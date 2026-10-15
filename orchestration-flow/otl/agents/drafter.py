"""drafter_node — publish a consistent terminal response or review note.

The single sink of the graph. The orchestrator always routes here to finalize, and its `final_mode`
(not any specialist flag) selects the mode:

  * EVIDENCE-GATE (`evidence_gate_decision.triggered`) — a configured verify_evidence rule fired BEFORE
    analysis ran (jade/orchestrator/specialists all skipped — see graph.py). Missing auditor support
    deterministically denies the exception. Supplied evidence that cannot be opened or accessed produces
    a deterministic HITL deferral instead. A model call cannot improve either fixed message, so this path
    intentionally consumes no drafter tokens.
  * TERMINAL  (`final_mode == "terminal"`) — the orchestrator judged a defensible rebuttal possible.
    The model supplies small semantic fields and ranks the supported arguments; code then assembles
    them strongest-first and enforces a concise, externally readable co-owner response. Raw ledger/cost-object identifiers,
    filenames, profile codes, and full rule text remain in the report's stage findings instead.
  * HITL      (`hitl_review` / `final_mode == "needs_evidence"`) — human judgement, intervention, or
    follow-up remains required. The orchestrator may have been unable to reach a defensible conclusion,
    or an attestation certificate may remain outstanding. Do NOT fabricate an unresolved analytical
    conclusion: write a deferral note with (a) findings so far, (b) why processing stopped, and (c) what
    evidence, judgement, or follow-up is needed.
    Disposition = "needs_evidence". The orchestrator has already set `hitl_review=True` +
    `hitl_reason`/`hitl_stage`/`hitl_evidence_trail` (and writes `hitl_review=False` explicitly when it
    finalizes terminal, so the gate can never be raised by a stale value); the full evidence trail is
    preserved by the output writer, so the note references it rather than restating it.

`disposition == "needs_evidence"` always implies `hitl_review=True`: it is reserved for human judgement,
intervention, inaccessible supplied evidence, or operational follow-up.
Every attestation-certificate request finishes with disposition `needs_evidence`, HITL stage
`attestation_certificate`, and an LLM-generated attestation paragraph; mixed cases retain their
otherwise-supported terminal wording before that paragraph.

Grounding rule: the drafter never opens a workbook and never invents a figure. It reasons only over a
publication-sized digest of the distilled sub-states. The complete technical findings and verbatim CFR
clause remain serialized separately in the report, so the outward response can state the supported
business basis without copying the trace into the letter.

PASSED TO THE LLM (structured output):
  * system : DRAFTER_SYSTEM (prompts.py)
  * user   : TERMINAL or HITL mode instructions plus the compact findings digest (the deterministic
             EVIDENCE-GATE path does not call the model)
  * schema : _DrafterExtraction — semantic response parts, assembled and checked in code
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field

from otl.authority import (OTL_GOVERNING_INSTRUMENT, OTL_MAJOR_CONSTRUCTION_PROVISION,
                           OTL_TECHNICAL_EXCLUSIONS_PROVISION, agreement_applies_to_asset)
from otl import llm_adapter as llm
from otl.prompts import DRAFTER_SYSTEM
from otl.state import SUBSTATE_KEYS, DrafterState
from otl.trace import trace_event
from otl.agents._common import (active_auditor_context_block, people_coverage,
                                rebuttal_context_block)

_TERMINAL_DISPOSITIONS = {"deny", "partial", "accept"}
_CO_OWNER_GUIDELINE_WORDS = 220
_ANALYTIC_QUESTIONS = {"job_title", "location", "hierarchy"}

_CREDIT_GRANTED_HEADING = "Credit Granted"
_CREDIT_PARTIALLY_GRANTED_HEADING = "Credit Partially Granted"
_CREDIT_DENIED_HEADING = "Credit Denied"
_CREDIT_DENIED_DOCUMENTATION_HEADING = "Credit Denied - supporting documentation provided"
_RESPONSE_WITHHELD_HEADING = "Response Withheld - pending information / Under Review"
_INSUFFICIENT_AUDITOR_EVIDENCE_HEADING = "Insufficient Auditor Evidence"

_ATTESTATION_SYSTEM = """You write a concise auditor-facing response to an attestation-certificate
request. Use the auditor's actual wording and available case context to identify the certificate
naturally. State that the requested certificate will be provided when it becomes available. Do not
promise a date, discuss internal workflow, mention HITL or agents, decide the labor-credit dispute, or
add unsupported facts. Return one direct sentence."""

_ADDITIONAL_SUPPORT_SYSTEM = """You write a concise auditor-facing response to a current unresolved request for evidence,
documents, records, data, support, or factual information. Use the auditor's actual wording and available
case context to identify what was requested naturally. State that the requested material or information
will be provided when it becomes available or is compiled. Do not promise a date, discuss internal
workflow, mention HITL or agents, decide the labor-credit dispute, or add unsupported facts. Return one
or two direct sentences."""

_PRESENTATION_FIELDS: dict[str, tuple[str, ...]] = {
    "intake": (
        "nl_summary", "assets_in_question", "dollars", "collected_evidence",
    ),
    "category_check": ("nl_summary", "auditor_question"),
    "verify_evidence": (
        "pob_data_found", "heli_data_found", "hierarchy_data_found", "evidence_gate_decision",
    ),
    "jade_analyzer": (
        "nl_summary", "overhead_mode", "overhead_rate", "amount_basis", "amount_computed",
        "amount_questioned", "amount_in_range", "jade_amount_discrepancy",
        "service_months_by_person", "service_dates_by_person",
        # The major-construction screen. The flag, the people it covers and the one-line basis travel,
        # but only when the screen fired (see `_nl_summary_digest`). `jv_object_type_by_person` never
        # travels — those are raw ledger cell values, trace material like `ledger_person_values`, and the
        # outward response strips internal identifiers anyway.
        "major_construction_charge", "major_construction_people", "major_construction_basis",
        # The settled person list resolved from ledger, POS, and intake inputs. `ledger_person_values` is
        # deliberately absent — those are raw cell contents, not
        # findings, and they belong in the trace only.
        "people_found_in_ledger", "people_named_count",
    ),
    "location": (
        "nl_summary", "found_in_pob", "found_in_heli_logs", "pob_people_location", "origin",
        "destination", "start_date", "end_date",
    ),
    "job_title": (
        "nl_summary", "identified_names_staff_listing", "positions_by_person",
        "selected_job_profiles_by_person", "role_status_by_person", "technical_rationale_by_person",
        "ipt_status_by_person", "exempt_role_status_by_person", "job_profile_status_by_person",
        "technical_job_title", "technical_rationale",
    ),
    "matching": (
        "nl_summary", "identified_cfr_category", "governing_instrument",
        "governing_source_document", "technical_employee_section",
        "technical_employee_application", "identified_settlement_section",
        "identified_settlement_application", "cfr_primary", "cfr_secondary", "cfr_additional",
        "subpart_labels", "cfr_rationale", "cfr_source_reference",
        "cfr_copas_resolution", "cfr_copas_resolution_by_person",
        "cfr_mapping_status_by_person", "matched_cfr_profiles_by_person",
        "cfr_primary_by_person", "cfr_secondary_by_person", "cfr_additional_by_person",
        "subpart_labels_by_person", "cfr_source_reference_by_person",
    ),
    "supervision": (
        "nl_summary", "above_first_level", "reports_summary", "questioned_amounts",
        "credit_amount", "chargeable_amount", "needs_human_review", "review_reason",
        "supervision_status_by_person", "supervision_record_status_by_person",
    ),
}

_WBS_CODE = re.compile(r"\bC\.(?:[A-Z0-9]+\.)+[A-Z0-9]+\b", re.IGNORECASE)
_ACCOUNT_CODE = re.compile(r"\bA\d{6,9}\b", re.IGNORECASE)
_RAW_FILE = re.compile(
    r"\b[^\n,;:]{1,100}\.(?:xlsx?|xlsm|csv|pdf)\b"
    r"(?:\s*\([^)]{0,80}\)|\s*,\s*(?:sheet|tab|worksheet)[^),.;]{0,50}\)?)?",
    re.IGNORECASE,
)
_INTERNAL_EXTERNAL_MARKER = re.compile(
    r"\b(?:JADE|CJI3|Jobs_Catalogue|HITL|LLM|SME|orchestrator|pipeline|language model|"
    r"A\d{6,9}|C\.(?:[A-Z0-9]+\.)+[A-Z0-9]+)\b|\.(?:xlsx?|xlsm|csv)\b",
    re.IGNORECASE,
)


class _DrafterExtraction(BaseModel):
    """Small semantic fields that code turns into the public response shape.

    The Field descriptions below become the JSON schema the model fills; each names which mode uses it.
    """
    disposition: str = Field(
        default="",
        description="TERMINAL: deny (charge stands / auditor wrong) | accept (exception valid, reverse "
                    "the charge) | partial (mixed). EVIDENCE-GATE and HITL: needs_evidence.")
    executive_summary: str = Field(
        default="",
        description="TERMINAL: 1-2 sentences stating the outcome and controlling reason without "
                    "restating the auditor's allegation.")
    amount_analysis: str = Field(
        default="", description="TERMINAL: at most 2 sentences on whether the amount reconciles and why.")
    evidence_basis: str = Field(
        default="", description="TERMINAL: 1-2 short sentences containing only the decisive non-date "
                                "role or supervision facts. Do not repeat their implication or authority, "
                                "and do not turn a governing requirement such as Joint Property benefit "
                                "into an established case fact unless the findings expressly prove it.")
    date_basis: str = Field(
        default="", description="TERMINAL: 1-3 client-ready factual sentences comparing the charged "
                                "period with contrary location dates, using only supplied date evidence. "
                                "Do not invent precision when exact charge dates or month counts are absent.")
    governing_basis: str = Field(
        default="", description="TERMINAL: 1-2 plain-language sentences stating the supported OTL "
                                "basis and naming the exact supplied agreement sections and CFR ranges.")
    argument_priority: list[str] = Field(
        default_factory=list,
        description="TERMINAL: strongest-to-weakest supported bases, using only date_coverage, "
                    "supervision, role_authority, amount_reconciliation.")
    challenged_premise: str = Field(
        default="",
        description="TERMINAL internal reasoning only: the primary factual assertion about the actual "
                    "transactions/person/period that drives the adjustment—not a legal conclusion or "
                    "secondary objection. This is not repeated in the published response.")
    argument_roles: dict[str, str] = Field(
        default_factory=dict,
        description="TERMINAL: selected argument label -> direct_refutation | independent_basis | "
                    "strengthening | corroboration, based on its relationship to challenged_premise.")
    premise_relationships: dict[str, str] = Field(
        default_factory=dict,
        description="TERMINAL internal reasoning: selected argument label -> contradicts | limits_scope | "
                    "accepts_or_assumes | does_not_address. limits_scope means the evidence shows the "
                    "primary assertion is too broad or cannot be applied to the full charged population. "
                    "An analysis saying a premise is too broad, too narrow, or cannot be applied to all "
                    "charged items must use limits_scope, never does_not_address. "
                    "If the primary factual premise were fully true and the argument would still work, "
                    "it accepts_or_assumes the premise and cannot be a direct_refutation.")
    argument_analysis: dict[str, str] = Field(
        default_factory=dict,
        description="TERMINAL: for each selected argument label, one plain sentence of no more than "
                    "25 words explaining its logical contribution. Do not repeat facts or authority.")
    conclusion: str = Field(
        default="", description="TERMINAL: 1-2 sentences giving the disposition and any supported credit amount.")
    co_owner_response: str = Field(
        default="",
        description="HITL only: a send-ready status paragraph without internal workflow language. "
                    "Terminal publication is assembled in code from the ranked semantic fields.")
    hitl_nl_summary: str = Field(
        default="",
        description="HITL: at most 4 short sentences covering only the material findings established so far.")
    hitl_blocker: str = Field(
        default="", description="HITL: 1-2 sentences identifying the concrete unresolved factual, "
                                "contractual, or classification issue in auditor-facing business language.")
    hitl_impact: str = Field(
        default="", description="HITL: one sentence explaining why the unresolved issue prevents the "
                                "proper chargeability, allocation, or credit from being determined.")
    hitl_next_step: str = Field(
        default="", description="HITL: 1-2 sentences stating the exact evidence, factual confirmation, "
                                "or professional judgement required to complete the review.")
    nl_summary: str = Field(
        default="", description="A concise but substantive analysis of the disposition, its material support, "
                                "and remaining uncertainty or open points. Do not list every fact.")


class _AttestationExtraction(BaseModel):
    response: str = Field(
        min_length=1,
        description="One or two concise auditor-facing sentences responding to the actual attestation "
                    "certificate request and stating it will be provided when available. Do not promise a date.",
    )


class _AdditionalSupportExtraction(BaseModel):
    response: str = Field(
        min_length=1,
        description="One or two concise auditor-facing sentences responding to the actual evidence, "
                    "document, data, support, or information request. Do not promise a date.",
    )


def _dump(value: Any) -> Any:
    """Serialize a sub-state (Pydantic model or already a dict) to a plain dict for the prompt."""
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value if isinstance(value, dict) else {}


def _edge_excerpt(value: Any, *, head: int, tail: int) -> str:
    """Retain both setup and conclusion from a long specialist summary."""
    text = " ".join(str(value or "").split())
    if len(text) <= head + tail:
        return text
    return f"{text[:head].rstrip()} … {text[-tail:].lstrip()}"


def _date_evidence_excerpt(state: dict) -> dict[str, Any]:
    """Expose existing specialist date findings without changing those stages.

    Older reports did not carry typed charge-month fields. Their summaries still put the ledger-period
    setup near the beginning and the location-coverage caveat near the end, so bounded edge excerpts
    preserve both while keeping the publication prompt manageable.
    """
    jade = _dump(state.get("jade_analyzer"))
    location = _dump(state.get("location"))
    starts = location.get("start_date") or {}
    ends = location.get("end_date") or {}
    if not (starts and ends and (jade.get("nl_summary") or jade.get("jade_line_item_texts"))):
        return {}
    excerpt: dict[str, Any] = {
        "location_start_dates": starts,
        "location_end_dates": ends,
    }
    if jade.get("nl_summary"):
        excerpt["charge_record_summary_excerpt"] = _edge_excerpt(
            jade["nl_summary"], head=850, tail=1100,
        )
    if jade.get("jade_line_item_texts"):
        excerpt["charge_line_text_sample"] = list(jade["jade_line_item_texts"] or [])[:20]
    if location.get("nl_summary"):
        excerpt["location_summary_excerpt"] = _edge_excerpt(
            location["nl_summary"], head=800, tail=1800,
        )
    return excerpt


def _nl_summary_digest(state: dict) -> dict:
    """Return only facts needed to write the response, not the technical trace.

    The full stage models remain in the serialized report. Keeping raw paths, resolved column names,
    cost-object lists, full job descriptions, and the verbatim CFR clause out of this prompt both lowers
    token use and makes it less likely that internal detail leaks into the co-owner paragraph.
    """
    digest: dict[str, Any] = {}
    for key in SUBSTATE_KEYS:
        # `wiki` is an independent report sidecar. Keep the exclusion explicit so a future
        # presentation-field addition cannot accidentally make precedent text a drafting input.
        if key in {"drafter", "orchestrator", "wiki", "agreement_retriever"}:
            continue
        d = _dump(state.get(key))
        allowed = _PRESENTATION_FIELDS.get(key, ())
        compact = {
            field: _compact_prompt_value(d[field], string_limit=3_000 if field == "nl_summary" else 1200)
            for field in allowed
            if d.get(field) not in (None, "", [], {})
        }
        if key == "jade_analyzer" and not d.get("major_construction_charge"):
            # The major-construction screen only speaks when it FIRED. A False flag — or a basis line
            # saying the cited ledger carries no such column — is a fact against the operator's charge.
            # The orchestrator sees it either way and weighs it; the response writer should not be handed
            # a negative it has no use for.
            for field in ("major_construction_charge", "major_construction_people",
                          "major_construction_basis"):
                compact.pop(field, None)
        if compact:
            digest[key] = compact
    reused = state.get("rebuttal_reused_nl_summaries") or {}
    if reused:
        digest["reused_prior_cycle_nl_summaries"] = {
            key: _compact_prompt_value(value, string_limit=2_000)
            for key, value in reused.items()
        }
    coverage = _date_coverage_context(state)
    if _date_coverage_sentence(state):
        digest["date_coverage_comparison"] = coverage
    elif date_excerpt := _date_evidence_excerpt(state):
        digest["date_evidence_excerpt"] = date_excerpt
    orchestrator = _dump(state.get("orchestrator"))
    decisions = list(orchestrator.get("decision_log") or [])
    if decisions:
        digest["orchestrator_case_assessment"] = [
            {
                field: decision[field]
                for field in ("hop", "chose", "rationale", "final_mode")
                if decision.get(field) not in (None, "")
            }
            for decision in decisions[-6:]
            if isinstance(decision, dict)
        ]
    supported = _supported_argument_components(state)
    if supported:
        # Alphabetical order is deliberate: this inventories what is present without implying rank.
        digest["supported_argument_components_unordered"] = supported
    return digest


def _agreement_prompt_block(state: dict) -> str:
    """Render selected agreement chunks in full, without prompt compaction or model paraphrase.

    The retriever loads ``verbatim_text`` from SQLite after the reranking model returns IDs.  Keep the
    source blocks outside ``_nl_summary_digest`` so the generic string/list bounds can never truncate them.
    """
    agreement = _dump(state.get("agreement_retriever"))
    chunks = list(agreement.get("selected_chunks") or [])
    if not chunks:
        status = str(agreement.get("status") or "")
        return (
            f"\n\nAGREEMENT RETRIEVAL STATUS: {status or 'unavailable'}; no indexed agreement "
            "excerpt is available for quotation."
        ) if agreement else ""
    rendered: list[str] = [
        "\n\nVERBATIM AGREEMENT EXCERPTS — PRIMARY LEGAL GROUNDING:",
        "AGREEMENT CHARGEABILITY ASSESSMENT: "
        + str(agreement.get("agreement_chargeability") or "unclear")
        + "; citation chunk IDs: "
        + ", ".join(str(item.get("chunk_id")) for item in
                    (agreement.get("agreement_chargeability_citations") or []) if item.get("chunk_id")),
        "The text inside each AGREEMENT_CHUNK block was loaded from the source index after ID-only "
        "reranking. Quote agreement language only from these blocks and preserve quoted wording "
        "exactly. Never present selection_reason, specialist findings, or retrieval metadata as "
        "agreement language. A clause states the governing test; it does not prove that the case facts "
        "satisfy that test.",
    ]
    for chunk in chunks:
        labels = chunk.get("printed_page_labels") or []
        page_label = ", ".join(str(value) for value in labels) or (
            str(chunk.get("pdf_page_start"))
            if chunk.get("pdf_page_start") == chunk.get("pdf_page_end")
            else f"{chunk.get('pdf_page_start')}-{chunk.get('pdf_page_end')}"
        )
        attributes = {
            "id": chunk.get("chunk_id"),
            "document": chunk.get("document_name"),
            "track": chunk.get("legal_track"),
            "section": chunk.get("heading_path"),
            "pages": page_label,
            "party_scope": chunk.get("party_scope") or [],
            "extraction_methods": chunk.get("extraction_methods") or [],
            "quote_safe": bool(chunk.get("quote_safe")),
            "content_sha256": chunk.get("content_sha256"),
        }
        rendered.append(
            "[AGREEMENT_CHUNK " + json.dumps(attributes, ensure_ascii=False) + "]\n"
            + str(chunk.get("verbatim_text") or "")
            + "\n[/AGREEMENT_CHUNK]"
        )
    return "\n\n".join(rendered)


def _has_agreement_chunks(state: dict) -> bool:
    return bool(_dump(state.get("agreement_retriever")).get("selected_chunks"))


def _compact_prompt_value(value: Any, *, string_limit: int = 1200) -> Any:
    """Bound verbose specialist summaries while retaining their structured conclusions."""
    if isinstance(value, str):
        if len(value) <= string_limit:
            return value
        return value[:string_limit].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
    if isinstance(value, list):
        items = [_compact_prompt_value(item, string_limit=400) for item in value[:20]]
        if len(value) > 20:
            items.append(f"[{len(value) - 20} additional item(s) retained in the technical report]")
        return items
    if isinstance(value, dict):
        items = list(value.items())
        compact = {
            str(key): _compact_prompt_value(item, string_limit=500)
            for key, item in items[:12]
        }
        if len(items) > 12:
            compact["_additional_items"] = len(items) - 12
        return compact
    return value


def _iso_date(value: Any) -> date | None:
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _months_in_window(start: date, end: date) -> list[str]:
    months: list[str] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return months


def _date_coverage_context(state: dict) -> dict:
    """Build a deterministic charge-period vs contrary-location-window comparison.

    An aggregate is emitted only when all location aliases resolve to one unique interval. Otherwise
    the per-person typed values remain available and publication must not conflate several people's
    windows.
    """
    jade = _dump(state.get("jade_analyzer"))
    location = _dump(state.get("location"))
    month_map = jade.get("service_months_by_person") or {}
    date_map = jade.get("service_dates_by_person") or {}
    service_months = sorted({
        str(month) for values in month_map.values() for month in (values or [])
        if re.fullmatch(r"20\d{2}-(?:0[1-9]|1[0-2])", str(month))
    })
    service_dates = sorted({
        parsed for values in date_map.values() for value in (values or [])
        if (parsed := _iso_date(value)) is not None
    })
    starts = {_iso_date(value) for value in (location.get("start_date") or {}).values()}
    ends = {_iso_date(value) for value in (location.get("end_date") or {}).values()}
    starts.discard(None)
    ends.discard(None)
    context: dict[str, Any] = {
        "service_months_by_person": month_map,
        "service_dates_by_person": date_map,
        "location_start_dates": location.get("start_date") or {},
        "location_end_dates": location.get("end_date") or {},
    }
    if not any(context.values()):
        return {}
    if service_months and len(starts) == 1 and len(ends) == 1:
        start, end = next(iter(starts)), next(iter(ends))
        if start <= end:
            contrary_months = _months_in_window(start, end)
            context.update({
                "charged_service_months": service_months,
                "charged_service_month_count": len(service_months),
                "contrary_location_window": {"start": start.isoformat(), "end": end.isoformat()},
                "contrary_window_months": contrary_months,
                "overlapping_service_months": sorted(set(service_months) & set(contrary_months)),
                "contrary_months_without_service_entry": sorted(
                    set(contrary_months) - set(service_months)
                ),
                "explicit_service_dates_outside_window": [
                    value.isoformat() for value in service_dates if value < start or value > end
                ],
                "explicit_service_dates_inside_window": [
                    value.isoformat() for value in service_dates if start <= value <= end
                ],
                "explicit_service_date_count": len(service_dates),
            })
    return context


def _month_label(value: str) -> str:
    try:
        return datetime.strptime(value, "%Y-%m").strftime("%B %Y")
    except ValueError:
        return value


def _day_label(value: date) -> str:
    return f"{value.strftime('%B')} {value.day}, {value.year}"


def _date_coverage_sentence(state: dict) -> str:
    coverage = _date_coverage_context(state)
    count = int(coverage.get("charged_service_month_count") or 0)
    window = coverage.get("contrary_location_window") or {}
    start = _iso_date(window.get("start"))
    end = _iso_date(window.get("end"))
    if not (count and start and end):
        return ""
    overlap = list(coverage.get("overlapping_service_months") or [])
    sentences = [
        f"The labor records span {count} service month{'s' if count != 1 else ''}, while the cited "
        f"location evidence covers only {_day_label(start)} through {_day_label(end)}."
    ]
    if overlap:
        overlap_label = ", ".join(_month_label(value) for value in overlap)
        if len(overlap) == 1:
            overlap_sentence = (
                f"At the month level, that window touches only {overlap_label}, representing 1 of "
                f"{count} charged month{'s' if count != 1 else ''}"
            )
        else:
            overlap_sentence = (
                f"At the month level, that window touches {overlap_label}, representing "
                f"{len(overlap)} of {count} charged month{'s' if count != 1 else ''}"
            )
    else:
        overlap_sentence = "At the month level, that window does not touch any charged service month"
    missing = list(coverage.get("contrary_months_without_service_entry") or [])
    if missing:
        overlap_sentence += (
            "; the ledger has no service-month entry for "
            + ", ".join(_month_label(value) for value in missing)
        )
    sentences.append(overlap_sentence + ".")
    outside = [_iso_date(value) for value in coverage.get("explicit_service_dates_outside_window") or []]
    outside = [value for value in outside if value is not None]
    if outside:
        labels = [_day_label(value) for value in outside[:4]]
        date_list = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]
        if all(value < start for value in outside):
            timing = "before the location window"
        elif all(value > end for value in outside):
            timing = "after the location window"
        else:
            timing = "outside the location window"
        if len(outside) == 1:
            sentences.append(
                f"The supplied service record is dated {date_list}, which is {timing}."
            )
        else:
            sentences.append(
                f"The supplied service records are dated {date_list}, all {timing}."
            )
    return " ".join(sentences)


def _authority_context(state: dict) -> dict[str, Any]:
    matching = _dump(state.get("matching"))
    jade = _dump(state.get("jade_analyzer"))
    references = list(dict.fromkeys(
        str(value).strip()
        for field in ("cfr_primary", "cfr_secondary", "cfr_additional")
        for value in (matching.get(field) or {}).values()
        if str(value).strip()
    ))
    source_references = list(dict.fromkeys(
        str(value).strip() for value in (matching.get("cfr_source_reference") or {}).values()
        if str(value).strip()
    ))
    asset = str((state.get("case") or {}).get("asset") or "")
    return {
        "cfr_references": references,
        "cfr_source_references": source_references,
        # The major-construction screen jade ran per person. MCOH means the labor sits on a major
        # construction project, and this quotes its own dedicated authority — a COUA provision, not the
        # GOM Settlement Agreement `instrument` above — so its instrument/section travel separately.
        "major_construction": bool(jade.get("major_construction_charge")),
        "major_construction_instrument": str(OTL_MAJOR_CONSTRUCTION_PROVISION.get("instrument") or ""),
        "major_construction_section": str(OTL_MAJOR_CONSTRUCTION_PROVISION.get("section") or ""),
        "major_construction_application": str(OTL_MAJOR_CONSTRUCTION_PROVISION.get("statement") or ""),
        "agreement_in_scope": not asset or agreement_applies_to_asset(asset),
    }


def _join_english(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + ", and " + values[-1]


def _annotated_cfr_references(context: dict[str, Any]) -> str:
    """Pair workbook CFR ranges with workbook source labels such as APD, casing, and BOP."""
    source_segments = [
        segment.strip()
        for source in context.get("cfr_source_references") or []
        for segment in re.split(r"\s*;\s*", source)
        if segment.strip()
    ]
    annotated: list[str] = []
    for reference in context.get("cfr_references") or []:
        code = "§" + reference.split("§", 1)[1] if "§" in reference else reference
        label = ""
        for segment in source_segments:
            if code in segment or code.replace("§", "") in segment:
                label = segment.split(code, 1)[-1].strip(" –—-:,")
                break
        annotated.append(f"{reference} ({label})" if label else reference)
    return _join_english(annotated)


def _authority_application(state: dict) -> str:
    """Explain each pinpoint authority beside the factual proposition it supports.

    A CFR workbook match is worth surfacing even when matching did not select one of the three
    category routes.  The wording preserves the distinction: a selected category is applied as a
    chargeability route; an unselected CFR match is identified as role-mapping context and is not
    promoted into a separate category-based basis.
    """
    # When indexed excerpts exist, the drafter's governing_basis is grounded in those exact blocks.
    # Do not append the legacy single-Chevron registry as though it were the universal instrument.
    if _has_agreement_chunks(state):
        return ""
    context = _authority_context(state)
    if not context["agreement_in_scope"]:
        return ""
    sentences: list[str] = []
    # Tracks which instrument the most recently appended sentence named, so "The same agreement's …"
    # is only used when it is actually true — the MCOH sentence below quotes a COUA, a different
    # instrument from `context['instrument']` (the case's GOM Settlement Agreement), so it must never
    # trigger that lead-in for the sentences that follow it.
    last_instrument = ""
    if context["major_construction"] and context["major_construction_section"]:
        # The major-construction basis. It comes from a code-table lookup against the ledger's coded
        # JV-object-type column, so it can stand on its own: an MCOH case may reach the drafter without
        # `matching` having run at all, and the sentence below is the only authority such a case would
        # otherwise carry.
        application = str(context["major_construction_application"] or "").strip()
        if application:
            application = application[0].lower() + application[1:]
        sentences.append(
            f"The labor ledger charges this work as major construction overhead, so it was performed "
            f"in connection with the design, construction or installation of a Major Construction "
            f"project. Under the {context['major_construction_instrument']}, "
            f"{context['major_construction_section']} charges that work direct to the Joint Account: "
            + (application.rstrip(".") if application else
               "personnel performing technical services in connection with the project are charged "
               "direct regardless of location")
            + "."
        )
        last_instrument = context["major_construction_instrument"]
    if not _dump(state.get("matching")):
        # Everything below is read off `matching`. Without it there is no category and no technical
        # finding to apply, and the MCOH sentence (if any) stands alone.
        return " ".join(sentences)
    cfrs = _annotated_cfr_references(context)
    if cfrs:
        sentences.append(
            "The available role mapping references " + cfrs + ". Those citations describe regulatory "
            "role context only; whether the offsite technical labor is chargeable is determined from the "
            "asset's retrieved governing agreements."
        )
    return " ".join(sentences)


def _supervision_authority_application(state: dict) -> str:
    if not _dump(state.get("supervision")):
        return ""
    asset = str((state.get("case") or {}).get("asset") or "")
    if asset and not agreement_applies_to_asset(asset):
        return ""
    section = str(OTL_TECHNICAL_EXCLUSIONS_PROVISION.get("section") or "Attachment 1 §2")
    return (
        f"Under the {OTL_GOVERNING_INSTRUMENT}, {section} excludes supervision above the first "
        "operational level from Technical Employees and treats its associated costs as overhead. "
        "First-level operational supervision remains billable."
    )


def _case_header(state: dict) -> dict:
    case = state.get("case") or {}
    return {k: case.get(k) for k in ("case_id", "exc_id", "year", "asset", "title") if case.get(k)}


def _coverage_note(state: dict) -> str:
    """A disclosure line for the prompt when the person-list cap truncated the disputed people.

    Empty in the normal case. Without this the model reads the findings as complete and writes a
    rebuttal that implies every disputed person was checked, which for a capped case is not true.

    Read exclusively from `jade_analyzer`, which owns the final resolved person population.
    """
    examined, dropped, total = people_coverage(state)
    if not dropped:
        return ""
    return (
        f"\n\nPARTIAL COVERAGE — the findings cover only {len(examined)} of "
        f"{total or len(examined) + len(dropped)} named people. EXAMINED: "
        f"{json.dumps(list(examined), ensure_ascii=False)}. NOT EXAMINED: "
        f"{json.dumps(dropped, ensure_ascii=False)}. State this limitation plainly in the response and "
        f"name the people not examined — confine every conclusion to the people examined and do NOT "
        f"imply the whole charge was reviewed.")


def _chargeability_resolution(state: dict) -> dict:
    return _dump(state.get("chargeability_resolution"))


def _business_review_overview(state: dict) -> dict[str, Any]:
    resolution = _chargeability_resolution(state)
    people = []
    for subject, result in (resolution.get("person_results") or {}).items():
        people.append({
            "person_or_charge_group": subject,
            "amount": result.get("amount"),
            "status": result.get("disposition"),
        })
    fact_labels = {
        "location_status": "work location",
        "role_status": "technical-role classification",
        "ipt_status": "Integrated Project Team status",
        "exempt_role_status": "exempt-role status",
        "job_profile_status": "job-profile match",
        "supervision_status": "supervision level",
        "supervision_record_status": "supervision records",
        "cfr_copas_resolution": "CFR/COPAS regulatory basis",
        "agreement_chargeability": "governing agreement chargeability basis",
        "legal_chargeability_basis": "combined legal chargeability basis",
        "precedence_status": "relevant historical precedent",
        "mcoh": "major-construction classification",
    }
    missing: list[str] = []
    for raw in state.get("chargeability_hook_evaluations") or []:
        item = _dump(raw)
        if item.get("status") not in {"pending", "unresolved"}:
            continue
        for key in item.get("missing_conditions") or []:
            label = fact_labels.get(str(key), str(key).replace("_", " "))
            if label not in missing:
                missing.append(label)
    return {
        "people_and_amounts": people,
        "resolved_deny_amount": resolution.get("resolved_deny_amount", 0),
        "resolved_accept_amount": resolution.get("resolved_accept_amount", 0),
        "unresolved_amount": resolution.get("unresolved_amount", 0),
        "unresolved_people_or_groups": resolution.get("unresolved_people", []),
        "remaining_information_overview": missing,
    }


def _terminal_user(state: dict, nl_summaries: dict) -> str:
    locked = _chargeability_resolution(state)
    return (
        "MODE: TERMINAL — the evidence was sufficient and no stage deferred. Draft the operator's "
        "rebuttal to this OTL audit exception, grounded ONLY in the summaries below.\n\n"
        f"CASE: {json.dumps(_case_header(state), ensure_ascii=False)}\n\n"
        f"{active_auditor_context_block(state)}\n\n"
        f"NATURAL-LANGUAGE SUMMARIES (per stage, already distilled — do not invent anything beyond these):\n"
        f"{json.dumps(nl_summaries, ensure_ascii=False, default=str)}"
        f"{_coverage_note(state)}"
        f"{_agreement_prompt_block(state)}\n\n"
        f"{rebuttal_context_block(state)}\n\n"
        f"LOCKED CHARGEABILITY RESULT: {json.dumps(locked, ensure_ascii=False, default=str)}\n"
        "Use the locked disposition and scoped amounts. Do not independently change the disposition.\n\n"
        "Write these semantic fields; code will apply the headings and final formatting:\n"
        "- disposition: one of deny | partial | accept (deny = the charge stands / auditor is wrong; "
        "accept = the exception is valid and the charge should be reversed; partial = mixed).\n"
        "- argument_priority: rank only the supported bases from strongest to weakest using these exact "
        "labels: date_coverage | supervision | role_authority | amount_reconciliation. Decide from this "
        "case. The unordered supported_argument_components list identifies the components actually "
        "present. Evaluate every listed component and include each one in argument_priority, "
        "premise_relationships, argument_roles, and argument_analysis; code, not the word target, decides "
        "which corroboration is omitted from client prose. First populate challenged_premise with the "
        "PRIMARY factual assertion about what actually "
        "happened in the charged transactions/person/period and that drives the proposed adjustment "
        "(internal reasoning only; it will not be published). If several objections appear, use narrative "
        "emphasis and orchestrator_case_assessment to distinguish the primary claim from secondary "
        "eligibility/documentation objections. A factual premise is not an unstated legal conclusion such "
        "as 'onsite presence is required.' For every selected label, first populate premise_relationships "
        "using contradicts | limits_scope | accepts_or_assumes | does_not_address. Use limits_scope when "
        "the evidence does not prove the opposite fact but shows the primary assertion is too broad or "
        "cannot be applied to the full charged population. Apply this counterfactual strictly: "
        "if the primary factual premise were completely true, would the argument still establish its "
        "point? If yes, it accepts_or_assumes the premise; making the premise irrelevant or non-dispositive "
        "does not contradict it. Then populate argument_roles, classifying each selected label relative to "
        "that primary premise: direct_refutation only when premise_relationships says contradicts or "
        "limits_scope and it shows the primary premise is false, "
        "too narrow, or does not apply to the charged transactions/person/period; independent_basis when it "
        "accepts the premise but establishes a separate dispositive contractual or factual basis; "
        "strengthening when it adds meaningful support; or corroboration when it only confirms arithmetic "
        "or consistency. Rank in that order. Evidence answering a secondary role/documentation objection "
        "does not become the direct refutation of a different primary location/activity premise. An "
        "alternative eligibility route does not factually refute the primary premise merely because it "
        "makes that premise non-dispositive. Any direct_refutation paired with accepts_or_assumes or "
        "does_not_address is logically invalid. No argument type has a universal priority. Before returning, "
        "cross-check each relationship against its argument_analysis: wording that says the premise is too "
        "broad, too narrow, covers only part of the population, or cannot be applied to all charged items "
        "requires limits_scope, while does_not_address is reserved for evidence with no bearing on that "
        "premise. Re-rank argument_priority after that consistency check.\n"
        "- argument_analysis: for each selected label, one plain analytical sentence of no more than "
        "25 words explaining its logical contribution. Do not repeat facts, authority, or another "
        "field's wording. For date_coverage, state only what the comparison establishes or fails to "
        "establish; do not repeat dates, counts, or overlap facts.\n"
        "- executive_summary: outcome and controlling reason, 1-2 sentences. Do not restate what the "
        "auditor alleged, requested, questioned, asserted, or claimed.\n"
        "- amount_analysis: whether the questioned amount reconciles, at most 2 sentences.\n"
        "- evidence_basis: only the decisive non-date role or supervision facts, 1-2 short sentences. "
        "A governing requirement, "
        "including Joint Property benefit, is not a case fact unless the specialist findings expressly "
        "establish it.\n"
        "- date_basis: when date_coverage_comparison or date_evidence_excerpt is present, write 1-3 "
        "client-ready factual sentences comparing the charged period with the contrary location dates. "
        "Use exact counts and dates only when supplied. If older summary evidence establishes only a "
        "broader charged period and a narrow location window, state that bounded comparison without "
        "inventing missing dates, months, or overlap counts. Do not state the inference here. When a "
        "typed date_coverage_comparison is present, expressly quantify its charged service months, "
        "window, overlap, missing ledger month, and explicit charge dates outside the window.\n"
        "- governing_basis: use the VERBATIM AGREEMENT EXCERPTS as the primary agreement authority when "
        "they are present. Compare materially different shared and bilateral tracks without merging their "
        "party scopes. Name the instrument, heading/section and supplied page label beside each proposition. "
        "You may quote a short controlling passage, but every quoted word must appear exactly inside its "
        "AGREEMENT_CHUNK block. Include CFR ranges supplied in matching. When no indexed excerpt is present, "
        "the existing curated authority fallback may be used. When matching selected no category, describe "
        "CFR citations only as role-mapping context, not as a separate category-based chargeability route.\n"
        "- conclusion: the outcome and supported credit amount, 1-2 sentences.\n"
        "Code will assemble the co-owner paragraph in the ranked order and place each agreement/CFR "
        "reference beside the proposition it supports. Aim for roughly 220 words, but preserve material "
        "reasoning and governing support when a clear response needs more space. Use natural variation "
        "in sentence length. Prefer periods and commas; do not use em dashes. Leave co_owner_response "
        "and all hitl_* fields empty."
    )


def _hitl_user(state: dict, nl_summaries: dict, why_stopped: str) -> str:
    overview = _business_review_overview(state)
    return (
        "MODE: HITL — processing stopped before a defensible conclusion. Do NOT fabricate a "
        "conclusion. Write a deferral note.\n\n"
        f"CASE: {json.dumps(_case_header(state), ensure_ascii=False)}\n\n"
        f"WHY PROCESSING STOPPED: {why_stopped}\n"
        # Always a post-analysis deferral — the evidence gate returns earlier and never reaches HITL mode.
        f"STAGE THAT DEFERRED: {state.get('hitl_stage') or 'orchestrator'}\n\n"
        f"{active_auditor_context_block(state)}\n\n"
        f"NATURAL-LANGUAGE SUMMARIES SO FAR (per stage — everything gathered before the stop):\n"
        f"{json.dumps(nl_summaries, ensure_ascii=False, default=str)}"
        f"\n\nBUSINESS REVIEW OVERVIEW:\n{json.dumps(overview, ensure_ascii=False, default=str)}"
        f"{_coverage_note(state)}"
        f"{_agreement_prompt_block(state)}\n\n"
        f"{rebuttal_context_block(state)}\n\n"
        "The complete deferral evidence trail is retained in the technical report; do not recreate it "
        "in the response.\n\n"
        "Write FOUR compact semantic fields; code will assemble the auditor-facing paragraph in this "
        "order and apply the technical-note headings:\n"
        "- hitl_nl_summary: a compact business overview followed by each listed person's or charge group's "
        "recommended amount and resolved/unresolved status. Include material reconciled amounts and "
        "supported classifications, but do not name internal rules or hooks. "
        "Select findings that help explain the unresolved decision; omit audit tags, raw account codes, "
        "file mechanics, and other trace detail that does not affect the remaining question.\n"
        "- hitl_blocker: 1-2 sentences summarizing the unresolved factual, contractual, or classification "
        "themes across the population. Do not provide a person-by-person missing-data inventory.\n"
        "- hitl_impact: one sentence explaining why that unresolved question prevents chargeability, "
        "allocation, or the proper credit from being determined.\n"
        "- hitl_next_step: 1-2 sentences giving an overview of the evidence, confirmation, or professional "
        "judgement still needed. A later workflow will create targeted actions, so do not enumerate a "
        "technical task for every person. Say 'please provide' only when identified records are genuinely "
        "missing from the current record. When the evidence is already present but requires expert "
        "classification or judgement, state the confirmation or determination required without implying "
        "that the auditor failed to submit documents. Never say that a reviewer, agent, or system should "
        "take the next step.\n"
        "Do not repeat the auditor's allegation, imply acceptance or denial, or mention agents, models, "
        "HITL, routing, a pipeline, or internal review mechanics. Aim for roughly 120-180 words overall, "
        "but preserve a second unresolved issue or necessary record when material. Leave "
        "co_owner_response and all terminal semantic fields empty. Set disposition = needs_evidence."
    )


def _required_evidence(ve: dict, questions: list[str], gap: Any = None) -> list[str]:
    """The source material absent from, or needing an accessible copy in, the auditor record.

    Two nodes can fail the gate, so there are two sources for this list.

    `gap` is the `evidence_gap` channel. The node that failed the gate writes it, naming the data it
    could not find, and when it is set we simply pass it through — that node knew exactly what was
    missing. Today only `jade_analyzer` writes it (the labor ledger).

    Everything below is the location-question path, for `verify_evidence`, which writes no gap. Its
    gate applies whenever the question set contains `location` (see `verify_evidence._evidence_gate`),
    and it fails precisely when POB and/or heli is absent. So we name POB / helicopter data whenever its
    flag says it is absent. `job_title` and `hierarchy` questions alone never gate (they don't turn on
    the auditor's cited POB/heli), so they never reach this path. With no recognized category the
    gate PASSES (the orchestrator defers to HITL), so the no-category branch here is only a defensive
    fallback.
    """
    if isinstance(gap, (list, tuple)):
        named = [str(g).strip() for g in gap if str(g or "").strip()]
        if named:
            return named

    req: list[str] = []
    qset = {q for q in (questions or [])}

    def _need_pob_heli() -> None:
        if not ve.get("pob_data_found"):
            req.append("personnel-on-board (POB) records for the charge period")
        if not ve.get("heli_data_found"):
            req.append("helicopter manifests / flight logs for the charge period")

    if "location" in qset or not qset:
        _need_pob_heli()
    return req


def _gate_subject(gap: Any = None) -> str:
    """What kind of evidence the gate found missing, for the response's opening sentence and findings.

    Kept coarse on purpose. The gap entries themselves are already specific, and they are named in the
    very next sentence, so the opener only has to be true.
    """
    return "supporting documentation" if gap else "location evidence"


def _required_sources_text(required: list[str]) -> str:
    """Render deterministic evidence labels in one client-facing sentence fragment."""
    if required:
        if len(required) == 1:
            return required[0]
        return ", ".join(required[:-1]) + " and " + required[-1]
    return ("personnel-on-board records and helicopter manifests or flight logs for the "
            "charge period")


def _unsupported_evidence_denial(required: list[str], subject: str = "location evidence") -> str:
    """Deterministic denial when the auditor did not supply required support."""
    sources = _required_sources_text(required)
    return (
        f"This exception is denied because sufficient {subject} was not provided. "
        f"The cited material does not include {sources}. No credit is due."
    )


def _unreadable_evidence_response(required: list[str], subject: str = "supporting documentation") -> str:
    """Deterministic HITL response for supplied evidence that cannot be accessed."""
    sources = _required_sources_text(required)
    return (
        f"A final determination cannot be made because the supplied {subject} could not be opened or "
        f"accessed. Please provide an accessible copy of {sources}."
    )


def _normalize_text(value: Any) -> str:
    """Collapse model formatting into clean paragraph text."""
    text = str(value or "").replace("\u00a0", " ")
    text = re.sub(r"(?m)^\s*(?:[-*]\s+|#+\s*)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()


def _clean_external_text(value: Any) -> str:
    """Keep technical identifiers in the report, but not in outward-facing prose."""
    text = _normalize_text(value)
    text = re.sub(
        r"^This exception questions (\$[\d,]+(?:\.\d{2})?)\s+in\b",
        r"The auditor requests a \1 credit for",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\blabor charged for ([A-Z][A-Za-z' .-]{1,60}?) to (the [A-Z])",
        r"labor for \1 charged to \2",
        text,
    )
    def _source_label(match: re.Match[str], label: str) -> str:
        at_sentence_start = match.start() == 0 or text[max(0, match.start() - 2):match.start()] in {
            ". ", "! ", "? ",
        }
        return label.capitalize() if at_sentence_start else label

    text = _RAW_FILE.sub(lambda match: _source_label(match, "the source records"), text)
    text = re.sub(r"\b(?:on|in|from)\s+(?:worksheet|sheet|tab)\s*['\"]?[^,.;:]{1,50}['\"]?",
                  "in the source records", text, flags=re.IGNORECASE)
    text = re.sub(r"\bJADE\b", lambda match: _source_label(match, "the labor ledger"), text,
                  flags=re.IGNORECASE)
    text = re.sub(r"\bCJI3\b", lambda match: _source_label(match, "the ledger"), text,
                  flags=re.IGNORECASE)
    text = re.sub(r"\bJobs_Catalogue\b|\bJobs? Catalogue\b",
                  "the documented role description", text, flags=re.IGNORECASE)
    text = _WBS_CODE.sub("the challenged cost object", text)
    text = re.sub(r"\b(?:WBS|cost object)\s+the challenged cost object\b",
                  "the challenged cost object", text, flags=re.IGNORECASE)
    text = re.sub(
        r"(?:the challenged cost object\s*(?:,|and)\s*)+the challenged cost object",
        "the challenged cost objects", text, flags=re.IGNORECASE,
    )
    text = re.sub(r"\bWBS\s+objects?\b", "cost objects", text, flags=re.IGNORECASE)
    text = re.sub(r"\bWBS(?:es)?\b", "cost object", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcost object objects\b", "cost objects", text, flags=re.IGNORECASE)
    text = _ACCOUNT_CODE.sub("the applicable labor account", text)
    text = re.sub(r"\baccount\s+the applicable labor account\b",
                  "the applicable labor account", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthe Cost\s*\+\s*OH column sums to\b",
                  "the labor ledger totals", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthe labor ledger ledger\b", "the labor ledger", text,
                  flags=re.IGNORECASE)
    text = re.sub(r"\b([Tt]he source records) (?:contains|confirms|shows)\b",
                  lambda match: f"{match.group(1)} " + {
                      "contains": "contain", "confirms": "confirm", "shows": "show",
                  }[match.group(0).split()[-1].lower()], text)
    text = re.sub(r"\s*\(?(?:staff|employee|position)\s*(?:no\.?|number|id)?\s*\d{5,12}\)?",
                  "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+JG\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:the\s+)?(?:AI|LLM)\s+(?:cannot|could not)\s+(?:determine|decide|infer)\b",
                  "the current record does not establish", text, flags=re.IGNORECASE)
    text = re.sub(r"\bHITL\b", "review", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:the\s+)?(?:orchestrator|pipeline)\b", "the review", text,
                  flags=re.IGNORECASE)
    text = re.sub(r"\b(?:specialist|agent)\s+(?:stage\s+)?", "review ", text,
                  flags=re.IGNORECASE)
    text = re.sub(r"\b(?:a|the) (?:human |SME )?reviewer must confirm\s+",
                  "Confirmation is required regarding ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:a|the) (?:human |SME )?reviewer must resolve\s+",
                  "The review must resolve ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:a|the) reviewer should confirm\s+",
                  "Confirmation is needed regarding ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:a|the) reviewer should determine\s+",
                  "A determination is needed regarding ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:the\s+)?role catalogue\b", "the available role documentation", text,
                  flags=re.IGNORECASE)
    text = re.sub(r"\bSME review\b", "additional review", text, flags=re.IGNORECASE)
    text = re.sub(
        r"(^|(?<=[.!?])\s+)the review\b",
        lambda match: f"{match.group(1)}The review",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bthe\s+the\b", "the", text, flags=re.IGNORECASE)
    # Do not let a model turn the agreement's Joint-Property-benefit requirement into a case fact.
    # Removing only the unsupported trailing qualifier preserves the grounded role/service description.
    text = re.sub(
        r"\s+(?:performed|provided|used)\s+(?:principally\s+)?for the (?:principal\s+)?benefit "
        r"of (?:the\s+)?(?:Joint Property|Joint Operations)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Client prose uses commas or sentence breaks instead of model-favored em dashes. Preserve en
    # dashes in section titles and CFR ranges such as §250.410–418.
    text = re.sub(r"\s+—\s+", ", ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text.strip()


def _attestation_response(state: dict) -> str:
    """Generate the concise case-specific attestation paragraph only when one was requested."""
    if not state.get("attestation_requested"):
        return ""
    category = _dump(state.get("category_check"))
    user = (
        f"{active_auditor_context_block(state)}\n\n"
        "CATEGORY-CHECK FINDINGS:\n"
        f"{str(category.get('nl_summary') or '').strip()}\n\n"
        "Respond only to the attestation or affiliate certificate request."
    )
    ext = llm.structured("drafter", _ATTESTATION_SYSTEM, user, _AttestationExtraction)
    return _clean_external_text(ext.response)


def _additional_support_response(state: dict, *, requested: bool | None = None) -> str:
    """Generate the concise case-specific response to the auditor's information request."""
    if requested is None:
        requested = bool(state.get("additional_support_requested"))
    if not requested:
        return ""
    category = _dump(state.get("category_check"))
    user = (
        f"{active_auditor_context_block(state)}\n\n"
        "CATEGORY-CHECK FINDINGS:\n"
        f"{str(category.get('nl_summary') or '').strip()}\n\n"
        "Respond only to the auditor's evidence, document, data, support, or information request."
    )
    ext = llm.structured("drafter", _ADDITIONAL_SUPPORT_SYSTEM, user, _AdditionalSupportExtraction)
    return _clean_external_text(ext.response)


def _manual_hitl_reason(*, attestation_requested: bool, additional_support_requested: bool) -> str:
    if attestation_requested and additional_support_requested:
        return ("Human follow-up is required to provide the requested information and attestation "
                "certificate.")
    if additional_support_requested:
        return ("Human follow-up is required to provide the evidence, documents, or information "
                "requested by the auditor.")
    return "Human follow-up is required to provide the requested attestation certificate."


def _manual_hitl_stage(*, additional_support_requested: bool) -> str:
    return "additional_support_request" if additional_support_requested else "attestation_certificate"


def _append_paragraph(text: str, paragraph: str) -> str:
    text = str(text or "").strip()
    paragraph = str(paragraph or "").strip()
    if not paragraph or paragraph in text:
        return text
    return f"{text}\n\n{paragraph}" if text else paragraph


def _sentences(value: Any) -> list[str]:
    text = _normalize_text(value)
    if not text:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9$])", text) if part.strip()]


def _clip_sentences(value: Any, max_words: int) -> str:
    """Prefer whole sentences; use an ellipsis only for a single pathological model sentence."""
    chosen: list[str] = []
    used = 0
    for sentence in _sentences(value):
        words = sentence.split()
        if used + len(words) <= max_words:
            chosen.append(sentence)
            used += len(words)
        elif not chosen:
            chosen.append(" ".join(words[:max_words]).rstrip(" ,;:") + "…")
            break
        else:
            break
    return " ".join(chosen).strip()


def _questioned_amount(state: dict) -> float:
    for key, field in (("jade_analyzer", "amount_questioned"), ("intake", "dollars")):
        value = _dump(state.get(key)).get(field)
        try:
            if float(value or 0) > 0:
                return float(value)
        except (TypeError, ValueError):
            pass
    return 0.0


def _money(value: float) -> str:
    return f"${float(value):,.2f}"


def _default_conclusion(state: dict, disposition: str) -> str:
    questioned = _questioned_amount(state)
    if disposition == "deny":
        return "This exception is denied. No credit is due."
    if disposition == "accept":
        credit = f" A full credit of {_money(questioned)} is due." if questioned else ""
        return f"This exception is accepted.{credit}".strip()
    supervision = _dump(state.get("supervision"))
    credit_amount = float(supervision.get("credit_amount") or 0)
    credit = f" A credit of {_money(credit_amount)} is due." if credit_amount else ""
    return f"This exception is partially accepted.{credit}".strip()


def _terminal_close(disposition: str) -> str:
    if disposition == "deny":
        return "This exception is denied. No credit is due."
    if disposition == "accept":
        return "This exception is accepted."
    return "This exception is partially accepted."


def _co_owner_close(state: dict, disposition: str) -> str:
    if disposition == "deny":
        return _terminal_close(disposition)
    if disposition == "accept":
        amount = _questioned_amount(state)
        credit = f"A full credit of {_money(amount)} is due. " if amount else ""
        return credit + _terminal_close(disposition)
    credit_amount = float(_dump(state.get("supervision")).get("credit_amount") or 0)
    credit = f"A credit of {_money(credit_amount)} is due. " if credit_amount else ""
    return credit + _terminal_close(disposition)


def _strip_outcome_sentences(value: Any) -> str:
    outcome = re.compile(
        r"\b(?:no credit is due|exception (?:denied|accepted|partially accepted)|"
        r"(?:this|the) exception remains open|"
        r"this exception (?:is|should be) (?:denied|accepted|partially accepted)|"
        r"the exception (?:is|should be) (?:denied|accepted|partially accepted)|"
        r"exception\b.{0,80}\b(?:is|should be) (?:denied|accepted|partially accepted))\b",
        re.IGNORECASE,
    )
    return " ".join(sentence for sentence in _sentences(value) if not outcome.search(sentence)).strip()


def _compose_bounded_response(parts: list[str], close: str, max_words: int) -> str:
    """Join de-duplicated whole sentences under a hard word budget, reserving room for the close."""
    close_words = len(close.split())
    budget = max(1, max_words - close_words)
    chosen: list[str] = []
    fingerprints: set[str] = set()
    used = 0
    for part in parts:
        for sentence in _sentences(_strip_outcome_sentences(part)):
            fingerprint = re.sub(r"\W+", "", sentence).lower()
            words = sentence.split()
            if not fingerprint or fingerprint in fingerprints or used + len(words) > budget:
                continue
            chosen.append(sentence)
            fingerprints.add(fingerprint)
            used += len(words)
    body = " ".join(chosen).strip()
    return f"{body} {close}".strip()


def _without_required_topics(value: Any, *, dates: bool, authority: bool) -> str:
    """Remove model-authored versions of facts that code will append canonically.

    This prevents a vague or incomplete model sentence from sitting beside the exact deterministic
    sentence while preserving the model's amount, role-duty, and business-rule explanation.
    """
    kept: list[str] = []
    for sentence in _sentences(value):
        lower = sentence.lower()
        if dates and (
            "service month" in lower or "location window" in lower or "contrary period" in lower
            or "location records" in lower or "pob" in lower or "helicopter" in lower
            or bool(re.search(r"\b(?:20\d{2}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/20\d{2})\b", sentence))
        ):
            continue
        if authority and (
            "settlement agreement" in lower or "attachment 1" in lower or "30 cfr" in lower
            or "governing provisions" in lower
        ):
            continue
        kept.append(sentence)
    return " ".join(kept)


def _without_auditor_restatement(value: Any) -> str:
    """Remove model-authored recaps of the auditor's request or allegation."""
    return " ".join(
        sentence for sentence in _sentences(value)
        if not re.match(
            r"^(?:(?:the\s+)?auditor(?:'s)?\b|(?:this|the)\s+exception\s+"
            r"(?:questions|concerns|challenges|alleges)\b)",
            sentence,
            flags=re.IGNORECASE,
        )
    ).strip()


def _without_role_conclusions(value: Any) -> str:
    """Keep role facts here and leave their legal implication to the authority paragraph."""
    conclusion_markers = (
        "chargeable", "chargeability", "dispositive", "regardless of location",
        "regardless of where", "irrespective of location", "does not determine chargeability",
        "qualifies under", "meets the definition",
    )
    return " ".join(
        sentence for sentence in _sentences(value)
        if not any(marker in sentence.lower() for marker in conclusion_markers)
    ).strip()


def _argument_order(ext: _DrafterExtraction, state: dict) -> list[str]:
    """Validate the drafter's case-specific ranking without assigning fixed strength in code."""
    available = set(_supported_argument_components(state))
    if ext.amount_analysis:
        available.add("amount_reconciliation")
    order: list[str] = []
    for key in ext.argument_priority:
        if key in available and key not in order:
            order.append(key)
    for key in ext.argument_roles:
        if key in available and key not in order:
            order.append(key)
    # A model that populated the analytical map but omitted the parallel list still communicated an
    # order through JSON insertion order. This is a resilience path, not a hard-coded strength rule.
    for key in ext.argument_analysis:
        if key in available and key not in order:
            order.append(key)
    valid_roles = {
        "direct_refutation": 0,
        "independent_basis": 1,
        "strengthening": 2,
        "corroboration": 3,
    }
    if order:
        # The drafter supplies both the semantic classification and its preferred order. Classification
        # provides the generic hierarchy; the model's order breaks ties. No evidence type is named here.
        indexed = {key: index for index, key in enumerate(order)}
        if any(_argument_role(ext, key) in valid_roles for key in order):
            order.sort(key=lambda key: (
                valid_roles.get(_argument_role(ext, key), len(valid_roles)),
                indexed[key],
            ))
        return order
    # Defensive fallback for a malformed terminal extraction: include every supported component in a
    # stable presentation order, without asserting that the order represents evidentiary strength.
    for key in ("date_coverage", "role_authority", "supervision", "amount_reconciliation"):
        if key in available and key not in order:
            order.append(key)
    return order


def _supported_argument_components(state: dict) -> list[str]:
    """Inventory available components alphabetically, with no evidentiary rank."""
    jade = _dump(state.get("jade_analyzer"))
    labels: set[str] = set()
    if _date_coverage_sentence(state) or _date_evidence_excerpt(state):
        labels.add("date_coverage")
    if _dump(state.get("supervision")):
        labels.add("supervision")
    if _dump(state.get("job_title")) or _dump(state.get("matching")):
        labels.add("role_authority")
    if any(jade.get(field) not in (None, "", 0, [], {}) for field in (
        "nl_summary", "amount_computed", "amount_questioned", "amount_in_range",
    )):
        labels.add("amount_reconciliation")
    return sorted(labels)


def _argument_analysis(ext: _DrafterExtraction, key: str) -> str:
    value = (ext.argument_analysis or {}).get(key, "")
    return _clip_sentences(
        _without_auditor_restatement(_clean_external_text(value)),
        25,
    )


def _published_argument_analysis(ext: _DrafterExtraction, key: str) -> str:
    """Publish only a concise scope limitation needed to connect facts to the primary premise.

    The full analytical map remains in DrafterState for review. Other explanations are already expressed
    by their facts and governing basis, so repeating model-authored advocacy in the client response adds
    length and can overstate what the findings establish.
    """
    relationship = str((ext.premise_relationships or {}).get(key, "")).strip().lower()
    if relationship == "limits_scope":
        return "That limitation means the same conclusion cannot be applied to the full charged population."
    return ""


def _argument_role(ext: _DrafterExtraction, key: str) -> str:
    """Return a role after enforcing the generic premise/argument counterfactual."""
    role = str((ext.argument_roles or {}).get(key, "")).strip().lower()
    relationship = str((ext.premise_relationships or {}).get(key, "")).strip().lower()
    if relationship in {"contradicts", "limits_scope"}:
        return "direct_refutation"
    if role == "direct_refutation" and relationship == "accepts_or_assumes":
        return "independent_basis"
    if role == "direct_refutation" and relationship == "does_not_address":
        return "strengthening"
    return role if role in {
        "direct_refutation", "independent_basis", "strengthening", "corroboration",
    } else ""


def _primary_summary(ext: _DrafterExtraction, state: dict, primary: str, disposition: str) -> str:
    candidate = _without_auditor_restatement(_clean_external_text(ext.executive_summary))
    if candidate:
        return candidate
    analysis = _argument_analysis(ext, primary)
    if analysis:
        return analysis
    if primary == "supervision":
        return "The documented reporting level and governing overhead rule determine the adjustment."
    if primary == "role_authority":
        return "The documented duties satisfy the governing OTL chargeability provisions."
    if disposition == "deny":
        return "The available evidence supports the challenged charge."
    return "The available evidence supports the stated adjustment."


def _role_evidence(ext: _DrafterExtraction, state: dict) -> str:
    job_title = _dump(state.get("job_title"))
    fallback = job_title.get("technical_rationale") or job_title.get("nl_summary", "")
    role = _without_auditor_restatement(_clean_external_text(ext.evidence_basis or fallback))
    role = _without_required_topics(role, dates=True, authority=True)
    role = _without_role_conclusions(role)
    return _clip_sentences(role, 48)


def _role_authority_argument(ext: _DrafterExtraction, state: dict) -> str:
    role = _role_evidence(ext, state)
    analysis = _published_argument_analysis(ext, "role_authority")
    authority = _authority_application(state)
    if authority:
        return f"{role} {analysis} {authority}".strip()
    governing = _without_required_topics(
        _without_auditor_restatement(_clean_external_text(ext.governing_basis)),
        dates=False,
        authority=False,
    )
    return _compose_bounded_response([role, analysis, governing], "", 150)


def _supervision_argument(ext: _DrafterExtraction, state: dict) -> str:
    supervision = _dump(state.get("supervision"))
    evidence = _without_auditor_restatement(
        _clean_external_text(ext.evidence_basis or supervision.get("nl_summary", ""))
    )
    evidence = _clip_sentences(evidence, 60)
    analysis = _published_argument_analysis(ext, "supervision")
    if _has_agreement_chunks(state):
        authority = _without_auditor_restatement(_clean_external_text(ext.governing_basis))
    else:
        authority = _supervision_authority_application(state)
    return f"{evidence} {analysis} {authority}".strip()


def _terminal_sections(ext: _DrafterExtraction, state: dict, disposition: str) -> dict[str, str]:
    """Publish supported argument blocks in strongest-to-weakest order."""
    jade = _dump(state.get("jade_analyzer"))
    order = _argument_order(ext, state)
    primary = order[0] if order else ""
    sections: dict[str, str] = {
        "Executive Summary": _clip_sentences(
            _primary_summary(ext, state, primary, disposition), 55
        )
    }
    blocks = {
        "date_coverage": (
            "Date Coverage",
            f"{_date_coverage_sentence(state) or ext.date_basis} "
            f"{_published_argument_analysis(ext, 'date_coverage')}".strip(),
            125,
        ),
        "supervision": ("Supervision and Governing Basis", _supervision_argument(ext, state), 160),
        "role_authority": (
            "Role and Governing Basis", _role_authority_argument(ext, state), 190,
        ),
        "amount_reconciliation": (
            "Amount",
            ext.amount_analysis or jade.get("nl_summary", ""),
            70,
        ),
    }
    for key in order:
        title, value, limit = blocks[key]
        cleaned = _clip_sentences(_clean_external_text(value), limit)
        if cleaned:
            sections[title] = cleaned
    sections["Conclusion"] = _compose_bounded_response(
        [ext.conclusion or _default_conclusion(state, disposition)],
        _terminal_close(disposition),
        45,
    )
    return sections


def _render_sections(sections: dict[str, str]) -> str:
    return "\n\n".join(f"## {title}\n{text}" for title, text in sections.items() if text.strip())


def _response_heading(
    disposition: str,
    *,
    review_pending: bool = False,
    insufficient_auditor_evidence: bool = False,
    supporting_documentation_provided: bool = False,
) -> str:
    """Select the client-facing response type without asking the model to classify it."""
    if insufficient_auditor_evidence:
        return _INSUFFICIENT_AUDITOR_EVIDENCE_HEADING
    if review_pending or disposition in {"needs_evidence", "budget_exceeded"}:
        return _RESPONSE_WITHHELD_HEADING
    if disposition == "accept":
        return _CREDIT_GRANTED_HEADING
    if disposition == "partial":
        return _CREDIT_PARTIALLY_GRANTED_HEADING
    if supporting_documentation_provided:
        return _CREDIT_DENIED_DOCUMENTATION_HEADING
    return _CREDIT_DENIED_HEADING


def _publish_response_sections(heading: str, sections: dict[str, str]) -> dict[str, str]:
    """Wrap detailed analysis beneath exactly one response-type heading.

    ``response_sections`` is the report/API contract. The prior detailed headings remain available as
    level-three subsections inside the selected response type, preserving the existing analysis while
    making the top-level format stable across terminal, evidence-gate and HITL paths.
    """
    populated = [(title, text.strip()) for title, text in sections.items() if str(text).strip()]
    if not populated:
        return {}
    if len(populated) == 1 and populated[0][0] == "Response":
        body = populated[0][1]
    else:
        body = "\n\n".join(f"### {title}\n{text}" for title, text in populated)
    return {heading: body}


def _compose_ranked_units(
    units: list[tuple[str, bool]],
    close: str,
    guideline_words: int,
) -> str:
    """Compose ordered units under a soft guideline while never dropping mandatory reasoning."""
    prepared = [(_strip_outcome_sentences(text), mandatory) for text, mandatory in units if text]
    close_words = len(close.split())
    mandatory_words = sum(len(text.split()) for text, required in prepared if required)
    # Material facts/analysis set the real budget when they exceed the target. Component-level limits
    # still prevent essays, but there is intentionally no publication hard-stop at 220 words.
    budget = max(guideline_words, mandatory_words + close_words)
    selected: list[str] = []
    used = 0
    remaining_required = mandatory_words
    for text, required in prepared:
        words = len(text.split())
        if required:
            selected.append(text)
            used += words
            remaining_required -= words
        elif used + words + remaining_required + close_words <= budget:
            selected.append(text)
            used += words
    return " ".join([*selected, close]).strip()


def _terminal_co_owner(
    ext: _DrafterExtraction,
    sections: dict[str, str],
    disposition: str,
    state: dict,
) -> str:
    """Compose the outward paragraph from already ranked, source-grounded sections."""
    close = _co_owner_close(state, disposition)
    units: list[tuple[str, bool]] = []
    section_for = {
        "date_coverage": "Date Coverage",
        "supervision": "Supervision and Governing Basis",
        "role_authority": "Role and Governing Basis",
        "amount_reconciliation": "Amount",
    }
    for key in _argument_order(ext, state):
        text = sections.get(section_for[key], "")
        role = _argument_role(ext, key)
        if not text:
            continue
        if role != "corroboration":
            units.append((text, True))
        elif key != "amount_reconciliation":
            # A distinct fact-bearing corroborative block may add a useful second layer when it fits the
            # soft word guideline. Arithmetic corroboration stays in the technical response unless the
            # amount itself is classified as a substantive argument; it should not pad client prose.
            units.append((text, False))
    response = _compose_ranked_units(units, close, _CO_OWNER_GUIDELINE_WORDS)
    if _INTERNAL_EXTERNAL_MARKER.search(response):
        response = _clean_external_text(response)
    return response


def _fallback_hitl_nl_summary(nl_summaries: dict) -> str:
    summaries = [
        str(stage.get("nl_summary") or "") for stage in nl_summaries.values()
        if isinstance(stage, dict) and stage.get("nl_summary")
    ]
    return " ".join(summaries[-4:])


def _hitl_parts(ext: _DrafterExtraction, nl_summaries: dict, why_stopped: str) -> dict[str, str]:
    parts = {
        "What the review established": _clip_sentences(
            _strip_outcome_sentences(
                _clean_external_text(ext.hitl_nl_summary or _fallback_hitl_nl_summary(nl_summaries))
            ), 110),
        "What remains unresolved": _clip_sentences(
            _strip_outcome_sentences(_clean_external_text(ext.hitl_blocker or why_stopped)), 70),
        "Why it matters": _clip_sentences(
            _strip_outcome_sentences(_clean_external_text(
                ext.hitl_impact or (
                    "Without resolving that question, the proper treatment and any resulting adjustment "
                    "cannot be determined."
                )
            )), 55),
        "What is needed to complete the review": _clip_sentences(
            _strip_outcome_sentences(
            _clean_external_text(ext.hitl_next_step or
                                 "The outstanding issue must be resolved before the exception can be finalized.")
            ), 85),
    }
    return {title: text for title, text in parts.items() if text}


def _hitl_co_owner(parts: dict[str, str]) -> str:
    """Assemble every required auditor-facing element; never trust a free-form model paragraph."""
    close = "This exception remains open pending resolution of that issue."
    response = _compose_ranked_units(
        [(text, True) for text in parts.values()],
        close,
        180,
    )
    return _clean_external_text(response)


def drafter_node(state: dict) -> dict:
    budget_recovery = bool(state.get("budget_exceeded"))
    nl_summaries = _nl_summary_digest(state)
    questions = set(state.get("auditor_question") or [])
    verify_decision = _dump(state.get("evidence_gate_decision"))
    jade_decision = _dump(state.get("jade_hook_decision"))
    decision = jade_decision if jade_decision.get("triggered") else verify_decision
    ve = _dump(state.get("verify_evidence"))
    attestation_requested = bool(state.get("attestation_requested"))
    third_party_invoice_requested = bool(
        state.get("third_party_invoice_requested")
        or "third_party_invoice_requested" in questions
    )
    additional_support_requested = bool(state.get("additional_support_requested"))
    ed04_triggered = "ED-04" in set(verify_decision.get("gate_ids") or [])
    effective_third_party_request = third_party_invoice_requested and not ed04_triggered
    information_request = additional_support_requested or effective_third_party_request
    manual_followup_requested = attestation_requested or information_request
    request_only = manual_followup_requested and not questions.intersection(_ANALYTIC_QUESTIONS)
    # Budget recovery is deliberately one terminal LLM call. Separate attestation/information calls
    # are skipped so the hook cannot re-enter before the final HITL response is produced.
    attestation_response = "" if budget_recovery else _attestation_response(state)
    additional_support_response = (
        "" if budget_recovery else
        _additional_support_response(state, requested=information_request)
    )
    manual_response = _append_paragraph(attestation_response, additional_support_response)
    manual_stage = (
        _manual_hitl_stage(additional_support_requested=information_request)
        if manual_followup_requested else ""
    )
    manual_reason = (
        _manual_hitl_reason(
            attestation_requested=attestation_requested,
            additional_support_requested=information_request,
        )
        if manual_followup_requested else ""
    )

    # 1) EVIDENCE-GATE (checked first, unambiguous): the hard gate fired upstream, so nothing after it
    #    ran. Missing auditor support denies the exception. Inaccessible supplied evidence remains a
    #    genuine deferral because a human must resolve access before a determination is safe.
    if decision.get("triggered") and not budget_recovery:
        if decision.get("outcome_key") == "deny_jade_amount_mismatch":
            jade = _dump(state.get("jade_analyzer"))
            questioned = float(jade.get("amount_questioned") or 0.0)
            amount_text = _money(questioned)
            msg = (
                "The total amount provided in the JADE does not reconcile to the questioned amount of "
                f"{amount_text}. This exception is denied. No credit is due."
            )
            msg = _append_paragraph(msg, manual_response)
            hitl = manual_followup_requested
            response_sections = _publish_response_sections(
                _response_heading("deny", review_pending=hitl),
                {"Response": msg},
            )
            reasons = [str(value) for value in decision.get("reasons", []) or [] if str(value).strip()]
            sub = DrafterState(
                nl_summary=(reasons[0] if reasons else
                          "JADE/general-ledger amounts do not reconcile to the auditor's stated amount."),
                disposition="needs_evidence" if hitl else "deny",
                draft_response=_render_sections(response_sections),
                response_sections=response_sections,
                co_owner_response=msg,
                attestation_response=attestation_response,
                additional_support_response=additional_support_response,
                hitl_note=msg if hitl else "",
            )
            update: dict = {"drafter": sub, "hitl_review": hitl}
            if hitl:
                update["hitl_stage"] = manual_stage
                update["hitl_reason"] = manual_reason
            update.update(trace_event(
                state, "drafter", "drafted denial for JADE amount mismatch",
                inputs_digest=(f"gate=fail (jade_analyzer), questioned={questioned}, "
                               f"outcome={decision.get('outcome_key')}"),
                key_outputs={"disposition": sub.disposition,
                             "triggered_gate_ids": list(decision.get("gate_ids") or []),
                             "outcome_key": decision.get("outcome_key", ""),
                             "hitl_review": hitl},
                nl_summary=sub.nl_summary,
            ))
            return update

        questions = state.get("auditor_question") or []   # set by category_check (master channel)
        gap = state.get("evidence_gap") or []
        required = list(decision.get("required_evidence") or _required_evidence(ve, questions, gap))
        subject = "supporting documentation" if decision else _gate_subject(gap)
        decision_status = str(decision.get("status") or "")
        gate_status = str(state.get("evidence_gate_status") or (
            "unreadable_auditor_evidence" if decision_status == "unreadable"
            else "missing_auditor_evidence"
        ))
        unreadable = decision_status == "unreadable" or gate_status == "unreadable_auditor_evidence"
        msg = (_unreadable_evidence_response(required, subject)
               if unreadable else _unsupported_evidence_denial(required, subject))
        msg = _append_paragraph(msg, manual_response)
        response_sections = _publish_response_sections(
            _response_heading(
                "needs_evidence" if (unreadable or manual_followup_requested) else "deny",
                review_pending=unreadable,
                insufficient_auditor_evidence=not unreadable,
            ),
            {"Response": msg},
        )
        sub = DrafterState(
            nl_summary=(f"Supplied {subject} could not be opened or accessed; human follow-up is required."
                      if unreadable else
                      f"Exception denied: sufficient {subject} was not provided."),
            disposition="needs_evidence" if (unreadable or manual_followup_requested) else "deny",
            draft_response=_render_sections(response_sections),
            response_sections=response_sections,
            co_owner_response=msg,
            attestation_response=attestation_response,
            additional_support_response=additional_support_response,
            hitl_note=msg if (unreadable or manual_followup_requested) else "",
        )
        hitl = unreadable or manual_followup_requested
        update: dict = {"drafter": sub, "hitl_review": hitl}
        if unreadable and not manual_followup_requested:
            update["hitl_stage"] = "verify_evidence" if decision else "jade_analyzer"
            update["hitl_reason"] = "The supplied evidence could not be opened or accessed."
        elif manual_followup_requested:
            update["hitl_stage"] = manual_stage
            update["hitl_reason"] = manual_reason
        gate_source = "verify_evidence" if decision else ("jade_analyzer" if gap else "verify_evidence")
        update.update(trace_event(
            state, "drafter",
            "drafted HITL accessibility response (evidence gate)" if unreadable
            else "drafted denial for unsupported evidence (evidence gate)",
            inputs_digest=(f"gate=fail ({gate_source}), "
                           f"status={gate_status}, "
                           f"pob={ve.get('pob_data_found')}, heli={ve.get('heli_data_found')}, "
                           f"{len(required)} required source(s)"),
            key_outputs={"disposition": sub.disposition, "required_evidence": required,
                         "evidence_gate_status": gate_status,
                         "triggered_gate_ids": list(decision.get("gate_ids") or []),
                         "outcome_key": decision.get("outcome_key", ""),
                         "hitl_review": hitl},
            nl_summary=sub.nl_summary,
        ))
        return update

    # No labor specialist or legal-retrieval path is needed for a pure document/information request.
    if request_only and not budget_recovery:
        response_sections = _publish_response_sections(
            _response_heading("needs_evidence", review_pending=True),
            {"Response": manual_response},
        )
        sub = DrafterState(
            nl_summary=manual_reason,
            disposition="needs_evidence",
            draft_response=_render_sections(response_sections),
            response_sections=response_sections,
            co_owner_response=manual_response,
            attestation_response=attestation_response,
            additional_support_response=additional_support_response,
            hitl_note=manual_response,
        )
        update = {
            "drafter": sub,
            "hitl_review": True,
            "hitl_stage": manual_stage,
            "hitl_reason": manual_reason,
        }
        update.update(trace_event(
            state, "drafter", "drafted document/information-request response",
            inputs_digest="request-only category; no labor specialist route",
            key_outputs={"disposition": "needs_evidence", "hitl_review": True,
                         "attestation_requested": attestation_requested,
                         "third_party_invoice_requested": effective_third_party_request,
                         "additional_support_requested": additional_support_requested},
            nl_summary=sub.nl_summary,
        ))
        return update

    # 2) analysis ran — TERMINAL rebuttal or a HITL deferral note keyed off the orchestrator's decision.
    hitl = (bool(state.get("hitl_review"))
            or (state.get("final_mode") == "needs_evidence")
            or questions == {"other"})
    if hitl:
        why_stopped = state.get("hitl_reason") or (
            "The auditor's question could not be mapped to a supported analysis category; "
            "human clarification is required to identify the appropriate review path."
            if questions == {"other"}
            else "a specialist stage could not resolve its question."
        )
        user = _hitl_user(state, nl_summaries, why_stopped)
    else:
        user = _terminal_user(state, nl_summaries)

    ext = llm.structured(
        "drafter", DRAFTER_SYSTEM, user, _DrafterExtraction,
        allow_budget_overrun=budget_recovery,
    )

    if hitl:
        parts = _hitl_parts(ext, nl_summaries, why_stopped)
        if attestation_response:
            parts["Attestation Certificate"] = attestation_response
        if additional_support_response:
            parts["Requested Information"] = additional_support_response
        response_sections = _publish_response_sections(
            _response_heading(
                "budget_exceeded" if budget_recovery else "needs_evidence",
                review_pending=True,
            ),
            parts,
        )
        note = _render_sections(response_sections)
        co_owner = _hitl_co_owner(parts)
        sub = DrafterState(
            nl_summary=ext.nl_summary or f"Deferred at {state.get('hitl_stage') or 'orchestrator'}.",
            disposition="budget_exceeded" if budget_recovery else "needs_evidence",
            draft_response=note if budget_recovery else "",
            response_sections=response_sections,
            co_owner_response=co_owner,
            attestation_response=attestation_response,
            additional_support_response=additional_support_response,
            hitl_note=note,
        )
    else:
        locked_disposition = str(_chargeability_resolution(state).get("disposition") or "").lower()
        disposition = locked_disposition or ext.disposition.strip().lower()
        if disposition not in _TERMINAL_DISPOSITIONS:
            disposition = "deny"
        final_argument_order = _argument_order(ext, state)
        sections = _terminal_sections(ext, state, disposition)
        co_owner = _terminal_co_owner(ext, sections, disposition, state)
        if attestation_response:
            sections["Attestation Certificate"] = attestation_response
            co_owner = _append_paragraph(co_owner, attestation_response)
        if additional_support_response:
            sections["Requested Information"] = additional_support_response
            co_owner = _append_paragraph(co_owner, additional_support_response)
        response_sections = _publish_response_sections(
            _response_heading(
                disposition,
                review_pending=manual_followup_requested,
                supporting_documentation_provided=bool(
                    state.get("supporting_documentation_provided")
                ),
            ),
            sections,
        )
        sub = DrafterState(
            nl_summary=ext.nl_summary or f"Terminal rebuttal: {disposition}.",
            disposition="needs_evidence" if manual_followup_requested else disposition,
            draft_response=_render_sections(response_sections),
            response_sections=response_sections,
            attestation_response=attestation_response,
            additional_support_response=additional_support_response,
            challenged_premise=" ".join(str(ext.challenged_premise).split()),
            argument_priority=final_argument_order,
            argument_roles={
                key: role for key in final_argument_order
                if (role := _argument_role(ext, key))
            },
            premise_relationships={
                key: relationship for key in final_argument_order
                if (relationship := str(
                    (ext.premise_relationships or {}).get(key, "")
                ).strip().lower()) in {
                    "contradicts", "limits_scope", "accepts_or_assumes", "does_not_address",
                }
            },
            argument_analysis={
                key: analysis for key in final_argument_order
                if (analysis := _argument_analysis(ext, key))
            },
            co_owner_response=co_owner,
        )

    update = {"drafter": sub}
    if hitl:
        # Backfill the HITL channels so the report always carries them when disposition=needs_evidence.
        update["hitl_review"] = True
        update["hitl_reason"] = why_stopped
        update["hitl_stage"] = state.get("hitl_stage") or "orchestrator"
    if manual_followup_requested and not budget_recovery:
        update["hitl_review"] = True
        update["hitl_reason"] = manual_reason
        update["hitl_stage"] = manual_stage
    update.update(trace_event(
        state, "drafter",
        ("drafted budget-exceeded HITL deferral note" if budget_recovery else
         "drafted HITL deferral note" if hitl else "drafted terminal rebuttal"),
        inputs_digest=(f"mode={'hitl' if hitl else 'terminal'}, "
                       f"stages={[k for k in nl_summaries]}, "
                       f"gate_triggered={bool(decision.get('triggered'))}"),
        key_outputs={"disposition": sub.disposition,
                     "has_hitl_note": bool(sub.hitl_note),
                     "sections": list(sub.response_sections.keys()),
                     "argument_priority": list(sub.argument_priority),
                     "argument_roles": dict(sub.argument_roles),
                     "premise_relationships": dict(sub.premise_relationships),
                     "co_owner_words": len(sub.co_owner_response.split())},
        nl_summary=sub.nl_summary,
    ))
    return update


__all__ = ["drafter_node"]
