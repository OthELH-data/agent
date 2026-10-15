"""intake_node — deterministic case facts plus a light-tier analysis of extractor output.

The intake model receives the extractor's cross-attachment synthesis and compact attachment analysis,
then produces a concise narrative plus a non-authoritative list of explicitly named people. JADE resolves
those leads alongside ledger values and POS suggestions into the downstream person population. Assets and
dollars come from the resolved case state, while ``collected_evidence`` comes from resolved workbooks and
supporting PDFs after response-name filtering.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from code_classifier import classify
from otl import llm_adapter as llm
from otl.agreements.corpus import canonical_asset
from otl.prompts import INTAKE_SYSTEM
from otl.state import IntakeState
from otl.trace import trace_event
from otl.agents._common import (active_auditor_context_block, compact_synthesis, error_type_of,
                                seed_dollars, synthesis_narrative)


def _asset_pairs(values, fallback: str = "") -> list[tuple[str, str]]:
    """Return unique ``(agreement_corpus_key, source_display_name)`` pairs."""
    candidates = list(values or []) if isinstance(values, (list, tuple)) else []
    if fallback:
        candidates.append(fallback)
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in candidates:
        as_seen = str(raw or "").strip()
        canonical = canonical_asset(as_seen)
        if canonical and canonical not in seen:
            seen.add(canonical)
            out.append((canonical, as_seen))
    return out


def _filter_assets(values, fallback: str = "") -> list[str]:
    """Compatibility helper returning canonical asset keys only."""
    return [canonical for canonical, _as_seen in _asset_pairs(values, fallback)]


def _resolved_evidence(state: dict) -> list[str]:
    """Stable names for extractor-resolved, auditor-side workbook and PDF attachments."""
    paths = [
        *(state.get("exception_exhibits") or []),
        *(state.get("supporting_pdfs") or []),
    ]
    names = [Path(str(path or "")).name for path in paths]
    return list(dict.fromkeys(str(name) for name in names if str(name or "").strip()))


class _IntakeSummary(BaseModel):
    nl_summary: str = Field(
        default="",
        description="A concise but substantive analysis of the notable intake facts, relevant evidence "
                    "context, and any material uncertainty. Do not list every detail.")
    people_found: list[str] = Field(
        default_factory=list,
        description="Distinct people explicitly named in the supplied extractor synthesis or attachment "
                    "analysis. Preserve the supplied spelling. Exclude organizations, roles, and inferred "
                    "or guessed names. This is a lead list only; JADE resolves it with ledger and POS "
                    "evidence before it becomes the downstream population.")


def intake_node(state: dict) -> dict:
    case = state.get("case") or {}
    cycle = int(case.get("rebuttal_cycle", state.get("rebuttal_cycle", 0)) or 0)
    latest_reply = str((state.get("latest_auditor_reply") or {}).get("text") or "").strip()
    narrative_parts = [active_auditor_context_block(state).strip()]
    extractor_output = state.get("synthesis")
    extractor_output = extractor_output if isinstance(extractor_output, dict) else {}
    synthesis = synthesis_narrative(state) if extractor_output else ""
    attachment_analysis = compact_synthesis(extractor_output) if extractor_output else ""
    if synthesis:
        narrative_parts.append(f"Extractor synthesis:\n{synthesis}")
    if attachment_analysis:
        narrative_parts.append(f"Extractor attachment analysis (compacted, row-free):\n{attachment_analysis}")
    user = (
        "\n\n".join(narrative_parts) + "\n\n"
        "Write a concise but substantive analysis of the exception: state the notable issue, material "
        "context, and any relevant uncertainty or open point. Do not identify assets, calculate an amount, "
        "list evidence, or state a disposition. Do not repeat every fact. Separately populate people_found "
        "with every distinct person explicitly named in the extractor synthesis or attachment analysis. "
        "Preserve source spelling; do not infer names, expand initials, or include organizations, roles, or "
        "generic references to employees. The list is a lead list only and does not establish that a person "
        "was charged to the exception."
    )
    ext = llm.structured("intake", INTAKE_SYSTEM, user, _IntakeSummary)

    dollars = seed_dollars(state)
    code = classify(error_type_of(state))
    asset_pairs = _asset_pairs([], fallback=(state.get("case") or {}).get("asset", ""))
    assets = [canonical for canonical, _as_seen in asset_pairs]
    assets_as_seen = [as_seen for _canonical, as_seen in asset_pairs]
    evidence = _resolved_evidence(state)
    sub = IntakeState(
        nl_summary=ext.nl_summary,
        assets_in_question=assets,
        assets_as_seen=assets_as_seen,
        dollars=dollars,
        collected_evidence=evidence,
        error_code=code["code"],
        code_family=code["family"],
        people_found=ext.people_found,
    )

    update: dict = {"intake": sub}
    update.update(trace_event(
        state, "intake", "analyzed extractor synthesis and attachment summaries alongside case facts",
        inputs_digest=(f"cycle={cycle}; original narrative plus "
                       f"{'latest auditor reply' if latest_reply else 'no latest auditor reply'}; "
                       f"{'extractor synthesis' if synthesis else 'no extractor synthesis'} and "
                       "compacted attachment analysis; assets, amount, and evidence derived in code"),
        key_outputs={"assets": sub.assets_in_question, "assets_as_seen": sub.assets_as_seen,
                     "dollars": sub.dollars,
                     "evidence": sub.collected_evidence,
                     "error_code": sub.error_code,
                     "code_family": sub.code_family,
                     "people_found": sub.people_found},
        nl_summary=sub.nl_summary or (
            f"{len(sub.assets_in_question)} supported asset(s), ${sub.dollars:,.2f}"),
    ))
    return update


__all__ = ["_IntakeSummary", "_filter_assets", "_resolved_evidence", "intake_node"]
