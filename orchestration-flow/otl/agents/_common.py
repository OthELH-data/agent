"""Shared helpers for the spine nodes (intake, category_check, verify_evidence).

These keep raw workbook rows out of the prompt. The classified-exceptions record supplies the baseline
case facts; an enabled and locally available extractor synthesis adds interpretive tab metadata and the
extractor's account of how the auditor built the amount. Only code-exec specialists touch full cells.

They are also where the spine drops uncited operator material. Classified attachment names are filtered
while the catalog is normalized; staged synthesis attachments are reconciled with the extractor evidence
manifest here at read time so auditor-cited workbooks remain available.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from otl.response_filter import drop_responses, is_operator_response, is_response_tab
from shared.rebuttal_history import prompt_rebuttal_context

_AMOUNT_RE = re.compile(r"-?\$?\s*[\d,]+(?:\.\d+)?")
_VERSION_RE = re.compile(r"[\s._-]*(?:v[\s._]?(\d{1,3})|\((\d{1,3})\))\s*$", re.I)


def sub_get(state: dict, key: str, field: str, default: Any = None) -> Any:
    """Read `field` off a sub-state that may be a Pydantic model OR a plain dict (post-checkpoint)."""
    obj = state.get(key)
    if obj is None:
        return default
    if hasattr(obj, field):
        return getattr(obj, field)
    if isinstance(obj, dict):
        return obj.get(field, default)
    return default


MAX_DISPUTED_PEOPLE = 0            # no cap when --max-people is not passed
_max_people = MAX_DISPUTED_PEOPLE


def max_disputed_people() -> int:
    """How many disputed people each specialist examines: all of them unless explicitly capped.

    The ONLY knob is the CLI flag — deliberately not env/.env-configurable, so the cap a run used is
    visible in the command that launched it rather than in ambient state. Read at CALL time (not
    import), so `set_max_disputed_people` takes effect for the whole run. The default and any value <= 0
    mean "no cap".
    """
    return _max_people


def set_max_disputed_people(n: int) -> int:
    """Apply the `--max-people` flag. Returns the applied value (<= 0 meaning "no cap").

    Call once at startup, before the graph runs; `cap_people` re-reads it per call. This is a process
    global, so it does NOT cross a process boundary: `run_batch.py` forwards `--max-people` on each
    worker's command line (see `_run_case`) rather than relying on inheritance.
    """
    global _max_people
    _max_people = int(n)
    return _max_people


_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def _name_tokens(name: str) -> frozenset[str]:
    """The words that identify a person, order-free: lowercased, punctuation dropped, initials and
    generational suffixes removed.

    Order-free is the point. The two sources write the same person differently — the auditor writes
    "Kenneth Oyler" and the ledger's line-item text carries "Oyler, Kenneth Willmann" — and neither the
    word order nor the comma tells us which word is the surname. A word SET sidesteps the question.

    Single letters go because they are initials, not names ("K. Oyler" should not read as its own
    person). That leaves a name written entirely in initials with no tokens at all, so callers must
    handle the empty set rather than treat it as a match.
    """
    words = re.sub(r"[^0-9a-z]+", " ", name.lower()).split()
    return frozenset(w for w in words if len(w) > 1 and w not in _NAME_SUFFIXES)


def cap_people(names: Any) -> tuple[list[str], list[str]]:
    """Dedupe (one entry per person, first spelling wins), optionally applying an explicit cap.

    Returns (kept, dropped). Every ReAct specialist does per-person work over large workbooks (a
    staff-listing stream, a catalogue lookup, an org-tree walk), so callers may explicitly cap the list
    with `--max-people`. The normal path examines everyone; the budget hook is responsible for stopping
    a run that exceeds its configured budget. Dedupe runs first, so several spellings of ONE person
    cannot consume the budget by themselves.

    Two names are the same person when one's word set contains the other's: "Kenneth Oyler" and
    "Oyler, Kenneth Willmann" collapse, while "Adams, Nancy" and "Adams, Sheila" do not. The rule is
    applied only when it is **unambiguous** — if a name relates to two or more names already seen (a
    bare "Adams" against both Adamses), it is kept as its own entry instead. Guessing which one it meant
    could drop a real person from the whole rebuttal; a redundant entry only costs steps.
    """
    kept: list[str] = []
    dropped: list[str] = []
    seen: set[str] = set()
    token_sets: list[frozenset[str]] = []
    limit = max_disputed_people()
    for raw in (names if isinstance(names, (list, tuple)) else []):
        name = str(raw).strip()
        if not name:
            continue
        key = " ".join(name.lower().split())
        if key in seen:
            continue
        tokens = _name_tokens(name)
        if tokens and len([t for t in token_sets if t <= tokens or tokens <= t]) == 1:
            continue                      # one earlier name, same person — its spelling already stands
        seen.add(key)
        if tokens:
            token_sets.append(tokens)
        (kept if (limit <= 0 or len(kept) < limit) else dropped).append(name)
    return kept, dropped


def disputed_people(state: dict) -> list[str]:
    """The final resolved people a specialist should examine, with any explicit cap re-applied."""
    ledger_names = sub_get(state, "jade_analyzer", "people_found_in_ledger", []) or []
    kept, _ = cap_people(ledger_names)
    return kept


def people_coverage(state: dict) -> tuple[list[str], list[str], int]:
    """Derive examined, cap-overflow, and total counts from the sole stored resolved person list."""
    ledger_names = sub_get(state, "jade_analyzer", "people_found_in_ledger", []) or []
    kept, dropped = cap_people(ledger_names)
    return kept, dropped, len(kept) + len(dropped)


def agent_directive(state: dict, node: str) -> str:
    """The orchestrator's focused directive for this specialist run ('' if none was written)."""
    return str((state.get("agent_instructions") or {}).get(node, "") or "")


def directive_block(state: dict, node: str) -> str:
    """A promptable block carrying the orchestrator's directive — empty when there is none.

    Appended to a specialist's ReAct task (jade/location/job_title) or user prompt (matching) so the
    hub can steer a run (e.g. re-run with a sharper focus) without overriding the agent's own rules.
    """
    d = agent_directive(state, node)
    if not d:
        return ""
    return ("\n\nORCHESTRATOR DIRECTIVE (focus hint for THIS run — bounded by your own rules; never "
            f"fabricate to satisfy it): {d}")


def _active_cycle(state: dict) -> int:
    case = state.get("case") or {}
    for value in (state.get("rebuttal_cycle"), case.get("rebuttal_cycle"),
                  (state.get("synthesis") or {}).get("rebuttal_cycle")):
        if value not in (None, ""):
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return 0


def active_auditor_text(state: dict, limit: int = 12_000) -> str:
    """The active baseline plus any later auditor reply, without interpretive instructions.

    Search/retrieval callers need the auditor's words rather than the prompt framing. The original
    exception is always retained; a later reply is appended because it may continue, narrow, replace,
    withdraw, or resolve part of that baseline.
    """
    original = str(state.get("auditor_narrative") or "").strip()
    latest = str((state.get("latest_auditor_reply") or {}).get("text") or "").strip()
    parts = [value for value in (original, latest) if value]
    text = "\n\n".join(parts)
    if len(text) <= limit:
        return text
    if not latest:
        return text[: limit - 1].rstrip() + "…"
    # Preserve both surfaces under pressure; the latest reply often contains the cycle-specific delta.
    original_limit = max(1, int(limit * 0.6))
    latest_limit = max(1, limit - original_limit - 3)
    return original[:original_limit].rstrip() + " …\n\n" + latest[:latest_limit].rstrip() + "…"


def active_auditor_context_block(state: dict, limit: int = 12_000) -> str:
    """Prompt framing shared by every agent that reasons over baseline/reply chronology."""
    cycle = _active_cycle(state)
    original = str(state.get("auditor_narrative") or "").strip()
    latest = str((state.get("latest_auditor_reply") or {}).get("text") or "").strip()
    if not original and not latest:
        return ""
    if latest:
        chronology = (
            "A later auditor reply exists. Read it together with the active original baseline. The "
            "reply may clarify, narrow, continue, replace, withdraw, or resolve a baseline request; "
            "infer that effect only from explicit wording and the supplied current-cycle context."
        )
    elif cycle == 0:
        chronology = (
            "This is rebuttal cycle 0. The original exception baseline is the active auditor request. "
            "No later auditor reply exists, and that empty field does not make the baseline historical, "
            "resolved, withdrawn, or inactive."
        )
    else:
        chronology = (
            f"This is rebuttal cycle {cycle}, and no later auditor reply is present. The absence of a "
            "reply does not by itself erase the active original baseline. Treat a baseline request as "
            "withdrawn, resolved, replaced, or inactive only when the supplied current-cycle or prior-"
            "cycle context explicitly establishes that result."
        )
    original_text = original or "(none supplied)"
    latest_text = latest or "(none supplied — apply the chronology rule above)"
    block = (
        "\n\n<ACTIVE_AUDITOR_REQUEST_CONTEXT>\n"
        f"CHRONOLOGY RULE:\n{chronology}\n\n"
        f"ACTIVE ORIGINAL EXCEPTION BASELINE:\n{original_text}\n\n"
        f"LATEST AUDITOR REPLY FOR CYCLE {cycle}:\n{latest_text}\n"
        "</ACTIVE_AUDITOR_REQUEST_CONTEXT>"
    )
    if len(block) <= limit:
        return block
    text = active_auditor_text(state, max(1, limit - 800))
    return (
        "\n\n<ACTIVE_AUDITOR_REQUEST_CONTEXT>\n"
        f"CHRONOLOGY RULE:\n{chronology}\n\n"
        f"ACTIVE AUDITOR TEXT (bounded):\n{text}\n"
        "</ACTIVE_AUDITOR_REQUEST_CONTEXT>"
    )


def synthesis_narrative(state: dict, limit: int = 6000) -> str:
    """The extractor's free-text `synthesis` — its interpretive account of HOW the auditor built the
    exception (the amount layers, the affiliate named, the WBS/cost objects the charge was booked to).

    Distinct from `state["auditor_narrative"]` (the auditor's own classified words) and from
    `compact_synthesis` (the structural attachment/tab inventory): this is the extractor's reasoning,
    the single richest orienting paragraph in the packet. Trimmed to `limit` chars; '' when absent.
    """
    s = str((state.get("synthesis") or {}).get("synthesis", "") or "").strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


def synthesis_block(state: dict, limit: int = 6000, specialist: str = "") -> str:
    """A labelled prompt block carrying the extractor's synthesis narrative — '' when absent.

    Appended to spine prompts and specialist tasks so every reasoning node shares the extractor's
    account of the exception. It is ORIENTING context (how the auditor got to the figure), never a
    substitute for the evidence a specialist must reconstruct from the workbooks.
    """
    s = synthesis_narrative(state, limit)
    synthesis = ""
    if s:
        synthesis = (
            "\n\nEXTRACTOR SYNTHESIS (the extractor's interpretive account of how the auditor built "
            f"this exception — orienting context, NOT a substitute for the cited evidence):\n{s}"
        )
    return synthesis + rebuttal_context_block(state, specialist)


def compact_classified_exception(state: dict) -> str:
    """Bounded structured view of the classified-exceptions baseline, excluding response prose.

    The catalog's auditor description is already passed separately through ``state["auditor_narrative"]``.
    Evidence attachment identity is carried independently by the final manifest-derived state fields,
    rather than by the classified catalog.
    """
    record = state.get("classified_exception") or {}
    if not isinstance(record, dict) or not record:
        return "{}"
    classification = record.get("classification") or {}
    view = {
        "error_label": record.get("error_label", ""),
        "error_type": record.get("error_type", ""),
        "error_major": record.get("error_major", ""),
        "matched_category": record.get("matched_category", ""),
        "short_description": record.get("short_description", ""),
        "gross_credit_requested": record.get("gross_credit_requested", ""),
        "classification": classification if isinstance(classification, dict) else {},
        "table_references": record.get("table_references", []) or [],
    }
    return json.dumps(view, default=str, ensure_ascii=False)


def classified_exception_block(state: dict, limit: int = 6000) -> str:
    """Prompt block for the authoritative classified-exceptions case record."""
    text = compact_classified_exception(state)
    if text == "{}":
        return ""
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return (
        "\n\nCLASSIFIED EXCEPTION RECORD (baseline case facts):\n"
        + text
    )


def case_context_block(state: dict, limit: int = 6000, specialist: str = "") -> str:
    """Shared case context: classified baseline plus optional extractor synthesis overlay."""
    return active_auditor_context_block(state, limit=max(limit, 8_000)) + classified_exception_block(
        state, limit=limit
    ) + synthesis_block(
        state, limit=limit, specialist=specialist
    )


def rebuttal_context_block(state: dict, specialist: str = "") -> str:
    """Clearly separate current-cycle grounds from full or specialist-filtered prior analysis."""
    if not state.get("rebuttal_history"):
        return ""
    current_posture = str(state.get("rebuttal_current_posture") or "").strip()
    if specialist == "orchestrator":
        routing = {
            "prior_agents": state.get("rebuttal_prior_agents") or [],
            "specialist_revisit_allowed": state.get("rebuttal_specialist_revisit") or {},
            "current_questioning_grounds": state.get("rebuttal_specialist_current_grounds") or {},
            "revisit_rationales": state.get("rebuttal_specialist_rationales") or {},
            "reusable_prior_nl_summaries": state.get("rebuttal_reused_nl_summaries") or {},
        }
        return prompt_rebuttal_context(
            json.dumps(routing, indent=2, ensure_ascii=False, default=str),
            current_context=current_posture or "No current posture analysis was produced.",
            context_label="per-specialist cycle-delta routing matrix",
        )
    if specialist:
        prior = str((state.get("rebuttal_specialist_contexts") or {}).get(specialist) or "").strip()
        ground = str(
            (state.get("rebuttal_specialist_current_grounds") or {}).get(specialist) or ""
        ).strip()
        current = "\n".join(filter(None, (
            f"Current posture: {current_posture}" if current_posture else "",
            f"Current {specialist} questioning ground: {ground}" if ground else (
                f"No new {specialist}-specific questioning ground was identified."
            ),
        )))
        return prompt_rebuttal_context(
            prior or f"No prior {specialist} finding was available.",
            current_context=current,
            context_label=f"cherry-picked history for {specialist}",
        )
    return prompt_rebuttal_context(
        str(state.get("rebuttal_context") or ""),
        current_context=current_posture or "Current submission details appear in the main prompt above.",
        context_label="complete prior rebuttal history",
    )


def parse_amount(value: Any) -> float:
    """Best-effort '$47,365.25' / '47,365.25' / 47365.25 → 47365.25; 0.0 when unparseable."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    m = _AMOUNT_RE.search(str(value))
    if not m:
        return 0.0
    try:
        return float(m.group(0).replace("$", "").replace(",", "").strip())
    except ValueError:
        return 0.0


def seed_dollars(state: dict) -> float:
    """The disputed amount seeded from the resolved classified case amount; 0.0 if absent."""
    return parse_amount((state.get("case") or {}).get("amount"))


def error_type_of(state: dict) -> str:
    """The auditor's COPAS error_type string for the deterministic classifier.

    Sourced best-effort: an explicit `case['error_type']`, else the classified exception JSON `description` (e.g.
    'CBOH - Offsite Tech - SIEP Affiliate Labor'). Labor Issues rows usually carry no numeric COPAS prefix, 
    so this commonly yields '' → family 'unknown' (design doc line 11); routing never depends on it.
    """
    case = state.get("case") or {}
    return (case.get("error_type") or case.get("description") or "")


def _discovery_workbook_names(synth: dict, key: str) -> list[str]:
    """Stable, lock-file-free workbook basenames from the extractor evidence manifest."""
    discovery = synth.get("evidence_discovery") or {}
    names: list[str] = []
    seen: set[str] = set()
    for path in discovery.get(key, []) or []:
        name = Path(str(path or "")).name
        folded = name.casefold()
        if not name or name.startswith("~$") or folded in seen:
            continue
        seen.add(folded)
        names.append(name)
    return names


def filtered_attachments(synth: dict, warn=None) -> list[dict]:
    """The synthesis attachments after manifest-aware ownership filtering.

    Every spine consumer goes through here, so the compacted prompt view and the cited-evidence view
    cannot disagree about what exists. Uncited operator-response files/tabs are removed. A workbook in
    ``evidence_discovery.cited_attachment_workbooks`` is retained regardless of response-like naming,
    because the extractor manifest records that the auditor explicitly made it evidence. Office lock
    files are always discarded.

    Ordinary workbooks still use two filter levels: whole response workbooks, then response tabs inside
    surviving workbooks. Explicitly cited workbooks bypass both levels as a unit.

    Returns the attachment dicts as-is apart from a rewritten `tabs` list, so callers can read any field.
    """
    atts = list(synth.get("attachments") or [])
    case = synth.get("title") or synth.get("exception_code") or "this exception"
    cited_files = set(_discovery_workbook_names(synth, "cited_attachment_workbooks"))
    candidate_files = [
        str(a.get("file_name", "") or "") for a in atts
        if not str(a.get("file_name", "") or "").startswith("~$")
        and str(a.get("file_name", "") or "") not in cited_files
    ]
    keep_files, _ = drop_responses(candidate_files,
                                   kind="workbook", source=case, warn=warn)
    keep_files = set(keep_files) | cited_files

    out: list[dict] = []
    for att in atts:
        fname = str(att.get("file_name", "") or "")
        if not fname or fname.startswith("~$") or fname not in keep_files:
            continue
        tabs = list(att.get("tabs") or [])
        if fname in cited_files:
            out.append({**att, "tabs": tabs})
            continue
        keep_tabs, _ = drop_responses([t.get("tab_name", "") for t in tabs],
                                      kind="tab", source=fname or "this workbook", warn=warn)
        keep_tabs = set(keep_tabs)
        out.append({**att, "tabs": [t for t in tabs if t.get("tab_name", "") in keep_tabs]})
    return out


def cited_attachments(synth: dict, warn=None) -> tuple[list[str], dict[str, list[str]], list[dict]]:
    """Reshape `filtered_attachments` into the three views verify_evidence cites from:
      * cited_workbooks  — kept file names,
      * cited_sheets     — {file name: [kept tab names]},
      * kept_tabs        — [{file, tab_name, purpose, demonstrates, key_columns}] for the LLM to read.
    """
    workbooks: list[str] = []
    sheets: dict[str, list[str]] = {}
    kept_tabs: list[dict] = []
    for att in filtered_attachments(synth, warn=warn):
        fname = att.get("file_name", "")
        tab_names: list[str] = []
        for tab in att.get("tabs") or []:
            tname = tab.get("tab_name", "")
            tab_names.append(tname)
            kept_tabs.append({
                "file": fname,
                "tab_name": tname,
                "purpose": tab.get("purpose", ""),
                "demonstrates": tab.get("demonstrates", ""),
                "key_columns": tab.get("key_columns", []),
            })
        workbooks.append(fname)
        sheets[fname] = tab_names
    # The discovery manifest is authoritative even when extractor interpretation of a workbook failed
    # and therefore produced no attachment summary. Keep that workbook on the labor pipeline's source
    # list so deterministic profiling/code execution can still inspect it directly.
    for fname in _discovery_workbook_names(synth, "workbooks"):
        if fname not in sheets:
            workbooks.append(fname)
            sheets[fname] = []
    return workbooks, sheets, kept_tabs


def person_tab_hints(state: dict) -> dict:
    """`{file name: {tab name: notes}}` — what the extractor already worked out about each tab.

    The extractor read every tab once and wrote down what it holds, and for a per-person tab it usually
    names the person outright: the notes for `Adams N` say "a single affiliate employee, Nancy Adams
    (Employee ID 10203215)". That is the fact that tells `Adams N` from `Adams S`, which a tab name
    alone cannot. So the manifest matches people against these notes instead of re-deriving the mapping.

    The text is used only inside Python, for matching. It never goes into a prompt — one tab's notes run
    to a thousand characters, and there can be 63 tabs.

    Built on `filtered_attachments`, so uncited operator-response material is gone while workbooks the
    auditor explicitly cited remain available for person-to-tab matching.
    """
    hints: dict[str, dict[str, str]] = {}
    for att in filtered_attachments(state.get("synthesis") or {}):
        fname = att.get("file_name", "")
        if not fname:
            continue
        per_tab: dict[str, str] = {}
        for tab in att.get("tabs") or []:
            tname = str(tab.get("tab_name", "") or "")
            if not tname:
                continue
            parts = [tname, str(tab.get("purpose", "") or ""), str(tab.get("demonstrates", "") or ""),
                     str(tab.get("notes", "") or "")]
            per_tab[tname] = " ".join(p for p in parts if p)
        if per_tab:
            hints[fname] = per_tab
    return hints


def _figure_values(inv: dict) -> list[str]:
    """The specific dollar/label figures the auditor keyed on — VALUES only.

    `figures_used` entries carry a long `why_relevant` rationale; the spine only needs the figures
    themselves (e.g. "212,850.00"), so drop the prose and keep the values. Tolerates plain-string
    entries too.
    """
    out: list[str] = []
    for f in inv.get("figures_used") or []:
        v = f.get("value") if isinstance(f, dict) else f
        if v not in (None, ""):
            out.append(str(v))
    return out


def _invoice_view(inv: dict, demonstrates_limit: int = 240) -> dict:
    """A compact, row-free view of ONE parsed invoice for the spine prompts.

    Carries the identifying fields, the billed TOTAL, the specific figures the auditor keyed on, and a
    BOUNDED `demonstrates`/`ties_to_amount` — enough for a reasoning node to weigh the invoice as
    evidence for the questioned amount. Kept lean *by construction* (figure values only, no
    `why_relevant`/`ties_to_exhibit` prose, `demonstrates` capped) so ALL invoices survive the
    `compact_synthesis` char budget rather than the first one consuming it. Omits raw line-item rows
    (jade reconstructs figures from the workbooks); amount/vendor fall back from `parsed` to
    `extraction`.
    """
    parsed = inv.get("parsed") or {}
    extraction = inv.get("extraction") or {}
    total = parsed.get("total_amount")
    if total in (None, ""):
        total = extraction.get("total_amount")
    dem = str(inv.get("demonstrates", "") or "")
    if len(dem) > demonstrates_limit:
        dem = dem[: demonstrates_limit - 1] + "…"
    return {
        "file_name": inv.get("file_name", ""),
        "vendor": parsed.get("vendor") or extraction.get("vendor") or inv.get("file_name", ""),
        "invoice_number": parsed.get("invoice_number") or extraction.get("invoice_number") or "",
        "invoice_date": parsed.get("invoice_date") or extraction.get("invoice_date") or "",
        "po_number": parsed.get("po_number") or extraction.get("po_number") or "",
        "currency": parsed.get("currency") or extraction.get("currency") or "",
        "total_amount": total,
        "figures_used": _figure_values(inv),
        "ties_to_amount": inv.get("ties_to_amount", ""),
        "demonstrates": dem,
    }


def _cap_items(items: list[dict], budget: int) -> tuple[list[dict], int]:
    """Keep leading `items` whose serialized size fits `budget`; return (kept, dropped_count).

    Caps a list by dropping WHOLE trailing items rather than truncating mid-item, so the view stays
    coherent. Always keeps at least the first item (so a lone fat item still shows).
    """
    kept: list[dict] = []
    used = 0
    for it in items:
        size = len(json.dumps(it, default=str, ensure_ascii=False))
        if kept and used + size > budget:
            break
        kept.append(it)
        used += size
    return kept, len(items) - len(kept)


def compact_synthesis(synth: dict, attachments_limit: int = 12000,
                      invoices_limit: int = 12000) -> str:
    """A compact, row-free JSON view of the synthesis for the light-tier spine prompts.

    Includes the header, each attachment's file/role/summary + its tabs' purpose/demonstrates, and the
    parsed invoices — their identifying fields, billed totals, and the figures the auditor keyed on
    (see `_invoice_view`) — never raw cells. Attachments and invoices get SEPARATE char budgets so a
    fat attachment section can't starve the invoices (which serialize after it); each is capped by
    dropping whole trailing items, with an `_omitted` count when that happens.

    Attachments come from `filtered_attachments`, so uncited operator-response material is gone before
    the text is built. Workbooks explicitly cited by the auditor remain visible by design.
    """
    attachments = [
        {
            "file_name": a.get("file_name", ""),
            "role": a.get("role", ""),
            "overall_summary": a.get("overall_summary", ""),
            "tabs": [
                {"tab_name": t.get("tab_name", ""), "purpose": t.get("purpose", ""),
                 "demonstrates": t.get("demonstrates", "")}
                for t in (a.get("tabs") or [])
            ],
        }
        for a in filtered_attachments(synth)
    ]
    invoices = [_invoice_view(i) for i in (synth.get("invoices") or [])]

    attachments, att_omitted = _cap_items(attachments, attachments_limit)
    invoices, inv_omitted = _cap_items(invoices, invoices_limit)

    view: dict[str, Any] = {
        "title": synth.get("title", ""),
        "exception_code": synth.get("exception_code", ""),
        "exception_category": synth.get("exception_category", ""),
        "attachments": attachments,
        "invoices": invoices,
        "likely_basis_for_amount": synth.get("likely_basis_for_amount", []),
    }
    if att_omitted:
        view["attachments_omitted"] = att_omitted
    if inv_omitted:
        view["invoices_omitted"] = inv_omitted
    return json.dumps(view, default=str, ensure_ascii=False)


def _version_of(name: str) -> int:
    """Version suffix number for sort key: 0 = base file, N = vN or (N). Handles v2, v_1, v.2, (1)…

    Only the version matters — the year token in corpus filenames is one year behind the audit year
    (risk #5), so it is ignored. Caps (N) at 3 digits so large SharePoint IDs are not treated as
    version numbers.
    """
    m = _VERSION_RE.search(Path(name).stem)
    return int(m.group(1) or m.group(2)) if m else 0


def resolve_staff_listing(common_reports_dir: Any) -> str | None:
    """The consolidated staff listing for THIS case's year — env override `OTL_STAFF_LISTING`, else the
    latest-VERSION 'Consolidated Staff Listing' workbook in the case's per-year Common-Reports folder.

    Each audit year's `Common Reports for all assets` folder carries its OWN staff listing alongside the
    POB + heli logs (design doc line 34 / risk #6: the listing must track the case's year, not a global
    latest). The filename year token is one year behind the folder year (risk #5) so it is ignored — only
    the `_vN` version matters, latest wins. The `staff_listing_with_CFR_mapping` reference (which lacks
    'consolidated') is naturally excluded by the keyword filter, and .xlsb listings (e.g. 2021) are
    picked up too.

    Shared by the `job_title` and `supervision` specialists (both walk the same person listing), so it
    lives here rather than as a private helper on either agent.
    """
    override = os.getenv("OTL_STAFF_LISTING")
    if override and Path(override).exists():
        return override
    return pick_latest_by_keyword(common_reports_dir, ["consolidated", "staff", "listing"])


def pick_latest_by_keyword(directory: Any, keywords: list[str],
                           exts: tuple[str, ...] = (".xlsx", ".xlsb", ".xls")) -> str | None:
    """Return the latest-version workbook in `directory` whose name contains ALL `keywords`.

    Case-insensitive substring match; Excel lock files (`~$…`) are skipped. Ties on version break by
    name (deterministic). Returns an absolute path str, or None when the directory is missing/empty of
    matches. Used to select the POB / heli / staff-listing workbooks by ROLE + latest version rather
    than by the (off-by-one) filename year token.
    """
    d = Path(directory) if directory else None
    if not d or not d.is_dir():
        return None
    kws = [k.lower() for k in keywords]
    cands = [
        p for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("~$")
        and all(k in p.name.lower() for k in kws)
    ]
    if not cands:
        return None
    return str(max(cands, key=lambda p: (_version_of(p.name), p.name)))


__all__ = [
    "is_operator_response", "is_response_tab", "drop_responses", "sub_get", "agent_directive",
    "directive_block", "active_auditor_text", "active_auditor_context_block",
    "synthesis_narrative", "synthesis_block",
    "compact_classified_exception", "classified_exception_block", "case_context_block",
    "rebuttal_context_block", "parse_amount",
    "seed_dollars", "error_type_of", "filtered_attachments", "cited_attachments",
    "person_tab_hints", "compact_synthesis", "pick_latest_by_keyword", "resolve_staff_listing",
    "MAX_DISPUTED_PEOPLE", "max_disputed_people", "set_max_disputed_people",
    "cap_people", "disputed_people", "people_coverage",
]
