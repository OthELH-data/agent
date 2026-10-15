"""The dedicated skill pass: apply a skill to ALL extracted values via a chunked, LLM-driven scan.

The code does NO column/value logic. It slices the fully-extracted rows into chunks and, for each
chunk, asks the LLM (guided by the skill's markdown instructions) to pull out whatever the skill
needs and to say whether — combined with the findings so far — it now has enough. While the LLM
reports `sufficient=False` and chunks remain, the scan advances to the next chunk; it early-exits the
moment the LLM is satisfied, and records a gap if the chunks are exhausted first.

Every chunk pass goes through the shared, tiktoken-tracked `structured_call`, so all passes count
toward the run's `token_usage` in the final JSON.
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field

from extractor.blocks import primary, render_blocks
from extractor.column_profile import render_profile
from extractor.llm_adapter import structured_call, text_call
from shared.tab_schemas import KNOWN_TYPES

# A tab is "relevant" (and therefore must be fully scanned before the scan may declare sufficiency)
# if it was recognized as a known auditor data type, or its name contains one of these tokens. This
# guarantees coverage of every Recon / BoM Recon / JADE / CJI3 / iWells tab — see the skill's Step 0
# input map. Tab-level routing only; no value/column logic.
_RELEVANT_TAB_KW = (
    "recon", "reconciliation", "bom", "b.o.m", "jade", "cji3", "iwells", "i-wells", "iwell",
    "ivl", "suspense", "tally", "schematic", "inventory", "material", "exception", "tubular",
    "octg", "casing", "tubing", "vendor",
)

# Tunable scan parameters. Bigger chunks -> fewer passes -> less repeated prompt overhead, at the
# cost of a larger single prompt.
_CHUNK_ROWS = 300
_FINDINGS_RENDER_CHARS = 100000  # once the verbatim "findings so far" tail exceeds this, fold it into
                                 # a rolling LLM summary instead of hard-truncating
_FINDINGS_SUMMARY_TIER = "light"  # cheap tier used to compress older findings into the rolling summary
_ANALYSIS_RENDER_CHARS = 100000  # budget for findings rendered into the synthesis prompt. Findings are
                                 # produced in row order, so a prefix cut silently drops the exhibit's
                                 # later rows — and with them its later dates and part of every total.
                                 # Overflow is therefore COMPRESSED oldest-first, never dropped; this
                                 # is sized so a typical exception (tens of findings) never compresses.
_CHUNK_PROFILE_CHARS = 600       # per-tab profile budget in a chunk prompt, where it repeats every
                                 # chunk. Only trims text columns; typed ones are always kept.
# A JADE ledger bigger than this many rows is NOT required for coverage — it can be huge, so the scan
# may finish (sufficient) after the other required tabs and only dip into it if still not satisfied.
_BIG_JADE_ROWS = 1000


class SkillScanResult(BaseModel):
    """LLM output for one chunk of a skill scan."""
    relevant_findings: list[str] = Field(
        default_factory=list,
        description="facts/rows/values in THIS chunk that the skill needs, verbatim, each with a "
                    "tab + row reference (e.g. \"iWells tab row 42: 5.5\\\" 23# tubing, 234 jts / "
                    "10,278 ft\"); empty if this chunk has nothing relevant")
    sufficient: bool = Field(
        description="True ONLY if, combined with the findings gathered so far, you now have "
                    "everything the skill needs to explain how the auditor reached their conclusion; "
                    "False to keep scanning further chunks")
    note: str = Field(default="", description="what is still missing if not sufficient, or a caveat")


SYSTEM_SKILL_SCAN = (
    "You are applying a domain SKILL to an auditor's exhibit, scanning it one chunk of rows at a time. "
    "You are given the skill's instructions, the auditor's exception description, the findings gathered "
    "from earlier chunks, and ONE chunk of rows (positional cells, with the tab's header). Extract into "
    "relevant_findings any rows/values in THIS chunk that the skill calls for — copied verbatim, each "
    "with a tab + row reference so a human can trace it. "
    "COVERAGE REQUIREMENT: the skill's Step 0 input map lists the source tabs it depends on (e.g. the "
    "Recon / BoM Recon tab, the JADE lines, the CJI3 export, the iWells report). You must examine EVERY "
    "such relevant tab before you have enough — do NOT set `sufficient` True while any relevant tab "
    "(the prompt lists which ones remain) is still unscanned, even if the numbers you have already seem "
    "to add up. Set `sufficient` True only once the findings so far PLUS this chunk give you everything "
    "the skill needs AND no relevant tabs remain to cover; otherwise set it False to keep scanning. "
    "Never invent, infer, or compute a value that is not printed in the chunk; if the chunk has nothing "
    "relevant, return empty relevant_findings and sufficient=False. "
    "Do NOT guess the meaning of an abbreviation or code you do not recognize — carry it verbatim."
)


def _tab_id(sh: dict) -> tuple[str, str]:
    return (sh.get("file", ""), sh.get("tab", ""))


def _is_relevant(sh: dict) -> bool:
    """A tab the skill must cover: a recognized data type, or a name matching a relevant token."""
    if sh.get("detected_type") in KNOWN_TYPES:
        return True
    name = (sh.get("tab") or "").lower()
    return any(k in name for k in _RELEVANT_TAB_KW)


def _is_big_jade(sh: dict) -> bool:
    """A large JADE ledger — relevant, but NOT mandatory to fully traverse (it can be huge).

    A WBS-filtered JADE (`wbs_filtered`) has already been narrowed to the codes of interest, so it is
    never treated as the skippable big ledger — it must be fully covered.
    """
    if sh.get("wbs_filtered"):
        return False
    return sh.get("detected_type") == "JADE" and len(sh.get("rows") or []) > _BIG_JADE_ROWS


def _chunk_plan(sheets_json: list[dict], chunk_rows: int):
    """Build the ordered chunk list + the set of required (mandatory-coverage) tab ids.

    Mandatory relevant tabs are scanned first (smallest-first — a compact Recon/Exception tab before a
    ledger); then the big JADE ledger(s), which are relevant but exempt from mandatory coverage; then
    any other tabs. Each chunk carries its tab id, whether it's relevant, and whether it's that tab's
    final chunk (so the caller can track coverage). `required_ids` excludes big JADE ledgers, so the
    scan may reach sufficiency after the other required tabs without grinding through the big ledger.
    """
    usable = [s for s in sheets_json if not s.get("error") and (s.get("rows") or [])]
    relevant = [s for s in usable if _is_relevant(s)]
    mandatory = sorted((s for s in relevant if not _is_big_jade(s)), key=lambda s: len(s["rows"]))
    big_jade = sorted((s for s in relevant if _is_big_jade(s)), key=lambda s: len(s["rows"]))
    other = sorted((s for s in usable if not _is_relevant(s)), key=lambda s: len(s["rows"]))
    required_ids = {_tab_id(s) for s in mandatory}
    chunks: list[dict] = []
    for s in mandatory + big_jade + other:
        rows = s["rows"]
        n = len(rows)
        # Rendered once per tab and shared by all its chunks. A chunk sees a few hundred rows out of
        # possibly thousands, so the tab's real range and totals are invisible to it — this supplies
        # them. Kept deliberately tight, because it repeats in every chunk prompt of the tab: period,
        # date and number columns are never withheld by the budget, so a small cap trims only the text
        # columns, whose values the chunk's own rows already show.
        prof = render_profile(s.get("column_profile") or [], s.get("total_rows") or [],
                              max_chars=_CHUNK_PROFILE_CHARS)
        # Which table this chunk's rows belong to. A chunk sees a window of rows with no way to tell
        # that they sit under a second table's header, so without this it reads them under the first
        # table's column names. Empty for the single-table tabs, which is nearly all of them.
        tab_blocks = s.get("blocks") or []
        lay = render_blocks(tab_blocks, primary(tab_blocks))
        for start in range(0, n, chunk_rows):
            chunks.append({
                "file": s.get("file", ""), "tab": s.get("tab", ""),
                "dtype": s.get("detected_type"), "header": s.get("header") or [],
                "profile": prof, "layout": lay,
                "start": start, "rows": rows[start:start + chunk_rows],
                "tab_id": _tab_id(s),
                "relevant": _is_relevant(s),          # true relevance (incl. big JADE)
                "mandatory": _tab_id(s) in required_ids,  # must be covered before sufficiency
                "is_tab_last": start + chunk_rows >= n,
            })
    return chunks, required_ids


SYSTEM_FINDINGS_SUMMARY = (
    "You are compressing the running findings of a skill scan over an auditor's exhibit. You are given "
    "the prior running summary (may be empty) and a batch of newer verbatim findings, each with a "
    "tab + row reference. Fold them into ONE compact running summary that PRESERVES every distinct "
    "quantity, amount, material/MMR code, WBS/cost-object code, document number, and tab/row reference "
    "needed to reconstruct how the auditor reached their conclusion. Keep numbers and references "
    "verbatim; drop only redundant prose and exact duplicates. Never invent, infer, or adjudicate — "
    "only condense what is present. Return the summary text only, no preamble."
)


def _summarize_findings(prior_summary: str, findings: list[str]) -> str:
    """Fold older verbatim findings into a compact running summary.

    Routed through the tracked `text_call`, so the tokens it consumes are recorded in the run's
    token_usage automatically.
    """
    body = "\n".join(f"- {f}" for f in findings)
    user = (
        f"PRIOR RUNNING SUMMARY:\n{prior_summary or '(none)'}\n\n"
        f"NEWER FINDINGS TO FOLD IN:\n{body}\n\n"
        "Return the updated compact running summary."
    )
    return text_call(_FINDINGS_SUMMARY_TIER, SYSTEM_FINDINGS_SUMMARY, user).strip()


def _render_findings(summary: str, recent: list[str]) -> str:
    """Render the "findings so far" echo: the rolling summary (if any) plus the verbatim recent tail."""
    parts: list[str] = []
    if summary:
        parts.append(f"SUMMARY OF EARLIER FINDINGS:\n{summary}")
    if recent:
        parts.append("\n".join(f"- {f}" for f in recent))
    return "\n\n".join(parts) if parts else "(none yet)"


def apply_skill(skill, sheets_json: list[dict], description: str, meta: dict | None = None,
                tier: str = "med", chunk_rows: int = _CHUNK_ROWS) -> dict:
    """Scan all extracted values in chunks, applying `skill`, until sufficient or chunks exhausted.

    Returns {skill_name, findings, sufficient, chunks_scanned, total_chunks, gaps}. All LLM passes go
    through the tracked structured_call, so they count toward token_usage.
    """
    meta = meta or {}
    system = SYSTEM_SKILL_SCAN + "\n\nSKILL INSTRUCTIONS:\n" + skill.instructions

    all_chunks, required_ids = _chunk_plan(sheets_json, chunk_rows)
    findings: list[str] = []
    findings_summary = ""   # rolling summary of findings already folded out of the verbatim tail
    summarized_upto = 0     # findings[:summarized_upto] have been folded into findings_summary
    sufficient = False
    scanned = 0
    covered: set[tuple[str, str]] = set()

    def _remaining_required() -> list[str]:
        return sorted(tab for (_f, tab) in (required_ids - covered))

    for ch in all_chunks:
        remaining = _remaining_required()
        cover_note = (
            f"REQUIRED TABS STILL TO COVER (do NOT set sufficient=True yet): {remaining}"
            if remaining else
            "All required tabs have now been scanned — you MAY set sufficient=True if you have enough information"
            "(a large JADE ledger, shown last, is optional supporting detail, not required). Otherwise, set sufficient=False."
        )
        opt = "" if ch["mandatory"] else " (OPTIONAL — large JADE ledger, not required for coverage)"
        focus = meta.get("focus") or {}
        focus_hint = (" | context of interest: " + "; ".join(f"{k}={v}" for k, v in focus.items())
                      if focus else "")
        # What the profile actually covers. On a tab holding several stacked tables it covers the one
        # the layout marks, not the tab — saying "the whole tab" there would be the same overstatement
        # the profile exists to prevent.
        scope = "the marked table" if ch.get("layout") else "the WHOLE tab"
        user = (
            f"AUDITOR EXCEPTION DESCRIPTION:\n{description}\n\n"
            f"QUESTIONED/CONTEXT: {meta.get('title', '')} | amount hint: {meta.get('disputed_amount')}"
            f"{focus_hint}\n\n"
            f"{cover_note}\n\n"
            f"FINDINGS SO FAR:\n{_render_findings(findings_summary, findings[summarized_upto:])}\n\n"
            f"CHUNK — file={ch['file']!r} tab={ch['tab']!r} type={ch['dtype']}{opt} "
            f"rows {ch['start']}-{ch['start'] + len(ch['rows']) - 1}\n"
            f"HEADER (positional): {json.dumps(ch['header'])}\n"
            + (f"{ch['layout']}\n" if ch.get("layout") else "")
            + (f"{ch['profile']}\n"
               f"(The profile above covers {scope} in full, not just this chunk. Where a finding needs "
               f"the full range, count or total, take it from there and say so — do not compute it from "
               f"the rows below, which are only part of it.)\n"
               if ch.get("profile") else "")
            + f"ROWS (positional cells):\n{json.dumps(ch['rows'])}\n\n"
            "Extract the skill-relevant findings from THIS chunk and set `sufficient`."
        )
        res = structured_call(tier, system, user, SkillScanResult)
        scanned += 1
        if res.relevant_findings:
            findings.extend(res.relevant_findings)
        # Once the verbatim tail outgrows the render cap, summarize it (rather than truncate) so the
        # next pass still sees earlier findings compactly. The fold is a tracked call → tokens counted.
        recent = findings[summarized_upto:]
        if len("\n".join(f"- {f}" for f in recent)) > _FINDINGS_RENDER_CHARS:
            findings_summary = _summarize_findings(findings_summary, recent)
            summarized_upto = len(findings)
        if ch["is_tab_last"]:
            covered.add(ch["tab_id"])
        # Coverage gate: sufficiency is only honored once every relevant tab is fully scanned.
        if res.sufficient and required_ids.issubset(covered):
            sufficient = True
            break

    gaps: list[str] = []
    if not sufficient:
        gaps.append("scanned all relevant tabs but the skill's needs were not fully met")

    return {
        "skill_name": skill.name,
        "findings": findings,
        # The rolling fold built during the scan, carried out so `render_analysis` can reuse it
        # instead of paying for the same compression a second time.
        "findings_summary": findings_summary,
        "summarized_upto": summarized_upto,
        "sufficient": sufficient,
        "chunks_scanned": scanned,
        "total_chunks": len(all_chunks),
        "required_tabs": sorted(tab for (_f, tab) in required_ids),
        "covered_tabs": sorted(tab for (_f, tab) in covered),
        "gaps": gaps,
    }


def _verbatim_tail_start(findings: list[str], max_chars: int) -> int:
    """Index where the verbatim tail must begin for it to fit `max_chars`; 0 if everything fits.

    Walks backwards from the NEWEST finding, because the newest are the ones the old prefix-truncating
    renderer always threw away. The last finding is always kept, even if it alone busts the budget —
    a single oversized verbatim quote is better than a block with no verbatim evidence at all.
    """
    total = 0
    last = len(findings) - 1
    for i in range(last, -1, -1):
        total += len(findings[i]) + 3          # "- " + "\n"
        if total > max_chars and i < last:
            return i + 1
    return 0


def _render_analysis_findings(analysis: dict, findings: list[str], max_chars: int) -> str:
    """Render EVERY finding into the synthesis block — verbatim, or compressed if over budget.

    Findings arrive in exhibit row order, so truncating to a prefix always discards the exhibit's
    later rows. That does not read as missing data downstream; it reads as a complete answer over an
    incomplete population — a date range that stops early, a total that is short by the dropped rows.
    So overflow is folded oldest-first through the same `_summarize_findings` pass the scan already
    uses, which is instructed to keep every amount, code, document number and row reference verbatim.
    """
    n = len(findings)
    split = _verbatim_tail_start(findings, max_chars)
    if split == 0:
        return f"(all {n} findings, verbatim):\n" + "\n".join(f"- {f}" for f in findings)

    # The scan may already have folded a head of this list. Reuse that fold when it covers at least
    # what the budget requires; otherwise extend it over the findings it has not yet absorbed.
    prior = analysis.get("findings_summary") or ""
    upto = int(analysis.get("summarized_upto") or 0)
    if prior and upto >= split:
        summary, split = prior, upto
    else:
        summary = _summarize_findings(prior, findings[upto:split])

    tail = findings[split:]
    body = "\n".join(f"- {f}" for f in tail)
    if not summary:
        # Compression produced nothing. Do not imply the remainder is the whole population.
        return (
            f"(INCOMPLETE — {n} findings were gathered but findings 1-{split} could not be rendered. "
            f"Every range, date and total you state from this block must be qualified as covering "
            f"ONLY findings {split + 1}-{n}, not the full exhibit.):\n{body}"
        )
    return (
        f"({n} findings total. Findings 1-{split} are compressed into the summary below, with their "
        f"amounts, codes and row references preserved verbatim; findings {split + 1}-{n} follow in "
        f"full. Nothing has been dropped — read ranges and totals across BOTH parts.)\n"
        f"SUMMARY OF EARLIER FINDINGS:\n{summary}\n\n{body}"
    )


def render_analysis(analysis: dict | None, max_chars: int = _ANALYSIS_RENDER_CHARS) -> str:
    """Render a skill analysis into a synthesis-prompt block (empty string if none)."""
    if not analysis:
        return ""
    findings = analysis.get("findings") or []
    if findings:
        lines = _render_analysis_findings(analysis, findings, max_chars)
    else:
        lines = "(no relevant findings surfaced)"
    gaps = analysis.get("gaps") or []
    gap_block = ("\nCOVERAGE GAPS:\n" + "\n".join(f"- {g}" for g in gaps)) if gaps else ""
    status = ("all needed evidence located" if analysis.get("sufficient")
              else "scan did NOT confirm it found everything needed")
    req = analysis.get("required_tabs") or []
    cov = analysis.get("covered_tabs") or []
    cover_block = f"\nRELEVANT TABS: required={req}; fully scanned={cov}" if req else ""
    return (
        f"\n\nSKILL ANALYSIS — {analysis.get('skill_name')} "
        f"(scanned {analysis.get('chunks_scanned')}/{analysis.get('total_chunks')} chunks; {status}):\n"
        f"{lines}{cover_block}{gap_block}"
    )
