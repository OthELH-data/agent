"""Assemble and persist the OTL exception report (the pipeline's output artifact).

One JSON per exception — `rebuttal_<cycle>_otl.json` — written as a SIBLING to the extractor's
`rebuttal_<cycle>_synthesis.json` (never clobbering it), with the live `rebuttal_<cycle>_otl_trace.jsonl`
alongside. The report embeds the full stage-by-stage findings, the routed trace, the natural-language
summaries, and a token/time footer (the shared `TokenUsageReport`, subclassed here to nest a
per-node breakdown inside each tier — decision: reuse the extractor toolchain).

Backends mirror the reader (INPUT_BASE) with OUTPUT_BASE: `OUTPUT_TARGET=local` (default) writes to
`OUTPUT_BASE/<year>/<asset>/<exc>/…`; `OUTPUT_TARGET=sharepoint` uploads to
`SHAREPOINT_WRITE_FOLDER/<same rel path>` and falls back to OUTPUT_BASE on a 401 (SharePointSink's own
behaviour). The trace JSONL is streamed live to a LOCAL path during the run (a UI can tail it); after
the run it is also pushed verbatim through the sink (`write_trace`) so a SharePoint target keeps a
standalone `rebuttal_<cycle>_otl_trace.jsonl` alongside the report — and it remains embedded in the
report's `trace` array regardless.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from shared.output_paths import LocalSink, SharePointSink, pipeline_output_base
from shared.token_schemas import TierTokenUsage, TokenUsageReport

from .state import SUBSTATE_KEYS


# --------------------------------------------------------------------------------------------------
# token footer — the shared report, plus the per-node breakdown only the graph can produce
#
# Subclassed rather than extended in shared/token_schemas.py so the extractor's own artifacts are
# untouched (its TierTokenUsage ignores the extra `by_node` key `USAGE.report()` now emits). The
# by_tier field has to be re-declared: pydantic validates the nested model by its declared type, so
# `dict[str, TierTokenUsage]` would silently drop `by_node` on the way in.
# --------------------------------------------------------------------------------------------------
class NodeTokenUsage(BaseModel):
    """One graph node's slice of a tier. Every node that ran appears, even with zero LLM calls.

    Same fields and same conventions as the tier: the three input counts are disjoint and add up to
    `prompt_tokens`, and `thinking_tokens` is a subset of `tokens_out`.
    """
    model: str = Field(default="", description="model id that served this node")
    source: str = Field(
        default="", description="'gateway' (provider-reported), 'local-estimate' (tiktoken), or 'mixed'")
    visits: int = Field(default=0, description="times this node executed in the graph")
    calls: int = Field(default=0, description="LLM invocations by this node (retries included)")
    tokens_in: int = Field(default=0, description="new (uncached) input tokens")
    cache_read_tokens: int = 0
    # Claude only — OpenAI does not bill cache writes as their own class, so this is 0 on a GPT tier
    # because the class does not exist there, not because nothing was cached.
    cache_creation_tokens: int = 0
    prompt_tokens: int = Field(default=0, description="the three input counts added up")
    tokens_out: int = 0
    thinking_tokens: int = Field(
        default=0, description="a subset of tokens_out — thinking on Claude, reasoning on GPT")
    tokens_total: int = Field(default=0, description="prompt_tokens + tokens_out")


class OTLTierTokenUsage(TierTokenUsage):
    """A tier's usage plus who spent it. The by_node entries sum back to the tier's own counters."""
    by_node: dict[str, NodeTokenUsage] = Field(
        default_factory=dict,
        description="per-node usage keyed by graph node name; sums to this tier's totals. Each "
                    "node's entry also equals the sum of that node's `tokens` blocks in `trace`.")


class OTLTokenUsageReport(TokenUsageReport):
    """The shared token footer with the per-node breakdown nested inside each tier."""
    by_tier: dict[str, OTLTierTokenUsage] = Field(
        default_factory=dict, description="per-tier usage keyed by tier (high/med/light)")


class OTLProcessingStatus(BaseModel):
    """Whether the labor cycle completed or finalized safely after a runtime-hook stop."""
    status: Literal["complete", "partial"] = "complete"
    stopped_stage: str = ""
    stop_reason: str = ""
    message: str = ""
    hooks: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------------------------------
# report model — case header on top, findings in the middle, token/time footer at the bottom
# --------------------------------------------------------------------------------------------------
class OTLExceptionReport(BaseModel):
    """The serialized outcome of one OTL exception run (field order = serialization order)."""
    # --- case header ---
    run_id: str = ""
    case_id: str = ""
    exc_id: str = ""
    year: str = ""
    asset: str = ""
    title: str = ""
    auditor_narrative: str = ""
    questioned_amount: float = 0.0
    rebuttal_cycle: int = 0
    prior_rebuttal_cycles: list[int] = Field(default_factory=list)
    rebuttal_history_compacted: bool = False
    rebuttal_analysis_path: str = ""
    rebuttal_current_posture: str = ""
    rebuttal_specialist_revisit: dict[str, bool] = Field(default_factory=dict)
    rebuttal_reused_nl_summaries: dict[str, str] = Field(default_factory=dict)
    processing: OTLProcessingStatus = Field(default_factory=OTLProcessingStatus)

    # --- outcome mirrors (quick-scan; full detail lives under `stages`) ---
    disposition: str = ""
    identified_cfr_category: str = ""
    cfr_clause: str = ""
    # `hitl_review` = human judgement, action, follow-up, or intervention remains required. Every
    # `needs_evidence` disposition is a HITL/deferred outcome; unsupported auditor evidence is a `deny`.
    hitl_review: bool = False
    hitl_stage: str = ""
    hitl_reason: str = ""
    evidence_information_message: str = ""

    # --- terminal enrichment + drafter output, surfaced side-by-side for the reviewer ---
    # The wiki sidecar is intentionally a separate object; no field is merged into the response.
    wiki_output: dict[str, Any] = Field(default_factory=dict)
    # Agreement retrieval is a drafter input, but remains independently reviewable with exact chunks,
    # source hashes, pages, tracks, and the two retrieval queries preserved here.
    agreement_output: dict[str, Any] = Field(default_factory=dict)
    draft_response: str = ""
    response_sections: dict[str, str] = Field(default_factory=dict)
    co_owner_response: str = ""
    hitl_note: str = ""

    # --- per-stage findings (ordered by SUBSTATE_KEYS) ---
    stages: dict[str, dict] = Field(default_factory=dict)

    # --- cross-cutting ---
    open_questions: list[str] = Field(default_factory=list)
    hooks: dict[str, Any] = Field(default_factory=dict)
    hitl_evidence_trail: dict = Field(default_factory=dict)
    nl_summaries_by_node: dict[str, str] = Field(default_factory=dict)
    trace: list[dict] = Field(default_factory=list)

    # --- footer ---
    token_usage: OTLTokenUsageReport = Field(default_factory=OTLTokenUsageReport)
    # Mirrors token_usage.cost.total_usd, hoisted here so the run's cost is scannable next to its
    # duration. None when no rates were configured, which is different from a run that cost nothing.
    estimated_cost_usd: float | None = None
    elapsed_minutes: float = 0.0


# --------------------------------------------------------------------------------------------------
# path helpers (shared relative layout with the extractor synthesis)
# --------------------------------------------------------------------------------------------------
def otl_rel_dir(year: str, asset: str, exc_id: str) -> str:
    return f"{year}/{asset}/{exc_id}"


def otl_report_rel_path(year: str, asset: str, exc_id: str, cycle: int = 0) -> str:
    return f"{otl_rel_dir(year, asset, exc_id)}/rebuttal_{cycle}_otl.json"


def otl_trace_rel_path(year: str, asset: str, exc_id: str, cycle: int = 0) -> str:
    return f"{otl_rel_dir(year, asset, exc_id)}/rebuttal_{cycle}_otl_trace.jsonl"


def local_trace_path(year: str, asset: str, exc_id: str, cycle: int = 0,
                     base: Path | None = None) -> Path:
    """Absolute LOCAL path the trace JSONL is streamed to during the run (also the local sibling)."""
    root = base or pipeline_output_base()
    return root / otl_trace_rel_path(year, asset, exc_id, cycle)


# --------------------------------------------------------------------------------------------------
# sink (OTL WRITE folder + OUTPUT_BASE_PIPELINE fallback — distinct from extractor output)
# --------------------------------------------------------------------------------------------------
def make_otl_sink():
    """Output sink selected by OUTPUT_TARGET, using ``OUTPUT_BASE_PIPELINE`` locally."""
    target = (os.getenv("OUTPUT_TARGET") or "local").strip().lower()
    local_base = pipeline_output_base()
    if target == "sharepoint":
        url = os.getenv("SHAREPOINT_URL")
        folder = os.getenv("SHAREPOINT_WRITE_FOLDER")
        if not (url and folder):
            sys.exit("OUTPUT_TARGET=sharepoint requires SHAREPOINT_URL and SHAREPOINT_WRITE_FOLDER in .env")
        return SharePointSink(url, folder, fallback_base=local_base)
    if target != "local":
        sys.exit(f"Unknown OUTPUT_TARGET={target!r} (expected 'local' or 'sharepoint')")
    return LocalSink(local_base)


# --------------------------------------------------------------------------------------------------
# assemble + write
# --------------------------------------------------------------------------------------------------
def _dump(value: Any) -> dict:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return dict(value) if isinstance(value, dict) else {}


def _questioned_amount(state: dict, stages: dict) -> float:
    """The disputed figure: prefer JADE's reconstructed amount, else the deterministic intake seed."""
    for key, field in (("jade_analyzer", "amount_questioned"), ("intake", "dollars")):
        val = (stages.get(key) or {}).get(field) or 0.0
        try:
            if float(val) > 0:
                return float(val)
        except (TypeError, ValueError):
            continue
    return 0.0


def assemble_report(state: dict, *, run_id: str, elapsed_minutes: float,
                    token_usage: dict | None, trace_events: list[dict] | None) -> OTLExceptionReport:
    """Build the OTLExceptionReport from the final graph state + the run's token/trace metadata."""
    case = state.get("case") or {}

    # per-stage sub-states, ordered; drop the bulky raw synthesis payload from verify_evidence
    stages: dict[str, dict] = {}
    for key in SUBSTATE_KEYS:
        d = _dump(state.get(key))
        if key == "verify_evidence":
            d.pop("synthesis_json", None)
        if d:
            stages[key] = d

    matching = stages.get("matching", {})
    wiki = stages.get("wiki", {})
    agreement = stages.get("agreement_retriever", {})
    drafter = stages.get("drafter", {})
    usage = OTLTokenUsageReport(**(token_usage or {}))
    evidence_notice = str(state.get("evidence_exclusion_notice") or "")
    evidence_information_message = (
        evidence_notice
        if (drafter.get("disposition") == "deny"
            and not state.get("exception_exhibits")
            and bool(state.get("all_workbooks_operator_responses"))
            and evidence_notice)
        else ""
    )

    return OTLExceptionReport(
        run_id=run_id,
        case_id=case.get("case_id", ""),
        exc_id=case.get("exc_id", ""),
        year=str(case.get("year", "")),
        asset=str(case.get("asset", "")),
        title=case.get("title", ""),
        auditor_narrative=str(state.get("auditor_narrative") or ""),
        questioned_amount=_questioned_amount(state, stages),
        rebuttal_cycle=int(state.get("rebuttal_cycle") or 0),
        prior_rebuttal_cycles=[
            int(item.get("cycle") or 0) for item in (state.get("rebuttal_history") or [])
        ],
        rebuttal_history_compacted=bool(state.get("rebuttal_history_compacted")),
        rebuttal_analysis_path=str(state.get("rebuttal_analysis_path") or ""),
        rebuttal_current_posture=str(state.get("rebuttal_current_posture") or ""),
        rebuttal_specialist_revisit=dict(state.get("rebuttal_specialist_revisit") or {}),
        rebuttal_reused_nl_summaries=dict(state.get("rebuttal_reused_nl_summaries") or {}),
        processing=OTLProcessingStatus.model_validate(_dump(state.get("processing"))),
        disposition=drafter.get("disposition", ""),
        identified_cfr_category=matching.get("identified_cfr_category", ""),
        cfr_clause=matching.get("cfr_clause", ""),
        hitl_review=bool(state.get("hitl_review")),
        hitl_stage=state.get("hitl_stage", "") or "",
        hitl_reason=state.get("hitl_reason", "") or "",
        evidence_information_message=evidence_information_message,
        wiki_output=wiki,
        agreement_output=agreement,
        draft_response=drafter.get("draft_response", ""),
        response_sections=drafter.get("response_sections", {}) or {},
        co_owner_response=drafter.get("co_owner_response", ""),
        hitl_note=drafter.get("hitl_note", ""),
        stages=stages,
        open_questions=list(state.get("open_questions") or []),
        hooks={
            **dict(state.get("hook_status") or {}),
            "chargeability": {
                "scopes": [
                    item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                    for item in state.get("chargeability_scopes") or []
                ],
                "evaluations": [
                    item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                    for item in state.get("chargeability_hook_evaluations") or []
                ],
                "resolution": _dump(state.get("chargeability_resolution")),
            },
        },
        hitl_evidence_trail=_dump(state.get("hitl_evidence_trail")),
        nl_summaries_by_node=dict(state.get("nl_summaries_by_node") or {}),
        trace=list(trace_events or state.get("trace") or []),
        token_usage=usage,
        estimated_cost_usd=usage.cost.total_usd,
        elapsed_minutes=round(float(elapsed_minutes), 3),
    )


def write_report(report: OTLExceptionReport, *, sink, year: str, asset: str, exc_id: str,
                 cycle: int = 0) -> str:
    """Write the report JSON via the sink; returns the location string (local path or SharePoint URL)."""
    rel = otl_report_rel_path(year, asset, exc_id, cycle)
    # Stage/trace payloads may retain native values from workbook evidence (for example ``date``).
    # JSON mode converts those recursively before either sink calls ``json.dumps``.
    return sink.write(rel, report.model_dump(mode="json"))


def write_trace(sink, *, trace_path: Path | str, year: str, asset: str, exc_id: str,
                cycle: int = 0) -> str | None:
    """Push the live-streamed trace JSONL through the sink (verbatim), a sibling to the report.

    The trace is streamed to `trace_path` locally during the run; this uploads that same file's bytes
    to the sink's `rebuttal_<cycle>_otl_trace.jsonl` so a SharePoint target keeps a standalone copy
    (not just the array embedded in the report). Line-delimited JSONL is preserved as-is via
    `sink.write_text`. Returns the written location, or None if the trace file is missing/empty.
    """
    src = Path(trace_path)
    if not src.is_file():
        return None
    text = src.read_text(encoding="utf-8")
    if not text.strip():
        return None
    rel = otl_trace_rel_path(year, asset, exc_id, cycle)
    return sink.write_text(rel, text)


__all__ = [
    "OTLExceptionReport", "OTLProcessingStatus", "assemble_report", "write_report", "write_trace", "make_otl_sink",
    "otl_rel_dir", "otl_report_rel_path", "otl_trace_rel_path", "local_trace_path",
]
