"""First OTL node: load prior analysis files and compact them only when context pressure requires it."""
from __future__ import annotations

import json
import os

from pydantic import BaseModel, Field

from shared.rebuttal_history import (
    DEFAULT_COMPACTION_RATIO,
    compact_rebuttal_history,
    extract_agent_history,
    initialize_cycle_analysis,
    load_rebuttal_history,
    prior_agents,
    write_agent_analysis,
    write_cycle_delta_analysis,
)

from otl import llm_adapter as llm
from otl.trace import trace_event
from otl.agents._common import (active_auditor_context_block, compact_classified_exception,
                                compact_synthesis)


_SPECIALISTS = ("jade_analyzer", "location", "job_title", "matching", "supervision")
_DISPATCH_SPECIALISTS = ("location", "job_title", "matching", "supervision")


_COMPACTION_SYSTEM = """\
You compact prior-cycle analysis for a production joint-venture audit rebuttal. Preserve continuity
without preserving deep implementation detail.

For every cycle and substantive agent represented in the source, retain:
- the agent's main finding;
- material uncertainty, evidence limitation, or unresolved question;
- how the finding supported, contradicted, narrowed, or did not affect the final disposition;
- the released disposition, credit/no-credit result, and any exact material amount;
- what changed between cycles when the source establishes a change.

Remove raw workbook rows, long file inventories, execution traces, token data, repeated prose, and
low-level retrieval mechanics. Never retain classified Operator response prose. Preserve a concise
version of each pipeline-generated rebuttal response together with analytical findings and disposition.
Never invent facts, authority, amounts, or relationships. If the source
does not establish how a finding affected disposition, say that concisely. Return compact markdown with
cycle headings and short agent bullets, suitable for another agent to parse."""

_DELTA_SYSTEM = """\
You compare the CURRENT auditor rebuttal submission with PRIOR rebuttal-cycle agent analysis for the
same exception. Produce both a robust current-posture analysis and a conservative per-specialist revisit
assessment.

CURRENT POSTURE: current_posture_nl_summary must be a concise but substantive plain-English analysis that
another downstream agent can understand without rereading the full history. Cover the original disputed
issue, material prior findings and disposition, the latest auditor reply, what changed or remained
unchanged this cycle, current evidence/document/information requests, and unresolved decision-relevant
points. Explain how the current submission affects the prior posture. Do not reproduce every agent field,
raw record, file list, implementation detail, or classified Operator response prose. Do not draft the
next response.

Specialists: jade_analyzer (amount/ledger/population), location (POB/heli/physical location), job_title
(role and duties), matching (CFR/OTL category), supervision (above-first-level hierarchy).

For each specialist, identify only its relevant prior finding and the current questioning ground. Set
new_information_present=true when the current submission changes or newly challenges that specialist's
facts, people, dates/period, amount, documents, evidence, role, authority theory, or conclusion. Set
should_revisit=false ONLY when the specialist previously ran and the current submission adds no new
information or questioning ground within that specialist's scope. Uncertainty must fail open to
should_revisit=true. Never treat silence about one specialist as new information for another.

Apply dependencies carefully: a new disputed person can affect JADE, location, job title, matching and
supervision; a changed role/profile affects both job_title and matching; changed questioned amounts can
affect JADE and supervision allocation; changed dates/POB/heli evidence can affect location but does not by
itself reopen an unchanged role. Mark every genuinely affected specialist, even when the auditor names
only the upstream ground.

The relevant_prior_context field must be a concise cherry-picked summary for that specialist alone,
including its prior main finding, material limitation, and relationship to the prior final disposition.
Do not copy the whole history. Do not invent facts.

AUDITOR CHRONOLOGY: the original exception is the active baseline, not historical agent context. For
cycle 0 it is the active request even if no later auditor reply exists. For later cycles, read it with the
latest reply; an empty latest reply does not erase it, and withdrawal, resolution, or replacement requires
explicit support in the supplied context."""


class _SpecialistAssessment(BaseModel):
    current_questioning_ground: str = Field(default="")
    relevant_prior_context: str = Field(default="")
    new_information_present: bool = Field(default=True)
    should_revisit: bool = Field(default=True)
    rationale: str = Field(default="")


class _CycleDeltaAssessment(BaseModel):
    current_posture_nl_summary: str = Field(
        default="", description="Concise but substantive current posture covering the original issue, "
                                "material prior findings/disposition, latest auditor reply, current-cycle "
                                "changes or continuations, active requests, and unresolved relevant points."
    )
    specialists: dict[str, _SpecialistAssessment] = Field(default_factory=dict)


def _ratio() -> float:
    raw = str(os.getenv("REBUTTAL_HISTORY_COMPACT_RATIO") or DEFAULT_COMPACTION_RATIO).strip()
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_COMPACTION_RATIO


def _compact(source: str, target_tokens: int) -> str:
    return llm.text(
        "rebuttal_compaction",
        _COMPACTION_SYSTEM,
        f"Target at most {target_tokens} tokens. Compact this prior rebuttal history:\n\n{source}",
    )


def _current_context(state: dict) -> str:
    intake = state.get("intake")
    narrative = str(state.get("auditor_narrative") or "").strip()
    if hasattr(intake, "model_dump"):
        intake = intake.model_dump(mode="json")
    elif not isinstance(intake, dict):
        intake = {}
    current = {
        "active_auditor_request_context": active_auditor_context_block(state).strip(),
        "case": state.get("case") or {},
        "auditor_narrative": narrative,
        "latest_auditor_reply": state.get("latest_auditor_reply") or {},
        "intake": intake,
        "classified_exception": json.loads(compact_classified_exception(state)),
        "current_evidence_inventory": json.loads(compact_synthesis(state.get("synthesis") or {})),
    }
    return json.dumps(current, indent=2, ensure_ascii=False, default=str)


def _assess_cycle_delta(current: str, prior: str) -> _CycleDeltaAssessment:
    return llm.structured(
        "rebuttal_history",
        _DELTA_SYSTEM,
        "<CURRENT_REBUTTAL_CONTEXT>\n"
        f"{current}\n"
        "</CURRENT_REBUTTAL_CONTEXT>\n\n"
        "<PRIOR_REBUTTAL_CONTEXT>\n"
        f"{prior}\n"
        "</PRIOR_REBUTTAL_CONTEXT>\n\n"
        f"Return assessments for exactly these specialists: {list(_SPECIALISTS)}.",
        _CycleDeltaAssessment,
    )


def rebuttal_history_node(state: dict) -> dict:
    case = state.get("case") or {}
    cycle = case.get("rebuttal_cycle") or 0
    history = load_rebuttal_history(case, current_cycle=cycle)
    compacted = compact_rebuttal_history(
        history,
        case=case,
        context_window=llm.context_window_for_node(node='rebuttal_history'),
        compactor_context_window=llm.context_window_for_node("rebuttal_compaction"),
        token_counter=llm.estimate_text_tokens,
        llm_compactor=_compact,
        ratio=_ratio(),
    )
    analysis_path = initialize_cycle_analysis(
        case,
        cycle=cycle,
        latest_auditor_reply=state.get("latest_auditor_reply") or {},
    )
    # Intake intentionally runs before this node so it can describe only the current submission. Record
    # that completed finding now that the active-cycle file exists.
    write_agent_analysis(
        {**state, "rebuttal_analysis_path": analysis_path},
        "intake",
        {"intake": state.get("intake")},
    )

    current_context = _current_context(state)
    seen_agents = prior_agents(history)
    assessment_error = ""
    assessed = _CycleDeltaAssessment()
    if history:
        try:
            assessed = _assess_cycle_delta(current_context, compacted["context"])
        except Exception as exc:
            # Never suppress a specialist because the delta classifier failed.
            assessment_error = f"{type(exc).__name__}: {str(exc)[:300]}"

    intake = state.get("intake") or {}
    if hasattr(intake, "model_dump"):
        intake = intake.model_dump(mode="json")
    elif not isinstance(intake, dict):
        intake = {}
    current_posture = (
        str(assessed.current_posture_nl_summary or "").strip()
        or str(intake.get("nl_summary") or "").strip()
    )

    specialist_contexts: dict[str, str] = {}
    specialist_grounds: dict[str, str] = {}
    revisit: dict[str, bool] = {}
    rationales: dict[str, str] = {}
    reused: dict[str, str] = {}
    reused_status: dict[str, dict[str, str]] = {}
    for specialist in _SPECIALISTS:
        item = assessed.specialists.get(specialist, _SpecialistAssessment())
        fallback = extract_agent_history(history, specialist)
        specialist_contexts[specialist] = str(item.relevant_prior_context or fallback).strip()
        specialist_grounds[specialist] = str(item.current_questioning_ground or "").strip()
        has_prior = specialist in seen_agents
        should_revisit = (
            True if not has_prior or assessment_error
            else bool(item.new_information_present or item.should_revisit)
        )
        revisit[specialist] = should_revisit
        rationales[specialist] = str(item.rationale or "").strip()
        if has_prior and not should_revisit:
            reused[specialist] = specialist_contexts[specialist]
            reused_status[specialist] = {
                "status": "objective_met",
                "reason": "Prior-cycle finding reused because no new specialist-specific information was submitted.",
            }

    write_cycle_delta_analysis(
        analysis_path,
        posture=current_posture,
        prior_completed_agents=sorted(seen_agents),
        revisit=revisit,
        current_grounds=specialist_grounds,
        rationales=rationales,
        reused_nl_summaries=reused,
    )

    update = {
        "rebuttal_cycle": cycle,
        "rebuttal_history": history,
        "rebuttal_context": compacted["context"],
        "rebuttal_history_compacted": bool(compacted["compacted"]),
        "rebuttal_history_source_tokens": int(compacted["source_tokens"]),
        "rebuttal_history_threshold_tokens": int(compacted["threshold_tokens"]),
        "rebuttal_history_cache_path": str(compacted.get("cache_path") or ""),
        "rebuttal_analysis_path": analysis_path,
        "rebuttal_current_context": current_context,
        "rebuttal_current_posture": current_posture,
        "rebuttal_prior_agents": sorted(seen_agents),
        "rebuttal_specialist_contexts": specialist_contexts,
        "rebuttal_specialist_current_grounds": specialist_grounds,
        "rebuttal_specialist_revisit": revisit,
        "rebuttal_specialist_rationales": rationales,
        "rebuttal_reused_nl_summaries": reused,
        "rebuttal_delta_assessment_error": assessment_error,
    }
    if reused_status:
        update["agent_status"] = reused_status
    update.update(trace_event(
        state,
        "rebuttal_history",
        "loaded and compacted prior rebuttal analysis" if compacted["compacted"]
        else "loaded prior rebuttal analysis",
        inputs_digest=f"cycle={cycle}, prior_cycles={[item['cycle'] for item in history]}",
        key_outputs={
            "prior_cycles": [item["cycle"] for item in history],
            "compacted": bool(compacted["compacted"]),
            "source_tokens": int(compacted["source_tokens"]),
            "threshold_tokens": int(compacted["threshold_tokens"]),
            "cache_hit": bool(compacted.get("cache_hit", False)),
            "prior_agents": sorted(seen_agents),
            "revisit": revisit,
            "assessment_error": assessment_error,
            "analysis_path": analysis_path,
        },
        nl_summary=(
            f"loaded {len(history)} prior rebuttal cycle(s); "
            f"history {'compacted' if compacted['compacted'] else 'kept verbatim'}; "
            f"reusing {sum(not value for key, value in revisit.items() if key in _DISPATCH_SPECIALISTS)} "
            "prior specialist finding(s)"
        ),
    ))
    return update


__all__ = ["rebuttal_history_node"]
