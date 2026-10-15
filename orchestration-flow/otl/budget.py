"""Graceful labor-pipeline finalization after the configured spend hook stops processing."""
from __future__ import annotations

import sys

from shared.hooks import BudgetHookBlocked, HOOKS
from shared.tokens import USAGE


HOOK_MODULE = "labor"


def _stop_message(exc: BudgetHookBlocked) -> str:
    spent = "unavailable" if exc.spent_usd is None else f"${exc.spent_usd:.6f}"
    if exc.reason == "limit_reached":
        lead = (
            f"Labor pipeline budget exceeded: the configured ${exc.limit_usd:.2f} limit was "
            f"reached before the {exc.stage} LLM call (recorded spend {spent})."
        )
    else:
        lead = (
            f"The labor pipeline budget hook stopped processing before the {exc.stage} LLM call "
            f"because {exc.reason.replace('_', ' ')} (recorded spend {spent}, configured limit "
            f"${exc.limit_usd:.2f})."
        )
    return (
        f"{lead} Processing stopped, the final drafter ran once using the findings collected so far, "
        "and the case was routed to HITL."
    )


def _stop_update(exc: BudgetHookBlocked, state: dict) -> dict:
    message = _stop_message(exc)
    questions = list(state.get("open_questions") or [])
    if message not in questions:
        questions.append(message)
    return {
        "budget_exceeded": True,
        "processing": {
            "status": "partial",
            "stopped_stage": exc.stage,
            "stop_reason": exc.reason,
            "message": message,
        },
        "hitl_review": True,
        "hitl_stage": exc.stage,
        "hitl_reason": message,
        "hitl_evidence_trail": {
            "stopped_stage": exc.stage,
            "stop_reason": exc.reason,
            "nl_summaries_by_node": dict(state.get("nl_summaries_by_node") or {}),
        },
        "final_mode": "needs_evidence",
        "open_questions": questions,
    }


def _attach_snapshot(state: dict) -> dict:
    final = dict(state or {})
    usage_report = USAGE.report()
    snapshot = HOOKS.budget_snapshot(
        module=HOOK_MODULE,
        usage_report=usage_report,
        usage_generation=USAGE.generation(),
    )
    processing = dict(final.get("processing") or {})
    processing_hooks = dict(processing.get("hooks") or {})
    processing_hooks["budget"] = snapshot
    processing["hooks"] = processing_hooks
    final["processing"] = processing

    hook_status = dict(final.get("hook_status") or {})
    hook_status["budget"] = snapshot
    final["hook_status"] = hook_status
    return final


def invoke_with_budget_recovery(graph, init_state: dict, config: dict) -> dict:
    """Invoke the graph, or checkpoint-jump to its real drafter after a budget stop.

    Marking the recovery update as the completed ``wiki`` node schedules only ``drafter`` next. This
    preserves the graph's normal instrumentation and output shape while preventing any remaining
    analysis, retrieval, or specialist node from spending after the hook fires.
    """
    try:
        final = graph.invoke(init_state, config=config)
    except BudgetHookBlocked as exc:
        snapshot = graph.get_state(config)
        partial = dict(getattr(snapshot, "values", None) or init_state)
        update = _stop_update(exc, partial)
        print(f"  !! {update['processing']['message']}", file=sys.stderr)
        graph.update_state(config, update, as_node="wiki")
        final = graph.invoke(None, config=config)
    return _attach_snapshot(final)


__all__ = ["HOOK_MODULE", "invoke_with_budget_recovery"]
