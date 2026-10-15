"""Runtime interception hooks for shared LLM calls."""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any

from .hooks_config import HookDefinition, hook_definition


class HookBlocked(RuntimeError):
    """A configured hook intentionally prevented an operation."""


class BudgetHookBlocked(HookBlocked):
    """The cycle budget hook prevented an LLM request before network I/O."""

    def __init__(self, *, module: str, stage: str, reason: str,
                 limit_usd: float, spent_usd: float | None):
        self.module = module
        self.stage = stage
        self.reason = reason
        self.limit_usd = limit_usd
        self.spent_usd = spent_usd
        amount = "unavailable" if spent_usd is None else f"${spent_usd:.6f}"
        super().__init__(
            f"{module}.hooks.budget blocked {stage}: {reason}; "
            f"spent={amount}, limit=${limit_usd:.2f}"
        )


@dataclass(frozen=True)
class BudgetParameters:
    scope: str
    limit_usd: float
    block_when_pricing_unavailable: bool


@dataclass
class _BudgetState:
    stopped_stage: str = ""
    reason: str = ""
    skipped_calls: int = 0
    intercepted_calls: int = 0
    last_spent_usd: float | None = 0.0
    pricing_available: bool = True
    triggered: bool = False


def _budget_parameters(definition: HookDefinition) -> BudgetParameters:
    params = definition.parameters
    allowed_intercepts = {"structured_call", "text_call"}
    unknown_intercepts = set(definition.intercepts) - allowed_intercepts
    if unknown_intercepts:
        names = ", ".join(sorted(unknown_intercepts))
        raise ValueError(f"budget hook has unknown intercept(s): {names}")
    if not definition.intercepts:
        raise ValueError("budget hook must configure at least one intercept")
    allowed_parameters = {
        "scope", "limit_usd", "block_when_pricing_unavailable",
    }
    unknown_parameters = set(params) - allowed_parameters
    if unknown_parameters:
        names = ", ".join(sorted(unknown_parameters))
        raise ValueError(f"budget hook has unknown parameter(s): {names}")
    scope = params.get("scope", "rebuttal_cycle")
    if scope not in {"rebuttal_cycle", "case"}:
        raise ValueError("budget hook parameter `scope` must be 'rebuttal_cycle' or 'case'")
    limit = params.get("limit_usd")
    if isinstance(limit, bool) or not isinstance(limit, (int, float)):
        raise ValueError("budget hook parameter `limit_usd` must be a number")
    limit = float(limit)
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("budget hook parameter `limit_usd` must be finite and > 0")
    block_unpriced = params.get("block_when_pricing_unavailable", True)
    if not isinstance(block_unpriced, bool):
        raise ValueError(
            "budget hook parameter `block_when_pricing_unavailable` must be boolean"
        )
    return BudgetParameters(scope, limit, block_unpriced)


def _call_count(usage_report: dict) -> int:
    return sum(int(tier.get("calls", 0) or 0) for tier in usage_report.get("by_tier", {}).values())


def _cost_state(usage_report: dict) -> tuple[float | None, bool]:
    calls = _call_count(usage_report)
    if calls == 0:
        return 0.0, True
    cost = usage_report.get("cost") or {}
    available = bool(cost.get("priced")) and not bool(cost.get("unpriced_tiers"))
    amount = cost.get("total_usd") if available else None
    return (float(amount) if amount is not None else None), available


class HookRuntime:
    """Execute configured hooks around intercepted call surfaces.

    State is keyed by ``(module, usage_generation)``. ``USAGE.reset()`` increments the generation,
    which naturally gives every rebuttal cycle a fresh hook state and budget.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._budget_states: dict[tuple[str, int], _BudgetState] = {}

    def _definition(self, module: str, call_type: str) -> HookDefinition | None:
        definition = hook_definition(module, "budget")
        if not definition or not definition.enabled or call_type not in definition.intercepts:
            return None
        return definition

    def before_call(self, *, module: str, call_type: str, stage: str,
                    usage_report: dict, usage_generation: int) -> None:
        """Block a configured intercepted call when the current cycle can spend no more."""
        definition = self._definition(module, call_type)
        if definition is None:
            return
        params = _budget_parameters(definition)
        spent, pricing_available = _cost_state(usage_report)
        key = (module, usage_generation)
        with self._lock:
            state = self._budget_states.setdefault(key, _BudgetState())
            state.intercepted_calls += 1
            state.last_spent_usd = spent
            state.pricing_available = pricing_available
            reason = ""
            if not pricing_available and params.block_when_pricing_unavailable:
                reason = "pricing_unavailable"
            elif spent is not None and spent >= params.limit_usd:
                reason = "limit_reached"
            if not reason:
                return
            state.triggered = True
            state.stopped_stage = stage
            state.reason = reason
            state.skipped_calls += 1
        raise BudgetHookBlocked(
            module=module, stage=stage, reason=reason,
            limit_usd=params.limit_usd, spent_usd=spent,
        )

    def after_call(self, *, module: str, call_type: str, stage: str,
                   usage_report: dict, usage_generation: int) -> None:
        """Refresh hook state from the provider usage recorded for a completed call.

        This never raises: the completed response remains usable. The next intercepted call is the
        one that is prevented when the refreshed total has reached the limit.
        """
        definition = self._definition(module, call_type)
        if definition is None:
            return
        params = _budget_parameters(definition)
        spent, pricing_available = _cost_state(usage_report)
        key = (module, usage_generation)
        with self._lock:
            state = self._budget_states.setdefault(key, _BudgetState())
            state.last_spent_usd = spent
            state.pricing_available = pricing_available
            if not pricing_available and params.block_when_pricing_unavailable:
                state.triggered = True
                state.reason = "pricing_unavailable"
            elif spent is not None and spent >= params.limit_usd:
                state.triggered = True
                state.reason = "limit_reached"

    def budget_snapshot(self, *, module: str, usage_report: dict,
                        usage_generation: int) -> dict[str, Any]:
        """Return JSON-ready status for the current module/cycle budget hook."""
        definition = hook_definition(module, "budget")
        if definition is None:
            return {"enabled": False}
        if not definition.enabled:
            return {
                "enabled": False,
                "intercepts": list(definition.intercepts),
                "parameters": dict(definition.parameters),
            }
        params = _budget_parameters(definition)
        spent, pricing_available = _cost_state(usage_report)
        with self._lock:
            state = self._budget_states.get((module, usage_generation), _BudgetState())
            return {
                "enabled": definition.enabled,
                "triggered": state.triggered,
                "intercepts": list(definition.intercepts),
                "parameters": dict(definition.parameters),
                "values": {
                    "spent_usd": spent,
                    "pricing_available": pricing_available,
                    "completed_calls": _call_count(usage_report),
                    "intercepted_calls": state.intercepted_calls,
                    "skipped_calls": state.skipped_calls,
                },
                "stopped_stage": state.stopped_stage,
                "reason": state.reason,
            }


HOOKS = HookRuntime()


__all__ = ["BudgetHookBlocked", "HOOKS", "HookBlocked", "HookRuntime"]
