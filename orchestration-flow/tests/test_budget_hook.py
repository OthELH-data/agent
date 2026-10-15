"""Labor budget stops must still produce a typed HITL report through the real final drafter."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ORCH_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ORCH_DIR.parent
for path in (str(REPO_ROOT), str(ORCH_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from otl import budget  # noqa: E402
from otl.agents import drafter  # noqa: E402
from otl.output_writer import assemble_report  # noqa: E402
from shared.hooks import BudgetHookBlocked  # noqa: E402


class _Graph:
    def __init__(self, blocked: BudgetHookBlocked, partial: dict, final: dict) -> None:
        self.blocked = blocked
        self.partial = partial
        self.final = final
        self.invocations: list[object] = []
        self.updated: tuple[dict, dict, str] | None = None

    def invoke(self, value, config):
        self.invocations.append(value)
        if len(self.invocations) == 1:
            raise self.blocked
        return {**self.final, **(self.updated[1] if self.updated else {})}

    def get_state(self, _config):
        return SimpleNamespace(values=self.partial)

    def update_state(self, config, update, as_node):
        self.updated = (config, update, as_node)


class LaborBudgetHookTests(unittest.TestCase):
    def test_checkpoint_recovery_stops_work_and_schedules_only_the_drafter(self):
        blocked = BudgetHookBlocked(
            module="labor", stage="location", reason="limit_reached",
            limit_usd=7.0, spent_usd=7.2,
        )
        graph = _Graph(
            blocked,
            partial={"case": {"case_id": "case"}, "nl_summaries_by_node": {"intake": "done"}},
            final={"case": {"case_id": "case"}, "drafter": {"disposition": "budget_exceeded"}},
        )
        hook_snapshot = {
            "enabled": True, "triggered": True, "stopped_stage": "location",
            "reason": "limit_reached", "parameters": {"limit_usd": 7.0},
        }

        with (
            patch.object(budget.HOOKS, "budget_snapshot", return_value=hook_snapshot),
            patch.object(budget.USAGE, "report", return_value={}),
            patch.object(budget.USAGE, "generation", return_value=4),
        ):
            final = budget.invoke_with_budget_recovery(graph, {"case": {}}, {"configurable": {}})

        self.assertEqual([{"case": {}}, None], graph.invocations)
        self.assertEqual("wiki", graph.updated[2])
        update = graph.updated[1]
        self.assertTrue(update["budget_exceeded"])
        self.assertTrue(update["hitl_review"])
        self.assertEqual("location", update["hitl_stage"])
        self.assertEqual("budget_exceeded", final["drafter"]["disposition"])
        self.assertEqual("partial", final["processing"]["status"])
        self.assertEqual(hook_snapshot, final["processing"]["hooks"]["budget"])
        self.assertIn("budget exceeded", final["hitl_reason"].lower())

    def test_budget_recovery_drafter_runs_once_and_emits_hitl_budget_disposition(self):
        extraction = drafter._DrafterExtraction(
            hitl_nl_summary="The completed stages established the available labor facts.",
            hitl_blocker="The remaining review was not completed.",
            hitl_impact="A final chargeability conclusion cannot yet be determined.",
            hitl_next_step="A human reviewer should complete the outstanding assessment.",
            nl_summary="Available findings were preserved for review.",
        )
        state = {
            "case": {"case_id": "case", "exc_id": "0001", "title": "Labor"},
            "budget_exceeded": True,
            "hitl_review": True,
            "hitl_stage": "location",
            "hitl_reason": "Labor pipeline budget exceeded before location.",
            "final_mode": "needs_evidence",
            "nl_summaries_by_node": {"intake": "Parsed the auditor request."},
        }

        with patch.object(drafter.llm, "structured", return_value=extraction) as call:
            update = drafter.drafter_node(state)

        self.assertEqual(1, call.call_count)
        self.assertTrue(call.call_args.kwargs["allow_budget_overrun"])
        self.assertEqual("budget_exceeded", update["drafter"].disposition)
        self.assertTrue(update["hitl_review"])
        self.assertTrue(update["drafter"].draft_response)
        self.assertTrue(update["drafter"].hitl_note)
        self.assertEqual(
            list(update["drafter"].response_sections),
            ["Response Withheld - pending information / Under Review"],
        )

    def test_budget_recovery_overrides_evidence_gate_and_request_only_early_returns(self):
        extraction = drafter._DrafterExtraction(
            hitl_nl_summary="The completed stages established the available facts.",
            hitl_blocker="The remaining review was not completed.",
            hitl_impact="A final conclusion cannot yet be determined.",
            hitl_next_step="The outstanding assessment must be completed.",
        )
        base = {
            "case": {"case_id": "case", "exc_id": "0001", "title": "Labor"},
            "budget_exceeded": True,
            "hitl_review": True,
            "hitl_stage": "drafter",
            "hitl_reason": "Labor pipeline budget exceeded before drafter.",
            "final_mode": "needs_evidence",
        }
        cases = (
            {**base, "evidence_gate_decision": {
                "triggered": True, "outcome_key": "deny_missing_evidence",
            }},
            {**base, "attestation_requested": True, "auditor_question": []},
        )

        for state in cases:
            with self.subTest(state=state), patch.object(
                drafter.llm, "structured", return_value=extraction,
            ) as call:
                update = drafter.drafter_node(state)

            self.assertEqual(1, call.call_count)
            self.assertTrue(call.call_args.kwargs["allow_budget_overrun"])
            self.assertEqual("budget_exceeded", update["drafter"].disposition)
            self.assertIn("budget exceeded", update["hitl_reason"].lower())

    def test_partial_budget_report_validates_with_defaults_and_clear_reason(self):
        message = (
            "Labor pipeline budget exceeded: the configured $7.00 limit was reached before location."
        )
        report = assemble_report(
            {
                "case": {"case_id": "case", "exc_id": "0001", "title": "Labor"},
                "processing": {
                    "status": "partial", "stopped_stage": "location",
                    "stop_reason": "limit_reached", "message": message,
                    "hooks": {"budget": {"enabled": True, "triggered": True}},
                },
                "drafter": {
                    "disposition": "budget_exceeded", "draft_response": "Review deferred.",
                    "hitl_note": "Review deferred.",
                },
                "hitl_review": True,
                "hitl_stage": "location",
                "hitl_reason": message,
            },
            run_id="case", elapsed_minutes=0.1, token_usage={}, trace_events=[],
        )
        payload = report.model_dump()

        self.assertEqual("budget_exceeded", payload["disposition"])
        self.assertTrue(payload["hitl_review"])
        self.assertEqual("partial", payload["processing"]["status"])
        self.assertIn("$7.00", payload["processing"]["message"])
        self.assertEqual({}, payload["wiki_output"])
        self.assertEqual({}, payload["agreement_output"])
        self.assertEqual([], payload["open_questions"])
        self.assertIsInstance(payload["token_usage"], dict)


if __name__ == "__main__":
    unittest.main()
