"""Configuration and interception coverage for module-scoped runtime hooks."""
from __future__ import annotations

import json
import inspect
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.hooks import BudgetHookBlocked, HookRuntime
from shared.hooks_config import clear_hooks_config_cache, hook_definition


HOOKS_YAML = """\
schema_version: 1
extractor:
  hooks:
    budget:
      enabled: true
      intercepts: [structured_call, text_call]
      parameters:
        scope: rebuttal_cycle
        limit_usd: 7.0
        block_when_pricing_unavailable: true
labor:
  hooks: {}
"""


def usage_report(spent: float, *, calls: int = 1, priced: bool = True) -> dict:
    return {
        "by_tier": {"high": {"calls": calls}},
        "cost": {
            "priced": priced,
            "unpriced_tiers": [] if priced else ["high"],
            "total_usd": spent if priced else None,
        },
    }


class HookConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "hooks.yaml"
        self.path.write_text(HOOKS_YAML, encoding="utf-8")
        self.env = patch.dict(os.environ, {"HOOKS_CONFIG_FILE": str(self.path)})
        self.env.start()
        self.addCleanup(self.env.stop)
        clear_hooks_config_cache()
        self.addCleanup(clear_hooks_config_cache)

    def test_module_scoped_hook_loads_from_yaml(self):
        budget = hook_definition("extractor", "budget")

        self.assertIsNotNone(budget)
        self.assertEqual(7.0, budget.parameters["limit_usd"])
        self.assertIsNone(hook_definition("labor", "budget"))

    def test_budget_refresh_keeps_completed_call_and_blocks_the_next(self):
        runtime = HookRuntime()
        runtime.after_call(
            module="extractor", call_type="structured_call", stage="TabInterpretation",
            usage_report=usage_report(7.25), usage_generation=4,
        )

        with self.assertRaises(BudgetHookBlocked) as caught:
            runtime.before_call(
                module="extractor", call_type="text_call", stage="rebuttal_summary",
                usage_report=usage_report(7.25), usage_generation=4,
            )

        self.assertEqual("rebuttal_summary", caught.exception.stage)
        snapshot = runtime.budget_snapshot(
            module="extractor", usage_report=usage_report(7.25), usage_generation=4,
        )
        self.assertTrue(snapshot["triggered"])
        self.assertEqual(1, snapshot["values"]["skipped_calls"])

    def test_new_usage_generation_gets_a_fresh_cycle_budget(self):
        runtime = HookRuntime()
        with self.assertRaises(BudgetHookBlocked):
            runtime.before_call(
                module="extractor", call_type="structured_call", stage="old_cycle",
                usage_report=usage_report(7.0), usage_generation=10,
            )

        runtime.before_call(
            module="extractor", call_type="structured_call", stage="new_cycle",
            usage_report=usage_report(0.0, calls=0), usage_generation=11,
        )

    def test_extractor_hook_does_not_intercept_labor_module(self):
        HookRuntime().before_call(
            module="labor", call_type="structured_call", stage="drafter",
            usage_report=usage_report(99.0), usage_generation=1,
        )


class LLMHookIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from shared import llm
            from extractor.schemas import SynthesisResult
            from shared.tokens import USAGE
        except ModuleNotFoundError as exc:
            self.skipTest(f"project LLM dependencies are not installed: {exc}")
        self.llm = llm
        self.SynthesisResult = SynthesisResult
        self.USAGE = USAGE
        USAGE.reset()
        self.addCleanup(USAGE.reset)

    def test_structured_retry_checks_hook_before_each_attempt(self):
        blocked = BudgetHookBlocked(
            module="extractor", stage="SynthesisResult", reason="limit_reached",
            limit_usd=7.0, spent_usd=7.2,
        )
        reply = self.llm.LLMReply(
            text="not json", usage={"input_tokens": 1, "output_tokens": 1}
        )

        with (
            patch.object(self.llm.HOOKS, "before_call", side_effect=[None, blocked]) as before,
            patch.object(self.llm.HOOKS, "after_call"),
            patch.object(self.llm, "_invoke_tier",
                         return_value=(reply, "claude-opus-5")) as invoke,
        ):
            with self.assertRaises(BudgetHookBlocked):
                self.llm.structured_call(
                    "high", "system", "user", self.SynthesisResult, retries=1,
                    hook_module="extractor",
                )

        self.assertEqual(2, before.call_count)
        self.assertEqual(1, invoke.call_count)

    def test_budget_exempt_terminal_call_still_records_usage_and_refreshes_hook_state(self):
        reply = self.llm.LLMReply(
            text='{"synthesis":"Partial evidence summary.","open_questions":["What remains?"],'
                 '"likely_basis_for_amount":[]}',
            usage={"input_tokens": 11, "output_tokens": 7},
        )

        with (
            patch.object(self.llm.HOOKS, "before_call") as before,
            patch.object(self.llm.HOOKS, "after_call") as after,
            patch.object(self.llm, "_invoke_tier",
                         return_value=(reply, "claude-opus-5")),
        ):
            result = self.llm.structured_call(
                "high", "system", "user", self.SynthesisResult,
                hook_module="extractor", hook_stage="final_synthesis",
                allow_budget_overrun=True,
            )

        self.assertEqual("Partial evidence summary.", result.synthesis)
        before.assert_not_called()
        after.assert_called_once()
        usage = self.USAGE.report()
        self.assertEqual(1, usage["by_tier"]["high"]["calls"])
        self.assertEqual(11, usage["total_tokens_in"])
        self.assertEqual(7, usage["total_tokens_out"])

    def test_shared_calls_require_an_explicit_hook_module(self):
        parameter = inspect.signature(self.llm.structured_call).parameters["hook_module"]
        self.assertIs(parameter.default, inspect.Parameter.empty)
        parameter = inspect.signature(self.llm.text_call).parameters["hook_module"]
        self.assertIs(parameter.default, inspect.Parameter.empty)

    def test_extractor_adapter_passes_its_module_variable(self):
        from extractor import llm_adapter

        expected = self.SynthesisResult(synthesis="done")
        with patch.object(llm_adapter._llm, "structured_call", return_value=expected) as call:
            result = llm_adapter.structured_call(
                "high", "system", "user", self.SynthesisResult,
            )

        self.assertIs(result, expected)
        self.assertEqual("extractor", llm_adapter.HOOK_MODULE)
        self.assertEqual(llm_adapter.HOOK_MODULE, call.call_args.kwargs["hook_module"])

    def test_budget_stop_runs_final_synthesis_and_aggregates_its_metrics(self):
        try:
            import run as extractor_run
        except ModuleNotFoundError as exc:
            self.skipTest(f"project extractor dependencies are not installed: {exc}")
        blocked = BudgetHookBlocked(
            module="extractor", stage="SkillSelection", reason="limit_reached",
            limit_usd=7.0, spent_usd=7.1,
        )

        def terminal_synthesis(description, attachments, meta, **kwargs):
            self.assertEqual("Auditor description", description)
            self.assertEqual([], attachments)
            self.assertTrue(kwargs["allow_budget_overrun"])
            self.USAGE.record(
                "high", "claude-opus-5", 11, 7, source="gateway",
                usage={"input_tokens": 11, "output_tokens": 7},
            )
            return extractor_run.ExhibitEvidenceReport(
                case_id=meta["case_id"],
                exception_code=meta["exception_code"],
                title=meta["title"],
                auditor_description=description,
                synthesis="Partial evidence summary.",
                open_questions=["What remains?"],
            )

        with (
            patch.object(extractor_run, "extract_exception_description",
                         return_value="Auditor description"),
            patch.object(extractor_run, "extract_rebuttal_dialogue", return_value=""),
            patch.object(extractor_run, "split_rebuttal_dialogue", return_value=[]),
            patch.object(extractor_run, "resolve_skill", side_effect=blocked),
            patch.object(extractor_run, "synthesize", side_effect=terminal_synthesis) as synthesize,
        ):
            report = extractor_run.build_report(
                Path("report.pdf"), [Path("evidence.xlsx")], "0040", "Labor", "case",
                auto_invoices=False, report_text="report text",
            )

        self.assertEqual("partial", report.processing.status)
        self.assertEqual("SkillSelection", report.processing.stopped_stage)
        self.assertEqual(["evidence.xlsx"], [a.file_name for a in report.attachments])
        synthesize.assert_called_once()
        self.assertEqual("Partial evidence summary.", report.synthesis)
        self.assertIn("What remains?", report.open_questions)
        self.assertEqual(1, report.token_usage.by_tier["high"].calls)
        self.assertEqual(11, report.token_usage.total_tokens_in)
        self.assertEqual(7, report.token_usage.total_tokens_out)
        self.assertTrue(report.token_usage.cost.priced)
        self.assertIsNotNone(report.token_usage.cost.total_usd)
        payload = report.model_dump(
            mode="json", exclude_none=False, exclude_unset=False, exclude_defaults=False,
        )
        self.assertIn("processing", payload)
        self.assertIn("token_usage", payload)
        self.assertEqual("Partial evidence summary.", payload["synthesis"])
        json.dumps(payload, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
