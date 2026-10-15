"""The labor adapter must identify its module on every shared LLM call."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import BaseModel


ORCH_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ORCH_DIR.parent
for path in (str(REPO_ROOT), str(ORCH_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from otl import llm_adapter  # noqa: E402


class Reply(BaseModel):
    value: str = ""


class LaborHookModuleTests(unittest.TestCase):
    def test_structured_passes_labor_module_variable(self):
        expected = Reply(value="ok")
        with patch.object(llm_adapter._xllm, "structured_call", return_value=expected) as call:
            result = llm_adapter.structured("intake", "system", "user", Reply)

        self.assertIs(result, expected)
        self.assertEqual("labor", llm_adapter.HOOK_MODULE)
        self.assertEqual(llm_adapter.HOOK_MODULE, call.call_args.kwargs["hook_module"])

    def test_text_passes_labor_module_variable(self):
        with patch.object(llm_adapter._xllm, "text_call", return_value="ok") as call:
            result = llm_adapter.text("intake", "system", "user")

        self.assertEqual("ok", result)
        self.assertEqual(llm_adapter.HOOK_MODULE, call.call_args.kwargs["hook_module"])

    def test_structured_can_mark_only_the_terminal_drafter_as_budget_recovery(self):
        expected = Reply(value="ok")
        with patch.object(llm_adapter._xllm, "structured_call", return_value=expected) as call:
            result = llm_adapter.structured(
                "drafter", "system", "user", Reply, allow_budget_overrun=True,
            )

        self.assertIs(result, expected)
        self.assertTrue(call.call_args.kwargs["allow_budget_overrun"])

    def test_react_turn_is_covered_by_the_labor_budget_hook(self):
        reply = SimpleNamespace(text="done")
        with (
            patch.object(llm_adapter.HOOKS, "before_call") as before,
            patch.object(llm_adapter.HOOKS, "after_call") as after,
            patch.object(llm_adapter._xllm, "_invoke_tier", return_value=(reply, "model")),
            patch.object(llm_adapter._xllm, "record_reply_usage"),
        ):
            result = llm_adapter.invoke_messages("jade_analyzer", [])

        self.assertEqual("done", result)
        self.assertEqual("labor", before.call_args.kwargs["module"])
        self.assertEqual("text_call", before.call_args.kwargs["call_type"])
        self.assertEqual("jade_analyzer", before.call_args.kwargs["stage"])
        self.assertEqual("labor", after.call_args.kwargs["module"])


if __name__ == "__main__":
    unittest.main()
