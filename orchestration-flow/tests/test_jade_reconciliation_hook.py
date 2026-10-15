"""Deterministic ED-02 reconciliation hook coverage."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import otl  # noqa: F401  — bootstrap shared imports

from shared.hooks_config import clear_hooks_config_cache, node_hook_definitions
from otl.agents import jade_analyzer
from otl.agents.drafter import drafter_node
from otl.graph import route_from_jade_analyzer
from otl.decision_hooks import (JadeAnalyzerHooks, JadeReconciliationFacts,
                                compute_reconciliation, evaluate_jade_hooks)
from otl.output_writer import assemble_report


def _facts(computed: float, questioned: float = 100.0, *, basis: str = "sum",
           loop_status: str = "finished", model_in_range: bool = False,
           model_discrepancy: bool = True) -> JadeReconciliationFacts:
    return JadeReconciliationFacts(
        loop_status=loop_status,
        amount_basis=basis,
        amount_computed=computed,
        amount_questioned=questioned,
        model_amount_in_range=model_in_range,
        model_jade_amount_discrepancy=model_discrepancy,
    )


class JadeReconciliationCalculationTests(unittest.TestCase):
    def test_node_class_owns_jade_policy(self) -> None:
        self.assertEqual("jade_analyzer", JadeAnalyzerHooks.NODE)
        self.assertEqual(0.025, JadeAnalyzerHooks.RECONCILIATION_TOLERANCE)

    def test_exact_and_inclusive_boundaries_pass(self) -> None:
        for computed in (100.0, 97.5, 102.5):
            with self.subTest(computed=computed):
                check = compute_reconciliation(_facts(computed))
                self.assertTrue(check.valid)
                self.assertTrue(check.in_range)

    def test_code_pass_result_wins_over_model_discrepancy_flag(self) -> None:
        evaluations, decision, _snapshot, check = evaluate_jade_hooks(facts=_facts(100.0))
        self.assertTrue(check.in_range)
        self.assertFalse(decision.triggered)
        self.assertEqual("pass", evaluations[0].status)
        self.assertTrue(check.details["model_jade_amount_discrepancy"])
        self.assertTrue(check.details["code_amount_in_range"])

    def test_outside_boundary_triggers_mismatch(self) -> None:
        _evaluations, decision, _snapshot, check = evaluate_jade_hooks(
            facts=_facts(102.5001, model_in_range=True, model_discrepancy=False)
        )
        self.assertTrue(check.valid)
        self.assertFalse(check.in_range)
        self.assertTrue(decision.triggered)
        self.assertEqual(["ED-02"], decision.gate_ids)
        self.assertEqual("mismatch", decision.status)
        self.assertEqual("deny_jade_amount_mismatch", decision.outcome_key)
        self.assertEqual("deny", decision.disposition)
        self.assertFalse(decision.hitl_review)
        self.assertIn("do not reconcile", decision.reasons[0])
        self.assertTrue(check.details["model_amount_in_range"])
        self.assertFalse(check.details["model_jade_amount_discrepancy"])
        self.assertFalse(check.details["code_amount_in_range"])

    def test_negative_amounts_use_signed_comparison_with_absolute_tolerance(self) -> None:
        self.assertTrue(compute_reconciliation(_facts(-102.5, questioned=-100.0)).in_range)
        self.assertFalse(compute_reconciliation(_facts(100.0, questioned=-100.0)).in_range)

    def test_half_basis_is_valid(self) -> None:
        check = compute_reconciliation(_facts(100.0, basis="half"))
        self.assertTrue(check.valid)
        self.assertTrue(check.in_range)

    def test_invalid_or_unfinished_results_do_not_trigger(self) -> None:
        cases = (
            _facts(100.0, basis=""),
            _facts(0.0),
            _facts(100.0, questioned=0.0),
            _facts(float("inf")),
            _facts(100.0, loop_status="max_steps"),
        )
        for facts in cases:
            with self.subTest(facts=facts.model_dump()):
                evaluations, decision, _snapshot, check = evaluate_jade_hooks(facts=facts)
                self.assertFalse(check.valid)
                self.assertFalse(decision.triggered)
                self.assertEqual("not_applicable", evaluations[0].status)


class JadeReconciliationConfigTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_hooks_config_cache()

    def test_default_config_registers_ed02_only_under_jade(self) -> None:
        rules = node_hook_definitions("labor", "jade_analyzer")
        self.assertEqual(["ED-02"], list(rules))
        self.assertEqual("jade_amount_reconciliation", rules["ED-02"].detector)

    def test_disabled_ed02_does_not_route_to_drafter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hooks.yaml"
            path.write_text(
                """schema_version: 1
labor:
  hooks:
    jade_analyzer:
      ED-02:
        enabled: false
        detector: jade_amount_reconciliation
        reason: Amounts do not reconcile.
        outcomes:
          mismatch: deny_jade_amount_mismatch
""",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"HOOKS_CONFIG_FILE": str(path)}):
                clear_hooks_config_cache()
                evaluations, decision, _snapshot, check = evaluate_jade_hooks(facts=_facts(200.0))
        self.assertTrue(check.valid)
        self.assertEqual("disabled", evaluations[0].status)
        self.assertFalse(decision.triggered)
        self.assertEqual("orchestrator", route_from_jade_analyzer({
            "jade_hook_decision": decision.model_dump(mode="json"),
            "analysis_blocked": False,
        }))

    def test_missing_public_reason_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hooks.yaml"
            path.write_text(
                """schema_version: 1
labor:
  hooks:
    jade_analyzer:
      ED-02:
        enabled: true
        detector: jade_amount_reconciliation
        outcomes:
          mismatch: deny_jade_amount_mismatch
""",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"HOOKS_CONFIG_FILE": str(path)}):
                clear_hooks_config_cache()
                with self.assertRaisesRegex(ValueError, "reason"):
                    node_hook_definitions("labor", "jade_analyzer")


class JadeReconciliationIntegrationTests(unittest.TestCase):
    def test_jade_node_overrides_model_flags_and_routes_ed02_to_drafter(self) -> None:
        manifest = {"workbooks": [{"name": "JADE.xlsx", "shapes": []}], "unreadable": []}
        loop = SimpleNamespace(
            status="finished", steps=1, last_observation="",
            result={
                "nl_summary": "JADE reconstruction completed.",
                "amount_basis": "sum",
                "amount_computed": 200.0,
                "amount_questioned": 100.0,
                "amount_in_range": True,
                "jade_amount_discrepancy": False,
            },
        )
        state = {
            "case": {"case_id": "GreatWhite_2024_0010"},
            "intake": {"dollars": 100.0, "people_found": ["Intake Lead"]},
            "exception_exhibits": ["JADE.xlsx"],
        }
        with patch.object(jade_analyzer, "_jade_workbooks", return_value=["JADE.xlsx"]), \
             patch.object(jade_analyzer, "build_manifest", return_value=manifest), \
             patch.object(jade_analyzer.ledger_people, "find_ledger_tabs", return_value=[{"tab": "JADE"}]), \
             patch.object(jade_analyzer.ledger_people, "harvest_with_diagnostics",
                          return_value=(["Alice Example"], ["JADE.xlsx!JADE"], [])), \
             patch.object(jade_analyzer.ledger_people, "extract_people", return_value=["Alice Example"]) as extract_people, \
             patch.object(jade_analyzer.ledger_people, "charged_amount_by_person",
                          return_value={"Alice Example": 200.0}), \
             patch.object(jade_analyzer, "major_construction_jv_object_types", return_value={}), \
             patch.object(jade_analyzer, "CodeExecutor", return_value=object()), \
             patch.object(jade_analyzer, "run_agent_loop", return_value=loop):
            result = jade_analyzer.jade_analyzer_node(state)

        extract_people.assert_called_once_with(
            "jade_people", ["Alice Example"], intake_people=["Intake Lead"],
        )
        self.assertFalse(result["jade_analyzer"].amount_in_range)
        self.assertTrue(result["jade_analyzer"].jade_amount_discrepancy)
        self.assertTrue(result["jade_hook_decision"]["triggered"])
        self.assertIn("jade_analyzer", result["hook_status"])
        self.assertEqual("drafter", route_from_jade_analyzer(result))

    def test_drafter_uses_generic_ed02_denial_and_questioned_amount(self) -> None:
        state = {
            "jade_hook_decision": {
                "triggered": True,
                "gate_ids": ["ED-02"],
                "status": "mismatch",
                "outcome_key": "deny_jade_amount_mismatch",
                "reasons": ["JADE/general-ledger amounts do not reconcile to the auditor's stated amount."],
            },
            "jade_analyzer": {"amount_questioned": 281351.04},
            "auditor_question": ["job_title"],
            "case": {"case_id": "Mars_2024_0018"},
        }
        with patch("otl.agents.drafter.llm.structured", side_effect=AssertionError("LLM called")):
            result = drafter_node(state)

        response = result["drafter"].co_owner_response
        self.assertEqual("deny", result["drafter"].disposition)
        self.assertFalse(result["hitl_review"])
        self.assertEqual(
            "The total amount provided in the JADE does not reconcile to the questioned amount of "
            "$281,351.04. This exception is denied. No credit is due.",
            response,
        )

    def test_separate_manual_request_preserves_hitl_precedence(self) -> None:
        state = {
            "jade_hook_decision": {
                "triggered": True, "gate_ids": ["ED-02"], "status": "mismatch",
                "outcome_key": "deny_jade_amount_mismatch", "reasons": ["Mismatch."],
            },
            "jade_analyzer": {"amount_questioned": 100.0},
            "additional_support_requested": True,
            "auditor_question": ["job_title", "additional_support_requested"],
            "case": {"case_id": "Mars_2024_0018"},
        }
        support = SimpleNamespace(response="The separately requested schedule will be provided.")
        with patch("otl.agents.drafter.llm.structured", return_value=support):
            result = drafter_node(state)
        self.assertEqual("needs_evidence", result["drafter"].disposition)
        self.assertTrue(result["hitl_review"])
        self.assertIn("No credit is due.", result["drafter"].co_owner_response)
        self.assertIn("separately requested", result["drafter"].co_owner_response)

    def test_jade_hook_snapshot_and_decision_serialize_in_report(self) -> None:
        report = assemble_report(
            {
                "case": {"case_id": "Mars_2024_0018", "exc_id": "0018"},
                "jade_analyzer": {
                    "amount_questioned": 100.0,
                    "jade_hook_decision": {"triggered": True, "gate_ids": ["ED-02"]},
                    "jade_hook_evaluations": [{"gate_id": "ED-02", "status": "mismatch"}],
                },
                "hook_status": {"jade_analyzer": {
                    "triggered": True, "triggered_gate_ids": ["ED-02"],
                }},
                "drafter": {"disposition": "deny", "draft_response": "No credit is due."},
            },
            run_id="Mars_2024_0018", elapsed_minutes=0.1,
            token_usage={}, trace_events=[],
        )
        self.assertEqual(["ED-02"], report.hooks["jade_analyzer"]["triggered_gate_ids"])
        self.assertEqual(
            ["ED-02"], report.stages["jade_analyzer"]["jade_hook_decision"]["gate_ids"],
        )


if __name__ == "__main__":
    unittest.main()
