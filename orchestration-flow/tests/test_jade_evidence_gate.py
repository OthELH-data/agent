"""JADE stops safely when verification was bypassed and no ledger can be analyzed."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from otl.agents.jade_analyzer import jade_analyzer_node


class JadeEvidenceGateTests(unittest.TestCase):
    def _run(self, workbooks: list[dict], unreadable: list[dict]) -> dict:
        manifest = {"workbooks": workbooks, "unreadable": unreadable}
        state = {"case": {"case_id": "GreatWhite_2024_0010"}, "intake": {"dollars": 100.0}}
        with patch("otl.agents.jade_analyzer._jade_workbooks", return_value=["Summary.xlsx", "JADE.xlsx"]), \
             patch("otl.agents.jade_analyzer.build_manifest", return_value=manifest):
            return jade_analyzer_node(state)

    def test_readable_workbooks_without_a_ledger_defer_operationally(self) -> None:
        result = self._run([{"name": "Summary.xlsx", "shapes": []}], [])

        self.assertTrue(result["analysis_blocked"])
        self.assertTrue(result["hitl_review"])
        self.assertEqual(result["final_mode"], "needs_evidence")
        self.assertEqual(result["jade_analyzer"].evidence_gate_status, "")
        self.assertEqual(result["jade_analyzer"].evidence_access_issues, [])

    def test_unreadable_ledger_defers_for_access_follow_up(self) -> None:
        unreadable = [{"name": "JADE Detail.xlsx", "errors": {"profile": "encrypted workbook"}}]
        result = self._run([], unreadable)

        self.assertTrue(result["analysis_blocked"])
        self.assertEqual(result["hitl_stage"], "jade_analyzer")
        self.assertIn("JADE Detail.xlsx", result["jade_analyzer"].evidence_access_issues[0])

    def test_mixed_readable_summary_and_unreadable_workbook_defers(self) -> None:
        unreadable = [{"name": "JADE Detail.xlsx", "errors": {"profile": "cannot open"}}]
        result = self._run([{"name": "Summary.xlsx", "shapes": []}], unreadable)

        self.assertTrue(result["analysis_blocked"])
        self.assertTrue(result["jade_analyzer"].evidence_access_issues)

    def test_readable_ledger_without_usable_detail_denies(self) -> None:
        manifest = {"workbooks": [{"name": "JADE.xlsx", "shapes": []}], "unreadable": []}
        state = {"case": {"case_id": "GreatWhite_2024_0010"}, "intake": {"dollars": 100.0}}
        with patch("otl.agents.jade_analyzer._jade_workbooks", return_value=["JADE.xlsx"]), \
             patch("otl.agents.jade_analyzer.build_manifest", return_value=manifest), \
             patch("otl.agents.jade_analyzer.ledger_people.find_ledger_tabs", return_value=[{"tab": "JADE"}]), \
             patch("otl.agents.jade_analyzer.ledger_people.harvest_with_diagnostics", return_value=([], [], [])):
            result = jade_analyzer_node(state)

        self.assertTrue(result["analysis_blocked"])
        self.assertFalse(result["jade_analyzer"].evidence_access_issues)


if __name__ == "__main__":
    unittest.main()
