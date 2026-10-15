from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from otl.agents import supervision as supervision_agent
from otl.supervision_evidence import SupervisionEvidence, analyze_supervision


def _index(rows, selected):
    return {"sources": [{
        "role": "staff", "selected": selected,
        "entries": [{
            "columns": {"reports_to": {"Manager ID": 1}},
            "staff_rows": rows,
        }],
    }]}


def _row(name, key, manager, month, source_row):
    return (name, key, manager, "", "Position", "", month, "", "", "Staff", source_row)


class DeterministicSupervisionTests(unittest.TestCase):
    def test_above_first_level_and_first_level_are_derived_from_org_tree(self):
        rows = [
            _row("Alice Alpha", "1", "", "January", 2),
            _row("Bob Beta", "2", "1", "January", 3),
            _row("Cara Gamma", "3", "2", "January", 4),
            _row("Dan Delta", "4", "1", "January", 5),
        ]
        selected = {
            "Alice Alpha": [{"name_in_file": "Alice Alpha"}],
            "Bob Beta": [{"name_in_file": "Bob Beta"}],
            "Cara Gamma": [{"name_in_file": "Cara Gamma"}],
        }
        result = analyze_supervision(_index(rows, selected), list(selected))
        self.assertTrue(result.above_first_level["Alice Alpha"])
        self.assertFalse(result.above_first_level["Bob Beta"])
        self.assertFalse(result.above_first_level["Cara Gamma"])
        self.assertEqual(result.hierarchy_periods_by_person["Alice Alpha"][0].direct_report_count, 2)
        self.assertEqual(result.hierarchy_periods_by_person["Alice Alpha"][0].manager_report_count, 1)

    def test_conflicting_monthly_levels_are_unresolved(self):
        rows = [
            _row("Alice Alpha", "1", "", "January", 2),
            _row("Bob Beta", "2", "1", "January", 3),
            _row("Cara Gamma", "3", "2", "January", 4),
            _row("Alice Alpha", "1", "", "February", 5),
            _row("Bob Beta", "2", "1", "February", 6),
        ]
        result = analyze_supervision(_index(rows, {"Alice Alpha": [{"name_in_file": "Alice Alpha"}]}),
                                     ["Alice Alpha"])
        self.assertEqual(result.supervision_status_by_person["Alice Alpha"], "unresolved")
        self.assertIn("changes across monthly snapshots", result.unresolved_reasons["Alice Alpha"][-1])

    def test_missing_staff_identity_is_recorded_as_missing(self):
        result = analyze_supervision(_index([], {}), ["Missing Person"])
        self.assertEqual(result.supervision_status_by_person["Missing Person"], "unresolved")
        self.assertEqual(result.supervision_record_status_by_person["Missing Person"], "missing")


class SupervisionNodeFlowTests(unittest.TestCase):
    def _state(self):
        return {
            "case": {"case_id": "case-1"},
            "jade_analyzer": {
                "people_found_in_ledger": ["Alice Alpha", "Bob Beta"],
                "charged_amount_by_person": {"Alice Alpha": 70.0, "Bob Beta": 30.0},
            },
            "common_reports_dir": "/reports", "sandbox_read_roots": ["/reports"],
        }

    def test_normal_deterministic_path_never_constructs_code_executor(self):
        evidence = SupervisionEvidence(
            above_first_level={"Alice Alpha": True, "Bob Beta": False},
            supervision_status_by_person={
                "Alice Alpha": "above_first_level", "Bob Beta": "not_above_first_level"},
            supervision_record_status_by_person={"Alice Alpha": "found", "Bob Beta": "found"},
            reports_summary={"Alice Alpha": "2 direct reports", "Bob Beta": "0 direct reports"},
        )
        with patch.object(supervision_agent, "resolve_staff_listing", return_value="/reports/staff.xlsx"), \
                patch.object(supervision_agent, "build_manifest", return_value={}), \
                patch.object(supervision_agent, "build_person_index", return_value={}), \
                patch.object(supervision_agent, "analyze_supervision", return_value=evidence), \
                patch.object(supervision_agent, "CodeExecutor") as executor:
            update = supervision_agent.supervision_node(self._state())
        executor.assert_not_called()
        self.assertEqual(update["supervision"].supervision_analysis_mode, "deterministic")
        self.assertEqual(update["supervision"].credit_amount, 70.0)
        self.assertEqual(update["supervision"].chargeable_amount, 30.0)

    def test_fallback_receives_only_unresolved_people(self):
        evidence = SupervisionEvidence(
            above_first_level={"Alice Alpha": True},
            supervision_status_by_person={"Alice Alpha": "above_first_level", "Bob Beta": "unresolved"},
            supervision_record_status_by_person={"Alice Alpha": "found", "Bob Beta": "ambiguous"},
            unresolved_reasons={"Bob Beta": ["conflicting reporting relationships"]},
        )
        loop = SimpleNamespace(steps=1, status="finished", result={
                "above_first_level": {"Bob Beta": False},
                "supervision_status_by_person": {"Bob Beta": "not_above_first_level"},
                "supervision_record_status_by_person": {"Bob Beta": "found"},
        })
        with patch.object(supervision_agent, "resolve_staff_listing", return_value="/reports/staff.xlsx"), \
                patch.object(supervision_agent, "build_manifest", return_value={}), \
                patch.object(supervision_agent, "build_person_index", return_value={}), \
                patch.object(supervision_agent, "analyze_supervision", return_value=evidence), \
                patch.object(supervision_agent, "_fallback", return_value=loop) as fallback:
            update = supervision_agent.supervision_node(self._state())
        self.assertEqual(fallback.call_args.kwargs["people"], ["Bob Beta"])
        self.assertEqual(update["supervision"].supervision_status_by_person["Alice Alpha"],
                         "above_first_level")
        self.assertEqual(update["supervision"].supervision_status_by_person["Bob Beta"],
                         "not_above_first_level")


if __name__ == "__main__":
    unittest.main()
