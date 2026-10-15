from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from otl.agents.jade_analyzer import jade_analyzer_node
from otl.jade_ledger import assign_people, extract_jade_ledger, monthly_mcoh_facts
from otl.workbook_manifest import build_manifest


DATA_ROOT = "/Users/harshakolachina/Documents/Projects/Shell JV Audit/Data Files/Exception Reports and Rebuttals by year"


class RealJadeCases(unittest.TestCase):
    def _extract(self, paths, questioned, people):
        for path in paths:
            if not os.path.exists(path):
                self.skipTest(f"real case workbook unavailable: {path}")
        result = extract_jade_ledger(build_manifest(paths), questioned)
        assign_people(result.rows, people)
        return result

    def test_great_white_2024_0046_mcoh_and_duplicate_population(self):
        base = f"{DATA_ROOT}/2024/GreatWhite/Rebuttal 0/Auditor Report/0046"
        result = self._extract([
            f"{base}/GW EXC046 Jade Lines.xlsx",
            f"{base}/JADE line for IR 34.xlsx",
        ], 51144.33, ["Dupre Michael H", "Andy Kihm", "Kihm Andrew Douglas"])
        self.assertEqual(len(result.rows), 34)
        self.assertEqual(result.amount_basis, "sum")
        self.assertEqual(result.amount_computed, 51144.33)
        self.assertTrue(result.amount_in_range)
        self.assertTrue(any(row.mcoh_status == "mcoh" for row in result.rows))
        self.assertEqual({row.person for row in result.rows if row.person},
                         {"Dupre Michael H", "Andy Kihm", "Kihm Andrew Douglas"})
        self.assertTrue(monthly_mcoh_facts(result.rows))

    def test_great_white_2024_0046_node_uses_structured_path_not_code(self):
        base = f"{DATA_ROOT}/2024/GreatWhite/Rebuttal 0/Auditor Report/0046"
        paths = [f"{base}/GW EXC046 Jade Lines.xlsx", f"{base}/JADE line for IR 34.xlsx"]
        for path in paths:
            if not os.path.exists(path):
                self.skipTest(f"real case workbook unavailable: {path}")
        state = {
            "case": {"case_id": "GreatWhite_2024_0046", "asset": "Great White"},
            "intake": {"dollars": 51144.33},
            "exception_exhibits": paths,
        }
        with patch("otl.agents.jade_analyzer.ledger_people.extract_people", return_value=[
            "Dupre Michael H", "Andy Kihm", "Kihm Andrew Douglas",
        ]), patch("otl.agents.jade_analyzer.run_agent_loop") as fallback:
            update = jade_analyzer_node(state)
        self.assertEqual(update["jade_analyzer"].jade_analysis_mode, "structured")
        self.assertEqual(update["jade_analyzer"].amount_computed, 51144.33)
        self.assertTrue(update["jade_analyzer"].major_construction_charge)
        fallback.assert_not_called()

    def test_mars_2025_0023_removes_summary_double_count(self):
        path = f"{DATA_ROOT}/2025/Mars et al/Rebuttal 1/Auditor Report/0023/Exc 23 BP Rebuttal 1.xlsx"
        result = self._extract([path], 3265475.55, ["Cado, Blake", "Harris, Bo", "Gloudemans, Dan"])
        self.assertEqual(result.amount_basis, "sum")
        self.assertEqual(result.amount_computed, 3265475.55)
        self.assertEqual(len(result.rows), 982)
        self.assertTrue(any(row.mcoh_status == "mcoh" for row in result.rows))
        self.assertTrue(any(row.mcoh_status == "non_mcoh" for row in result.rows))

    def test_great_white_2023_0003_line_text_name_and_non_mcoh(self):
        path = f"{DATA_ROOT}/2023/Great White/Rebuttal 0/Audit Report/0003/IR 24 Affiliate Coding Error.xlsx"
        result = self._extract([path], 47365.25, ["Kenneth Oyler"])
        self.assertEqual(len(result.rows), 31)
        self.assertEqual(result.amount_basis, "sum")
        self.assertEqual(result.amount_computed, 47365.25)
        self.assertEqual({row.person for row in result.rows}, {"Kenneth Oyler"})
        self.assertTrue(all(row.mcoh_status == "non_mcoh" for row in result.rows))


if __name__ == "__main__":
    unittest.main()
