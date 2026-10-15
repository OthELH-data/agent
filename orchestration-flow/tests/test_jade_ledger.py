from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from otl.jade_ledger import extract_jade_ledger, monthly_mcoh_facts
from otl.workbook_manifest import build_manifest


class UnattributedLedgerTests(unittest.TestCase):
    def test_mcoh_row_survives_without_any_person_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unnamed_mcoh.xlsx"
            wb = Workbook()
            ws = wb.active
            ws.title = "JADE"
            ws.append([
                "Joint Venture(JV)", "JV:JOA", "Equity group", "Cost Object WBS/CC (CO)",
                "Account (ACCT)", "WBS:JV Object Type", "Actual Amout in Local Currency",
                "Qty - Local UoM", "Entry Date SL",
            ])
            ws.append(["JV1", "JOA1", "EQ1", "C.US.TEST", "A9820120", "C116", 1000, 6, "01/18/2023"])
            wb.save(path)
            wb.close()

            result = extract_jade_ledger(build_manifest([str(path)]), 1000)

        self.assertEqual(len(result.rows), 1)
        self.assertIsNone(result.rows[0].person)
        self.assertEqual(result.rows[0].mcoh_status, "mcoh")
        self.assertEqual(result.rows[0].qty_local_uom, 6)
        self.assertEqual(monthly_mcoh_facts(result.rows), [])


if __name__ == "__main__":
    unittest.main()
