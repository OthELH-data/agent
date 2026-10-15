"""Regression tests for sibling-PDF discovery."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from extractor.invoice_extract import discover_invoices


class InvoiceDiscoveryTests(unittest.TestCase):
    def test_keeps_supporting_pdfs_and_excludes_only_reports_and_correspondence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            exhibit = folder / "AE01.xlsx"
            exhibit.touch()

            excluded = {
                "Audit Report - Final [2024SOV_E100].pdf",
                "External Report.pdf",
                "Rebuttal 1.pdf",
                "Operator Response.pdf",
                "Auditor Reply.pdf",
                "Final [2024SOV_E100].pdf",
            }
            kept = {
                "Vendor Invoice.pdf",
                "POB January.pdf",
                "Helicopter Log.pdf",
                "IR 24 Support.pdf",
                "JIB Detail.pdf",
                "Fleet Status.pdf",
                "Withdrawal Detail.pdf",
                "Cover Letter.pdf",
                "Memo.pdf",
            }
            for name in excluded | kept:
                (folder / name).touch()

            discovered = discover_invoices([exhibit])

            self.assertEqual({path.name for path in discovered}, kept)


if __name__ == "__main__":
    unittest.main()
