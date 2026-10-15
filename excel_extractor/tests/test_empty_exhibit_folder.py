"""Tests for `exception_folder_is_empty` — the guard that keeps an evidence-less cycle from running.

Silvertip 2022 exception 0002 is the case behind it. At Rebuttal 1 the exception is closed and denied,
so the auditor filed `02/` with nothing in it. The report carries no Description block for a closed
exception either, so the description fallback scraped a neighbouring exception's paragraph (the
Transocean rig-rate one) and the cycle synthesized against the wrong text and the wrong amount.

The distinction this guard draws: an EMPTY per-exception folder skips the cycle, while a folder that
holds files but no usable evidence — and the flat layout, which has no folder to be empty — still runs
and records 'insufficient evidence'.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.rebuttals import Cycle, exception_folder_is_empty  # noqa: E402


class EmptyExhibitFolderTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.auditor = Path(self._tmp.name) / "Auditors Report - Rebuttal - 1"
        self.auditor.mkdir()
        self.cycle = Cycle(number=1, auditor_dir=self.auditor)

    def tearDown(self):
        self._tmp.cleanup()

    def test_empty_numbered_folder_is_empty(self):
        (self.auditor / "02").mkdir()
        self.assertTrue(exception_folder_is_empty(self.cycle, "0002"))

    def test_dotfiles_and_empty_subfolders_do_not_count_as_content(self):
        folder = self.auditor / "02"
        folder.mkdir()
        (folder / ".DS_Store").write_bytes(b"\x00")
        (folder / "nested").mkdir()
        self.assertTrue(exception_folder_is_empty(self.cycle, "0002"))

    def test_folder_with_a_workbook_is_not_empty(self):
        folder = self.auditor / "02"
        folder.mkdir()
        (folder / "SIL AE-02 JADE.LBR BURDEN.xlsx").write_bytes(b"stub")
        self.assertFalse(exception_folder_is_empty(self.cycle, "0002"))

    def test_nested_file_counts_as_content(self):
        nested = self.auditor / "02" / "02"
        nested.mkdir(parents=True)
        (nested / "support.xlsx").write_bytes(b"stub")
        self.assertFalse(exception_folder_is_empty(self.cycle, "0002"))

    def test_folder_with_only_unusable_files_still_runs(self):
        # Files present but no spreadsheet: that is 'insufficient evidence', not an empty filing.
        folder = self.auditor / "02"
        folder.mkdir()
        (folder / "cover letter.docx").write_bytes(b"stub")
        self.assertFalse(exception_folder_is_empty(self.cycle, "0002"))

    def test_prefixed_folder_name_matches(self):
        (self.auditor / "Ex 02").mkdir()
        self.assertTrue(exception_folder_is_empty(self.cycle, "0002"))

    def test_other_exceptions_empty_folder_is_not_this_ones(self):
        (self.auditor / "03").mkdir()
        (self.auditor / "02" / "x").mkdir(parents=True)
        (self.auditor / "02" / "x" / "e.xlsx").write_bytes(b"stub")
        self.assertFalse(exception_folder_is_empty(self.cycle, "0002"))

    def test_flat_layout_has_no_folder_to_be_empty(self):
        (self.auditor / "EX02-support.xlsx").write_bytes(b"stub")
        self.assertFalse(exception_folder_is_empty(self.cycle, "0002"))

    def test_missing_auditor_side(self):
        self.assertFalse(exception_folder_is_empty(Cycle(number=1), "0002"))
        self.assertFalse(
            exception_folder_is_empty(Cycle(number=1, auditor_dir=Path("/nope/nowhere")), "0002"))


if __name__ == "__main__":
    unittest.main()
