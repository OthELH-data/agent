"""Tests for populated vs declared column width in `extract_sheet`.

Excel does not shrink a sheet's declared used-range when content is deleted. Sheet1 of the
GreatWhite_2023_0003 exhibit holds 72 columns and declares 16,378. That declared width used to size
`header`, `records` and the preview, so a single tab shipped a 66 KB header of empty strings into every
chunk prompt — 2.2x the 30 KB of actual rows — and told the interpreter it had "+16338 more columns".

These drive `extract_sheet` through a stub worksheet rather than a real file. That stub interface
(`.title`, `.max_row`, `.max_column`, `.iter_rows(values_only=True)`) is exactly the contract the
pyxlsb and xlrd adapters implement, so testing it covers all three readers at once.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extractor import excel_bounded as eb  # noqa: E402


class _FakeSheet:
    """The slice of openpyxl's read-only API that `extract_sheet` consumes.

    `declared_cols` is set independently of the rows, which is the whole point: it reproduces a sheet
    whose dimension record outlived its data.
    """

    def __init__(self, rows, title="Sheet1", declared_cols=None, declared_rows=None):
        self._rows = rows
        self.title = title
        self.max_row = declared_rows if declared_rows is not None else len(rows)
        self.max_column = (declared_cols if declared_cols is not None
                           else max((len(r) for r in rows), default=0))

    def iter_rows(self, values_only=True):
        for r in self._rows:
            yield tuple(r)


class PopulatedWidthTest(unittest.TestCase):

    def test_declared_width_does_not_size_the_payload(self):
        rows = [["Period", "Amount", "Note"],
                ["003.2022", 542, "affiliate"],
                ["012.2022", 271, "affiliate"]]
        sh = eb.extract_sheet(_FakeSheet(rows, declared_cols=16_378))
        self.assertEqual(sh["populated_cols"], 3)
        self.assertEqual(sh["reported_cols"], 16_378, "declared width is kept for provenance")
        self.assertEqual(len(sh["header"]), 3)
        self.assertEqual(len(sh["records"][0]), 3)
        self.assertNotIn("", sh["header"], "no empty declared columns should survive")

    def test_trailing_empty_cells_in_rows_do_not_widen_it(self):
        rows = [["A", "B", None, None, None],
                [1, 2, None, None, None]]
        sh = eb.extract_sheet(_FakeSheet(rows, declared_cols=5))
        self.assertEqual(sh["populated_cols"], 2)

    def test_interior_gaps_keep_their_column_slots(self):
        """`raw_grid` cites cells as c<idx> and the synthesis quotes those, so gaps must not compact."""
        rows = [["A", None, None, None, None, "F"],
                [1, None, None, None, None, 6]]
        sh = eb.extract_sheet(_FakeSheet(rows, declared_cols=900))
        self.assertEqual(sh["populated_cols"], 6, "width runs to the last populated index, not a count")
        self.assertEqual(len(sh["header"]), 6)
        self.assertEqual(sh["header"][5], "F", "column F must stay at index 5")
        self.assertEqual(sh["rows"][0], ["1", "", "", "", "", "6"])
        self.assertIn("c6=6", sh["raw_grid"], "the 1-based c<idx> address must be unchanged")

    def test_labelled_column_with_no_data_is_kept(self):
        """A header with no values under it is a fact about the exhibit, not noise to trim."""
        rows = [["Period", "Amount", "JV Partner Number", "JV Partner Name"],
                ["003.2022", 542, None, None]]
        sh = eb.extract_sheet(_FakeSheet(rows, declared_cols=4_000))
        self.assertEqual(sh["populated_cols"], 4)
        self.assertIn("JV Partner Name", sh["header"])

    def test_data_wider_than_the_header_is_not_cut(self):
        rows = [["A", "B"],
                [1, 2, 3, 4, 5]]
        sh = eb.extract_sheet(_FakeSheet(rows, declared_cols=5))
        self.assertEqual(sh["populated_cols"], 5)
        self.assertEqual(sh["rows"][0], ["1", "2", "3", "4", "5"])
        self.assertIn("col5", sh["records"][0], "unlabelled columns fall back to positional names")

    def test_no_row_is_ever_truncated_by_the_width(self):
        rows = [["A", "B", "C"],
                [1, 2, 3],
                [1, None, None],
                [1, 2, 3]]
        sh = eb.extract_sheet(_FakeSheet(rows, declared_cols=12_000))
        for i, r in enumerate(sh["rows"]):
            self.assertLessEqual(len(r), sh["populated_cols"], f"row {i} exceeds the populated width")
        self.assertEqual(sh["populated_cols"], 3)

    def test_well_formed_sheet_is_unchanged(self):
        rows = [["A", "B", "C"], [1, 2, 3]]
        sh = eb.extract_sheet(_FakeSheet(rows))
        self.assertEqual(sh["populated_cols"], sh["reported_cols"])
        self.assertEqual(len(sh["header"]), 3)

    def test_preview_is_not_sized_to_the_declared_width(self):
        rows = [["A", "B"], [1, 2]]
        narrow = eb.extract_sheet(_FakeSheet(rows, declared_cols=2))["preview_md"]
        wide = eb.extract_sheet(_FakeSheet(rows, declared_cols=16_378))["preview_md"]
        self.assertEqual(narrow, wide, "the preview must not grow with the declared width")

    def test_empty_sheet_does_not_crash(self):
        sh = eb.extract_sheet(_FakeSheet([], declared_cols=16_378))
        self.assertEqual(sh["populated_cols"], 0)
        self.assertEqual(sh["header"], [])
        self.assertEqual(sh["rows"], [])

    def test_header_only_sheet(self):
        sh = eb.extract_sheet(_FakeSheet([["A", "B", "C"]], declared_cols=500))
        self.assertEqual(sh["populated_cols"], 3)
        self.assertEqual(sh["rows"], [])

    def test_sheets_to_json_carries_the_populated_width(self):
        rows = [["Period", "Amount"], ["003.2022", 542]]
        book = {"sheets": [eb.extract_sheet(_FakeSheet(rows, declared_cols=16_378))]}
        entry = eb.sheets_to_json(book)[0]
        self.assertEqual(entry["populated_cols"], 2)
        self.assertEqual(len(entry["header"]), 2)


class RealWorkbookWidthTest(unittest.TestCase):
    """Integration check against the exhibit this fix came from. Skipped when the file is absent."""

    SRC = ("/Users/harshakolachina/Documents/Projects/Shell JV Audit/Data Files/"
           "Exception Reports and Rebuttals by year/2023/Great White/Rebuttal 0/"
           "Audit Report/0003/IR 24 Affiliate Coding Error.xlsx")

    def setUp(self):
        if not os.path.exists(self.SRC):
            self.skipTest("source exhibit not available on this machine")
        self.book = eb.extract_excel_full(self.SRC)

    def _sheet(self, name):
        return next(s for s in self.book["sheets"] if s.get("name") == name)

    def test_sheet1_populated_width(self):
        sh = self._sheet("Sheet1")
        self.assertEqual(sh["populated_cols"], 72)
        self.assertEqual(sh["reported_cols"], 16_378)

    def test_header_payload_is_smaller_than_the_rows_payload(self):
        import json
        sh = self._sheet("Sheet1")
        header_chars = len(json.dumps(sh["header"]))
        rows_chars = len(json.dumps(sh["rows"]))
        self.assertLess(header_chars, rows_chars,
                        "the header must no longer outweigh the data it labels")

    def test_the_columns_the_old_cut_lost_are_present(self):
        header = [str(h).lower() for h in self._sheet("Sheet1")["header"]]
        for want in ("document date", "posting date", "cost + oh", "fiscal year/period"):
            self.assertTrue(any(want in h for h in header), f"{want!r} missing from the header")

    def test_narrow_tab_is_untouched(self):
        sh = self._sheet("Oyler")
        self.assertEqual(sh["populated_cols"], sh["reported_cols"])


if __name__ == "__main__":
    unittest.main()
