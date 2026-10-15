"""Tests for `_select_columns` / `_render_tab` — which columns of a wide tab the prompt actually shows.

The bug these guard against: the sampled table kept the first 40 columns of a tab. Sheet1 of the
GreatWhite_2023_0003 exhibit has 72 populated columns, and the four the exception is argued over sit
past that cut — `Document Date` (c43), `Entry Date SL` (c44), `Posting date` (c46) and `Cost + OH`
(c66). So the model was shown 40 columns of joint-venture codes and document numbers, and none of the
dates or the money. Column position carries no meaning in a ledger export, which is what makes a
positional cut a coin flip on the evidence that matters.

The contract under test:

  - a typed column (period, date, number) is never dropped in favour of a text column
  - a column carrying a stated total is never dropped
  - every column is labelled with its `c<n>` index, so a figure read here can be cited
  - what was withheld is NAMED, not just counted, so "40 columns not shown" cannot be read as
    "the rest were the unimportant ones"

Everything runs offline on synthetic tabs.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extractor import explain  # noqa: E402
from extractor.column_profile import profile_columns  # noqa: E402


def tab_from(header, rows):
    """Build the slice of a sheet dict that `_render_tab` reads, profile included."""
    width = max(len(header), max((len(r) for r in rows), default=0))
    profs, totals = profile_columns(header, rows, width)
    return {
        "header": header,
        "populated_cols": width,
        "rows": [[("" if v is None else str(v)) for v in r] for r in rows],
        "column_profile": profs,
        "total_rows": totals,
    }


class SelectionTest(unittest.TestCase):

    def test_a_typed_column_past_the_cap_is_kept_over_text_before_it(self):
        """The Sheet1 shape in miniature: filler text first, the dates and the money last."""
        header = [f"Note {i}" for i in range(30)] + ["Posting date", "Cost + OH"]
        rows = [[f"free text {d}-{j}" for j in range(30)] + [f"{d:02d}.03.2022", 100.0 + d]
                for d in range(1, 9)]
        tab = tab_from(header, rows)
        kinds = {p["header"]: p["kind"] for p in tab["column_profile"]}
        self.assertEqual(kinds["Posting date"], "date", "test premise: it must profile as a date")
        sel = explain._select_columns(tab, len(header), cap=8)
        self.assertIn(30, sel, "the date column was dropped for text ahead of it")
        self.assertIn(31, sel, "the money column was dropped for text ahead of it")

    def test_a_column_carrying_a_stated_total_is_kept(self):
        """A column the sheet itself totals is the one an exception's amount comes from."""
        header = [f"Code {i}" for i in range(20)] + ["Amount"]
        rows = [[f"x{i}{j}" for j in range(20)] + [100.0] for i in range(6)]
        rows.append([None] * 20 + [600.0])            # the sheet's own totals row
        tab = tab_from(header, rows)
        self.assertEqual([t["index"] for t in tab["total_rows"]], [6], "test premise: total detected")
        sel = explain._select_columns(tab, len(header), cap=4)
        self.assertIn(20, sel)

    def test_selection_comes_back_in_sheet_order(self):
        """The table must read left to right as the sheet does, whatever the ranking decided."""
        header = ["Note", "Amount", "Other", "Date"]
        rows = [["a", 1.0, "b", "01.03.2022"], ["c", 2.0, "d", "02.03.2022"]]
        sel = explain._select_columns(tab_from(header, rows), 4, cap=3)
        self.assertEqual(sel, sorted(sel))

    def test_a_constant_text_column_loses_to_one_that_varies(self):
        """A column holding the same value on every row cannot tell one row from another, and the
        profile already states the single value it holds."""
        header = ["Constant", "Varies"]
        rows = [["SAME", f"WBS-{i}"] for i in range(5)]
        sel = explain._select_columns(tab_from(header, rows), 2, cap=1)
        self.assertEqual(sel, [1], "the constant column was shown instead of the varying one")

    def test_a_constant_number_column_is_not_demoted(self):
        """A rate that happens not to vary is still part of the arithmetic, so shape wins here."""
        header = ["Varies", "OH RATE"]
        rows = [[f"WBS-{i}", 0.025] for i in range(5)]
        sel = explain._select_columns(tab_from(header, rows), 2, cap=1)
        self.assertEqual(sel, [1])

    def test_free_text_loses_to_a_labelling_column(self):
        """A column where nearly every row differs takes a date column's width and settles nothing."""
        header = ["Free", "Label"]
        rows = [[f"unique note {i}", "FREIGHT" if i % 2 else "FUEL"] for i in range(10)]
        sel = explain._select_columns(tab_from(header, rows), 2, cap=1)
        self.assertEqual(sel, [1])

    def test_a_narrow_tab_shows_every_column(self):
        header = ["A", "B", "C"]
        rows = [[1, "x", "01.03.2022"], [2, "y", "02.03.2022"]]
        self.assertEqual(explain._select_columns(tab_from(header, rows), 3), [0, 1, 2])

    def test_without_a_profile_it_falls_back_to_the_leading_columns(self):
        """The old behaviour, and the only thing left to do with no shape information."""
        tab = {"header": [f"c{i}" for i in range(50)], "populated_cols": 50}
        self.assertEqual(explain._select_columns(tab, 50, cap=5), [0, 1, 2, 3, 4])

    def test_the_cap_is_honoured(self):
        header = [f"Amount {i}" for i in range(40)]
        rows = [[float(i + j) for j in range(40)] for i in range(4)]
        sel = explain._select_columns(tab_from(header, rows), 40)
        self.assertEqual(len(sel), explain._MAX_TABLE_COLS)


class RenderTabTest(unittest.TestCase):

    def test_each_column_is_labelled_with_its_index(self):
        """A figure quoted from this table has to be citable the same way as one from the raw grid,
        which addresses cells as `c<n>`. The indices here are 1-based to match it."""
        header = ["Period", "Amount"]
        rows = [["003.2022", 542.0], ["012.2022", 271.0]]
        out = explain._render_tab(tab_from(header, rows), 10)
        first = out.splitlines()[0]
        self.assertIn("c1 Period", first)
        self.assertIn("c2 Amount", first)

    def test_rows_are_read_positionally_so_duplicate_headers_survive(self):
        """JADE ships Key/Text pairs under one label. Header-keyed records collapse them, and the two
        then show the same value — which is how a code and its description became one column."""
        header = ["Equity group", "Equity group", "Amount"]
        rows = [["E06", "E06-per Traci", 542.0], ["E07", "E07-per Traci", 271.0]]
        out = explain._render_tab(tab_from(header, rows), 10)
        self.assertIn("E06 | E06-per Traci", out, "the duplicate-labelled columns collapsed")

    def test_the_withheld_columns_are_named_not_just_counted(self):
        """A bare count invites the assumption that the missing columns were the unimportant ones."""
        header = [f"Note {i}" for i in range(30)] + ["Posting date"]
        rows = [[f"free text {i}-{j}" for j in range(30)] + [f"{i:02d}.03.2022"]
                for i in range(1, 9)]
        out = explain._render_tab(tab_from(header, rows), 10)
        self.assertIn("columns not shown here", out)
        self.assertIn("Note ", out.splitlines()[-1], "no withheld column was named")
        self.assertIn("COLUMN PROFILE", out.splitlines()[-1],
                      "the note must say where the withheld columns can still be read")

    def test_fewer_columns_buys_back_rows(self):
        """The cell budget is rows x columns, so narrowing the table lets more rows through. At 72
        columns the budget allowed 16 rows; the same budget over 24 columns allows 50."""
        header = [f"Amount {i}" for i in range(72)]
        rows = [[float(i + j) for j in range(72)] for i in range(60)]
        out = explain._render_tab(tab_from(header, rows), 60)
        body = len(out.splitlines()) - 3          # header, rule, and the trailing note
        self.assertGreater(body, 16, "narrowing the table did not buy any rows back")

    def test_a_tab_that_failed_to_read_says_so(self):
        self.assertIn("could not read tab", explain._render_tab({"error": "boom"}, 10))

    def test_records_still_work_when_a_caller_has_no_positional_rows(self):
        """Not every caller supplies `rows`. The header-keyed path stays available for them."""
        tab = {"header": ["A", "B"], "populated_cols": 2,
               "records": [{"A": "1", "B": "2"}, {"A": "3", "B": "4"}]}
        out = explain._render_tab(tab, 10)
        self.assertIn("| 1 | 2 |", out)
        self.assertIn("| 3 | 4 |", out)


class RealExhibitTest(unittest.TestCase):
    """The four columns the exception is actually about. Skips when the exhibit is absent."""

    SRC = ("/Users/harshakolachina/Documents/Projects/Shell JV Audit/Data Files/"
           "Exception Reports and Rebuttals by year/2023/Great White/Rebuttal 0/"
           "Audit Report/0003/IR 24 Affiliate Coding Error.xlsx")

    WANTED = ("Document Date", "Entry Date SL", "Posting date", "Cost + OH")

    def setUp(self):
        if not os.path.exists(self.SRC):
            self.skipTest("source exhibit not available on this machine")
        from extractor import excel_bounded as eb
        book = eb.extract_excel_full(self.SRC)
        self.sheet = next(s for s in book["sheets"] if s["name"] == "Sheet1")

    def test_the_dates_and_the_money_survive_the_column_budget(self):
        sh = self.sheet
        sel = set(explain._select_columns(sh, sh["populated_cols"]))
        header = sh["header"]
        for name in self.WANTED:
            idx = [i for i, h in enumerate(header) if str(h).strip() == name]
            with self.subTest(column=name):
                self.assertTrue(idx, f"{name} is missing from the tab's header")
                self.assertTrue(any(i in sel for i in idx),
                                f"{name} (c{idx[0] + 1}) was dropped from the sampled table")
                self.assertTrue(all(i >= 40 for i in idx),
                                f"test premise: {name} must sit past the old 40-column cut")

    def test_the_rendered_table_holds_the_charge_dates_running_to_december(self):
        """The window the synthesis got wrong. December must be visible in the table, not only in the
        profile — the rows are what the model quotes from."""
        out = explain._render_tab(self.sheet, 40)
        self.assertIn("Posting date", out.splitlines()[0])
        self.assertIn("12.2022", out, "no December charge row reached the prompt")


if __name__ == "__main__":
    unittest.main()
