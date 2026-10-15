"""Tests for `blocks` — telling apart the separate tables stacked inside one worksheet tab.

The bug this guards against: the `Oyler` tab of `IR 24 Affiliate Coding Error.xlsx` holds a helicopter
trip table and, below a blank row, an unrelated personnel table. Read as one table, the second table's
HEADER ROW becomes a data row of the first, so `TRIP ID` ends up holding `41341, 41880, 'Current POB'`
and every column profiles as "mixed". No figure from that tab can be stated correctly until the two
tables are told apart.

The opposite mistake is just as bad and easier to make, so most of these tests are about NOT splitting:

  - a blank row alone is not a boundary. Sheet1 has a blank row above its totals row, and splitting
    there would strand the totals row away from the rows it reconciles against.
  - a row of text alone is not a header. A run of description rows must stay with its table.
  - coverage must tile every row. A block set with a gap silently drops rows from the statistics, which
    is the same class of failure as reporting a sampled range as if it covered everything.

Everything here runs offline on synthetic rows. The real-exhibit test skips when the file is absent.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extractor import blocks as bk  # noqa: E402


def seg(header, rows, width=None):
    """Segment some rows, defaulting the width to the widest row."""
    width = width if width is not None else max(len(header), max((len(r) for r in rows), default=0))
    return bk.segment(header, rows, width)


def spans(blocks):
    """Each block as `(start, end, header_row)` — the shape assertions care about."""
    return [(b["start"], b["end"], b["header_row"]) for b in blocks]


def assert_tiles(case, blocks, n_rows):
    """Every row belongs to exactly one block, and the ranges are contiguous with no gaps."""
    case.assertTrue(blocks, "segmentation returned no blocks at all")
    case.assertEqual(blocks[0]["start"], 0, "coverage does not start at the first row")
    case.assertEqual(blocks[-1]["end"], n_rows - 1, "coverage does not reach the last row")
    for k in range(len(blocks) - 1):
        case.assertEqual(blocks[k + 1]["start"], blocks[k]["end"] + 1,
                         f"gap or overlap between block {k} and block {k + 1}")


class SingleTableTest(unittest.TestCase):
    """The common case. A tab holding one table must come back as exactly one block, unchanged."""

    def test_a_clean_ledger_is_one_block(self):
        rows = [[i, float(i * 10), dt.date(2022, 1, 1)] for i in range(1, 21)]
        blocks = seg(["ID", "Amount", "Date"], rows)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(spans(blocks), [(0, 19, None)])
        self.assertEqual(blocks[0]["n_rows"], 20)
        self.assertEqual(blocks[0]["header"], ["ID", "Amount", "Date"])

    def test_an_empty_sheet_yields_no_blocks(self):
        self.assertEqual(bk.segment([], [], 0), [])
        self.assertEqual(bk.segment(["A"], [], 1), [])

    def test_a_sheet_of_only_blank_rows_is_one_block(self):
        """Nothing to split, and nothing to report — but the coverage invariant still has to hold."""
        rows = [[None, None], [None, None], ["", ""]]
        blocks = seg(["A", "B"], rows)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["n_rows"], 0)
        self.assertIsNone(blocks[0]["data_start"])
        assert_tiles(self, blocks, 3)


class BlankRowIsNotABoundaryTest(unittest.TestCase):
    """A gap alone means nothing. This is the Sheet1 totals-row case, which must not split."""

    def test_a_totals_row_below_a_blank_row_stays_with_its_table(self):
        """The exact Sheet1 shape: 31 charge rows, a blank row, then the totals row.

        Splitting here would put the totals row in a block of its own, and the sum it is meant to
        reconcile against would then be computed over rows that exclude it — or over rows that do not
        include the charges. Either way the reconciliation the auditor's own sheet asserts is lost.
        """
        rows = [[f"C{i}", 100.0, dt.date(2022, i % 12 + 1, 1)] for i in range(31)]
        rows.append([None, None, None])
        rows.append([None, 3100.0, None])            # totals row: narrow span, no label
        blocks = seg(["Code", "Amount", "Date"], rows)
        self.assertEqual(len(blocks), 1, f"blank row split the table: {spans(blocks)}")
        self.assertEqual(blocks[0]["end"], 32)
        assert_tiles(self, blocks, 33)

    def test_a_narrow_row_inside_a_wide_table_is_containment_not_a_new_table(self):
        """A subtotal row fills a few columns of a wide table. Contained is not disjoint.

        Overlap is measured against the narrower span, so a row sitting inside the table's columns
        scores as fully overlapping. Measured against the wider span it would look like a different
        table and be split off.
        """
        rows = [[f"C{i}", "desc", 10.0, 20.0, 30.0] for i in range(8)]
        rows.append([None] * 5)
        rows.append([None, None, 80.0, 160.0, 240.0])
        blocks = seg(["Code", "Desc", "A", "B", "C"], rows)
        self.assertEqual(len(blocks), 1, f"contained subtotal row was split off: {spans(blocks)}")

    def test_many_blank_separated_runs_of_one_table_stay_one_block(self):
        """A sparse export has gaps everywhere. None of them is a table boundary."""
        rows = []
        for i in range(10):
            rows.append([f"C{i}", float(i), dt.date(2022, 6, 1)])
            rows.append([None, None, None])
        blocks = seg(["Code", "Amount", "Date"], rows)
        self.assertEqual(len(blocks), 1, f"gaps split a single table: {spans(blocks)}")
        assert_tiles(self, blocks, len(rows))


class TextRowIsNotAHeaderTest(unittest.TestCase):
    """Text alone proves nothing. A header is text WHERE THE TABLE KEEPS TYPED VALUES."""

    def test_a_run_of_description_rows_does_not_split(self):
        """These rows put text in text columns. There is no contradiction, so no boundary."""
        rows = [
            [1, 10.0, "opening balance"],
            [2, 20.0, "freight"],
            [None, None, None],
            [3, 30.0, "port charges"],
        ]
        blocks = seg(["ID", "Amount", "Note"], rows)
        self.assertEqual(len(blocks), 1, f"description rows split the table: {spans(blocks)}")

    def test_an_all_text_table_never_splits_on_the_header_signal(self):
        """With no typed columns there is nothing for a label to contradict, so leave it whole.

        A guess here would be worse than doing nothing: an all-text tab is exactly where a header row
        and a data row are indistinguishable by shape.
        """
        rows = [["a", "b", "c"], ["d", "e", "f"], ["Name", "Note", "Ref"], ["g", "h", "i"]]
        blocks = seg(["A", "B", "C"], rows)
        self.assertEqual(len(blocks), 1, f"all-text rows were split: {spans(blocks)}")

    def test_a_two_column_table_cannot_split_on_the_header_signal(self):
        """A known and deliberate limit, recorded so it is not mistaken for a bug.

        Three labels is the minimum evidence, so on a two-column tab a stacked table is only found by
        the disjoint-columns signal. The alternative is worse: with two cells, an ordinary note like
        "Adjusted | see below" contradicts a period-and-number table just as convincingly as a real
        header would, and splitting on it would cut a table in half.
        """
        rows = [["003.2022", 542.0],
                ["004.2022", 271.0],
                [None, None],
                ["Region", "POB"],
                ["GOM", 2688]]
        blocks = seg(["Period", "Amount"], rows)
        self.assertEqual(len(blocks), 1, spans(blocks))

    def test_two_stray_labels_are_a_note_not_a_header(self):
        """Below the minimum cell count. A merged title or a footnote must not split a table."""
        rows = [[1, 10.0, dt.date(2022, 1, 1)],
                [2, 20.0, dt.date(2022, 2, 1)],
                [None] * 3,
                ["see attached", None, None],
                [3, 30.0, dt.date(2022, 3, 1)]]
        blocks = seg(["ID", "Amount", "Date"], rows)
        self.assertEqual(len(blocks), 1, f"a stray note split the table: {spans(blocks)}")


class StackedTableTest(unittest.TestCase):
    """The Oyler case, which is what this module exists for."""

    def test_a_second_table_after_a_blank_row_is_split_off(self):
        """A header row contradicting the established column shapes is the corroborating signal."""
        rows = [
            [1, 100.0, dt.date(2022, 1, 1), "a"],
            [2, 200.0, dt.date(2022, 2, 1), "b"],
            [3, 300.0, dt.date(2022, 3, 1), "c"],
            [None] * 4,
            ["Code", "Qty", "Total", "Unit"],      # a header: text where the table keeps typed values
            ["W1", 5, 50.0, "kg"],
            ["W2", 6, 60.0, "kg"],
        ]
        blocks = seg(["ID", "Amount", "Date", "Tag"], rows)
        self.assertEqual(len(blocks), 2, f"stacked tables were read as one: {spans(blocks)}")
        self.assertEqual(blocks[1]["header_row"], 4)
        self.assertEqual(blocks[1]["header"][:4], ["Code", "Qty", "Total", "Unit"])
        assert_tiles(self, blocks, len(rows))

    def test_a_disjoint_column_range_splits_without_any_header(self):
        """The other corroborating signal: a run occupying substantially different columns."""
        rows = [
            [1, 100.0, None, None, None, None],
            [2, 200.0, None, None, None, None],
            [None] * 6,
            [None, None, None, None, 9.0, 8.0],
            [None, None, None, None, 7.0, 6.0],
        ]
        blocks = seg(["A", "B", "C", "D", "E", "F"], rows)
        self.assertEqual(len(blocks), 2, f"disjoint ranges were read as one: {spans(blocks)}")
        self.assertIsNone(blocks[1]["header_row"], "no header row was present to claim")
        assert_tiles(self, blocks, len(rows))

    def test_the_second_block_reports_its_own_row_count_not_the_padding(self):
        """`data_start`/`data_end` exclude the separator blanks and the block's own header row.

        A range padded out to the separator would overstate the table's size, which is the same kind of
        overstatement — a figure quoted over the wrong row set — that this package exists to prevent.
        """
        rows = [
            [1, 100.0, dt.date(2022, 1, 1)],
            [2, 200.0, dt.date(2022, 2, 1)],
            [None] * 3,
            [None] * 3,
            ["Code", "Qty", "Total"],
            ["W1", 5, 50.0],
        ]
        blocks = seg(["ID", "Amount", "Date"], rows)
        self.assertEqual(len(blocks), 2, spans(blocks))
        first, second = blocks
        self.assertEqual((first["data_start"], first["data_end"]), (0, 1))
        self.assertEqual(first["n_rows"], 2, "trailing blanks were counted as rows")
        self.assertEqual((second["data_start"], second["data_end"]), (5, 5))
        self.assertEqual(second["n_rows"], 1, "the header row was counted as a data row")

    def test_blank_header_cells_do_not_render_as_the_word_none(self):
        """A discovered header row is padded to the sheet width. Those cells must render empty.

        Otherwise the string "None" reaches the prompt looking like a column name.
        """
        rows = [
            [1, 100.0, dt.date(2022, 1, 1), "a", "p"],
            [2, 200.0, dt.date(2022, 2, 1), "b", "q"],
            [None] * 5,
            ["Code", "Qty", "Total", None, None],
            ["W1", 5, 50.0, None, None],
        ]
        blocks = seg(["ID", "Amount", "Date", "Tag", "Extra"], rows)
        self.assertEqual(len(blocks), 2, spans(blocks))
        self.assertEqual(blocks[1]["header"], ["Code", "Qty", "Total", "", ""])


class InteriorHeaderTest(unittest.TestCase):
    """Stacked tables with NO blank row between them — the harder case."""

    def test_two_tables_jammed_together_split_at_the_interior_header(self):
        rows = [
            [1, 100.0, dt.date(2022, 1, 1), "a", "b", "c"],
            [2, 200.0, dt.date(2022, 2, 1), "a", "b", "c"],
            [3, 300.0, dt.date(2022, 3, 1), "a", "b", "c"],
            ["Code", "Qty", "Total", None, None, None],    # no blank row above it
            ["W1", 5, 50.0, None, None, None],
            ["W2", 6, 60.0, None, None, None],
        ]
        blocks = seg(["ID", "Amt", "Date", "X", "Y", "Z"], rows)
        self.assertEqual(len(blocks), 2, f"jammed tables were read as one: {spans(blocks)}")
        self.assertEqual(spans(blocks), [(0, 2, None), (3, 5, 3)])
        assert_tiles(self, blocks, len(rows))

    def test_the_candidate_is_judged_against_the_rows_above_it_only(self):
        """The reason the reference window is local, stated as a test.

        Measured over the whole run, the candidate header's own text cells join the sample and drag the
        very columns they should contradict toward "mixed". In a short run one text cell among five
        numbers is enough. The column then reads as untyped, there is nothing to contradict, and the
        split never happens. Over the rows above the candidate, those columns are cleanly typed.
        """
        rows = [
            [1, 100.0, dt.date(2022, 1, 1), "a", "b", "c"],
            [2, 200.0, dt.date(2022, 2, 1), "a", "b", "c"],
            [3, 300.0, dt.date(2022, 3, 1), "a", "b", "c"],
            ["Code", "Qty", "Total", None, None, None],
            ["W1", 5, 50.0, None, None, None],
            ["W2", 6, 60.0, None, None, None],
        ]
        width = 6
        whole = bk._kinds_for(rows, width, 0, 5)
        above = bk._kinds_for(rows, width, 0, 2)
        self.assertNotIn(whole.get(1), bk._TYPED,
                         "whole-run kinds should be polluted — that is the flaw being guarded")
        self.assertIn(above.get(1), bk._TYPED, "the rows above the candidate are cleanly typed")
        self.assertFalse(bk._contradicts_kinds(rows[3], width, whole))
        self.assertTrue(bk._contradicts_kinds(rows[3], width, above))

    def test_a_rectangular_run_is_never_scanned_inside(self):
        """The cost bound and a correctness guard at once.

        A run whose columns are all-or-nothing is already one table. Looking inside it could only
        produce a false split, so the scan is skipped — which also keeps the expense off clean ledgers.
        """
        rows = [[i, float(i), dt.date(2022, 1, 1)] for i in range(1, 13)]
        rows[6] = ["Code", "Qty", "Total"]     # a header-shaped row inside a rectangular run
        blocks = seg(["ID", "Amt", "Date"], rows)
        self.assertEqual(len(blocks), 1, f"a rectangular run was split: {spans(blocks)}")

    def test_three_stacked_tables_all_split(self):
        """Each candidate is judged against the table it would terminate, not the first one."""
        rows = [
            [1, 100.0, dt.date(2022, 1, 1), "a", "b"],
            [2, 200.0, dt.date(2022, 2, 1), "a", "b"],
            ["Code", "Qty", "Total", None, None],
            ["W1", 5, 50.0, None, None],
            ["W2", 6, 60.0, None, None],
            ["Region", "Area", "POB", None, None],
            ["GOM", "West", 12, None, None],
        ]
        blocks = seg(["ID", "Amt", "Date", "X", "Y"], rows)
        self.assertEqual(len(blocks), 3, f"expected three tables: {spans(blocks)}")
        self.assertEqual([b["header_row"] for b in blocks], [None, 2, 5])
        assert_tiles(self, blocks, len(rows))


class RectangularityTest(unittest.TestCase):
    """Reported so the model knows when to read a block positionally instead of as a table."""

    def test_a_clean_table_is_rectangular(self):
        rows = [[i, float(i), "x"] for i in range(1, 11)]
        blocks = seg(["A", "B", "C"], rows)
        self.assertTrue(blocks[0]["rectangular"])
        self.assertEqual(blocks[0]["rectangularity"], 1.0)

    def test_a_ragged_block_is_flagged(self):
        """Columns populated about half the time are what two tables in one range produce."""
        rows = []
        for i in range(12):
            if i % 2:
                rows.append([i, float(i), None, None, None, None])
            else:
                rows.append([i, None, "x", "y", "z", "w"])
        blocks = seg(["A", "B", "C", "D", "E", "F"], rows)
        ragged = [b for b in blocks if not b["rectangular"]]
        self.assertTrue(ragged, f"a ragged block was reported as clean: "
                                f"{[b['rectangularity'] for b in blocks]}")


class BoundsTest(unittest.TestCase):
    """Past the size limits a sheet is a ledger by construction. Return it whole."""

    def test_a_huge_sheet_is_not_segmented(self):
        rows = [[i, float(i)] for i in range(bk._MAX_SEGMENT_ROWS + 1)]
        rows[5] = ["Code", "Qty"]
        blocks = seg(["A", "B"], rows)
        self.assertEqual(len(blocks), 1)

    def test_a_sheet_with_too_many_runs_is_not_segmented(self):
        """Dozens of separated runs is a sparse ledger, not a stack of tables."""
        rows = []
        for i in range(bk._MAX_RUNS + 2):
            rows.append([i, float(i), dt.date(2022, 1, 1)])
            rows.append([None, None, None])
        blocks = seg(["A", "B", "C"], rows)
        self.assertEqual(len(blocks), 1)


class PrimaryAndRowsTest(unittest.TestCase):
    """Which block the column profile describes, and which rows it is computed over."""

    def test_the_primary_block_is_the_one_with_the_most_rows(self):
        rows = [
            [1, 100.0, dt.date(2022, 1, 1)],
            [None] * 3,
            ["Code", "Qty", "Total"],
            ["W1", 5, 50.0],
            ["W2", 6, 60.0],
            ["W3", 7, 70.0],
        ]
        blocks = seg(["ID", "Amt", "Date"], rows)
        self.assertEqual(len(blocks), 2, spans(blocks))
        self.assertIs(bk.primary(blocks), blocks[1])

    def test_the_earliest_block_wins_a_tie(self):
        rows = [
            [1, 100.0, dt.date(2022, 1, 1)],
            [2, 200.0, dt.date(2022, 2, 1)],
            [None] * 3,
            ["Code", "Qty", "Total"],
            ["W1", 5, 50.0],
            ["W2", 6, 60.0],
        ]
        blocks = seg(["ID", "Amt", "Date"], rows)
        self.assertEqual(len(blocks), 2, spans(blocks))
        self.assertEqual(blocks[0]["n_rows"], blocks[1]["n_rows"])
        self.assertIs(bk.primary(blocks), blocks[0])

    def test_primary_of_nothing_is_none(self):
        self.assertIsNone(bk.primary([]))

    def test_block_rows_excludes_the_blocks_own_header_row(self):
        """The profile must be computed over data, not over the labels naming it."""
        rows = [
            [1, 100.0, dt.date(2022, 1, 1)],
            [None] * 3,
            ["Code", "Qty", "Total"],
            ["W1", 5, 50.0],
            ["W2", 6, 60.0],
        ]
        blocks = seg(["ID", "Amt", "Date"], rows)
        self.assertEqual(len(blocks), 2, spans(blocks))
        got = bk.block_rows(blocks[1], rows)
        self.assertEqual(got, [["W1", 5, 50.0], ["W2", 6, 60.0]])

    def test_block_rows_of_a_single_block_is_every_row(self):
        rows = [[i, float(i)] for i in range(6)]
        blocks = seg(["A", "B"], rows)
        self.assertEqual(bk.block_rows(blocks[0], rows), rows)


class RenderTest(unittest.TestCase):
    """What the prompt is told. Silence for one table; an explicit layout for several."""

    def test_one_table_renders_nothing(self):
        rows = [[i, float(i)] for i in range(6)]
        blocks = seg(["A", "B"], rows)
        self.assertEqual(bk.render_blocks(blocks, bk.primary(blocks)), "")

    def test_the_layout_names_each_table_and_marks_the_profiled_one(self):
        rows = [
            [1, 100.0, dt.date(2022, 1, 1)],
            [2, 200.0, dt.date(2022, 2, 1)],
            [3, 300.0, dt.date(2022, 3, 1)],
            [None] * 3,
            ["Code", "Qty", "Total"],
            ["W1", 5, 50.0],
        ]
        blocks = seg(["ID", "Amt", "Date"], rows)
        self.assertEqual(len(blocks), 2, spans(blocks))
        out = bk.render_blocks(blocks, bk.primary(blocks))
        self.assertIn("2 separate tables", out)
        self.assertIn("never be read under another's column name", out)
        self.assertIn("COLUMN PROFILE below describes THIS table only", out)
        self.assertIn("header on row 5", out, "1-based row numbers, matching the rest of the prompt")
        self.assertIn("Code", out)
        self.assertIn("1 row", out, "one row must not be described as '1 rows'")


class _FakeSheet:
    """The slice of openpyxl's read-only API that `extract_sheet` consumes.

    Same stub the populated-width tests drive, and the same reason: it is exactly the contract the
    pyxlsb and xlrd adapters implement, so one stub covers all three readers.
    """

    def __init__(self, rows, title="Sheet1"):
        self._rows = rows
        self.title = title
        self.max_row = len(rows)
        self.max_column = max((len(r) for r in rows), default=0)

    def iter_rows(self, values_only=True):
        for r in self._rows:
            yield tuple(r)


class WiringTest(unittest.TestCase):
    """`extract_sheet` profiles the largest table and says which one that was."""

    def test_a_single_table_tab_is_profiled_exactly_as_before(self):
        """The no-change guarantee. Nearly every tab takes this path, so it must not move."""
        from extractor import excel_bounded as eb
        from extractor.column_profile import profile_columns
        rows = [["Period", "Amount", "Date"],
                ["003.2022", 542.0, dt.date(2022, 3, 1)],
                ["012.2022", 271.0, dt.date(2022, 12, 31)]]
        sh = eb.extract_sheet(_FakeSheet(rows))
        self.assertEqual(len(sh["blocks"]), 1)
        expected, exp_totals = profile_columns(sh["header"], rows[1:], sh["populated_cols"])
        self.assertEqual(sh["column_profile"], expected, "single-table profiling changed")
        self.assertEqual(sh["total_rows"], exp_totals)

    def test_a_stacked_tab_is_profiled_over_the_larger_table_only(self):
        """Profiling both at once is what put a column NAME in the trip-id column and made every
        column read as "mixed". The narrower question gives a clean answer."""
        from extractor import excel_bounded as eb
        rows = [["Period", "Amount", "Date"],
                ["003.2022", 542.0, dt.date(2022, 3, 1)],
                ["004.2022", 271.0, dt.date(2022, 4, 1)],
                ["012.2022", 100.0, dt.date(2022, 12, 31)],
                [None, None, None],
                ["Region", "POB", "Type"],
                ["GOM", 2688, "DRILL SHIP"]]
        sh = eb.extract_sheet(_FakeSheet(rows))
        self.assertEqual(len(sh["blocks"]), 2, spans(sh["blocks"]))
        kinds = {p["header"]: p["kind"] for p in sh["column_profile"]}
        self.assertEqual(kinds["Period"], "period", f"profiled across both tables: {kinds}")
        self.assertEqual(kinds["Date"], "date", f"profiled across both tables: {kinds}")
        self.assertEqual(kinds["Amount"], "number", f"profiled across both tables: {kinds}")

    def test_a_totals_row_index_is_reported_in_the_sheets_own_numbering(self):
        """The index comes out relative to the block it was found in. The prompt prints it as a row
        number and the synthesis cites it, so it has to be shifted back to the sheet's numbering."""
        from extractor import excel_bounded as eb
        rows = [["Region", "POB", "Type"],      # the small table comes FIRST
                ["GOM", 2688, "DRILL SHIP"],
                [None, None, None],
                ["Item", "Amount", "Currency"],  # the larger table's own header
                ["Freight", 100.0, "USD"],
                ["Fuel", 200.0, "USD"],
                ["Port", 300.0, "USD"],
                ["Total", 600.0, "USD"]]
        sh = eb.extract_sheet(_FakeSheet(rows))
        self.assertEqual(len(sh["blocks"]), 2, spans(sh["blocks"]))
        main = bk.primary(sh["blocks"])
        self.assertEqual(main["header_row"], 2, "the larger table has its own header row")
        self.assertEqual([t["index"] for t in sh["total_rows"]], [6],
                         "the totals row is data row 6 of the SHEET, not row 3 of the block")
        # And the sheet's own rows confirm what is at that index.
        self.assertEqual(sh["rows"][6][0], "Total")

    def test_the_blocks_ride_through_to_the_skill_payload(self):
        """A chunk sees a window of rows and cannot tell which table they came from on its own."""
        from extractor import excel_bounded as eb
        rows = [["Period", "Amount", "Date"],
                ["003.2022", 542.0, dt.date(2022, 3, 1)],
                ["004.2022", 271.0, dt.date(2022, 4, 1)],
                [None, None, None],
                ["Region", "POB", "Type"],
                ["GOM", 2688, "DRILL SHIP"]]
        book = {"sheets": [eb.extract_sheet(_FakeSheet(rows))]}
        payload = eb.sheets_to_json(book)
        self.assertEqual(len(payload[0]["blocks"]), 2)


class RealExhibitTest(unittest.TestCase):
    """The two real tabs. Sheet1 must stay whole; Oyler must come apart. Skips if absent."""

    SRC = ("/Users/harshakolachina/Documents/Projects/Shell JV Audit/Data Files/"
           "Exception Reports and Rebuttals by year/2023/Great White/Rebuttal 0/"
           "Audit Report/0003/IR 24 Affiliate Coding Error.xlsx")

    def setUp(self):
        if not os.path.exists(self.SRC):
            self.skipTest("source exhibit not available on this machine")
        from extractor import excel_bounded as eb
        self.sheets = {s["name"]: s for s in eb.extract_excel_full(self.SRC)["sheets"]}

    def test_sheet1_is_one_table_so_its_totals_row_is_not_stranded(self):
        sh = self.sheets["Sheet1"]
        blocks = sh["blocks"]
        self.assertEqual(len(blocks), 1, f"Sheet1 was split: {spans(blocks)}")
        self.assertTrue(blocks[0]["rectangular"])
        self.assertEqual(blocks[0]["span"], [0, 71])

    def test_oyler_comes_apart_into_the_trip_table_and_the_personnel_table(self):
        """The tab that produced `TRIP ID = [41341, 41880, 'Current POB']` when read as one table."""
        sh = self.sheets["Oyler"]
        blocks = sh["blocks"]
        self.assertEqual(len(blocks), 2, f"Oyler stayed one table: {spans(blocks)}")
        trips, people = blocks
        self.assertIn("TRIP ID", trips["header"])
        self.assertEqual(trips["n_rows"], 3)
        self.assertIsNotNone(people["header_row"])
        self.assertIn("Current POB", people["header"],
                      "'Current POB' is a column NAME in the second table, not a trip id")
        self.assertEqual(people["n_rows"], 1)

    def test_every_row_of_both_tabs_belongs_to_exactly_one_block(self):
        for name in ("Sheet1", "Oyler"):
            sh = self.sheets[name]
            with self.subTest(sheet=name):
                assert_tiles(self, sh["blocks"], len(sh["rows"]))


if __name__ == "__main__":
    unittest.main()
