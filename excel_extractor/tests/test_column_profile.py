"""Tests for `column_profile` — the module that computes ranges and totals instead of inferring them.

The bug this guards against: on GreatWhite_2023_0003 the synthesis read a bounded sample of Sheet1 and
reported the charge window as "March 2022 - August 2022" when the charges run to December, and a total
of $27,499.67 against $47,365.25. A sample cannot answer either question, so these tests are mostly
about the places where a sample or a shortcut would give a confident wrong answer:

  - a period (`003.2022`) must not be read as a March date, since both use dots
  - a date range must come from parsed dates, never text order ("31.03.2022" sorts before "30.09.2021")
  - the extremes must come from every row, including rows no sample would visit
  - a trailing totals row must be excluded from sums, or every figure doubles

Everything runs offline on synthetic rows. The one real-file test skips when the exhibit is absent.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extractor import column_profile as cp  # noqa: E402


def profile(header, rows, width=None):
    """Profile some rows and return `({header_or_c<n> -> profile}, total_rows)`."""
    width = width if width is not None else max(len(header), max((len(r) for r in rows), default=0))
    profs, totals = cp.profile_columns(header, rows, width)
    return {p["header"] or f"c{p['col']}": p for p in profs}, totals


class ClassificationTest(unittest.TestCase):
    """A column's kind comes from the shape of its values, never from its header."""

    def test_fiscal_period_is_not_read_as_a_date(self):
        """`003.2022` is period 3 of 2022. Read as DD.MM it would become 3 January 2022."""
        by, _ = profile(["Fiscal year/Period"], [["003.2022"], ["004.2022"], ["012.2022"]])
        p = by["Fiscal year/Period"]
        self.assertEqual(p["kind"], "period")
        self.assertEqual((p["min"], p["max"]), ("003.2022", "012.2022"))

    def test_special_periods_up_to_016_are_periods(self):
        by, _ = profile(["P"], [["013.2022"], ["016.2022"]])
        self.assertEqual(by["P"]["kind"], "period")

    def test_a_three_part_dotted_value_is_a_date_not_a_period(self):
        by, _ = profile(["Document Date"], [["31.03.2022"], ["30.09.2022"]])
        self.assertEqual(by["Document Date"]["kind"], "date")

    def test_period_and_date_are_distinguished_with_no_header_help(self):
        """The same run of digits, one column each, told apart on part count alone."""
        by, _ = profile(["", ""], [["003.2022", "03.03.2022"], ["004.2022", "04.04.2022"]])
        self.assertEqual(by["c1"]["kind"], "period")
        self.assertEqual(by["c2"]["kind"], "date")

    def test_native_datetimes_are_dates(self):
        by, _ = profile(["D"], [[dt.datetime(2022, 3, 31, 9, 0)], [dt.date(2022, 12, 31)]])
        p = by["D"]
        self.assertEqual(p["kind"], "date")
        self.assertEqual((p["min"], p["max"]), ("2022-03-31", "2022-12-31"))

    def test_iso_and_slash_dates(self):
        by, _ = profile(["A", "B"], [["2022-03-31", "3/31/2022"], ["2022-12-31", "12/31/2022"]])
        self.assertEqual(by["A"]["kind"], "date")
        self.assertEqual(by["B"]["max"], "2022-12-31", "slash dates read MM/DD/YYYY")

    def test_impossible_primary_reading_falls_back_to_the_other_order(self):
        """`03.31.2022` has no 31st month, so it is read as March 31 rather than dropped."""
        by, _ = profile(["D"], [["03.31.2022"], ["03.15.2022"]])
        self.assertEqual(by["D"]["kind"], "date")
        self.assertEqual(by["D"]["max"], "2022-03-31")

    def test_numbers_arriving_as_formatted_text_are_numbers(self):
        by, _ = profile(["Amt"], [["1,224.00"], ["$ 6,504"], ["(271.00)"]])
        p = by["Amt"]
        self.assertEqual(p["kind"], "number")
        self.assertAlmostEqual(p["sum"], 1224.00 + 6504 - 271, places=2)
        self.assertAlmostEqual(p["min"], -271.0, places=2, msg="parentheses mean a credit")
        self.assertIn("formatted text", p.get("note", ""))

    def test_booleans_are_not_numbers(self):
        by, _ = profile(["Flag"], [[True], [False], [True]])
        self.assertNotEqual(by["Flag"]["kind"], "number")

    def test_a_bare_year_in_a_date_column_does_not_make_it_numeric(self):
        """Number is tried last precisely so `2022` cannot win a column of dates.

        `_parse_number` accepts "2022" quite happily, so if number were tried first every date column
        containing a stray year would be counted partly numeric.
        """
        rows = [[f"{d:02d}.03.2022"] for d in range(1, 10)] + [["2022"]]
        by, _ = profile(["D"], rows)
        self.assertEqual(by["D"]["kind"], "date")
        self.assertEqual(by["D"]["max"], "2022-03-09", "the stray year is not read as a date")

    def test_a_bare_year_on_its_own_is_a_number(self):
        by, _ = profile(["Y"], [["2022"], ["2023"]])
        self.assertEqual(by["Y"]["kind"], "number", "only the dated shapes are dates")

    def test_one_junk_value_does_not_change_the_kind(self):
        """9 dates and 1 label is 90% agreement, which still classifies as a date column."""
        rows = [[f"{d:02d}.03.2022"] for d in range(1, 10)] + [["n/a"]]
        by, _ = profile(["D"], rows)
        self.assertEqual(by["D"]["kind"], "date")
        self.assertEqual(by["D"]["count"], 9, "the junk value is not counted as a date")

    def test_a_genuinely_mixed_column_is_labelled_mixed(self):
        rows = [["003.2022"], ["31.03.2022"], ["hello"], [1224], ["world"]]
        by, _ = profile(["X"], rows)
        self.assertEqual(by["X"]["kind"], "mixed")

    def test_a_single_value_is_not_enough_to_claim_a_kind(self):
        by, _ = profile(["X"], [["003.2022"], [None], [None]])
        self.assertEqual(by["X"]["kind"], "text", "one value cannot establish a column's shape")

    def test_a_wholly_empty_column_is_omitted(self):
        by, _ = profile(["A", "Empty"], [[1224, None], [271, None]])
        self.assertIn("A", by)
        self.assertNotIn("Empty", by)


class RangeTest(unittest.TestCase):
    """Ranges must come from parsed values over every row."""

    def test_date_range_is_not_lexical(self):
        """As text, '31.03.2022' < '30.09.2021'. Sorting the strings inverts this range."""
        by, _ = profile(["D"], [["30.09.2021"], ["31.12.2022"], ["31.03.2022"]])
        p = by["D"]
        self.assertEqual((p["min"], p["max"]), ("2021-09-30", "2022-12-31"))

    def test_period_range_and_full_enumeration(self):
        """The 0003 shape: nine periods from 003 to 012, with 011 genuinely absent."""
        months = [3, 4, 5, 6, 7, 8, 9, 10, 12]
        rows = [[f"{m:03d}.2022"] for m in months for _ in range(3)]
        by, _ = profile(["Fiscal year/Period"], rows)
        p = by["Fiscal year/Period"]
        self.assertEqual((p["min"], p["max"]), ("003.2022", "012.2022"))
        self.assertEqual(p["distinct"], 9)
        self.assertEqual(p["values"], [f"{m:03d}.2022" for m in months])
        self.assertNotIn("011.2022", p["values"], "an absent period must not be invented")

    def test_periods_across_a_year_boundary_order_by_year_first(self):
        by, _ = profile(["P"], [["012.2021"], ["003.2022"], ["001.2022"]])
        p = by["P"]
        self.assertEqual((p["min"], p["max"]), ("012.2021", "003.2022"))

    def test_the_extreme_in_the_very_last_row_is_still_found(self):
        """Kinds come from a strided sample, but statistics must cover every row.

        5,000 rows means the stride skips the tail, so the sample never sees row 4,999. If the
        statistics were sampled too, this range would stop in January — the original bug exactly.
        """
        rows = [["01.01.2022", "100"] for _ in range(4_999)] + [["31.12.2022", "999"]]
        self.assertNotIn(4_999, cp._sample_indices(5_000, cp._KIND_SAMPLE_ROWS))
        by, _ = profile(["D", "Amt"], rows)
        self.assertEqual(by["D"]["max"], "2022-12-31")
        self.assertEqual(by["D"]["count"], 5_000)
        self.assertEqual(by["Amt"]["max"], 999.0)
        self.assertAlmostEqual(by["Amt"]["sum"], 4_999 * 100 + 999, places=2)
        self.assertEqual(by["D"]["basis"], "all rows")

    def test_a_kind_that_only_appears_late_is_still_classified(self):
        """The sample strides across the whole sheet, so it is not blind to the second half."""
        rows = [["01.01.2022"] for _ in range(2_000)] + [["31.12.2022"] for _ in range(2_000)]
        by, _ = profile(["D"], rows)
        self.assertEqual(by["D"]["kind"], "date")
        self.assertEqual((by["D"]["min"], by["D"]["max"]), ("2022-01-01", "2022-12-31"))

    def test_sample_indices_span_the_sheet(self):
        idx = cp._sample_indices(100_000, 600)
        self.assertLessEqual(len(idx), 600)
        self.assertEqual(idx[0], 0)
        self.assertGreater(idx[-1], 99_000, "the sample must reach the end of the sheet")

    def test_text_column_statistics_say_they_are_sampled(self):
        rows = [[f"name-{j}"] for j in range(5_000)]
        by, _ = profile(["N"], rows)
        p = by["N"]
        self.assertEqual(p["kind"], "text")
        self.assertIn("sample", p["basis"], "a sampled figure must not be presented as a full count")
        self.assertLess(p["of_rows"], 5_000)


class DistinctValuesTest(unittest.TestCase):

    def test_few_distinct_values_are_listed_in_full(self):
        rows = [[f"E{i:02d}"] for i in range(11)] * 2
        by, _ = profile(["Equity group"], rows)
        p = by["Equity group"]
        self.assertEqual(p["distinct"], 11)
        self.assertEqual(len(p["values"]), 11)
        self.assertNotIn("examples", p)

    def test_many_distinct_values_give_a_count_and_spread_examples(self):
        rows = [[f"DOC{j:05d}"] for j in range(200)]
        by, _ = profile(["Doc"], rows)
        p = by["Doc"]
        self.assertEqual(p["distinct"], 200)
        self.assertNotIn("values", p, "200 values must not be dumped into the prompt")
        self.assertEqual(len(p["examples"]), 3)
        self.assertEqual(p["examples"][0], "DOC00000")
        self.assertEqual(p["examples"][-1], "DOC00199", "examples span the column, not just its head")

    def test_distinct_tracking_is_bounded(self):
        """A 200k-row column of unique amounts must not grow an unbounded tally."""
        rows = [[float(j) + 0.5] for j in range(cp._DISTINCT_CAP + 500)]
        by, _ = profile(["Amt"], rows)
        p = by["Amt"]
        self.assertTrue(p.get("distinct_capped"))
        self.assertLessEqual(p["distinct"], cp._DISTINCT_CAP + 1)
        self.assertAlmostEqual(p["max"], float(cp._DISTINCT_CAP + 499) + 0.5, places=2,
                               msg="capping the tally must not cap the range")


class TotalRowTest(unittest.TestCase):
    """A trailing summary row is ordinary in an exhibit, and summing through it doubles every figure."""

    def test_totals_row_is_detected_and_excluded(self):
        rows = [["003.2022", "A", 100.0], ["004.2022", "B", 200.0], ["005.2022", "C", 300.0],
                [None, None, 600.0]]
        by, totals = profile(["P", "Name", "Amt"], rows)
        self.assertEqual([t["index"] for t in totals], [3])
        p = by["Amt"]
        self.assertAlmostEqual(p["sum"], 600.0, places=2, msg="the totals row must not be summed")
        self.assertEqual(p["count"], 3)
        self.assertEqual(p["of_rows"], 3, "the totals row is not a data row")
        self.assertAlmostEqual(p["stated_total"], 600.0, places=2)
        self.assertTrue(p["reconciles"])

    def test_the_extremes_exclude_the_totals_row(self):
        """The total is normally the largest value in its column, so a naive max reports it."""
        rows = [[float(v)] for v in range(1, 11)] + [[55.0]]
        by, totals = profile(["Amt"], rows)
        self.assertEqual(len(totals), 1)
        self.assertAlmostEqual(by["Amt"]["max"], 10.0, places=2)
        self.assertAlmostEqual(by["Amt"]["min"], 1.0, places=2)

    def test_a_stated_total_that_disagrees_is_still_recognised_as_a_total(self):
        """A trailing row 6.7% off the rows above it is a WRONG total, not a data row.

        Declining to recognise it would be the worse mistake: its 640 would be summed in and the
        column reported as 1,240 instead of 600. So it is excluded from the sum, and the gap is
        reported rather than smoothed over.
        """
        rows = [["A", 100.0], ["B", 200.0], ["C", 300.0], [None, 640.0]]
        by, totals = profile(["Name", "Amt"], rows)
        self.assertEqual([t["index"] for t in totals], [3])
        self.assertEqual(totals[0]["matched_cols"], [], "nothing reconciled exactly")
        self.assertEqual(totals[0]["near_cols"], [1], "but it is the right order of magnitude")
        p = by["Amt"]
        self.assertAlmostEqual(p["sum"], 600.0, places=2, msg="the wrong total must not be summed in")
        self.assertFalse(p["reconciles"], "reporting must not call a 40.00 gap a match")
        self.assertAlmostEqual(p["total_diff"], 40.0, places=2)
        line = cp._profile_line(p)
        self.assertIn("does NOT match", line)
        self.assertIn("40", line)

    def test_a_non_reconciling_totals_row_is_called_out_in_the_render(self):
        rows = [["A", 100.0], ["B", 200.0], ["C", 300.0], [None, 640.0]]
        profs, totals = cp.profile_columns(["Name", "Amt"], rows, 2)
        out = cp.render_profile(profs, totals)
        self.assertIn("do NOT agree with the rows above", out)

    def test_a_trailing_figure_of_the_wrong_magnitude_is_not_a_total(self):
        """The near-match band is 20%, so a stray note is still kept as data."""
        rows = [["A", 100.0], ["B", 200.0], ["C", 300.0], [None, 7.0]]
        by, totals = profile(["Name", "Amt"], rows)
        self.assertEqual(totals, [])
        self.assertAlmostEqual(by["Amt"]["sum"], 607.0, places=2)

    def test_a_dense_trailing_row_is_not_mistaken_for_a_total(self):
        """Only a SPARSE trailing row is a candidate, so a full final data row stays data."""
        rows = [["003.2022", "A", 100.0], ["004.2022", "B", 200.0], ["005.2022", "C", 300.0],
                ["006.2022", "D", 600.0]]
        by, totals = profile(["P", "Name", "Amt"], rows)
        self.assertEqual(totals, [], "a fully populated row is data even if its value sums the others")
        self.assertAlmostEqual(by["Amt"]["sum"], 1200.0, places=2)

    def test_a_sparse_trailing_row_that_reconciles_nothing_is_kept(self):
        rows = [["A", 100.0], ["B", 200.0], [None, 7.0]]
        by, totals = profile(["Name", "Amt"], rows)
        self.assertEqual(totals, [], "sparse alone is not enough; the arithmetic must agree")
        self.assertAlmostEqual(by["Amt"]["sum"], 307.0, places=2)

    def test_two_matching_columns_give_high_confidence(self):
        rows = [["A", 100.0, 1.0], ["B", 200.0, 2.0], [None, 300.0, 3.0]]
        _by, totals = profile(["Name", "Amt", "Qty"], rows)
        self.assertEqual(totals[0]["confidence"], "high")
        self.assertEqual(sorted(totals[0]["matched_cols"]), [1, 2])

    def test_one_matching_column_is_only_probable(self):
        rows = [["A", 100.0, 1.0], ["B", 200.0, 2.0], [None, 300.0, 99.0]]
        _by, totals = profile(["Name", "Amt", "Qty"], rows)
        self.assertEqual(totals[0]["confidence"], "probable")
        self.assertEqual(totals[0]["matched_cols"], [1])

    def test_a_stacked_subtotal_and_total_both_resolve(self):
        """Detection runs newest-first so a confirmed total is not counted into the next candidate."""
        rows = [["A", 100.0], ["B", 200.0], ["C", 300.0], [None, 600.0], [None, 600.0]]
        by, totals = profile(["Name", "Amt"], rows)
        self.assertEqual([t["index"] for t in totals], [3, 4])
        self.assertAlmostEqual(by["Amt"]["sum"], 600.0, places=2)

    def test_totals_row_does_not_pollute_text_column_values(self):
        rows = [["A", 100.0], ["B", 200.0], ["Total", 300.0]]
        by, totals = profile(["Name", "Amt"], rows)
        self.assertEqual(len(totals), 1)
        self.assertNotIn("Total", by["Name"].get("values", []),
                         "a summary label is not one of the column's data values")

    def test_detection_reaches_beyond_the_last_row(self):
        """Blank rows can sit under the total, so the candidate window is not just row n-1."""
        rows = [["A", 100.0], ["B", 200.0], [None, 300.0], [None, None], [None, None]]
        by, totals = profile(["Name", "Amt"], rows)
        self.assertEqual([t["index"] for t in totals], [2])
        self.assertAlmostEqual(by["Amt"]["sum"], 300.0, places=2)

    def test_detect_total_rows_wrapper(self):
        rows = [["A", 100.0], ["B", 200.0], [None, 300.0]]
        self.assertEqual([t["index"] for t in cp.detect_total_rows(["Name", "Amt"], rows, 2)], [2])

    def test_totals_in_a_long_sheet_are_still_found(self):
        """The full pass and the candidate window must agree on a sheet big enough to be sampled."""
        rows = [[100.0] for _ in range(5_000)] + [[500_000.0]]
        by, totals = profile(["Amt"], rows)
        self.assertEqual([t["index"] for t in totals], [5_000])
        self.assertAlmostEqual(by["Amt"]["sum"], 500_000.0, places=2)

    def test_a_row_that_merely_repeats_the_row_above_is_not_a_total(self):
        """The sum of ONE row is that row, so equality there means repetition, not summation.

        This is the `Oyler` trip table. Three helicopter trips share an asset id, a call sign id and a
        trip id, so the second trip equalled the first on eight columns and was read as a
        high-confidence totals row — excluding a real trip from every figure the tab reports. Column
        agreement is not corroboration when the agreement comes from repeated identifiers.
        """
        rows = [[481, 621, 41341, 1], [481, 621, 41341, 2], [481, 182, 41880, 3]]
        _, totals = profile(["ASSET ID", "CALL SIGN ID", "TRIP ID", "LEG"], rows)
        self.assertEqual(totals, [], "a repeated identifier row was called a totals row")

    def test_a_labelled_total_after_a_single_row_is_still_believed(self):
        """The row-count guard covers unlabelled arithmetic only. A label is its own corroboration."""
        rows = [["Freight", 100.0], ["Total", 100.0]]
        _, totals = profile(["Item", "Amt"], rows)
        self.assertEqual([t["index"] for t in totals], [1])

    def test_a_two_row_table_where_the_second_row_doubles_the_first_is_not_a_total(self):
        """Even arithmetic that looks like summing needs more than one row beneath it to mean anything.

        Without a label, `100` then `200` is just as likely two charges as a charge and its total, and
        wrongly excluding the second halves everything the tab reports.
        """
        rows = [[100.0, 5.0], [200.0, 10.0]]
        _, totals = profile(["Amt", "Qty"], rows)
        self.assertEqual(totals, [])


class EmptyInputTest(unittest.TestCase):

    def test_no_rows(self):
        self.assertEqual(cp.profile_columns(["A"], [], 1), ([], []))

    def test_zero_width(self):
        self.assertEqual(cp.profile_columns([], [[1]], 0), ([], []))

    def test_rows_shorter_than_the_width_do_not_crash(self):
        by, _ = profile(["A", "B", "C"], [[1], [1, 2], [1, 2, 3]], width=3)
        self.assertIn("A", by)

    def test_render_of_nothing_is_empty(self):
        self.assertEqual(cp.render_profile([], []), "")


class NativeValueTest(unittest.TestCase):
    """The extractor profiles NATIVE cell values, so these are the inputs that actually arrive.

    A real date cell reaches us as a `datetime` and a money cell as a `float`. Profiling the display
    strings instead looks equivalent and is not, which cost real debugging time on the exhibit: see
    `test_fractional_cents_are_not_rounded_away`.
    """

    def test_fractional_cents_are_not_rounded_away(self):
        """Sub-cent values must be summed at full precision, then rounded once at the end.

        The exhibit computes `Cost + OH` as amount x 1.025, which lands on half-cents (277.775). Its
        DISPLAY strings are rounded to two decimals, so adding those up drifts a cent every few rows
        and the sheet's own total starts to look wrong. Here the 31 native rows plus a stated total
        reconcile exactly, and the same figures as text do not — a rounding artifact would surface
        downstream as a coding error in the exhibit, which is the expensive kind of false finding.
        """
        native = [[271 * 1.025] for _ in range(31)]        # 277.775, a half cent, as in the exhibit
        self.assertEqual(round(native[0][0], 3), 277.775)
        # The sheet's total cell is a formula over those same cells, so it carries the same precision.
        stated = sum(r[0] for r in native)
        by, totals = profile(["Cost + OH"], native + [[stated]])
        self.assertEqual([t["index"] for t in totals], [31])
        self.assertTrue(by["Cost + OH"]["reconciles"],
                        "native floats must agree with the sheet's stated total")

        # The same column as the two-decimal strings Excel DISPLAYS. Every row loses half a cent, so
        # by row 31 the column disagrees with the sheet's own total by about sixteen cents.
        as_text = [[f"{r[0]:.2f}"] for r in native] + [[f"{stated:.2f}"]]
        by_text, _ = profile(["Cost + OH"], as_text)
        self.assertFalse(by_text["Cost + OH"]["reconciles"],
                         "display strings drift; this is why we profile native values")
        self.assertGreater(abs(by_text["Cost + OH"]["total_diff"]), 0.10)

    def test_a_native_date_range_spans_every_row(self):
        rows = [[dt.datetime(2022, m, 15), 100.0] for m in range(1, 13)]
        by, _ = profile(["Posting date", "Amt"], rows)
        self.assertEqual((by["Posting date"]["min"], by["Posting date"]["max"]),
                         ("2022-01-15", "2022-12-15"))
        self.assertEqual(by["Amt"]["sum"], 1_200.0)

    def test_native_ints_and_floats_mix_in_one_column(self):
        """openpyxl returns an int for a whole number and a float otherwise, in the same column."""
        by, _ = profile(["Amt"], [[542], [271.5], [2439], [6504.25]])
        p = by["Amt"]
        self.assertEqual(p["kind"], "number")
        self.assertAlmostEqual(p["sum"], 9_756.75, places=2)
        self.assertAlmostEqual(p["min"], 271.5, places=2)

    def test_a_native_bool_is_not_counted_as_a_number(self):
        """Excel checkbox columns arrive as bools. Summing them invents a meaningless figure."""
        by, _ = profile(["Flag"], [[True], [False], [True], [False]])
        self.assertNotEqual(by["Flag"]["kind"], "number")


class RenderTest(unittest.TestCase):

    def test_the_header_states_that_the_figures_are_computed(self):
        profs, totals = cp.profile_columns(["P"], [["003.2022"], ["012.2022"]], 1)
        out = cp.render_profile(profs, totals)
        self.assertIn("computed in code over ALL 2 data rows", out)
        self.assertIn("never inferred from the sampled rows", out)

    def test_a_totals_row_is_announced_in_1_based_terms(self):
        rows = [["A", 100.0], ["B", 200.0], [None, 300.0]]
        profs, totals = cp.profile_columns(["Name", "Amt"], rows, 2)
        out = cp.render_profile(profs, totals)
        self.assertIn("Data row 3 is a TOTALS row", out)
        self.assertIn("EXCLUDED from every sum", out)

    def test_period_notation_is_explained(self):
        profs, totals = cp.profile_columns(["P"], [["003.2022"], ["012.2022"]], 1)
        self.assertIn("PPP.YYYY", cp.render_profile(profs, totals))

    def test_typed_columns_are_never_withheld_by_the_budget(self):
        header = ["P"] + [f"t{i}" for i in range(60)]
        rows = [["003.2022"] + [f"v{i}-{j}" for i in range(60)] for j in range(30)]
        profs, totals = cp.profile_columns(header, rows, len(header))
        out = cp.render_profile(profs, totals, max_chars=200)
        self.assertIn("003.2022", out, "the period column must survive any budget")
        self.assertIn("not profiled here", out, "withheld columns must be disclosed, not silent")

    def test_reconciling_columns_are_ordered_before_other_numeric_columns(self):
        header = ["DocNum", "Amt"]
        rows = [[130532608, 100.0], [130532609, 200.0], [None, 300.0]]
        profs, totals = cp.profile_columns(header, rows, 2)
        out = cp.render_profile(profs, totals)
        self.assertLess(out.index("'Amt'"), out.index("'DocNum'"),
                        "the column the exhibit totals is the reconciliation; it leads")

    def test_a_mostly_mixed_tab_is_flagged_as_stacked_tables(self):
        """Two tables one above the other produce per-column ranges that span both. Say so."""
        rows = [[f"hdr{i}" for i in range(5)]] + [[i * 10 + j for i in range(5)] for j in range(4)] \
               + [[f"other{i}" for i in range(5)]] + [[f"x{i}-{j}" for i in range(5)] for j in range(4)]
        profs, totals = cp.profile_columns([""] * 5, rows, 5)
        out = cp.render_profile(profs, totals)
        self.assertIn("SEVERAL stacked tables", out)
        self.assertIn("read the rows directly", out)

    def test_a_clean_tab_carries_no_stacked_table_warning(self):
        rows = [["003.2022", "A", 100.0], ["004.2022", "B", 200.0], ["005.2022", "C", 300.0]]
        profs, totals = cp.profile_columns(["P", "N", "Amt"], rows, 3)
        self.assertNotIn("stacked tables", cp.render_profile(profs, totals))

    def test_money_keeps_both_decimal_places(self):
        profs, totals = cp.profile_columns(["Amt"], [[6666.60], [277.77]], 1)
        out = cp.render_profile(profs, totals)
        self.assertIn("6,666.60", out, "trailing zeros in money must not be stripped")
        self.assertNotIn("6,666.6 ", out)

    def test_parsed_values_are_rendered_readably_not_as_tuples(self):
        profs, totals = cp.profile_columns(["X"], [["003.2022"], ["31.03.2022"], ["a"], [1], ["b"]], 1)
        out = cp.render_profile(profs, totals)
        self.assertNotIn("(", out.split("distinct")[-1], "raw parse payloads must not reach the prompt")


class RealExhibitTest(unittest.TestCase):
    """The five Sheet1 facts the synthesis got wrong. Skipped when the exhibit is absent.

    These read the profile the extractor itself produced (`sh["column_profile"]`) rather than
    re-profiling `sh["rows"]`. Two reasons. It is the artifact the prompts actually receive, so the
    wiring is covered as well as the arithmetic. And `rows` holds DISPLAY strings, which give a
    different answer here: `Cost + OH` is computed at half-cent precision (277.775), so each string
    is rounded to two decimals and the roundings accumulate to an 8-cent drift over 31 rows. Summing
    the strings yields 47,365.17 and makes the sheet's own total look 8 cents wrong. Summing the
    native floats yields 47,365.25, which is what the sheet says. The gap was ours, not the sheet's.
    """

    SRC = ("/Users/harshakolachina/Documents/Projects/Shell JV Audit/Data Files/"
           "Exception Reports and Rebuttals by year/2023/Great White/Rebuttal 0/"
           "Audit Report/0003/IR 24 Affiliate Coding Error.xlsx")

    def setUp(self):
        if not os.path.exists(self.SRC):
            self.skipTest("source exhibit not available on this machine")
        from extractor import excel_bounded as eb
        book = eb.extract_excel_full(self.SRC)
        self.book = book
        sh = next(s for s in book["sheets"] if s["name"] == "Sheet1")
        self.sheet = sh
        self.by = {p["header"] or f"c{p['col']}": p for p in sh["column_profile"]}
        self.totals = sh["total_rows"]

    def test_the_period_range_runs_to_december(self):
        """The synthesis said March to August. The exhibit runs March to December."""
        p = self.by["Fiscal year/Period"]
        self.assertEqual((p["min"], p["max"]), ("003.2022", "012.2022"))
        self.assertEqual(p["distinct"], 9)
        self.assertIn("012.2022", p["values"])

    def test_the_date_columns_are_dates_not_periods(self):
        for name in ("Document Date", "Entry Date SL", "Posting date"):
            self.assertEqual(self.by[name]["kind"], "date", name)
        self.assertEqual(self.by["Document Date"]["max"], "2022-12-31")

    def test_the_charge_total_is_the_exception_amount(self):
        """The synthesis said $27,499.67 by adding up a truncated sample.

        The 31 charge rows come to the exception amount and agree with the sheet's own total to the
        cent. Asserting `reconciles` here is the point: a spurious discrepancy on this column would
        be read downstream as a coding error in the exhibit.
        """
        p = self.by["Cost + OH"]
        self.assertAlmostEqual(p["sum"], 47_365.25, places=2)
        self.assertAlmostEqual(p["stated_total"], 47_365.25, places=2)
        self.assertTrue(p["reconciles"])

    def test_the_local_currency_column_reconciles_exactly(self):
        p = self.by["Actual Amout in Local Currency"]
        self.assertAlmostEqual(p["sum"], 46_210.0, places=2)
        self.assertTrue(p["reconciles"])

    def test_the_totals_row_is_found_and_excluded(self):
        self.assertEqual([t["index"] for t in self.totals], [32])
        self.assertEqual(self.totals[0]["confidence"], "high")
        self.assertEqual(self.by["Cost + OH"]["of_rows"], 32)

    def test_the_profile_survives_the_hand_off_to_the_prompts(self):
        """`sheets_to_json` is what the interpreter and the chunk scan read, not the sheet dict."""
        from extractor import excel_bounded as eb
        tab = next(t for t in eb.sheets_to_json(self.book) if t["tab"] == "Sheet1")
        self.assertEqual(len(tab["column_profile"]), len(self.sheet["column_profile"]))
        self.assertEqual([t["index"] for t in tab["total_rows"]], [32])

    def test_the_render_states_the_facts_the_synthesis_got_wrong(self):
        """The tightest budget any prompt uses must still carry the period range and the total.

        The chunk prompt repeats the profile for every chunk, so its budget is the smallest one. A
        cap that dropped a typed column would put us back where we started, with the model reading
        ranges off a partial set of rows.
        """
        txt = cp.render_profile(self.sheet["column_profile"], self.totals,
                                max_chars=600)
        self.assertIn("012.2022", txt)          # the period range ran to December, not August
        self.assertIn("2022-12-31", txt)        # and so did the document dates
        self.assertIn("47,365.25", txt)         # the real total, not the sampled $27,499.67
        self.assertIn("TOTALS row", txt)        # and the row that would otherwise double it


if __name__ == "__main__":
    unittest.main()
