"""Tests for `report_lists_exception` — the guard that keeps a cycle off a report that omits it.

GreatWhite(BPOnly)_2022_0010 is the case behind it. Cycles 4 and 5 are late letters covering only the
three exceptions still open that round; exception 10 closed earlier, so it is absent. With no exhibit
folder either, the description fallback matched `10` inside `BP Audit No.: 2022-101` in the letterhead
and returned exception 001's SCM Fee paragraph as exception 0010's — plus its $62,929.08 as the
questioned amount. Corpus-wide, 31 cycle/exception pairs sat in that state.

The line to hold: a code on its own line anywhere in the report (detail block OR summary table) means
the report covers the exception, and the cycle runs. Only a bare-substring occurrence does not.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extractor.description import report_lists_exception  # noqa: E402

# The shape of a detail block: code alone on a line, then the narrative.
DETAIL_REPORT = """[page 1]
EXCEPTION DETAILS
Auditor Company: BP
ID
Description
Gross Credit Requested
Type
010
Coding Error - Contractor Labor
50,479.87
M
Description: The Joint Account was charged $50,479.87 for contractor labor as shown on the
attached JADE lines.
"""

# A late letter: only the still-open exceptions, and the target code appears solely inside the audit
# numbers in the letterhead. This is what cycles 4 and 5 of GW BP Only 2022 look like.
LETTER_WITHOUT_THE_EXCEPTION = """[page 1]
BP Audit No.: 2022-101
Shell Audit No.: 2022SOV_E102B
Property: Great White East

EXCEPTION DETAILS
ID
Description
Type
001
SCM Fee
M
Description: The Operator charged the Great White Joint Account $62,929.08 on various Sumitomo
Corporation Of Americas invoices for SCM Fee Consignment charges.
"""

# Silvertip 2022 Rebuttal 1: exception 02 is closed, so it has no narrative block — but it IS carried
# in the closed-exceptions summary table, so the report does cover it.
CLOSED_SUMMARY_TABLE = """[page 1]
CLOSED EXCEPTIONS
ID
Description
Gross Credit Requested
Type
02
Incorrect Labor Burden Charged
13,616.65
M
03
Coding Error - MRC Global
81,153.75
M
"""


class ReportListsExceptionTest(unittest.TestCase):
    def test_detail_block_lists_the_exception(self):
        self.assertTrue(report_lists_exception(DETAIL_REPORT, "0010"))

    def test_padding_agnostic(self):
        for code in ("10", "010", "0010"):
            self.assertTrue(report_lists_exception(DETAIL_REPORT, code), code)

    def test_substring_in_an_audit_number_is_not_an_entry(self):
        # "10" occurs inside 2022-101 and 2022SOV_E102B; neither makes the report about exception 10.
        self.assertFalse(report_lists_exception(LETTER_WITHOUT_THE_EXCEPTION, "0010"))

    def test_other_exceptions_in_the_same_letter_are_still_found(self):
        self.assertTrue(report_lists_exception(LETTER_WITHOUT_THE_EXCEPTION, "0001"))

    def test_summary_table_row_counts_even_without_a_narrative(self):
        self.assertTrue(report_lists_exception(CLOSED_SUMMARY_TABLE, "0002"))
        self.assertTrue(report_lists_exception(CLOSED_SUMMARY_TABLE, "0003"))

    def test_absent_from_a_summary_table(self):
        self.assertFalse(report_lists_exception(CLOSED_SUMMARY_TABLE, "0010"))

    def test_trailing_whitespace_on_the_code_line(self):
        # PDF text often leaves a trailing space or tab after the code.
        self.assertTrue(report_lists_exception("ID\n0010 \t\nDescription: x\n", "0010"))

    def test_empty_report_text(self):
        self.assertFalse(report_lists_exception("", "0010"))


if __name__ == "__main__":
    unittest.main()
