"""Tests for extracting the report's operator/auditor rebuttal chronology."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extractor.description import (  # noqa: E402
    extract_exception_description, extract_rebuttal_dialogue, split_rebuttal_dialogue,
)


REPORT_WITH_COLON_DATES = """EXCEPTION DETAILS
ID
Description
0003
Affiliate Labor
Description: The Joint Account was charged $751,928.37 for affiliate labor.
Operator Reply: 08/13/2024 The cost certificate will be provided when obtained.
Operator's Reply on August 29, 2025: Cost certificates were attached.
Auditor's Reply on November 26, 2025: The material is redacted and does not cover 2023.
0004
Description: A different exception.
"""


class RebuttalDialogueTest(unittest.TestCase):
    def test_colon_and_on_date_labels_are_preserved_as_dialogue(self):
        dialogue = extract_rebuttal_dialogue(REPORT_WITH_COLON_DATES, "0003")

        self.assertIn("Operator Reply: 08/13/2024", dialogue)
        self.assertIn("Auditor's Reply on November 26, 2025", dialogue)
        self.assertNotIn("A different exception", dialogue)

    def test_colon_label_is_excluded_from_the_concise_description(self):
        description = extract_exception_description(REPORT_WITH_COLON_DATES, "0003")

        self.assertEqual(description, "The Joint Account was charged $751,928.37 for affiliate labor.")

    def test_dialogue_is_split_into_explicit_operator_and_auditor_entries(self):
        entries = split_rebuttal_dialogue(
            extract_rebuttal_dialogue(REPORT_WITH_COLON_DATES, "0003"))

        self.assertEqual([entry["party"] for entry in entries],
                         ["operator", "operator", "auditor"])
        self.assertEqual(entries[0]["reply"],
                         "08/13/2024 The cost certificate will be provided when obtained.")
        self.assertIn("does not cover 2023", entries[2]["reply"])


if __name__ == "__main__":
    unittest.main()
