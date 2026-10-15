"""Regression coverage for description-only auditor attachment citation extraction."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from extractor import explain
from extractor.schemas import AuditorCitedAttachmentNames


class AuditorAttachmentCitationTests(unittest.TestCase):
    def test_light_tier_prompt_contains_only_the_exception_description(self) -> None:
        description = "See attachment JADE Support.xlsx and the attached Invoice 123.pdf."
        expected = AuditorCitedAttachmentNames(
            attachment_names=["JADE Support.xlsx", "Invoice 123.pdf"]
        )

        with patch.object(explain, "structured_call", return_value=expected) as structured:
            result = explain.extract_auditor_cited_attachment_names(description)

        self.assertEqual(result, ["JADE Support.xlsx", "Invoice 123.pdf"])
        tier, _system, user, schema = structured.call_args.args[:4]
        self.assertEqual(tier, "light")
        self.assertEqual(schema, AuditorCitedAttachmentNames)
        self.assertEqual(user, "AUDITOR EXCEPTION DESCRIPTION:\n" + description)
        self.assertNotIn("CANDIDATE", user)
        self.assertNotIn("DISCOVERED", user)


if __name__ == "__main__":
    unittest.main()
