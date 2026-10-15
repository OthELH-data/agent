"""Regression coverage for graph-input narrative seeding."""
from __future__ import annotations

import unittest

from otl.input_sources import graph_input_state


class GraphInputStateTests(unittest.TestCase):
    def test_classified_description_seeds_active_auditor_narrative(self) -> None:
        state = graph_input_state(
            case_state={"case_id": "marsetal_2024_0012", "description": "Fallback description"},
            classified={"description": "Active auditor exception description."},
            resolved={
                "rebuttal_dir": "/case",
                "auditor_report_pdf": "/case/report.pdf",
                "exception_exhibits": [],
                "supporting_pdfs": [],
                "common_reports_dir": "/common",
                "otl_information_dir": "/otl",
                "sandbox_read_roots": ["/case"],
            },
            synthesis={}, run_id="marsetal_2024_0012", trace_path="/tmp/trace.jsonl",
            rebuttal_cycle=0,
        )

        self.assertEqual("Active auditor exception description.", state["auditor_narrative"])


if __name__ == "__main__":
    unittest.main()
