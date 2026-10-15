"""Regression coverage for JSON-safe OTL report output."""
from __future__ import annotations

from datetime import date
import unittest

from otl.output_writer import OTLExceptionReport, assemble_report, write_report


class _RecordingSink:
    def __init__(self) -> None:
        self.relative_path = ""
        self.data = None

    def write(self, relative_path: str, data: dict) -> str:
        self.relative_path = relative_path
        self.data = data
        return f"memory://{relative_path}"


class OutputWriterTests(unittest.TestCase):
    def test_assemble_report_preserves_active_auditor_narrative(self) -> None:
        report = assemble_report(
            {
                "case": {"case_id": "marsetal_2024_0012", "exc_id": "0012"},
                "auditor_narrative": "The auditor disputes the designer labor charges.",
            },
            run_id="marsetal_2024_0012", elapsed_minutes=0.1, token_usage={}, trace_events=[],
        )

        self.assertEqual(
            "The auditor disputes the designer labor charges.",
            report.model_dump()["auditor_narrative"],
        )

    def test_write_report_converts_nested_dates_to_json_values(self) -> None:
        sink = _RecordingSink()
        report = OTLExceptionReport(
            stages={"intake": {"period_end": date(2024, 1, 31)}},
            trace=[{"event_date": date(2024, 2, 1)}],
        )

        location = write_report(report, sink=sink, year="2024", asset="Europa", exc_id="0008")

        self.assertEqual("memory://2024/Europa/0008/rebuttal_0_otl.json", location)
        self.assertEqual("2024-01-31", sink.data["stages"]["intake"]["period_end"])
        self.assertEqual("2024-02-01", sink.data["trace"][0]["event_date"])


if __name__ == "__main__":
    unittest.main()
