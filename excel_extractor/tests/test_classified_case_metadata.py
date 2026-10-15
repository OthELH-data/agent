"""Coverage for classified-exceptions overrides of extractor report header facts."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run


class ClassifiedCaseMetadataTests(unittest.TestCase):
    def test_main_forwards_catalog_metadata_to_cycle_runner(self) -> None:
        case = SimpleNamespace(
            entity_dir=Path("/corpus/2024/Great White"), exc_id="0007",
            case_id="GreatWhite_2024_0007", exception_type="Fallback title",
        )
        metadata = run._ExtractorCaseMetadata(
            year="2025", asset="Europa", exception_category="OTL Labor",
            case_id="Europa_2025_0007", title="Catalog title",
            auditor_description="Catalog description.",
        )
        with (
            patch.object(run.sys, "argv", ["run.py", "--case-id", case.case_id]),
            patch.object(run, "default_corpus_base", return_value=Path("/corpus")),
            patch.object(run, "resolve_case_id", return_value=case),
            patch.object(run, "_classified_case_metadata", return_value=metadata),
            patch.object(run, "ensure_run_id", return_value="test-run"),
            patch.object(run, "make_sink", return_value=object()),
            patch.object(run, "extractor_excel_parsing_enabled", return_value=False),
            patch.object(run, "run_rebuttal_cycles") as run_cycles,
        ):
            run.main()

        self.assertEqual(run_cycles.call_args.args[2:4], ("Catalog title", "Europa_2025_0007"))
        self.assertEqual(run_cycles.call_args.kwargs["category"], "OTL Labor")
        self.assertEqual(run_cycles.call_args.kwargs["year"], "2025")
        self.assertEqual(run_cycles.call_args.kwargs["asset"], "Europa")
        self.assertEqual(run_cycles.call_args.kwargs["auditor_description"], "Catalog description.")

    def test_missing_catalog_is_a_true_no_op(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "classified_exceptions.json"
            with patch.object(run, "classified_exceptions_path", return_value=missing), \
                    patch.object(run, "find_classified_exception") as lookup:
                metadata = run._classified_case_metadata(
                    Path("/corpus/2024/Great White"), "0010", "GreatWhite_2024_0010", "Labor",
                )

        self.assertIsNone(metadata)
        lookup.assert_not_called()

    @patch.object(run, "infer_year_asset", return_value=("2024", "Great White"))
    def test_catalog_supplies_all_report_header_values_when_classification_exists(self, _asset) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "classified_exceptions.json"
            catalog.touch()
            record = {
                "year": "2025", "asset": "Europa", "exception_id": "7",
                "case_id": "Europa_2025_0007", "description": "Catalog auditor description.",
                "classification": {
                    "exception_category": "OTL Labor",
                    "title": "Affiliate payroll charge",
                },
            }
            with patch.object(run, "classified_exceptions_path", return_value=catalog), \
                    patch.object(run, "find_classified_exception", return_value=record), \
                    patch.object(run, "lookup_exception_type") as csv_lookup:
                metadata = run._classified_case_metadata(
                    Path("/corpus/2024/Great White"), "0007", "fallback", "Fallback title",
                )

        self.assertEqual(metadata.year, "2025")
        self.assertEqual(metadata.asset, "Europa")
        self.assertEqual(metadata.exception_category, "OTL Labor")
        self.assertEqual(metadata.case_id, "Europa_2025_0007")
        self.assertEqual(metadata.title, "Affiliate payroll charge")
        self.assertEqual(metadata.auditor_description, "Catalog auditor description.")
        csv_lookup.assert_not_called()

    @patch.object(run, "infer_year_asset", return_value=("2024", "Great White"))
    def test_missing_classification_uses_designated_labor_csv_for_category_and_title(self, _asset) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "classified_exceptions.json"
            catalog.touch()
            record = {
                "year": "2024", "asset": "Great White", "exception_id": "10",
                "description": "Catalog description.", "classification": {},
            }
            with patch.object(run, "classified_exceptions_path", return_value=catalog), \
                    patch.object(run, "find_classified_exception", return_value=record), \
                    patch.object(run, "lookup_exception_type", return_value="Labor (OTL)") as csv_lookup:
                metadata = run._classified_case_metadata(
                    Path("/corpus/2024/Great White"), "0010", "fallback", "Fallback title",
                )

        self.assertEqual(metadata.exception_category, "Labor (OTL)")
        self.assertEqual(metadata.title, "Labor (OTL)")
        csv_lookup.assert_called_once_with("2024", "Great White", "0010", [run.DEFAULT_CASES_CSV])


if __name__ == "__main__":
    unittest.main()
