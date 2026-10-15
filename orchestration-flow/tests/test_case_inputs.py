from __future__ import annotations

import json
import os
import tempfile
import unittest
import csv
from pathlib import Path
from unittest.mock import patch

from shared.classified_exceptions import (
    clear_classified_exceptions_cache,
    find_classified_exception,
)
from shared.pipeline_config import clear_pipeline_config_cache
from otl.sharepoint_reader import SharePointReader
from otl.input_sources import (classified_case_state, expand_rebuttal_cases, extractor_synthesis,
                               graph_input_state, limit_exceptions, rebuttal_cycle_for_case)
from extractor.case_resolver import Case, resolve_cases
from otl.agents.intake import _IntakeSummary, _filter_assets, _resolved_evidence, intake_node


class CaseInputTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_pipeline_config_cache()
        clear_classified_exceptions_cache()

    def _files(self, *, extractor_enabled: bool, catalog: dict | None = None):
        directory = tempfile.TemporaryDirectory()
        root = Path(directory.name)
        catalog_path = root / "classified.json"
        catalog_path.write_text(json.dumps(catalog or {}), encoding="utf-8")
        config_path = root / "pipeline.yaml"
        config_path.write_text(
            "schema_version: 1\n"
            "input_sources:\n"
            "  classified_exceptions:\n"
            "    enabled: true\n"
            f"    path: {catalog_path}\n"
            "  extractor_outputs:\n"
            f"    enabled: {'true' if extractor_enabled else 'false'}\n"
            "    directory_env: TEST_INPUT_BASE\n"
            "    read_folder_env: TEST_SHAREPOINT_READ_FOLDER\n"
            "node_gates: {}\n",
            encoding="utf-8",
        )
        return directory, root, config_path

    def test_classified_lookup_omits_attachment_catalog_metadata(self) -> None:
        catalog = {
            "2024": {
                "Great White": {
                    "40": {
                        "exception_id": "40",
                        "error_label": "Error Copy",
                        "error_type": "01.0 Coding Error",
                        "gross_credit_requested": "1,234.56",
                        "description": "Auditor claim",
                        "table_references": [{"id": "table 1"}],
                        "classification": {"category": "Labor (OTL, Affiliates, Payroll)"},
                        "rebuttals": [
                            {
                                "rebuttal": "Rebuttal 0",
                                "attachments": ["Old.xlsx"],
                                "replies": [{"role": "Operator", "date": "2024-01-01",
                                             "text": "Old operator response."}],
                            },
                            {
                                "rebuttal": "Rebuttal 1",
                                "attachments": ["JADE.xlsx", "JADE back to auditor.xlsx"],
                                "replies": [
                                    {"role": "Operator", "date": "2024-01-01",
                                     "text": "Cumulative operator response to exclude."},
                                    {"role": "Auditor", "date": "2024-02-01",
                                     "text": "Provide the location support."},
                                    {"role": "Operator", "date": "2024-03-01",
                                     "text": "Second operator response to exclude."},
                                    {"role": "Auditor", "date": "2024-04-01",
                                     "text": "The new job-title ground also remains open."},
                                ],
                            },
                        ],
                    }
                }
            }
        }
        directory, _root, config = self._files(extractor_enabled=False, catalog=catalog)
        self.addCleanup(directory.cleanup)

        with patch.dict(os.environ, {"OTL_PIPELINE_CONFIG": str(config)}):
            clear_pipeline_config_cache()
            clear_classified_exceptions_cache()
            record = find_classified_exception("2024", "GreatWhite", "0040")

        self.assertIsNotNone(record)
        self.assertEqual(record["gross_credit_requested"], "1,234.56")
        self.assertNotIn("attachments", record)
        self.assertEqual(record["rebuttal_cycle"], 1)
        self.assertEqual(
            [reply["text"] for reply in record["auditor_rebuttals"]],
            ["Provide the location support.", "The new job-title ground also remains open."],
        )
        self.assertEqual(
            record["latest_auditor_reply"]["text"],
            "The new job-title ground also remains open.",
        )
        self.assertNotIn("operator response", str(record).lower())

    def test_temporal_split_context_overrides_case_facts(self) -> None:
        catalog = {"2024": {"Europa": {"3": {
            "exception_id": "3", "gross_credit_requested": "catalog amount",
            "description": "catalog description", "classification": {"category": "catalog"},
        }}}}
        directory, root, config = self._files(extractor_enabled=False, catalog=catalog)
        self.addCleanup(directory.cleanup)
        entity = root / "corpus" / "2024" / "Europa"
        entity.mkdir(parents=True)
        case = Case(entity_dir=entity, exc_id="0003", title="Labor (OTL, Affiliates, Payroll)",
                    case_id="Europa_2024_0003", context={
                        "csv_year": "2024", "csv_site": "Europa", "csv_exception_id": "3.0",
                        "gross_credit_requested": "751,928.37", "description": "temporal split claim",
                        "source_file": "2024/Europa/report.pdf", "source_type": "Auditor Report",
                        "classification_category": "Labor (OTL, Affiliates, Payroll)",
                        "classification_confidence": "high", "classification_rationale": "test rationale",
                    })
        with patch.dict(os.environ, {"OTL_PIPELINE_CONFIG": str(config)}):
            clear_pipeline_config_cache()
            clear_classified_exceptions_cache()
            record, state = classified_case_state(case, "2024", "Europa")
        self.assertEqual(record["gross_credit_requested"], "751,928.37")
        self.assertEqual(record["description"], "temporal split claim")
        self.assertEqual(record["classification"]["category"], "Labor (OTL, Affiliates, Payroll)")
        self.assertEqual(record["classification"]["confidence"], "high")
        self.assertEqual(state["source_file"], "2024/Europa/report.pdf")

    def test_temporal_split_schema_preserves_context_and_recovers_blank_site(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / "2025" / "GreatWhite").mkdir(parents=True)
        csv_path = root / "temporal.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "year", "site", "exception_ID", "gross_credit_requested", "description",
                "source_file", "source_type", "classification_category",
                "classification_confidence", "classification_rationale",
            ])
            writer.writeheader()
            writer.writerow({
                "year": "2025", "site": "", "exception_ID": "3.0",
                "gross_credit_requested": "740,864.00", "description": "Affiliate labor claim",
                "source_file": "2025/GreatWhite - Chevron only/Rebuttal 0/report.pdf",
                "source_type": "Auditor Report", "classification_category": "Labor (OTL)",
                "classification_confidence": "high", "classification_rationale": "Labor evidence",
            })
        [case] = resolve_cases(root, csv_path)
        self.assertEqual((case.case_id, case.exc_id, case.title),
                         ("GreatWhite_2025_0003", "0003", "Labor (OTL)"))
        self.assertEqual(case.context["gross_credit_requested"], "740,864.00")
        self.assertEqual(case.context["csv_site"], "GreatWhite")
        self.assertEqual(case.context["site_derived_from_source_file"], "true")

    def test_cycle_selection_takes_only_the_newest_auditor_event_from_cumulative_snapshot(self) -> None:
        catalog = {
            "2024": {"Great White": {"40": {
                "exception_id": "40", "description": "Original claim", "classification": {},
                "rebuttals": [
                    {"rebuttal": "Rebuttal 1", "replies": [
                        {"role": "Operator", "text": "Actual operator response 1"},
                        {"role": "Auditor", "date": "2024-01-01", "text": "Auditor reply 1"},
                    ]},
                    {"rebuttal": "Rebuttal 2", "replies": [
                        {"role": "Operator", "text": "Actual operator response 1"},
                        {"role": "Auditor", "date": "2024-01-01", "text": "Auditor reply 1"},
                        {"role": "Operator", "text": "Actual operator response 2"},
                        {"role": "Auditor", "date": "2024-02-01", "text": "Auditor reply 2"},
                    ]},
                ],
            }}}
        }
        directory, _root, config = self._files(extractor_enabled=False, catalog=catalog)
        self.addCleanup(directory.cleanup)

        with patch.dict(os.environ, {"OTL_PIPELINE_CONFIG": str(config)}):
            clear_pipeline_config_cache()
            clear_classified_exceptions_cache()
            cycle1 = find_classified_exception("2024", "GreatWhite", "0040", 1)
            cycle2 = find_classified_exception("2024", "GreatWhite", "0040", 2)

        self.assertEqual(cycle1["latest_auditor_reply"]["text"], "Auditor reply 1")
        self.assertEqual(cycle2["latest_auditor_reply"]["text"], "Auditor reply 2")
        self.assertNotIn("Actual operator response", str(cycle1))
        self.assertNotIn("Actual operator response", str(cycle2))

    def test_requested_cycle_expands_chronologically_through_prerequisites(self) -> None:
        catalog = {
            "2024": {"Great White": {"40": {
                "exception_id": "40", "description": "Claim", "classification": {},
                "rebuttals": [
                    {"rebuttal": f"Rebuttal {cycle}", "replies": []} for cycle in range(5)
                ],
            }}}
        }
        directory, root, config = self._files(extractor_enabled=False, catalog=catalog)
        self.addCleanup(directory.cleanup)
        entity = root / "2024" / "Great White"
        entity.mkdir(parents=True)
        case = Case(entity_dir=entity, exc_id="0040", title="Labor Issues",
                    case_id="GreatWhite_2024_0040", rebuttal_cycle=4)

        with patch.dict(os.environ, {"OTL_PIPELINE_CONFIG": str(config)}):
            clear_pipeline_config_cache()
            clear_classified_exceptions_cache()
            expanded = expand_rebuttal_cases([case])

        self.assertEqual([item.rebuttal_cycle for item in expanded], [0, 1, 2, 3, 4])

    def test_csv_rebuttal_column_cannot_set_cycle(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        entity = root / "2024" / "Great White"
        entity.mkdir(parents=True)
        csv_path = root / "cases.csv"
        csv_path.write_text(
            "year,asset,exc_id,topic,rebuttal\n"
            "2024,Great White,0040,Labor Issues,Rebuttal 9\n",
            encoding="utf-8",
        )

        cases = resolve_cases(root, csv_path)

        self.assertEqual(len(cases), 1)
        self.assertIsNone(cases[0].rebuttal_cycle)

    def test_limit_counts_unique_exceptions_not_rebuttal_rows(self) -> None:
        root = Path("/tmp/corpus/2024")
        cases = [
            Case(root / "Great White", "0001", "Labor Issues", "GreatWhite_2024_0001", cycle)
            for cycle in (1, 2, 3, 4)
        ] + [
            Case(root / "Ursa", "0011", "Labor Issues", "Ursa_2024_0011", 1),
            Case(root / "Europa", "0026", "Labor Issues", "Europa_2024_0026", 1),
        ]

        limited = limit_exceptions(cases, 2)

        self.assertEqual(
            [(case.case_id, case.rebuttal_cycle) for case in limited],
            [
                ("GreatWhite_2024_0001", 1),
                ("GreatWhite_2024_0001", 2),
                ("GreatWhite_2024_0001", 3),
                ("GreatWhite_2024_0001", 4),
                ("Ursa_2024_0011", 1),
            ],
        )

    def test_disabled_extractor_source_never_reads_outputs(self) -> None:
        directory, root, config = self._files(extractor_enabled=False)
        self.addCleanup(directory.cleanup)
        reader = SharePointReader()

        with patch.dict(os.environ, {
            "OTL_PIPELINE_CONFIG": str(config),
            "TEST_SHAREPOINT_READ_FOLDER": "/Extractor Output",
        }), patch.object(reader, "local_synthesis", side_effect=AssertionError("reader called")):
            clear_pipeline_config_cache()
            self.assertEqual(
                extractor_synthesis(reader, "2024", "GreatWhite", "0040", 0), {}
            )

    def test_enabled_extractor_source_reads_sharepoint_output_when_present(self) -> None:
        directory, root, config = self._files(extractor_enabled=True)
        self.addCleanup(directory.cleanup)
        payload = {"rebuttal_cycle": 2, "synthesis": "optional extractor context"}
        reader = SharePointReader()
        reader.target = "sharepoint"

        with patch.dict(os.environ, {
            "OTL_PIPELINE_CONFIG": str(config),
            "TEST_SHAREPOINT_READ_FOLDER": "/Extractor Output",
        }), patch.object(reader, "sharepoint_synthesis", return_value=payload) as read:
            clear_pipeline_config_cache()
            self.assertEqual(
                extractor_synthesis(reader, "2024", "GreatWhite", "0040", 2), payload
            )
            self.assertEqual(reader.read_folder, "/Extractor Output")
            read.assert_called_once_with("2024", "GreatWhite", "0040", 2)

    def test_enabled_extractor_source_skips_when_sharepoint_folder_is_unset(self) -> None:
        directory, root, config = self._files(extractor_enabled=True)
        self.addCleanup(directory.cleanup)
        reader = SharePointReader()
        reader.target = "sharepoint"

        with patch.dict(os.environ, {
            "OTL_PIPELINE_CONFIG": str(config),
            "TEST_SHAREPOINT_READ_FOLDER": "",
        }), patch.object(reader, "sharepoint_synthesis", side_effect=AssertionError("reader called")):
            clear_pipeline_config_cache()
            self.assertEqual(
                extractor_synthesis(reader, "2024", "GreatWhite", "0040", 0), {}
            )

    def test_enabled_extractor_source_reads_local_output_for_a_local_target(self) -> None:
        directory, root, config = self._files(extractor_enabled=True)
        self.addCleanup(directory.cleanup)
        input_dir = root / "outputs"
        input_dir.mkdir()
        payload = {"rebuttal_cycle": 2, "synthesis": "local extractor context"}
        reader = SharePointReader()
        reader.target = "local"

        with patch.dict(os.environ, {
            "OTL_PIPELINE_CONFIG": str(config),
            "TEST_INPUT_BASE": str(input_dir),
        }), patch.object(reader, "local_synthesis", return_value=payload) as read:
            clear_pipeline_config_cache()
            self.assertEqual(
                extractor_synthesis(reader, "2024", "GreatWhite", "0040", 2), payload
            )
            self.assertEqual(reader.input_base, str(input_dir.resolve()))
            read.assert_called_once_with("2024", "GreatWhite", "0040", 2)

    def test_local_extractor_source_reads_only_the_requested_rebuttal_cycle(self) -> None:
        directory, root, config = self._files(extractor_enabled=True)
        self.addCleanup(directory.cleanup)
        input_dir = root / "outputs"
        exception_dir = input_dir / "2024" / "GreatWhite" / "0040"
        exception_dir.mkdir(parents=True)
        cycle_zero = {"rebuttal_cycle": 0, "synthesis": "cycle zero"}
        cycle_five = {"rebuttal_cycle": 5, "synthesis": "cycle five"}
        (exception_dir / "rebuttal_0_synthesis.json").write_text(
            json.dumps(cycle_zero), encoding="utf-8"
        )
        (exception_dir / "rebuttal_5_synthesis.json").write_text(
            json.dumps(cycle_five), encoding="utf-8"
        )
        reader = SharePointReader()
        reader.target = "local"

        with patch.dict(os.environ, {
            "OTL_PIPELINE_CONFIG": str(config),
            "TEST_INPUT_BASE": str(input_dir),
        }):
            clear_pipeline_config_cache()
            self.assertEqual(
                extractor_synthesis(reader, "2024", "GreatWhite", "0040", 0),
                cycle_zero,
            )
            self.assertEqual(
                extractor_synthesis(reader, "2024", "GreatWhite", "0040", 5),
                cycle_five,
            )
            self.assertEqual(
                extractor_synthesis(reader, "2024", "GreatWhite", "0040", 3),
                {},
            )

    def test_rebuttal_cycle_zero_is_preserved(self) -> None:
        case = Case(
            entity_dir="/data/2024/GreatWhite",
            exc_id="0040",
            title="Labor",
            case_id="GreatWhite_2024_0040",
            rebuttal_cycle=0,
        )

        self.assertEqual(
            rebuttal_cycle_for_case(case, {"rebuttal_cycle": 5}),
            0,
        )

    def test_disabled_or_missing_extractor_output_never_enters_graph_state(self) -> None:
        state = graph_input_state(
            case_state={"case_id": "GreatWhite_2024_0040"},
            classified={"description": "Auditor claim"},
            resolved={
                "rebuttal_dir": "/case",
                "auditor_report_pdf": "/case/report.pdf",
                "exception_exhibits": ["/case/JADE.xlsx"],
                "common_reports_dir": "/common",
                "otl_information_dir": "/otl",
                "sandbox_read_roots": ["/case"],
            },
            synthesis={},
            run_id="GreatWhite_2024_0040",
            trace_path="/tmp/trace.jsonl",
            rebuttal_cycle=1,
        )

        self.assertNotIn("synthesis", state)
        self.assertFalse(state["extractor_synthesis_loaded"])
        self.assertEqual(state["auditor_narrative"], "Auditor claim")
        self.assertEqual(state["open_questions"], [])
        self.assertEqual(state["latest_auditor_reply"], {})

    def test_intake_assets_are_normalized_to_agreement_corpus_keys(self) -> None:
        self.assertEqual(
            _filter_assets(
                ["GreatWhite (BP Only)", "Mars et al", "Atlantis", "Tobago", "mars"],
                fallback="Silvertip",
            ),
            ["great_white_bp_only", "mars", "tobago", "silvertip"],
        )

    def test_intake_evidence_comes_from_filtered_resolved_workbooks_and_pdfs(self) -> None:
        state = {
            "exception_exhibits": [
                "/case/JADE Labor.xlsx",
                "/case/IR 24 back to auditor.xlsx",
            ],
            "supporting_pdfs": [
                "/case/Support.pdf",
                "/case/JADE Labor.xlsx",
            ],
        }
        self.assertEqual(_resolved_evidence(state), ["JADE Labor.xlsx", "Support.pdf"])

    def test_intake_extracts_people_from_synthesis_and_attachment_analysis(self) -> None:
        extraction = _IntakeSummary(
            nl_summary="The exception concerns a Great White offsite labor charge.",
            people_found=["Smith Jonathan S", "Szymanski, Michael James"],
        )
        state = {
            "case": {"asset": "Great White", "amount": "1,234.56"},
            "auditor_narrative": "The Great White Joint Account was charged.",
            "exception_exhibits": ["/case/JADE.xlsx", "/case/Response back to auditor.xlsx"],
            "supporting_pdfs": ["/case/Support.pdf"],
            "synthesis": {
                "synthesis": "The ledger narrative names Smith Jonathan S.",
                "attachments": [{
                    "file_name": "JADE.xlsx",
                    "role": "auditor evidence",
                    "overall_summary": "Names Szymanski, Michael James.",
                    "tabs": [],
                }],
            },
        }

        with patch("otl.agents.intake.llm.structured", return_value=extraction) as structured:
            intake = intake_node(state)["intake"].model_dump()

        self.assertEqual(intake["assets_in_question"], ["great_white"])
        self.assertEqual(intake["assets_as_seen"], ["Great White"])
        self.assertEqual(intake["dollars"], 1234.56)
        self.assertEqual(intake["collected_evidence"], ["JADE.xlsx", "Support.pdf"])
        self.assertEqual(intake["error_code"], "")
        self.assertEqual(intake["code_family"], "unknown")
        self.assertEqual(intake["people_found"], ["Smith Jonathan S", "Szymanski, Michael James"])
        prompt = structured.call_args.args[2]
        self.assertIn("Extractor synthesis:", prompt)
        self.assertIn("Smith Jonathan S", prompt)
        self.assertIn("Extractor attachment analysis", prompt)
        self.assertIn("Szymanski, Michael James", prompt)

    def test_intake_uses_latest_auditor_reply_after_cycle_zero(self) -> None:
        extraction = _IntakeSummary(
            nl_summary="The latest reply requests additional location support."
        )
        state = {
            "case": {
                "asset": "Great White",
                "amount": "1,234.56",
                "description": "Original exception about labor classification.",
                "rebuttal_cycle": 1,
            },
            "auditor_narrative": "Original exception about labor classification.",
            "latest_auditor_reply": {
                "role": "Auditor",
                "text": "Please provide the location support for the challenged period.",
            },
            "exception_exhibits": [],
            "supporting_pdfs": [],
        }

        with patch("otl.agents.intake.llm.structured", return_value=extraction) as structured:
            intake_node(state)

        prompt = structured.call_args.args[2]
        self.assertIn("Latest auditor reply for rebuttal cycle 1", prompt)
        self.assertIn("Please provide the location support", prompt)
        self.assertIn("Original exception about labor classification", prompt)


if __name__ == "__main__":
    unittest.main()
