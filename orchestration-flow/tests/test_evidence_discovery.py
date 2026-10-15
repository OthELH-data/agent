"""Regression coverage for shared extractor → OTL evidence discovery."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from otl.case_resolver import resolve_otl_case
from otl.agents._common import cited_attachments, filtered_attachments
from otl.input_sources import classified_case_state, graph_input_state
from otl.output_writer import assemble_report
from shared.case_resolver import Case
from shared.rebuttals import (Cycle, EvidenceDiscovery, find_exception_evidence,
                              union_auditor_cited_attachments)


class EvidenceDiscoveryTests(unittest.TestCase):
    @patch("otl.input_sources.find_classified_exception")
    def test_missing_classification_block_normalizes_to_empty_mapping(self, find_classified) -> None:
        case = Case(Path("/case/2024/Great White"), "0010", "Labor", "GreatWhite_2024_0010", 1)
        find_classified.return_value = {
            "year": "2024", "asset": "Great White", "exception_id": "0010",
            "description": "Auditor claim", "rebuttal_cycle": 1,
        }

        classified, case_state = classified_case_state(case, "2024", "Great White")
        self.assertEqual(classified.get("classification"), {})
        self.assertEqual(case_state["classification"], {})

        state = graph_input_state(
            case_state={"case_id": case.case_id}, classified=classified,
            resolved={
                "rebuttal_dir": "/case", "auditor_report_pdf": "/case/report.pdf",
                "exception_exhibits": ["/case/JADE.xlsx"], "supporting_pdfs": [],
                "common_reports_dir": "/common", "otl_information_dir": "/otl",
                "sandbox_read_roots": ["/case"],
            },
            synthesis={}, run_id=case.case_id, trace_path="/tmp/trace", rebuttal_cycle=1,
        )
        self.assertEqual(state["case"]["classification"], {})

    def test_all_operator_workbooks_are_excluded_and_manifest_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            entity = Path(directory) / "2024" / "Great White"
            auditor = entity / "Rebuttal 1" / "Audit Report"
            exhibits = auditor / "0010"
            exhibits.mkdir(parents=True)
            (exhibits / "Operator Response.xlsx").touch()
            (exhibits / "Shell Findings.xlsx").touch()

            discovery = find_exception_evidence(
                Cycle(number=1, auditor_dir=auditor), "0010"
            )

            self.assertEqual(discovery.workbooks, [])
            self.assertTrue(discovery.all_workbooks_operator_responses)
            self.assertEqual(
                {path.name for path in discovery.excluded_operator_workbooks},
                {"Operator Response.xlsx", "Shell Findings.xlsx"},
            )
            self.assertIn("No auditor-side workbook evidence", discovery.evidence_exclusion_notice)

            restored = EvidenceDiscovery.from_manifest(discovery.to_manifest(entity), entity)
            self.assertIsNotNone(restored)
            self.assertTrue(restored.all_workbooks_operator_responses)
            self.assertEqual(
                {path.name for path in restored.excluded_operator_workbooks},
                {"Operator Response.xlsx", "Shell Findings.xlsx"},
            )

            resolved = resolve_otl_case(
                Case(entity, "0010", "Labor", "GreatWhite_2024_0010", 1),
                [Cycle(number=1, auditor_dir=auditor)],
                evidence_manifest=discovery.to_manifest(entity),
            )
            self.assertEqual(resolved["exception_exhibits"], [])
            self.assertTrue(resolved["all_workbooks_operator_responses"])
            self.assertIn("No auditor-side workbook evidence", resolved["evidence_exclusion_notice"])

    def test_manifest_is_consumed_without_a_second_pdf_filter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            entity = Path(directory) / "2024" / "Great White"
            auditor = entity / "Rebuttal 1" / "Audit Report"
            auditor.mkdir(parents=True)
            safe_pdf = auditor / "Ledger Support.pdf"
            response_pdf = auditor / "Operator Response.pdf"
            safe_pdf.touch()
            response_pdf.touch()

            discovery = EvidenceDiscovery(
                cycle_number=1,
                auditor_dir=auditor,
                supporting_pdfs=[safe_pdf],
                excluded_operator_pdfs=[response_pdf],
            )
            resolved = resolve_otl_case(
                Case(entity, "0010", "Labor", "GreatWhite_2024_0010", 1),
                [Cycle(number=1, auditor_dir=auditor)],
                evidence_manifest=discovery.to_manifest(entity),
            )

            self.assertEqual([Path(path).name for path in resolved["supporting_pdfs"]],
                             [safe_pdf.name])
            self.assertEqual([Path(path).name for path in resolved["excluded_operator_pdfs"]],
                             [response_pdf.name])

    def test_auditor_citations_override_workbook_response_filter_and_round_trip_to_otl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            entity = Path(directory) / "2024" / "Great White"
            auditor = entity / "Rebuttal 1" / "Audit Report"
            exhibits = auditor / "0010"
            exhibits.mkdir(parents=True)
            discovered = exhibits / "JADE.xlsx"
            lock_file = exhibits / "~$JADE.xlsx"
            cited_book = auditor / "Auditor Cited Support.xlsx"
            cited_pdf = auditor / "Auditor Cited Invoice.pdf"
            operator_book = auditor / "Operator Response.xlsx"
            operator_lock_file = auditor / "~$Operator Response.xlsx"
            for path in (discovered, lock_file, cited_book, cited_pdf, operator_book,
                         operator_lock_file):
                path.touch()

            discovery = find_exception_evidence(Cycle(number=1, auditor_dir=auditor), "0010")
            self.assertNotIn(lock_file, discovery.workbooks)
            union_auditor_cited_attachments(discovery, [
                "Auditor Cited Support.xlsx", "Auditor Cited Invoice.pdf", "Operator Response.xlsx",
            ])

            self.assertEqual(
                {path.name for path in discovery.workbooks},
                {"JADE.xlsx", "Auditor Cited Support.xlsx", "Operator Response.xlsx"},
            )
            self.assertEqual(
                [path.name for path in discovery.supporting_pdfs], ["Auditor Cited Invoice.pdf"],
            )
            self.assertEqual(
                discovery.auditor_cited_attachment_names,
                ["Auditor Cited Support.xlsx", "Auditor Cited Invoice.pdf", "Operator Response.xlsx"],
            )
            self.assertEqual(
                {path.name for path in discovery.cited_attachment_workbooks},
                {"Auditor Cited Support.xlsx", "Operator Response.xlsx"},
            )
            self.assertNotIn("Operator Response.xlsx",
                             {path.name for path in discovery.excluded_operator_workbooks})
            self.assertNotIn(operator_lock_file, discovery.cited_attachment_workbooks)

            manifest = discovery.to_manifest(entity)
            restored = EvidenceDiscovery.from_manifest(manifest, entity)
            self.assertIsNotNone(restored)
            self.assertEqual(manifest["auditor_cited_attachment_names"], discovery.auditor_cited_attachment_names)
            self.assertEqual({path.name for path in restored.workbooks},
                             {"JADE.xlsx", "Auditor Cited Support.xlsx", "Operator Response.xlsx"})

            resolved = resolve_otl_case(
                Case(entity, "0010", "Labor", "GreatWhite_2024_0010", 1),
                [Cycle(number=1, auditor_dir=auditor)], evidence_manifest=manifest,
            )
            self.assertEqual(
                {Path(path).name for path in resolved["exception_exhibits"]},
                {"JADE.xlsx", "Auditor Cited Support.xlsx", "Operator Response.xlsx"},
            )
            self.assertEqual(
                set(resolved["final_attachment_names"]),
                {"JADE.xlsx", "Auditor Cited Support.xlsx", "Operator Response.xlsx",
                 "Auditor Cited Invoice.pdf"},
            )
            state = graph_input_state(
                case_state={"attachments": ["stale catalog attachment.xlsx"]}, classified={},
                resolved=resolved, synthesis={}, run_id="case", trace_path="/tmp/trace", rebuttal_cycle=1,
            )
            self.assertEqual(state["case"]["attachments"], resolved["final_attachment_names"])
            self.assertEqual(state["final_attachments"], resolved["final_attachment_names"])

    def test_extensionless_auditor_citation_promotes_response_workbook_with_upload_year(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            entity = Path(directory) / "2024" / "Ursa"
            auditor = entity / "Rebuttal 0" / "Auditor report"
            exhibits = auditor / "060"
            exhibits.mkdir(parents=True)
            response_book = exhibits / "response to IR 62-- Ursa Staff TW _ 2023.xlsx"
            response_lock = exhibits / "~$response to IR 62-- Ursa Staff TW _ 2023.xlsx"
            response_book.touch()
            response_lock.touch()

            diagnostics: list[str] = []
            discovery = find_exception_evidence(
                Cycle(number=0, auditor_dir=auditor), "0060", warn=diagnostics.append,
            )
            self.assertEqual(discovery.workbooks, [])
            self.assertEqual(discovery.excluded_operator_workbooks, [response_book])
            self.assertEqual(diagnostics, [
                "WARN: every auditor-side workbook here looks like an operator response "
                "(response to IR 62-- Ursa Staff TW _ 2023.xlsx) — excluding all of them; no "
                "auditor-side workbook evidence is safe to analyze."
            ])

            union_auditor_cited_attachments(
                discovery, ["response to IR-62-Ursa Staff TW"],
            )

            self.assertEqual(discovery.workbooks, [response_book])
            self.assertEqual(discovery.cited_attachment_workbooks, [response_book])
            self.assertEqual(discovery.excluded_operator_workbooks, [])
            self.assertEqual(discovery.evidence_exclusion_notice, "")
            manifest = discovery.to_manifest(entity)
            self.assertEqual(
                manifest["workbooks"],
                ["Rebuttal 0/Auditor report/060/response to IR 62-- Ursa Staff TW _ 2023.xlsx"],
            )

            resolved = resolve_otl_case(
                Case(entity, "0060", "Labor", "ursa_2024_0060", 0),
                [Cycle(number=0, auditor_dir=auditor)], evidence_manifest=manifest,
            )
            self.assertEqual(
                [Path(path).resolve() for path in resolved["exception_exhibits"]],
                [response_book.resolve()],
            )
            self.assertEqual(
                resolved["final_attachment_names"],
                ["response to IR 62-- Ursa Staff TW _ 2023.xlsx"],
            )

            synth = {
                "title": "Labor",
                "attachments": [
                    {
                        "file_name": response_book.name,
                        "tabs": [{"tab_name": "Operator Response", "purpose": "employee listing"}],
                    },
                    {
                        "file_name": response_lock.name,
                        "tabs": [{"tab_name": "Operator Response"}],
                    },
                ],
                "evidence_discovery": manifest,
            }
            labor_attachments = filtered_attachments(synth)
            self.assertEqual([item["file_name"] for item in labor_attachments], [response_book.name])
            self.assertEqual(
                [tab["tab_name"] for tab in labor_attachments[0]["tabs"]],
                ["Operator Response"],
            )
            cited_workbooks, cited_sheets, _tabs = cited_attachments(synth)
            self.assertEqual(cited_workbooks, [response_book.name])
            self.assertEqual(cited_sheets, {response_book.name: ["Operator Response"]})

            manifest_only_workbooks, manifest_only_sheets, _tabs = cited_attachments({
                **synth,
                "attachments": [],
            })
            self.assertEqual(manifest_only_workbooks, [response_book.name])
            self.assertEqual(manifest_only_sheets, {response_book.name: []})

    @patch("otl.case_resolver.extract_auditor_cited_attachment_names")
    @patch("otl.case_resolver.extract_exception_description", return_value="See Cited Support.xlsx.")
    @patch("otl.case_resolver.extract_pdf", return_value={"text": "report"})
    def test_fallback_uses_the_same_citation_union(self, _pdf, _description, cited_names) -> None:
        with tempfile.TemporaryDirectory() as directory:
            entity = Path(directory) / "2024" / "Great White"
            auditor = entity / "Rebuttal 1" / "Audit Report"
            exhibits = auditor / "0010"
            exhibits.mkdir(parents=True)
            (exhibits / "JADE.xlsx").touch()
            (auditor / "Cited Support.xlsx").touch()
            (auditor / "Auditor Report.pdf").touch()
            cited_names.return_value = ["Cited Support.xlsx"]

            resolved = resolve_otl_case(
                Case(entity, "0010", "Labor", "GreatWhite_2024_0010", 1),
                [Cycle(number=1, auditor_dir=auditor, auditor_report=auditor / "Auditor Report.pdf")],
            )

            self.assertEqual(
                {Path(path).name for path in resolved["exception_exhibits"]},
                {"JADE.xlsx", "Cited Support.xlsx"},
            )
            cited_names.assert_called_once_with("See Cited Support.xlsx.")

    def test_report_notice_is_reviewer_only_for_empty_evidence_denial(self) -> None:
        notice = "All candidate workbooks were excluded as operator responses."
        state = {
            "case": {"case_id": "GreatWhite_2024_0010", "exc_id": "0010",
                     "year": "2024", "asset": "Great White", "title": "Labor"},
            "rebuttal_cycle": 1,
            "drafter": {"disposition": "deny", "draft_response": "This exception is denied."},
            "exception_exhibits": [],
            "all_workbooks_operator_responses": True,
            "evidence_exclusion_notice": notice,
        }

        report = assemble_report(state, run_id="case", elapsed_minutes=0.0,
                                 token_usage={}, trace_events=[])

        self.assertEqual(report.evidence_information_message, notice)
        self.assertEqual(report.draft_response, "This exception is denied.")

        state["exception_exhibits"] = ["JADE.xlsx"]
        report = assemble_report(state, run_id="case", elapsed_minutes=0.0,
                                 token_usage={}, trace_events=[])
        self.assertEqual(report.evidence_information_message, "")


if __name__ == "__main__":
    unittest.main()
