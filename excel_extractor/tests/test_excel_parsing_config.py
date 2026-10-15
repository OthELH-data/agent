from __future__ import annotations

import inspect
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

EXTRACTOR_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXTRACTOR_DIR.parent
for path in (str(REPO_ROOT), str(EXTRACTOR_DIR)):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

import run as extractor_run  # noqa: E402
import run_batch as extractor_batch  # noqa: E402
from extractor import explain  # noqa: E402
from shared.rebuttals import Cycle  # noqa: E402
from extractor.schemas import (  # noqa: E402
    ExhibitEvidenceReport,
    InvoiceInterpretation,
    SynthesisResult,
)
from shared.extractor_settings import (  # noqa: E402
    clear_extractor_settings_cache,
    extractor_excel_parsing_enabled,
)


class ExcelParsingConfigTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_extractor_settings_cache()

    def _config(self, excel_block: str | None) -> tuple[tempfile.TemporaryDirectory, Path]:
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "otl_pipeline.yaml"
        block = f"    excel_parsing:\n{excel_block}" if excel_block is not None else ""
        path.write_text(
            "schema_version: 1\n"
            "input_sources:\n"
            "  extractor_outputs:\n"
            "    enabled: true\n"
            f"{block}"
            "node_gates: {}\n",
            encoding="utf-8",
        )
        return directory, path

    def test_standalone_extractor_reads_disabled_setting(self) -> None:
        directory, path = self._config("      enabled: false\n")
        self.addCleanup(directory.cleanup)

        with patch.dict(os.environ, {"OTL_PIPELINE_CONFIG": str(path)}):
            clear_extractor_settings_cache()
            self.assertFalse(extractor_excel_parsing_enabled())

    def test_standalone_extractor_defaults_enabled_when_setting_is_absent(self) -> None:
        directory, path = self._config(None)
        self.addCleanup(directory.cleanup)

        with patch.dict(os.environ, {"OTL_PIPELINE_CONFIG": str(path)}):
            clear_extractor_settings_cache()
            self.assertTrue(extractor_excel_parsing_enabled())

    def test_main_passes_shared_setting_to_cycle_runner(self) -> None:
        case = SimpleNamespace(
            entity_dir=Path("/corpus/2024/GreatWhite"),
            exc_id="0040",
            title="Labor (OTL, Affiliates, Payroll)",
            case_id="GreatWhite_2024_0040",
        )
        sink = object()
        with (
            patch.object(sys, "argv", ["run.py", "--case-id", case.case_id]),
            patch.object(extractor_run, "default_corpus_base", return_value=Path("/corpus")),
            patch.object(extractor_run, "resolve_case_id", return_value=case),
            patch.object(extractor_run, "ensure_run_id", return_value="test-run"),
            patch.object(extractor_run, "make_sink", return_value=sink),
            patch.object(extractor_run, "extractor_excel_parsing_enabled", return_value=False),
            patch.object(extractor_run, "run_rebuttal_cycles") as run_cycles,
            redirect_stdout(io.StringIO()),
        ):
            extractor_run.main()

        self.assertFalse(run_cycles.call_args.kwargs["excel_parsing_enabled"])

    def test_cycle_runner_passes_setting_to_build_report(self) -> None:
        report_pdf = Path("/corpus/2024/GreatWhite/Rebuttal 0/Audit Report/report.pdf")
        workbook = report_pdf.parent / "0040" / "support.xlsx"
        cycle = Cycle(number=0, auditor_dir=report_pdf.parent, auditor_report=report_pdf)
        sink = Mock()
        sink.folder_exists.return_value = False

        with (
            patch.object(extractor_run, "discover_rebuttal_cycles", return_value=[cycle]),
            patch.object(extractor_run, "infer_year_asset", return_value=("2024", "GreatWhite")),
            patch.object(extractor_run, "find_exception_evidence", return_value=([workbook], [])),
            patch.object(extractor_run, "exception_folder_is_empty", return_value=False),
            patch.object(extractor_run, "extract_pdf", return_value={"text": "Exception 0040"}),
            patch.object(extractor_run, "extract_auditor_cited_attachment_names", return_value=[]),
            patch.object(extractor_run, "build_report", side_effect=SystemExit("stop")) as build_report,
            redirect_stdout(io.StringIO()),
        ):
            extractor_run.run_rebuttal_cycles(
                Path("/corpus/2024/GreatWhite"), "0040", "Labor", "GreatWhite_2024_0040", sink,
                excel_parsing_enabled=False,
            )

        self.assertFalse(build_report.call_args.kwargs["excel_parsing_enabled"])
        manifest = build_report.call_args.kwargs["evidence_discovery"]
        self.assertEqual(manifest["cycle_number"], 0)
        self.assertEqual(manifest["workbooks"], ["Rebuttal 0/Audit Report/0040/support.xlsx"])

    def test_direct_call_defaults_remain_workbook_enabled(self) -> None:
        build_default = inspect.signature(extractor_run.build_report).parameters[
            "excel_parsing_enabled"
        ].default
        cycle_default = inspect.signature(extractor_run.run_rebuttal_cycles).parameters[
            "excel_parsing_enabled"
        ].default
        self.assertIs(build_default, True)
        self.assertIs(cycle_default, True)

    def test_batch_workers_inherit_config_path(self) -> None:
        case = SimpleNamespace(case_id="GreatWhite_2024_0040", title="Labor")
        completed = SimpleNamespace(stdout="", stderr="", returncode=0)
        config_path = "/deployment/otl_pipeline.yaml"

        with (
            patch.dict(os.environ, {"OTL_PIPELINE_CONFIG": config_path}),
            patch.object(extractor_batch.subprocess, "run", return_value=completed) as run_process,
        ):
            extractor_batch._run_case(case, 1, 1)
            self.assertEqual(run_process.call_args.kwargs["env"]["OTL_PIPELINE_CONFIG"], config_path)

    def test_disabled_mode_skips_every_workbook_stage_but_keeps_invoices(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        invoice_path = Path(directory.name) / "invoice.pdf"
        invoice_path.write_bytes(b"stub")
        report_result = SimpleNamespace(token_usage=None, elapsed_minutes=None)
        invoice_evidence = SimpleNamespace(
            extraction_method="text",
            n_pages=1,
            extraction=SimpleNamespace(line_items=[]),
            ties_to_amount="supporting",
        )
        forbidden = (
            "resolve_skill", "collect_wbs_candidates", "select_wbs_focus", "extract_excel_full",
            "extract_excel_bounded", "interpret_attachment", "sheets_to_json", "apply_skill",
        )

        patches = [patch.object(extractor_run, name) for name in forbidden]
        mocks = [p.start() for p in patches]
        self.addCleanup(lambda: [p.stop() for p in patches])
        for mocked in mocks:
            mocked.side_effect = AssertionError("workbook stage called while Excel parsing disabled")

        with (
            patch.object(extractor_run, "extract_exception_description", return_value="Auditor text"),
            patch.object(extractor_run, "extract_rebuttal_dialogue", return_value="Operator named ABC"),
            patch.object(extractor_run, "split_rebuttal_dialogue", return_value=[]),
            patch.object(extractor_run, "extract_invoice", return_value={"file_name": "invoice.pdf"}),
            patch.object(extractor_run, "build_invoice_evidence",
                         return_value=invoice_evidence) as build_invoice,
            patch.object(extractor_run, "synthesize", return_value=report_result) as synthesize,
            patch.object(extractor_run, "TokenUsageReport") as token_report,
            redirect_stdout(io.StringIO()),
        ):
            token_report.model_validate.return_value = None
            result = extractor_run.build_report(
                Path("report.pdf"), [Path("large.xlsx")], "0040", "Labor", "case",
                invoice_paths=[invoice_path], auto_invoices=False, report_text="report text",
                excel_parsing_enabled=False,
            )

        self.assertIs(result, report_result)
        self.assertFalse(build_invoice.call_args.kwargs["excel_parsing_enabled"])
        self.assertIn("PRIOR-ROUND REBUTTAL DIALOGUE", build_invoice.call_args.args[0])
        self.assertEqual(synthesize.call_args.args[1], [])
        self.assertFalse(synthesize.call_args.kwargs["excel_parsing_enabled"])

    def test_non_workbook_synthesis_records_mode_and_avoids_false_absence_claim(self) -> None:
        synthesis = SynthesisResult(
            likely_basis_for_amount=[],
            synthesis="Invoice amounts may contribute, but the workbook bridge was not analyzed.",
            open_questions=["Review the workbook bridge."],
        )
        meta = {
            "case_id": "GreatWhite_2024_0040",
            "exception_code": "0040",
            "title": "Labor",
            "year": "2024",
            "asset": "GreatWhite",
            "exception_category": "Labor",
            "disputed_amount": 100.0,
            "currency": "USD",
        }
        with patch.object(explain, "structured_call", return_value=synthesis) as structured:
            report = explain.synthesize(
                "Auditor description", [], meta, excel_parsing_enabled=False,
            )

        system, user = structured.call_args.args[1:3]
        self.assertFalse(report.excel_parsing_enabled)
        self.assertTrue(system.startswith(explain.SYSTEM_NON_WORKBOOK_SYNTH))
        self.assertNotIn(explain.SYSTEM_SYNTH, system)
        self.assertNotIn("overrides any instruction above", system)
        self.assertIn("workbook parsing disabled by configuration", user)
        self.assertNotIn("auditor provided NO Excel evidence", user)

    def test_synthesis_prioritizes_the_selected_auditor_reply_over_raw_dialogue(self) -> None:
        synthesis = SynthesisResult(
            likely_basis_for_amount=[], synthesis="Auditor position considered.", open_questions=[],
        )
        meta = {
            "case_id": "GreatWhite_2024_0040", "exception_code": "0040", "title": "Labor",
        }
        reply = {
            "rebuttal_cycle": 2,
            "date": "2024-04-15",
            "text": "The auditor still requests the payroll support.",
        }
        with patch.object(explain, "structured_call", return_value=synthesis) as structured:
            explain.synthesize(
                "Auditor description", [], meta,
                operator_dialogue="Operator says the support was already provided.",
                latest_auditor_reply=reply,
            )

        user = structured.call_args.args[2]
        self.assertIn("LATEST AUDITOR REPLY FOR THIS CYCLE", user)
        self.assertIn("The auditor still requests the payroll support.", user)
        self.assertIn("prioritize it over any conflicting attribution", user)
        self.assertIn("Operator says the support was already provided.", user)

    def test_pdf_auditor_reply_is_the_later_cycle_fallback(self) -> None:
        entries = [
            {"party": "operator", "reply": "The operator's first response."},
            {"party": "auditor", "reply": "The auditor's earlier request."},
            {"party": "operator", "reply": "The operator's follow-up."},
            {"party": "auditor", "reply": "The auditor's latest request."},
        ]

        reply = extractor_run._latest_pdf_auditor_reply(entries, 3)

        self.assertEqual(3, reply["rebuttal_cycle"])
        self.assertEqual("The auditor's latest request.", reply["text"])
        self.assertEqual("audit_report_pdf", reply["source"])
        self.assertEqual({}, extractor_run._latest_pdf_auditor_reply(entries, 0))

    def test_cycle_catalog_reply_reaches_the_report_builder(self) -> None:
        report_pdf = Path("/corpus/2024/GreatWhite/Rebuttal 2/Audit Report/report.pdf")
        invoice = report_pdf.parent / "0040" / "invoice.pdf"
        cycle = Cycle(number=2, auditor_dir=report_pdf.parent, auditor_report=report_pdf)
        sink = Mock()
        sink.folder_exists.return_value = False
        catalog_reply = {
            "rebuttal_cycle": 2,
            "date": "2024-04-15",
            "text": "The auditor requests payroll support.",
        }

        with (
            patch.object(extractor_run, "discover_rebuttal_cycles", return_value=[cycle]),
            patch.object(extractor_run, "infer_year_asset", return_value=("2024", "GreatWhite")),
            patch.object(extractor_run, "find_exception_evidence", return_value=([], [invoice])),
            patch.object(extractor_run, "exception_folder_is_empty", return_value=False),
            patch.object(extractor_run, "extract_pdf", return_value={"text": "report"}),
            patch.object(extractor_run, "extract_exception_description", return_value="Auditor text"),
            patch.object(extractor_run, "extract_auditor_cited_attachment_names", return_value=[]),
            patch.object(extractor_run, "union_auditor_cited_attachments", side_effect=lambda x, _y: x),
            patch.object(extractor_run, "_latest_catalog_auditor_reply", return_value=catalog_reply),
            patch.object(extractor_run, "build_report", side_effect=SystemExit("stop")) as build_report,
            redirect_stdout(io.StringIO()),
        ):
            extractor_run.run_rebuttal_cycles(
                Path("/corpus/2024/GreatWhite"), "0040", "Labor", "GreatWhite_2024_0040", sink,
                excel_parsing_enabled=False,
            )

        self.assertEqual(catalog_reply, build_report.call_args.kwargs["latest_auditor_reply"])
        self.assertEqual(2, build_report.call_args.kwargs["rebuttal_cycle"])

    def test_pdf_fallback_reaches_attachment_interpretation_and_synthesis(self) -> None:
        auditor_entries = [
            {"party": "operator", "reply": "Operator response."},
            {"party": "auditor", "reply": "Please provide the payroll support."},
        ]
        attachment_contexts: list[str] = []

        def synthesize(description, attachments, meta, **kwargs):
            self.assertEqual("Auditor description", description)
            self.assertEqual("Please provide the payroll support.",
                             kwargs["latest_auditor_reply"]["text"])
            return ExhibitEvidenceReport(
                case_id=meta["case_id"], exception_code=meta["exception_code"],
                title=meta["title"], auditor_description=description,
                attachments=attachments, synthesis="Synthesis.",
            )

        with (
            patch.object(extractor_run, "extract_exception_description",
                         return_value="Auditor description"),
            patch.object(extractor_run, "extract_rebuttal_dialogue", return_value="dialogue"),
            patch.object(extractor_run, "split_rebuttal_dialogue", return_value=auditor_entries),
            patch.object(extractor_run, "resolve_skill", return_value=None),
            patch.object(extractor_run, "extract_excel_bounded", return_value={"sheets": []}),
            patch.object(extractor_run, "interpret_attachment",
                         side_effect=lambda context, *_args, **_kwargs: attachment_contexts.append(context)),
            patch.object(extractor_run, "synthesize", side_effect=synthesize),
            redirect_stdout(io.StringIO()),
        ):
            report = extractor_run.build_report(
                Path("report.pdf"), [Path("evidence.xlsx")], "0040", "Labor", "case",
                auto_invoices=False, report_text="report text", rebuttal_cycle=2,
            )

        self.assertEqual("Synthesis.", report.synthesis)
        self.assertEqual(1, len(attachment_contexts))
        self.assertIn("LATEST AUDITOR REPLY", attachment_contexts[0])
        self.assertIn("Please provide the payroll support.", attachment_contexts[0])

    def test_non_workbook_invoice_prompt_does_not_claim_ledger_reconciliation(self) -> None:
        interpretation = InvoiceInterpretation(
            demonstrates="The invoice identifies the vendor and amount.",
            ties_to_amount="supporting",
        )
        with patch.object(explain, "structured_call", return_value=interpretation) as structured:
            explain.contextualize_invoice(
                "Auditor description", explain.InvoiceExtraction(invoice_number="INV-1"), [],
                excel_parsing_enabled=False,
            )

        system, user = structured.call_args.args[1:3]
        self.assertIn("Workbook parsing was disabled", system)
        self.assertIn("no ledger/workpaper comparison is available", user)

    def test_invoice_only_cycle_is_not_skipped_when_report_listing_is_absent(self) -> None:
        report_pdf = Path("/corpus/2024/GreatWhite/Rebuttal 0/Audit Report/report.pdf")
        invoice = report_pdf.parent / "0040" / "invoice.pdf"
        cycle = Cycle(number=0, auditor_dir=report_pdf.parent, auditor_report=report_pdf)
        sink = Mock()
        sink.folder_exists.return_value = False

        with (
            patch.object(extractor_run, "discover_rebuttal_cycles", return_value=[cycle]),
            patch.object(extractor_run, "infer_year_asset", return_value=("2024", "GreatWhite")),
            patch.object(extractor_run, "find_exception_evidence", return_value=([], [invoice])),
            patch.object(extractor_run, "exception_folder_is_empty", return_value=False),
            patch.object(extractor_run, "extract_pdf", return_value={"text": "report"}),
            patch.object(extractor_run, "report_lists_exception", return_value=False),
            patch.object(extractor_run, "extract_auditor_cited_attachment_names", return_value=[]),
            patch.object(extractor_run, "build_report", side_effect=SystemExit("stop")) as build_report,
            redirect_stdout(io.StringIO()),
        ):
            extractor_run.run_rebuttal_cycles(
                Path("/corpus/2024/GreatWhite"), "0040", "Labor", "GreatWhite_2024_0040", sink,
                excel_parsing_enabled=False,
            )

        build_report.assert_called_once()


if __name__ == "__main__":
    unittest.main()
