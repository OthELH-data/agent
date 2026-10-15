from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

EXTRACTOR_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXTRACTOR_DIR.parent
for path in (str(REPO_ROOT), str(EXTRACTOR_DIR)):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

import run as extractor_run  # noqa: E402
from extractor.schemas import ExhibitEvidenceReport  # noqa: E402
from shared.hooks import BudgetHookBlocked  # noqa: E402
from shared.rebuttals import Cycle, EvidenceDiscovery  # noqa: E402


class _Sink:
    def __init__(self) -> None:
        self.writes: list[tuple[str, dict]] = []

    def folder_exists(self, _rel_folder: str) -> bool:
        return False

    def write(self, rel_path: str, data: dict) -> str:
        self.writes.append((rel_path, data))
        return f"local/{rel_path}"


class BudgetPartialOutputTests(unittest.TestCase):
    def test_citation_budget_stop_still_writes_partial_cycle(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        report_path = root / "Audit Report.pdf"
        report_path.write_bytes(b"placeholder")
        cycle = Cycle(number=0, auditor_dir=root, auditor_report=report_path)
        evidence = EvidenceDiscovery(cycle_number=0, auditor_dir=root, auditor_report=report_path)
        partial = ExhibitEvidenceReport(
            year="2024", asset="Europa", exception_category="Labor", case_id="europa_2024_0003",
            exception_code="0003", title="Labor", auditor_description="Auditor claim",
        )
        partial.processing.status = "partial"
        partial.processing.stopped_stage = "AuditorCitedAttachmentNames"
        sink = _Sink()
        blocked = BudgetHookBlocked(
            module="extractor", stage="AuditorCitedAttachmentNames", reason="limit_reached",
            limit_usd=7.0, spent_usd=7.2,
        )

        with (
            patch.object(extractor_run, "discover_rebuttal_cycles", return_value=[cycle]),
            patch.object(extractor_run, "find_exception_evidence", return_value=evidence),
            patch.object(extractor_run, "extract_pdf", return_value={"text": "Auditor claim"}),
            patch.object(extractor_run, "report_lists_exception", return_value=True),
            patch.object(extractor_run, "extract_auditor_cited_attachment_names", side_effect=blocked),
            patch.object(extractor_run, "build_report", return_value=partial) as build_report,
            patch.object(extractor_run, "cache_report_metrics", return_value={}),
            patch.object(extractor_run, "mark_metric_write"),
            patch.object(extractor_run, "_print_result"),
        ):
            written = extractor_run.run_rebuttal_cycles(
                root, "0003", "Labor", "europa_2024_0003", sink,
                auto_invoices=False, year="2024", asset="Europa", category="Labor",
            )

        self.assertEqual(written, ["local/2024/Europa/0003/rebuttal_0_synthesis.json"])
        self.assertEqual(len(sink.writes), 1)
        self.assertEqual(build_report.call_args.kwargs["evidence_discovery"]["auditor_cited_attachment_names"], [])


if __name__ == "__main__":
    unittest.main()
