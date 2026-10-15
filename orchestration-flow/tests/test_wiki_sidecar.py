"""The precedent wiki is a terminal report sidecar, never a drafting input."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from otl.agents.drafter import _nl_summary_digest
from otl.agents.wiki import _WikiSelection, wiki_node
from otl.graph import build_otl_graph
from otl.output_writer import assemble_report
from wiki import kb_fs


class WikiSidecarTests(unittest.TestCase):
    def test_empty_wiki_is_valid_and_skips_llm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "index.md").write_text("# Index\n\n_(none yet)_\n", encoding="utf-8")
            for section in kb_fs.SECTIONS:
                (root / section).mkdir()

            with patch.object(kb_fs, "KB", root), patch(
                "otl.agents.wiki.llm.structured", side_effect=AssertionError("LLM called")
            ):
                result = wiki_node({"case": {"case_id": "case-1"}})

        self.assertEqual(result["wiki"].status, "empty")
        self.assertEqual(result["wiki"].precedent_strength, "none")
        self.assertEqual(result["wiki"].precedent_context, "")

    def test_populated_wiki_returns_validated_pages_as_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for section in kb_fs.SECTIONS:
                (root / section).mkdir()
            (root / "index.md").write_text(
                "# Index\n\n## Types\n- `otl-labor`: offsite technical labor\n",
                encoding="utf-8",
            )
            (root / "types" / "otl-labor.md").write_text(
                "---\ncategory: labor\n---\n# OTL labor\nHistorical pattern.",
                encoding="utf-8",
            )
            selection = _WikiSelection(
                types=["otl-labor", "invented-page"],
                precedent_strength="strong",
                match_reason="same OTL labor pattern",
            )
            state = {
                "case": {"case_id": "case-2", "asset": "Mars", "year": "2025"},
                "intake": {"nl_summary": "Offsite technical labor charge.", "error_code": "", "code_family": "otl_labor"},
                "rebuttal_current_posture": "Offsite technical labor charge.",
                "auditor_question": ["location", "job_title"],
            }

            with patch.object(kb_fs, "KB", root), patch(
                "otl.agents.wiki.llm.structured", return_value=selection
            ):
                result = wiki_node(state)

        wiki = result["wiki"]
        self.assertEqual(wiki.status, "matched")
        self.assertEqual(wiki.precedent_strength, "strong")
        self.assertEqual(wiki.selected_pages["types"], ["otl-labor"])
        self.assertNotIn("invented-page", str(wiki.selected_pages))
        self.assertIn("Historical pattern", wiki.precedent_context)

    def test_wiki_state_is_explicitly_excluded_from_drafter_digest(self) -> None:
        sentinel = "WIKI-MUST-NOT-REACH-DRAFTER"
        digest = _nl_summary_digest({
            "wiki": {"precedent_context": sentinel, "match_reason": sentinel},
            "intake": {"nl_summary": "Current-case facts."},
            "rebuttal_current_posture": "Current-case facts.",
        })

        self.assertNotIn(sentinel, str(digest))

    def test_report_surfaces_wiki_and_drafter_outputs_side_by_side(self) -> None:
        report = assemble_report(
            {
                "case": {"case_id": "case-3", "exc_id": "3", "year": "2025", "asset": "Mars"},
                "wiki": {
                    "status": "matched",
                    "precedent_strength": "weak",
                    "precedent_context": "independent precedent",
                },
                "drafter": {
                    "disposition": "deny",
                    "draft_response": "current-case response",
                    "co_owner_response": "send-ready response",
                },
            },
            run_id="run-1",
            elapsed_minutes=0.1,
            token_usage=None,
            trace_events=[],
        )
        payload = report.model_dump()

        self.assertEqual(payload["wiki_output"]["precedent_context"], "independent precedent")
        self.assertEqual(payload["draft_response"], "current-case response")
        self.assertEqual(payload["stages"]["wiki"]["status"], "matched")
        self.assertEqual(payload["stages"]["drafter"]["disposition"], "deny")

    def test_only_orchestrator_terminal_routes_enter_agreement_then_wiki_before_drafter(self) -> None:
        graph = build_otl_graph()
        edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}

        self.assertIn(("orchestrator", "agreement_retriever"), edges)
        self.assertIn(("agreement_retriever", "wiki"), edges)
        self.assertIn(("wiki", "drafter"), edges)
        self.assertNotIn(("agreement_retriever", "drafter"), edges)
        self.assertIn(("verify_evidence", "drafter"), edges)
        self.assertIn(("jade_analyzer", "drafter"), edges)
        self.assertNotIn(("orchestrator", "drafter"), edges)

    def test_mixed_attestation_and_job_title_reaches_wiki_and_agreement(self) -> None:
        visits: list[str] = []

        def mark(name: str, update: dict):
            def _node(_state):
                visits.append(name)
                return update
            return _node

        graph = build_otl_graph(nodes={
            "intake": mark("intake", {"intake": {"nl_summary": "Current request."}}),
            "rebuttal_history": mark("history", {"rebuttal_history": []}),
            "category_check": mark("category", {
                "auditor_question": ["job_title", "attestation_certificate"],
                "attestation_requested": True,
            }),
            "jade_analyzer": mark("jade", {"analysis_blocked": False}),
            "orchestrator": mark("orchestrator", {"next_agent": "drafter"}),
            "wiki": mark("wiki", {"wiki": {"nl_summary": "Precedent checked."}}),
            "agreement_retriever": mark("agreement", {
                "agreement_retriever": {"nl_summary": "Agreement authority selected."},
            }),
            "drafter": mark("drafter", {"drafter": {"disposition": "needs_evidence"}}),
        })

        with patch("otl.graph.node_gate_enabled", return_value=False):
            graph.invoke(
                {"case": {"case_id": "case-1"}},
                config={"configurable": {"thread_id": "mixed-attestation-route"}},
            )

        self.assertEqual(
            visits,
            ["intake", "history", "category", "jade", "orchestrator", "agreement", "wiki", "drafter"],
        )


if __name__ == "__main__":
    unittest.main()
