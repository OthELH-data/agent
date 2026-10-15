from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from otl.agents import orchestrator
from otl.agents._common import (active_auditor_context_block, case_context_block,
                                rebuttal_context_block)
from otl.agents.rebuttal_history import (
    _COMPACTION_SYSTEM,
    _CycleDeltaAssessment,
    _SpecialistAssessment,
    rebuttal_history_node,
)
from otl.graph import (
    build_otl_graph,
    route_from_category_check,
    route_from_verify_evidence,
)
from otl.llm_adapter import tier_for
from shared.rebuttal_history import (
    compact_rebuttal_history,
    initialize_cycle_analysis,
    load_rebuttal_history,
    write_agent_analysis,
)


CASE = {
    "case_id": "GreatWhite_2024_0044",
    "year": "2024",
    "asset": "GreatWhite",
    "exc_id": "0044",
}


class RebuttalHistoryTests(unittest.TestCase):
    def _configure_local_history(self, directory: str | Path) -> None:
        patcher = patch.dict(os.environ, {
            "OUTPUT_TARGET": "local",
            "LOCAL_REBUTTAL_FOLDER": str(directory),
        })
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_rebuttal_compaction_uses_medium_tier(self) -> None:
        self.assertEqual(tier_for("rebuttal_compaction"), "med")
        self.assertEqual(tier_for("rebuttal_history"), "high")

    def test_cycle_zero_uses_intake_findings_as_current_posture_without_delta_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._configure_local_history(directory)
            state = {
                "case": {**CASE, "rebuttal_cycle": 0},
                "intake": {"nl_summary": "Initial exception posture from intake."},
            }
            with patch("otl.agents.rebuttal_history.llm.structured",
                       side_effect=AssertionError("delta LLM called for cycle zero")):
                update = rebuttal_history_node(state)

            self.assertEqual(update["rebuttal_current_posture"],
                             "Initial exception posture from intake.")

    def test_cycle_zero_active_context_does_not_treat_empty_reply_as_no_request(self) -> None:
        state = {
            "case": {**CASE, "rebuttal_cycle": 0},
            "auditor_narrative": "Please furnish a third-party cost certificate or credit the account.",
            "latest_auditor_reply": {},
        }

        block = active_auditor_context_block(state)

        self.assertIn("original exception baseline is the active auditor request", block)
        self.assertIn("empty field does not make the baseline historical", block)
        self.assertIn("Please furnish a third-party cost certificate", block)
        self.assertIn("(none supplied — apply the chronology rule above)", block)

    def test_shared_specialist_context_contains_active_auditor_framing(self) -> None:
        block = case_context_block({
            "case": {**CASE, "rebuttal_cycle": 1},
            "auditor_narrative": "Original location challenge.",
            "latest_auditor_reply": {"text": "The location challenge remains open."},
        })

        self.assertIn("<ACTIVE_AUDITOR_REQUEST_CONTEXT>", block)
        self.assertIn("Read it together with the active original baseline", block)
        self.assertIn("Original location challenge", block)
        self.assertIn("The location challenge remains open", block)

    def test_later_cycle_uses_delta_current_posture_findings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._configure_local_history(directory)
            initialize_cycle_analysis(CASE, cycle=0)
            state = {
                "case": {**CASE, "rebuttal_cycle": 1},
                "intake": {"nl_summary": "Current reply requests support."},
            }
            assessment = _CycleDeltaAssessment(
                current_posture_nl_summary=(
                    "The prior denial remains material, while the current reply requests new location support."
                )
            )
            with patch("otl.agents.rebuttal_history.llm.context_window_for_node",
                       return_value=1_000_000), patch(
                "otl.agents.rebuttal_history.llm.structured", return_value=assessment
            ):
                update = rebuttal_history_node(state)

            self.assertEqual(update["rebuttal_current_posture"],
                             assessment.current_posture_nl_summary)

    def test_verify_evidence_gate_can_be_bypassed_to_jade(self) -> None:
        with patch("otl.graph.node_gate_enabled", return_value=False) as enabled:
            self.assertEqual(route_from_category_check({"auditor_question": ["location"]}), "jade_analyzer")
        enabled.assert_called_once_with("verify_evidence")

    def test_verify_evidence_gate_runs_when_enabled(self) -> None:
        with patch("otl.graph.node_gate_enabled", return_value=True):
            self.assertEqual(route_from_category_check({"auditor_question": ["location"]}), "verify_evidence")

    def test_loads_all_prior_cycles_but_excludes_active_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._configure_local_history(directory)
            for cycle in (0, 1, 2):
                path = Path(initialize_cycle_analysis(CASE, cycle=cycle))
                path.write_text(path.read_text() + f"\ncycle-{cycle}-finding\n", encoding="utf-8")

            history = load_rebuttal_history(CASE, current_cycle=2)

            self.assertEqual([record["cycle"] for record in history], [0, 1])
            self.assertNotIn("cycle-2-finding", str(history))

    def test_compaction_triggers_at_85_percent_and_is_cache_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._configure_local_history(directory)
            records = [{"cycle": 0, "path": "r0", "content": "finding disposition " * 100}]
            calls: list[str] = []

            def compact(source: str, target: int) -> str:
                calls.append(source)
                return f"## Cycle 0\n- Main finding retained.\n- Final disposition relationship retained.\n- target {target}"

            kwargs = dict(
                case=CASE,
                context_window=100,
                compactor_context_window=1_000,
                token_counter=lambda text: len(text.split()),
                llm_compactor=compact,
                ratio=0.85,
            )
            first = compact_rebuttal_history(records, **kwargs)
            second = compact_rebuttal_history(records, **kwargs)

            self.assertTrue(first["compacted"])
            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(len(calls), 1)
            self.assertEqual(first["context"], second["context"])

    def test_history_below_threshold_does_not_call_llm(self) -> None:
        result = compact_rebuttal_history(
            [{"cycle": 0, "path": "r0", "content": "small finding"}],
            case=CASE,
            context_window=1_000,
            compactor_context_window=1_000,
            token_counter=lambda text: len(text.split()),
            llm_compactor=lambda _source, _target: self.fail("compactor called"),
            ratio=0.85,
        )
        self.assertFalse(result["compacted"])

    def test_exactly_85_percent_triggers_compaction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._configure_local_history(directory)
            calls: list[str] = []

            def count(text: str) -> int:
                return 2 if text.startswith("COMPRESSED") else 85

            result = compact_rebuttal_history(
                [{"cycle": 0, "path": "r0", "content": "boundary finding"}],
                case=CASE,
                context_window=100,
                compactor_context_window=1_000,
                token_counter=count,
                llm_compactor=lambda source, _target: calls.append(source) or "COMPRESSED history",
                ratio=0.85,
            )
            self.assertTrue(result["compacted"])
            self.assertEqual(len(calls), 1)

    def test_specialist_sections_are_upserted_not_duplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._configure_local_history(directory)
            path = initialize_cycle_analysis(CASE, cycle=1)
            state = {"rebuttal_analysis_path": path}

            write_agent_analysis(
                state, "location", {"location": {"nl_summary": "Initial review found A absent from the POB evidence.", "found_in_pob": {"A": False}}},
            )
            write_agent_analysis(
                state, "location", {"location": {"nl_summary": "Updated review found A present in the POB evidence.", "found_in_pob": {"A": True}}},
            )
            text = Path(path).read_text(encoding="utf-8")

            self.assertEqual(text.count("<!-- rebuttal-agent:location:start -->"), 1)
            self.assertNotIn("- Findings: Initial review found A absent", text)
            self.assertIn("- Summary: Updated review found A present", text)
            self.assertIn('"A": true', text)

    def test_agent_history_sections_are_tight_and_report_omissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._configure_local_history(directory)
            path = initialize_cycle_analysis(CASE, cycle=1)
            write_agent_analysis(
                {"rebuttal_analysis_path": path},
                "jade_analyzer",
                {"jade_analyzer": {
                    "nl_summary": "Long finding " * 1_000,
                    "charged_amount_by_person": {
                        f"Person {index}": index * 100.0 for index in range(80)
                    },
                    "major_construction_by_person": {
                        f"Person {index}": False for index in range(80)
                    },
                }},
            )

            text = Path(path).read_text(encoding="utf-8")
            self.assertLess(len(text), 12_000)
            self.assertIn("chars omitted", text)
            self.assertIn("_entries_omitted", text)

    def test_cycle_markdown_contains_original_exception_and_auditor_only_rebuttal_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._configure_local_history(directory)
            path = initialize_cycle_analysis(
                {**CASE, "description": "Original auditor exception.",
                 "amount": "47,365.25", "short_description": "Affiliate labor"},
                cycle=2,
                latest_auditor_reply={
                    "role": "Auditor", "date": "2025-02-01", "text": "Latest auditor reply."
                },
            )

            text = Path(path).read_text(encoding="utf-8")
            self.assertIn("## Original Exception Details", text)
            self.assertIn("Original auditor exception.", text)
            self.assertIn("## Newest Auditor Response for This Cycle", text)
            self.assertIn("Latest auditor reply.", text)

    def test_drafter_write_records_bounded_agent_generated_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._configure_local_history(directory)
            path = initialize_cycle_analysis(CASE, cycle=1)
            write_agent_analysis(
                {"rebuttal_analysis_path": path},
                "drafter",
                {"drafter": {
                    "nl_summary": "The charges are supported by the available evidence and the prior denial remains appropriate.",
                    "disposition": "deny",
                    "draft_response": "No credit is due based on the documented support.",
                }},
            )

            text = Path(path).read_text(encoding="utf-8")
            self.assertIn('"disposition": "deny"', text)
            self.assertIn("## Agent-Generated Rebuttal Response", text)
            self.assertIn("No credit is due based on the documented support.", text)

    def test_every_specialist_can_write_its_own_section(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._configure_local_history(directory)
            path = initialize_cycle_analysis(CASE, cycle=1)
            state = {"rebuttal_analysis_path": path}
            specialists = ("jade_analyzer", "location", "job_title", "matching", "supervision")
            for specialist in specialists:
                write_agent_analysis(
                    state, specialist, {specialist: {"nl_summary": f"{specialist} main finding with material support and no unresolved point."}},
                )
            text = Path(path).read_text(encoding="utf-8")
            for specialist in specialists:
                self.assertIn(f"## Agent: {specialist}", text)
                self.assertIn(f"{specialist} main finding", text)

    def test_first_node_initializes_active_file_and_compacts_prior_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._configure_local_history(directory)
            prior = Path(initialize_cycle_analysis(CASE, cycle=0))
            prior.write_text(prior.read_text() + ("\nimportant finding and disposition\n" * 30), encoding="utf-8")
            state = {"case": {**CASE, "rebuttal_cycle": 1}, "synthesis": {"rebuttal_cycle": 1}}

            with patch(
                "otl.agents.rebuttal_history.llm.context_window_for_node",
                side_effect=lambda node: 40 if node == "rebuttal_history" else 1_000,
            ), patch(
                "otl.agents.rebuttal_history.llm.estimate_text_tokens",
                side_effect=lambda text: len(text.split()),
            ), patch(
                "otl.agents.rebuttal_history.llm.text",
                return_value="## Cycle 0\n- Main finding.\n- Supported the final deny disposition.",
            ), patch(
                "otl.agents.rebuttal_history.llm.structured",
                return_value=_CycleDeltaAssessment(current_posture_nl_summary="No new specialist ground."),
            ) as compact:
                update = rebuttal_history_node(state)

            self.assertTrue(update["rebuttal_history_compacted"])
            self.assertEqual([item["cycle"] for item in update["rebuttal_history"]], [0])
            self.assertTrue(Path(update["rebuttal_analysis_path"]).is_file())
            self.assertIn("Compacted Prior Rebuttal Analysis", update["rebuttal_context"])
            self.assertEqual(compact.call_count, 1)
            self.assertIn("final disposition", _COMPACTION_SYSTEM.lower())

    def test_history_node_cherry_picks_context_and_suppresses_unchanged_specialist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._configure_local_history(directory)
            prior_path = initialize_cycle_analysis(CASE, cycle=0)
            write_agent_analysis(
                {"rebuttal_analysis_path": prior_path},
                "location",
                {"location": {"nl_summary": "The prior location review found the person was not on the charged asset."}},
            )
            assessment = _CycleDeltaAssessment(
                current_posture_nl_summary="The auditor repeated the amount objection only.",
                specialists={
                    "location": _SpecialistAssessment(
                        current_questioning_ground="No new location ground.",
                        relevant_prior_context="Prior location finding: person was not on the asset; it was not dispositive.",
                        new_information_present=False,
                        should_revisit=False,
                        rationale="No new person, period, location record, or location allegation.",
                    )
                },
            )
            state = {
                "case": {**CASE, "rebuttal_cycle": 1},
                "synthesis": {"rebuttal_cycle": 1, "auditor_description": "Amount repeated."},
                "auditor_narrative": "Amount repeated.",
                "intake": {"nl_summary": "The current submission repeats the amount objection without a new specialist-specific ground."},
            }
            with patch(
                "otl.agents.rebuttal_history.llm.context_window_for_node", return_value=1_000_000
            ), patch("otl.agents.rebuttal_history.llm.structured", return_value=assessment):
                update = rebuttal_history_node(state)

            self.assertFalse(update["rebuttal_specialist_revisit"]["location"])
            self.assertIn("Prior location finding", update["rebuttal_specialist_contexts"]["location"])
            self.assertIn("location", update["rebuttal_reused_nl_summaries"])
            self.assertEqual(update["agent_status"]["location"]["status"], "objective_met")
            active = Path(update["rebuttal_analysis_path"]).read_text(encoding="utf-8")
            self.assertIn("## Rebuttal Cycle Delta", active)
            self.assertIn('"revisit_this_cycle": false', active)

    def test_specialist_prompt_delineates_current_ground_and_only_its_prior_context(self) -> None:
        state = {
            "rebuttal_history": [{"cycle": 0}],
            "rebuttal_current_posture": "A new location period was questioned.",
            "rebuttal_specialist_current_grounds": {
                "location": "New December POB period.",
                "job_title": "No new role question.",
            },
            "rebuttal_specialist_contexts": {
                "location": "LOCATION-ONLY-PRIOR",
                "job_title": "JOB-TITLE-ONLY-PRIOR",
            },
        }
        block = rebuttal_context_block(state, "location")
        self.assertIn("<CURRENT_REBUTTAL_CONTEXT>", block)
        self.assertIn("New December POB period.", block)
        self.assertIn("LOCATION-ONLY-PRIOR", block)
        self.assertNotIn("JOB-TITLE-ONLY-PRIOR", block)

    def test_orchestrator_does_not_reinvoke_unchanged_prior_specialists(self) -> None:
        state = {
            "auditor_question": ["location"],
            "rebuttal_history": [{"cycle": 0}],
            "rebuttal_specialist_revisit": {
                "location": False,
                "job_title": False,
                "matching": False,
            },
            "rebuttal_reused_nl_summaries": {
                "location": "Prior location finding.",
                "job_title": "Prior role finding.",
                "matching": "Prior matching finding.",
            },
            "agent_status": {
                "jade_analyzer": {"status": "objective_met", "reason": ""},
                "location": {"status": "objective_met", "reason": "reused"},
                "job_title": {"status": "objective_met", "reason": "reused"},
                "matching": {"status": "objective_met", "reason": "reused"},
            },
            "agent_run_counts": {},
            "orchestrator_hops": 0,
        }
        with patch("otl.agents.orchestrator.llm.structured", side_effect=AssertionError("LLM called")):
            update = orchestrator.orchestrator_node(state)
        self.assertEqual(update["next_agent"], "drafter")
        self.assertEqual(update["final_mode"], "terminal")
        self.assertEqual(update["agent_run_counts"], {})

    def test_orchestrator_allows_specialist_when_cycle_has_new_ground(self) -> None:
        state = {
            "auditor_question": ["location"],
            "rebuttal_history": [{"cycle": 0}],
            "rebuttal_specialist_revisit": {
                "location": True,
                "job_title": False,
                "matching": False,
            },
            "agent_status": {"jade_analyzer": {"status": "objective_met", "reason": ""}},
            "agent_run_counts": {},
            "orchestrator_hops": 0,
        }
        update = orchestrator.orchestrator_node(state)
        self.assertEqual(update["next_agent"], "location")
        self.assertEqual(update["agent_run_counts"]["location"], 1)

    def test_orchestrator_can_add_context_beside_cherry_picked_history(self) -> None:
        state = {
            "auditor_question": ["location"],
            "rebuttal_history": [{"cycle": 0}],
            "rebuttal_current_posture": "The auditor submitted a new December manifest.",
            "rebuttal_specialist_current_grounds": {
                "location": "December location coverage is newly disputed."
            },
            "rebuttal_specialist_revisit": {"location": True, "job_title": False, "matching": False},
            "rebuttal_specialist_contexts": {"location": "Prior location finding for November."},
            "agent_status": {
                "jade_analyzer": {"status": "inconclusive", "reason": "non-transport ambiguity"}
            },
            "agent_run_counts": {},
            "orchestrator_hops": 0,
        }
        decision = orchestrator._OrchestratorDecision(
            next_agent="location",
            instruction="Resolve the new December coverage.",
            context_to_pass="Compare the newly filed manifest with the current charged period.",
            rationale="New location evidence requires review.",
            nl_summary="New December evidence requires location analysis before the case can be resolved.",
        )
        with patch("otl.agents.orchestrator.llm.structured", return_value=decision):
            update = orchestrator.orchestrator_node(state)

        directive = update["agent_instructions"]["location"]
        self.assertIn("Resolve the new December coverage.", directive)
        self.assertIn("December location coverage is newly disputed.", directive)
        self.assertIn("Compare the newly filed manifest", directive)
        self.assertNotIn("Prior location finding for November.", directive)
        self.assertIn("Prior location finding for November.", rebuttal_context_block(state, "location"))

    def test_unchanged_prior_jade_finding_skips_jade_spine_node(self) -> None:
        state = {
            "evidence_gate_decision": {"triggered": False},
            "rebuttal_history": [{"cycle": 0}],
            "rebuttal_specialist_revisit": {"jade_analyzer": False},
        }
        self.assertEqual(route_from_verify_evidence(state), "orchestrator")
        state["rebuttal_specialist_revisit"]["jade_analyzer"] = True
        self.assertEqual(route_from_verify_evidence(state), "jade_analyzer")

    def test_graph_wrapper_writes_each_specialist_during_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._configure_local_history(directory)
            path = initialize_cycle_analysis(CASE, cycle=0)
            routes = iter(("location", "job_title", "matching", "supervision", "drafter"))

            def stage(name: str, **values):
                return lambda _state: {name: {"nl_summary": f"{name} finding with material support and no unresolved point.", **values}}

            nodes = {
                "rebuttal_history": lambda _state: {
                    "rebuttal_cycle": 0,
                    "rebuttal_history": [],
                    "rebuttal_context": "",
                    "rebuttal_analysis_path": path,
                },
                "intake": stage("intake"),
                "category_check": lambda _state: {
                    "category_check": {"nl_summary": "Location and hierarchy questions were identified.",
                                       "auditor_question": ["location", "hierarchy"]},
                    "auditor_question": ["location", "hierarchy"],
                },
                "verify_evidence": lambda _state: {
                    "verify_evidence": {"nl_summary": "The cited evidence meets the required deterministic gate.",
                                        "evidence_gate_decision": {"triggered": False}},
                    "evidence_gate_decision": {"triggered": False},
                },
                "jade_analyzer": lambda _state: {
                    "jade_analyzer": {"nl_summary": "The ledger reconstruction supports the questioned amount of 10.", "amount_questioned": 10},
                    "analysis_blocked": False,
                },
                "orchestrator": lambda _state: {"next_agent": next(routes), "final_mode": ""},
                "location": stage("location", found_in_pob={"A": True}),
                "job_title": stage("job_title", technical_job_title=True),
                "matching": stage("matching", identified_cfr_category="Regulatory Activity"),
                "supervision": stage("supervision", credit_amount=0),
                "wiki": stage("wiki", status="empty"),
                "agreement_retriever": stage("agreement_retriever", status="matched"),
                "drafter": stage("drafter", disposition="deny", co_owner_response="No credit is due."),
            }
            graph = build_otl_graph(nodes=nodes)
            with patch("otl.graph.node_gate_enabled", return_value=True):
                graph.invoke(
                    {"run_id": "test", "case": CASE},
                    config={"configurable": {"thread_id": "rebuttal-history-test"},
                            "recursion_limit": 50},
                )
            text = Path(path).read_text(encoding="utf-8")

            for specialist in (
                "category_check", "verify_evidence", "jade_analyzer",
                "orchestrator", "location", "job_title", "matching", "supervision", "wiki",
                "agreement_retriever", "drafter",
            ):
                self.assertIn(f"## Agent: {specialist}", text)
            self.assertIn("ledger reconstruction supports", text)
            for specialist in ("location", "job_title", "matching", "supervision"):
                self.assertIn(f"{specialist} finding", text)
            self.assertIn("- Summary: drafter finding", text)
            self.assertIn('"disposition": "deny"', text)
            self.assertIn("## Agent-Generated Rebuttal Response", text)
            self.assertIn("No credit is due.", text)


if __name__ == "__main__":
    unittest.main()
