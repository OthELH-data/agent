"""Routing coverage for every composite OTL auditor-question combination."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from otl.agents import category_check, orchestrator
from otl.graph import route_from_category_check


class CompositeRoutingTests(unittest.TestCase):
    def test_other_sets_hitl_at_category_check_before_direct_drafter_route(self) -> None:
        extraction = category_check._CategoryExtraction(
            auditor_question=["other"],
            nl_summary="The request does not map to a supported labor-analysis category.",
        )
        with patch.object(category_check.llm, "structured", return_value=extraction):
            result = category_check.category_check_node({"synthesis": {}, "auditor_narrative": "Please address a document request."})

        self.assertEqual(result["auditor_question"], ["other"])
        self.assertTrue(result["hitl_review"])
        self.assertEqual(result["hitl_stage"], "category_check")
        self.assertIn("human clarification", result["hitl_reason"])
        self.assertEqual(route_from_category_check(result), "drafter")

    def test_attestation_only_routes_directly_to_drafter(self) -> None:
        extraction = category_check._CategoryExtraction(
            auditor_question=["attestation_certificate"],
            nl_summary="The auditor requested an attestation certificate and raised no labor-analysis question.",
        )
        with patch.object(category_check.llm, "structured", return_value=extraction):
            result = category_check.category_check_node({"synthesis": {}, "auditor_narrative": "Certificate requested."})

        self.assertTrue(result["attestation_requested"])
        self.assertEqual(result["auditor_question"], ["attestation_certificate"])
        self.assertEqual(route_from_category_check(result), "drafter")

    def test_cycle_zero_prompt_treats_original_attestation_request_as_active_baseline(self) -> None:
        extraction = category_check._CategoryExtraction(
            auditor_question=["attestation_certificate", "additional_support_requested"],
            nl_summary="The active baseline requests a cost certificate and supporting rate records.",
        )
        state = {
            "case": {"rebuttal_cycle": 0},
            "synthesis": {},
            "auditor_narrative": (
                "This was the subject of IR-027 during fieldwork. The Operator is requested to furnish "
                "a third-party cost certificate or other attestation and provide rate support."
            ),
            "latest_auditor_reply": {},
        }
        with patch.object(category_check.llm, "structured", return_value=extraction) as structured:
            result = category_check.category_check_node(state)

        prompt = structured.call_args.args[2]
        self.assertIn("ACTIVE ORIGINAL EXCEPTION BASELINE", prompt)
        self.assertIn("original exception baseline is the active auditor request", prompt)
        self.assertIn("empty field does not make the baseline historical", prompt)
        self.assertIn("An IR number or fieldwork reference", prompt)
        self.assertEqual(
            result["auditor_question"],
            ["attestation_certificate", "additional_support_requested"],
        )

    def test_mixed_attestation_preserves_analytic_routing(self) -> None:
        extraction = category_check._CategoryExtraction(
            auditor_question=["job_title", "attestation_certificate", "other"],
            nl_summary="The auditor disputes the role and separately requests an attestation certificate.",
        )
        with patch.object(category_check.llm, "structured", return_value=extraction):
            result = category_check.category_check_node({"synthesis": {}, "auditor_narrative": "Role and certificate requested."})

        self.assertTrue(result["attestation_requested"])
        self.assertEqual(result["auditor_question"], ["job_title", "attestation_certificate"])
        with patch("otl.graph.node_gate_enabled", return_value=False):
            self.assertEqual(route_from_category_check(result), "jade_analyzer")

    def test_additional_support_only_routes_directly_to_drafter(self) -> None:
        extraction = category_check._CategoryExtraction(
            auditor_question=["additional_support_requested"],
            nl_summary="The auditor currently requests the underlying labor support and an explanatory response.",
        )
        with patch.object(category_check.llm, "structured", return_value=extraction):
            result = category_check.category_check_node({
                "synthesis": {},
                "auditor_narrative": "Please provide the labor ledger support and explain the allocation.",
            })

        self.assertTrue(result["additional_support_requested"])
        self.assertEqual(result["auditor_question"], ["additional_support_requested"])
        self.assertEqual(route_from_category_check(result), "drafter")

    def test_third_party_invoice_request_routes_through_verification(self) -> None:
        extraction = category_check._CategoryExtraction(
            auditor_question=["third_party_invoice_requested"],
            nl_summary="The auditor specifically requests the underlying vendor invoice.",
        )
        with patch.object(category_check.llm, "structured", return_value=extraction):
            result = category_check.category_check_node({
                "synthesis": {},
                "auditor_narrative": "Please provide the underlying third-party invoice.",
            })

        self.assertTrue(result["third_party_invoice_requested"])
        self.assertFalse(result["additional_support_requested"])
        self.assertEqual(result["auditor_question"], ["third_party_invoice_requested"])
        self.assertEqual(route_from_category_check(result), "verify_evidence")
        with patch("otl.graph.node_gate_enabled", return_value=False):
            self.assertEqual(route_from_category_check(result), "drafter")

    def test_mixed_additional_support_preserves_analytic_routing(self) -> None:
        extraction = category_check._CategoryExtraction(
            auditor_question=["location", "additional_support_requested"],
            nl_summary="The auditor disputes location and currently requests supporting records.",
        )
        with patch.object(category_check.llm, "structured", return_value=extraction):
            result = category_check.category_check_node({
                "synthesis": {},
                "auditor_narrative": "Please provide the POB support and explain the charged location.",
            })

        self.assertTrue(result["additional_support_requested"])
        with patch("otl.graph.node_gate_enabled", return_value=False):
            self.assertEqual(route_from_category_check(result), "jade_analyzer")

    def test_cycle_delta_reintroduces_continued_job_title_and_location_grounds(self) -> None:
        extraction = category_check._CategoryExtraction(
            auditor_question=["additional_support_requested"],
            nl_summary="The active reply requests additional supporting documentation.",
        )
        state = {
            "case": {"description": "Original job-title and location exception."},
            "synthesis": {"auditor_description": "Please provide the requested support."},
            "latest_auditor_reply": {"text": "Please provide the additional role and location support."},
            "rebuttal_history": [{"cycle": 0}],
            "rebuttal_current_posture": "The request continues the prior role and location grounds.",
            "rebuttal_specialist_revisit": {
                "location": True,
                "job_title": True,
                "matching": True,
                "supervision": False,
            },
            "rebuttal_specialist_current_grounds": {
                "location": "The requested records continue the prior location-support issue.",
                "job_title": "The requested records continue the prior job-duty issue.",
                "matching": "Role chargeability remains dependent on the requested duty support.",
                "supervision": "",
            },
        }
        with patch.object(category_check.llm, "structured", return_value=extraction):
            result = category_check.category_check_node(state)

        self.assertEqual(
            result["category_check"].auditor_question,
            ["additional_support_requested"],
        )
        self.assertEqual(
            result["auditor_question"],
            ["location", "job_title", "additional_support_requested"],
        )
        self.assertNotIn("hitl_review", result)
        with patch("otl.graph.node_gate_enabled", return_value=False):
            self.assertEqual(route_from_category_check(result), "jade_analyzer")

    def test_prior_categories_are_not_carried_without_current_delta_grounds(self) -> None:
        extraction = category_check._CategoryExtraction(
            auditor_question=["additional_support_requested"],
            nl_summary="The active reply requests generic additional support only.",
        )
        state = {
            "synthesis": {"auditor_description": "Please provide additional support."},
            "latest_auditor_reply": {"text": "Please provide additional support."},
            "rebuttal_history": [{"cycle": 0, "content": "Prior job-title and location findings."}],
            "rebuttal_specialist_revisit": {"location": False, "job_title": False, "matching": False},
            "rebuttal_specialist_current_grounds": {},
        }
        with patch.object(category_check.llm, "structured", return_value=extraction) as structured:
            result = category_check.category_check_node(state)

        self.assertEqual(result["auditor_question"], ["additional_support_requested"])
        self.assertEqual(route_from_category_check(result), "drafter")
        self.assertNotIn("PRIOR_REBUTTAL_CONTEXT", structured.call_args.args[2])

    def test_category_prompt_distinguishes_current_request_from_historical_ir_reference(self) -> None:
        extraction = category_check._CategoryExtraction(
            auditor_question=["other"],
            nl_summary="The narrative only references a completed historical IR exchange.",
        )
        with patch.object(category_check.llm, "structured", return_value=extraction) as structured:
            result = category_check.category_check_node({
                "synthesis": {},
                "auditor_narrative": "IR-024 was answered during fieldwork.",
            })
        user_prompt = structured.call_args.args[2]
        self.assertIn("does not make the request historical", user_prompt)
        self.assertIn("explicitly shows the request was satisfied", user_prompt)
        self.assertEqual(result["auditor_question"], ["other"])

    def test_category_check_keeps_all_recognized_composite_questions(self) -> None:
        extraction = SimpleNamespace(
            auditor_question=["hierarchy", "location", "job_title", "location", "unsupported"],
            nl_summary="The exception combines location and hierarchy challenges that require separate specialist review.",
        )
        with patch.object(category_check.llm, "structured", return_value=extraction):
            result = category_check.category_check_node({"synthesis": {}})

        self.assertEqual(result["auditor_question"], ["hierarchy", "location", "job_title"])

    def test_every_nonempty_question_combination_has_an_ordered_route(self) -> None:
        expected_targets = {
            frozenset({"job_title"}): ["job_title", "matching"],
            frozenset({"location"}): ["location", "job_title", "matching"],
            frozenset({"hierarchy"}): ["supervision"],
            frozenset({"job_title", "location"}): ["location", "job_title", "matching"],
            frozenset({"job_title", "hierarchy"}): ["job_title", "supervision", "matching"],
            frozenset({"location", "hierarchy"}): [
                "location", "job_title", "supervision", "matching"
            ],
            frozenset({"job_title", "location", "hierarchy"}): [
                "location", "job_title", "supervision", "matching"
            ],
        }

        for questions, route in expected_targets.items():
            question_list = list(questions)
            state = {"auditor_question": question_list, "agent_status": {}}
            run_counts = {}

            self.assertEqual(orchestrator._relevant_targets(question_list), route)
            for target in route:
                self.assertEqual(orchestrator._adaptive_prescribed_target(state, run_counts), target)
                run_counts[target] = 1

            self.assertEqual(orchestrator._adaptive_prescribed_target(state, run_counts), "")
            self.assertEqual(
                orchestrator._invoke_job_title_after_location(state), "location" in questions
            )

    def test_hierarchy_composite_pins_supervision_before_matching_after_inconclusive_title(self) -> None:
        state = {
            "auditor_question": ["job_title", "hierarchy"],
            "agent_run_counts": {"job_title": 1},
            "agent_status": {"job_title": {"status": "inconclusive"}},
        }

        self.assertTrue(
            orchestrator._invoke_supervision_before_matching(state, state["agent_run_counts"])
        )

    def test_technical_role_shortcut_waits_for_hierarchy_review(self) -> None:
        state = {
            "auditor_question": ["job_title", "hierarchy"],
            "agent_run_counts": {"job_title": 1, "matching": 1},
            "matching": {"identified_cfr_category": ""},
            "job_title": {"technical_job_title": True},
            "agent_status": {"supervision": {"status": "inconclusive"}},
        }

        self.assertFalse(orchestrator._technical_role_override(state))
        state["agent_status"]["supervision"]["status"] = "objective_met"
        self.assertTrue(orchestrator._technical_role_override(state))


if __name__ == "__main__":
    unittest.main()
