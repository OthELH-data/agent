from __future__ import annotations

import unittest
from unittest.mock import patch

from otl.authority import (OTL_CATEGORY_PROVISIONS, OTL_GOVERNING_INSTRUMENT,
                           agreement_applies_to_asset, agreement_path, registry_path)
from otl.agents.drafter import (
    _AdditionalSupportExtraction,
    _AttestationExtraction,
    _DrafterExtraction,
    _clean_external_text,
    _business_review_overview,
    _compose_ranked_units,
    _coverage_note,
    _nl_summary_digest,
    _publish_response_sections,
    _response_heading,
    _hitl_co_owner,
    _hitl_parts,
    _hitl_user,
    _render_sections,
    _terminal_co_owner,
    _terminal_sections,
    drafter_node,
)


class DrafterPresentationTests(unittest.TestCase):
    def test_response_heading_matrix_matches_publication_format(self) -> None:
        self.assertEqual(_response_heading("accept"), "Credit Granted")
        self.assertEqual(_response_heading("partial"), "Credit Partially Granted")
        self.assertEqual(_response_heading("deny"), "Credit Denied")
        self.assertEqual(
            _response_heading("deny", supporting_documentation_provided=True),
            "Credit Denied - supporting documentation provided",
        )
        self.assertEqual(
            _response_heading("needs_evidence", review_pending=True),
            "Response Withheld - pending information / Under Review",
        )
        self.assertEqual(
            _response_heading("deny", insufficient_auditor_evidence=True),
            "Insufficient Auditor Evidence",
        )

        published = _publish_response_sections(
            "Credit Denied",
            {"Executive Summary": "Summary.", "Conclusion": "No credit is due."},
        )
        self.assertEqual(list(published), ["Credit Denied"])
        self.assertIn("### Executive Summary", published["Credit Denied"])
        self.assertIn("### Conclusion", published["Credit Denied"])

    def test_chargeability_hitl_context_is_business_facing_overview(self) -> None:
        state = {
            "case": {"case_id": "case-1"},
            "chargeability_resolution": {
                "person_results": {
                    "Alex Example": {"disposition": "deny", "amount": 70000, "hooks": ["GR-01"]},
                    "Blair Example": {"disposition": "needs_evidence", "amount": 30000, "hooks": []},
                },
                "resolved_deny_amount": 70000,
                "resolved_accept_amount": 0,
                "unresolved_amount": 30000,
                "unresolved_people": ["Blair Example"],
            },
            "chargeability_hook_evaluations": [{
                "status": "pending", "missing_conditions": ["role_status", "supervision_status"],
            }],
        }
        overview = _business_review_overview(state)
        self.assertEqual(overview["resolved_deny_amount"], 70000)
        self.assertEqual(overview["unresolved_people_or_groups"], ["Blair Example"])
        self.assertEqual(
            overview["remaining_information_overview"],
            ["technical-role classification", "supervision level"],
        )
        prompt = _hitl_user(state, {}, "Some questioned labor remains unresolved.")
        self.assertIn("BUSINESS REVIEW OVERVIEW", prompt)
        self.assertIn("Alex Example", prompt)
        self.assertNotIn("GR-01", prompt)

    def test_additional_support_only_is_direct_hitl_needs_evidence(self) -> None:
        state = {
            "additional_support_requested": True,
            "auditor_question": ["additional_support_requested"],
            "auditor_narrative": "Please provide the labor ledger support and explain the allocation.",
            "case": {"case_id": "GreatWhite_2024_0010"},
        }
        extraction = _AdditionalSupportExtraction(
            response="The requested labor ledger support and allocation explanation will be provided when available."
        )
        with patch("otl.agents.drafter.llm.structured", return_value=extraction) as structured:
            result = drafter_node(state)

        prompt = structured.call_args.args[2]
        self.assertIn("ACTIVE ORIGINAL EXCEPTION BASELINE", prompt)
        self.assertIn("original exception baseline is the active auditor request", prompt)
        self.assertIn("empty field does not make the baseline historical", prompt)
        self.assertEqual(result["drafter"].disposition, "needs_evidence")
        self.assertEqual(result["drafter"].additional_support_response, extraction.response)
        self.assertEqual(
            list(result["drafter"].response_sections),
            ["Response Withheld - pending information / Under Review"],
        )
        self.assertTrue(result["hitl_review"])
        self.assertEqual(result["hitl_stage"], "additional_support_request")

    def test_mixed_additional_support_keeps_terminal_wording_but_stores_needs_evidence(self) -> None:
        state = {
            "additional_support_requested": True,
            "auditor_question": ["job_title", "additional_support_requested"],
            "auditor_narrative": "The role is disputed and the labor support is requested.",
            "final_mode": "terminal",
            "case": {"case_id": "GreatWhite_2024_0010", "asset": "Great White"},
            "job_title": {
                "technical_job_title": True,
                "technical_rationale": "The documented duties are technical engineering work.",
            },
            "matching": {"identified_cfr_category": ""},
        }
        support = _AdditionalSupportExtraction(
            response="The requested labor support and explanatory information will be provided when available."
        )
        terminal = _DrafterExtraction(
            disposition="deny",
            executive_summary="The technical role is chargeable by function.",
            governing_basis="The available findings support direct chargeability.",
            conclusion="No credit is due.",
            nl_summary="The technical role supports denial; the requested support remains pending.",
        )
        with patch("otl.agents.drafter.llm.structured", side_effect=[support, terminal]):
            result = drafter_node(state)

        self.assertEqual(result["drafter"].disposition, "needs_evidence")
        self.assertIn("No credit is due.", result["drafter"].draft_response)
        self.assertIn(support.response, result["drafter"].draft_response)
        self.assertEqual(
            list(result["drafter"].response_sections),
            ["Response Withheld - pending information / Under Review"],
        )
        self.assertTrue(result["hitl_review"])
        self.assertEqual(result["hitl_stage"], "additional_support_request")

    def test_attestation_only_is_direct_hitl_needs_evidence(self) -> None:
        state = {
            "attestation_requested": True,
            "auditor_question": ["attestation_certificate"],
            "auditor_narrative": "Please provide an affiliate labor attestation certificate.",
            "case": {"case_id": "GreatWhite_2024_0010"},
        }
        extraction = _AttestationExtraction(
            response="The requested affiliate labor attestation certificate will be provided when available."
        )
        with patch("otl.agents.drafter.llm.structured", return_value=extraction):
            result = drafter_node(state)

        self.assertEqual(result["drafter"].disposition, "needs_evidence")
        self.assertEqual(result["drafter"].attestation_response, extraction.response)
        self.assertTrue(result["hitl_review"])
        self.assertEqual(result["hitl_stage"], "attestation_certificate")

    def test_mixed_attestation_keeps_terminal_wording_but_stores_needs_evidence(self) -> None:
        state = {
            "attestation_requested": True,
            "auditor_question": ["job_title", "attestation_certificate"],
            "auditor_narrative": "The role is disputed and an attestation certificate is requested.",
            "final_mode": "terminal",
            "case": {"case_id": "GreatWhite_2024_0010", "asset": "Great White"},
            "job_title": {
                "technical_job_title": True,
                "technical_rationale": "The documented duties are technical engineering work.",
            },
            "matching": {"identified_cfr_category": ""},
        }
        attestation = _AttestationExtraction(
            response="The requested technical labor attestation certificate will be provided when available."
        )
        terminal = _DrafterExtraction(
            disposition="deny",
            executive_summary="The technical role is chargeable by function.",
            governing_basis="The available findings support direct chargeability.",
            conclusion="No credit is due.",
            nl_summary="The technical role supports denial; the attestation remains pending.",
        )
        with patch("otl.agents.drafter.llm.structured", side_effect=[attestation, terminal]):
            result = drafter_node(state)

        self.assertEqual(result["drafter"].disposition, "needs_evidence")
        self.assertIn("No credit is due.", result["drafter"].draft_response)
        self.assertIn(attestation.response, result["drafter"].draft_response)
        self.assertTrue(result["hitl_review"])
        self.assertEqual(result["hitl_stage"], "attestation_certificate")

    def test_evidence_gate_is_deterministic_and_skips_llm(self) -> None:
        state = {
            "evidence_gate_decision": {
                "triggered": True, "gate_ids": ["ED-03"], "status": "missing",
                "required_evidence": [
                    "personnel-on-board (POB) records for the charge period",
                    "helicopter manifests / flight logs for the charge period",
                ],
            },
            "auditor_question": ["location"],
            "verify_evidence": {"pob_data_found": False, "heli_data_found": False},
            "case": {"case_id": "Tobago_2025_0001"},
        }

        with patch("otl.agents.drafter.llm.structured", side_effect=AssertionError("LLM called")):
            result = drafter_node(state)

        response = result["drafter"].co_owner_response
        self.assertFalse(result["hitl_review"])
        self.assertEqual(result["drafter"].disposition, "deny")
        self.assertEqual(
            list(result["drafter"].response_sections),
            ["Insufficient Auditor Evidence"],
        )
        self.assertIn("This exception is denied", response)
        self.assertIn("personnel-on-board (POB) records", response)
        self.assertIn("helicopter manifests / flight logs", response)
        self.assertNotIn("remains open", response)

    def test_evidence_gate_denies_when_each_individual_location_source_is_missing(self) -> None:
        for pob, heli, absent in (
            (False, True, "personnel-on-board (POB) records"),
            (True, False, "helicopter manifests / flight logs"),
        ):
            with self.subTest(pob=pob, heli=heli):
                state = {
                    "evidence_gate_status": "missing_auditor_evidence",
                    "evidence_gate_decision": {
                        "triggered": True, "gate_ids": ["ED-03"], "status": "missing",
                        "required_evidence": [absent],
                    },
                    "auditor_question": ["location"],
                    "verify_evidence": {"pob_data_found": pob, "heli_data_found": heli},
                    "case": {"case_id": "Tobago_2025_0001"},
                }
                with patch("otl.agents.drafter.llm.structured", side_effect=AssertionError("LLM called")):
                    result = drafter_node(state)

                self.assertEqual(result["drafter"].disposition, "deny")
                self.assertFalse(result["hitl_review"])
                self.assertIn(absent, result["drafter"].draft_response)

    def test_ledger_gate_names_the_ledger_rather_than_pob_records(self) -> None:
        # `jade_analyzer` fails the same gate when the auditor cited no usable labor ledger, and it names
        # the gap itself. The response denies the unsupported exception using THAT, not POB/heli records.
        gap = "the JADE labor ledger detail behind the questioned charge"
        state = {
            "evidence_gap": [gap],
            "evidence_gate_decision": {
                "triggered": True, "gate_ids": ["ED-01"], "status": "missing",
                "required_evidence": [gap],
            },
            "auditor_question": ["location"],
            "verify_evidence": {"pob_data_found": True, "heli_data_found": True},
            "case": {"case_id": "GreatWhite_2023_0003"},
        }

        with patch("otl.agents.drafter.llm.structured", side_effect=AssertionError("LLM called")):
            result = drafter_node(state)

        response = result["drafter"].co_owner_response
        self.assertFalse(result["hitl_review"])
        self.assertEqual(result["drafter"].disposition, "deny")
        self.assertIn("This exception is denied", response)
        self.assertIn(gap, response)
        self.assertNotIn("personnel-on-board", response)
        self.assertNotIn("helicopter", response)
        self.assertNotIn("location evidence", response)

    def test_unreadable_evidence_gate_defers_without_internal_language(self) -> None:
        gap = "the JADE labor ledger detail behind the questioned charge"
        state = {
            "evidence_gate_status": "unreadable_auditor_evidence",
            "evidence_gap": [gap],
            "evidence_gate_decision": {
                "triggered": True, "gate_ids": ["ED-01"], "status": "unreadable",
                "required_evidence": [gap],
            },
            "auditor_question": ["location"],
            "case": {"case_id": "GreatWhite_2023_0003"},
        }

        with patch("otl.agents.drafter.llm.structured", side_effect=AssertionError("LLM called")):
            result = drafter_node(state)

        response = result["drafter"].co_owner_response.lower()
        self.assertEqual(result["drafter"].disposition, "needs_evidence")
        self.assertTrue(result["hitl_review"])
        self.assertEqual(
            list(result["drafter"].response_sections),
            ["Response Withheld - pending information / Under Review"],
        )
        self.assertEqual(result["hitl_stage"], "verify_evidence")
        self.assertIn("could not be opened or accessed", response)
        self.assertIn(gap.lower(), response)
        for internal in ("agent", "model", "pipeline", "system", "hitl"):
            self.assertNotIn(internal, response)

    def test_ed04_denial_suppresses_duplicate_invoice_response_without_hitl(self) -> None:
        state = {
            "evidence_gate_status": "missing_auditor_evidence",
            "evidence_gap": ["Shell invoice required before requesting third-party invoice support"],
            "evidence_gate_decision": {
                "triggered": True,
                "gate_ids": ["ED-04"],
                "status": "missing",
                "required_evidence": [
                    "Shell invoice required before requesting third-party invoice support",
                ],
                "outcome_key": "deny_missing_evidence",
            },
            "third_party_invoice_requested": True,
            "additional_support_requested": False,
            "auditor_question": ["third_party_invoice_requested"],
            "case": {"case_id": "Ursa_2025_0033"},
        }

        with patch("otl.agents.drafter.llm.structured", side_effect=AssertionError("LLM called")):
            result = drafter_node(state)

        self.assertEqual("deny", result["drafter"].disposition)
        self.assertFalse(result["hitl_review"])
        self.assertIn("No credit is due.", result["drafter"].co_owner_response)

    def test_separate_support_request_wins_hitl_but_keeps_no_credit_wording(self) -> None:
        state = {
            "evidence_gate_status": "missing_auditor_evidence",
            "evidence_gate_decision": {
                "triggered": True,
                "gate_ids": ["ED-04"],
                "status": "missing",
                "required_evidence": ["Shell invoice"],
                "outcome_key": "deny_missing_evidence",
            },
            "third_party_invoice_requested": True,
            "additional_support_requested": True,
            "auditor_question": ["third_party_invoice_requested", "additional_support_requested"],
            "case": {"case_id": "Ursa_2025_0033"},
        }
        support = _AdditionalSupportExtraction(
            response="The separately requested supporting schedule will be provided."
        )
        with patch("otl.agents.drafter.llm.structured", return_value=support):
            result = drafter_node(state)

        self.assertEqual("needs_evidence", result["drafter"].disposition)
        self.assertTrue(result["hitl_review"])
        self.assertIn("No credit is due.", result["drafter"].co_owner_response)
        self.assertIn("separately requested", result["drafter"].co_owner_response)

    def test_coverage_note_reads_the_jade_person_list(self) -> None:
        # The honesty mechanism. jade harvested 6 people and the cap examined 2, so a note naming the
        # four not examined is required — reading intake alone here would call the run fully covered.
        state = {
            "jade_analyzer": {
                "people_found_in_ledger": [
                    "Kenneth Oyler", "Sheila Adams", "Nancy Adams",
                    "Rob Vance", "Dale Innes", "Amy Fox",
                ],
            },
        }

        with patch("otl.agents._common._max_people", 2):
            note = _coverage_note(state)

        self.assertIn("PARTIAL COVERAGE", note)
        self.assertIn("only 2 of 6 named people", note)
        self.assertIn("Nancy Adams", note)
        self.assertIn("Amy Fox", note)
        self.assertIn("Sheila Adams", note)

    def test_coverage_note_is_empty_when_the_cap_did_not_bite(self) -> None:
        state = {
            "jade_analyzer": {"people_found_in_ledger": ["Kenneth Oyler", "Sheila Adams"]},
        }
        self.assertEqual(_coverage_note(state), "")

    def test_coverage_note_is_empty_before_jade_runs(self) -> None:
        self.assertEqual(_coverage_note({"intake": {"assets_in_question": ["Great White"]}}), "")

    def test_prompt_digest_excludes_trace_only_fields_and_bounds_long_text(self) -> None:
        state = {
            "jade_analyzer": {
                "nl_summary": "x " * 2000,
                "amount_questioned": 47365.25,
                "resolved_columns": {"wbs": "secret column"},
                "jade_wbs_codes": ["C.US.GWH.PD.21.004.0942A"],
            },
            "matching": {
                "identified_cfr_category": "Regulatory Activity",
                "cfr_clause": "verbatim clause must remain outside the prompt",
                "matched_job_profiles": {"internal": "internal"},
            },
        }

        digest = _nl_summary_digest(state)

        self.assertNotIn("resolved_columns", digest["jade_analyzer"])
        self.assertNotIn("jade_wbs_codes", digest["jade_analyzer"])
        self.assertLessEqual(len(digest["jade_analyzer"]["nl_summary"]), 3_001)
        self.assertNotIn("cfr_clause", digest["matching"])
        self.assertNotIn("matched_job_profiles", digest["matching"])
        self.assertEqual(
            digest["supported_argument_components_unordered"],
            ["amount_reconciliation", "role_authority"],
        )

    def test_terminal_response_has_ranked_sections_and_publication_cleanup(self) -> None:
        ext = _DrafterExtraction(
            disposition="deny",
            executive_summary=(
                "The auditor requests a $47,365.25 credit for labor charged to Great White."
            ),
            amount_analysis=(
                "JADE lines on account A7393160 and WBS C.US.GWH.PD.21.004.0942A reconcile "
                "the questioned amount."
            ),
            evidence_basis=(
                "Jobs_Catalogue identifies the role as Production Technology Lead JG2. "
                "The documented duties are technical in nature."
            ),
            governing_basis=(
                "The role is chargeable by function under the Regulatory Activity OTL provision."
            ),
            conclusion="The charge is proper. No credit is due.",
            co_owner_response=(
                "Exception 0003 is denied. JADE reconciles the amount on account A7393160 and "
                "WBS C.US.GWH.PD.21.004.0942A. Jobs_Catalogue supports the technical role. "
                "No credit is due."
            ),
        )
        state = {
            "jade_analyzer": {"amount_questioned": 47365.25},
            "matching": {"identified_cfr_category": "Regulatory Activity"},
        }

        sections = _terminal_sections(ext, state, "deny")
        response = _terminal_co_owner(ext, sections, "deny", state)
        draft = _render_sections(sections)

        self.assertEqual(
            list(sections),
            ["Executive Summary", "Role and Governing Basis", "Amount", "Conclusion"],
        )
        self.assertIn("## Executive Summary", draft)
        self.assertNotRegex(response, r"(?i)JADE|Jobs_Catalogue|A7393160|C\.US\.GWH|\bJG2\b")
        self.assertNotIn("labor ledger ledger", response.lower())
        self.assertLessEqual(len(response.split()), 220)
        self.assertTrue(response.endswith("This exception is denied. No credit is due."))
        self.assertEqual(response.lower().count("no credit is due"), 1)

    def test_filename_cleanup_keeps_a_grammatical_source_sentence(self) -> None:
        raw = (
            "The JADE ledger (IR 24 Affiliate Coding Error.xlsx, Sheet1) contains "
            "31 lines on account A7393160."
        )

        cleaned = _clean_external_text(raw)

        self.assertEqual(
            cleaned,
            "The source records contain 31 lines on the applicable labor account.",
        )
        self.assertEqual(
            _clean_external_text("This exception questions $47,365.25 in affiliate labor charges."),
            "The auditor requests a $47,365.25 credit for affiliate labor charges.",
        )
        self.assertEqual(
            _clean_external_text(
                "The auditor requests a credit for affiliate labor charged for Kenneth Oyler to the "
                "Great White Joint Account."
            ),
            "The auditor requests a credit for affiliate labor for Kenneth Oyler charged to the "
            "Great White Joint Account.",
        )
        self.assertEqual(
            _clean_external_text(
                "The services are drafting deliverables performed for the benefit of the Joint Property."
            ),
            "The services are drafting deliverables.",
        )
        self.assertEqual(
            _clean_external_text("A reviewer should confirm the role classification."),
            "Confirmation is needed regarding the role classification.",
        )
        self.assertEqual(
            _clean_external_text("The role has no exact match in the role catalogue."),
            "The role has no exact match in the available role documentation.",
        )

    def test_empty_terminal_paragraph_falls_back_to_sections(self) -> None:
        ext = _DrafterExtraction(
            disposition="accept",
            executive_summary="The auditor requests a full adjustment of $148,758.20.",
            amount_analysis="The labor ledger reconciles the questioned amount.",
            evidence_basis="All four challenged employees are above first-level supervision.",
            governing_basis="Above-first-level supervision is recovered through overhead.",
            conclusion="A full credit of $148,758.20 is due.",
        )
        state = {
            "jade_analyzer": {"amount_questioned": 148758.20},
            "supervision": {"credit_amount": 148758.20},
        }

        sections = _terminal_sections(ext, state, "accept")
        response = _terminal_co_owner(ext, sections, "accept", state)

        self.assertIn("$148,758.20", response)
        self.assertIn("above first-level supervision", response)
        self.assertTrue(response.endswith("This exception is accepted."))

        posted = ext.model_copy(update={
            "co_owner_response": "Full credit of $148,758.20 will be posted.",
        })
        posted_response = _terminal_co_owner(posted, sections, "accept", state)
        self.assertIn("A full credit of $148,758.20 is due.", posted_response)
        self.assertNotIn("will be posted", posted_response)

    def test_terminal_response_guarantees_pinpoint_authority_and_date_comparison(self) -> None:
        ext = _DrafterExtraction(
            disposition="deny",
            executive_summary="The auditor requests a $47,365.25 credit for affiliate labor.",
            amount_analysis="The questioned amount reconciles to the labor ledger.",
            evidence_basis=(
                "The role performs technical well-delivery work and is chargeable by function. "
                "Physical location is not dispositive."
            ),
            governing_basis="The role falls within the Regulatory Activity category.",
            argument_priority=["date_coverage", "role_authority", "amount_reconciliation"],
            challenged_premise=(
                "The contrary location window demonstrates that the challenged charges supported "
                "work elsewhere."
            ),
            argument_roles={
                "date_coverage": "direct_refutation",
                "role_authority": "independent_basis",
                "amount_reconciliation": "corroboration",
            },
            argument_analysis={
                "date_coverage": (
                    "This directly answers the challenged location premise because the supplied "
                    "window does not cover the dated charges or the broader charge population."
                ),
                "role_authority": (
                    "The role evidence independently strengthens the result by establishing a "
                    "supported OTL chargeability route."
                ),
                "amount_reconciliation": "The amount reconstruction corroborates the recorded charge.",
            },
            conclusion="No credit is due.",
            co_owner_response=(
                "The auditor requests a $47,365.25 credit for affiliate labor. "
                "The role performs technical well-delivery work and is chargeable by function. "
                "Physical location is not dispositive. No credit is due."
            ),
        )
        state = {
            "jade_analyzer": {
                "amount_questioned": 47365.25,
                "service_months_by_person": {
                    "Kenneth Oyler": [
                        "2022-01", "2022-02", "2022-03", "2022-04", "2022-05", "2022-06",
                        "2022-07", "2022-08", "2022-09", "2022-10", "2022-12",
                    ],
                },
                "service_dates_by_person": {
                    "Kenneth Oyler": ["2022-10-19", "2022-10-20"],
                },
            },
            "location": {
                "start_date": {"Kenneth Oyler": "2022-10-31"},
                "end_date": {"Kenneth Oyler": "2022-11-11"},
            },
            "job_title": {
                "technical_job_title": True,
                "technical_rationale": (
                    "The Lead Production Technologist performs well design, well integrity, and "
                    "production optimization work."
                ),
            },
            "matching": {
                "identified_cfr_category": "Regulatory Activity",
                "governing_instrument": "2019 Chevron GOM Settlement Agreement",
                "technical_employee_section": (
                    "Attachment 1 §2 – Definition of Technical Employees"
                ),
                "identified_settlement_section": "Attachment 1 §3.A – Regulatory Activity",
                "cfr_primary": {"Production Technology Lead": "30 CFR §250.410–418"},
                "cfr_secondary": {"Production Technology Lead": "30 CFR §250.420–428"},
                "cfr_additional": {"Production Technology Lead": "30 CFR §250.730–739"},
                "cfr_source_reference": {
                    "Production Technology Lead": (
                        "§250.410–418 APD; §250.420–428 casing/cementing; §250.730–739 BOP"
                    ),
                },
            },
        }

        sections = _terminal_sections(ext, state, "deny")
        response = _terminal_co_owner(ext, sections, "deny", state)

        self.assertEqual(
            list(sections),
            [
                "Executive Summary", "Date Coverage", "Role and Governing Basis", "Amount",
                "Conclusion",
            ],
        )
        self.assertNotRegex(sections["Executive Summary"], r"(?i)^the auditor")
        self.assertNotRegex(response, r"(?i)\bthe auditor\b")
        for text in (sections["Date Coverage"], response):
            self.assertIn("11 service months", text)
            self.assertIn("October 31, 2022 through November 11, 2022", text)
            self.assertIn("window touches only October 2022", text)
            self.assertIn("1 of 11 charged months", text)
            self.assertIn("no service-month entry for November 2022", text)
            self.assertIn("October 19, 2022 and October 20, 2022", text)
            self.assertIn("before the location window", text)
        for text in (sections["Role and Governing Basis"], response):
            self.assertNotIn("2019 Chevron GOM Settlement Agreement", text)
            self.assertNotIn("Attachment 1 §2", text)
            self.assertNotIn("Attachment 1 §3.A", text)
            self.assertIn("30 CFR §250.410–418", text)
            self.assertIn("30 CFR §250.420–428", text)
            self.assertIn("30 CFR §250.730–739", text)
            self.assertIn("APD", text)
            self.assertIn("casing/cementing", text)
            self.assertIn("BOP", text)
            self.assertIn("offsite technical labor", text)
            self.assertIn("retrieved governing agreements", text)
        self.assertTrue(response.endswith("This exception is denied. No credit is due."))
        self.assertNotIn("—", response)
        self.assertNotIn("Physical location is not dispositive", response)

    def test_terminal_response_surfaces_unselected_cfr_mapping_with_limitation(self) -> None:
        ext = _DrafterExtraction(
            disposition="deny",
            evidence_basis=(
                "The documented duties include well design, well integrity, and production optimization."
            ),
            governing_basis="The documented role is technical in nature.",
            argument_priority=["role_authority"],
            argument_roles={"role_authority": "independent_basis"},
            premise_relationships={"role_authority": "accepts_or_assumes"},
        )
        state = {
            "case": {"asset": "Great White"},
            "job_title": {"technical_job_title": True},
            "matching": {
                "identified_cfr_category": "",
                "cfr_primary": {
                    "Production Technology Lead JG2": "30 CFR §250.410–418",
                },
                "cfr_secondary": {
                    "Production Technology Lead JG2": "30 CFR §250.420–428",
                },
                "cfr_additional": {
                    "Production Technology Lead JG2": "30 CFR §250.730–739",
                },
                "cfr_source_reference": {
                    "Production Technology Lead JG2": (
                        "§250.410–418 APD; §250.420–428 casing/cementing; §250.730–739 BOP"
                    ),
                },
            },
        }

        sections = _terminal_sections(ext, state, "deny")
        response = _terminal_co_owner(ext, sections, "deny", state)

        for text in (sections["Role and Governing Basis"], response):
            self.assertNotIn("Attachment 1 §2", text)
            self.assertIn("30 CFR §250.410–418 (APD)", text)
            self.assertIn("30 CFR §250.420–428 (casing/cementing)", text)
            self.assertIn("30 CFR §250.730–739 (BOP)", text)
            self.assertIn("role context", text)
            self.assertIn("retrieved governing agreements", text)
            self.assertNotIn("Attachment 1 §3.A", text)
            self.assertNotIn("regulatory OTL", text)

    def test_argument_priority_is_case_specific_not_hardcoded_by_evidence_type(self) -> None:
        ext = _DrafterExtraction(
            disposition="deny",
            argument_priority=["role_authority", "date_coverage"],
            evidence_basis="The documented role performs technical engineering work.",
            governing_basis="The role is chargeable under the governing OTL provisions.",
        )
        state = {
            "jade_analyzer": {
                "service_months_by_person": {"Person": ["2022-01"]},
                "service_dates_by_person": {"Person": ["2022-01-01"]},
            },
            "location": {
                "start_date": {"Person": "2022-02-01"},
                "end_date": {"Person": "2022-02-02"},
            },
            "job_title": {"technical_job_title": True},
            "matching": {"identified_cfr_category": "Regulatory Activity"},
        }

        sections = _terminal_sections(ext, state, "deny")

        self.assertEqual(
            list(sections),
            ["Executive Summary", "Role and Governing Basis", "Date Coverage", "Conclusion"],
        )

    def test_drafter_uses_existing_date_summaries_without_upstream_schema_changes(self) -> None:
        ext = _DrafterExtraction(
            disposition="deny",
            date_basis=(
                "The charge records span fiscal periods March through December 2022, while the "
                "location records cover only October 31 through November 11, 2022."
            ),
            evidence_basis="The documented role performs technical engineering work.",
            argument_priority=["date_coverage", "role_authority"],
            argument_roles={
                "date_coverage": "direct_refutation",
                "role_authority": "independent_basis",
            },
            premise_relationships={
                "date_coverage": "limits_scope",
                "role_authority": "accepts_or_assumes",
            },
        )
        state = {
            "jade_analyzer": {
                "nl_summary": "The charge records span fiscal periods 03.2022 through 12.2022.",
                "jade_line_item_texts": ["Person 0322", "Person 1222"],
            },
            "location": {
                "nl_summary": (
                    "The location evidence covers only 10/31/2022 through 11/11/2022, a narrow "
                    "portion of the broader charged period."
                ),
                "start_date": {"Person": "2022-10-31"},
                "end_date": {"Person": "2022-11-11"},
            },
            "job_title": {"technical_job_title": True},
        }

        digest = _nl_summary_digest(state)
        sections = _terminal_sections(ext, state, "deny")
        response = _terminal_co_owner(ext, sections, "deny", state)

        self.assertNotIn("date_coverage_comparison", digest)
        self.assertIn("date_evidence_excerpt", digest)
        self.assertIn("date_coverage", digest["supported_argument_components_unordered"])
        self.assertIn("March through December 2022", response)
        self.assertIn("October 31 through November 11, 2022", response)
        self.assertLess(
            response.index("March through December 2022"),
            response.index("documented role performs technical engineering work"),
        )

    def test_generic_argument_roles_put_direct_refutation_before_independent_basis(self) -> None:
        ext = _DrafterExtraction(
            disposition="deny",
            argument_priority=["role_authority", "date_coverage"],
            argument_roles={
                "role_authority": "independent_basis",
                "date_coverage": "direct_refutation",
            },
            evidence_basis="The documented role performs technical engineering work.",
        )
        state = {
            "jade_analyzer": {
                "service_months_by_person": {"Person": ["2022-01"]},
                "service_dates_by_person": {"Person": ["2022-01-01"]},
            },
            "location": {
                "start_date": {"Person": "2022-02-01"},
                "end_date": {"Person": "2022-02-02"},
            },
            "job_title": {"technical_job_title": True},
            "matching": {"identified_cfr_category": "Regulatory Activity"},
        }

        sections = _terminal_sections(ext, state, "deny")

        self.assertEqual(
            list(sections),
            ["Executive Summary", "Date Coverage", "Role and Governing Basis", "Conclusion"],
        )
    def test_terminal_state_records_case_specific_argument_reasoning(self) -> None:
        ext = _DrafterExtraction(
            disposition="deny",
            executive_summary="The source records support the charge.",
            evidence_basis="The documented role performs technical engineering work.",
            argument_priority=["role_authority", "date_coverage"],
            challenged_premise="The charged work occurred outside the charged activity.",
            argument_roles={
                "role_authority": "independent_basis",
                "date_coverage": "direct_refutation",
            },
            premise_relationships={
                "role_authority": "accepts_or_assumes",
                "date_coverage": "contradicts",
            },
            argument_analysis={
                "role_authority": "The governing rule independently permits the documented work.",
                "date_coverage": "The transaction dates directly contradict the challenged premise.",
            },
        )
        state = {
            "final_mode": "terminal",
            "jade_analyzer": {
                "service_months_by_person": {"Person": ["2022-01"]},
                "service_dates_by_person": {"Person": ["2022-01-01"]},
            },
            "location": {
                "start_date": {"Person": "2022-02-01"},
                "end_date": {"Person": "2022-02-02"},
            },
            "job_title": {"technical_job_title": True},
            "matching": {"identified_cfr_category": "Regulatory Activity"},
        }

        with patch("otl.agents.drafter.llm.structured", return_value=ext):
            result = drafter_node(state)

        drafted = result["drafter"]
        self.assertEqual(
            drafted.challenged_premise,
            "The charged work occurred outside the charged activity.",
        )
        self.assertEqual(drafted.argument_priority, ["date_coverage", "role_authority"])
        self.assertEqual(
            drafted.argument_roles,
            {
                "date_coverage": "direct_refutation",
                "role_authority": "independent_basis",
            },
        )
        self.assertEqual(
            drafted.premise_relationships,
            {
                "date_coverage": "contradicts",
                "role_authority": "accepts_or_assumes",
            },
        )
        self.assertIn("date_coverage", drafted.argument_analysis)

    def test_counterfactual_prevents_relevance_argument_from_becoming_direct_refutation(self) -> None:
        ext = _DrafterExtraction(
            argument_priority=["role_authority", "date_coverage"],
            argument_roles={
                "role_authority": "direct_refutation",
                "date_coverage": "direct_refutation",
            },
            premise_relationships={
                "role_authority": "accepts_or_assumes",
                "date_coverage": "contradicts",
            },
            evidence_basis="The documented role performs technical engineering work.",
        )
        state = {
            "jade_analyzer": {
                "service_months_by_person": {"Person": ["2022-01"]},
                "service_dates_by_person": {"Person": ["2022-01-01"]},
            },
            "location": {
                "start_date": {"Person": "2022-02-01"},
                "end_date": {"Person": "2022-02-02"},
            },
            "job_title": {"technical_job_title": True},
            "matching": {"identified_cfr_category": "Regulatory Activity"},
        }

        sections = _terminal_sections(ext, state, "deny")

        self.assertEqual(
            list(sections),
            ["Executive Summary", "Date Coverage", "Role and Governing Basis", "Conclusion"],
        )

    def test_scope_limitation_is_a_direct_answer_without_hardcoding_evidence_type(self) -> None:
        ext = _DrafterExtraction(
            argument_priority=["role_authority", "date_coverage"],
            argument_roles={
                "role_authority": "independent_basis",
                "date_coverage": "corroboration",
            },
            premise_relationships={
                "role_authority": "accepts_or_assumes",
                "date_coverage": "limits_scope",
            },
            evidence_basis="The documented role performs technical engineering work.",
        )
        state = {
            "jade_analyzer": {
                "service_months_by_person": {"Person": ["2022-01"]},
                "service_dates_by_person": {"Person": ["2022-01-01"]},
            },
            "location": {
                "start_date": {"Person": "2022-02-01"},
                "end_date": {"Person": "2022-02-02"},
            },
            "job_title": {"technical_job_title": True},
            "matching": {"identified_cfr_category": "Regulatory Activity"},
        }

        sections = _terminal_sections(ext, state, "deny")

        self.assertEqual(
            list(sections),
            ["Executive Summary", "Date Coverage", "Role and Governing Basis", "Conclusion"],
        )
        self.assertIn(
            "same conclusion cannot be applied to the full charged population",
            sections["Date Coverage"],
        )

    def test_co_owner_word_target_can_expand_for_required_reasoning(self) -> None:
        required = " ".join(["material"] * 230) + "."

        response = _compose_ranked_units(
            [(required, True)],
            "This exception is denied. No credit is due.",
            220,
        )

        self.assertGreater(len(response.split()), 220)
        self.assertIn(required, response)
        self.assertTrue(response.endswith("This exception is denied. No credit is due."))

    def test_co_owner_does_not_use_arithmetic_corroboration_as_filler(self) -> None:
        ext = _DrafterExtraction(
            disposition="deny",
            evidence_basis="The documented duties are technical.",
            amount_analysis="The questioned amount reconciles exactly.",
            argument_priority=["role_authority", "amount_reconciliation"],
            argument_roles={
                "role_authority": "independent_basis",
                "amount_reconciliation": "corroboration",
            },
        )
        state = {
            "jade_analyzer": {"amount_questioned": 100.0},
            "job_title": {"technical_job_title": True},
        }

        sections = _terminal_sections(ext, state, "deny")
        response = _terminal_co_owner(ext, sections, "deny", state)

        self.assertIn("documented duties are technical", response)
        self.assertNotIn("amount reconciles", response)
        self.assertLess(len(response.split()), 220)

    def test_co_owner_can_include_distinct_fact_bearing_corroboration(self) -> None:
        ext = _DrafterExtraction(
            disposition="deny",
            evidence_basis="The documented duties are technical.",
            amount_analysis="The questioned amount reconciles exactly.",
            argument_priority=[
                "role_authority", "amount_reconciliation", "date_coverage",
            ],
            argument_roles={
                "role_authority": "independent_basis",
                "amount_reconciliation": "corroboration",
                "date_coverage": "corroboration",
            },
            premise_relationships={
                "role_authority": "accepts_or_assumes",
                "amount_reconciliation": "does_not_address",
                "date_coverage": "does_not_address",
            },
        )
        state = {
            "jade_analyzer": {
                "amount_questioned": 100.0,
                "service_months_by_person": {"Person": ["2022-01"]},
                "service_dates_by_person": {"Person": ["2022-01-01"]},
            },
            "location": {
                "start_date": {"Person": "2022-02-01"},
                "end_date": {"Person": "2022-02-02"},
            },
            "job_title": {"technical_job_title": True},
        }

        sections = _terminal_sections(ext, state, "deny")
        response = _terminal_co_owner(ext, sections, "deny", state)

        self.assertLess(
            response.index("documented duties are technical"),
            response.index("labor records span 1 service month"),
        )
        self.assertNotIn("amount reconciles", response)
        self.assertTrue(response.endswith("This exception is denied. No credit is due."))

    def test_authority_registry_covers_every_category_and_source(self) -> None:
        self.assertEqual(OTL_GOVERNING_INSTRUMENT, "2019 Chevron GOM Settlement Agreement")
        self.assertTrue(registry_path().is_file())
        self.assertEqual(
            agreement_path().name,
            "C00025092-001 _ 2019 CVX GOM SETTLEMENT AGREEMENT.pdf",
        )
        self.assertTrue(agreement_applies_to_asset("GreatWhite - Chevron only"))
        self.assertTrue(agreement_applies_to_asset("Perdido Regional Host"))
        self.assertFalse(agreement_applies_to_asset("Unrelated Asset"))
        self.assertEqual(
            set(OTL_CATEGORY_PROVISIONS),
            {
                "Regulatory Activity",
                "Feasibility and Project Studies",
                "Planners & Schedulers, maintenance and turnaround, rig administration",
            },
        )
        for provision in OTL_CATEGORY_PROVISIONS.values():
            self.assertTrue(provision["section"].startswith("Attachment 1 §3."))
            self.assertTrue(provision["quote"])
            self.assertTrue(provision["application_summary"])

    def test_hitl_output_is_compact_structured_and_outward_safe(self) -> None:
        ext = _DrafterExtraction(
            disposition="needs_evidence",
            hitl_nl_summary="JADE reconciled the amount; no person-level conclusion was established.",
            hitl_blocker="The orchestrator could not classify the dispute.",
            hitl_impact=(
                "Without the classification, the proper chargeability treatment cannot be determined."
            ),
            hitl_next_step="A human reviewer must confirm the dispute category and supporting records.",
        )
        parts = _hitl_parts(ext, {}, "the pipeline stopped")
        note = _render_sections(parts)
        response = _hitl_co_owner(parts)

        self.assertIn("## What the review established", note)
        self.assertIn("## What remains unresolved", note)
        self.assertIn("## Why it matters", note)
        self.assertIn("## What is needed to complete the review", note)
        self.assertIn("The review could not classify the dispute.", note)
        self.assertIn("proper chargeability treatment cannot be determined", response)
        self.assertIn("Confirmation is required regarding the dispute category", response)
        self.assertNotRegex(response, r"(?i)JADE|orchestrator|pipeline|HITL|LLM|SME|human reviewer")
        self.assertLessEqual(len(response.split()), 220)
        self.assertTrue(response.endswith("This exception remains open pending resolution of that issue."))

        candidate = ext.model_copy(update={
            "co_owner_response": (
                "This exception is accepted. No further review is required."
            ),
        })
        candidate_parts = _hitl_parts(candidate, {}, "the pipeline stopped")
        candidate_response = _hitl_co_owner(candidate_parts)
        self.assertEqual(candidate_response, response)
        self.assertNotIn("accepted", candidate_response.lower())
        self.assertEqual(candidate_response.lower().count("exception remains open"), 1)


if __name__ == "__main__":
    unittest.main()
