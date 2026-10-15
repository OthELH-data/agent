from __future__ import annotations

import unittest
from pathlib import Path

from otl.chargeability_hooks import build_charge_scopes, evaluate_hooks, evaluate_state, hook_rules
from otl.state import LedgerRow, SourceRowRef


def _row(row: int, person: str, *, mcoh: bool, amount: float = 100.0) -> LedgerRow:
    return LedgerRow(
        source_ref=SourceRowRef(workbook="JADE.xlsx", sheet="JADE", excel_row=row),
        person=person, amount=amount, jv_object_type="C116" if mcoh else "C102",
        mcoh_status="mcoh" if mcoh else "non_mcoh",
    )


def _state(rows, *, roles, locations, cfr="chargeable", precedence="unavailable_or_unclear"):
    people = list(roles)
    return {
        "jade_analyzer": {"ledger_rows": rows, "people_found_in_ledger": people},
        "job_title": {
            "role_status_by_person": roles,
            "ipt_status_by_person": {p: "non_ipt" for p in people},
            "exempt_role_status_by_person": {p: "non_exempt" for p in people},
            "job_profile_status_by_person": {p: "matched" for p in people},
        },
        "location": {
            "found_in_pob": {p: locations[p] == "onsite" for p in people},
            "found_in_heli_logs": {p: False for p in people},
        },
        "supervision": {"above_first_level": {p: False for p in people}},
        "matching": {
            "cfr_copas_resolution": cfr,
            "cfr_copas_resolution_by_person": {p: cfr for p in people},
            "cfr_source_reference_by_person": {p: ["CFR source"] for p in people},
        },
        "agreement_retriever": {"agreement_chargeability": "unclear"},
        "wiki": {"precedence_status": precedence},
    }


class ChargeabilityConsolidationTests(unittest.TestCase):
    def test_all_chargeability_rules_are_loaded_from_shared_hooks_yaml(self):
        rules = hook_rules()
        self.assertEqual(len(rules), 29)
        self.assertIn("GR-01", rules)
        self.assertIn("SE-25", rules)
        self.assertFalse(Path("configs/labor_chargeability_hooks.yaml").exists())

    def test_any_location_rule_can_fire_when_location_is_unresolved(self):
        state = _state([_row(2, "A", mcoh=True)], roles={"A": "technical"},
                       locations={"A": "offsite"}, cfr="chargeable")
        state["location"] = {
            "location_status": {"A": "unresolved"},
            "location_rationale": {"A": "Two close source-name matches remain."},
        }
        evaluations = evaluate_hooks(state, build_charge_scopes(state))
        fired = {item.hook_id for item in evaluations if item.status == "fired"}
        self.assertIn("GR-01", fired)

    def test_location_specific_rules_do_not_fire_when_location_is_unresolved(self):
        state = _state([_row(2, "A", mcoh=False)], roles={"A": "technical"},
                       locations={"A": "offsite"}, cfr="unclear")
        state["location"] = {
            "location_status": {"A": "unresolved"},
            "location_rationale": {"A": "The source period is incomplete."},
        }
        evaluations = evaluate_hooks(state, build_charge_scopes(state))
        by_id = {item.hook_id: item.status for item in evaluations}
        self.assertEqual(by_id["GR-02"], "unresolved")
        self.assertEqual(by_id["GR-05"], "unresolved")

    def test_same_disposition_hooks_consolidate(self):
        state = _state([_row(2, "A", mcoh=True)], roles={"A": "technical"},
                       locations={"A": "onsite"}, cfr="chargeable")
        result = evaluate_state(state, final=True)["chargeability_resolution"]
        self.assertEqual(result.disposition, "deny")
        self.assertIn("GR-01", next(iter(result.fired_hooks.values())))
        self.assertIn("SE-06", next(iter(result.fired_hooks.values())))

    def test_conflicting_hooks_on_same_scope_require_review(self):
        state = _state([_row(2, "A", mcoh=True)], roles={"A": "technical"},
                       locations={"A": "onsite"}, cfr="not_chargeable")
        result = evaluate_state(state, final=True)["chargeability_resolution"]
        self.assertEqual(result.disposition, "needs_evidence")
        self.assertIn("A", result.unresolved_people)

    def test_different_scopes_for_same_person_require_review(self):
        rows = [_row(2, "A", mcoh=True), _row(3, "A", mcoh=False)]
        state = _state(rows, roles={"A": "non_technical"}, locations={"A": "offsite"},
                       cfr="unclear", precedence="supports_chargeability")
        result = evaluate_state(state, final=True)["chargeability_resolution"]
        self.assertEqual(result.disposition, "needs_evidence")

    def test_different_people_can_form_partial_case(self):
        rows = [_row(2, "A", mcoh=True, amount=70), _row(3, "B", mcoh=False, amount=30)]
        state = _state(rows, roles={"A": "technical", "B": "non_technical"},
                       locations={"A": "onsite", "B": "offsite"}, cfr="not_chargeable")
        # A: GR-01 deny (SE-07 is human-review, so use per-person authority below).
        state["matching"]["cfr_copas_resolution"] = "unclear"
        state["matching"]["cfr_copas_resolution_by_person"] = {"A": "unclear", "B": "unclear"}
        state["wiki"]["precedence_status"] = "supports_chargeability"
        # B remains GR-06 accept; A resolves through GR-01 + SE-09 deny.
        result = evaluate_state(state, final=True)["chargeability_resolution"]
        self.assertEqual(result.disposition, "partial")
        self.assertEqual(result.resolved_deny_amount, 70)
        self.assertEqual(result.resolved_accept_amount, 30)

    def test_one_unresolved_person_makes_case_needs_evidence(self):
        rows = [_row(2, "A", mcoh=True, amount=70), _row(3, "B", mcoh=False, amount=30)]
        state = _state(rows, roles={"A": "technical", "B": "unclear"},
                       locations={"A": "onsite", "B": "offsite"}, cfr="chargeable")
        state["location"]["found_in_pob"].pop("B")
        state["location"]["found_in_heli_logs"].pop("B")
        result = evaluate_state(state, final=True)["chargeability_resolution"]
        self.assertEqual(result.disposition, "needs_evidence")
        self.assertGreater(result.unresolved_amount, 0)

    def test_agreement_chargeability_can_trigger_legal_basis_without_cfr(self):
        state = _state([_row(2, "A", mcoh=True)], roles={"A": "technical"},
                       locations={"A": "onsite"}, cfr="unclear")
        state["agreement_retriever"] = {
            "agreement_chargeability": "chargeable",
            "agreement_chargeability_citations": [{"chunk_id": "agreement-1"}],
        }
        evaluated = evaluate_state(state, final=True)
        self.assertTrue(evaluated["legal_chargeability_basis_by_person"]["A"])
        fired = {item.hook_id for item in evaluated["chargeability_hook_evaluations"]
                 if item.status == "fired"}
        self.assertIn("SE-06", fired)

    def test_cfr_chargeable_wins_when_agreement_is_non_chargeable(self):
        state = _state([_row(2, "A", mcoh=True)], roles={"A": "technical"},
                       locations={"A": "onsite"}, cfr="chargeable")
        state["agreement_retriever"] = {"agreement_chargeability": "not_chargeable"}
        evaluated = evaluate_state(state, final=True)
        self.assertTrue(evaluated["legal_chargeability_basis_by_person"]["A"])

    def test_ipt_registry_fact_drives_the_offsite_exempt_rule(self):
        state = _state([_row(2, "A", mcoh=False)], roles={"A": "technical"},
                       locations={"A": "offsite"}, cfr="unclear",
                       precedence="supports_chargeability")
        state["job_title"]["ipt_status_by_person"]["A"] = "ipt"
        state["supervision"]["above_first_level"]["A"] = True
        result = evaluate_state(state, final=True)
        fired = {item.hook_id for item in result["chargeability_hook_evaluations"]
                 if item.status == "fired"}
        self.assertIn("SE-25", fired)
        self.assertEqual(result["chargeability_resolution"].disposition, "deny")

    def test_supervision_missing_record_fires_human_review_gate(self):
        state = _state([_row(2, "A", mcoh=False)], roles={"A": "technical"},
                       locations={"A": "onsite"}, cfr="chargeable")
        state["supervision"] = {
            "supervision_status_by_person": {"A": "unresolved"},
            "supervision_record_status_by_person": {"A": "missing"},
            "supervision_escalation_reasons_by_person": {"A": ["no staff hierarchy row"]},
        }
        evaluated = evaluate_state(state, final=True)
        fired = {item.hook_id for item in evaluated["chargeability_hook_evaluations"]
                 if item.status == "fired"}
        self.assertIn("SE-05", fired)
        self.assertEqual(evaluated["chargeability_resolution"].disposition, "needs_evidence")


if __name__ == "__main__":
    unittest.main()
