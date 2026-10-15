"""Production configuration and decision coverage for verify_evidence hooks."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import otl  # noqa: F401  — bootstrap repository-root shared imports

from shared.hooks_config import (clear_hooks_config_cache, hook_definition,
                                 node_hook_definitions)
from otl.agents import verify_evidence
from otl.graph import route_from_verify_evidence
from otl.decision_hooks import EvidenceHookFacts, VerifyEvidenceHooks, evaluate_evidence_hooks
from otl.output_writer import assemble_report
from otl.prompts import CATEGORY_CHECK_SYSTEM, VERIFY_EVIDENCE_SYSTEM


def _ledger_manifest() -> dict:
    shape = {
        "id": "S1", "tabs": ["JADE"], "header_row": 0,
        "header": ["Employee", "Line Item Text Orig"], "detected_type": "JADE",
    }
    return {
        "workbooks": [{
            "name": "Evidence.xlsx", "path": "/tmp/Evidence.xlsx", "sheets": ["JADE"],
            "shapes": [shape], "tabs": {"JADE": {"shape": "S1", "detected_type": "JADE"}},
        }],
        "unreadable": [],
    }


def _synthesis(*, invoices: list[dict] | None = None) -> dict:
    payload = {
        "attachments": [{
            "file_name": "Evidence.xlsx",
            "tabs": [{
                "tab_name": "JADE", "purpose": "JADE ledger extract",
                "demonstrates": "Line-item charges", "key_columns": ["Employee"],
            }],
        }],
        "open_questions": [],
    }
    if invoices is not None:
        payload["invoices"] = invoices
    return payload


class VerifyEvidenceHookConfigTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_hooks_config_cache()

    def test_existing_budget_and_direct_node_rules_coexist(self) -> None:
        self.assertTrue(hook_definition("extractor", "budget").enabled)
        rules = node_hook_definitions("labor", "verify_evidence")
        self.assertEqual(["ED-01", "ED-03", "ED-04", "ED-05"], list(rules))
        self.assertEqual("labor_ledger", rules["ED-01"].detector)
        self.assertIsNone(hook_definition("labor", "verify_evidence"))
        jade_rules = node_hook_definitions("labor", "jade_analyzer")
        self.assertEqual(["ED-02"], list(jade_rules))
        self.assertEqual("jade_amount_reconciliation", jade_rules["ED-02"].detector)

    def test_individual_rule_disable_does_not_trigger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hooks.yaml"
            path.write_text(
                """schema_version: 1
labor:
  hooks:
    verify_evidence:
      ED-01:
        enabled: false
        detector: labor_ledger
        evidence_labels: {ledger: ledger extract}
        outcomes:
          missing: deny_missing_evidence
          unreadable: needs_evidence_unreadable
""",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"HOOKS_CONFIG_FILE": str(path)}):
                clear_hooks_config_cache()
                evaluations, decision, _snapshot = evaluate_evidence_hooks(
                    node="verify_evidence", facts=EvidenceHookFacts(ledger_found=False),
                )
        self.assertEqual("disabled", evaluations[0].status)
        self.assertFalse(decision.triggered)

    def test_invalid_rule_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hooks.yaml"
            path.write_text(
                """schema_version: 1
labor:
  hooks:
    verify_evidence:
      MISSING_LEDGER:
        enabled: true
        detector: labor_ledger
        evidence_labels: {ledger: ledger extract}
        outcomes:
          missing: deny_missing_evidence
          unreadable: needs_evidence_unreadable
""",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"HOOKS_CONFIG_FILE": str(path)}):
                clear_hooks_config_cache()
                with self.assertRaisesRegex(ValueError, "ED-##"):
                    node_hook_definitions("labor", "verify_evidence")


class VerifyEvidenceHookDecisionTests(unittest.TestCase):
    def test_node_class_owns_verify_registry(self) -> None:
        self.assertEqual("verify_evidence", VerifyEvidenceHooks.NODE)
        self.assertEqual(
            {"labor_ledger", "location_logs", "shell_invoice_prerequisite", "hierarchy_support"},
            set(VerifyEvidenceHooks.DETECTORS),
        )

    def test_aggregates_all_applicable_missing_sources_in_ed_order(self) -> None:
        facts = EvidenceHookFacts(
            questions=["location", "hierarchy", "third_party_invoice_requested"],
        )
        _evaluations, decision, _snapshot = evaluate_evidence_hooks(
            node="verify_evidence", facts=facts,
        )
        self.assertEqual(["ED-01", "ED-03", "ED-04", "ED-05"], decision.gate_ids)
        self.assertEqual("missing", decision.status)
        self.assertEqual("deny", decision.disposition)
        self.assertFalse(decision.hitl_review)
        self.assertEqual(5, len(decision.required_evidence))

    def test_unreadable_workbook_takes_precedence(self) -> None:
        facts = EvidenceHookFacts(
            questions=["location"], workbook_access_issues=["Locked.xlsx: encrypted"],
        )
        _evaluations, decision, _snapshot = evaluate_evidence_hooks(
            node="verify_evidence", facts=facts,
        )
        self.assertEqual("unreadable", decision.status)
        self.assertEqual("needs_evidence", decision.disposition)
        self.assertTrue(decision.hitl_review)


class VerifyEvidenceNodeHookTests(unittest.TestCase):
    def test_invoice_request_is_classified_by_category_check_not_verify_evidence(self) -> None:
        self.assertIn("third_party_invoice_requested", CATEGORY_CHECK_SYSTEM)
        self.assertNotIn("third_party_invoice_requested", VERIFY_EVIDENCE_SYSTEM)
        self.assertNotIn("third_party_invoice_requested", verify_evidence._VerifyExtraction.model_fields)
        self.assertIn("ED-05 requires hierarchy_data_found", VERIFY_EVIDENCE_SYSTEM)

    def test_ed01_denies_when_no_ledger_tab_exists(self) -> None:
        state = {
            "auditor_question": ["job_title"],
            "synthesis": {},
            "exception_exhibits": [],
        }
        with patch.object(verify_evidence, "build_manifest", return_value={"workbooks": [], "unreadable": []}), \
             patch.object(verify_evidence.llm, "structured", side_effect=AssertionError("LLM called")):
            result = verify_evidence.verify_evidence_node(state)

        self.assertTrue(result["evidence_gate_decision"]["triggered"])
        self.assertEqual(["ED-01"], result["evidence_gate_decision"]["gate_ids"])
        self.assertEqual("missing_auditor_evidence", result["evidence_gate_status"])
        self.assertIn("verify_evidence", result["hook_status"])

    def test_shell_invoice_from_extractor_satisfies_ed04(self) -> None:
        ext = verify_evidence._VerifyExtraction()
        state = {
            "auditor_question": ["third_party_invoice_requested"],
            "third_party_invoice_requested": True,
            "synthesis": _synthesis(invoices=[{
                "file_name": "invoice.pdf",
                "parsed": {"vendor": "SHELL INT'L EXPLORATION AND PROD'N INC"},
            }]),
            "exception_exhibits": ["/tmp/Evidence.xlsx"],
        }
        with patch.object(verify_evidence, "build_manifest", return_value=_ledger_manifest()), \
             patch.object(verify_evidence.llm, "structured", return_value=ext):
            result = verify_evidence.verify_evidence_node(state)

        self.assertFalse(result["evidence_gate_decision"]["triggered"])
        self.assertTrue(result["verify_evidence"].shell_invoice_found)
        self.assertEqual("drafter", route_from_verify_evidence({**state, **result}))

    def test_ed04_missing_invoice_consumes_request_without_hitl(self) -> None:
        ext = verify_evidence._VerifyExtraction()
        state = {
            "auditor_question": ["third_party_invoice_requested"],
            "third_party_invoice_requested": True,
            "synthesis": _synthesis(invoices=[]),
            "exception_exhibits": ["/tmp/Evidence.xlsx"],
        }
        with patch.object(verify_evidence, "build_manifest", return_value=_ledger_manifest()), \
             patch.object(verify_evidence.llm, "structured", return_value=ext):
            result = verify_evidence.verify_evidence_node(state)

        decision = result["evidence_gate_decision"]
        self.assertEqual(["ED-04"], decision["gate_ids"])
        self.assertFalse(decision["hitl_review"])

    def test_prior_structured_shell_invoice_satisfies_ed04(self) -> None:
        history = [{
            "cycle": 0,
            "content": """<!-- rebuttal-agent:verify_evidence:start -->
## Agent: verify_evidence

### Decision-relevant details

```json
{"shell_invoice_found": true, "shell_invoice_sources": ["Prior Shell invoice.pdf"],
 "shell_invoice_match_basis": ["prior seller was SIEP"],
 "evidence_gate_decision": {"triggered": false, "gate_ids": []}}
```
<!-- rebuttal-agent:verify_evidence:end -->""",
        }]
        found, sources, basis = verify_evidence._shell_invoice_facts({}, history)
        self.assertTrue(found)
        self.assertEqual(["Prior Shell invoice.pdf"], sources)
        self.assertEqual(["prior seller was SIEP"], basis)

    def test_bill_to_only_does_not_count_as_shell_invoice(self) -> None:
        found, _sources, _basis = verify_evidence._shell_invoice_facts({
            "invoices": [{
                "file_name": "Vendor 123.pdf",
                "extraction": {"vendor": "Acme Services", "bill_to": "Shell Offshore Inc."},
            }],
        }, [])
        self.assertFalse(found)

    def test_operator_response_invoice_is_excluded(self) -> None:
        found, _sources, _basis = verify_evidence._shell_invoice_facts({
            "invoices": [{
                "file_name": "Operator Response Shell invoice.pdf",
                "parsed": {"vendor": "Shell Offshore Inc."},
            }],
        }, [])
        self.assertFalse(found)

    def test_hook_snapshot_is_serialized_in_report(self) -> None:
        report = assemble_report(
            {
                "case": {"case_id": "GreatWhite_2024_0010", "exc_id": "0010"},
                "hook_status": {"verify_evidence": {"triggered": True, "triggered_gate_ids": ["ED-01"]}},
                "drafter": {"disposition": "deny", "draft_response": "No credit is due."},
            },
            run_id="GreatWhite_2024_0010", elapsed_minutes=0.1,
            token_usage={}, trace_events=[],
        )
        self.assertEqual(["ED-01"], report.hooks["verify_evidence"]["triggered_gate_ids"])


if __name__ == "__main__":
    unittest.main()
