"""Configured deterministic decision hooks for OTL graph nodes.

Add future node hooks here in three places: a typed facts model, a detector function, and the relevant
node registry/evaluator section. ``configs/hooks.yaml`` remains the source of rule enablement and
auditor-facing labels/reasons; agent nodes only assemble grounded facts and consume typed decisions.
"""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field

from shared.hooks_config import EvidenceRuleDefinition, node_hook_definitions


# ==================================================================================================
# Shared decision models and helpers
# ==================================================================================================

GateEvaluationStatus = Literal[
    "disabled", "not_applicable", "pass", "missing", "unreadable", "mismatch",
]


class EvidenceGateEvaluation(BaseModel):
    gate_id: str
    detector: str
    status: GateEvaluationStatus
    required_evidence: list[str] = Field(default_factory=list)
    access_issues: list[str] = Field(default_factory=list)
    rationale: str = ""
    public_reason: str = ""
    details: dict = Field(default_factory=dict)
    outcome_key: str = ""


class EvidenceGateDecision(BaseModel):
    triggered: bool = False
    gate_ids: list[str] = Field(default_factory=list)
    status: Literal["", "missing", "unreadable", "mismatch"] = ""
    required_evidence: list[str] = Field(default_factory=list)
    access_issues: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    outcome_key: str = ""
    disposition: Literal["", "deny", "needs_evidence"] = ""
    route: Literal["", "drafter"] = ""
    hitl_review: bool = False


def _labels(rule: EvidenceRuleDefinition, *keys: str) -> list[str]:
    return [rule.evidence_labels[key] for key in keys if key in rule.evidence_labels]


def _gate_sort_key(gate_id: str) -> tuple[int, str]:
    try:
        return int(gate_id.split("-", 1)[1]), gate_id
    except (IndexError, ValueError):
        return 9999, gate_id


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if str(value or "").strip()))


def _snapshot(node: str, rules: dict, evaluations: list[EvidenceGateEvaluation],
              decision: EvidenceGateDecision) -> dict:
    return {
        "configured": bool(rules),
        "configured_gate_ids": list(rules),
        "evaluations": [item.model_dump(mode="json") for item in evaluations],
        "triggered": decision.triggered,
        "triggered_gate_ids": list(decision.gate_ids),
        "outcome_key": decision.outcome_key,
        "stopped_stage": node if decision.triggered else "",
    }


# ==================================================================================================
# verify_evidence facts and detectors — ED-01 / ED-03 / ED-04 / ED-05
# ==================================================================================================

class EvidenceHookFacts(BaseModel):
    questions: list[str] = Field(default_factory=list)
    ledger_found: bool = False
    workbook_access_issues: list[str] = Field(default_factory=list)
    pob_data_found: bool = False
    heli_data_found: bool = False
    hierarchy_data_found: bool = False
    shell_invoice_found: bool = False
    shell_invoice_sources: list[str] = Field(default_factory=list)


def _verify_ledger(rule: EvidenceRuleDefinition,
                   facts: EvidenceHookFacts) -> EvidenceGateEvaluation:
    if facts.ledger_found:
        return EvidenceGateEvaluation(
            gate_id="", detector=rule.detector, status="pass",
            rationale="A readable JADE/general-ledger tab was identified.",
        )
    status: Literal["missing", "unreadable"] = (
        "unreadable" if facts.workbook_access_issues else "missing"
    )
    return EvidenceGateEvaluation(
        gate_id="", detector=rule.detector, status=status,
        required_evidence=_labels(rule, "ledger"),
        access_issues=list(facts.workbook_access_issues),
        rationale=("Candidate workbook evidence could not be opened or accessed."
                   if status == "unreadable" else
                   "No readable JADE or general-ledger tab was identified."),
        outcome_key=rule.outcomes[status],
    )


def _verify_location(rule: EvidenceRuleDefinition,
                     facts: EvidenceHookFacts) -> EvidenceGateEvaluation:
    if "location" not in set(facts.questions):
        return EvidenceGateEvaluation(
            gate_id="", detector=rule.detector, status="not_applicable",
            rationale="The active question set does not include location.",
        )
    missing: list[str] = []
    if not facts.pob_data_found:
        missing.extend(_labels(rule, "pob"))
    if not facts.heli_data_found:
        missing.extend(_labels(rule, "heli"))
    if not missing:
        return EvidenceGateEvaluation(
            gate_id="", detector=rule.detector, status="pass",
            rationale="Both POB and helicopter evidence were identified.",
        )
    status: Literal["missing", "unreadable"] = (
        "unreadable" if facts.workbook_access_issues else "missing"
    )
    return EvidenceGateEvaluation(
        gate_id="", detector=rule.detector, status=status,
        required_evidence=missing,
        access_issues=list(facts.workbook_access_issues),
        rationale="One or more location evidence sources were not established.",
        outcome_key=rule.outcomes[status],
    )


def _verify_shell_invoice(rule: EvidenceRuleDefinition,
                          facts: EvidenceHookFacts) -> EvidenceGateEvaluation:
    if "third_party_invoice_requested" not in set(facts.questions):
        return EvidenceGateEvaluation(
            gate_id="", detector=rule.detector, status="not_applicable",
            rationale="The active question set does not include a third-party invoice request.",
        )
    if facts.shell_invoice_found:
        return EvidenceGateEvaluation(
            gate_id="", detector=rule.detector, status="pass",
            rationale="A Shell invoice was established in current or prior structured evidence.",
        )
    return EvidenceGateEvaluation(
        gate_id="", detector=rule.detector, status="missing",
        required_evidence=_labels(rule, "shell_invoice"),
        rationale="A third-party invoice was requested without a supporting Shell invoice.",
        outcome_key=rule.outcomes["missing"],
    )


def _verify_hierarchy(rule: EvidenceRuleDefinition,
                      facts: EvidenceHookFacts) -> EvidenceGateEvaluation:
    if "hierarchy" not in set(facts.questions):
        return EvidenceGateEvaluation(
            gate_id="", detector=rule.detector, status="not_applicable",
            rationale="The active question set does not include hierarchy.",
        )
    if facts.hierarchy_data_found:
        return EvidenceGateEvaluation(
            gate_id="", detector=rule.detector, status="pass",
            rationale="Hierarchy-level evidence was identified.",
        )
    status: Literal["missing", "unreadable"] = (
        "unreadable" if facts.workbook_access_issues else "missing"
    )
    return EvidenceGateEvaluation(
        gate_id="", detector=rule.detector, status=status,
        required_evidence=_labels(rule, "hierarchy"),
        access_issues=list(facts.workbook_access_issues),
        rationale="Hierarchy-level evidence was not established.",
        outcome_key=rule.outcomes[status],
    )


class VerifyEvidenceHooks:
    """ED-01/03/04/05 detector registry and aggregate decision builder."""

    NODE = "verify_evidence"
    DETECTORS = {
        "labor_ledger": _verify_ledger,
        "location_logs": _verify_location,
        "shell_invoice_prerequisite": _verify_shell_invoice,
        "hierarchy_support": _verify_hierarchy,
    }

    @classmethod
    def evaluate(cls, facts: EvidenceHookFacts) -> tuple[
        list[EvidenceGateEvaluation], EvidenceGateDecision, dict
    ]:
        rules = node_hook_definitions("labor", cls.NODE)
        evaluations: list[EvidenceGateEvaluation] = []
        for gate_id, rule in sorted(rules.items(), key=lambda item: _gate_sort_key(item[0])):
            if not rule.enabled:
                evaluation = EvidenceGateEvaluation(
                    gate_id=gate_id, detector=rule.detector, status="disabled",
                    rationale="The rule is disabled in hooks.yaml.",
                )
            else:
                evaluation = cls.DETECTORS[rule.detector](rule, facts)
                evaluation.gate_id = gate_id
            evaluations.append(evaluation)

        triggered = [item for item in evaluations if item.status in {"missing", "unreadable"}]
        status: Literal["", "missing", "unreadable"] = ""
        if any(item.status == "unreadable" for item in triggered):
            status = "unreadable"
        elif triggered:
            status = "missing"
        selected = next((item for item in triggered if item.status == status), None)
        decision = EvidenceGateDecision(
            triggered=bool(triggered),
            gate_ids=[item.gate_id for item in triggered],
            status=status,
            required_evidence=_dedupe([
                label for item in triggered for label in item.required_evidence
            ]),
            access_issues=_dedupe([
                issue for item in triggered for issue in item.access_issues
            ]),
            reasons=_dedupe([item.public_reason for item in triggered]),
            outcome_key=selected.outcome_key if selected else "",
            disposition=("needs_evidence" if status == "unreadable" else "deny" if status else ""),
            route="drafter" if triggered else "",
            hitl_review=status == "unreadable",
        )
        return evaluations, decision, _snapshot(cls.NODE, rules, evaluations, decision)


def evaluate_evidence_hooks(*, node: str, facts: EvidenceHookFacts) -> tuple[
    list[EvidenceGateEvaluation], EvidenceGateDecision, dict
]:
    """Compatibility wrapper; new call sites should use ``VerifyEvidenceHooks.evaluate``."""
    if node != VerifyEvidenceHooks.NODE:
        raise ValueError(f"verify-evidence hooks cannot evaluate node {node!r}")
    return VerifyEvidenceHooks.evaluate(facts)


# ==================================================================================================
# jade_analyzer facts and detector — ED-02
# ==================================================================================================

class JadeReconciliationFacts(BaseModel):
    loop_status: str = ""
    amount_basis: str = ""
    amount_computed: float = 0.0
    amount_questioned: float = 0.0
    model_amount_in_range: bool = False
    model_jade_amount_discrepancy: bool = False


class ReconciliationCheck(BaseModel):
    valid: bool = False
    in_range: bool = False
    reason: str = ""
    details: dict = Field(default_factory=dict)


class JadeAnalyzerHooks:
    """ED-02 reconciliation calculation, detector, and decision builder."""

    NODE = "jade_analyzer"
    RECONCILIATION_TOLERANCE = 0.025

    @classmethod
    def compute_reconciliation(cls, facts: JadeReconciliationFacts) -> ReconciliationCheck:
        basis = str(facts.amount_basis or "").strip().lower()
        computed = float(facts.amount_computed or 0.0)
        questioned = float(facts.amount_questioned or 0.0)
        details = {
            "loop_status": facts.loop_status,
            "amount_basis": basis,
            "amount_computed": computed,
            "amount_questioned": questioned,
            "model_amount_in_range": facts.model_amount_in_range,
            "model_jade_amount_discrepancy": facts.model_jade_amount_discrepancy,
            "tolerance_percent": cls.RECONCILIATION_TOLERANCE * 100,
        }
        invalid_reason = ""
        if facts.loop_status != "finished":
            invalid_reason = "JADE reconstruction did not finish."
        elif basis not in {"sum", "half"}:
            invalid_reason = "JADE reconstruction did not return a valid sum/half basis."
        elif not math.isfinite(computed) or not math.isfinite(questioned):
            invalid_reason = "JADE reconstruction returned a non-finite amount."
        elif computed == 0 or questioned == 0:
            invalid_reason = "The computed or auditor-stated amount is zero or unavailable."
        if invalid_reason:
            return ReconciliationCheck(valid=False, reason=invalid_reason, details=details)

        tolerance = abs(questioned) * cls.RECONCILIATION_TOLERANCE
        difference = abs(computed - questioned)
        in_range = difference <= tolerance
        details.update({
            "absolute_difference": difference,
            "percentage_difference": difference / abs(questioned) * 100,
            "tolerance_amount": tolerance,
            "code_amount_in_range": in_range,
            "code_jade_amount_discrepancy": not in_range,
        })
        return ReconciliationCheck(valid=True, in_range=in_range, details=details)

    @classmethod
    def evaluate(cls, facts: JadeReconciliationFacts) -> tuple[
        list[EvidenceGateEvaluation], EvidenceGateDecision, dict, ReconciliationCheck
    ]:
        rules = node_hook_definitions("labor", cls.NODE)
        check = cls.compute_reconciliation(facts)
        evaluations: list[EvidenceGateEvaluation] = []
        for gate_id, rule in sorted(rules.items(), key=lambda item: _gate_sort_key(item[0])):
            if not rule.enabled:
                evaluation = EvidenceGateEvaluation(
                    gate_id=gate_id, detector=rule.detector, status="disabled",
                    rationale="The rule is disabled in hooks.yaml.", details=check.details,
                )
            elif not check.valid:
                evaluation = EvidenceGateEvaluation(
                    gate_id=gate_id, detector=rule.detector, status="not_applicable",
                    rationale=check.reason, details=check.details,
                )
            elif check.in_range:
                evaluation = EvidenceGateEvaluation(
                    gate_id=gate_id, detector=rule.detector, status="pass",
                    rationale="The selected JADE amount reconciles within the inclusive ±2.5% tolerance.",
                    details=check.details,
                )
            else:
                evaluation = EvidenceGateEvaluation(
                    gate_id=gate_id, detector=rule.detector, status="mismatch",
                    rationale="The selected JADE amount is outside the inclusive ±2.5% tolerance.",
                    public_reason=rule.reason.strip(), details=check.details,
                    outcome_key=rule.outcomes["mismatch"],
                )
            evaluations.append(evaluation)

        triggered = [item for item in evaluations if item.status == "mismatch"]
        decision = EvidenceGateDecision(
            triggered=bool(triggered),
            gate_ids=[item.gate_id for item in triggered],
            status="mismatch" if triggered else "",
            reasons=_dedupe([item.public_reason for item in triggered]),
            outcome_key=triggered[0].outcome_key if triggered else "",
            disposition="deny" if triggered else "",
            route="drafter" if triggered else "",
            hitl_review=False,
        )
        return evaluations, decision, _snapshot(cls.NODE, rules, evaluations, decision), check


RECONCILIATION_TOLERANCE = JadeAnalyzerHooks.RECONCILIATION_TOLERANCE


def compute_reconciliation(facts: JadeReconciliationFacts) -> ReconciliationCheck:
    """Compatibility wrapper; new call sites should use ``JadeAnalyzerHooks.compute_reconciliation``."""
    return JadeAnalyzerHooks.compute_reconciliation(facts)


def evaluate_jade_hooks(*, facts: JadeReconciliationFacts) -> tuple[
    list[EvidenceGateEvaluation], EvidenceGateDecision, dict, ReconciliationCheck
]:
    """Compatibility wrapper; new call sites should use ``JadeAnalyzerHooks.evaluate``."""
    return JadeAnalyzerHooks.evaluate(facts)


__all__ = [
    "EvidenceGateDecision", "EvidenceGateEvaluation", "EvidenceHookFacts",
    "JadeAnalyzerHooks", "JadeReconciliationFacts", "ReconciliationCheck",
    "RECONCILIATION_TOLERANCE", "VerifyEvidenceHooks",
    "compute_reconciliation", "evaluate_evidence_hooks", "evaluate_jade_hooks",
]
