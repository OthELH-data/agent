"""Characterize cited evidence and execute configured ``labor.hooks.verify_evidence`` rules.

The node profiles manifest-approved auditor workbooks, caches the compact workbook manifest for JADE,
and gives the med model only row-free tab metadata. Uncited response material is filtered, while a
response-named workbook explicitly cited by the auditor remains evidence. The model detects POB, heli,
and hierarchy evidence.
The third-party-invoice category comes from category_check; ledger presence, Shell-invoice presence,
every ED outcome, aggregation, disposition, and route are deterministic.

Configured ED-01/03/04/05 failures are aggregated into one typed decision. Missing evidence denies;
concretely unreadable evidence defers to HITL. Disabling the node in pipeline.yaml bypasses all four
rules. Extractor open questions remain unweighted downstream hints.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

from otl import llm_adapter as llm
from otl.prompts import VERIFY_EVIDENCE_SYSTEM
from otl.state import VerifyEvidenceState
from otl.trace import trace_event
from otl.workbook_manifest import build_manifest
from otl.response_filter import is_operator_response
from otl import ledger_people
from otl.decision_hooks import EvidenceHookFacts, VerifyEvidenceHooks
from otl.agents._common import cited_attachments, sub_get, case_context_block
from shared.rebuttal_history import extract_agent_details


class _VerifyExtraction(BaseModel):
    """The med model's read of the manifest-approved cited tabs.

    The model only DETECTS cited evidence. Configured ED rules compute every pass/fail outcome downstream
    in code. The Field descriptions below become the JSON schema the model sees, so accurate evidence
    detection is the model's whole job here.
    """
    pob_data_found: bool = Field(
        default=False,
        description="True iff the cited evidence contains personnel-on-board (POB) data for the "
                    "questioned people over the charge period.")
    heli_data_found: bool = Field(
        default=False,
        description="True iff the cited evidence contains helicopter-manifest / flight-log data "
                    "(POB and heli may share one combined tab).")
    hierarchy_data_found: bool = Field(
        default=False,
        description="True iff the cited evidence proves the SUPERVISION "
                    "HIERARCHY of the questioned people — an org chart, reporting structure, or "
                    "manager/reports mapping.")
    nl_summary: str = Field(
        default="", description="A concise analysis of cited evidence, notable gaps, and why those gaps "
                                "matter to the current question. Do not list every tab or column.")


def _source_workbook_paths(state: dict, cited_workbooks: list[str]) -> list[str]:
    # The final evidence-discovery manifest already applied ownership filtering and the auditor-citation
    # union. Do not independently re-filter the state list here.
    paths = list(state.get("exception_exhibits") or [])
    cited = {str(name) for name in cited_workbooks if str(name or "").strip()}
    return [str(path) for path in paths if not cited or Path(str(path)).name in cited]


def _manifest_attachments(manifest: dict) -> tuple[list[str], dict[str, list[str]], list[dict]]:
    """Reshape deterministic workbook profiling into the metadata the verification model reads."""
    workbooks: list[str] = []
    sheets: dict[str, list[str]] = {}
    tabs: list[dict] = []
    for workbook in manifest.get("workbooks", []) or []:
        name = str(workbook.get("name") or Path(str(workbook.get("path") or "")).name)
        tab_names = [str(tab) for tab in workbook.get("sheets", []) or []]
        workbooks.append(name)
        sheets[name] = tab_names
        shapes = {str(shape.get("id")): shape for shape in workbook.get("shapes", []) or []}
        tab_records = workbook.get("tabs", {}) or {}
        for tab_name in tab_names:
            tab_record = tab_records.get(tab_name, {}) or {}
            shape = shapes.get(str(tab_record.get("shape")), {})
            detected = str(tab_record.get("detected_type") or shape.get("detected_type") or "")
            tabs.append({
                "file": name,
                "tab_name": tab_name,
                "purpose": detected or str(tab_record.get("confidence") or "workbook tab"),
                "demonstrates": "Source workbook tab schema identified from the attachment list.",
                "key_columns": list(shape.get("header") or []),
            })
    return workbooks, sheets, tabs


def _manifest_access_issues(manifest: dict) -> list[str]:
    issues: list[str] = []
    for item in manifest.get("unreadable", []) or []:
        name = str(item.get("name") or Path(str(item.get("path") or "")).name or "workbook")
        errors = item.get("errors") or {}
        detail = "; ".join(f"{key}: {value}" for key, value in errors.items())
        issues.append(f"{name}: {detail or 'could not be opened or accessed'}")
    return issues


_SHELL_SELLER_RE = re.compile(r"\b(?:shell|siep)\b", re.IGNORECASE)


def _shell_invoice_facts(synth: dict, history: list[dict]) -> tuple[bool, list[str], list[str]]:
    """Find a Shell/SIEP seller invoice in current extractor output or structured prior history."""
    sources: list[str] = []
    basis: list[str] = []
    invoices = synth.get("invoices") if isinstance(synth, dict) else None
    for invoice in invoices if isinstance(invoices, list) else []:
        if not isinstance(invoice, dict):
            continue
        name = str(invoice.get("file_name") or Path(str(invoice.get("source_path") or "")).name)
        if name and is_operator_response(name):
            continue
        normalized_name = " ".join(re.sub(r"[^a-z0-9]+", " ", name.lower()).split())
        filename_match = (
            ("shell" in normalized_name or "siep" in normalized_name)
            and "invoice" in normalized_name
        )
        sellers: list[str] = []
        for section_name in ("parsed", "extraction"):
            section = invoice.get(section_name) or {}
            if isinstance(section, dict):
                for key in ("vendor", "seller"):
                    value = str(section.get(key) or "").strip()
                    if value:
                        sellers.append(value)
        seller = next((value for value in sellers if _SHELL_SELLER_RE.search(value)), "")
        if not filename_match and not seller:
            continue
        source = name or str(invoice.get("source_path") or "Shell invoice")
        sources.append(source)
        if filename_match:
            basis.append(f"{source}: filename identifies a Shell invoice")
        if seller:
            basis.append(f"{source}: extracted seller/vendor is {seller}")

    if not sources:
        for details in extract_agent_details(history, "verify_evidence"):
            if not details.get("shell_invoice_found"):
                continue
            prior_sources = [str(value) for value in details.get("shell_invoice_sources", []) or []]
            sources.extend(prior_sources or [f"prior rebuttal cycle {details.get('cycle', 0)}"])
            prior_basis = [str(value) for value in details.get("shell_invoice_match_basis", []) or []]
            basis.extend(prior_basis or ["established in prior structured verify_evidence findings"])
            break
    return bool(sources), list(dict.fromkeys(sources)), list(dict.fromkeys(basis))


def verify_evidence_node(state: dict) -> dict:
    synth = state.get("synthesis") or {}
    cited_workbooks, cited_sheets, kept_tabs = cited_attachments(synth)
    source_paths = _source_workbook_paths(state, cited_workbooks)
    manifest = build_manifest(source_paths)
    manifest_workbooks, manifest_sheets, manifest_tabs = _manifest_attachments(manifest)
    if not kept_tabs:
        cited_workbooks, cited_sheets, kept_tabs = manifest_workbooks, manifest_sheets, manifest_tabs
    open_questions = list(synth.get("open_questions") or [])
    # category_check ran immediately upstream in the spine, so the dispute categories are available and
    # drive which evidence the gate requires (POB/heli whenever location is included; hierarchy proof is
    # informational). Read the master channel, falling back to the sub-state.
    questions = (state.get("auditor_question")
                 or sub_get(state, "category_check", "auditor_question", []) or [])

    if kept_tabs:
        category = sub_get(state, "category_check", "nl_summary", "") or ""
        user = (
            "The auditor cited these workbook tabs (uncited operator-response material removed). Read "
            "their purpose/demonstrates/columns — do NOT ask for raw rows:\n"
            f"{json.dumps(kept_tabs, ensure_ascii=False, default=str)}"
            + case_context_block(state)
            + f"\n\nThe auditor's dispute category(ies): {json.dumps(questions, ensure_ascii=False)}. "
            f"Dispute category(ies) explanation: {str(category).strip()}\n\nDetect which of these are present "
            "in the cited evidence and set the flags accordingly: "
            "POB (personnel-on-board) data (pob_data_found); helicopter-manifest data (heli_data_found) "
            "— POB & heli may share ONE combined tab; and proof of the SUPERVISION HIERARCHY of the "
            "questioned people — an org chart, reporting structure, or manager/reports mapping "
            "(hierarchy_data_found). The pipeline decides all pass/fail outcomes in code; detect the "
            "evidence flags accurately and do not give a disposition."
        )
        ext = llm.structured("verify_evidence", VERIFY_EVIDENCE_SYSTEM, user, _VerifyExtraction)
    else:
        ext = _VerifyExtraction(
            nl_summary="No citable auditor-side workbook tabs remained after response filtering."
        )

    access_issues = _manifest_access_issues(manifest)
    ledger_found = bool(ledger_people.find_ledger_tabs(manifest))
    shell_found, shell_sources, shell_basis = _shell_invoice_facts(
        synth, list(state.get("rebuttal_history") or [])
    )
    facts = EvidenceHookFacts(
        questions=list(questions),
        ledger_found=ledger_found,
        workbook_access_issues=access_issues,
        pob_data_found=ext.pob_data_found,
        heli_data_found=ext.heli_data_found,
        hierarchy_data_found=ext.hierarchy_data_found,
        shell_invoice_found=shell_found,
        shell_invoice_sources=shell_sources,
    )
    evaluations, decision, hook_snapshot = VerifyEvidenceHooks.evaluate(facts)
    gate_status = (
        "unreadable_auditor_evidence" if decision.status == "unreadable"
        else "missing_auditor_evidence" if decision.status == "missing"
        else ""
    )
    sub = VerifyEvidenceState(
        nl_summary=ext.nl_summary,
        synthesis_json=synth,
        pob_data_found=ext.pob_data_found,
        heli_data_found=ext.heli_data_found,
        hierarchy_data_found=ext.hierarchy_data_found,
        ledger_data_found=ledger_found,
        workbook_access_issues=access_issues,
        shell_invoice_found=shell_found,
        shell_invoice_sources=shell_sources,
        shell_invoice_match_basis=shell_basis,
        evidence_gate_status=gate_status,
        cited_workbooks=cited_workbooks,
        cited_sheets=cited_sheets,
        open_questions=open_questions,
        evidence_gate_evaluations=[item.model_dump(mode="json") for item in evaluations],
        evidence_gate_decision=decision.model_dump(mode="json"),
    )

    update: dict = {
        "verify_evidence": sub,
        "evidence_gate_status": sub.evidence_gate_status,
        "evidence_gap": list(decision.required_evidence),
        "evidence_gate_decision": sub.evidence_gate_decision,
        "evidence_gate_evaluations": sub.evidence_gate_evaluations,
        "auditor_workbook_manifest": manifest,
        "hook_status": {"verify_evidence": hook_snapshot},
        # forwarded to the specialists as hints (never a gate)
        "open_questions": open_questions,
    }
    update.update(trace_event(
        state, "verify_evidence", "characterized cited evidence",
        inputs_digest=f"{len(kept_tabs)} cited tab(s) across {len(cited_workbooks)} workbook(s)",
        key_outputs={"evidence_gate_status": sub.evidence_gate_status,
                     "triggered_gate_ids": list(decision.gate_ids),
                     "required_evidence": list(decision.required_evidence),
                     "auditor_question": list(questions),
                     "pob": sub.pob_data_found, "heli": sub.heli_data_found,
                     "hierarchy": sub.hierarchy_data_found, "ledger": sub.ledger_data_found,
                     "shell_invoice": sub.shell_invoice_found,
                     "third_party_invoice_requested": "third_party_invoice_requested" in set(questions),
                     "cited_sheets": sub.cited_sheets,
                     "open_questions": open_questions},
        nl_summary=sub.nl_summary or (
            f"POB={sub.pob_data_found}, heli={sub.heli_data_found}, "
            f"hierarchy={sub.hierarchy_data_found}, ledger={sub.ledger_data_found}"),
    ))
    return update


__all__ = ["verify_evidence_node"]
