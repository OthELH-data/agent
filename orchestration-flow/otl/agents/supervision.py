"""Deterministic supervision analysis with unresolved-person code escalation."""
from __future__ import annotations

from pathlib import Path

from otl.chargeability_hooks import evaluate_state
from otl.code_exec import CodeExecutor
from otl.person_index import build_person_index, index_json, person_index_block
from otl.prompts import SUPERVISION_SYSTEM
from otl.react_loop import run_agent_loop
from otl.state import SupervisionState
from otl.supervision_evidence import SupervisionEvidence, analyze_supervision
from otl.trace import trace_event
from otl.workbook_manifest import build_manifest, manifest_block
from otl.agents._common import (case_context_block, directive_block, disputed_people, parse_amount,
                                person_tab_hints, resolve_staff_listing, sub_get)


def _questioned_amounts(state: dict, people: list[str]) -> dict[str, float]:
    charged = dict(sub_get(state, "jade_analyzer", "charged_amount_by_person", {}) or {})
    return {person: parse_amount(charged.get(person)) for person in people if charged.get(person) is not None}


def _state_from_evidence(evidence: SupervisionEvidence, amounts: dict[str, float], *, mode: str) -> SupervisionState:
    credit = sum(amount for person, amount in amounts.items()
                 if evidence.supervision_status_by_person.get(person) == "above_first_level")
    chargeable = sum(amount for person, amount in amounts.items()
                     if evidence.supervision_status_by_person.get(person) == "not_above_first_level")
    unresolved = [person for person, status in evidence.supervision_status_by_person.items()
                  if status == "unresolved"]
    return SupervisionState(
        nl_summary=(
            f"Deterministic hierarchy analysis resolved "
            f"{len(evidence.supervision_status_by_person) - len(unresolved)} person(s) and left "
            f"{len(unresolved)} unresolved."
        ),
        above_first_level=evidence.above_first_level,
        reports_summary=evidence.reports_summary,
        questioned_amounts=amounts,
        credit_amount=round(credit, 2), chargeable_amount=round(chargeable, 2),
        needs_human_review=bool(unresolved),
        review_reason=("Supervision level remains unresolved for: " + ", ".join(unresolved[:10])
                       if unresolved else ""),
        supervision_status_by_person=evidence.supervision_status_by_person,
        supervision_record_status_by_person=evidence.supervision_record_status_by_person,
        hierarchy_periods_by_person={
            person: [item.model_dump(mode="json") for item in values]
            for person, values in evidence.hierarchy_periods_by_person.items()
        },
        supervision_analysis_mode=mode,
        supervision_escalation_reasons_by_person=evidence.unresolved_reasons,
    )


def _fallback(state: dict, *, people: list[str], staff_listing: str | None,
              manifest: dict, index: dict, reasons: dict[str, list[str]]):
    run_id, case = state.get("run_id", ""), state.get("case") or {}
    inputs = {
        "staff_listing": staff_listing or "", "disputed_people": people,
        "escalation_reasons": reasons, "open_questions": list(state.get("open_questions") or []),
    }
    task = (
        "Resolve only the supplied unresolved people's reporting chains. Deterministic monthly org-tree "
        "analysis completed everyone else. Use person_index() first and inspect raw rows only where the "
        "listed ambiguity requires it. For each person return above_first_level, reports_summary, "
        "supervision_status_by_person (above_first_level | not_above_first_level | unresolved), "
        "supervision_record_status_by_person (found | missing | ambiguous), and nl_summary. A person is above "
        "first level only when they have direct reports and at least one direct report has reports. Do not "
        "return another person and do not decide the final disposition.\n\n"
        f"UNRESOLVED PEOPLE: {people}\nREASONS: {reasons}"
        + case_context_block(state, specialist="supervision")
        + directive_block(state, "supervision")
        + manifest_block(manifest)
        + person_index_block(index, people)
    )
    executor = CodeExecutor(run_id, "supervision", read_roots=state.get("sandbox_read_roots"),
                            manifest=manifest, person_index=index_json(index))
    return run_agent_loop(
        node="supervision", system=SUPERVISION_SYSTEM, task=task, executor=executor, inputs=inputs,
        max_steps=8, trace_path=state.get("trace_path"), run_id=run_id,
        case_id=case.get("case_id", ""),
    )


def _merge_fallback(sub: SupervisionState, result: dict, unresolved: list[str]) -> SupervisionState:
    raw = result or {}
    above = raw.get("above_first_level") if isinstance(raw.get("above_first_level"), dict) else {}
    summaries = raw.get("reports_summary") if isinstance(raw.get("reports_summary"), dict) else {}
    statuses = (raw.get("supervision_status_by_person")
                if isinstance(raw.get("supervision_status_by_person"), dict) else {})
    records = (raw.get("supervision_record_status_by_person")
               if isinstance(raw.get("supervision_record_status_by_person"), dict) else {})
    for person in unresolved:
        if person in above:
            value = str(above[person]).strip().casefold() in {"true", "yes", "1", "y"}
            sub.above_first_level[person] = value
            sub.supervision_status_by_person[person] = (
                "above_first_level" if value else "not_above_first_level")
        elif person in statuses:
            sub.supervision_status_by_person[person] = str(statuses[person])
        if person in summaries:
            sub.reports_summary[person] = str(summaries[person])
        if person in records:
            sub.supervision_record_status_by_person[person] = str(records[person])
        if sub.supervision_status_by_person.get(person) in {"above_first_level", "not_above_first_level"}:
            sub.supervision_escalation_reasons_by_person.pop(person, None)
    sub.supervision_analysis_mode = "code_escalation"
    sub.nl_summary = str(raw.get("nl_summary") or sub.nl_summary)
    remaining = [person for person, status in sub.supervision_status_by_person.items()
                 if status == "unresolved"]
    sub.needs_human_review = bool(remaining)
    sub.review_reason = (
        "Supervision level remains unresolved for: " + ", ".join(remaining[:10]) if remaining else "")
    sub.credit_amount = round(sum(
        amount for person, amount in sub.questioned_amounts.items()
        if sub.supervision_status_by_person.get(person) == "above_first_level"), 2)
    sub.chargeable_amount = round(sum(
        amount for person, amount in sub.questioned_amounts.items()
        if sub.supervision_status_by_person.get(person) == "not_above_first_level"), 2)
    return sub


def supervision_node(state: dict) -> dict:
    people = disputed_people(state)
    staff_listing = resolve_staff_listing(state.get("common_reports_dir") or "")
    manifest = build_manifest([staff_listing], people=people, tab_hints=person_tab_hints(state))
    index = build_person_index([("staff", [staff_listing] if staff_listing else [])], people, manifest)
    amounts = _questioned_amounts(state, people)
    try:
        evidence = analyze_supervision(index, people)
    except Exception as exc:
        evidence = SupervisionEvidence(
            supervision_status_by_person={person: "unresolved" for person in people},
            supervision_record_status_by_person={person: "ambiguous" for person in people},
            unresolved_reasons={person: [f"deterministic hierarchy analysis failed: {type(exc).__name__}: {exc}"]
                                for person in people},
        )
    sub = _state_from_evidence(evidence, amounts, mode="deterministic")
    unresolved = [person for person in people
                  if sub.supervision_status_by_person.get(person) == "unresolved"]
    fallback_steps = 0
    fallback_status = ""
    if unresolved:
        loop = _fallback(
            state, people=unresolved, staff_listing=staff_listing, manifest=manifest, index=index,
            reasons={person: sub.supervision_escalation_reasons_by_person.get(person, [])
                     for person in unresolved},
        )
        fallback_steps, fallback_status = loop.steps, loop.status
        sub = _merge_fallback(sub, loop.result or {}, unresolved)

    remaining = [person for person in people
                 if sub.supervision_status_by_person.get(person) == "unresolved"]
    status = "objective_met" if not remaining else "inconclusive"
    reason = "" if not remaining else sub.review_reason
    update: dict = {
        "supervision": sub,
        "agent_status": {"supervision": {"status": status, "reason": reason}},
    }
    update.update(evaluate_state({**state, **update}, final=False))
    update.update(trace_event(
        state, "supervision", f"classified supervision level ({sub.supervision_analysis_mode})",
        inputs_digest=(f"{len(people)} person(s); staff_listing="
                       f"{Path(staff_listing).name if staff_listing else None}"),
        key_outputs={
            "mode": sub.supervision_analysis_mode, "above_first_level": sub.above_first_level,
            "supervision_status_by_person": sub.supervision_status_by_person,
            "credit_amount": sub.credit_amount, "chargeable_amount": sub.chargeable_amount,
            "fallback_people": unresolved, "fallback_steps": fallback_steps,
            "fallback_status": fallback_status, "status": status,
        },
        nl_summary=sub.nl_summary,
    ))
    return update


__all__ = ["supervision_node"]
