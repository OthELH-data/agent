"""Structured-first job-title specialist with unresolved-person code escalation."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from otl.chargeability_hooks import evaluate_state
from otl.code_exec import CodeExecutor
from otl.job_title_evidence import (JobTitlePersonFinding, PreparedJobTitleEvidence, analyze_packets,
                                    exempt_status, prepare_evidence)
from otl.person_index import build_person_index, index_json, person_index_block
from otl.prompts import JOB_TITLE_SYSTEM
from otl.react_loop import run_agent_loop
from otl.state import JobTitleState
from otl.trace import trace_event
from otl.workbook_manifest import build_manifest, manifest_block
from otl.agents._common import (case_context_block, directive_block, disputed_people, person_tab_hints,
                                pick_latest_by_keyword, resolve_staff_listing)


def _jobs_catalogue(otl_dir: str) -> str | None:
    override = os.getenv("OTL_JOBS_CATALOGUE")
    if override and Path(override).exists():
        return override
    return pick_latest_by_keyword(otl_dir, ["catalogue"])


def _structured_state(prepared: PreparedJobTitleEvidence, findings: list[JobTitlePersonFinding],
                      reasons: dict[str, list[str]], *, mode: str) -> JobTitleState:
    packets = {packet.person: packet for packet in prepared.packets}
    by_person = {finding.person: finding for finding in findings}
    positions: dict[str, list[str]] = {}
    title_periods: dict[str, list[dict]] = {}
    profiles_by_person: dict[str, list[str]] = {}
    role_status: dict[str, str] = {}
    rationale_by_person: dict[str, str] = {}
    profile_status: dict[str, str] = {}
    ipt_by_person: dict[str, str] = {}
    ipt_evidence: dict[str, dict] = {}
    exempt_by_person: dict[str, str] = {}
    names_staff: dict[str, str] = {}
    profile_summaries: dict[str, str] = {}
    profile_descriptions: dict[str, str] = {}

    for person, packet in packets.items():
        person_positions = list(dict.fromkeys(period.position for period in packet.staff_title_periods
                                              if period.position))
        positions[person] = person_positions
        title_periods[person] = [period.model_dump(mode="json") for period in packet.staff_title_periods]
        if person_positions:
            names_staff[person] = person_positions[-1]
        finding = by_person.get(person)
        if finding:
            profiles = list(dict.fromkeys([
                *finding.selected_job_profiles,
                *(profile for values in finding.selected_job_profiles_by_period.values()
                  for profile in values),
            ]))
            profiles_by_person[person] = profiles
            role_status[person] = finding.technical_status
            rationale_by_person[person] = finding.technical_rationale
            profile_status[person] = finding.job_profile_status
            candidates = {candidate.job_profile: candidate
                          for values in packet.catalog_candidates_by_period.values()
                          for candidate in values}
            for profile in profiles:
                candidate = candidates.get(profile)
                if candidate:
                    profile_summaries[profile] = candidate.job_summary
                    profile_descriptions[profile] = candidate.job_description
        else:
            profiles_by_person[person] = []
            role_status[person] = "unclear"
            rationale_by_person[person] = "; ".join(reasons.get(person) or [])
            profile_status[person] = "unclear" if packet.staff_title_periods else "no_match"

        if packet.ipt_registry_match:
            ipt_by_person[person] = "ipt"
            ipt_evidence[person] = packet.ipt_registry_match
        else:
            ipt_by_person[person] = "non_ipt"
        exempt_by_person[person] = exempt_status(
            positions=person_positions, profiles=profiles_by_person.get(person, []),
            ipt_match=packet.ipt_registry_match,
        )

    values = list(role_status.values())
    legacy: bool | None = (
        True if values and all(value == "technical" for value in values)
        else False if values and all(value == "non_technical" for value in values)
        else None
    )
    technical_count = sum(value == "technical" for value in values)
    nontechnical_count = sum(value == "non_technical" for value in values)
    unclear_count = sum(value == "unclear" for value in values)
    summary = (
        f"Resolved job-title evidence for {len(values)} person(s): {technical_count} technical, "
        f"{nontechnical_count} non-technical, {unclear_count} unclear; "
        f"{sum(value == 'ipt' for value in ipt_by_person.values())} confirmed IPT member(s)."
    )
    return JobTitleState(
        nl_summary=summary, identified_names_staff_listing=names_staff,
        identified_job_data=profile_summaries, identified_job_descriptions=profile_descriptions,
        technical_job_title=legacy,
        technical_rationale=(summary if legacy is None else " ".join(
            value for value in rationale_by_person.values() if value)[:2000]),
        positions_by_person=positions, title_periods_by_person=title_periods,
        selected_job_profiles_by_person=profiles_by_person,
        role_status_by_person=role_status, technical_rationale_by_person=rationale_by_person,
        ipt_status_by_person=ipt_by_person, ipt_registry_evidence_by_person=ipt_evidence,
        exempt_role_status_by_person=exempt_by_person,
        job_profile_status_by_person=profile_status,
        job_title_analysis_mode=mode, job_title_escalation_reasons_by_person=reasons,
    )


def _fallback(state: dict, *, people: list[str], staff_listing: str | None,
              jobs_catalogue: str | None, otl_dir: str, manifest: dict, index: dict,
              reasons: dict[str, list[str]]):
    run_id, case = state.get("run_id", ""), state.get("case") or {}
    inputs = {
        "staff_listing": [staff_listing] if staff_listing else [],
        "jobs_catalogue": [jobs_catalogue] if jobs_catalogue else [],
        "otl_information_dir": otl_dir, "disputed_people": people,
        "escalation_reasons": reasons, "open_questions": list(state.get("open_questions") or []),
    }
    task = (
        "Resolve only the listed unresolved job-title people and gaps. The deterministic pass already "
        "completed everyone else, so do not analyze or return another person. Use person_index() before "
        "opening raw files. For each person return role_status_by_person (technical | non_technical | "
        "unclear), technical_rationale_by_person, selected_job_profiles_by_person, "
        "job_profile_status_by_person (matched | no_match | multiple_close | unclear), positions_by_person, "
        "and nl_summary. Select catalogue profiles only from the active 'Job Catalogue' worksheet. "
        "Never change IPT status; code owns it.\n\n"
        f"UNRESOLVED PEOPLE: {people}\nREASONS: {reasons}"
        + case_context_block(state, specialist="job_title")
        + directive_block(state, "job_title")
        + manifest_block(manifest)
        + person_index_block(index, people)
    )
    executor = CodeExecutor(run_id, "job_title", read_roots=state.get("sandbox_read_roots"),
                            manifest=manifest, person_index=index_json(index))
    return run_agent_loop(
        node="job_title", system=JOB_TITLE_SYSTEM, task=task, executor=executor, inputs=inputs,
        max_steps=8, trace_path=state.get("trace_path"), run_id=run_id,
        case_id=case.get("case_id", ""),
    )


def _merge_fallback(sub: JobTitleState, result: dict, unresolved: list[str]) -> JobTitleState:
    raw = result or {}
    maps = {
        "role_status_by_person": sub.role_status_by_person,
        "technical_rationale_by_person": sub.technical_rationale_by_person,
        "selected_job_profiles_by_person": sub.selected_job_profiles_by_person,
        "job_profile_status_by_person": sub.job_profile_status_by_person,
        "positions_by_person": sub.positions_by_person,
    }
    for field, target in maps.items():
        incoming = raw.get(field) if isinstance(raw.get(field), dict) else {}
        for person in unresolved:
            if person not in incoming:
                continue
            value = incoming[person]
            target[person] = ([str(item) for item in value]
                              if field in {"selected_job_profiles_by_person", "positions_by_person"}
                              and isinstance(value, (list, tuple)) else str(value))
    for person in unresolved:
        if sub.role_status_by_person.get(person) in {"technical", "non_technical"}:
            sub.job_title_escalation_reasons_by_person.pop(person, None)
        positions = sub.positions_by_person.get(person, [])
        profiles = sub.selected_job_profiles_by_person.get(person, [])
        registry = sub.ipt_registry_evidence_by_person.get(person)
        sub.exempt_role_status_by_person[person] = exempt_status(
            positions=positions, profiles=profiles, ipt_match=registry)
        if positions:
            sub.identified_names_staff_listing[person] = positions[-1]
    if isinstance(raw.get("identified_job_data"), dict):
        sub.identified_job_data.update({str(k): str(v) for k, v in raw["identified_job_data"].items()})
    if isinstance(raw.get("identified_job_descriptions"), dict):
        sub.identified_job_descriptions.update(
            {str(k): str(v) for k, v in raw["identified_job_descriptions"].items()})
    sub.job_title_analysis_mode = "code_escalation"
    sub.nl_summary = str(raw.get("nl_summary") or sub.nl_summary)
    values = list(sub.role_status_by_person.values())
    sub.technical_job_title = (
        True if values and all(value == "technical" for value in values)
        else False if values and all(value == "non_technical" for value in values)
        else None
    )
    return sub


def job_title_node(state: dict) -> dict:
    people = disputed_people(state)
    otl_dir = state.get("otl_information_dir") or ""
    staff_listing = resolve_staff_listing(state.get("common_reports_dir") or "")
    jobs_catalogue = _jobs_catalogue(otl_dir)
    manifest = build_manifest([staff_listing, jobs_catalogue, otl_dir], people=people,
                              tab_hints=person_tab_hints(state))
    index = build_person_index([("staff", [staff_listing] if staff_listing else [])], people, manifest)

    findings: list[JobTitlePersonFinding] = []
    reasons: dict[str, list[str]] = {}
    shard_count = 0
    try:
        if not staff_listing or not jobs_catalogue:
            raise ValueError("staff listing or Jobs Catalogue could not be resolved")
        prepared = prepare_evidence(
            state=state, index=index, people=people, staff_listing=staff_listing,
            jobs_catalogue=jobs_catalogue,
        )
        findings, structured_reasons, shard_count = analyze_packets(
            prepared.packets, context=case_context_block(state, specialist="job_title"))
        reasons = dict(prepared.unresolved_reasons)
        for person, values in structured_reasons.items():
            reasons.setdefault(person, []).extend(
                value for value in values if value not in reasons.get(person, []))
    except Exception as exc:
        reasons = {person: [f"structured job-title analysis failed: {type(exc).__name__}: {exc}"]
                   for person in people}
        prepared = PreparedJobTitleEvidence(
            packets=[], unresolved_reasons=reasons,
            staff_source=Path(staff_listing).name if staff_listing else "",
            catalog_source=Path(jobs_catalogue).name if jobs_catalogue else "",
        )

    sub = _structured_state(prepared, findings, reasons, mode="structured")
    unresolved = [person for person in people if sub.role_status_by_person.get(person) == "unclear"
                  or reasons.get(person)]
    fallback_steps = 0
    fallback_status = ""
    if unresolved:
        loop = _fallback(
            state, people=unresolved, staff_listing=staff_listing, jobs_catalogue=jobs_catalogue,
            otl_dir=otl_dir, manifest=manifest, index=index,
            reasons={person: reasons.get(person, []) for person in unresolved},
        )
        fallback_steps, fallback_status = loop.steps, loop.status
        sub = _merge_fallback(sub, loop.result or {}, unresolved)

    remaining = [person for person in people if sub.role_status_by_person.get(person) not in {
        "technical", "non_technical"
    }]
    status = "objective_met" if not remaining else "inconclusive"
    reason = "" if not remaining else "Job-title status remains unresolved for: " + ", ".join(remaining[:10])
    update: dict = {
        "job_title": sub,
        "agent_status": {"job_title": {"status": status, "reason": reason}},
    }
    update.update(evaluate_state({**state, **update}, final=False))
    update.update(trace_event(
        state, "job_title", f"resolved positions ({sub.job_title_analysis_mode})",
        inputs_digest=(f"{len(people)} person(s); staff_listing="
                       f"{Path(staff_listing).name if staff_listing else None}; catalogue="
                       f"{Path(jobs_catalogue).name if jobs_catalogue else None}"),
        key_outputs={
            "mode": sub.job_title_analysis_mode, "structured_shards": shard_count,
            "fallback_people": unresolved, "fallback_steps": fallback_steps,
            "fallback_status": fallback_status, "role_status_by_person": sub.role_status_by_person,
            "ipt_status_by_person": sub.ipt_status_by_person,
            "exempt_role_status_by_person": sub.exempt_role_status_by_person, "status": status,
        },
        nl_summary=sub.nl_summary,
    ))
    return update


__all__ = ["job_title_node"]
