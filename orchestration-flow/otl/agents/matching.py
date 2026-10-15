"""Structured-first CFR evidence mapper; governing chargeability is deferred to agreement retrieval."""
from __future__ import annotations

import os
from pathlib import Path

from otl.chargeability_hooks import evaluate_state
from otl.code_exec import CodeExecutor
from otl.matching_evidence import (MatchingPersonFinding, analyze_packets, prepare_packets)
from otl.prompts import MATCHING_SYSTEM
from otl.react_loop import run_agent_loop
from otl.state import MatchingState
from otl.trace import trace_event
from otl.workbook_manifest import build_manifest, manifest_block
from otl.agents._common import case_context_block, directive_block, pick_latest_by_keyword


def _cfr_mapping(otl_dir: str) -> str | None:
    override = os.getenv("OTL_CFR_MAPPING")
    if override and Path(override).exists():
        return override
    return pick_latest_by_keyword(otl_dir, ["cfr", "mapping"])


def _state_from_findings(state: dict, packets, findings: list[MatchingPersonFinding],
                         reasons: dict[str, list[str]], *, mode: str) -> MatchingState:
    packet_map = {packet.person: packet for packet in packets}
    finding_map = {finding.person: finding for finding in findings}
    status_by_person: dict[str, str] = {}
    profiles_by_person: dict[str, list[str]] = {}
    primary_by_person: dict[str, list[str]] = {}
    secondary_by_person: dict[str, list[str]] = {}
    additional_by_person: dict[str, list[str]] = {}
    labels_by_person: dict[str, list[str]] = {}
    rationale_by_person: dict[str, list[str]] = {}
    references_by_person: dict[str, list[str]] = {}
    chargeability_by_person: dict[str, str] = {}
    legacy_profiles: dict[str, str] = {}
    legacy_primary: dict[str, str] = {}
    legacy_secondary: dict[str, str] = {}
    legacy_additional: dict[str, str] = {}
    legacy_labels: dict[str, str] = {}
    legacy_rationale: dict[str, str] = {}
    legacy_references: dict[str, str] = {}

    for person, packet in packet_map.items():
        finding = finding_map.get(person)
        status_by_person[person] = finding.mapping_status if finding else "unclear"
        chosen = set(finding.selected_candidate_ids if finding else [])
        selected = [candidate for candidate in packet.candidates if candidate.candidate_id in chosen]
        profiles_by_person[person] = list(dict.fromkeys(
            candidate.job_profile_description for candidate in selected))
        primary_by_person[person] = list(dict.fromkeys(candidate.cfr_primary for candidate in selected
                                                      if candidate.cfr_primary))
        secondary_by_person[person] = list(dict.fromkeys(candidate.cfr_secondary for candidate in selected
                                                        if candidate.cfr_secondary))
        additional_by_person[person] = list(dict.fromkeys(candidate.cfr_additional for candidate in selected
                                                         if candidate.cfr_additional))
        labels_by_person[person] = list(dict.fromkeys(candidate.subpart_labels for candidate in selected
                                                     if candidate.subpart_labels))
        rationale_by_person[person] = list(dict.fromkeys(candidate.rationale for candidate in selected
                                                        if candidate.rationale))
        references_by_person[person] = list(dict.fromkeys(candidate.source_reference for candidate in selected
                                                         if candidate.source_reference))
        authority_text = " ".join(
            f"{candidate.cfr_primary} {candidate.cfr_secondary} {candidate.cfr_additional} "
            f"{candidate.subpart_labels} {candidate.rationale} {candidate.notes}"
            for candidate in selected
        ).casefold()
        negative = any(pattern in authority_text for pattern in (
            "not chargeable", "non-chargeable", "non chargeable", "not applicable",
            "not separately chargeable", "covered by overhead",
        ))
        positive = any(
            candidate.cfr_primary or candidate.cfr_secondary or candidate.cfr_additional
            or candidate.subpart_labels for candidate in selected
        )
        chargeability_by_person[person] = (
            "not_chargeable" if negative else "chargeable" if positive else "unclear"
        )
        for candidate in selected:
            key = candidate.job_profile_description
            legacy_profiles[key] = candidate.job_profile_description
            legacy_primary[key] = candidate.cfr_primary
            legacy_secondary[key] = candidate.cfr_secondary
            legacy_additional[key] = candidate.cfr_additional
            legacy_labels[key] = candidate.subpart_labels
            legacy_rationale[key] = candidate.rationale
            legacy_references[key] = candidate.source_reference

    matched = sum(value == "matched" for value in status_by_person.values())
    no_match = sum(value == "no_match" for value in status_by_person.values())
    unresolved = len(status_by_person) - matched - no_match
    summary = (
        f"Mapped CFR reference evidence for {matched} person(s); {no_match} had no supported mapping and "
        f"{unresolved} remain unresolved. Chargeability is deferred to the governing agreement review."
    )
    chargeability_values = set(chargeability_by_person.values())
    cfr_resolution = (
        "chargeable" if "chargeable" in chargeability_values
        else "not_chargeable" if "not_chargeable" in chargeability_values
        else "unclear"
    )
    return MatchingState(
        nl_summary=summary, identified_cfr_category="", cfr_clause="", cfr_resolution="unclear",
        cfr_copas_resolution=cfr_resolution,
        cfr_copas_resolution_by_person=chargeability_by_person,
        matched_job_profiles=legacy_profiles, cfr_primary=legacy_primary,
        cfr_secondary=legacy_secondary, cfr_additional=legacy_additional,
        subpart_labels=legacy_labels, cfr_rationale=legacy_rationale,
        cfr_source_reference=legacy_references,
        cfr_mapping_status_by_person=status_by_person,
        matched_cfr_profiles_by_person=profiles_by_person,
        cfr_primary_by_person=primary_by_person, cfr_secondary_by_person=secondary_by_person,
        cfr_additional_by_person=additional_by_person, subpart_labels_by_person=labels_by_person,
        cfr_rationale_by_person=rationale_by_person,
        cfr_source_reference_by_person=references_by_person,
        matching_analysis_mode=mode, matching_escalation_reasons_by_person=reasons,
    )


def _fallback(state: dict, *, people: list[str], mapping_path: str | None,
              manifest: dict, reasons: dict[str, list[str]]):
    run_id, case = state.get("run_id", ""), state.get("case") or {}
    inputs = {
        "cfr_mapping": mapping_path or "", "unresolved_people": people,
        "escalation_reasons": reasons,
    }
    task = (
        "Resolve only the supplied unresolved people's CFR workbook mappings. Deterministic normalization "
        "completed everyone else. Match the job-title positions, selected profiles, summaries and "
        "descriptions to the workbook's Job Profile Description and Job Title fields. Return "
        "cfr_mapping_status_by_person, matched_cfr_profiles_by_person, cfr_primary_by_person, "
        "cfr_secondary_by_person, cfr_additional_by_person, subpart_labels_by_person, "
        "cfr_rationale_by_person, cfr_source_reference_by_person, cfr_copas_resolution_by_person and "
        "nl_summary. cfr_copas_resolution_by_person may be chargeable only when the mapped CFR fields "
        "provide a substantive regulatory basis, not_chargeable only when the CFR mapping explicitly says "
        "so, otherwise unclear. Do not apply settlement/accounting rules.\n\n"
        f"UNRESOLVED PEOPLE: {people}\nREASONS: {reasons}"
        + case_context_block(state, specialist="matching")
        + directive_block(state, "matching")
        + manifest_block(manifest)
    )
    executor = CodeExecutor(run_id, "matching", read_roots=state.get("sandbox_read_roots"),
                            manifest=manifest)
    return run_agent_loop(
        node="matching", system=MATCHING_SYSTEM, task=task, executor=executor, inputs=inputs,
        max_steps=8, trace_path=state.get("trace_path"), run_id=run_id,
        case_id=case.get("case_id", ""),
    )


def _merge_fallback(sub: MatchingState, result: dict, unresolved: list[str]) -> MatchingState:
    raw = result or {}
    fields = (
        "cfr_mapping_status_by_person", "matched_cfr_profiles_by_person", "cfr_primary_by_person",
        "cfr_secondary_by_person", "cfr_additional_by_person", "subpart_labels_by_person",
        "cfr_rationale_by_person", "cfr_source_reference_by_person",
    )
    for field in fields:
        target = getattr(sub, field)
        incoming = raw.get(field) if isinstance(raw.get(field), dict) else {}
        for person in unresolved:
            if person not in incoming:
                continue
            value = incoming[person]
            target[person] = ([str(item) for item in value]
                              if isinstance(value, (list, tuple, set)) else [str(value)])
    statuses = raw.get("cfr_mapping_status_by_person") if isinstance(
        raw.get("cfr_mapping_status_by_person"), dict) else {}
    for person in unresolved:
        if person in statuses:
            sub.cfr_mapping_status_by_person[person] = str(statuses[person])
        if sub.cfr_mapping_status_by_person.get(person) in {"matched", "no_match"}:
            sub.matching_escalation_reasons_by_person.pop(person, None)
    incoming_resolution = (raw.get("cfr_copas_resolution_by_person")
                           if isinstance(raw.get("cfr_copas_resolution_by_person"), dict) else {})
    for person in unresolved:
        if person in incoming_resolution:
            sub.cfr_copas_resolution_by_person[person] = str(incoming_resolution[person])
    values = set(sub.cfr_copas_resolution_by_person.values())
    sub.cfr_copas_resolution = (
        "chargeable" if "chargeable" in values
        else "not_chargeable" if "not_chargeable" in values else "unclear"
    )
    sub.matching_analysis_mode = "code_escalation"
    sub.nl_summary = str(raw.get("nl_summary") or sub.nl_summary)
    return sub


def matching_node(state: dict) -> dict:
    otl_dir = state.get("otl_information_dir") or ""
    mapping_path = _cfr_mapping(otl_dir)
    manifest = build_manifest([mapping_path, otl_dir])
    try:
        if not mapping_path:
            raise ValueError("CFR mapping workbook could not be resolved")
        packets = prepare_packets(state, mapping_path)
        findings, reasons, shard_count = analyze_packets(
            packets, context=case_context_block(state, specialist="matching"))
    except Exception as exc:
        job = state.get("job_title") or {}
        if hasattr(job, "model_dump"):
            job = job.model_dump()
        people = list((job.get("positions_by_person") or {}).keys())
        packets, findings, shard_count = [], [], 0
        reasons = {person: [f"structured CFR mapping failed: {type(exc).__name__}: {exc}"]
                   for person in people}

    sub = _state_from_findings(state, packets, findings, reasons, mode="structured")
    unresolved = [person for person, status in sub.cfr_mapping_status_by_person.items()
                  if status in {"multiple_close", "unclear"} or reasons.get(person)]
    fallback_steps = 0
    fallback_status = ""
    if unresolved:
        loop = _fallback(
            state, people=unresolved, mapping_path=mapping_path, manifest=manifest,
            reasons={person: reasons.get(person, []) for person in unresolved},
        )
        fallback_steps, fallback_status = loop.steps, loop.status
        sub = _merge_fallback(sub, loop.result or {}, unresolved)

    remaining = [person for person, status in sub.cfr_mapping_status_by_person.items()
                 if status in {"multiple_close", "unclear"}]
    status = "objective_met" if not remaining else "inconclusive"
    reason = "" if not remaining else "CFR mapping remains unresolved for: " + ", ".join(remaining[:10])
    update: dict = {
        "matching": sub,
        "agent_status": {"matching": {"status": status, "reason": reason}},
    }
    update.update(evaluate_state({**state, **update}, final=False))
    update.update(trace_event(
        state, "matching", f"mapped CFR evidence ({sub.matching_analysis_mode})",
        inputs_digest=(f"{len(packets)} person packet(s); "
                       f"cfr_mapping={Path(mapping_path).name if mapping_path else None}"),
        key_outputs={
            "mode": sub.matching_analysis_mode, "structured_shards": shard_count,
            "mapping_status_by_person": sub.cfr_mapping_status_by_person,
            "cfr_copas_resolution": sub.cfr_copas_resolution,
            "fallback_people": unresolved, "fallback_steps": fallback_steps,
            "fallback_status": fallback_status, "status": status,
        },
        nl_summary=sub.nl_summary,
    ))
    return update


__all__ = ["matching_node"]
