"""Deterministic monthly org-tree normalization and above-first-level classification."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field


class HierarchyPeriodFinding(BaseModel):
    period: str
    person_key: str
    reports_to: str = ""
    direct_report_count: int = 0
    manager_report_count: int = 0
    above_first_level: bool
    source_rows: list[int] = Field(default_factory=list)


class SupervisionEvidence(BaseModel):
    above_first_level: dict[str, bool] = Field(default_factory=dict)
    supervision_status_by_person: dict[str, str] = Field(default_factory=dict)
    supervision_record_status_by_person: dict[str, str] = Field(default_factory=dict)
    hierarchy_periods_by_person: dict[str, list[HierarchyPeriodFinding]] = Field(default_factory=dict)
    reports_summary: dict[str, str] = Field(default_factory=dict)
    unresolved_reasons: dict[str, list[str]] = Field(default_factory=dict)


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _norm(value: Any) -> str:
    return re.sub(r"\D", "", _clean(value)) or _clean(value).casefold()


def _period_key(value: str) -> tuple[int, str]:
    months = {name.casefold(): i for i, name in enumerate(
        ("January", "February", "March", "April", "May", "June", "July", "August",
         "September", "October", "November", "December"), 1)}
    text = _clean(value)
    return months.get(text.casefold(), 99), text


def _selected_keys(index: dict, person: str) -> set[str]:
    keys: set[str] = set()
    for source in index.get("sources") or []:
        if source.get("role") != "staff":
            continue
        selected_names = {match.get("name_in_file") for match in
                          (source.get("selected") or {}).get(person) or []}
        for entry in source.get("entries") or []:
            for row in entry.get("staff_rows") or []:
                if row[0] in selected_names and row[1]:
                    keys.add(_norm(row[1]))
    return keys


def analyze_supervision(index: dict, people: list[str]) -> SupervisionEvidence:
    # period -> employee key -> compact rows. Multiple conflicting manager values within a month remain
    # visible and make only the affected person unresolved.
    periods: dict[str, dict[str, list[tuple]]] = defaultdict(lambda: defaultdict(list))
    has_reports_to_role = False
    for source in index.get("sources") or []:
        if source.get("role") != "staff":
            continue
        for entry in source.get("entries") or []:
            if (entry.get("columns") or {}).get("reports_to"):
                has_reports_to_role = True
            for row in entry.get("staff_rows") or []:
                name, person_key, reports_to, _manager, position, _profile, period, *_rest = row
                key = _norm(person_key)
                if key:
                    periods[_clean(period) or "undated"][key].append(
                        (_norm(reports_to), name, position, int(row[-1] or 0)))

    above: dict[str, bool] = {}
    statuses: dict[str, str] = {}
    record_status: dict[str, str] = {}
    findings_by_person: dict[str, list[HierarchyPeriodFinding]] = {}
    summaries: dict[str, str] = {}
    reasons: dict[str, list[str]] = defaultdict(list)

    if not has_reports_to_role:
        for person in people:
            statuses[person] = "unresolved"
            record_status[person] = "missing"
            reasons[person].append("staff listing has no reporting-chain column")
        return SupervisionEvidence(
            supervision_status_by_person=statuses,
            supervision_record_status_by_person=record_status,
            unresolved_reasons=dict(reasons),
        )

    for person in people:
        keys = _selected_keys(index, person)
        if not keys:
            statuses[person] = "unresolved"
            record_status[person] = "missing"
            reasons[person].append("person has no resolved staff-listing employee key")
            continue
        if len(keys) > 1:
            statuses[person] = "unresolved"
            record_status[person] = "ambiguous"
            reasons[person].append("person maps to multiple employee keys")
            continue
        person_key = next(iter(keys))
        period_findings: list[HierarchyPeriodFinding] = []
        for period in sorted(periods, key=_period_key):
            rows = periods[period]
            own = rows.get(person_key) or []
            if not own:
                continue
            own_managers = {row[0] for row in own if row[0]}
            if len(own_managers) > 1:
                reasons[person].append(f"conflicting reporting relationships in {period}")
                continue
            direct_keys = [key for key, values in rows.items()
                           if any(value[0] == person_key for value in values)]
            reports_to_values = {value[0] for values in rows.values() for value in values if value[0]}
            manager_reports = [key for key in direct_keys if key in reports_to_values]
            period_findings.append(HierarchyPeriodFinding(
                period=period, person_key=person_key,
                reports_to=next(iter(own_managers), ""), direct_report_count=len(set(direct_keys)),
                manager_report_count=len(set(manager_reports)),
                above_first_level=bool(direct_keys and manager_reports),
                source_rows=sorted({value[3] for value in own if value[3]}),
            ))
        if not period_findings:
            statuses[person] = "unresolved"
            record_status[person] = "missing" if not reasons.get(person) else "ambiguous"
            reasons[person].append("no usable hierarchy snapshot for the person")
            continue
        values = {item.above_first_level for item in period_findings}
        findings_by_person[person] = period_findings
        record_status[person] = "found"
        if len(values) > 1 or reasons.get(person):
            statuses[person] = "unresolved"
            record_status[person] = "ambiguous"
            if len(values) > 1:
                reasons[person].append("supervision level changes across monthly snapshots")
            continue
        resolved = next(iter(values))
        above[person] = resolved
        statuses[person] = "above_first_level" if resolved else "not_above_first_level"
        max_direct = max(item.direct_report_count for item in period_findings)
        max_managers = max(item.manager_report_count for item in period_findings)
        summaries[person] = (
            f"{max_direct} direct report(s); {max_managers} direct report(s) also have reports; "
            f"{len(period_findings)} monthly snapshot(s) reviewed."
        )

    return SupervisionEvidence(
        above_first_level=above, supervision_status_by_person=statuses,
        supervision_record_status_by_person=record_status,
        hierarchy_periods_by_person=findings_by_person, reports_summary=summaries,
        unresolved_reasons=dict(reasons),
    )


__all__ = ["HierarchyPeriodFinding", "SupervisionEvidence", "analyze_supervision"]
