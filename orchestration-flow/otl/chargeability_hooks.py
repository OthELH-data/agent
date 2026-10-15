"""Declarative GR/SE chargeability hook evaluation.

Nodes own facts.  This module owns only rule matching and conservative final aggregation.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from shared.hooks_config import chargeability_hook_definitions

from otl.state import ChargeScope, ChargeabilityResolution, HookEvaluation, HookFact


_FACT_AGENT = {
    "location_status": "location",
    "role_status": "job_title",
    "ipt_status": "job_title",
    "exempt_role_status": "job_title",
    "job_profile_status": "job_title",
    "supervision_status": "supervision",
    "supervision_record_status": "supervision",
    "precedence_status": "wiki",
}


def _dump(value: Any) -> dict:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return dict(value) if isinstance(value, dict) else {}


def hook_rules() -> dict[str, dict[str, Any]]:
    return {
        str(key): value.model_dump(mode="python")
        for key, value in chargeability_hook_definitions().items()
    }


def _ref(row: dict) -> str:
    src = _dump(row.get("source_ref"))
    return f"{src.get('workbook', '')}!{src.get('sheet', '')}:{src.get('excel_row', '')}"


def _month(value: Any) -> str:
    text = str(value or "")
    return text[:7] if len(text) >= 7 else ""


def _person_role_facts(state: dict, person: str) -> dict[str, HookFact]:
    out: dict[str, HookFact] = {}
    job = _dump(state.get("job_title"))
    if job:
        for fact, field in (
            ("role_status", "role_status_by_person"),
            ("ipt_status", "ipt_status_by_person"),
            ("exempt_role_status", "exempt_role_status_by_person"),
            ("job_profile_status", "job_profile_status_by_person"),
        ):
            value = (job.get(field) or {}).get(person)
            if value:
                out[fact] = HookFact(value=value, source_node="job_title")

    loc = _dump(state.get("location"))
    if loc:
        explicit = (loc.get("location_status") or {}).get(person)
        if explicit:
            out["location_status"] = HookFact(
                value=explicit, status=("unresolved" if explicit == "unresolved" else "resolved"),
                source_node="location", rationale=str((loc.get("location_rationale") or {}).get(person) or ""),
            )
            explicit = True
        else:
            explicit = False
        pob = loc.get("found_in_pob") or {}
        heli = loc.get("found_in_heli_logs") or {}
        if not explicit and (person in pob or person in heli):
            onsite = bool(pob.get(person)) or bool(heli.get(person))
            out["location_status"] = HookFact(
                value="onsite" if onsite else "offsite", source_node="location")

    sup = _dump(state.get("supervision"))
    if sup:
        explicit_status = (sup.get("supervision_status_by_person") or {}).get(person)
        record_status = (sup.get("supervision_record_status_by_person") or {}).get(person)
        if explicit_status:
            out["supervision_status"] = HookFact(
                value=explicit_status,
                status=("unresolved" if explicit_status == "unresolved" else "resolved"),
                source_node="supervision",
                rationale=str((sup.get("supervision_escalation_reasons_by_person") or {}).get(person) or ""),
            )
        if record_status == "missing":
            out["supervision_record_status"] = HookFact(
                value="missing", source_node="supervision",
                rationale="No usable staff-listing hierarchy record was found for this person.",
            )
        above = sup.get("above_first_level") or {}
        if not explicit_status and person in above:
            out["supervision_status"] = HookFact(
                value="above_first_level" if above[person] else "not_above_first_level",
                source_node="supervision",
            )
        elif not explicit_status and sup.get("needs_human_review"):
            out["supervision_record_status"] = HookFact(
                value="missing", source_node="supervision",
                rationale=str(sup.get("review_reason") or "supervision record unavailable"),
            )

    agreement = _dump(state.get("agreement_retriever"))
    matching = _dump(state.get("matching"))
    cfr_resolution = ((matching.get("cfr_copas_resolution_by_person") or {}).get(person)
                      or matching.get("cfr_copas_resolution") or "unclear")
    cfr_refs = list((matching.get("cfr_source_reference_by_person") or {}).get(person) or [])
    if cfr_resolution:
        out["cfr_copas_resolution"] = HookFact(
            value=cfr_resolution, source_node="matching", source_refs=[str(value) for value in cfr_refs],
        )
    agreement_resolution = str(agreement.get("agreement_chargeability") or "unclear")
    agreement_refs = [str(item.get("chunk_id")) for item in
                      (agreement.get("agreement_chargeability_citations") or []) if item.get("chunk_id")]
    if agreement:
        out["agreement_chargeability"] = HookFact(
            value=agreement_resolution, source_node="agreement_retriever", source_refs=agreement_refs)
    if "chargeable" in {cfr_resolution, agreement_resolution}:
        legal_basis: bool | None = True
    elif "not_chargeable" in {cfr_resolution, agreement_resolution}:
        legal_basis = False
    else:
        legal_basis = None
    out["legal_chargeability_basis"] = HookFact(
        value=legal_basis, source_node="matching+agreement_retriever",
        source_refs=[*cfr_refs, *agreement_refs],
        rationale=(f"CFR/COPAS mapping={cfr_resolution}; "
                   f"agreement authority={agreement_resolution}."),
    )

    wiki = _dump(state.get("wiki"))
    if wiki:
        out["precedence_status"] = HookFact(
            value=str(wiki.get("precedence_status") or "unavailable_or_unclear"),
            source_node="wiki",
        )
    return out


def build_charge_scopes(state: dict) -> list[ChargeScope]:
    jade = _dump(state.get("jade_analyzer"))
    rows = [_dump(row) for row in jade.get("ledger_rows") or []]
    monthly = [_dump(item) for item in jade.get("mcoh_monthly_facts") or []]
    monthly_by_key = {(str(x.get("person")), str(x.get("submission_month"))): x for x in monthly}

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for index, row in enumerate(rows):
        person = str(row.get("person") or "")
        mcoh_status = str(row.get("mcoh_status") or "unresolved")
        period = _month(row.get("entry_date_sl")) if mcoh_status == "mcoh" else ""
        subject = person or f"unattributed:{_ref(row) or index}"
        groups[(subject, mcoh_status, period)].append(row)

    scopes: list[ChargeScope] = []
    for (subject, mcoh_status, period), selected in groups.items():
        person = "" if subject.startswith("unattributed:") else subject
        refs = [_ref(row) for row in selected]
        amount = sum(float(row.get("amount") or 0.0) for row in selected)
        facts: dict[str, HookFact] = {}
        if mcoh_status in {"mcoh", "non_mcoh"}:
            facts["mcoh"] = HookFact(
                value=(mcoh_status == "mcoh"), source_node="jade_analyzer", source_refs=refs)
        if person:
            facts.update(_person_role_facts(state, person))
        if person and mcoh_status == "mcoh" and period:
            item = monthly_by_key.get((person, period))
            if item:
                agreement = _dump(state.get("agreement_retriever"))
                threshold_hours = agreement.get("mcoh_threshold_hours")
                threshold = float(threshold_hours) if threshold_hours is not None else float(
                    item.get("threshold_hours") or 8.0)
                facts["mcoh_below_monthly_threshold"] = HookFact(
                    value=float(item.get("hours") or 0.0) < threshold,
                    source_node=("agreement_retriever" if threshold_hours is not None else "jade_analyzer"),
                    source_refs=refs,
                )
        scopes.append(ChargeScope(
            scope_id=f"{subject}|{mcoh_status}|{period or 'all'}",
            person=person or None, period=period or None, row_refs=refs,
            questioned_amount=round(amount, 2), facts=facts,
        ))
    return scopes


def _condition_result(scope: ChargeScope, conditions: dict[str, list[Any]], state: dict) -> tuple[str, list[str], dict[str, HookFact]]:
    missing: list[str] = []
    support: dict[str, HookFact] = {}
    for key, allowed in conditions.items():
        fact = scope.facts.get(key)
        if fact is None:
            if key == "supervision_record_status" and state.get("supervision"):
                supervision = _dump(state.get("supervision"))
                if scope.person in (supervision.get("above_first_level") or {}):
                    return "not_matched", [], support
            agent = _FACT_AGENT.get(key)
            if agent and state.get(agent):
                return "unresolved", [key], support
            missing.append(key)
            continue
        support[key] = fact
        if fact.status == "unresolved":
            return "unresolved", [key], support
        if fact.value not in list(allowed or []):
            return "not_matched", [], support
    return ("pending" if missing else "matched"), missing, support


def evaluate_hooks(state: dict, scopes: list[ChargeScope] | None = None) -> list[HookEvaluation]:
    scopes = scopes or build_charge_scopes(state)
    out: list[HookEvaluation] = []
    for scope in scopes:
        for hook_id, rule in hook_rules().items():
            status, missing, support = _condition_result(scope, rule.get("all") or {}, state)
            if status == "matched" and rule.get("any"):
                any_results = [_condition_result(scope, group or {}, state) for group in rule["any"]]
                matched = next((r for r in any_results if r[0] == "matched"), None)
                if matched:
                    support.update(matched[2])
                elif any(r[0] == "pending" for r in any_results):
                    status = "pending"
                    missing = sorted({x for r in any_results for x in r[1]})
                elif any(r[0] == "unresolved" for r in any_results):
                    status = "unresolved"
                else:
                    status = "not_matched"
            out.append(HookEvaluation(
                hook_id=hook_id, scope_id=scope.scope_id,
                status="fired" if status == "matched" else status,
                missing_conditions=missing, supporting_facts=support,
                decision=str(rule.get("decision") or ""),
                disposition=str(rule.get("disposition") or ""),
                outcome=str(rule.get("outcome") or ""),
            ))
    return out


def required_agents(evaluations: list[HookEvaluation]) -> list[str]:
    wanted: list[str] = []
    for raw in evaluations:
        item = raw if isinstance(raw, HookEvaluation) else HookEvaluation.model_validate(raw)
        if item.status != "pending":
            continue
        for fact in item.missing_conditions:
            agent = _FACT_AGENT.get(fact)
            if agent and agent not in wanted and agent not in {"wiki"}:
                wanted.append(agent)
    return wanted


def resolve_hooks(scopes: list[ChargeScope], evaluations: list[HookEvaluation], *, final: bool) -> ChargeabilityResolution:
    by_scope = defaultdict(list)
    uncertain_by_scope: dict[str, bool] = defaultdict(bool)
    for item in evaluations:
        if item.status == "fired":
            by_scope[item.scope_id].append(item)
        elif item.status in {"pending", "unresolved"}:
            uncertain_by_scope[item.scope_id] = True
    scope_by_id = {scope.scope_id: scope for scope in scopes}
    person_scopes: dict[str, list[tuple[str, float, list[str]]]] = defaultdict(list)
    unresolved_people: list[str] = []
    fired_hooks: dict[str, list[str]] = {}

    for scope in scopes:
        person_key = scope.person or scope.scope_id
        facts = scope.facts
        threshold = facts.get("mcoh_below_monthly_threshold")
        fired = by_scope.get(scope.scope_id, [])
        dispositions = {item.disposition for item in fired if item.disposition}
        hook_ids = [item.hook_id for item in fired]

        # The supplied business rule is a decisive pre-gate for identified-person MCOH months.
        if threshold and threshold.value is True:
            dispositions.add("accept")
            hook_ids.append("MCOH_MONTHLY_THRESHOLD")

        terminal = {d for d in dispositions if d in {"deny", "accept", "partial"}}
        has_review = "needs_evidence" in dispositions
        if has_review or len(terminal) != 1 or (final and uncertain_by_scope.get(scope.scope_id, False)):
            unresolved_people.append(person_key)
            person_scopes[person_key].append(("needs_evidence", float(scope.questioned_amount or 0), hook_ids))
        else:
            disposition = next(iter(terminal))
            person_scopes[person_key].append((disposition, float(scope.questioned_amount or 0), hook_ids))
        fired_hooks[scope.scope_id] = hook_ids

    people_results: dict[str, dict[str, Any]] = {}
    deny_amount = accept_amount = unresolved_amount = 0.0
    for person, results in person_scopes.items():
        dispositions = {r[0] for r in results}
        amount = sum(r[1] for r in results)
        hooks = list(dict.fromkeys(h for r in results for h in r[2]))
        if "needs_evidence" in dispositions or len(dispositions) != 1:
            disposition = "needs_evidence"
            unresolved_amount += amount
            if person not in unresolved_people:
                unresolved_people.append(person)
        else:
            disposition = next(iter(dispositions))
            if disposition == "deny":
                deny_amount += amount
            elif disposition == "accept":
                accept_amount += amount
            else:
                deny_amount += amount / 2
                accept_amount += amount / 2
        people_results[person] = {"disposition": disposition, "amount": round(amount, 2), "hooks": hooks}

    if unresolved_people:
        disposition = "needs_evidence"
        status = "needs_evidence" if final else "pending"
    else:
        resolved = {row["disposition"] for row in people_results.values()}
        if not resolved:
            disposition, status = "needs_evidence", ("needs_evidence" if final else "pending")
        elif resolved == {"deny"}:
            disposition, status = "deny", "resolved"
        elif resolved == {"accept"}:
            disposition, status = "accept", "resolved"
        else:
            disposition, status = "partial", "resolved"

    summary = (
        f"{len(people_results) - len(set(unresolved_people))} resolved subject(s), "
        f"{len(set(unresolved_people))} unresolved; deny={deny_amount:.2f}, "
        f"accept={accept_amount:.2f}, unresolved={unresolved_amount:.2f}."
    )
    return ChargeabilityResolution(
        status=status, disposition=disposition, fired_hooks=fired_hooks,
        person_results=people_results, resolved_deny_amount=round(deny_amount, 2),
        resolved_accept_amount=round(accept_amount, 2), unresolved_amount=round(unresolved_amount, 2),
        unresolved_people=list(dict.fromkeys(unresolved_people)), summary=summary,
    )


def evaluate_state(state: dict, *, final: bool = False) -> dict[str, Any]:
    scopes = build_charge_scopes(state)
    evaluations = evaluate_hooks(state, scopes)
    resolution = resolve_hooks(scopes, evaluations, final=final)
    legal_by_person: dict[str, bool | None] = {}
    for scope in scopes:
        if scope.person and "legal_chargeability_basis" in scope.facts:
            legal_by_person.setdefault(scope.person, scope.facts["legal_chargeability_basis"].value)
    values = set(legal_by_person.values())
    legal_case = True if True in values else False if False in values else None
    return {
        "chargeability_scopes": scopes,
        "chargeability_hook_evaluations": evaluations,
        "chargeability_resolution": resolution,
        "legal_chargeability_basis": legal_case,
        "legal_chargeability_basis_by_person": legal_by_person,
    }


__all__ = ["build_charge_scopes", "evaluate_hooks", "evaluate_state", "hook_rules",
           "required_agents", "resolve_hooks"]
