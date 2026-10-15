"""orchestrator_node — the sequential hub-and-spoke supervisor (high tier).

The hub between the deterministic prep spine and the terminal drafter. `jade_analyzer` runs in the
spine (before this node), so on each pass the orchestrator reads every specialist's returned sub-state
+ status — jade's amount reconstruction already among them — then decides ONE of two things:

  * dispatch exactly ONE specialist next (location | job_title | matching | supervision) with a short
    focused `instruction`, OR
  * finalize to the drafter with `final_mode` "terminal" (a rebuttal is supportable) or
    "needs_evidence" (a real gap blocks any defensible conclusion).

The chosen specialist returns here (static edge), and the orchestrator re-decides with the new
findings. Composite questions are expanded into the union of their specialist paths while retaining
their required dependencies — in particular, `location` always precedes `job_title`, and when
`hierarchy` is also raised, `supervision` precedes `matching`.

Single-visit rule: each specialist is dispatched AT MOST ONCE. The orchestrator only advertises the
not-yet-run specialists as options, and refuses to route to one it has already run — so it can never
revisit the same node. Once every relevant specialist has run, it finalizes. (This makes the flow
near-linear and bounded independently of the loop guard below.)

Loop guard (decision #3): the node also counts its own passes (`orchestrator_hops`) and each
specialist's dispatches (`agent_run_counts`). Hitting MAX_HOPS or MAX_AGENT_RUNS forces a graceful
finalize to the drafter with `final_mode="needs_evidence"` — the LLM is NOT consulted on that hop, so
the graph can never infinite-loop. `recursion_limit` (run.py) is only the engine-level seatbelt above
this.

Specialists never force HITL anymore; they only report `agent_status`. The orchestrator alone owns the
terminal decision and the `hitl_*` channels when it finalizes to `needs_evidence`.

PASSED TO THE LLM (structured output):
  * system : ORCHESTRATOR_SYSTEM (prompts.py)
  * user   : the compact DECISION SURFACE (`_digest_for_llm`) — case header, auditor question +
             narrative, extractor synthesis narrative, the verify_evidence flags, open_questions, each
             specialist's `agent_status` + salient row-free findings so far, run counts, and the
             `available_specialists` it may dispatch (single-visit) — plus the hop number and the
             dispatch/finalize instruction. NEVER raw workbook rows.
  * schema : _OrchestratorDecision — its Field descriptions reach the model via the JSON schema.

Deterministic guards enforced in CODE before the LLM is consulted (so the graph can never infinite-loop
or mis-finalize these cases):
  * loop guard                    — MAX_HOPS / MAX_AGENT_RUNS tripped → needs_evidence.
  * no-dispute-category guard     — category_check found no dispute category → needs_evidence (no routing
                                    path; a human confirms which data sources to check).
  * location → job_title pin      — whenever category_check identified `location` and `location` has
                                    run, dispatch `job_title` before any later specialist (physical
                                    presence alone cannot close a location question; a technical role
                                    is chargeable by function). See `_invoke_job_title_after_location`.
                                    This is the only guard that DISPATCHES rather than finalizes.
  * technical-role override (S1)  — matching found no CFR category but job_title is technical → terminal
                                    (chargeable by function, not location).
  * non-technical-role defer (S2) — matching found no CFR category and job_title is NOT technical →
                                    needs_evidence (AI cannot judge technical defensibility).
Dispatch is also category-aware: `available_specialists` is filtered to the specialists RELEVANT to the
dispute category(ies) (`_CATEGORY_TARGETS`), so e.g. `location` is never offered for a job_title-only
question.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field
from otl.chargeability_hooks import required_agents

from otl import llm_adapter as llm
from otl.prompts import ORCHESTRATOR_SYSTEM
from otl.state import OrchestratorState
from otl.trace import trace_event
from otl.agents._common import (active_auditor_text, compact_classified_exception,
                                people_coverage, rebuttal_context_block, sub_get,
                                synthesis_narrative)

# Loop-guard caps (module constants so a test can lower them). Hitting either forces finalize.
MAX_HOPS = 50
MAX_AGENT_RUNS = 50

# The specialists the hub can dispatch, listed in the prescribed composite dependency order.
# jade_analyzer is NOT here — it runs once in the spine, before
# the orchestrator, and is never re-dispatched. Each of these is likewise dispatched at most once
# (single-visit rule): the orchestrator advertises only the not-yet-run ones and refuses to revisit.
_DISPATCH_TARGETS = ("location", "job_title", "supervision", "matching")
_DIGEST_LIMIT = 12000

# Category-aware routing (single source of truth for which specialists a dispute category can reach).
# The auditor's dispute category(ies) from category_check decide the routing path. A composite case gets
# the union of its category paths, in `_DISPATCH_TARGETS` dependency order:
#   * job_title -> resolve the role (job_title) and, if technical, defend it on the OTL / CFR basis
#                  (matching). LOCATION IS NOT OFFERED: a technical role is chargeable by function, so
#                  POB / physical presence is not dispositive for a pure job-title question.
#   * location  -> the location specialist (POB/heli), plus job_title+matching because "were they even
#                  offsite technical labor?" is the fallback basis when POB can't place them.
#   * hierarchy -> the supervision specialist (above/below first level from the org tree), in addition
#                  to any specialists needed by a co-raised location or job-title question.
# Empty / unrecognized category -> () : no routing path. Handled by the no-dispute-category guard, which
# finalizes to HITL asking the human which data sources to check (never a blind dispatch-all).
_CATEGORY_TARGETS: dict[str, tuple[str, ...]] = {
    "job_title": ("job_title", "matching"),
    "location": ("location", "job_title", "matching"),
    "hierarchy": ("supervision",),
}


class _OrchestratorDecision(BaseModel):
    """The supervisor's per-hop decision. The Field descriptions below become the JSON schema the
    model fills, so they are the model's contract for each field."""
    next_agent: str = Field(
        default="",
        description="The single specialist to dispatch next — one of the current available_specialists "
                    "(location | job_title | matching | supervision) — OR 'drafter' to finalize. "
                    "jade_analyzer already ran in the spine and is never a valid choice.")
    finalize: bool = Field(
        default=False,
        description="True to route to the drafter and end (equivalent to next_agent=='drafter').")
    final_mode: str = Field(
        default="",
        description="Only when finalizing: 'terminal' if a defensible rebuttal is supportable from the "
                    "findings, or 'needs_evidence' if a real gap blocks any defensible conclusion.")
    instruction: str = Field(
        default="",
        description="When dispatching: a short, focused directive telling the chosen specialist exactly "
                    "what to establish this run.")
    context_to_pass: str = Field(
        default="",
        description="When dispatching: concise additional current-case context the specialist needs "
                    "beyond its cherry-picked prior-cycle context and current questioning ground. "
                    "Include dependencies or findings from other agents when useful; omit raw rows and "
                    "unrelated history.")
    hitl_reason: str = Field(
        default="",
        description="Required when final_mode=='needs_evidence': the concrete reason a human must "
                    "review (the specific unresolved gap or blocker).")
    hitl_stage: str = Field(
        default="", description="Optional; the stage that deferred. Defaults to 'orchestrator'.")
    rationale: str = Field(
        default="", description="One-line justification for this choice (recorded in the decision_log).")
    nl_summary: str = Field(
        default="", description="A concise analysis of the routing decision, decisive findings, "
                                "unresolved work, and material uncertainty. Do not list every detail.")


def _dump(value: Any) -> dict:
    """Serialize a sub-state (Pydantic model or already a dict) to a plain dict."""
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value if isinstance(value, dict) else {}


def _trunc(s: str, limit: int) -> str:
    s = s or ""
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _questions(state: dict) -> list[str]:
    """The auditor's dispute category(ies) from category_check — the master channel, falling back to the
    category_check sub-state. Drives category-aware dispatch and the no-dispute-category guard."""
    questions = list(state.get("auditor_question")
                     or sub_get(state, "category_check", "auditor_question", []) or [])
    evaluations = list(state.get("chargeability_hook_evaluations") or [])
    for agent in required_agents(evaluations):
        question = {
            "location": "location", "job_title": "job_title",
            "matching": "job_title", "supervision": "hierarchy",
        }.get(agent)
        if question and question not in questions:
            questions.append(question)
    return questions


def _relevant_targets(questions: list[str]) -> list[str]:
    """The specialists the auditor's dispute category(ies) can reach, in `_DISPATCH_TARGETS` order.
    Empty when no category is recognized (no routing path — see `_no_dispute_category`)."""
    allowed: set[str] = set()
    for q in questions or []:
        allowed.update(_CATEGORY_TARGETS.get(q, ()))
    return [t for t in _DISPATCH_TARGETS if t in allowed]


def _available_targets(run_counts: dict, questions: list[str], state: dict | None = None) -> list[str]:
    """The specialists RELEVANT to the dispute category(ies) that have not yet been dispatched
    (single-visit rule), in `_DISPATCH_TARGETS` order. jade already ran in the spine. A specialist
    irrelevant to the question raised is never offered — e.g. `location` is not offered for a pure
    job_title dispute (POB is not dispositive for a technical-role question)."""
    relevant = _relevant_targets(questions)
    revisit = (state or {}).get("rebuttal_specialist_revisit") or {}
    has_history = bool((state or {}).get("rebuttal_history"))
    return [
        target for target in relevant
        if int(run_counts.get(target, 0) or 0) == 0
        and (not has_history or bool(revisit.get(target, True)))
    ]


def _model_error_agent(state: dict) -> str:
    """The first specialist whose agentic model call failed after transport retries."""
    for agent, status in (state.get("agent_status") or {}).items():
        reason = str((status or {}).get("reason") or "")
        if str((status or {}).get("status") or "") == "inconclusive" and (
            "(llm_error)" in reason or "LLM request failed:" in reason
        ):
            return str(agent)
    return ""


def _adaptive_prescribed_target(state: dict, run_counts: dict) -> str:
    """Return the next zero-judgement hop for a registry-prescribed route.

    The category registry already fixes the dependency order. Paying a high-tier
    supervisor to repeat that order adds no reasoning value. If any completed
    specialist is inconclusive, return empty and preserve the agentic supervisor
    path. Every recognized composite is mechanically prescribed by the union of
    its categories. The base role path is location -> job_title -> matching; when
    hierarchy is raised, supervision is inserted before matching. Specialists not
    required by the selected categories are omitted.
    """
    questions = _questions(state)
    recognized = [q for q in questions if q in _CATEGORY_TARGETS]
    if not recognized:
        return ""
    if any(
        str((status or {}).get("status") or "") == "inconclusive"
        for status in (state.get("agent_status") or {}).values()
    ):
        return ""
    available = _available_targets(run_counts, recognized, state)
    return available[0] if available else ""


def _adaptive_instruction(state: dict, target: str) -> str:
    """Focused default directive for a registry-prescribed specialist hop.

    Every specialist already receives ``open_questions`` as non-gating hints. Repeating the first four
    in every directive contaminates unrelated judgements (for example, asking matching to defer its CFR
    classification over a location-coverage question). The deterministic route therefore states only
    the target's owned objective.
    """
    objectives = {
        "location": (
            "Establish each disputed person's dated POB and travel location for the charged period, "
            "including coverage gaps and whether the dated records overlap the charged work."
        ),
        "job_title": (
            "Resolve each disputed person's position and duties, then determine whether the work is "
            "technical in nature under the Technical Employees definition."
        ),
        "matching": (
            "Trace the resolved role to the CFR-mapping source and applicable verified agreement "
            "authority; distinguish a specific CFR route from the broader technical-employee basis."
        ),
        "supervision": (
            "Resolve each disputed person's reporting level for the charged population and determine "
            "whether the person is above first-level supervision."
        ),
    }
    return objectives.get(target, f"Resolve the {target} evidence question.")


def _invoke_job_title_after_location(state: dict) -> bool:
    """True iff category_check included `location` in the auditor's question set.

    A location question cannot be closed on physical presence alone: a role that is TECHNICAL IN NATURE
    is chargeable by FUNCTION, not location, so "was the person on the charged asset?" always carries
    "and what was their role?" behind it. When this is True the orchestrator dispatches `job_title`
    DETERMINISTICALLY on the hop after `location` returns (see `orchestrator_node`), including when
    location is part of a composite case.

    This is intentionally asymmetric: there is no job_title→location equivalent, because whether a role
    is technical does not depend on where the person was — hence `_CATEGORY_TARGETS["job_title"]` omits
    `location` entirely.

    Derived from `auditor_question` on every read, so it cannot drift from what category_check found;
    it is mirrored onto `OrchestratorState.invoke_job_title_after_location` purely for the trace.
    """
    return "location" in set(_questions(state))


def _auto_job_title_instruction(state: dict) -> str:
    """The directive for the deterministic post-location `job_title` dispatch. Composed in code (the LLM
    is not consulted on that hop), but still focused by what `location` actually found."""
    loc = _dump(state.get("location"))
    absent = [p for p, found in (loc.get("found_in_pob") or {}).items() if not found]
    parts = [
        "The auditor raised a location question, so role-based chargeability is the fallback basis and "
        "must be established before this case can be closed. Resolve each disputed person's "
        "position from the staff listing + Jobs_Catalogue and judge whether the role is technical in "
        "nature (technical_job_title) — a technical role is chargeable by FUNCTION, independently of "
        "where the person physically was.",
    ]
    if loc.get("nl_summary"):
        parts.append(f"What location established: {_trunc(str(loc['nl_summary']), 1_200)}")
    if absent:
        parts.append("Absent from the POB report: " + ", ".join(absent[:10]) + ".")
    return " ".join(parts)


def _invoke_supervision_before_matching(state: dict, run_counts: dict) -> bool:
    """True when a composite hierarchy review must finish before role matching.

    For any question set that includes hierarchy and reaches the role/CFR path, the required sequence is
    `job_title → supervision → matching`. This guard preserves that dependency even if an earlier
    specialist was inconclusive and the normal adaptive route therefore yields to the supervisor.
    """
    available = _available_targets(run_counts, _questions(state), state)
    return (
        "hierarchy" in set(_questions(state))
        and int(run_counts.get("job_title", 0) or 0) > 0
        and "supervision" in available
        and "matching" in available
    )


def _no_dispute_category(state: dict) -> bool:
    """True when category_check could not identify which dispute the auditor is raising (empty
    auditor_question). There is then NO routing path, so the orchestrator finalizes to HITL asking the
    human which data sources to check (rather than blindly dispatching every specialist)."""
    return not _questions(state)


def _untechnical_role_defer(state: dict) -> bool:
    """Scenario 2 (the mirror of `_technical_role_override`): a job_title question whose duties are NOT
    technical. When `matching` has RUN but classified no CFR category, AND `job_title` has RUN and found
    the role is NOT technical, the AI cannot determine whether the duty is defensible as technical — so
    the case defers to HITL (needs_evidence), it does not fabricate a chargeable rebuttal. Checked AFTER
    `_technical_role_override`, so the two are mutually exclusive (technical wins → terminal)."""
    if state.get("chargeability_scopes"):
        return False
    run_counts = state.get("agent_run_counts") or {}
    matching_ran = int(run_counts.get("matching", 0) or 0) > 0
    job_title_ran = int(run_counts.get("job_title", 0) or 0) > 0
    no_category = not str(sub_get(state, "matching", "identified_cfr_category", "") or "").strip()
    technical = bool(sub_get(state, "job_title", "technical_job_title", False))
    hierarchy_complete = (
        "hierarchy" not in set(_questions(state))
        or int(run_counts.get("supervision", 0) or 0) > 0
    )
    return matching_ran and job_title_ran and no_category and not technical and hierarchy_complete


def _hierarchy_review_complete(state: dict) -> bool:
    """Whether a co-raised hierarchy question has a completed supervision finding.

    A role-based terminal shortcut can resolve the role/location side of an exception, but it must not
    erase an independent hierarchy question. Requiring `objective_met` rather than merely a dispatch
    means an inconclusive supervision pass remains available to the normal final-mode logic.
    """
    if "hierarchy" not in set(_questions(state)):
        return True
    return str((state.get("agent_status") or {}).get("supervision", {}).get("status") or "") == "objective_met"


def _technical_role_override(state: dict) -> bool:
    """Request H (ref: 2025 Tobago exc 0001): a role `job_title` established is TECHNICAL IN NATURE is
    chargeable by FUNCTION, not location — even when `matching` cannot classify a CFR category and even
    when the person is absent from POB. So when `matching` has RUN but produced no category AND
    `job_title.technical_job_title` is True, the case is terminal (a chargeable rebuttal), not a human
    deferral. POB / physical presence is not dispositive, so we do not chase `location` either."""
    if state.get("chargeability_scopes"):
        return False
    run_counts = state.get("agent_run_counts") or {}
    matching_ran = int(run_counts.get("matching", 0) or 0) > 0
    no_category = not str(sub_get(state, "matching", "identified_cfr_category", "") or "").strip()
    technical = bool(sub_get(state, "job_title", "technical_job_title", False))
    return matching_ran and no_category and technical and _hierarchy_review_complete(state)


def _infer_final_mode(state: dict) -> str:
    """The mode to finalize with when the orchestrator is FORCED to finalize (every specialist has
    already run, or the LLM tried to revisit one). Terminal iff every specialist that ran — jade
    included — reported `objective_met`; otherwise defer to a human (needs_evidence). This keeps a
    fully-resolved case terminal without letting a stuck one fabricate a conclusion.

    Request H override: a technical role with an unclassified CFR category is still terminal (chargeable
    by function) — see `_technical_role_override`."""
    if _technical_role_override(state):
        return "terminal"
    statuses = [(v or {}).get("status") for v in (state.get("agent_status") or {}).values()]
    return "terminal" if statuses and all(s == "objective_met" for s in statuses) else "needs_evidence"


def _salient_nl_summaries(state: dict) -> dict:
    """A compact, row-free view of each specialist's findings (only stages that have run)."""
    out: dict[str, Any] = {}
    intake = _dump(state.get("intake"))
    if intake:
        out["intake"] = {
            "assets_in_question": intake.get("assets_in_question", []),
            "collected_evidence": intake.get("collected_evidence", []),
        }
    jade = _dump(state.get("jade_analyzer"))
    if jade:
        out["jade_analyzer"] = {
            "nl_summary": jade.get("nl_summary", ""),
            "amount_basis": jade.get("amount_basis", ""),
            "amount_in_range": jade.get("amount_in_range"),
            "jade_amount_discrepancy": jade.get("jade_amount_discrepancy"),
            "people_found_in_ledger": jade.get("people_found_in_ledger", []),
            "people_named_count": jade.get(
                "people_named_count", len(jade.get("people_found_in_ledger", []) or [])
            ),
            # The major-construction screen jade ran per person while it scanned the ledger: a code-table
            # lookup against the person's coded JV-object-type value, not a text read. A matched code
            # means the labor sits on a major construction project, which is directly chargeable under
            # the settlement's Agreed OTL Definition and Billing Standards — a basis that turns on
            # neither the CFR category nor where the person physically was, though it does not override
            # an above-first-level-supervision exclusion. The supervisor weighs it; nothing in code
            # routes on it.
            "major_construction_charge": jade.get("major_construction_charge"),
        }
        if jade.get("major_construction_charge"):
            # The reading behind the flag, so the supervisor can see WHICH people it covers and what the
            # ledger actually said — a screen that fired for one person of five is not a whole-case basis.
            out["jade_analyzer"]["major_construction_people"] = jade.get("major_construction_people", [])
            out["jade_analyzer"]["jv_object_type_by_person"] = jade.get("jv_object_type_by_person", {})
        if jade.get("major_construction_basis"):
            # Carried either way: when the screen did NOT fire this says whether the column was read and
            # held something else, or was absent from the cited ledger altogether.
            out["jade_analyzer"]["major_construction_basis"] = jade.get("major_construction_basis", "")
        examined, dropped, total = people_coverage(state)
        if dropped:
            out["jade_analyzer"]["population_coverage"] = {
                "examined": examined,
                "not_examined": dropped,
                "identified_count": total,
            }
    loc = _dump(state.get("location"))
    if loc:
        out["location"] = {
            "nl_summary": loc.get("nl_summary", ""),
            "found_in_pob": loc.get("found_in_pob", {}),
            "found_in_heli_logs": loc.get("found_in_heli_logs", {}),
            "pob_people_location": loc.get("pob_people_location", {}),
        }
    jt = _dump(state.get("job_title"))
    if jt:
        out["job_title"] = {
            "nl_summary": jt.get("nl_summary", ""),
            "identified_names_staff_listing": jt.get("identified_names_staff_listing", {}),
            "identified_job_data": {k: _trunc(str(v), 500)
                                    for k, v in (jt.get("identified_job_data") or {}).items()},
            "technical_job_title": jt.get("technical_job_title"),
            "technical_rationale": _trunc(str(jt.get("technical_rationale", "")), 500),
            "role_status_by_person": jt.get("role_status_by_person", {}),
            "ipt_status_by_person": jt.get("ipt_status_by_person", {}),
            "exempt_role_status_by_person": jt.get("exempt_role_status_by_person", {}),
            "job_profile_status_by_person": jt.get("job_profile_status_by_person", {}),
        }
    mt = _dump(state.get("matching"))
    if mt:
        out["matching"] = {"nl_summary": mt.get("nl_summary", ""),
                           "identified_cfr_category": mt.get("identified_cfr_category", ""),
                           "matched_job_profiles": mt.get("matched_job_profiles", {}),
                           "subpart_labels": mt.get("subpart_labels", {})}
    sup = _dump(state.get("supervision"))
    if sup:
        out["supervision"] = {"nl_summary": sup.get("nl_summary", ""),
                              "above_first_level": sup.get("above_first_level", {}),
                              "supervision_status_by_person": sup.get("supervision_status_by_person", {}),
                              "supervision_record_status_by_person": sup.get("supervision_record_status_by_person", {}),
                              "reports_summary": sup.get("reports_summary", {}),
                              "credit_amount": sup.get("credit_amount"),
                              "chargeable_amount": sup.get("chargeable_amount"),
                              "needs_human_review": sup.get("needs_human_review")}
    reused = state.get("rebuttal_reused_nl_summaries") or {}
    if reused:
        out["reused_prior_cycle_nl_summaries"] = reused
    return out


def _digest_for_llm(state: dict, hops: int, run_counts: dict) -> str:
    """The compact decision surface handed to the supervisor.

    Carries the extractor's interpretive `synthesis` narrative (how the auditor built the exception) —
    the richest single orienting field — but never the full synthesis payload or raw workbook rows.
    """
    case = state.get("case") or {}
    ve = _dump(state.get("verify_evidence"))
    questions = _questions(state)
    digest = {
        "hop": hops,
        "case": {k: case.get(k) for k in ("case_id", "exc_id", "year", "asset", "title")
                 if case.get(k)},
        "auditor_question": questions,
        # True iff a location question is present — then job_title is dispatched after location even in
        # a composite case (the runtime enforces it); see _invoke_job_title_after_location.
        "invoke_job_title_after_location": _invoke_job_title_after_location(state),
        "active_auditor_request": _trunc(active_auditor_text(state), 2500),
        "classified_exception": json.loads(compact_classified_exception(state)),
        "extractor_synthesis": synthesis_narrative(state, 3500),
        "evidence": {
            "evidence_gate_decision": ve.get("evidence_gate_decision", {}),
            "pob_data_found": ve.get("pob_data_found"),
            "heli_data_found": ve.get("heli_data_found"),
            "hierarchy_data_found": ve.get("hierarchy_data_found"),
            "cited_workbooks": ve.get("cited_workbooks", []),
            "cited_sheets": ve.get("cited_sheets", {}),
        },
        "open_questions": state.get("open_questions") or [],
        "agent_status": state.get("agent_status") or {},
        "agent_run_counts": run_counts,
        # single-visit rule: you may ONLY dispatch a specialist listed here; each runs at most once.
        # This is already filtered to the specialists RELEVANT to the dispute category(ies) above —
        # jade_analyzer already ran in the spine, and e.g. `location` is absent for a job_title-only
        # question (POB is not dispositive for a technical-role question).
        "available_specialists": _available_targets(run_counts, questions, state),
        "rebuttal_cycle_delta": {
            "current_posture": state.get("rebuttal_current_posture") or "",
            "revisit_allowed": state.get("rebuttal_specialist_revisit") or {},
            "current_grounds": state.get("rebuttal_specialist_current_grounds") or {},
            "reused_nl_summaries": state.get("rebuttal_reused_nl_summaries") or {},
        },
        "nl_summaries_so_far": _salient_nl_summaries(state),
    }
    s = json.dumps(digest, ensure_ascii=False, default=str)
    return s if len(s) <= _DIGEST_LIMIT else s[: _DIGEST_LIMIT - 1] + "…"


def _default_context_for_target(state: dict, target: str) -> str:
    """Bounded dependency context automatically carried beside the history node's cherry-picked slice."""
    nl_summaries = _salient_nl_summaries(state)
    dependency_keys = {
        "location": ("intake", "jade_analyzer"),
        "job_title": ("intake", "jade_analyzer", "location"),
        "supervision": ("intake", "jade_analyzer", "job_title"),
        "matching": ("jade_analyzer", "job_title", "supervision"),
    }.get(target, ())
    payload = {
        "current_posture": state.get("rebuttal_current_posture") or "",
        "current_questioning_ground": (
            (state.get("rebuttal_specialist_current_grounds") or {}).get(target) or ""
        ),
        "relevant_current_nl_summaries": {
            key: nl_summaries[key] for key in dependency_keys if key in nl_summaries
        },
    }
    if not any(payload.values()):
        return ""
    return json.dumps(payload, ensure_ascii=False, default=str)


def _build_update(state: dict, *, hops: int, run_counts: dict, prev_log: list,
                  chose: str, rationale: str, nl_summary: str, final_mode: str = "",
                  instruction: str = "", context_to_pass: str = "", hitl_reason: str = "",
                  hitl_stage: str = "") -> dict:
    """Assemble the node's state update + trace for one decision (dispatch or finalize)."""
    finalizing = chose == "drafter"
    status_seen = {k: (v or {}).get("status") for k, v in (state.get("agent_status") or {}).items()}
    log_entry = {"hop": hops, "chose": chose, "rationale": _trunc(rationale, 400),
                 "status_seen": status_seen}
    if context_to_pass:
        log_entry["context_to_pass"] = _trunc(context_to_pass, 600)
    if finalizing:
        log_entry["final_mode"] = final_mode

    orch = OrchestratorState(
        nl_summary=nl_summary,
        decision_log=[*prev_log, log_entry],
        final_mode=final_mode if finalizing else "",
        hops=hops,
        invoke_job_title_after_location=_invoke_job_title_after_location(state),
    )

    update: dict = {
        "orchestrator": orch,
        "next_agent": chose,
        "orchestrator_hops": hops,
        "agent_run_counts": run_counts,
    }
    if finalizing:
        update["final_mode"] = final_mode
        # `final_mode` is the ONLY thing that raises the HITL gate on this path, and it is written
        # EXPLICITLY in both directions so the drafter's mode selector can never be flipped by a stale
        # channel value: needs_evidence → the full hitl_* trail; terminal → hitl_review hard-set False.
        if final_mode == "needs_evidence":
            update["hitl_review"] = True
            update["hitl_reason"] = hitl_reason or "the orchestrator could not resolve the exception."
            update["hitl_stage"] = hitl_stage or "orchestrator"
            update["hitl_evidence_trail"] = {"decision_log": orch.decision_log,
                                             "nl_summaries": _salient_nl_summaries(state)}
        else:
            update["hitl_review"] = False
    else:
        # merge_dict channel — accumulates one directive per dispatched specialist
        context_parts = [
            value for value in (_default_context_for_target(state, chose), context_to_pass.strip())
            if value
        ]
        enriched = instruction.strip()
        if context_parts:
            enriched += (
                "\n\nORCHESTRATOR-SELECTED ADDITIONAL CONTEXT "
                "(use beside the separately supplied cherry-picked rebuttal history):\n"
                + "\n".join(context_parts)
            )
        update["agent_instructions"] = {chose: enriched}

    action = (f"finalize → drafter ({final_mode})" if finalizing else f"dispatch → {chose}")
    update.update(trace_event(
        state, "orchestrator", action,
        inputs_digest=f"hop={hops}, run_counts={run_counts}, status={status_seen}",
        key_outputs={"next_agent": chose, "final_mode": final_mode if finalizing else "",
                     "hitl_review": update.get("hitl_review") if finalizing else None,
                     "instruction": _trunc(instruction, 200) if not finalizing else ""},
        nl_summary=nl_summary or action,
    ))
    return update


def orchestrator_node(state: dict) -> dict:
    hops = int(state.get("orchestrator_hops", 0) or 0) + 1
    run_counts = dict(state.get("agent_run_counts") or {})
    prev_log = list(sub_get(state, "orchestrator", "decision_log", []) or [])

    # ---- loop guard: trip BEFORE trusting the LLM so the graph can never infinite-loop ----
    over_hops = hops >= MAX_HOPS
    maxed_agent = max(run_counts.values(), default=0) >= MAX_AGENT_RUNS
    if over_hops or maxed_agent:
        reason = (("orchestrator hop cap reached" if over_hops else "a specialist re-run cap reached")
                  + " (loop guard) — finalizing to human review without a defensible conclusion.")
        return _build_update(state, hops=hops, run_counts=run_counts, prev_log=prev_log,
                             chose="drafter", final_mode="needs_evidence", rationale=reason,
                             hitl_reason=reason, hitl_stage="orchestrator",
                             nl_summary="Loop guard tripped before a defensible conclusion; the case requires human review.")

    # ---- no-dispute-category guard: category_check could not identify which dispute the auditor is
    #      raising, so there is NO routing path. Defer to HITL asking which data sources to check rather
    #      than blindly dispatching every specialist. ----
    if _no_dispute_category(state):
        reason = ("category_check could not identify which dispute the auditor is raising "
                  "(job_title / location / hierarchy), so there is no routing path — a human must "
                  "confirm which data sources to check.")
        return _build_update(state, hops=hops, run_counts=run_counts, prev_log=prev_log,
                             chose="drafter", final_mode="needs_evidence", rationale=reason,
                             hitl_reason=reason, hitl_stage="orchestrator",
                             nl_summary="No dispute category could be established, so no defensible specialist route is available.")

    # ---- location → job_title pin: job_title ALWAYS follows location, including composite cases. A
    #      location finding alone cannot close the case — a technical role is chargeable by FUNCTION
    #      regardless of where the person was — so dispatch job_title deterministically here (the LLM is
    #      NOT consulted on this hop). The `available` check makes this a no-op once job_title has run, so
    #      it fires at most once. ----
    if (_invoke_job_title_after_location(state)
            and int(run_counts.get("location", 0) or 0) > 0
            and "job_title" in _available_targets(run_counts, _questions(state), state)):
        reason = ("location question and location has run — the role question is always live (a "
                  "technical role is chargeable by function, not location), so job_title follows "
                  "automatically.")
        run_counts["job_title"] = int(run_counts.get("job_title", 0)) + 1
        return _build_update(state, hops=hops, run_counts=run_counts, prev_log=prev_log,
                             chose="job_title", instruction=_auto_job_title_instruction(state),
                             rationale=reason,
                             nl_summary="Location remains disputed; job-title analysis is required to assess the independent functional basis.")

    # ---- hierarchy → matching pin: a composite hierarchy question must complete supervision before
    #      role/CFR matching. This preserves job_title → supervision → matching even when an earlier
    #      specialist was inconclusive and the normal adaptive route would defer to the LLM. ----
    if _invoke_supervision_before_matching(state, run_counts):
        reason = ("hierarchy is part of this role-based composite and job_title has run — supervision "
                  "must resolve before matching.")
        run_counts["supervision"] = int(run_counts.get("supervision", 0) or 0) + 1
        return _build_update(
            state, hops=hops, run_counts=run_counts, prev_log=prev_log,
            chose="supervision", instruction=_adaptive_instruction(state, "supervision"),
            rationale=reason, nl_summary="A hierarchy component remains unresolved, so supervision must run before matching.",
        )

    # ---- adaptive cheap path: the category registry already determines the next dependency for every
    #      recognized category combination. Keep the high-tier supervisor only for an inconclusive
    #      specialist; do not spend it merely restating a fixed route. ----
    questions = _questions(state)
    available = _available_targets(run_counts, questions, state)
    if not available:
        mode = _infer_final_mode(state)
        reason = (
            "Every relevant specialist has completed; the typed specialist statuses determine the "
            f"final mode ({mode}) without another routing-model call."
        )
        return _build_update(
            state, hops=hops, run_counts=run_counts, prev_log=prev_log,
            chose="drafter", final_mode=mode, rationale=reason,
            hitl_reason="" if mode == "terminal" else reason,
            hitl_stage="orchestrator", nl_summary=f"Relevant specialists are complete; finalizing with {mode}.",
        )

    prescribed = _adaptive_prescribed_target(state, run_counts)
    if prescribed:
        run_counts[prescribed] = int(run_counts.get(prescribed, 0) or 0) + 1
        reason = (
            f"The recognized question set prescribes {prescribed} as the next unresolved dependency; "
            "no discretionary routing choice is present."
        )
        return _build_update(
            state, hops=hops, run_counts=run_counts, prev_log=prev_log,
            chose=prescribed, instruction=_adaptive_instruction(state, prescribed),
            rationale=reason, nl_summary=f"The recognized question set requires {prescribed} as the next unresolved dependency.",
        )

    # ---- otherwise, ask the supervisor for the next hop ----
    questions = _questions(state)
    available = _available_targets(run_counts, questions, state)   # relevant specialists not yet run
    user = (
        "Decide the next step for this OTL exception. You are on hop "
        f"{hops} of at most {MAX_HOPS}.\n\n"
        "DECISION SURFACE (everything gathered so far — compact, no raw rows):\n"
        f"{_digest_for_llm(state, hops, run_counts)}\n\n"
        f"{rebuttal_context_block(state, 'orchestrator')}\n\n"
        f"You may dispatch ONLY a specialist in available_specialists ({available or 'none left'}); "
        "each runs AT MOST ONCE and jade_analyzer already ran in the spine, so it is never an option. "
        "When no useful specialist remains, finalize. To finalize, set next_agent='drafter' (or "
        "finalize=true) and final_mode ∈ {terminal, needs_evidence}; give a concrete hitl_reason when "
        "needs_evidence. To dispatch, give a short focused instruction. Provide a concise but substantive "
        "findings analysis covering the decision, decisive facts, and unresolved work. Populate context_to_pass with any additional current-case findings or "
        "dependencies the selected specialist needs beyond its supplied cherry-picked history; leave it "
        "empty when no additional context is useful."
    )
    dec = llm.structured("orchestrator", ORCHESTRATOR_SYSTEM, user, _OrchestratorDecision)

    nxt = (dec.next_agent or "").strip()
    finalize = bool(dec.finalize) or nxt == "drafter"

    if finalize:
        final_mode = "terminal" if dec.final_mode.strip().lower() == "terminal" else "needs_evidence"
        return _build_update(state, hops=hops, run_counts=run_counts, prev_log=prev_log,
                             chose="drafter", final_mode=final_mode, rationale=dec.rationale,
                             hitl_reason=dec.hitl_reason, hitl_stage=dec.hitl_stage or "orchestrator",
                             nl_summary=dec.nl_summary)

    if nxt not in _DISPATCH_TARGETS:
        # Malformed / no valid next step (garbage, empty, or jade — which is spine-only). Finalize
        # conservatively to human review rather than route nowhere.
        reason = f"orchestrator returned no valid next step ({dec.next_agent!r})."
        return _build_update(state, hops=hops, run_counts=run_counts, prev_log=prev_log,
                             chose="drafter", final_mode="needs_evidence",
                             rationale=dec.rationale or reason, hitl_reason=reason,
                             hitl_stage="orchestrator",
                             nl_summary=dec.nl_summary or "No valid next step was returned, so the case requires human review.")

    if nxt not in available:
        # SINGLE-VISIT RULE: the chosen specialist already ran. Never revisit — finalize instead,
        # terminal iff everything that ran resolved, else needs_evidence (see _infer_final_mode).
        mode = _infer_final_mode(state)
        reason = (f"orchestrator tried to revisit '{nxt}' (already run); each specialist runs at most "
                  f"once — finalizing ({mode}).")
        return _build_update(state, hops=hops, run_counts=run_counts, prev_log=prev_log,
                             chose="drafter", final_mode=mode, rationale=dec.rationale or reason,
                             hitl_reason="" if mode == "terminal" else reason,
                             hitl_stage="orchestrator",
                             nl_summary=dec.nl_summary or f"A prohibited specialist revisit was blocked; finalizing with {mode}.")

    # dispatch this first-and-only run of the specialist
    run_counts[nxt] = int(run_counts.get(nxt, 0)) + 1
    return _build_update(state, hops=hops, run_counts=run_counts, prev_log=prev_log,
                         chose=nxt, instruction=dec.instruction,
                         context_to_pass=dec.context_to_pass, rationale=dec.rationale,
                         nl_summary=dec.nl_summary)


__all__ = ["orchestrator_node", "MAX_HOPS", "MAX_AGENT_RUNS"]
