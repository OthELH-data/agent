"""category_check_node — which questions the auditor is raising about the charged labor (light tier).

Chooses zero or more of {job_title, location, hierarchy, attestation_certificate,
third_party_invoice_requested, additional_support_requested, other} and writes them to the master
`auditor_question` channel, which the orchestrator reads to decide which specialists to dispatch.
Only a question the auditor actually raises is included. Categories are independent: any combination
may be returned when the auditor raises each corresponding question.

PASSED TO THE LLM (structured output):
  * system : CATEGORY_CHECK_SYSTEM (prompts.py)
  * user   : active-cycle auditor description/latest reply + compacted synthesis + bounded cycle-delta
             summary/current specialist grounds; full prior rebuttal history is deliberately excluded
  * schema : _CategoryExtraction — its Field descriptions reach the model via the JSON schema.
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field

from otl import llm_adapter as llm
from otl.prompts import CATEGORY_CHECK_SYSTEM
from otl.state import CategoryCheckState
from otl.trace import trace_event
from otl.agents._common import active_auditor_context_block, compact_synthesis

_ALLOWED = {
    "job_title", "location", "hierarchy", "attestation_certificate",
    "third_party_invoice_requested", "additional_support_requested", "other",
}
_ANALYTIC = {"job_title", "location", "hierarchy"}
_OTHER_HITL_REASON = (
    "The auditor's question could not be mapped to a supported analysis category; "
    "human clarification is required to identify the appropriate review path."
)


def _delta_questions(state: dict) -> list[str]:
    """Map current-cycle specialist grounds from delta analysis back to routing categories."""
    if not state.get("rebuttal_history"):
        return []
    revisit = state.get("rebuttal_specialist_revisit") or {}
    grounds = state.get("rebuttal_specialist_current_grounds") or {}
    questions: list[str] = []
    if revisit.get("location") and str(grounds.get("location") or "").strip():
        questions.append("location")
    if (
        (revisit.get("job_title") and str(grounds.get("job_title") or "").strip())
        or (revisit.get("matching") and str(grounds.get("matching") or "").strip())
    ):
        questions.append("job_title")
    if revisit.get("supervision") and str(grounds.get("supervision") or "").strip():
        questions.append("hierarchy")
    return questions


class _CategoryExtraction(BaseModel):
    """What the light model returns — the dispute categories the auditor actually raised.

    The Field descriptions below become the JSON schema the model sees.
    """
    auditor_question: list[str] = Field(
        default_factory=list,
        description="The dispute(s) the auditor is actually raising, each EXACTLY one of "
                    "'job_title' (is the person's role what was charged?), 'location' (was the person "
                    "physically where the labor was charged?), 'hierarchy' (is the person above "
                    "first-level supervision, i.e. not separately chargeable?), 'attestation_certificate' "
                    "(the auditor is requesting an attestation or affiliate cost certificate), "
                    "'third_party_invoice_requested' (the auditor specifically asks the operator to provide "
                    "an underlying third-party/vendor invoice), 'additional_support_requested' (the auditor "
                    "currently asks for other evidence, documents, records, data, support, calculations, or "
                    "a factual explanation), "
                    "or 'other' (none of the listed categories apply).")
    nl_summary: str = Field(
        default="", description="A concise analysis of the auditor's actual dispute categories, notable "
                                "grounds, and any ambiguity that affects routing. Do not list every detail.")


def category_check_node(state: dict) -> dict:
    synth = state.get("synthesis") or {}
    user = (
        "CLASSIFY THE UNRESOLVED REQUESTS IN THE ACTIVE AUDITOR CONTEXT BELOW. Do not carry forward a "
        "prior-cycle category merely because it appears only in historical findings, but do not mistake "
        "the active original exception baseline for prior-cycle history.\n"
        f"{active_auditor_context_block(state)}\n\n"
        f"Current-cycle's auditor-provided attachment synthesis (compacted, row-free):\n{compact_synthesis(synth)}"
        + '\n\nWhich questions is the auditor raising in this cycle? Choose from exactly "job_title", "location", '
        '"hierarchy", "attestation_certificate", "third_party_invoice_requested", '
        '"additional_support_requested", or "other". Choose "third_party_invoice_requested" for a current '
        'specific request for an underlying third-party/vendor invoice. Do not also choose '
        '"additional_support_requested" for that same request; choose it only when a separate request for '
        'other material or information remains. Choose either request category only for a current unresolved request. '
        'An IR number or fieldwork reference inside the ACTIVE ORIGINAL EXCEPTION BASELINE does not make '
        'the request historical. Apply the historical-request exclusion only to separate prior-cycle context '
        'that explicitly shows the request was satisfied, withdrawn, replaced, or resolved. Return only the '
        'questions actually raised.'
    )
    ext = llm.structured("category_check", CATEGORY_CHECK_SYSTEM, user, _CategoryExtraction)

    # Keep only allowed tokens, de-duped, order-stable. Categories are intentionally independent so
    # the orchestrator can analyze every question the auditor actually combined in one exception.
    model_questions = [q for q in dict.fromkeys(ext.auditor_question) if q in _ALLOWED]
    recognized = [q for q in model_questions if q != "other"]
    current_questions = recognized or ["other"]
    current_analytic = [q for q in current_questions if q in _ANALYTIC]
    delta_analytic = _delta_questions(state)
    effective_questions = list(dict.fromkeys([*current_analytic, *delta_analytic]))

    attestation_requested = "attestation_certificate" in current_questions
    third_party_invoice_requested = "third_party_invoice_requested" in current_questions
    additional_support_requested = "additional_support_requested" in current_questions
    if attestation_requested:
        effective_questions.append("attestation_certificate")
    if third_party_invoice_requested:
        effective_questions.append("third_party_invoice_requested")
    if additional_support_requested:
        effective_questions.append("additional_support_requested")
    if not effective_questions:
        effective_questions = ["other"]
    sub = CategoryCheckState(
        nl_summary=ext.nl_summary,
        auditor_question=current_questions,
        attestation_requested=attestation_requested,
        third_party_invoice_requested=third_party_invoice_requested,
        additional_support_requested=additional_support_requested,
    )

    update: dict = {
        "category_check": sub,
        "auditor_question": effective_questions,
        "attestation_requested": attestation_requested,
        "third_party_invoice_requested": third_party_invoice_requested,
        "additional_support_requested": additional_support_requested,
    }
    if effective_questions == ["other"]:
        update.update({
            "hitl_review": True,
            "hitl_stage": "category_check",
            "hitl_reason": _OTHER_HITL_REASON,
        })
    update.update(trace_event(
        state, "category_check", "determined auditor questions",
        inputs_digest="active-cycle narrative/reply + compacted synthesis + cycle-delta grounds",
        key_outputs={
            "current_auditor_question": current_questions,
            "delta_auditor_question": delta_analytic,
            "auditor_question": effective_questions,
            "attestation_requested": attestation_requested,
            "third_party_invoice_requested": third_party_invoice_requested,
            "additional_support_requested": additional_support_requested,
            "hitl_review": bool(update.get("hitl_review")),
        },
        nl_summary=sub.nl_summary or ("questions: " + (", ".join(effective_questions) or "none")),
    ))
    return update


__all__ = ["category_check_node"]
