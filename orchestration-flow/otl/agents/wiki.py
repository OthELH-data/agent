"""Read-only precedent lookup after agreement retrieval.

Raw pages remain an independent report sidecar and never enter drafter prose. The node emits one
bounded ``precedence_status`` fact, performs final declarative hook evaluation, and then routes to the
drafter. An unavailable corpus therefore becomes ``unavailable_or_unclear`` rather than invented
precedent support.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from wiki import kb_fs

from otl import llm_adapter as llm
from otl.chargeability_hooks import evaluate_state
from otl.state import WikiState
from otl.trace import trace_event
from otl.agents._common import active_auditor_text, rebuttal_context_block


_SECTIONS = ("types", "clauses", "precedents", "entities")
_MAX_INDEX_CHARS = 16_000
_MAX_QUERY_CHARS = 3_000
_MAX_CONTEXT_CHARS = 20_000


class _WikiSelection(BaseModel):
    types: list[str] = Field(default_factory=list)
    clauses: list[str] = Field(default_factory=list)
    precedents: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    precedent_strength: str = "none"
    precedence_status: str = Field(
        default="unavailable_or_unclear",
        description="Exactly supports_chargeability | supports_non_chargeability | unavailable_or_unclear")
    match_reason: str = ""


_SYSTEM = """You select relevant pages from a historical audit precedent wiki.

This is an independent reviewer sidecar. It does not draft, revise, endorse, or reject the OTL
response. Select only slugs explicitly present in the supplied index. Prefer a closely matching type
page, then directly relevant clause and precedent pages. Do not manufacture a match merely because
the wiki is non-empty. Use precedent_strength=strong only for a close dispute-pattern match,
weak for useful but incomplete/analogical context, and none when nothing is genuinely relevant."""
_SYSTEM += "\nSet precedence_status to supports_chargeability or supports_non_chargeability only when " \
           "the selected precedent clearly supports that direction; otherwise use unavailable_or_unclear."


def _dump(value: Any) -> dict:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return dict(value) if isinstance(value, dict) else {}


def _query_text(state: dict) -> str:
    case = state.get("case") or {}
    intake = _dump(state.get("intake"))
    category = _dump(state.get("category_check"))
    auditor_text = (
        active_auditor_text(state)
        or case.get("desc")
        or case.get("description")
        or ""
    )
    parts = [
        f"exception_id: {case.get('exc_id') or case.get('case_id') or ''}",
        f"asset: {case.get('asset') or ''}",
        f"year: {case.get('year') or ''}",
        f"title: {case.get('title') or ''}",
        f"current_posture: {state.get('rebuttal_current_posture') or intake.get('nl_summary') or ''}",
        f"error_code: {intake.get('error_code') or ''}",
        f"code_family: {intake.get('code_family') or ''}",
        f"auditor_questions: {category.get('auditor_question') or state.get('auditor_question') or []}",
        f"active_auditor_text: {auditor_text}",
    ]
    return "\n".join(parts)[:_MAX_QUERY_CHARS]


def _empty_result(state: dict, *, status: str, reason: str, provenance: dict) -> dict:
    query = _query_text(state)
    sub = WikiState(
        nl_summary=reason,
        status=status,
        query=query,
        precedent_strength="none",
        match_reason=reason,
        selected_pages={section: [] for section in _SECTIONS},
        kb_provenance=provenance,
        precedence_status="unavailable_or_unclear",
    )
    update = {"wiki": sub}
    update.update(_final_hook_update({**state, **update}))
    update.update(trace_event(
        state,
        "wiki",
        "completed independent precedent lookup",
        inputs_digest="read-only index-first lookup; output excluded from drafter inputs",
        key_outputs={
            "status": sub.status,
            "precedent_strength": sub.precedent_strength,
            "selected_pages": sub.selected_pages,
            "affects_drafter": False,
        },
        nl_summary=sub.nl_summary,
    ))
    return update


def _final_hook_update(state: dict) -> dict:
    evaluated = evaluate_state(state, final=True)
    resolution = evaluated["chargeability_resolution"]
    existing_needs = state.get("final_mode") == "needs_evidence" or bool(state.get("hitl_review"))
    if existing_needs or resolution.disposition == "needs_evidence":
        evaluated.update({
            "final_mode": "needs_evidence", "hitl_review": True,
            "hitl_stage": "chargeability_review",
            "hitl_reason": (
                "Some questioned labor remains unresolved or has competing chargeability outcomes; "
                "the supported and unresolved portions are summarized for business review."
            ),
        })
    else:
        evaluated.update({"final_mode": "terminal", "hitl_review": False})
    return evaluated


def wiki_node(state: dict) -> dict:
    """Select and return precedent pages without affecting any downstream decision or prose."""
    provenance: dict = {}
    try:
        provenance = kb_fs.provenance()
        if not provenance.get("available"):
            return _empty_result(
                state, status="unavailable", reason="precedent wiki is not available", provenance=provenance,
            )

        page_count = sum(
            int((provenance.get("section_counts") or {}).get(section) or 0)
            for section in _SECTIONS
        )
        index = kb_fs.read_index()
        if page_count == 0 or not index.strip():
            return _empty_result(
                state, status="empty", reason="precedent wiki has not been populated yet", provenance=provenance,
            )

        query = _query_text(state)
        selection = llm.structured(
            "wiki",
            _SYSTEM,
            f"WIKI INDEX:\n{index[:_MAX_INDEX_CHARS]}\n\nCURRENT OTL CASE:\n{query}"
            f"{rebuttal_context_block(state)}",
            _WikiSelection,
        )

        selected: dict[str, list[str]] = {}
        chunks: list[str] = []
        for section in _SECTIONS:
            raw_slugs = getattr(selection, section, []) or []
            valid = [
                slug for slug in dict.fromkeys(str(item).strip() for item in raw_slugs)
                if slug and kb_fs.exists(section, slug)
            ]
            selected[section] = valid
            for slug in valid:
                chunks.append(f"### [{section}/{slug}]\n{kb_fs.read_page(section, slug)}")

        context = "\n\n".join(chunks)[:_MAX_CONTEXT_CHARS]
        strength = str(selection.precedent_strength or "none").strip().lower()
        if strength not in {"strong", "weak", "none"}:
            strength = "none"
        if not context:
            strength = "none"
        elif strength == "none":
            strength = "weak"

        status = "matched" if context else "no_match"
        reason = str(selection.match_reason or "").strip()
        if not reason:
            reason = "relevant precedent pages selected" if context else "no relevant precedent pages found"
        sub = WikiState(
            nl_summary=f"{status} — {reason}",
            status=status,
            query=query,
            precedent_strength=strength,
            match_reason=reason,
            selected_pages=selected,
            precedent_context=context,
            kb_provenance=provenance,
            precedence_status=(
                selection.precedence_status
                if selection.precedence_status in {
                    "supports_chargeability", "supports_non_chargeability", "unavailable_or_unclear"
                }
                else "unavailable_or_unclear"
            ),
        )
        update = {"wiki": sub}
        update.update(_final_hook_update({**state, **update}))
        update.update(trace_event(
            state,
            "wiki",
            "completed independent precedent lookup",
            inputs_digest="index-first selection; selected page bodies returned only in wiki sidecar",
            key_outputs={
                "status": status,
                "precedent_strength": strength,
                "selected_pages": selected,
                "context_chars": len(context),
                "affects_drafter": False,
                "precedence_status": sub.precedence_status,
                "chargeability_disposition": update["chargeability_resolution"].disposition,
            },
            nl_summary=sub.nl_summary,
        ))
        return update
    except Exception as exc:
        return _empty_result(
            state,
            status="error",
            reason=f"precedent wiki lookup failed: {type(exc).__name__}: {str(exc)[:160]}",
            provenance=provenance,
        )


__all__ = ["wiki_node"]
