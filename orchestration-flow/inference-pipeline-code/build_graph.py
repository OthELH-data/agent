"""The two LangGraph pipelines (markdown Precedent-Wiki edition).

(A) BUILD / LEARNING — process an exception from scratch: summarize/classify,
    run the Excel Extractor, retrieve clauses, assemble & reconstruct the defense,
    run the directed Excel EVIDENCE agent to enrich the case, verify, interrupt for
    SME review when reasoning_flag is set, draft, then INGEST the resolved case into
    the markdown Precedent Wiki (compounding type/clause/precedent/entity pages).

(B) INFERENCE — for a new incoming exception: summarize, inventory/synthesize the
    attached workbooks, QUERY the Precedent Wiki (index-first,
    reuse/assist/novel), then draft reusing precedent when strong, else run the
    full build path. The directed Excel evidence agent still runs after precedent
    is identified so it knows what specifics to hunt for.

State flows through one typed AuditState (State1..State5 accrue).
"""
from __future__ import annotations
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import AuditState
from agents import core
from extractors import excel_extractor as ex
from extractors import excel_evidence as ee
from graph.evidence_inventory import (
    finding_evidence_items,
    inventory_from_case,
    merge_collected_evidence,
    state2_evidence_item,
)
from graph.evidence_planner import evidence_gap_finalize_node, evidence_planner_node
from graph.pob_agent import pob_agent_node
from wiki import ingest_writer as wiki_writer
from wiki import query as wiki_query


# ---------------------------------------------------------------- shared nodes
def intake(state: AuditState) -> dict:
    case = state["case"]
    return {"dollars": float(case.get("dollars") or 0.0),
            "collected_evidence": inventory_from_case(case),
            "log": [f"[intake] {case.get('exception_id')} {case.get('asset')} {case.get('year')}"]}


def _is_openpyxl_workbook(attachment: dict) -> bool:
    kind = str(attachment.get("kind") or "").lower().lstrip(".")
    path = str(attachment.get("path") or "").lower()
    return kind in {"xlsx", "xlsm"} or path.endswith((".xlsx", ".xlsm"))


def _first_xlsx(case: dict) -> str | None:
    for a in case.get("attachments", []) or []:
        if not isinstance(a, dict):
            continue
        if _is_openpyxl_workbook(a):
            return a.get("path")
    return None


def _preextracted_state(state: AuditState) -> dict:
    """Load staged auditor-workbook synthesis as an additive state slice.

    This runs before reconstruction in inference mode and again at the directed
    evidence stage.  The merge is idempotent, so both paths see the same evidence
    without duplicating artifacts.
    """
    try:
        from extractors import preextracted_evidence as preextracted

        mapped = preextracted.load_case_evidence(state.get("case", {}))
        if mapped:
            packet = dict(state.get("evidence_packet", {}) or {})
            packet["excel_evidence"] = mapped["excel_evidence"]
            raw_gaps = mapped["excel_evidence"].get("gap") or []
            gaps = raw_gaps if isinstance(raw_gaps, list) else [raw_gaps]
            gaps = [str(gap).strip() for gap in gaps if str(gap).strip()]
            questions = mapped.get("evidence_questions", []) or []
            out = {
                "excel_mode": "preextracted",
                "evidence_packet": packet,
                "excel_evidence_gaps": gaps,
                "evidence_questions": questions,
                "collected_evidence": merge_collected_evidence(
                    state.get("collected_evidence"), mapped.get("collected_evidence")
                ),
                "log": [
                    f"[excel-evidence] PRE-EXTRACTED "
                    f"source={mapped['excel_evidence'].get('source_case_id')} "
                    f"found={mapped['excel_evidence']['found']} "
                    f"rows={len(mapped['excel_evidence']['evidence_rows'])} "
                    f"questions={len(questions)} gaps={len(gaps)}"
                ],
            }
            if mapped.get("calculation_summary"):
                out["calculation_summary"] = mapped["calculation_summary"]
            if mapped.get("potential_discrepancies"):
                out["potential_discrepancies"] = mapped["potential_discrepancies"]
            return out

        packet = dict(state.get("evidence_packet", {}) or {})
        packet["excel_evidence"] = {
            "found": False,
            "proves": "",
            "evidence_rows": [],
            "supporting_fields": [],
            "gap": ["no pre-extracted JSON for this case"],
            "source": "preextracted",
            "source_case_id": None,
        }
        return {
            "excel_mode": "none",
            "evidence_packet": packet,
            "excel_evidence_gaps": ["no pre-extracted JSON for this case"],
            "log": [
                "[excel-evidence] PRE-EXTRACTED source-of-truth mode: no JSON found -> "
                "empty evidence (NO fallback to live extractor)"
            ],
        }
    except Exception as exc:
        gap = f"pre-extracted evidence adapter failed: {type(exc).__name__}: {str(exc)[:120]}"
        packet = dict(state.get("evidence_packet", {}) or {})
        packet["excel_evidence"] = {
            "found": False,
            "proves": "",
            "evidence_rows": [],
            "supporting_fields": [],
            "gap": [gap],
            "source": "preextracted",
        }
        return {
            "excel_mode": "none",
            "evidence_packet": packet,
            "excel_evidence_gaps": [gap],
            "log": [
                "[excel-evidence] PRE-EXTRACTED source-of-truth mode: "
                "adapter failure -> explicit gap (NO live fallback)"
            ],
        }


def excel_node(state: AuditState) -> dict:
    """Excel EXTRACTOR: summarize the workbook -> State-2 (the 5-column summary)."""
    import os
    if os.environ.get("USE_PREEXTRACTED_EVIDENCE") == "1":
        return _preextracted_state(state)
    xlsx = _first_xlsx(state.get("case", {}))
    if not xlsx:
        return {"excel_mode": "none", "log": ["[excel] no workbook attached"]}
    try:
        res = ex.extract(xlsx, mode="deep_agent")
        d = res.to_state2()
        attachment = next(
            (item for item in (state.get("case", {}) or {}).get("attachments", []) or []
             if isinstance(item, dict) and item.get("path") == xlsx),
            {"path": xlsx, "name": _xlsx_name(state.get("case", {})), "source": "auditor"},
        )
        d["collected_evidence"] = merge_collected_evidence(
            state.get("collected_evidence"),
            [state2_evidence_item(attachment, d)],
        )
        d["log"] = [f"[excel] mode={res.mode} discrepancies={len(res.potential_discrepancies or [])}"]
        return d
    except Exception as e:
        return {"excel_mode": "none", "log": [f"[excel] error: {str(e)[:80]}"]}


def excel_evidence_node(state: AuditState) -> dict:
    """Excel EVIDENCE agent: runs AFTER precedent/assembly identified what to prove.
    Directed by TWO context sources:
      (1) PRECEDENT (wiki) — what Shell historically argued for this dispute type and
          which facts decided it; and
      (2) the RECONSTRUCTION/assembly — this case's specific defense angle.
    Plus the EXTRACTOR's State-2 workbook summary tells it WHERE in the sheets to look.
    It digs into the workbook rows for the SPECIFIC supporting facts (job descriptions,
    WBS coding, hour splits) and enriches the evidence packet.

    TEMPORARY SWAP (flag-gated, off by default): when USE_PREEXTRACTED_EVIDENCE=1,
    a colleague already ran this excel-extractor stage and saved the results as
    JSON. We consume that instead of re-running the workbook dig — ONLY this
    stage is swapped; every other pipeline node is unchanged. Live extraction is
    used only when the flag is off. Missing/invalid pre-extracted data becomes an
    explicit evidence gap so a source-of-truth run cannot be contaminated."""
    import os
    if os.environ.get("USE_PREEXTRACTED_EVIDENCE") == "1":
        return _preextracted_state(state)
    attachments = (state.get("case", {}) or {}).get("attachments", []) or []
    xlsx_attachments = [a for a in attachments
                        if isinstance(a, dict) and a.get("path")
                        and _is_openpyxl_workbook(a)]
    if not xlsx_attachments:
        return {"log": ["[excel-evidence] no workbook — skipped"]}
    from graph.state_helpers import get_selected
    # (1) precedent context (inference mode threads this in); (2) this case's defense
    precedent_ctx = state.get("precedent_context", "") or ""
    defense_needs = (state.get("reconstruction_rationale")
                     or " ".join(str(s) for s in get_selected(state))
                     or state.get("summary", ""))
    needs_parts = []
    if precedent_ctx:
        needs_parts.append("PRECEDENT (what similar past cases turned on):\n" + precedent_ctx[:800])
    if defense_needs:
        needs_parts.append("THIS CASE'S DEFENSE ANGLE:\n" + str(defense_needs)[:800])
    # (3) EVIDENCE PLAN from the planner: type-skill playbook + authority + already-resolved
    #     JADE findings + the typed needs still to prove. This directs the workbook dig.
    plan = state.get("evidence_plan", {}) or {}
    if plan:
        authority = plan.get("authority", {}) or {}
        plan_bits = []
        if plan.get("type_skill"):
            plan_bits.append(f"TYPE PLAYBOOK: {plan['type_skill']}")
        if plan.get("type_skill_instructions"):
            plan_bits.append("TYPE RUNTIME CONTRACT:\n" + str(plan["type_skill_instructions"])[:4000])
        if authority.get("mfis"):
            plan_bits.append("RELEVANT MFIs: " + ", ".join(map(str, authority['mfis'][:4])))
        open_needs = [n.get("need") for n in plan.get("needs", [])
                      if n.get("source_status") not in ("resolved",) and n.get("need")]
        if open_needs:
            plan_bits.append("STILL TO PROVE: " + " | ".join(str(x) for x in open_needs[:4]))
        if plan.get("jade_findings"):
            plan_bits.append("JADE ALREADY CONFIRMS: " + "; ".join(list(plan["jade_findings"].keys())[:3]))
        if plan_bits:
            needs_parts.append("EVIDENCE PLAN (planner):\n" + "\n".join(plan_bits))
    needs = "\n\n".join(needs_parts) or state.get("summary", "")
    # extractor State-2 summary: tells the agent where to dig in the workbook
    extractor_summary = " | ".join(filter(None, [
        str(state.get("calculation_summary", "") or ""),
        "discrepancies: " + "; ".join(state.get("potential_discrepancies", []) or [])
            if state.get("potential_discrepancies") else "",
    ]))
    try:
        result = ee.investigate_case(xlsx_attachments, evidence_needs=needs[:5200],
                                     extractor_summary=extractor_summary)
        findings = result.get("excel_evidence_findings", []) or []
        gaps = [str(g).strip() for g in result.get("excel_evidence_gaps", []) or []
                if str(g).strip()]
        evidence_rows = [row for finding in findings
                         for row in (finding.get("evidence_rows", []) or [])][:12]
        supporting_fields = list(dict.fromkeys(
            field for finding in findings
            for field in (finding.get("supporting_fields", []) or [])))
        proves = [finding.get("proves") for finding in findings if finding.get("proves")]
        pkt = dict(state.get("evidence_packet", {}) or {})
        pkt["excel_evidence"] = {
            "found": bool(result.get("excel_evidence_found")),
            "proves": proves,
            "supporting_fields": supporting_fields,
            "evidence_rows": evidence_rows,
            "gap": gaps,
            "findings": findings,
        }
        inventory = merge_collected_evidence(
            state.get("collected_evidence"), finding_evidence_items(findings)
        )
        return {"evidence_packet": pkt,
                "excel_evidence_gaps": gaps,
                "collected_evidence": inventory,
                "log": [f"[excel-evidence] files={len(findings)} "
                        f"found={bool(result.get('excel_evidence_found'))} "
                        f"fields={len(supporting_fields)} gaps={len(gaps)}"]}
    except Exception as e:
        return {"log": [f"[excel-evidence] error: {str(e)[:80]}"]}


def _xlsx_name(case: dict) -> str:
    for a in case.get("attachments", []) or []:
        if isinstance(a, dict) and _is_openpyxl_workbook(a):
            return a.get("name", "workbook.xlsx")
    return "workbook.xlsx"


def sme_review(state: AuditState) -> dict:
    return {"hitl_required": True,
            "log": [f"[sme] REVIEW REQUIRED — flags={state.get('flags', [])}"]}


def agent_wiki_ingest(state: AuditState) -> dict:
    """State-5: INGEST the resolved case into the markdown Precedent Wiki.

    Leakage guard: if WIKI_INGEST_OFF=1 (set during eval), this is a NO-OP so a
    graded case can never write to the wiki it is being scored against.
    """
    import os
    if os.environ.get("WIKI_INGEST_OFF") == "1":
        return {"log": ["[wiki-ingest] SKIPPED (WIKI_INGEST_OFF=1)"]}
    try:
        r = wiki_writer.ingest(state)
        written = r.get("wiki_pages_written", []) or []
        type_pages = [p for p in written if str(p).startswith("types/")]
        return {
            "wiki_pages_written": written,
            "wiki_page_count": len(written),
            "wiki_page_id": type_pages[0] if type_pages else "",
            "pattern_exists_in_wiki": len(written) > 0,
            "log": [f"[wiki-ingest] wrote {len(written)} pages: {written}"]
        }
    except Exception as e:
        return {"log": [f"[wiki-ingest] error: {str(e)[:100]}"]}


# ---------------------------------------------------------------- routing
def _needs_sme(state: AuditState) -> Literal["sme", "draft"]:
    return "sme" if state.get("reasoning_flag") else "draft"


# ---------------------------------------------------------------- BUILD pipeline
def build_pipeline(checkpointer=None):
    g = StateGraph(AuditState)
    g.add_node("intake", intake)
    g.add_node("summarizer", core.summarizer_classifier)
    g.add_node("excel", excel_node)
    g.add_node("retrieve", core.resolve_and_retrieve)
    g.add_node("assembly", core.assembly)
    g.add_node("reconstruction", core.reconstruction)
    g.add_node("planner", evidence_planner_node)        # typed evidence needs + JADE resolve
    g.add_node("excel_evidence", excel_evidence_node)   # directed, after precedent
    g.add_node("gap_finalize", evidence_gap_finalize_node)
    g.add_node("synthesis", core.evidence_synthesis)    # decides FINAL stance after evidence
    g.add_node("verifier", core.verifier)
    g.add_node("sme", sme_review)
    g.add_node("drafter", core.response_drafter)
    g.add_node("wiki_ingest", agent_wiki_ingest)

    g.add_edge(START, "intake")
    g.add_edge("intake", "summarizer")
    g.add_edge("summarizer", "excel")
    g.add_edge("excel", "retrieve")
    g.add_edge("retrieve", "assembly")
    g.add_edge("assembly", "reconstruction")
    g.add_edge("reconstruction", "planner")             # provisional theory -> plan evidence needs
    g.add_edge("planner", "excel_evidence")             # dig for specifics we now know we need
    g.add_edge("excel_evidence", "gap_finalize")        # promote new gaps into typed requests
    g.add_edge("gap_finalize", "synthesis")             # weigh evidence -> FINAL stance
    g.add_edge("synthesis", "verifier")                 # ground-check the final stance
    g.add_conditional_edges("verifier", _needs_sme, {"sme": "sme", "draft": "drafter"})
    g.add_edge("sme", "drafter")
    g.add_edge("drafter", "wiki_ingest")
    g.add_edge("wiki_ingest", END)

    return g.compile(
        checkpointer=checkpointer or MemorySaver(),
        interrupt_before=["sme"],
    )


# ---------------------------------------------------------------- INFERENCE pipeline
def wiki_query_node(state: AuditState) -> dict:
    """Consult the markdown Precedent Wiki (index-first)."""
    r = wiki_query.query(state)
    strength = r.get("precedent_strength", "none")
    out = {"match_confidence": {"strong": 0.9, "weak": 0.4, "none": 0.0}.get(strength, 0.0),
           "route": {"strong": "reuse", "weak": "assist"}.get(strength, "novel"),
           "wiki_route": {"strong": "reuse", "weak": "assist"}.get(strength, "novel"),
           "pattern_exists_in_wiki": r.get("wiki_hit", False),
           "wiki_hit": r.get("wiki_hit", False),
           "wiki_selected": r.get("wiki_selected", {}),
           "wiki_match_method": r.get("wiki_match_method", "llm_index"),
           "wiki_semantic_scores": r.get("wiki_semantic_scores", {}),
           "precedent_strength": strength,
           "precedent_context": r.get("precedent_context", ""),
           "case_signature": r.get("case_signature", {}),
           "precedent_matches": r.get("precedent_matches", []),
           "precedent_candidate_matches": r.get("precedent_candidate_matches", []),
           "log": [f"[wiki-query] strength={strength} matcher="
                   f"{r.get('wiki_match_method', 'llm_index')} "
                   f"selected={r.get('wiki_selected')}"]}
    return out


def _inference_gate(state: AuditState) -> Literal["reuse", "build"]:
    route = state.get("wiki_route") or state.get("route")  # route is a legacy alias
    return "reuse" if route == "reuse" else "build"


def inference_pipeline(checkpointer=None):
    g = StateGraph(AuditState)
    g.add_node("intake", intake)
    g.add_node("summarizer", core.summarizer_classifier)
    g.add_node("excel", excel_node)                    # State-2 evidence before reconstruction/wiki routing
    g.add_node("wiki_query", wiki_query_node)
    g.add_node("reuse_retrieve", core.resolve_and_retrieve)  # reuse still needs a grounded clause pool + COPAS edition
    g.add_node("reuse_assembly", core.assembly)         # scope-aware governing-authority selection
    g.add_node("reuse_planner", evidence_planner_node)   # precedent gives approach -> plan JADE/evidence
    g.add_node("reuse_pob", pob_agent_node)              # agent-invoked POB/presence tool (ToolNode)
    g.add_node("excel_evidence", excel_evidence_node)
    g.add_node("reuse_gap_finalize", evidence_gap_finalize_node)
    g.add_node("reuse_synthesis", core.evidence_synthesis)  # confirm final stance from evidence
    g.add_node("reuse_verifier", core.verifier)          # reuse still requires final grounding
    g.add_node("reuse_draft", core.response_drafter)
    g.add_node("retrieve", core.resolve_and_retrieve)
    g.add_node("assembly", core.assembly)
    g.add_node("reconstruction", core.reconstruction)
    g.add_node("build_planner", evidence_planner_node)
    g.add_node("build_pob", pob_agent_node)              # agent-invoked POB/presence tool (ToolNode)
    g.add_node("build_excel_evidence", excel_evidence_node)
    g.add_node("build_gap_finalize", evidence_gap_finalize_node)
    g.add_node("build_synthesis", core.evidence_synthesis)
    g.add_node("verifier", core.verifier)
    g.add_node("build_draft", core.response_drafter)

    g.add_edge(START, "intake")
    g.add_edge("intake", "summarizer")
    g.add_edge("summarizer", "excel")
    g.add_edge("excel", "wiki_query")
    # STRONG precedent (reuse): the wiki gives the APPROACH, then the agent still
    # explores the excels (directed by that precedent) to CONFIRM the deciding facts
    # before drafting — it does NOT short-circuit past evidence.
    # WEAK/NONE (build): full retrieve -> assemble -> reconstruct -> excel dig.
    # STRONG precedent (reuse): precedent supplies the APPROACH, then the agent still
    # plans + gathers evidence (JADE/workbook) and runs synthesis to CONFIRM the final
    # stance before drafting — it never short-circuits past evidence.
    g.add_conditional_edges("wiki_query", _inference_gate,
                            {"reuse": "reuse_retrieve", "build": "retrieve"})
    # reuse: precedent supplies the APPROACH. We still resolve and assemble a
    # focused governing-authority set so a mechanically balanced but irrelevant
    # clause cannot flow directly into the response. Reconstruction stays OUT of
    # reuse because the precedent already supplies that theory.
    g.add_edge("reuse_retrieve", "reuse_assembly")
    g.add_edge("reuse_assembly", "reuse_planner")
    g.add_edge("reuse_planner", "reuse_pob")
    g.add_edge("reuse_pob", "excel_evidence")
    g.add_edge("excel_evidence", "reuse_gap_finalize")
    g.add_edge("reuse_gap_finalize", "reuse_synthesis")
    g.add_edge("reuse_synthesis", "reuse_verifier")
    g.add_edge("reuse_verifier", "reuse_draft")
    g.add_edge("reuse_draft", END)
    g.add_edge("retrieve", "assembly")
    g.add_edge("assembly", "reconstruction")
    g.add_edge("reconstruction", "build_planner")
    g.add_edge("build_planner", "build_pob")
    g.add_edge("build_pob", "build_excel_evidence")
    g.add_edge("build_excel_evidence", "build_gap_finalize")
    g.add_edge("build_gap_finalize", "build_synthesis")
    g.add_edge("build_synthesis", "verifier")
    g.add_edge("verifier", "build_draft")
    g.add_edge("build_draft", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())
