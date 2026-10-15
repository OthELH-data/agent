"""Typed pipeline state (State1..State5) matching the v2 architecture PDF.

The architecture defines a state that ACCUMULATES as it flows through the
build/learning pipeline:

  State1 { dollars, summary, docs_referenced, category }              <- Summarizer+Classifier
  State2 { state1, calculation_summary, keywords, calculation_reasoning,
           collected_evidence }                                          <- Excel Extractor
  State3 { dollars, dispute_summary, calculation_reasoning, keywords,
           potential_discrepancies, sections_from_governing_docs,
           evidence_packet }                                          <- Assembly agent
  State3'{ + reconstruction_rationale }                               <- Reconstruction agent
  StateSynth { evidence_coverage, proof_assessment, auditor_burden_status,
               disposition }                                           <- Evidence Synthesis
  State4 { verification_of_cited_sources, reasoning_flag }             <- Verifier agent
  State5 { audit_exception_types, dollar_value_bucket,
           pattern_exists_in_wiki, reconciliation_approach,
           reconciliation_steps, citations }                         <- Agent Wiki (logged)

We model this as ONE TypedDict that grows; each node fills its slice.
"""
from __future__ import annotations
from typing import Annotated, Any, Literal, TypedDict
import operator


# ---- sub-structures ------------------------------------------------------
class Attachment(TypedDict, total=False):
    name: str
    path: str
    kind: str            # pdf | xlsx | msg | image
    text: str            # extracted text (may be empty)
    vlm_needed: bool     # heuristic: scanned/needs a vision model
    source: str          # auditor | operator


class Citation(TypedDict, total=False):
    source_doc: str
    section: str
    page: int | None
    quote: str           # verbatim
    rule_id: str
    copas_edition: str


class EvidenceItem(TypedDict, total=False):
    """One artifact already present in the case record.

    Availability and sufficiency are intentionally separate: ``status`` says
    whether the artifact was listed/extracted/analyzed, while ``supports``
    records facts found in it.  Synthesis decides whether those facts satisfy a
    proof requirement.
    """
    evidence_id: str
    source_party: str
    artifact_name: str
    artifact_type: str
    status: str
    description: str
    supports: list[str]
    locations: list[str]
    provenance_role: str  # source_fact | extractor_interpretation | derived_calculation


class CasePacket(TypedDict, total=False):
    exception_id: str
    asset: str
    year: str
    title: str
    dollars: float
    auditor_request_text: str
    error_type: str          # auditor's COPAS error code (drives deterministic classification)
    attachments: list[Attachment]


# ---- nested sub-states (lead's shape: State1..State5 grouped) ------------
# These group the flat fields by pipeline stage. Introduced additively; the
# flat keys below remain for backward-compat during migration. New code should
# prefer the grouped accessors in graph/state_helpers.py.
try:
    from pydantic import BaseModel, Field

    class State1(BaseModel):          # Summarizer + Classifier
        dollars: float = 0.0
        summary: str = ""
        category: str = ""
        error_code: str = ""
        code_family: str = ""
        code_tier_hint: int | None = None
        classification_source: str = ""

    class State2(BaseModel):          # Excel Extractor
        calculation_summary: str = ""
        keywords: list[str] = Field(default_factory=list)
        calculation_reasoning: str = ""
        excel_mode: str = "none"
        potential_discrepancies: list[str] = Field(default_factory=list)

    class State3(BaseModel):          # Assembly + Reconstruction (provisional theory)
        dispute_summary: str = ""
        reconstruction_rationale: str = ""   # provisional Shell-first theory
        provisional_stance: str = ""          # lean disposition BEFORE evidence (not final)
        theory_hypotheses: list[str] = Field(default_factory=list)  # what must hold to win
        auditor_weak_points: list[str] = Field(default_factory=list)
        timing_flags: list[str] = Field(default_factory=list)

    class StateSynth(BaseModel):      # Evidence Synthesis (decides FINAL stance)
        final_stance_basis: str = ""          # why the evidence supports the final stance
        evidence_supports: list[str] = Field(default_factory=list)
        evidence_gaps: list[str] = Field(default_factory=list)

    class State4(BaseModel):          # Verifier
        reasoning_flag: bool = False   # default False — only True on genuine grounding failure
        verifier_notes: str = ""

    class State5(BaseModel):          # Wiki
        dollar_value_bucket: str = ""
        pattern_exists_in_wiki: bool = False
        reconciliation_approach: str = ""
        reconciliation_steps: list[str] = Field(default_factory=list)

    class ClauseStore(BaseModel):
        """SINGLE source of truth for clause data. Verbatim quotes live ONLY in
        `pool` (optionally externalized to scratch via `pool_ptr`); the selected
        and cited sets are just rule_id references — no duplicated quote text."""
        pool_ptr: str = ""                 # 'run_id/pool' pointer to the full clause list
        pool: list[dict] = Field(default_factory=list)   # inline fallback if not externalized
        selected_ids: list[str] = Field(default_factory=list)   # assembly's picks
        cited_ids: list[str] = Field(default_factory=list)      # final citations
except Exception:  # pydantic missing — grouped models unavailable, flat keys still work
    BaseModel = object  # type: ignore


# ---- the flowing pipeline state -----------------------------------------
class AuditState(TypedDict, total=False):
    # scratch-store pointer for this run (heavy blobs live on disk, not in state)
    run_id: str
    # single clause store (replaces retrieved_clauses/sections/citations triplication)
    clauses: dict
    # input
    case: CasePacket

    # State1 — Summarizer + Classifier
    dollars: float
    summary: str
    docs_referenced: list[str]
    category: str
    code_family: str         # deterministic sub-family from the error code
    code_tier_hint: int      # automation-tier hint (1/2/3) from the code registry
    error_code: str          # parsed COPAS code, e.g. '03.1'
    classification_source: str  # 'error_code:03.1' or 'llm_fallback'

    # State2 — Excel Extractor
    calculation_summary: str
    keywords: list[str]  # evidence metadata only; not a retrieval query
    calculation_reasoning: str
    excel_mode: Literal["deterministic", "deep_agent", "preextracted", "none"]
    potential_discrepancies: list[str]
    collected_evidence: list[EvidenceItem]  # attached artifacts, enriched as extractors run
    evidence_questions: list[str]           # reviewer follow-ups; not automatically missing proof

    # retrieval / applicability
    copas_edition: str
    agreement_chain: list[str]          # asset hierarchy (self + host ancestors)
    retrieved_clauses: list[Citation]
    otl_pinpoint_required: bool
    otl_pinpoint_status: str            # not_applicable | unavailable | unverified | verified

    # State3 — Assembly + Reconstruction (PROVISIONAL theory, not final stance)
    dispute_summary: str
    sections_from_governing_docs: list[Citation]
    contract_inventory: dict[str, Any]    # registered coverage, not universe completeness
    evidence_packet: dict[str, Any]
    reconstruction_rationale: str         # provisional Shell-first theory
    provisional_stance: str               # lean disposition before evidence (deny by default)
    theory_hypotheses: list[str]          # facts that must hold for the theory to win
    # Evidence Synthesis — decides the FINAL disposition AFTER evidence retrieval
    final_stance_basis: str
    evidence_supports: list[str]
    evidence_gaps: list[str]
    evidence_requests: list[str]       # missing DECISIVE facts to ask the auditor for (gap-driven)
    evidence_request_details: list[dict[str, Any]]  # owner/artifact/missing content/blocking
    precedent_decision: str            # adopt | adapt | ignore — how the precedent was used
    precedent_rationale: str           # one-line justification for the precedent decision
    excel_evidence_gaps: list[str]     # extractor-flagged gaps, carried as first-class signals
    gap_assessment: str                # decisive | non_decisive | closed | unassessed
    gap_rationale: str
    evidence_plan: dict[str, Any]      # planner output: typed needs + jade findings + needs_human + authority
    evidence_coverage: list[dict[str, Any]]  # planned requirements vs already-collected artifacts
    proof_assessment: list[dict[str, Any]]   # synthesis sufficiency decision per requirement/artifact
    auditor_burden_status: str         # met | unmet | unclear | not_applicable
    auditor_burden_rationale: str
    operator_evidence_required: bool   # true when Shell-side proof is still decisive despite burden result
    evidence_conclusive: bool          # true only when current evidence closes the decisive merits question
    otl_chargeability: dict[str, Any]  # structured technical/route/benefit decision record
    financial_scope: dict[str, Any]    # quantified population required for accept/partial
    decision_control_flags: list[str]  # deterministic post-synthesis safety findings
    evidence_request_conflicts: list[dict[str, Any]]  # requests that appear to duplicate attached artifacts
    auditor_weak_points: list[str]     # out-of-context clauses / unsupported claims
    timing_flags: list[str]            # TECO / premature-reconciliation notes
    verifier_notes: str

    # State4 — Verifier
    verification_of_cited_sources: list[dict]
    reasoning_flag: bool                # true => needs SME / not fully grounded

    # State5 — Wiki (query / ingest)
    audit_exception_types: list[str]
    dollar_value_bucket: str
    pattern_exists_in_wiki: bool
    wiki_page_id: str
    match_confidence: float
    reconciliation_approach: str
    reconciliation_steps: list[str]
    citations: list[Citation]
    # precedent query results (inference pipeline)
    wiki_hit: bool
    wiki_route: str                   # reuse | assist | novel
    wiki_selected: dict[str, list[str]]  # selected page slugs grouped by section
    wiki_match_method: str              # llm_index | llm_index+hf_embedding
    wiki_semantic_scores: dict[str, float]
    precedent_context: str            # full precedent text fed to agents
    precedent_strength: str           # strong | weak | none
    case_signature: dict[str, Any]     # structured incoming-case signature
    precedent_matches: list[dict[str, Any]]  # selected direct | analogical | weak/background
    precedent_candidate_matches: list[dict[str, Any]]  # all semantic candidates before LLM
    # ingest results (build pipeline)
    wiki_pages_written: list[str]     # page paths written this run
    wiki_page_count: int              # total pages written

    # draft / output
    draft_response: str
    response_sections: dict[str, str]     # original_ask | analysis | final_response | co_owner_response
    co_owner_response: str                # brief paragraph ready to paste into the co-owner response
    response_attachments: list[Any]       # explicitly confirmed outward attachments; never inferred
    response_template: str
    recommendation_status: str            # pending_human_approval | not_applicable
    disposition: str                    # deny | partial | accept | needs_evidence

    # control / audit trail (append-only log)
    log: Annotated[list[str], operator.add]
    route: str                          # legacy alias for wiki_route
    hitl_required: bool
