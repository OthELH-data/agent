"""Typed graph state for the OTL orchestration layer.

Two layers (design doc line 68):

* **Per-agent sub-states** — one Pydantic model per agent, each carrying an `nl_summary: str`
  (design doc line 66) plus that agent's typed findings. A node returns its sub-state under its own
  key, e.g. `{"jade_analyzer": JadeState(...), ...shared}`. Because each key has exactly one writer,
  LangGraph's default last-write-wins channel is correct — no reducer needed for the sub-state keys.

* **Master `OTLAuditState`** — a flat `TypedDict(total=False)` holding every sub-state under its key,
  the orchestrator's hub-and-spoke channels (`next_agent`, `final_mode`, the hop/run-count loop
  guards, `agent_instructions`, `agent_status`), the `auditor_question` / HITL fields, the resolver
  paths, the reviewer-flagged `open_questions` forwarded to specialists (hints, never a gate), and the
  trace channels. Routing is no longer edge-computed: a highest-tier `orchestrator` node picks the
  next hop each pass (hub-and-spoke), and the conditional edge just reads `next_agent`.

Trace channels use reducers so parallel/sequential appends accumulate instead of overwriting:
  * `messages`     — `add_messages` (the permanent, UI-readable message log)
  * `trace`        — `operator.add` (append-only structured events)
  * `nl_summaries_by_node` — `merge_dict` (agent → bounded narrative summaries)
"""
from __future__ import annotations

import operator
from datetime import date
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# The hard evidence gates distinguish an absent auditor record from a supplied record that cannot be
# read. The distinction determines whether the drafter denies the exception or defers it to HITL.
EvidenceGateStatus = Literal["", "missing_auditor_evidence", "unreadable_auditor_evidence"]


class SourceRowRef(BaseModel):
    """Stable internal locator for one source-ledger row."""
    workbook: str
    sheet: str
    excel_row: int


class LedgerRow(BaseModel):
    """One normalized JADE detail row; unavailable optional columns remain absent/None."""
    source_ref: SourceRowRef
    person: str | None = None
    employee_id: str | None = None
    wbs: str | None = None
    wbs_description: str | None = None
    account_code: str | None = None
    account_description: str | None = None
    amount: float | None = None
    jv_object_type: str | None = None
    mcoh_status: Literal["mcoh", "non_mcoh"] | None = None
    qty_local_uom: float | None = None
    entry_date_sl: date | None = None
    cost_center_code: str | None = None
    cost_center_name: str | None = None
    line_item_text: str | None = None


class McohMonthlyFact(BaseModel):
    person: str
    submission_month: str
    hours: float
    threshold_hours: float = 8.0
    below_threshold: bool
    row_refs: list[SourceRowRef] = Field(default_factory=list)
    amount: float = 0.0
    threshold_source: str = "configured_default_pending_terminal_authority"


class HookFact(BaseModel):
    value: Any
    status: Literal["resolved", "unresolved"] = "resolved"
    source_node: str
    source_refs: list[str] = Field(default_factory=list)
    rationale: str = ""


class ChargeScope(BaseModel):
    scope_id: str
    person: str | None = None
    period: str | None = None
    row_refs: list[str] = Field(default_factory=list)
    questioned_amount: float | None = None
    facts: dict[str, HookFact] = Field(default_factory=dict)


class HookEvaluation(BaseModel):
    hook_id: str
    scope_id: str
    status: Literal["pending", "fired", "not_matched", "unresolved"]
    missing_conditions: list[str] = Field(default_factory=list)
    supporting_facts: dict[str, HookFact] = Field(default_factory=dict)
    decision: str = ""
    disposition: str = ""
    outcome: str = ""


class ChargeabilityResolution(BaseModel):
    status: Literal["pending", "resolved", "needs_evidence"] = "pending"
    disposition: Literal["", "deny", "accept", "partial", "needs_evidence"] = ""
    fired_hooks: dict[str, list[str]] = Field(default_factory=dict)
    person_results: dict[str, dict[str, Any]] = Field(default_factory=dict)
    resolved_deny_amount: float = 0.0
    resolved_accept_amount: float = 0.0
    unresolved_amount: float = 0.0
    unresolved_people: list[str] = Field(default_factory=list)
    summary: str = ""


# --------------------------------------------------------------------------------------------------
# reducers
# --------------------------------------------------------------------------------------------------
def merge_dict(left: dict | None, right: dict | None) -> dict:
    """Shallow-merge reducer: right wins on key collision. Used for `nl_summaries_by_node`."""
    out = dict(left or {})
    out.update(right or {})
    return out


# --------------------------------------------------------------------------------------------------
# per-agent sub-states (each carries a bounded natural-language summary)
#
# NOTE on the Field(description=...) below: these are HUMAN-FACING documentation. Unlike the
# `_XxxExtraction` schemas in the agent modules (whose descriptions ARE serialized into the JSON schema
# the LLM fills), a sub-state is handed to a downstream LLM only via `model_dump()` — which emits the
# field VALUES, not their descriptions. So these descriptions document the state for maintainers; the
# text an LLM actually reads about a field lives on the corresponding extraction schema.
# --------------------------------------------------------------------------------------------------
class IntakeState(BaseModel):
    """intake: resolved case facts plus summary and people mentioned in extractor analysis."""
    nl_summary: str = Field(default="", description="Bounded analysis of notable intake facts and relevant gaps.")
    assets_in_question: list[str] = Field(
        default_factory=list,
        description="Canonical asset keys from configs/agreement_corpus.yaml, used for agreement scope.")
    assets_as_seen: list[str] = Field(
        default_factory=list,
        description="Corresponding asset names as supplied by the resolved case/folder for display and auditability.")
    dollars: float = Field(
        default=0.0, description="Disputed amount from the resolved classified case state.")
    collected_evidence: list[str] = Field(
        default_factory=list,
        description="Response-filtered names of resolved workbook and supporting-PDF attachments.")
    error_code: str = Field(
        default="", description="Canonical COPAS token, e.g. '09.1' ('' if unparseable). Deterministic; trace/triage only.")
    code_family: str = Field(
        default="",
        description="'otl_labor' | 'otl_timewriting' | 'affiliate' | 'unknown'. Deterministic; trace/triage only.")
    people_found: list[str] = Field(
        default_factory=list,
        description="Distinct people explicitly named in the extractor synthesis or attachment analysis. "
                    "Lead list only: JADE resolves it with ledger values and POS suggestions before "
                    "writing the authoritative downstream person population to people_found_in_ledger.")


class CategoryCheckState(BaseModel):
    """Current analytic questions and distinct attestation/invoice/support request categories.

    Drives the orchestrator's dispatch AND the verify_evidence category-driven gate. The categories
    are independent, so any subset may be present when the auditor raises multiple questions.
    """
    nl_summary: str = Field(default="", description="Bounded analysis of the dispute categories and notable ambiguities.")
    auditor_question: list[str] = Field(
        default_factory=list,
        description="Current-cycle categories explicitly identified by category_check. The master "
                    "`auditor_question` channel may additionally include analytic categories reintroduced "
                    "deterministically from current rebuttal cycle-delta grounds. 'other' appears only by itself.")
    attestation_requested: bool = Field(
        default=False,
        description="True when the auditor requested an attestation or affiliate certificate, whether alone or with other questions.")
    third_party_invoice_requested: bool = Field(
        default=False,
        description="True when the active cycle specifically requests an underlying third-party/vendor invoice.")
    additional_support_requested: bool = Field(
        default=False,
        description="True only for a current unresolved non-third-party-invoice request to provide evidence, documents, records, data, support, calculations, or information.")


class VerifyEvidenceState(BaseModel):
    """verify_evidence_sufficiency: is the auditor's cited evidence enough to proceed?

    `synthesis_json` (the full extractor payload) lives ONLY here. `cited_workbooks`/`cited_sheets`
    are emitted with the OPERATOR's material filtered out at BOTH levels — whole files and individual
    tabs (response / operator reply / rebuttal / to auditor / operator comments, all years, via the
    shared predicate in `extractor/response_names.py`) — so an operator-response sheet is never handed
    downstream even when the staged extractor JSON lists it. Names attributing a reply or comments to the
    AUDITOR are auditor-side and ARE kept.
    """
    nl_summary: str = Field(default="", description="Bounded analysis of cited evidence, gaps, and the resulting gate posture.")
    synthesis_json: dict = Field(
        default_factory=dict,
        description="The full extractor synthesis payload — lives ONLY here (dropped from the drafter "
                    "digest as too large). The interpreted narrative reaches downstream LLMs via "
                    "synthesis_block/synthesis_narrative instead.")
    pob_data_found: bool = Field(
        default=False, description="Auditor cited personnel-on-board data for the charge period.")
    heli_data_found: bool = Field(
        default=False, description="Auditor cited helicopter-manifest/flight-log data (may share POB's tab).")
    hierarchy_data_found: bool = Field(
        default=False,
        description="Auditor cited proof of the supervision hierarchy (org chart / reporting structure) "
                    "of the questioned people. Required when the active question includes hierarchy.")
    ledger_data_found: bool = Field(
        default=False,
        description="A readable JADE or general-ledger tab was identified deterministically.")
    workbook_access_issues: list[str] = Field(
        default_factory=list,
        description="Auditor workbook access failures used to distinguish missing from unreadable evidence.")
    shell_invoice_found: bool = Field(
        default=False,
        description="A Shell/SIEP seller invoice was established in current extractor evidence or prior structured history.")
    shell_invoice_sources: list[str] = Field(
        default_factory=list, description="Current or prior invoice filenames supporting shell_invoice_found.")
    shell_invoice_match_basis: list[str] = Field(
        default_factory=list, description="Filename/vendor/history facts that established the Shell invoice.")
    evidence_gate_status: EvidenceGateStatus = Field(
        default="",
        description="Aggregate configured cause: missing_auditor_evidence, "
                    "unreadable_auditor_evidence, or empty when every applicable rule passes.")
    cited_workbooks: list[str] = Field(
        default_factory=list,
        description="Cited workbook paths, operator-response files/tabs removed (all years).")
    cited_sheets: dict[str, list[str]] = Field(
        default_factory=dict, description="file -> [tab names] surviving the operator-response filter.")
    open_questions: list[str] = Field(
        default_factory=list,
        description="The extractor's reviewer-flagged gaps, passed through as downstream hints — NEVER a "
                    "gate. Mirrored onto the master `open_questions` channel.")
    evidence_gate_evaluations: list[dict[str, Any]] = Field(
        default_factory=list, description="Configured ED rule evaluations in stable gate order.")
    evidence_gate_decision: dict[str, Any] = Field(
        default_factory=dict, description="Aggregate configured hook decision consumed by routing and the drafter.")


class OrchestratorState(BaseModel):
    """orchestrator: the LLM supervisor's per-hop decision trail (hub-and-spoke, highest tier).

    The hub between `verify_evidence` and the terminal `drafter`. It reads every sub-state + status
    returned by the specialists and, each pass, dispatches exactly ONE specialist (with a custom
    instruction) or finalizes. Single writer — `decision_log` grows by read-modify-write across hops,
    so no reducer is needed. `final_mode` is the terminal decision the drafter keys off.
    """
    nl_summary: str = Field(default="", description="Bounded analysis of the routing decision, rationale, and unresolved work.")
    decision_log: list[dict] = Field(
        default_factory=list,
        description="Per-hop trail: {hop, chose, rationale, status_seen, [final_mode]}. Grows by "
                    "read-modify-write across hops (single writer).")
    final_mode: str = Field(
        default="", description="'terminal' | 'needs_evidence' — set on the finalize hop; the drafter keys off it.")
    hops: int = Field(default=0, description="Number of orchestrator passes so far (loop-guard mirror).")
    invoke_job_title_after_location: bool = Field(
        default=False,
        description="True iff category_check included 'location'. When True the orchestrator preserves "
                    "the pinned `location` then `job_title` sequence — physical presence alone cannot "
                    "close a location question because a technical role is chargeable by FUNCTION. "
                    "Recorded here per hop for the trace; the authoritative value is always derived "
                    "from `auditor_question`.")


class JadeState(BaseModel):
    """jade_analyzer: reconstruct the questioned amount from the JADE ledger, and name who it charges.

    Column names vary case-to-case (design doc line 21) — the agent REASONS the mapping rather than
    hard-coding it, and records what it chose in `resolved_columns` for the trace.

    This agent is also where the pipeline's person list is settled. Before its ReAct loop starts, the light
    tier resolves ledger-name-column values, POS suggestions, and intake's named-person leads through
    `otl/ledger_people.py`, writing the result to `people_found_in_ledger` — the source list every later
    specialist caps and reads via `agents/_common.disputed_people`. The `ledger_*` / `people_*` fields below
    are written by CODE, not by the ReAct `finish` payload.

    While it scans the ledger per person, the agent also runs the MAJOR-CONSTRUCTION SCREEN: JADE carries
    no "JV Object Type Description" text column, only a coded "WBS:JV Object Type" (or "JV Object Type")
    value per row — a short code such as "C142". The agent reads that code per person and checks it
    against `otl/jv_object_types.py`'s registry (an exact dict lookup its own sandboxed code performs),
    and flags MCOH (major construction overhead) when the code matches. MCOH labor is directly chargeable
    under the settlement's Agreed OTL Definition and Billing Standards, so the flag travels to the
    orchestrator as a chargeability fact rather than as a ledger curiosity — but it does NOT override the
    above-first-level-supervision exclusion; a person confirmed above first level is overhead regardless.
    """
    nl_summary: str = Field(default="", description="Bounded analysis of amount reconstruction, ledger support, and unresolved issues.")
    evidence_gate_status: EvidenceGateStatus = Field(
        default="",
        description="Legacy compatibility field. Configured evidence policy belongs to verify_evidence; "
                    "JADE leaves this empty for operationally inconclusive ledger analysis.")
    evidence_access_issues: list[str] = Field(
        default_factory=list,
        description="Technical file or worksheet access failures retained for the report and trace; never "
                    "copied verbatim into the auditor-facing response.")
    resolved_columns: dict[str, str] = Field(
        default_factory=dict,
        description="The column mapping the agent REASONED (not hard-coded): "
                    "{wbs, name, cost_plus_oh, oh_rate, base_amount} -> actual header. Recorded for the trace.")
    jade_wbs_codes: list[str] = Field(default_factory=list, description="WBS codes matched in the ledger.")
    jade_line_item_texts: list[str] = Field(
        default_factory=list, description="Line-item-text values (names live here when Employee is blank).")
    overhead_mode: str = Field(
        default="", description="'cost_plus_oh_column' | 'computed_2.5pct_on_actual'.")
    overhead_rate: float = Field(default=0.025, description="Overhead rate used (default 2.5%).")
    amount_basis: str = Field(default="", description="'sum' | 'half' — which reconstruction matched the questioned figure.")
    amount_computed: float = Field(default=0.0, description="The reconstructed amount on that basis.")
    amount_questioned: float = Field(default=0.0, description="The auditor's questioned figure being checked.")
    amount_in_range: bool = Field(default=False, description="Code-derived: selected sum/half basis is within inclusive ±2.5% of the questioned amount.")
    jade_amount_discrepancy: bool = Field(
        default=False, description="Code-derived: selected sum/half basis falls outside ±2.5%.")
    jade_hook_evaluations: list[dict[str, Any]] = Field(
        default_factory=list, description="Configured jade_analyzer hook evaluations.")
    jade_hook_decision: dict[str, Any] = Field(
        default_factory=dict, description="Aggregate jade_analyzer hook decision consumed by routing and drafter.")
    charged_amount_by_person: dict[str, float] = Field(
        default_factory=dict,
        description="Each final resolved person with matched JADE rows -> deterministic sum of that person's "
                    "Cost + OH rows, before any explicitly configured downstream specialist cap. These source-ledger "
                    "totals are not changed by the ReAct reconciliation basis (sum versus half).")
    ledger_rows: list[LedgerRow] = Field(
        default_factory=list,
        description="Normalized in-scope JADE detail rows. Optional fields are absent when their column "
                    "or cell could not be resolved; person is optional and never a prerequisite for a row.")
    mcoh_monthly_facts: list[McohMonthlyFact] = Field(
        default_factory=list,
        description="Identified-person MCOH hours grouped by Entry Date SL submission month.")
    cats_timewriting_present: bool = False
    asset_cost_center_people: list[str] = Field(default_factory=list)
    jade_analysis_mode: str = Field(default="", description="structured | code_escalation")
    jade_escalation_reasons: list[str] = Field(default_factory=list)
    # --- major-construction screen: run per person as the ledger is scanned (see agents/jade_analyzer.py) ---
    jv_object_type_by_person: dict[str, str | list[str]] = Field(
        default_factory=dict,
        description="grounded people_found_in_ledger person -> the coded JV object-type value(s) on that "
                    "person's charge rows, verbatim "
                    "as the ledger writes them (e.g. 'C142'). Read from the 'WBS:JV Object Type' column "
                    "or whatever the case's ledger calls it; empty when the cited ledger carries no such "
                    "column.")
    major_construction_by_person: dict[str, bool] = Field(
        default_factory=dict,
        description="grounded people_found_in_ledger person -> any of their charge rows carry a code registered in "
                    "otl/jv_object_types.py as MCOH (major construction overhead) — an exact code-table "
                    "lookup, not a text pattern. True is a CHARGEABILITY fact, not just a label: labor on "
                    "a major construction project is directly chargeable under the settlement's Agreed "
                    "OTL Definition and Billing Standards, independently of the CFR category and of "
                    "where the person physically was — though NOT independently of above-first-level "
                    "supervision, which is judged separately and excludes the person regardless.")
    major_construction_people: list[str] = Field(
        default_factory=list,
        description="The people flagged True above. Derived IN CODE from major_construction_by_person, "
                    "so the two can never disagree.")
    major_construction_charge: bool = Field(
        default=False,
        description="True when any examined person's rows carry MCOH. Derived in code, and surfaced to "
                    "the orchestrator, which weighs it as a chargeability basis (see "
                    "agents/orchestrator.py `_salient_nl_summaries`).")
    major_construction_basis: str = Field(
        default="",
        description="One line naming the column the screen read and what it held — or, when the screen "
                    "could not run, why (no such column in the cited ledger).")
    # --- the person list, settled before the ReAct loop (written in code; see otl/ledger_people.py) ---
    ledger_tabs_read: list[str] = Field(
        default_factory=list,
        description="The `file!tab` labels the name column was read from. Empty means no ledger tab was "
                    "found, which is the evidence gate.")
    ledger_person_values: list[str] = Field(
        default_factory=list,
        description="The distinct name-column values as the ledger holds them, before any name was "
                    "extracted. Kept for the trace, so a missed name can be traced to the value it was in. "
                    "Distinct from `jade_line_item_texts`, which the model writes for the amount math.")
    people_found_in_ledger: list[str] = Field(
        default_factory=list,
        description="The final names the light tier resolved from ledger values, local POS suggestions, "
                    "and intake leads, before the cap.")
    people_named_count: int = Field(
        default=0,
        description="Exactly len(people_found_in_ledger), derived in code after the ledger, POS, and "
                    "intake-lead LLM resolution pass. It is never read from the JADE ReAct payload.")


class LocationState(BaseModel):
    """location: where were the people physically, per POB + heli logs (design doc lines 30, 76)."""
    nl_summary: str = Field(default="", description="Bounded analysis of location evidence, material limitations, and unresolved points.")
    found_in_pob: dict[str, bool] = Field(default_factory=dict, description="person -> present in the POB report.")
    found_in_heli_logs: dict[str, bool] = Field(default_factory=dict, description="person -> present in the heli manifest.")
    pob_people_location: dict[str, str] = Field(default_factory=dict, description="person -> POB Location value.")
    origin: dict[str, str] = Field(default_factory=dict, description="person -> heli leg origin.")
    destination: dict[str, str] = Field(default_factory=dict, description="person -> heli leg destination.")
    start_date: dict[str, str] = Field(default_factory=dict, description="person -> POB arrival / charge-period start.")
    end_date: dict[str, str] = Field(default_factory=dict, description="person -> POB departure / charge-period end.")
    location_status: dict[str, str] = Field(
        default_factory=dict, description="person -> onsite | offsite | unresolved for the questioned period.")
    location_rationale: dict[str, str] = Field(
        default_factory=dict, description="person -> concise basis for the deterministic location status.")
    # NOTE: this agent reports WHERE people were, nothing more. It no longer carries a flag about which
    # specialist should run next — that decision moved to the orchestrator
    # (`OrchestratorState.invoke_job_title_after_location`), which derives it from `auditor_question`
    # rather than from anything this agent finds.


class JobTitleState(BaseModel):
    """job_title: resolve each person's position from the staff listing + jobs catalogue.

    Pulls BOTH the Jobs_Catalogue Job Summary AND Job Description per profile — the downstream
    `matching` agent uses the summary/description to corroborate its profile match against the CFR
    mapping workbook (design: 'make sure job_title pulls everything matching needs').
    """
    nl_summary: str = Field(default="", description="Bounded analysis of job-title evidence, role conclusions, and unresolved points.")
    identified_names_staff_listing: dict[str, str] = Field(
        default_factory=dict, description="person -> position/job profile from the staff listing.")
    identified_job_data: dict[str, str] = Field(
        default_factory=dict, description="job profile -> Job Summary (from the Jobs_Catalogue).")
    identified_job_descriptions: dict[str, str] = Field(
        default_factory=dict,
        description="job profile -> Job Description (same keys); matching uses BOTH to corroborate its "
                    "CFR-mapping profile match.")
    technical_job_title: bool | None = Field(
        default=None,
        description="Compatibility aggregate: True when all resolved people/title periods are technical, "
                    "False when all are non-technical, and None for mixed or unresolved populations. "
                    "Canonical conclusions live in role_status_by_person.")
    technical_rationale: str = Field(
        default="", description="One-two sentences citing the duties that make the role technical (or not).")
    positions_by_person: dict[str, list[str]] = Field(default_factory=dict)
    title_periods_by_person: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    selected_job_profiles_by_person: dict[str, list[str]] = Field(default_factory=dict)
    role_status_by_person: dict[str, str] = Field(
        default_factory=dict, description="person -> technical | non_technical | unclear.")
    technical_rationale_by_person: dict[str, str] = Field(default_factory=dict)
    ipt_status_by_person: dict[str, str] = Field(
        default_factory=dict, description="person -> ipt | non_ipt | unclear.")
    exempt_role_status_by_person: dict[str, str] = Field(
        default_factory=dict, description="person -> exempt | non_exempt | unclear.")
    job_profile_status_by_person: dict[str, str] = Field(
        default_factory=dict, description="person -> matched | no_match | multiple_close | unclear.")
    ipt_registry_evidence_by_person: dict[str, dict[str, Any]] = Field(default_factory=dict)
    job_title_analysis_mode: str = ""
    job_title_escalation_reasons_by_person: dict[str, list[str]] = Field(default_factory=dict)


class MatchingState(BaseModel):
    """matching: map each resolved person/profile onto the CFR reference workbook.

    This node reports regulatory mapping evidence only. Asset-specific offsite technical labor
    chargeability is decided later from the governing agreements selected by agreement_retriever.
    """
    nl_summary: str = Field(default="", description="Bounded analysis of authority mapping, material support, and unresolved points.")
    identified_cfr_category: str = Field(
        default="",
        description="Deprecated compatibility field. Matching no longer selects a chargeability category.")
    cfr_clause: str = Field(
        default="", description="Deprecated compatibility field. Governing authority comes from agreement retrieval.")
    # --- CFR mapping (profile -> value), matched on the workbook's "Job Profile Description" column ---
    matched_job_profiles: dict[str, str] = Field(
        default_factory=dict, description="job_title profile -> matched 'Job Profile Description' in the mapping workbook.")
    cfr_primary: dict[str, str] = Field(default_factory=dict, description="profile -> CFR Primary mapping.")
    cfr_secondary: dict[str, str] = Field(default_factory=dict, description="profile -> CFR Secondary mapping.")
    cfr_additional: dict[str, str] = Field(default_factory=dict, description="profile -> CFR Additional mapping.")
    subpart_labels: dict[str, str] = Field(default_factory=dict, description="profile -> Subpart Labels.")
    cfr_rationale: dict[str, str] = Field(default_factory=dict, description="profile -> mapping Rationale.")
    cfr_source_reference: dict[str, str] = Field(default_factory=dict, description="profile -> Source Reference.")
    cfr_resolution: str = Field(
        default="unclear", description="Deprecated compatibility field; matching no longer decides chargeability.")
    cfr_copas_resolution: str = Field(
        default="unclear", description="chargeable | not_chargeable | unclear, derived from CFR mapping evidence.")
    cfr_copas_resolution_by_person: dict[str, str] = Field(
        default_factory=dict, description="person -> chargeable | not_chargeable | unclear.")
    cfr_mapping_status_by_person: dict[str, str] = Field(
        default_factory=dict, description="person -> matched | no_match | multiple_close | unclear.")
    matched_cfr_profiles_by_person: dict[str, list[str]] = Field(default_factory=dict)
    cfr_primary_by_person: dict[str, list[str]] = Field(default_factory=dict)
    cfr_secondary_by_person: dict[str, list[str]] = Field(default_factory=dict)
    cfr_additional_by_person: dict[str, list[str]] = Field(default_factory=dict)
    subpart_labels_by_person: dict[str, list[str]] = Field(default_factory=dict)
    cfr_rationale_by_person: dict[str, list[str]] = Field(default_factory=dict)
    cfr_source_reference_by_person: dict[str, list[str]] = Field(default_factory=dict)
    matching_analysis_mode: str = ""
    matching_escalation_reasons_by_person: dict[str, list[str]] = Field(default_factory=dict)


class SupervisionState(BaseModel):
    """supervision: is each disputed person ABOVE first-level supervision (per the staff-listing org tree)?

    Fired when `category_check` returns a `hierarchy` dispute. Supervision at or below the first level IS
    chargeable to the joint account; anything ABOVE first level is NOT separately chargeable and must be
    accounted for as overhead. The agent walks the self-referential org tree (Employee ID ← Manager ID)
    on the same consolidated staff listing `job_title` uses: a person is above first level iff they have
    direct reports AND at least one direct report is itself a manager (has its own reports). It reconciles
    that classification against the per-person auditor-questioned amounts into credit/chargeable totals;
    the drafter (not this agent) derives the accept/partial/deny disposition from those raw facts.
    """
    nl_summary: str = Field(default="", description="Bounded analysis of supervision analysis, allocation, and unresolved points.")
    above_first_level: dict[str, bool] = Field(
        default_factory=dict,
        description="person -> above first-level supervision (has direct reports AND ≥1 report is itself "
                    "a manager). Above first level is NOT separately chargeable.")
    reports_summary: dict[str, str] = Field(
        default_factory=dict, description="person -> short human line, e.g. '6 direct reports; 3 have their own reports'.")
    questioned_amounts: dict[str, float] = Field(
        default_factory=dict, description="person -> the auditor-questioned dollar amount (for reconciliation).")
    credit_amount: float = Field(
        default=0.0, description="Σ amounts for above-first-level people (not chargeable → creditable).")
    chargeable_amount: float = Field(
        default=0.0, description="Σ amounts for people at/below first level (chargeable → charge stands).")
    needs_human_review: bool = Field(
        default=False, description="True when a person can't be located or the reporting chain is ambiguous.")
    review_reason: str = Field(default="", description="Why human review is needed (when needs_human_review).")
    supervision_status_by_person: dict[str, str] = Field(
        default_factory=dict,
        description="person -> above_first_level | not_above_first_level | unresolved.")
    supervision_record_status_by_person: dict[str, str] = Field(
        default_factory=dict, description="person -> found | missing | ambiguous.")
    hierarchy_periods_by_person: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    supervision_analysis_mode: str = ""
    supervision_escalation_reasons_by_person: dict[str, list[str]] = Field(default_factory=dict)


class WikiState(BaseModel):
    """wiki: read-only precedent lookup and final structured precedence fact.

    Raw precedent pages remain isolated from drafter prose. Only ``precedence_status`` participates in
    final declarative hook evaluation after agreement retrieval.
    """
    nl_summary: str = Field(default="", description="Bounded analysis of precedent lookup results and material limitations.")
    status: str = Field(
        default="empty",
        description="matched | no_match | empty | unavailable | error; never a graph routing gate.",
    )
    query: str = Field(default="", description="Compact current-case query used to select wiki pages.")
    precedent_strength: str = Field(
        default="none", description="strong | weak | none, reported for review only.")
    match_reason: str = Field(default="", description="Why the sidecar selected (or did not select) pages.")
    selected_pages: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Validated page slugs grouped by types, clauses, precedents, and entities.",
    )
    precedent_context: str = Field(
        default="",
        description="Selected markdown pages for independent reviewer inspection; not a drafter input.",
    )
    kb_provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="Resolved corpus path, section counts, and content fingerprint.",
    )
    precedence_status: str = Field(
        default="unavailable_or_unclear",
        description="supports_chargeability | supports_non_chargeability | unavailable_or_unclear.")


class AgreementChunk(BaseModel):
    """One source chunk selected by ID and rehydrated verbatim from the agreement index.

    ``verbatim_text`` is never accepted from an LLM response.  The agreement-retriever validates the
    ranked ID and loads this field from SQLite afterward, preserving a clean boundary between source
    language and model-generated selection metadata.
    """
    chunk_id: str
    document_id: str
    document_name: str
    document_type: str = ""
    legal_track: str
    party_scope: list[str] = Field(default_factory=list)
    asset_scopes: list[str] = Field(default_factory=list)
    heading_path: str = ""
    pdf_page_start: int
    pdf_page_end: int
    printed_page_labels: list[str] = Field(default_factory=list)
    extraction_methods: list[str] = Field(default_factory=list)
    verbatim_text: str
    content_sha256: str
    token_count: int = 0
    ocr_confidence: float = 0.0
    quote_safe: bool = False
    relevance: str = ""
    perspectives: list[str] = Field(default_factory=list)
    relationship: str = ""
    selection_reason: str = ""


class AgreementRetrievalState(BaseModel):
    """Agreement RAG output used by the drafter and preserved in the technical report."""
    nl_summary: str = Field(default="", description="Bounded analysis of selected agreement authority and retrieval limitations.")
    status: str = Field(
        default="unavailable",
        description="matched | no_match | unsupported_asset | unavailable | stale | error.",
    )
    canonical_asset: str = ""
    auditor_query: str = Field(
        default="", description="Original auditor exception text used as the primary retrieval query.")
    shell_findings_query: str = Field(
        default="", description="Deterministic additive query from specialist chargeability findings.")
    applicable_documents: list[str] = Field(default_factory=list)
    applicable_tracks: list[str] = Field(default_factory=list)
    covered_tracks: list[str] = Field(default_factory=list)
    uncovered_tracks: list[str] = Field(default_factory=list)
    selected_chunks: list[AgreementChunk] = Field(default_factory=list)
    index_provenance: dict[str, Any] = Field(default_factory=dict)
    copas_mfi_resolution: str = Field(
        default="unclear", description="supports_chargeability | supports_non_chargeability | unclear.")
    cfr_copas_resolution: str = Field(
        default="unclear", description="Deprecated compatibility field; CFR resolution belongs to matching.")
    agreement_chargeability: str = Field(
        default="unclear", description="chargeable | not_chargeable | unclear.")
    agreement_chargeability_citations: list[dict[str, Any]] = Field(default_factory=list)
    authority_rationale: str = ""
    mcoh_threshold_hours: float | None = None
    mcoh_threshold_source: str = ""


class DrafterState(BaseModel):
    """drafter: the terminal rebuttal OR the HITL deferral note (findings / why-stopped / what's-missing).

    The single graph sink. Which fields are populated depends on the mode the drafter ran in
    (EVIDENCE-GATE / TERMINAL / HITL); see drafter.py.
    """
    nl_summary: str = Field(default="", description="Bounded analysis of the disposition, support, and remaining uncertainty.")
    disposition: str = Field(
        default="", description="deny | partial | accept | needs_evidence | budget_exceeded."
    )
    draft_response: str = Field(
        default="", description="TERMINAL: the operator's rebuttal prose. EVIDENCE-GATE: the deterministic denial or accessibility-deferral message.")
    response_sections: dict[str, str] = Field(
        default_factory=dict,
        description="One client-facing response-type heading mapped to its rendered content. Detailed "
                    "terminal or HITL analysis appears as subsections beneath that heading.")
    challenged_premise: str = Field(
        default="",
        description="TERMINAL trace: the primary factual premise the drafter used to rank the case's arguments; "
                    "not repeated in the client-facing response.")
    argument_priority: list[str] = Field(
        default_factory=list,
        description="TERMINAL trace: supported argument labels in the final publication order.")
    argument_roles: dict[str, str] = Field(
        default_factory=dict,
        description="TERMINAL trace: argument label -> direct_refutation | independent_basis | "
                    "strengthening | corroboration.")
    premise_relationships: dict[str, str] = Field(
        default_factory=dict,
        description="TERMINAL trace: whether each argument contradicts, limits the scope of, "
                    "accepts/assumes, or does not address the primary factual premise.")
    argument_analysis: dict[str, str] = Field(
        default_factory=dict,
        description="TERMINAL trace: the drafter's concise case-specific reason for each classification.")
    co_owner_response: str = Field(default="", description="TERMINAL: concise co-owner summary.")
    attestation_response: str = Field(
        default="",
        description="Concise LLM-generated response to an attestation-certificate request, when present.")
    additional_support_response: str = Field(
        default="",
        description="Concise LLM-generated response to an evidence, document, or information request, when present.")
    hitl_note: str = Field(
        default="", description="HITL: deferral note — (a) findings so far, (b) why stopped, (c) what's needed.")


# --------------------------------------------------------------------------------------------------
# master state
# --------------------------------------------------------------------------------------------------
class OTLAuditState(TypedDict, total=False):
    # --- run/case header (all set by run.py before invoke; single writer) ---
    run_id: str
    case: dict[str, Any]                       # {case_id, exc_id, year, asset, title, amount, desc, ...}
    trace_path: str                            # absolute path of this case's append-only *_otl_trace.jsonl
    auditor_narrative: str                     # active-cycle auditor exception description
    latest_auditor_reply: dict[str, Any]        # final auditor reply in that cumulative latest exchange
    synthesis: dict[str, Any]                  # optional extractor synthesis; absent when disabled/missing
    extractor_synthesis_loaded: bool           # trace/header signal only; never a reasoning input itself
    processing: dict[str, Any]                 # complete/partial status and runtime-hook details
    budget_exceeded: bool                      # terminal recovery mode after the labor budget hook fires

    # --- durable cross-cycle rebuttal analysis memory ---
    rebuttal_history: list[dict[str, Any]]
    rebuttal_context: str
    rebuttal_history_compacted: bool
    rebuttal_history_source_tokens: int
    rebuttal_history_threshold_tokens: int
    rebuttal_history_cache_path: str
    rebuttal_analysis_path: str
    rebuttal_current_context: str
    rebuttal_current_posture: str
    rebuttal_prior_agents: list[str]
    rebuttal_specialist_contexts: dict[str, str]
    rebuttal_specialist_current_grounds: dict[str, str]
    rebuttal_specialist_revisit: dict[str, bool]
    rebuttal_specialist_rationales: dict[str, str]
    rebuttal_reused_nl_summaries: dict[str, str]
    rebuttal_delta_assessment_error: str

    # --- resolver paths (populated by run.py before invoke) ---
    rebuttal_dir: str
    auditor_report_pdf: str
    exception_exhibits: list[str]              # resolved auditor-side workbooks for this exception
    supporting_pdfs: list[str]                 # resolved supporting PDF attachments for this exception
    final_attachments: list[str]               # basename union of final manifest workbooks + PDFs; mirrors case.attachments
    all_workbooks_operator_responses: bool     # all candidate workbooks were excluded as operator-side
    evidence_exclusion_notice: str             # reviewer-only explanation for that exclusion
    excluded_operator_workbooks: list[str]     # retained for technical provenance, never agent evidence
    excluded_operator_pdfs: list[str]          # PDF counterparts retained for technical provenance only
    common_reports_dir: str                    # POB + heli logs (location agent)
    otl_information_dir: str                    # staff listing + Jobs_Catalogue.xlsx (job_title agent)
    sandbox_read_roots: list[str]              # absolute dirs the code-exec sandbox may read

    # --- per-agent sub-states (single writer each: default last-write-wins) ---
    intake: IntakeState
    category_check: CategoryCheckState
    verify_evidence: VerifyEvidenceState
    orchestrator: OrchestratorState
    jade_analyzer: JadeState
    location: LocationState
    job_title: JobTitleState
    matching: MatchingState
    supervision: SupervisionState
    wiki: WikiState
    agreement_retriever: AgreementRetrievalState
    drafter: DrafterState

    # --- shared data channels (read by the orchestrator; no longer routing gates) ---
    evidence_gate_status: EvidenceGateStatus   # missing_auditor_evidence -> deny; unreadable_auditor_evidence -> needs_evidence + HITL
    # The sources absent from, or needing an accessible copy in, the auditor record. The drafter derives
    # its deterministic evidence-gate wording from this channel or from verify_evidence's POB/heli flags.
    evidence_gap: list[str]
    evidence_gate_decision: dict[str, Any]       # aggregate configured verify-evidence hook decision
    evidence_gate_evaluations: list[dict[str, Any]]
    jade_hook_decision: dict[str, Any]           # aggregate configured jade-analyzer hook decision
    jade_hook_evaluations: list[dict[str, Any]]
    auditor_workbook_manifest: dict[str, Any]    # compact shape/header manifest cached for JADE reuse
    hook_status: Annotated[dict, merge_dict]     # node -> configured hook execution snapshot
    analysis_blocked: bool                       # operational inability to analyze, distinct from a configured ED gate
    auditor_question: list[str]                # the orchestrator derives its dispatch path (and invoke_job_title_after_location) from this
    third_party_invoice_requested: bool        # category_check found a current request for an underlying third-party/vendor invoice
    attestation_requested: bool                # category_check found an attestation-certificate request; drafter handles the wording and HITL follow-up
    additional_support_requested: bool         # category_check found a current unresolved request for additional material; historical IR references do not qualify
    supporting_documentation_provided: bool    # explicit publication hint for the "Credit Denied - supporting documentation provided" response type
    hitl_review: bool                          # human judgement, action, follow-up, or intervention remains required; includes analytical deferrals and operational attestation follow-up
    chargeability_scopes: list[ChargeScope]
    chargeability_hook_evaluations: list[HookEvaluation]
    chargeability_resolution: ChargeabilityResolution
    legal_chargeability_basis: bool | None
    legal_chargeability_basis_by_person: dict[str, bool | None]

    # --- orchestrator (hub-and-spoke supervisor) channels ---
    next_agent: str                            # the orchestrator's per-hop routing decision (read by the edge)
    final_mode: str                            # "terminal" | "needs_evidence" — drives the drafter's mode
    orchestrator_hops: int                     # loop-guard counter; force finalize at MAX_HOPS
    agent_run_counts: dict[str, int]           # per-specialist dispatch counts; force finalize at MAX_AGENT_RUNS
    # orchestrator -> per-specialist custom directive (merge_dict so successive hops accumulate,
    # mirroring nl_summaries_by_node; the target specialist reads its own key each run).
    agent_instructions: Annotated[dict, merge_dict]
    # specialist -> {"status": objective_met|inconclusive, "reason": str} returned to the hub. REQUIRES
    # merge_dict: specialists write in different super-steps, so last-write-wins would clobber earlier
    # statuses; keying by agent preserves all of them (a re-run right-wins its own key).
    agent_status: Annotated[dict, merge_dict]

    # reviewer-flagged gaps forwarded to the specialists as exploration hints — NEVER a gate
    open_questions: list[str]

    # --- HITL fields (assembled for the drafter's deferral note) ---
    hitl_stage: str                            # which agent deferred
    hitl_reason: str                           # why it stopped
    hitl_evidence_trail: dict[str, Any]        # findings-so-far snapshot attached to the note

    # --- trace channels (reducers) ---
    messages: Annotated[list, add_messages]           # permanent, UI-readable message log
    trace: Annotated[list[dict], operator.add]        # append-only structured events
    nl_summaries_by_node: Annotated[dict, merge_dict]  # agent -> bounded narrative summary


# Ordered list of the sub-state keys (used by the output writer to serialize every stage in order).
SUBSTATE_KEYS: list[str] = [
    "intake", "category_check", "verify_evidence", "orchestrator",
    "jade_analyzer", "location", "job_title", "matching", "supervision", "wiki",
    "agreement_retriever", "drafter",
]

__all__ = [
    "merge_dict", "OTLAuditState", "SUBSTATE_KEYS",
    "IntakeState", "CategoryCheckState", "VerifyEvidenceState",
    "OrchestratorState", "JadeState", "LocationState", "JobTitleState", "MatchingState",
    "SupervisionState", "WikiState", "AgreementChunk", "AgreementRetrievalState", "DrafterState",
    "SourceRowRef", "LedgerRow", "McohMonthlyFact", "HookFact", "ChargeScope",
    "HookEvaluation", "ChargeabilityResolution",
]
