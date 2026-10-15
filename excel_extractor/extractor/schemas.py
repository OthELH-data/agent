"""Pydantic output schemas for the exhibit-evidence explainer.

These are the shapes the LLM is forced to emit (via llm.structured_call) and the shape of the
final JSON artifact. Field descriptions double as prompt guidance: structured_call embeds the
JSON schema in the system message.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from shared.token_schemas import TokenUsageReport

TieStrength = Literal["direct", "supporting", "unclear"]
# direct     = this data point plausibly *is* (part of) the questioned figure / the stated reason
# supporting = it corroborates the auditor's reasoning but isn't the figure itself
# unclear    = present in the tab but its link to the exception can't be established from the data


class AuditorCitedAttachmentNames(BaseModel):
    """Filenames explicitly named in the current auditor exception description."""

    attachment_names: list[str] = Field(
        default_factory=list,
        description=("Exact attachment filenames explicitly cited in the auditor's exception description, "
                     "including their extension when printed. Empty when none are named."),
    )


class FigureRef(BaseModel):
    """A specific number or data point in an exhibit that the auditor likely relied on."""
    value: str = Field(description="the figure/data as it appears, e.g. '$52,274.74', '1,234.50', "
                                   "'SAP Support Specialist', or a column total")
    location: str = Field(description="where it lives: tab name + column header and/or cell/row hint, "
                                      "e.g. \"tab 'auditor summary', column 'Amount', total row\"")
    why_relevant: str = Field(description="how this figure/data supports the auditor's reasoning "
                                          "or the questioned amount")


class TabInterpretation(BaseModel):
    """What one worksheet tab demonstrates relative to the auditor's description."""
    tab_name: str
    purpose: str = Field(description="what this tab is / contains, in one or two sentences")
    demonstrates: str = Field(description="what this tab demonstrates in relation to the auditor's "
                                          "exception description")
    key_columns: list[str] = Field(default_factory=list,
                                    description="column headers that carry the relevant evidence")
    auditor_figures_used: list[FigureRef] = Field(
        default_factory=list,
        description="specific figures/data in THIS tab the auditor likely used for the reasoning "
                    "or the questioned amount; empty if the tab carries none")
    ties_to_amount: TieStrength = Field(
        description="strength of this tab's link to the questioned amount / stated reason")
    notes: str = Field(default="", description="caveats, uncertainty, or data-quality observations")


class AttachmentInterpretation(BaseModel):
    """What one Excel file demonstrates, aggregated over its tabs."""
    file_name: str
    role: str = Field(description="the file's role, e.g. 'auditor exception workpaper' or "
                                  "'allocation workpaper'")
    overall_summary: str = Field(description="what this file, as a whole, demonstrates for the "
                                             "exception")
    tabs: list[TabInterpretation] = Field(default_factory=list)


class InvoiceParsedFields(BaseModel):
    """Header fields lifted verbatim from an invoice's text by regex (a cheap fallback prior)."""
    invoice_number: str | None = None
    vendor: str | None = None
    invoice_date: str | None = None
    po_number: str | None = None
    currency: str | None = None
    total_amount: float | None = None


class InvoiceLineItem(BaseModel):
    """One line item transcribed verbatim from any billing table (leave a field null/empty if absent).

    Covers both classic invoice lines and the richer 'Coversheet' billing grids that break a charge
    out by network/activity code with several amount columns. Everything is copied as printed.
    """
    description: str = ""
    reference: str | None = Field(
        default=None,
        description="the line's identifier as printed: PO line number / network+activity code / item "
                    "no / document number (e.g. 'line 170', '30332979 0020 / item 170') — the bridge "
                    "to the ledger's PO-line; null if the line has none")
    quantity: str | None = None
    unit: str | None = None
    unit_price: str | None = None
    amount: str | None = Field(default=None, description="the primary/billed line amount as printed "
                                                         "(keep currency/commas)")
    amounts: dict[str, str] = Field(
        default_factory=dict,
        description="additional labeled amount columns when a line carries several, verbatim (e.g. "
                    "{'year_to_date': '...', 'project_to_date': '...', 'contract_value': '...', "
                    "'this_invoice': '...'}); empty for a simple single-amount line")
    section: str = Field(
        default="",
        description="which billing view/page this line came from (e.g. 'coversheet', 'invoice', "
                    "'milestone certificate') so the same charge isn't double-counted")


class InvoiceExtraction(BaseModel):
    """Complete faithful transcription of an invoice (the light-tier extraction pass).

    Everything is copied verbatim from the invoice text/OCR — never invented. Money is kept as
    strings to preserve the printed form (currency symbols, commas, parentheses).
    """
    invoice_number: str | None = None
    vendor: str | None = None
    bill_to: str | None = None
    invoice_date: str | None = None
    po_number: str | None = None
    currency: str | None = None
    line_items: list[InvoiceLineItem] = Field(
        default_factory=list,
        description="every value-bearing line across ALL billing views/tables in the document "
                    "(coversheet breakdown, formal invoice, milestone/contract tables, SAP outputs), "
                    "verbatim — not just the lines on the page titled 'Invoice'")
    subtotal: str | None = None
    tax: str | None = None
    total_amount: str | None = None
    other_fields: dict[str, str] = Field(
        default_factory=dict,
        description="any other labeled values present (terms, dates, references, well/AFE, etc.), verbatim")


class InvoiceInterpretation(BaseModel):
    """LLM structured output: what a vendor invoice demonstrates for the exception (context pass)."""
    demonstrates: str = Field(description="what this invoice demonstrates in relation to the "
                                          "auditor's exception description")
    figures_used: list[FigureRef] = Field(
        default_factory=list,
        description="specific figures/data in THIS invoice the auditor likely relied on (invoice "
                    "total, line-item amounts, rates, dates, PO/period); empty if none apply")
    ties_to_amount: TieStrength = Field(
        description="strength of this invoice's link to the questioned amount / stated reason")
    ties_to_exhibit: str = Field(
        default="",
        description="how this invoice reconciles to the auditor's Excel evidence — which exhibit "
                    "figures/line items it supports, matches, or contradicts (empty if none)")
    notes: str = Field(default="", description="caveats, uncertainty, or data-quality observations")


class InvoiceEvidence(BaseModel):
    """A vendor invoice attached as evidence: extraction facts + full transcription + interpretation."""
    file_name: str
    source_path: str
    n_pages: int = 0
    extraction_method: Literal["text", "ocr", "mixed", "empty"] = "text"
    ocr_pages: int = 0
    ocr_available: bool = False
    parsed: InvoiceParsedFields = Field(default_factory=InvoiceParsedFields)
    extraction: InvoiceExtraction = Field(default_factory=InvoiceExtraction)
    demonstrates: str = ""
    figures_used: list[FigureRef] = Field(default_factory=list)
    ties_to_amount: TieStrength = "unclear"
    ties_to_exhibit: str = ""
    notes: str = ""


class RebuttalDialogueEntry(BaseModel):
    """One dated or undated reply as printed in the auditor's running report."""
    party: Literal["operator", "auditor"] = Field(
        description="who made the reply")
    reply: str = Field(
        description="verbatim reply text after its report header; includes the printed date when present")


class HookExecutionStatus(BaseModel):
    """JSON-ready status for one configured runtime hook."""
    enabled: bool = False
    triggered: bool = False
    intercepts: list[str] = Field(default_factory=list)
    parameters: dict = Field(default_factory=dict)
    values: dict = Field(default_factory=dict)
    stopped_stage: str = ""
    reason: str = ""


class ProcessingStatus(BaseModel):
    """Whether the cycle completed fully or finalized after a controlled hook stop."""
    status: Literal["complete", "partial"] = "complete"
    stopped_stage: str = ""
    stop_reason: str = ""
    hooks: dict[str, HookExecutionStatus] = Field(default_factory=dict)


class ExhibitEvidenceReport(BaseModel):
    """Top-level artifact: what the auditor's evidence demonstrates for the exception.

    Field order is deliberate (Pydantic serializes in declaration order): case header on top →
    auditor description → workbook mode/evidence → invoices → synthesis → rebuttal chain → token/time at
    the very bottom.
    """
    # --- case header ---
    # (the disputed/questioned amount is not carried as a field — it lives in auditor_description)
    year: str = Field(default="", description="audit year, e.g. '2024'")
    asset: str = Field(default="", description="asset/entity, e.g. 'GreatWhite'")
    exception_category: str = Field(
        default="", description="exception category/topic, e.g. 'Material Reconciliation'")
    case_id: str
    exception_code: str
    title: str
    auditor_description: str = Field(description="the auditor's exception description, verbatim")
    # --- Excel evidence ---
    excel_parsing_enabled: bool = Field(
        default=True,
        description="whether workbook contents were parsed for this report; false means attachments "
                    "may exist but were intentionally omitted by configuration")
    attachments: list[AttachmentInterpretation] = Field(default_factory=list)
    evidence_discovery: dict[str, Any] = Field(
        default_factory=dict,
        description="cycle-specific evidence manifest: safe auditor workbooks/PDFs, cycle-side paths, "
                    "and any operator-response workbook exclusions")
    # --- invoices (between the Excel evidence and the synthesis) ---
    invoices: list[InvoiceEvidence] = Field(
        default_factory=list,
        description="vendor invoices found alongside the exhibits, fully transcribed (text/OCR) and "
                    "interpreted against available evidence; reconciled to workbook evidence only when "
                    "Excel parsing is enabled")
    # --- synthesis ---
    likely_basis_for_amount: list[FigureRef] = Field(
        default_factory=list,
        description="the strongest candidate figures/data across all evidence for how the "
                    "questioned amount was derived")
    synthesis: str = Field(default="", description="an expert JV auditor's narrative of what the "
                                                    "evidence collectively demonstrates about the "
                                                    "reasoning and the questioned amount; empty when "
                                                    "no synthesis could be generated")
    open_questions: list[str] = Field(
        default_factory=list,
        description="what the evidence does NOT show / gaps a reviewer should check")
    # --- rebuttal chain ---
    rebuttal_cycle: int = Field(
        default=0, description="which rebuttal cycle this report is for (folder number; 0 = initial)")
    rebuttal_id: str = Field(
        default="", description="stable id for this cycle's report, e.g. '<case_id>__r<cycle>'")
    previous_rebuttals: dict[str, str] = Field(
        default_factory=dict,
        description="prior rebuttal cycles' outcomes. When fewer than 3 priors exist: "
                    "{<prev_rebuttal_id>: <that cycle's synthesis>}. Once 3+ priors exist, the "
                    "oldest are compressed under key 'summarized_rebuttals' (a running summary) and "
                    "only the most recent (count mod 3) remain as verbatim {id: synthesis} entries")
    rebuttal_dialogue: str = Field(
        default="",
        description="verbatim prior-round operator responses and auditor replies for this exception, "
                    "as extracted from the current auditor report; empty when the report contains no "
                    "rebuttal dialogue")
    rebuttal_dialogue_entries: list[RebuttalDialogueEntry] = Field(
        default_factory=list,
        description="the same rebuttal dialogue split into an ordered list of operator and auditor "
                    "replies, for direct review without parsing the verbatim transcript")
    operator_dialogue_considered: bool = Field(
        default=False,
        description="whether the prior-round rebuttal dialogue (operator responses + auditor replies) "
                    "embedded in the audit report was fed into this synthesis")
    processing: ProcessingStatus = Field(
        default_factory=ProcessingStatus,
        description="cycle completion state and the runtime hooks that affected it")
    # --- run metrics (very bottom) ---
    token_usage: TokenUsageReport | None = Field(
        default=None,
        description="tokens in/out for the whole run and each model tier — reported by the provider "
                    "on both backends, tiktoken-estimated only when a reply carried no usage block "
                    "(see its `estimator` field)")
    elapsed_minutes: float | None = Field(
        default=None, description="wall-clock minutes to produce this exception's analysis")


class AttachmentSummary(BaseModel):
    """Per-file rollup emitted by the summarizer step (tabs are attached in code)."""
    role: str
    overall_summary: str


class SynthesisResult(BaseModel):
    """Cross-attachment synthesis emitted by the final step (metadata is attached in code)."""
    likely_basis_for_amount: list[FigureRef] = Field(default_factory=list)
    synthesis: str
    open_questions: list[str] = Field(default_factory=list)


class SkillSelection(BaseModel):
    """The single skill a light routing pass picks for an exception, from the candidate skills whose
    `Applies to families` headers are shown to the model — or the `no_skill` sentinel when none fit."""
    skill_slug: str = Field(
        description="Exactly ONE skill slug copied verbatim from the provided candidate list — the "
                    "skill whose 'Applies to families' best matches the auditor's exception. Return "
                    "the literal 'no_skill' if none of the candidate skills apply.")


class WbsFocus(BaseModel):
    """The WBS code(s) of interest for an exception, selected from the candidate codes present in the
    evidence — used to pre-filter a large ledger before the skill scan — plus a free-form dict of the
    facts the auditor's exception specifically questions. Skill-agnostic."""
    wbs_codes: list[str] = Field(
        default_factory=list,
        description="WBS/cost-object code(s) this exception concerns, chosen ONLY from the provided "
                    "candidate list. Leave empty if unsure.")
    focus: dict[str, str] = Field(
        default_factory=dict,
        description="Key->value facts drawn SPECIFICALLY from what the auditor's exception description "
                    "questions — not arbitrary observations. Keys are short labels; values are "
                    "verbatim from the description where possible. E.g. for a material exception "
                    "{'material': '4.5\" 26# tubing', 'length_discrepancy': '234 jts short'}; for a "
                    "labor exception {'labor_category': 'offsite tech', 'affiliate': 'SIEP'}. Leave "
                    "empty if the description states nothing specific.")
    unsure: bool = Field(
        default=False,
        description="True if you cannot confidently pick the WBS code(s) from the candidates; the "
                    "pipeline then analyzes the full ledger instead of narrowing it.")
