"""Orchestrate the LLM passes that explain what the auditor's exhibits demonstrate.

Three passes, all through the ICA gateway via llm.structured_call:
  1. interpret_tab        — per worksheet tab: what it shows + which figures the auditor used
  2. interpret_attachment — roll the tabs up into one per-file summary
  3. synthesize           — cross-attachment: how the exhibits underpin the amount & reasoning

Everything is grounded in the extracted tab data; the prompts forbid inventing values.
"""
from __future__ import annotations

import json

from shared import tab_schemas
from extractor.blocks import primary, render_blocks
from extractor.column_profile import render_profile
from extractor.llm_adapter import structured_call, text_call
from extractor.skill_engine import render_analysis as _render_skill_analysis
from shared.auditor_attachment_citations import (
    SYSTEM as ATTACHMENT_CITATIONS_SYSTEM, user_prompt as attachment_citations_prompt,
)
from extractor.schemas import (
    AttachmentInterpretation, AttachmentSummary, AuditorCitedAttachmentNames, ExhibitEvidenceReport, InvoiceEvidence,
    InvoiceExtraction, InvoiceInterpretation, InvoiceParsedFields, RebuttalDialogueEntry,
    SynthesisResult, TabInterpretation, WbsFocus,
)

# Rows to show the model per tab. Small "money" tabs (pivots/summaries) are fully visible;
# large ledger dumps are sampled — enough to convey structure without flooding the context.
_PROMPT_ROWS_SMALL = 45
_PROMPT_ROWS_LARGE = 15
_SMALL_TAB_ROWS = 80
_MAX_PROMPT_CELLS = 1200  # max cells (rows × cols) rendered to avoid context overflow
_MAX_TABLE_COLS = 24      # wide ledgers: how many columns the sampled table may show
_TYPED_KINDS = ("period", "date", "number")   # the kinds an exception is argued over
_ID_MAX_DISTINCT = 12     # a text column with at most this many distinct values labels its rows
# ...and no more than this share of the rows it covers. "Few" has to be relative as well as absolute:
# eight distinct values over eight rows distinguishes every row, which is an identifier or free text,
# while eight over five thousand is a label. The absolute cap alone calls both of them labels.
_ID_MAX_SHARE = 0.5
# Per-tab profile budget in the synthesis prompt, which may carry many tabs at once. Tighter than the
# interpreter's, and safe to tighten: period, date and number columns are never withheld by the budget,
# so this only trims low-signal text columns.
_SYNTH_PROFILE_CHARS = 1500

_GROUNDING = (
    "Rules: cite ONLY values/labels that literally appear in the data shown. Never invent or "
    "estimate a number. If a tab is unrelated to this exception, say so plainly and leave "
    "auditor_figures_used empty. "
    "Do NOT guess the meaning of an abbreviation, code, acronym, or shorthand you do not recognize: "
    "if its meaning is not established by the data or surrounding context, treat it as unknown, carry "
    "it verbatim, and say its meaning is unclear rather than inventing an expansion."
)

# --- WBS conditional knowledge block ---

_WBS_KNOWLEDGE = """

WBS CODE INTERPRETATION CONTEXT:
You always have access to the following WBS knowledge. Apply it when the data contains WBS
codes, allocation structures, settlement objects, or cost-object-level questions. Do NOT apply
WBS matching when the exception is solely about whether a job title or role is administrative
in nature / overhead — in that case the question is about role classification, not project
structure, and WBS codes are incidental identifiers rather than the analytical focus.

How to read a WBS code:
  First character — project family:
    C = Capital project | P = Expense project | O = Operating expense/project
  Next segment (usually US): geography/company/Blueprint structure (carried as prefix, not the
    primary accounting decision).
  Embedded field token (positions 4-6 after compaction) — asset/field:
    PER = Perdido Host | GWH = Great White | SIL = Silvertip | TOB = Tobago
  Project type token (positions 7-8 after compaction) — THE KEY CLASSIFIER:
    EW = multi-project/asset allocation expense (accumulator/transfer WBS; should settle to
         receiver WBS; residual value is a settlement-control issue, NOT automatically a direct
         recovery claim)
    EA = reimbursable accumulator expense (settlement required, typically to G/L)
    ER = retirement/abandonment expense (settlement required, typically to G/L)
    TP = third-party opex/reimbursable billing (settlement required, relevant to billing)
    DC = decommissioning (settlement-required expense-side)
    DD = production drilling capital (real capital well work, capital-project rules)
    DM = development accumulator (capital accumulator concept)
    DU = development subsea (manifolds, flowlines, umbilicals, subsea trees)
    EU = allocation-style object; treated as allocation context, NOT a final billability object
    FI, PD = other project types validated against the project-type master
  Later segments (year/sequence/lower-level element/activity) vary by project; do not assume
  universal meaning — check WBS master data for name, level, recovery indicator, responsible
  cost center, status, and AA element flag.

Settlement rules:
  - C (capital) WBS should settle to asset/AUC.
  - EW should settle to receiver WBS; trace via PPON where it went.
  - EA, ER, TP, DC generally should settle to G/L.
  - Some P/expense and O/opex project types may retain balance normally.
  - If WBS should settle but still holds value → flag as settlement/control context.
  - If WBS can retain balance → remaining balance is NOT automatically an exception.

Billability determination — WBS alone does NOT decide billability. Ask:
  1. Is this WBS a final landed economic object or an upstream transfer/accumulator?
  2. Should this WBS settle somewhere else?
  3. Did the cost land on field/project economics or remain in host/allocation mechanics?
  4. Does the cost object AND cost element together support billability?
  5. Is this original spend, settlement, cutback, allocation, accrual, reversal, or context?
"""

SYSTEM_TAB = (
    "You are a JV audit analyst tracing the evidentiary basis of an operator-cost exception "
    "raised by the auditor (Chevron). Given the auditor's exception description and ONE "
    "worksheet tab from the auditor's own Excel exhibit, explain what the tab demonstrates in "
    "relation to the description and pinpoint the specific figures/columns the auditor likely "
    "used to reach the reasoning and the questioned amount. "
    "If the tab ties specific INVOICE NUMBERS to dollar amounts (e.g. a reconciliation row 'PO ..., "
    "line 110, $713,801.27, invoice # 92535509'), you MUST record EACH such row as its own "
    "auditor_figures_used entry — value = the amount, location = the invoice # plus PO/line and any "
    "line-item text. These invoice-number-to-amount rows are the primary bridge the downstream "
    "invoice reconciliation depends on; never omit or collapse them. Do the same for ledger "
    "line-item texts and PO-line-to-amount rows. "
    + _GROUNDING
)

SYSTEM_FILE = (
    "You are a JV audit analyst. Given the exception description and per-tab interpretations of "
    "one Excel exhibit, state the file's role and summarize what the file as a whole "
    "demonstrates for the exception. " + _GROUNDING
)


def extract_auditor_cited_attachment_names(description: str, tier: str = "light") -> list[str]:
    """Return filenames the auditor explicitly cites, from the description alone.

    This intentionally receives neither discovered filenames nor parsed exhibit content.  The shared
    evidence-discovery layer later resolves the returned names to safe, exact basename matches.
    """
    result = structured_call(
        tier, ATTACHMENT_CITATIONS_SYSTEM, attachment_citations_prompt(description),
        AuditorCitedAttachmentNames,
    )
    return list(result.attachment_names or [])

SYSTEM_SYNTH = (
    "You are a senior JV audit analyst. Given the exception description, the questioned amount, "
    "and interpretations of every piece of the auditor's evidence, synthesize what the evidence "
    "collectively demonstrates: how the auditor most plausibly derived the questioned amount and "
    "how it supports the stated reasoning (e.g. an 'administrative in nature' claim). Also list "
    "evidence gaps a reviewer should check. "
    "If NO evidence is provided (no exhibits and no invoices), do NOT speculate: state plainly that "
    "the auditor supplied no supporting evidence for this exception and that there is insufficient "
    "information to assess the validity of the claim; leave likely_basis_for_amount empty and record "
    "the missing evidence in open_questions. "
    "If PRIOR REBUTTAL HISTORY is supplied, this is a later cycle of the same exception: read that "
    "history first and make the synthesis continuous with it — note how the auditor's position and "
    "the questioned amount have evolved across cycles, what was previously conceded/withdrawn/upheld, "
    "and whether this cycle's evidence resolves or extends the earlier open threads. "
    "If PRIOR-ROUND REBUTTAL DIALOGUE is supplied, actively consider it — it is the operator's "
    "responses and the auditor's replies from earlier rebuttal rounds (as recorded in the audit "
    "report); weigh how the operator disputed or explained the questioned amount, note where this "
    "round's evidence answers or ignores those points, and reflect that in your synthesis (the "
    "questioned amount and evidence remain the auditor's). "
    "When the auditor's evidence attributes amounts to specific invoice numbers, reconcile by invoice "
    "number: tie each referenced invoice # to its amount, confirm the corresponding invoice supports "
    "it, then sum the per-invoice amounts for the same ledger/PO line and compare the total to that "
    "ledger line and to the questioned amount (e.g. an 'XT5 Material' line billed across three "
    "invoices summing to one ledger figure). State what is and is not accounted for — but only from "
    "figures that actually appear in the evidence; do not compute a reconciliation the figures don't "
    "support. "
    "Write the `synthesis` as an expert JV auditor's narrative memo: refer to the evidence in domain "
    "terms — the auditor's workpapers, the JADE ledger extract, the CJI3 cost detail, the materials "
    "reconciliation, the bill of materials, the iWells installation record, supporting invoices — and "
    "NEVER use spreadsheet-mechanics words ('tab', 'sheet', 'Excel', 'workbook', 'cell', "
    "'spreadsheet') in the synthesis. " + _GROUNDING
)

SYSTEM_NON_WORKBOOK_SYNTH = (
    "You are a senior JV audit analyst. Workbook parsing was disabled for this run. You are given the "
    "auditor's exception description, the questioned amount, and the available non-workbook evidence, "
    "which may include prior rebuttal history, operator responses, auditor replies, invoices, and other "
    "extracted documents. Synthesize what that evidence demonstrates about the auditor's questioning. "
    "Do NOT state that the auditor failed to provide workbooks; state only that workbook contents were "
    "not analyzed in this run. Do NOT claim a complete reconciliation of the questioned amount. Explain "
    "how documented amounts could contribute or tie to the questioned amount, clearly distinguishing an "
    "explicit tie from a plausible but unverified one. Surface every relevant identifier or unusual fact "
    "that may explain the auditor's questioning, including invoice/PO/document/WBS/account codes, vendors, "
    "named employees, contractors, affiliates, roles, dates, quantities, rates, and amounts. Flag those "
    "facts for reviewer follow-up. If PRIOR REBUTTAL HISTORY is supplied, make the synthesis continuous "
    "with it and explain how the position, amount, and unresolved issues evolved. If PRIOR-ROUND REBUTTAL "
    "DIALOGUE is supplied, consider what each side argued and whether the available documents answer those "
    "points. Put any missing workbook-to-final-amount bridge in open_questions rather than inventing it. "
    "Write the synthesis as an expert JV auditor's narrative memo using domain terms, not spreadsheet-"
    "mechanics language. " + _GROUNDING
)

SYSTEM_INVOICE_EXTRACT = (
    "You are a meticulous data-entry clerk transcribing ONE vendor billing document into a structured "
    "record. The document is NOT necessarily a single classic invoice: it may bundle several billing "
    "views and supporting outputs — e.g. a leading COVERSHEET that breaks the charge out by "
    "network/activity code with several amount columns (year-to-date, project-to-date, contract "
    "value, amount-left, amount-for-this-invoice), a formal SAP 'Invoice' page, milestone-completion / "
    "milestone-summary tables, proof-of-completion or SAP screenshots (MRP / PO / stock lists), and "
    "contract-value or amount-remaining figures. "
    "Transcribe EVERY value-bearing line across ALL of these views and pages — do NOT limit yourself "
    "to the page literally titled 'Invoice'. For each line capture: description; reference (its PO "
    "line number / network+activity code / item no / document number — the identifier that ties it "
    "back to the ledger); quantity; unit; unit price; the primary amount; any additional labeled "
    "amount columns in `amounts`; and `section` (which view/page it came from, e.g. 'coversheet', "
    "'invoice', 'milestone certificate'). Also copy the header fields (invoice number, vendor, "
    "bill-to, dates, PO/contract, currency) and the subtotal / tax / total; put any other labeled "
    "values (payment terms, references, well/AFE, service period) into other_fields. "
    "Ignore pure legal boilerplate (standard terms & conditions) — it carries no reconcilable value. "
    "Transcribe verbatim — never invent, infer, or compute a value; if something is absent or "
    "illegible leave it null. Keep money exactly as printed (currency symbols, commas). " + _GROUNDING
)

SYSTEM_INVOICE = (
    "You are a senior JV audit analyst. You are given the auditor's exception description, the "
    "interpretations of the auditor's Excel evidence (JADE ledger / CJI3 cost detail / reconciliation "
    "workpapers), and the FULL transcription of ONE vendor billing document. A vendor invoice exists "
    "to SUPPORT and RECONCILE the auditor's ledger line items: the invoice's charges should tie back "
    "to specific JADE/CJI3 lines. "
    "Explain what the document demonstrates for the exception and RECONCILE it against the auditor's "
    "evidence. "
    "ANCHOR ON THE INVOICE NUMBER FIRST: the auditor's Excel evidence usually references specific "
    "invoice numbers (e.g. a reconciliation row 'PO 4513465745, line 110, $713,801.27, invoice # "
    "92535509'). Check whether THIS invoice's number appears in the auditor's evidence; if it does, "
    "reconcile this invoice's dollar values (and any other relevant figures — quantities, rates, PO "
    "lines) against the amount the auditor attributes to that invoice number, and state in "
    "`ties_to_exhibit` whether the invoice's contents support and match that referenced value. "
    "FALLBACK — only when no invoice number ties this document to the evidence: match invoice lines "
    "to the ledger lines BY MEANING (descriptions drift between systems — the invoice's '2.0 10K EVDT "
    "XT5 Material (F)' is the ledger's '10K 2.0 XT 5 Material (F)') and confirm the tie using the PO "
    "number, the PO line number, and the amount. "
    "Either way, in `ties_to_exhibit` state which ledger figures/lines each invoice value supports, "
    "matches, or contradicts, and whether the amounts reconcile; draw on ANY transcribed figure that "
    "reconciles (a coversheet activity-code amount, a milestone or contract value, a SAP-output "
    "number) — not just a formal invoice total. Pinpoint the specific figures the auditor likely "
    "relied on in `figures_used`. "
    "CRITICAL — do not force a match: if you cannot confidently identify the corresponding ledger "
    "line, or the values do not reconcile, say so plainly and leave `ties_to_exhibit` empty and "
    "`figures_used` empty. Never fabricate a match, a line reference, or a figure to make a "
    "reconciliation appear to work. " + _GROUNDING
)

SYSTEM_INVOICE_NO_EXCEL = (
    "You are a senior JV audit analyst reviewing ONE fully transcribed billing document against the "
    "auditor's exception description. Workbook parsing was disabled, so no ledger or reconciliation "
    "workpaper was analyzed. Explain what the document itself demonstrates and how its printed amounts "
    "could contribute or relate to the questioned amount, but do NOT claim that it reconciles to a "
    "ledger line or to the final questioned amount without an explicit bridge in the description or "
    "rebuttal dialogue. Identify and flag every potentially relevant vendor, employee, contractor, "
    "affiliate, invoice/PO/document/WBS/account code, role, date, quantity, rate, line description, and "
    "amount. Preserve uncertainty and leave ties_to_exhibit empty because workbook evidence was not "
    "examined. " + _GROUNDING
)

SYSTEM_REBUTTAL_SUMMARY = (
    "You are a senior JV audit analyst maintaining a running summary of an exception's rebuttal "
    "history across successive audit cycles. Given the prior running summary (may be empty) and the "
    "syntheses of the most recent cycles, produce ONE updated running summary that preserves the "
    "through-line: how the auditor's position and the questioned amount evolved cycle over cycle, "
    "what each side argued, what was conceded/withdrawn/upheld, and any unresolved threads a "
    "reviewer still needs. Be compact but keep concrete figures, dispositions, and cycle order. "
    "Do not invent anything not present in the inputs."
)


def _select_columns(tab: dict, n_cols: int, cap: int = _MAX_TABLE_COLS) -> list[int]:
    """Which columns the sampled table shows, as 0-based indices in sheet order.

    This replaces keeping the first N columns. On Sheet1 of the GreatWhite_2023_0003 exhibit that cut
    dropped `Document Date` (c43), `Entry Date SL` (c44), `Posting date` (c46) and `Cost + OH` (c66) —
    the three date columns the exception's window is argued over and the money column its amount comes
    from. Column position carries no meaning in a ledger export, so a positional cut is a coin flip on
    exactly the evidence that matters, and it lost.

    The ranking is the column profile's own shape signals, so it needs no schema and no extra model
    call. Periods, dates, numbers and any column carrying a stated total come first, because those are
    what an exception disputes. Then text columns that vary across rows and so tell one row from
    another — a cost code, a WBS description.

    Then free text, where nearly every row differs: it takes as much width as a date column and settles
    little, but it does at least tell one row from another, and a description sometimes names the
    affiliate the exception is about.

    Dropped first of all is a text column holding a single distinct value. It repeats itself on every
    row, so it separates nothing, and the profile already states the one value it holds — strictly less
    than free text carries. Typed columns are never demoted for being constant, because a rate or a total
    that happens not to vary is still part of the arithmetic.

    Falls back to the first `cap` columns when there is no profile, which is the old behaviour and the
    only thing left to do without shape information.
    """
    profs = tab.get("column_profile") or []
    if not profs:
        return list(range(min(n_cols, cap)))

    def rank(p: dict) -> int:
        if p.get("kind") in _TYPED_KINDS or p.get("stated_total") is not None:
            return 0
        distinct = p.get("distinct")
        if distinct is None or distinct <= 1:
            return 3                          # says the same thing on every row: shown last
        room = max(2, int((p.get("of_rows") or 0) * _ID_MAX_SHARE))
        return 1 if distinct <= min(_ID_MAX_DISTINCT, room) else 2

    ranked = sorted((p for p in profs if p.get("index", 0) < n_cols),
                    key=lambda p: (rank(p), p.get("index", 0)))
    return sorted(p["index"] for p in ranked[:cap])


def _render_tab(tab: dict, max_rows: int) -> str:
    """Build a compact markdown view of a tab for the prompt.

    Rows are read POSITIONALLY where the tab supplies them that way. Header-keyed records collide on
    duplicate labels — JADE's Key/Text pairs — and mis-key a sub-table stacked at a different offset,
    and the selection below is by column index, so reading by index is the only faithful way to honour
    it. Each column is labelled with its `c<n>` index too, so a figure quoted from this table can be
    cited the same way as one from the raw grid.
    """
    if tab.get("error"):
        return f"(could not read tab: {tab['error']})"
    header = tab.get("header") or tab.get("columns") or []
    rows = tab.get("rows")
    records = tab.get("records") or []
    n_cols = max(len(header), tab.get("populated_cols") or 0)
    sel = _select_columns(tab, n_cols)

    # Reduce rows if cells would exceed budget. Selecting fewer, better columns buys rows back: at 72
    # columns the budget allowed 16 rows, and the same budget over 24 columns allows 50.
    effective_rows = min(max_rows, max(3, _MAX_PROMPT_CELLS // max(len(sel), 1)))
    labels = [f"c{i + 1} {header[i]}" if i < len(header) and header[i] else f"c{i + 1}" for i in sel]
    lines = ["| " + " | ".join(labels) + " |",
             "| " + " | ".join("---" for _ in sel) + " |"]
    if rows is not None:
        body = rows[:effective_rows]
        n_scanned = len(rows)
        for row in body:
            lines.append("| " + " | ".join(str(row[i]) if i < len(row) else "" for i in sel) + " |")
    else:
        body = records[:effective_rows]
        n_scanned = len(records)
        keys = [header[i] if i < len(header) else "" for i in sel]
        for rec in body:
            lines.append("| " + " | ".join(str(rec.get(k, "")) for k in keys) + " |")

    notes = []
    if n_scanned > effective_rows:
        notes.append(f"+{n_scanned - effective_rows} more scanned rows")
    # Count withheld columns against the POPULATED width, not the declared one. A sheet's declared
    # width survives deleting its content, so this note used to read "+16338 more columns" on a tab
    # that really had 32 beyond the cap — which tells the model the exhibit is unreadably wide when it
    # is not. Fall back to the declared width only when populated_cols is absent.
    full_cols = tab.get("populated_cols") or tab.get("reported_cols") or len(header)
    if len(sel) < full_cols:
        # Name what was left out rather than only counting it, and say where it can still be read.
        # A bare count invites the model to assume the missing columns were the unimportant ones.
        held = [f"c{i + 1} {header[i]}" for i in range(min(full_cols, len(header)))
                if i not in set(sel) and header[i]]
        notes.append(f"{full_cols - len(sel)} columns not shown here"
                     + (f" ({', '.join(held[:12])}{' …' if len(held) > 12 else ''})" if held else "")
                     + " — every column's kind, range and total is in the COLUMN PROFILE above")
    return "\n".join(lines) + ("\n…(" + "; ".join(notes) + ")" if notes else "")


def interpret_tab(description: str, file_name: str, tab: dict, tier: str = "med",
                  ) -> TabInterpretation:
    # Lead with the populated width. The declared width is shown only when it disagrees, and labelled,
    # so "16378 cols" can't be read as 16,378 columns of evidence.
    pop = tab.get("populated_cols") or 0
    dec = tab.get("reported_cols") or 0
    cols = f"{pop} populated cols" if pop else f"{dec} cols"
    if pop and dec > pop:
        cols += f" ({dec} declared by the sheet, the rest hold no values)"
    reported = f"{tab.get('reported_rows')} rows x {cols}"
    if tab.get("truncated"):
        reported += " (raw ledger — sampled)"
    row_cap = _PROMPT_ROWS_SMALL if (tab.get("reported_rows") or 0) <= _SMALL_TAB_ROWS else _PROMPT_ROWS_LARGE
    system = SYSTEM_TAB + _WBS_KNOWLEDGE

    # If this tab was recognized as a known auditor data type, hand the model the standard schema so
    # it locates the exception-relevant rows/columns instead of re-deriving the dense SAP layout.
    detected = tab.get("detected_type")
    schema = tab_schemas.schema_description(detected)
    if schema:
        system += (
            f"\n\nRECOGNIZED DATA TYPE — {detected}. This tab is a known export with the standard "
            f"structure below; use it to find the elements relevant to THIS exception rather than "
            f"re-deriving the layout:\n{schema}"
        )

    # A raw positional dump of the cells. The header-keyed table above is clean for the primary
    # table but hides stacked/offset sub-tables (a support/reconciliation block at a different column
    # offset, invoice#↔amount rows) — those show up faithfully here, read by true column index.
    raw_grid = tab.get("raw_grid") or ""
    raw_section = (
        "\n\nRAW POSITIONAL CELLS (each cell shown as c<column-index>=<value>; use this to read any "
        "stacked or offset sub-table — e.g. an invoice#↔amount reconciliation block — that does not "
        "align with the header table above):\n" + raw_grid
    ) if raw_grid.strip() else ""

    # Ground truth, placed BEFORE the sampled table. `_render_tab` shows a few dozen rows out of
    # possibly thousands and a subset of the columns, so any range or total read off it is a guess about
    # the rows and columns it left out. The profile states those facts over every scanned row, and
    # covers every column, including the ones the table has no room for.
    prof = render_profile(tab.get("column_profile") or [], tab.get("total_rows") or [])
    profile_section = f"\n\n{prof}" if prof else ""

    # Placed before the profile, because it says what the profile is about. On a tab holding several
    # stacked tables the profile describes one of them, and the model has to know which before it reads
    # a single figure. Empty for a single-table tab, so nothing changes for nearly every exhibit.
    tab_blocks = tab.get("blocks") or []
    layout = render_blocks(tab_blocks, primary(tab_blocks))
    layout_section = f"\n\n{layout}" if layout else ""

    user = (
        f"AUDITOR EXCEPTION DESCRIPTION:\n{description}\n\n"
        f"EXHIBIT FILE: {file_name}\n"
        f"TAB: {tab.get('name')}  ({reported})"
        + (f"  [recognized as {detected}]" if detected else "") + "\n"
        f"COLUMNS: {tab.get('columns')}"
        + layout_section + profile_section + "\n\n"
        f"TAB DATA — an ILLUSTRATIVE SAMPLE of rows, not the whole tab. Use it to see what a row looks "
        f"like and to quote individual rows. Do NOT read a range, an earliest or latest date, a count "
        f"or a total off it; those come from the COLUMN PROFILE above:\n{_render_tab(tab, row_cap)}"
        + raw_section + "\n\n"
        "Interpret this tab per the schema."
    )
    return structured_call(tier, system, user, TabInterpretation)


def interpret_attachment(description: str, file_name: str, sheets: list[dict],
                         tier: str = "med",
                         on_tab=None,
                         target: AttachmentInterpretation | None = None) -> AttachmentInterpretation:
    result = target or AttachmentInterpretation(
        file_name=file_name, role="", overall_summary="", tabs=[]
    )
    tabs = result.tabs
    for sheet in sheets:
        ti = interpret_tab(description, file_name, sheet, tier=tier)
        tabs.append(ti)
        if on_tab:
            on_tab(sheet.get("name"), ti)

    digest = "\n".join(
        f"- {t.tab_name}: {t.demonstrates} [ties_to_amount={t.ties_to_amount}; "
        f"figures={[f.value for f in t.auditor_figures_used]}]"
        for t in tabs
    )
    system = SYSTEM_FILE + _WBS_KNOWLEDGE
    user = (
        f"AUDITOR EXCEPTION DESCRIPTION:\n{description}\n\n"
        f"EXHIBIT FILE: {file_name}\n"
        f"PER-TAB INTERPRETATIONS:\n{digest}\n\n"
        "Give the file's role and an overall summary per the schema."
    )
    summ = structured_call(tier, system, user, AttachmentSummary)
    result.role = summ.role
    result.overall_summary = summ.overall_summary
    return result


def extract_invoice_fields(invoice: dict, tier: str = "light") -> InvoiceExtraction:
    """Pass 1 (light): faithfully transcribe the whole invoice into structured fields + line items.

    Pure transcription — no exhibit context, no reasoning — so it runs on the cheap light tier.
    Returns an empty extraction when the invoice yielded no text (scanned image, OCR unavailable).
    """
    body = invoice.get("text") or ""
    if not body.strip():
        return InvoiceExtraction()
    method = invoice.get("extraction_method", "text")
    user = (
        f"INVOICE FILE: {invoice.get('file_name')}  "
        f"({invoice.get('n_pages')} pages; extraction={method})\n\n"
        f"INVOICE TEXT:\n{body}\n\n"
        "Transcribe this invoice completely per the schema."
    )
    return structured_call(tier, SYSTEM_INVOICE_EXTRACT, user, InvoiceExtraction)


SYSTEM_WBS = (
    "You identify which WBS / cost-object code(s) an auditor's exception concerns, and capture the "
    "specific facts the exception disputes. You are given the exception description, (optionally) the "
    "active domain skill, and a list of the candidate WBS codes that actually appear in the evidence, "
    "each with its human-readable description (its project/well name). A WBS code encodes the "
    "asset/well and project type (see the WBS interpretation context), so match the well / asset / "
    "AFE / project / item named in the exception to the candidate descriptions, and also select any "
    "code stated verbatim in the exception. "
    "Choose ONLY from the candidate list — never invent a code. If several codes plausibly belong to "
    "the disputed well/project, return all of them. If you genuinely cannot tell which apply, set "
    "unsure=true and leave wbs_codes empty (the pipeline will then analyze the whole ledger). "
    "Also fill `focus` with the facts the auditor's exception SPECIFICALLY questions — the disputed "
    "item/material, quantity or footage/length discrepancy, labor category, affiliate/entity, person, "
    "period, or whatever the description actually disputes — copied verbatim where possible. Do NOT "
    "add arbitrary observations that the description does not raise; leave `focus` empty if it states "
    "nothing specific."
)

_WBS_CAND_CHUNK = 600  # candidate pairs per selection call; larger sets are chunked + unioned


def _skill_hint(skill) -> str:
    """A short skill descriptor for the focus prompt: name + a truncated instruction preview."""
    if skill is None:
        return ""
    preview = (skill.instructions or "").strip()[:800]
    return f"ACTIVE SKILL: {skill.name}\nSKILL CONTEXT (excerpt):\n{preview}\n\n"


def _wbs_user_prompt(description: str, pairs: list[str], skill=None) -> str:
    return (
        f"{_skill_hint(skill)}"
        f"AUDITOR EXCEPTION DESCRIPTION:\n{description}\n\n"
        f"CANDIDATE WBS CODES ({len(pairs)}) — one per line as `code<TAB>description`:\n"
        + "\n".join(pairs) + "\n\n"
        "Select the WBS code(s) this exception concerns and fill `focus` per the schema. "
        "Choose codes only from the list."
    )


def select_wbs_focus(description: str, wbs_map: dict, skill=None, tier: str = "light") -> WbsFocus:
    """Light pass: pick the WBS code(s) of interest from the evidence's candidate {code: description}
    and capture the facts the exception disputes into a skill-agnostic `focus` dict.

    The codes usually are NOT in the description — they live in JADE's Cost Object column — so the
    model chooses from the candidates (matching each code's description to the well/AFE/project) and
    can only return codes that exist. Returns unsure=True (empty codes) when it can't decide; the
    caller then skips narrowing. For an unusually large candidate set the list is chunked and the
    selected codes/focus are unioned.
    """
    system = SYSTEM_WBS + _WBS_KNOWLEDGE
    pairs = [f"{c}\t{(wbs_map.get(c) or '').strip()}" for c in sorted(wbs_map or {})]
    if not pairs:
        # No candidates (unreadable/absent ledger) — still capture the disputed facts from the description.
        user = (f"{_skill_hint(skill)}"
                f"AUDITOR EXCEPTION DESCRIPTION:\n{description}\n\n"
                "No candidate WBS codes were found in the evidence. Fill `focus` with what the "
                "exception specifically disputes; leave wbs_codes empty and set unsure=true.")
        return structured_call(tier, system, user, WbsFocus)

    if len(pairs) <= _WBS_CAND_CHUNK:
        return structured_call(tier, system, _wbs_user_prompt(description, pairs, skill), WbsFocus)

    # Large candidate set: select per chunk, union codes, merge focus (first value per key), unsure only if all.
    codes: list[str] = []
    focus: dict[str, str] = {}
    all_unsure = True
    for start in range(0, len(pairs), _WBS_CAND_CHUNK):
        res = structured_call(tier, system,
                              _wbs_user_prompt(description, pairs[start:start + _WBS_CAND_CHUNK], skill),
                              WbsFocus)
        for c in res.wbs_codes:
            if c not in codes:
                codes.append(c)
        for k, v in (res.focus or {}).items():
            if v and not focus.get(k):
                focus[k] = v
        all_unsure = all_unsure and res.unsure
    return WbsFocus(wbs_codes=codes, focus=focus, unsure=all_unsure and not codes)


def _exhibit_context(attachments: list[AttachmentInterpretation]) -> str:
    """Digest of the Excel interpretations, as anchors for invoice reconciliation.

    The invoice pass reconciles by matching invoice lines to specific ledger lines, so it needs the
    ledger line-item texts, PO lines, and any invoice#↔PO-line↔amount rows — not just bare figure
    values. We surface, per tab, its `demonstrates` narrative plus each auditor figure as
    `value @ location — why` (bounded), which is where those anchors live.
    """
    if not attachments:
        return "(no Excel evidence was provided)"
    lines = []
    for a in attachments:
        lines.append(f"- {a.file_name} ({a.role}): {a.overall_summary}")
        for t in a.tabs:
            figs = t.auditor_figures_used[:20]
            if not figs and t.ties_to_amount == "unclear":
                continue
            lines.append(f"    · {t.tab_name} [{t.ties_to_amount}]: {t.demonstrates}")
            for f in figs:
                why = f" — {f.why_relevant}" if f.why_relevant else ""
                lines.append(f"        - {f.value} @ {f.location}{why}")
    return "\n".join(lines)


def contextualize_invoice(description: str, extraction: InvoiceExtraction,
                          attachments: list[AttachmentInterpretation],
                          tier: str = "med",
                          excel_parsing_enabled: bool = True) -> InvoiceInterpretation:
    """Pass 2 (med): interpret an invoice against workbook evidence when that mode is enabled.

    Med tier by default (per-invoice reasoning; the final cross-evidence tie-together is done by
    `synthesize` on the high tier). With workbook parsing disabled, relate the invoice directly to the
    exception/rebuttal context without claiming ledger reconciliation. Overridable to high via `tier`.
    """
    inv_no = extraction.invoice_number or "(none found)"
    if excel_parsing_enabled:
        system = SYSTEM_INVOICE + _WBS_KNOWLEDGE
        user = (
            f"AUDITOR EXCEPTION DESCRIPTION:\n{description}\n\n"
            f"THIS INVOICE'S NUMBER: {inv_no}  —  first look for this number in the auditor evidence "
            f"below; if present, reconcile against the amount attributed to it, else fall back to "
            f"PO/line/description matching.\n\n"
            f"AUDITOR EXCEL EVIDENCE (interpretations):\n{_exhibit_context(attachments)}\n\n"
            f"FULL INVOICE TRANSCRIPTION (JSON):\n{json.dumps(extraction.model_dump(), indent=2)}\n\n"
            "Interpret and reconcile this invoice per the schema."
        )
    else:
        system = SYSTEM_INVOICE_NO_EXCEL + _WBS_KNOWLEDGE
        user = (
            f"AUDITOR EXCEPTION DESCRIPTION AND REBUTTAL CONTEXT:\n{description}\n\n"
            f"THIS INVOICE'S NUMBER: {inv_no}\n\n"
            "WORKBOOK STATUS: parsing disabled; no ledger/workpaper comparison is available.\n\n"
            f"FULL INVOICE TRANSCRIPTION (JSON):\n{json.dumps(extraction.model_dump(), indent=2)}\n\n"
            "Interpret this document in non-workbook mode per the schema."
        )
    return structured_call(tier, system, user, InvoiceInterpretation)


def summarize_rebuttals(prior_summary: str, recent_syntheses: list[str], tier: str = "med") -> str:
    """Roll the just-completed window of rebuttal syntheses into one running summary.

    Runs on the high tier by default (this summary is the long-range memory every later cycle
    inherits). `prior_summary` may be empty on the first window. Tokens are recorded via text_call.
    """
    window = "\n\n".join(
        f"[cycle synthesis {i + 1}]\n{s}" for i, s in enumerate(recent_syntheses)
    )
    user = (
        f"PRIOR RUNNING SUMMARY (may be empty):\n{prior_summary or '(none yet)'}\n\n"
        f"NEW CYCLE SYNTHESES TO FOLD IN:\n{window}\n\n"
        "Produce the updated running summary."
    )
    return text_call(tier, SYSTEM_REBUTTAL_SUMMARY, user).strip()


def build_invoice_evidence(description: str, invoice: dict,
                           attachments: list[AttachmentInterpretation] = (),
                           extract_tier: str = "light",
                           context_tier: str = "med",
                           excel_parsing_enabled: bool = True,
                           target: InvoiceEvidence | None = None) -> InvoiceEvidence:
    """Two-pass invoice evidence: light transcription → mode-aware med contextualization."""
    result = target or InvoiceEvidence(
        file_name=invoice.get("file_name", ""),
        source_path=invoice.get("source", ""),
        n_pages=invoice.get("n_pages", 0),
        extraction_method=invoice.get("extraction_method", "text"),
        ocr_pages=invoice.get("ocr_pages", 0),
        ocr_available=bool(invoice.get("ocr_available")),
        parsed=InvoiceParsedFields.model_validate(invoice.get("parsed") or {}),
    )
    extraction = extract_invoice_fields(invoice, tier=extract_tier)
    result.extraction = extraction
    interp = contextualize_invoice(
        description, extraction, list(attachments), tier=context_tier,
        excel_parsing_enabled=excel_parsing_enabled,
    )
    result.demonstrates = interp.demonstrates
    result.figures_used = interp.figures_used
    result.ties_to_amount = interp.ties_to_amount
    result.ties_to_exhibit = interp.ties_to_exhibit
    result.notes = interp.notes
    return result


def _render_previous_rebuttals(previous_rebuttals: dict[str, str]) -> str:
    """Render the prior-cycle chain into a synthesis-prompt block.

    Handles both shapes produced by build_previous_rebuttals: a running summary under
    'summarized_rebuttals' plus any verbatim tail, or purely verbatim {rebuttal_id: synthesis}.
    """
    if not previous_rebuttals:
        return ""
    parts = []
    summary = previous_rebuttals.get("summarized_rebuttals")
    if summary:
        parts.append(f"[SUMMARY OF EARLIER CYCLES]\n{summary}")
    for rid, synth in previous_rebuttals.items():
        if rid == "summarized_rebuttals":
            continue
        parts.append(f"[CYCLE {rid} — full synthesis]\n{synth}")
    return (
        "\n\nPRIOR REBUTTAL HISTORY (earlier cycles of THIS exception, oldest first — your "
        "synthesis must take this into account and stay continuous with it):\n"
        + "\n\n".join(parts)
    )


def _render_operator_dialogue(operator_dialogue: str) -> str:
    """Render the exception's prior-round rebuttal dialogue (from the audit report) for the prompt."""
    if not (operator_dialogue or "").strip():
        return ""
    return (
        "\n\nPRIOR-ROUND REBUTTAL DIALOGUE (the operator's responses and the auditor's replies from "
        "earlier rebuttal rounds, as recorded in the audit report — consider how the operator "
        "disputed or explained the questioned amount and whether this round's evidence answers it):\n"
        + operator_dialogue
    )


def _render_latest_auditor_reply(latest_auditor_reply: dict | None) -> str:
    """Render the selected auditor-only reply as priority cycle context."""
    reply = latest_auditor_reply or {}
    text = str(reply.get("text") or "").strip()
    if not text:
        return ""
    cycle = reply.get("rebuttal_cycle")
    date = str(reply.get("date") or "").strip()
    label = f"Rebuttal {cycle}" if cycle is not None else "Auditor rebuttal"
    if date:
        label += f" — {date}"
    source = str(reply.get("source") or "catalog").strip().lower()
    source_note = (
        "cataloged auditor-only context" if source != "audit_report_pdf"
        else "the final Auditor-labeled reply parsed from the cycle PDF"
    )
    return (
        "\n\nLATEST AUDITOR REPLY FOR THIS CYCLE (priority context: " + source_note + "; "
        "prioritize it over any conflicting attribution in the raw PDF dialogue):\n"
        f"[{label}]\n{text}"
    )


def synthesize(description: str, attachments: list[AttachmentInterpretation],
               meta: dict, invoices: list[InvoiceEvidence] = (),
               previous_rebuttals: dict[str, str] | None = None,
               operator_dialogue: str = "",
               rebuttal_dialogue_entries: list[dict[str, str]] | None = None,
               latest_auditor_reply: dict | None = None,
               skill=None, skill_analysis: dict | None = None,
               tab_profiles: dict[str, dict] | None = None,
               excel_parsing_enabled: bool = True,
               tier: str = "high", allow_budget_overrun: bool = False) -> ExhibitEvidenceReport:
    invoices = list(invoices)
    previous_rebuttals = previous_rebuttals or {}
    rebuttal_dialogue_entries = rebuttal_dialogue_entries or []
    tab_profiles = tab_profiles or {}
    blocks = []
    for a in attachments:
        parts = []
        for t in a.tabs:
            parts.append(f"    * {t.tab_name} [{t.ties_to_amount}]: {t.demonstrates} "
                         f"figures={[(f.value, f.location) for f in t.auditor_figures_used]}")
            # The computed profile for this tab, if the caller supplied one. Everything else in this
            # prompt is an LLM's account of the exhibit; this is arithmetic over its rows, and it is
            # what the synthesis must use for any range or total it states.
            pd = tab_profiles.get(f"{a.file_name}::{t.tab_name}")
            if pd:
                pd_blocks = pd.get("blocks") or []
                # The layout first, so the profile below it is read as being about one named table
                # rather than the tab. This is where a wrong scope does the most damage: the synthesis
                # is what writes the figures into the rebuttal.
                lay = render_blocks(pd_blocks, primary(pd_blocks))
                if lay:
                    parts.append("\n".join(f"      {ln}" for ln in lay.splitlines()))
                prof = render_profile(pd.get("column_profile") or [], pd.get("total_rows") or [],
                                      max_chars=_SYNTH_PROFILE_CHARS)
                if prof:
                    parts.append("\n".join(f"      {ln}" for ln in prof.splitlines()))
        blocks.append(f"FILE: {a.file_name} (role: {a.role})\n  {a.overall_summary}\n"
                      + "\n".join(parts))

    invoice_section = ""
    if invoices:
        inv_lines = []
        for iv in invoices:
            figs = [(f.value, f.location) for f in iv.figures_used]
            ext = iv.extraction
            inv_lines.append(
                f"INVOICE: {iv.file_name} [{iv.ties_to_amount}]"
                f"\n  vendor={ext.vendor or iv.parsed.vendor} "
                f"number={ext.invoice_number or iv.parsed.invoice_number} "
                f"total={ext.total_amount or iv.parsed.total_amount} "
                f"date={ext.invoice_date or iv.parsed.invoice_date} "
                f"line_items={len(ext.line_items)}"
                f"\n  {iv.demonstrates}"
                + (f"\n  reconciliation: {iv.ties_to_exhibit}" if iv.ties_to_exhibit else "")
                + f"\n  figures={figs}"
            )
        invoice_section = "\n\nINVOICE EVIDENCE:\n" + "\n\n".join(inv_lines)

    history_section = _render_previous_rebuttals(previous_rebuttals)
    operator_section = _render_operator_dialogue(operator_dialogue)
    catalog_auditor_section = _render_latest_auditor_reply(latest_auditor_reply)
    if blocks:
        exhibit_body = "\n\n".join(blocks)
    elif excel_parsing_enabled:
        exhibit_body = "(none — the auditor provided NO Excel evidence for this exception)"
    else:
        exhibit_body = (
            "(workbook parsing disabled by configuration — workbook attachments may exist, but their "
            "contents were intentionally not analyzed in this run)"
        )

    system = (SYSTEM_SYNTH if excel_parsing_enabled else SYSTEM_NON_WORKBOOK_SYNTH) + _WBS_KNOWLEDGE
    skill_section = ""
    if skill is not None:
        system += (
            f"\n\nACTIVE SKILL — {skill.name}. Apply the following domain playbook to understand the "
            f"evidence and explain how the auditor reached their conclusion. It is an understanding "
            f"aid, not a verdict: do NOT emit a deny/credit/resolved disposition — explain the "
            f"auditor's reasoning and record what's missing in open_questions.\n{skill.instructions}"
        )
        skill_section = _render_skill_analysis(skill_analysis)
    user = (
        f"AUDITOR EXCEPTION DESCRIPTION:\n{description}\n\n"
        f"QUESTIONED AMOUNT: {meta.get('disputed_amount')} {meta.get('currency', 'USD')}\n\n"
        f"AUDITOR EVIDENCE:\n" + exhibit_body + invoice_section + skill_section + history_section
        + operator_section + catalog_auditor_section + "\n\n"
        + ("Where a COLUMN PROFILE appears above, it was computed in code over every row of that tab. "
           "Take every date range, earliest/latest date, period range, row count and column total from "
           "it. Do NOT derive any of those from the narrative figures or the sampled rows — those cover "
           "part of a tab, so a range or total read from them silently describes only that part.\n\n"
           if tab_profiles else "")
        + ("Synthesize per the schema without claiming a complete reconciliation: identify invoice and "
           "report figures that could contribute or tie to the questioned amount; flag relevant codes, "
           "vendors, employees, contractors, affiliates, roles, and other notable facts; then provide "
           "likely_basis_for_amount, synthesis, and open_questions"
           if not excel_parsing_enabled else
           "Synthesize per the schema: likely_basis_for_amount (strongest candidate figures across "
           "all evidence for how the amount was derived), synthesis, and open_questions")
        + (". Use the SKILL ANALYSIS findings above as the primary evidence for the reconciliation, "
           "and fold its coverage gaps into open_questions." if skill is not None else "")
        + "."
    )
    syn = structured_call(
        tier, system, user, SynthesisResult, hook_stage="final_synthesis",
        allow_budget_overrun=allow_budget_overrun,
    )
    return ExhibitEvidenceReport(
        year=meta.get("year", ""),
        asset=meta.get("asset", ""),
        exception_category=meta.get("exception_category", ""),
        case_id=meta["case_id"],
        exception_code=meta["exception_code"],
        title=meta["title"],
        auditor_description=description,
        excel_parsing_enabled=excel_parsing_enabled,
        attachments=attachments,
        invoices=invoices,
        likely_basis_for_amount=syn.likely_basis_for_amount,
        synthesis=syn.synthesis,
        open_questions=syn.open_questions,
        previous_rebuttals=previous_rebuttals,
        rebuttal_dialogue=operator_dialogue,
        rebuttal_dialogue_entries=[RebuttalDialogueEntry(**entry)
                                  for entry in rebuttal_dialogue_entries],
        operator_dialogue_considered=bool((operator_dialogue or "").strip()),
    )
