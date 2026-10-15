"""Recognize the recurring auditor data types and hand the LLM a ready-made schema for each.

Auditor exhibits are hand-compiled, so the same underlying data (a JADE ledger export, a CJI3 SAP
cost dump, a bill of materials, a materials reconciliation, an iWells installation extract) shows
up under wildly inconsistent tab names, with the header row not always at the top, and sometimes
two data types stacked in one tab. Tab names are therefore useless for identification — but each
type keeps a stable COLUMN SIGNATURE.

`detect_tab_type` scans the first rows of a sheet for those signatures and returns the type plus
the row that is its column header. `schema_description` returns a pre-written explanation of that
type (structure, key columns, where WBS codes live, what an auditor cites) so the interpretation
pass is told what the tab is and what to look for — instead of re-deriving the layout every time.
This is a processing-time / token optimization: the model stops reverse-engineering these dense
SAP exports and goes straight to the auditor-relevant rows.

Detection is deterministic; nothing here calls an LLM.
"""
from __future__ import annotations

import re

# How many leading rows to sniff for a header signature (title/metadata blocks push real headers
# down — a Tubular Reconciliation table header can sit around row 12-13).
SNIFF_ROWS = 15

KNOWN_TYPES = ("JADE", "CJI3", "iWells", "BoM", "Recon")


def _norm(value) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


# Signature groups, most-reliable first. Each: (type, tokens, min_hits, is_header).
#   * a token "matches" a cell when it equals the cell or is a substring of it (tolerates the
#     Gen-1 " - Key" suffixes and minor label drift);
#   * min_hits tokens must appear in ONE row for the group to fire;
#   * is_header=True means that row is the column header (records start below it); is_header=False
#     means we matched a title/metadata/summary row (type is known, but let the caller find the
#     header the default way).
# On ties the earlier group wins, so the crisp full-width signatures (iWells, CJI3, JADE) outrank
# the looser title/summary groups.
_GROUPS: list[tuple[str, list[str], int, bool]] = [
    ("iWells", ["network", "well common name", "event", "assembly", "component name",
                "# jts", "length (ft)"], 4, True),
    ("CJI3", ["wbs element", "cost element", "cost element name", "value in obj. crcy",
              "ref. document number", "document type", "material description"], 3, True),
    ("CJI3", ["object", "cost element", "cost element name", "value in obj. crcy",
              "wbs element"], 4, True),
    ("JADE", ["joint venture(jv)", "jv:joa", "equity group", "cost object wbs/cc (co)",
              "postngobj wbs/cc/or (pso)", "jib/jibe class", "line item text orig"], 3, True),
    ("BoM", ["shell common description", "sap short description", "qty needed", "shell mmr",
             "vendor part number", "total cost"], 3, True),
    ("Recon", ["jib date", "voucher ref", "amount charged", "average price per foot",
               "footage", "material no./description"], 2, True),
    ("Recon", ["bom/cn quantity", "sap li detail quantity", "installed qty per iwells",
               "quantity out of tolerance", "mdm diff. (chrgd - set)"], 2, True),
    ("JADE", ["allocation %", "exc amt", "total exc claim", "oh exc amt", "cvx net claim"], 2, True),
    ("Recon", ["tubular reconciliation", "materials reconciliation", "equipment reconciliation",
               "reconciliation summary", "materials recon", "tubular recon"], 1, False),
    ("Recon", ["account description", "difference", "grand total"], 2, False),
]


def _hits(tokens: list[str], cells: list[str]) -> int:
    n = 0
    for t in tokens:
        if any(t == c or (len(t) >= 4 and t in c) for c in cells):
            n += 1
    return n


def detect_tab_type(top_rows) -> tuple[str | None, int | None]:
    """Identify a sheet's data type from its first rows.

    `top_rows` is a list of rows (each a list of raw cell values). Returns
    (type, header_row_index) — header_row_index is the 0-based row that is the column header, or
    None when the match came from a title/summary row (type is still returned). (None, None) if no
    known type is recognized.
    """
    best = None  # (hits, -group_index, type, header_idx_or_None)
    for gi, (typ, tokens, min_hits, is_header) in enumerate(_GROUPS):
        ntokens = [_norm(t) for t in tokens]
        for ri, row in enumerate(top_rows[:SNIFF_ROWS]):
            cells = [_norm(c) for c in row if _norm(c)]
            if not cells:
                continue
            h = _hits(ntokens, cells)
            if h >= min_hits:
                cand = (h, -gi, typ, ri if is_header else None)
                if best is None or cand > best:
                    best = cand
                break  # first qualifying row for this group is enough
    if best is None:
        return None, None
    return best[2], best[3]


# --- Header-row choice for tabs no signature group recognizes ---------------------------------
#
# `detect_tab_type` only knows the five auditor families above. A POB report, a staff listing or a
# per-person labor tab matches none of them, and the usual fallback — "the first non-empty row is the
# header" — picks the wrong row on any sheet that opens with a merged title banner.
#
# The signal that separates the two is DISTINCTNESS, not width. A merged banner spreads ONE string
# across many cells: on a real POB tab, `Personnel On Board` appeared in 15 cells, so 15 non-empty
# cells held 1 distinct value. The true header one row below had 17 non-empty cells and 17 distinct
# values. A rule based on how many cells are filled, or on how word-like they look, picks the banner
# every time, because the banner passes both tests.
_MIN_HEADER_CELLS = 2       # a two-cell row can be a header; a one-cell row is a title
_MIN_DISTINCT_FRAC = 0.8    # the banner test: repeated cells mean a merged title, not column labels
_MIN_LABELISH_FRAC = 0.6    # column labels are words, not amounts (tolerates year/number columns)
_MIN_WIDTH_FRAC = 0.5       # a header spans most of the table, not a corner of it
_MAX_LABEL_CHARS = 80       # beyond this a cell is prose (a note or a disclaimer), not a label

_NUMERIC_RE = re.compile(r"^[-+(]?[\s$€£]*[\d,]+(\.\d+)?\s*[)%]?$")


def _is_labelish(value) -> bool:
    """True when a cell reads like a column label rather than a value.

    A label is a string with at least one letter that is not just a formatted number and is short
    enough to be a heading. Dates and numbers arrive as non-str types from every reader, so they are
    excluded by the type check alone.
    """
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or len(text) > _MAX_LABEL_CHARS:
        return False
    if _NUMERIC_RE.match(text):
        return False
    return any(ch.isalpha() for ch in text)


def choose_header_row(top_rows) -> tuple[int | None, str]:
    """Pick the column-header row of a sheet no signature group recognized.

    `top_rows` is the sniffed prefix of the sheet (a list of rows, each a list of raw cell values).
    Returns `(row_index, confidence)` where `row_index` is 0-based within `top_rows`:

      * `(i, "label-row")`         — row `i` looks like column labels.
      * `(None, "no-header(positional)")` — no row qualifies. The sheet is positional, which is a real
        and common shape here: the helicopter manifest and the staff listing's `Sheet1` both have no
        single header row. Read those by column index.
      * `(None, "empty")`          — nothing in the sniff window at all.

    A row is rejected when any of these holds, and the first row that survives wins:
      * fewer than `_MIN_HEADER_CELLS` non-empty cells,
      * distinct values below `_MIN_DISTINCT_FRAC` of non-empty cells (the merged-banner test),
      * label-like cells below `_MIN_LABELISH_FRAC` of non-empty cells,
      * non-empty cells below `_MIN_WIDTH_FRAC` of the widest row in the window.

    Reporting "no header" is deliberate. A wrong header row is worse than none, because every agent
    downstream inherits it and reads the whole sheet under the wrong column names.
    """
    rows = list(top_rows or [])[:SNIFF_ROWS]
    filled: list[list] = []
    for row in rows:
        filled.append([c for c in (row or []) if _norm(c)])
    widest = max((len(f) for f in filled), default=0)
    if widest == 0:
        return None, "empty"

    for idx, cells in enumerate(filled):
        n = len(cells)
        if n < _MIN_HEADER_CELLS:
            continue
        if n < _MIN_WIDTH_FRAC * widest:
            continue
        if len({_norm(c) for c in cells}) < _MIN_DISTINCT_FRAC * n:
            continue                      # merged banner: many cells, one value
        if sum(1 for c in cells if _is_labelish(c)) < _MIN_LABELISH_FRAC * n:
            continue                      # a data row, not a header
        return idx, "label-row"
    return None, "no-header(positional)"


_DESCRIPTIONS: dict[str, str] = {
    "JADE": (
        "JADE is a standardized SAP/BW joint-venture accounting ledger export — one row per posted "
        "cost line. Header is at the top (row 1-2, occasionally under a title block). Most attributes "
        "are adjacent Key+Text pairs: the first cell holds the code, the next the description (do not "
        "treat the repeat as a duplicate). Key columns: 'Joint Venture(JV)', 'JV:JOA', 'Equity group', "
        "'Fiscal year/Period' (period.year, e.g. 004.2024), 'Account (ACCT)' (GL account + name), "
        "'Cost Object WBS/CC (CO)' = the PRIMARY WBS code charged (+ description), 'PostngObj WBS/CC/OR "
        "(PSO)' = posting-object WBS (usually equal to CO), 'Partner Cost Object' = partner-side WBS on "
        "cross-charges, 'JIB/JIBE Class'/'Subclass', document numbers ('Doc # FI', 'Inv 1/2 Doc #', "
        "'Ref Doc #'), 'Vendor (Linked)', 'PO Item (POI)', 'Line Item Text Orig' (free-text narrative — "
        "a key audit field), 'Material / Service(MATSRV)', and 'CO:JV Object Type Cat' (CAPEX/OPEX). "
        "Amounts: 'Actual Amout in Local [Currency]' (the 'Amout' misspelling is in every file) and "
        "'Actual Amount in Trans'; quantity is 'Qty - Local UoM'. Dates ('Document Date', 'Posting "
        "date', dd.mm.yyyy) appear in 2023+ exports. WBS codes look like C.US.GWH..., O.US.132.PER..., "
        "P.US.PER.EW.... To find the disputed charges, filter rows by the WBS/cost object, account, or "
        "document/PO named in the auditor's description and read the amount and line-item text. "
        "Auditor-built tabs with 'OH%'/'Allocation %'/'Exc Amt'/'Total Exc Claim' are derived "
        "allocation calculations, not raw JADE."
    ),
    "CJI3": (
        "CJI3 is an SAP actual-cost line-item report (transaction CJI3) for a WBS/project. Header is "
        "row 1 (sometimes below a title, sometimes stacked under a JADE block on the same tab). "
        "Identifying columns: 'WBS element' (+ 'WBS Description') = the project/well WBS (C.US...., "
        "O.US....), 'Cost Element' (+ 'Cost element name'), 'Value in Obj. Crcy' = the posted amount, "
        "'Total quantity', 'Document type', 'Document Number'/'Ref. document number', 'Invoice Number', "
        "'Posting Date'/'Document Date', 'Vendor Name', 'Material'/'Material Description', 'Purchasing "
        "Document'/'Item', 'Recovery Indicator'. An older operator variant leads with an 'Object' column "
        "and places 'WBS element' mid-table. CJI3 is the SAP record of what actually posted to a WBS; "
        "auditors reconcile it against JADE and the vendor invoices. Locate disputed lines by cost "
        "element, document/invoice number, or material named in the description; the amount is 'Value in "
        "Obj. Crcy'."
    ),
    "iWells": (
        "iWells is a well-construction component/installation extract. It has NO financial amounts and "
        "NO WBS column — the well is identified by 'Network' (well network number) and 'Well common "
        "name'. Fixed ~22-column header at row 1: 'Network', 'Well common name', 'Event', event/report "
        "dates, 'Assembly', 'Status', 'Component name', 'Size (in)', 'Weight (ppf)', 'Grade', 'Top "
        "Connection', 'Item Description', '# Jts' (joints), 'Length (ft)', 'Manufacturer', 'Component "
        "serial number' (dates may appear as Excel serials). It is the physical record of what tubulars/"
        "components were actually run or installed in a well, used to corroborate installed quantities "
        "(footage, joint count) against what was billed. Match by well ('Network'/'Well common name') "
        "and component/size to the item in dispute, then sum '# Jts' / 'Length (ft)' for the installed "
        "quantity."
    ),
    "BoM": (
        "BoM (Bill of Materials) is the Shell 'Package Library' planned-materials template. A 'Bill of "
        "Materials' title sits above the header row (~row 7): 'Section Name', 'Nominal Section Outside "
        "Diameter', 'Vendor', 'Type of Equipment', 'Shell Common Description', 'SAP Short Description', "
        "'Weight or Wall Thickness', 'Grade', 'Connection', 'Qty Needed', 'Unit', 'Unit Cost', 'Total "
        "Cost', 'Vendor Part Number', 'Shell MMR', 'Qty Purchased/Transfer'd'. WBS appears in the top "
        "metadata block (WBS, Network, Asset, Well), not as a per-row column. BoM is the planned/"
        "purchased material list; auditors compare its 'Qty Needed'/'Qty Purchased' against installed "
        "(iWells) and billed (CJI3/JADE) quantities. Match items by 'Shell Common Description' / 'SAP "
        "Short Description' and size/grade."
    ),
    "Recon": (
        "A materials/tubular reconciliation the auditor built. Three layouts occur. (A) Tubular/"
        "Materials/Equipment Reconciliation: a '... RECONCILIATION' title, a metadata block (Property, "
        "Well, Project, WBS, Tubular Size), a per-item table (JIB Date, Voucher Ref, Material No./"
        "Description, Size, Weight, Grade, Vendor, Invoice No., Footage, Amount charged, Average Price "
        "Per Foot) and footer math — TOTAL CHARGED, SET/INSTALLED IN WELL, THREAD LOSS, OVER(UNDER) "
        "CHARGED, CREDIT (CHARGE) TO JOINT; the over/under-charge is the claimed amount. (B) JADE-vs-"
        "CJI3 account crosswalk: side-by-side 'Jade' and 'CJI3' blocks by account/year with a "
        "'Difference' column. (C) Reconciliation Summary / MDM quantity recon: 'BoM/CN Quantity' vs "
        "'SAP LI Detail Quantity' vs 'Installed qty per Iwells' with 'Quantity Out of Tolerance'. WBS "
        "is in the metadata/top cell. Recon ties BoM, CJI3/JADE and iWells together; its over-charge "
        "or out-of-tolerance quantity is what the exception claims."
    ),
}


def schema_description(tab_type: str | None) -> str:
    """The pre-written schema description for a detected type, or '' when unknown."""
    return _DESCRIPTIONS.get(tab_type or "", "")


# --- WBS / cost-object columns (JADE) -----------------------------------------------------------
# JADE stores each dimension as an adjacent Key+Text pair: a '<label> - Key …' column holding the
# CODE (e.g. 'C.US.GWH.PD.19.008.0942A') and a plain '<label>' column holding its DESCRIPTION
# (e.g. 'GWH DCR Swap 2020'). The code is what we filter rows on; the description is the human label
# the LLM uses to pick the WBS code(s) of interest. We target the PRIMARY charged WBS — 'Cost Object
# WBS/CC (CO)' — and deliberately NOT 'PostngObj (PSO)' (numeric posting objects) or 'Partner Cost
# Object', which would add noise and inflate the distinct-code set.
_WBS_LABELS = ("cost object wbs",)
_WBS_COL_FALLBACK = 10  # col K — the auditor's manual procedure filters JADE on "col K"


def wbs_key_columns(header: list, allow_fallback: bool = True) -> list[int]:
    """Positional indices of the WBS *code* columns — the '<label> - Key' variants.

    With `allow_fallback` (row filtering on a known JADE tab), falls back to col K (index 10) when no
    header label resolves. Candidate enumeration passes `allow_fallback=False` so the col-K guess
    never fires on an unrelated sheet.
    """
    idx = [i for i, h in enumerate(header)
           if any(lbl in _norm(h) for lbl in _WBS_LABELS) and "key" in _norm(h)]
    if idx or not allow_fallback:
        return idx
    return [_WBS_COL_FALLBACK] if len(header) > _WBS_COL_FALLBACK else []


def wbs_desc_column(header: list, key_idx: int) -> int | None:
    """Index of the description column paired with a '<label> - Key' code column.

    That is the column whose label equals the key label minus the ' - Key' suffix — almost always
    the immediate next column. Returns None if no paired description column is found.
    """
    if key_idx < 0 or key_idx >= len(header):
        return None
    # Strip the ' - Key …' suffix (may be followed by qualifiers, e.g. '- Key (Not Compounded)').
    target = re.sub(r"\s*-\s*key\b.*$", "", _norm(header[key_idx])).strip()
    if not target:
        return None
    # Prefer the immediate neighbour, then any other plain-label (non-Key) column with that label.
    order = [key_idx + 1] + [i for i in range(len(header)) if i != key_idx + 1]
    for i in order:
        if 0 <= i < len(header) and "key" not in _norm(header[i]) and _norm(header[i]) == target:
            return i
    return None


def normalize_wbs_codes(codes) -> list[str]:
    """Uppercase, strip whitespace, and drop trailing wildcard/punctuation so an auditor prefix like
    'C.US.GW**' matches a full JADE code 'C.US.GWH.PD.19...' (prefix match)."""
    out = []
    for c in codes or []:
        c = re.sub(r"[.*\s]+$", "", str(c).strip().upper())
        if c:
            out.append(c)
    return out


def wbs_row_matches(row: list, key_idxs: list[int], norm_codes: list[str]) -> bool:
    """True if any WBS *code* cell in `row` matches any normalized code (prefix or substring)."""
    if not norm_codes or not key_idxs:
        return False
    for i in key_idxs:
        if i < 0 or i >= len(row):
            continue
        cell = str(row[i] or "").strip().upper()
        if not cell:
            continue
        for code in norm_codes:
            if cell.startswith(code) or code in cell:
                return True
    return False
