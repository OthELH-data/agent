"""Split a worksheet's rows into the separate tables it actually contains.

Why this exists. `extract_sheet` takes the first non-empty row as THE header and every row below it as
data. That is right for a ledger export and wrong for a hand-made tab. The `Oyler` tab of `IR 24
Affiliate Coding Error.xlsx` holds a helicopter trip table (26 columns, 3 rows), a blank row, and then
an entirely different personnel table (16 columns, 1 row). Read as one table, the second table's
HEADER ROW becomes a data row of the first, so `TRIP ID` ends up holding the values
`41341, 41880, 'Current POB'` and every column profiles as "mixed". Nothing about that tab can be
stated correctly until the two tables are told apart.

A blank row is NOT enough to split on. Sheet1 of that same workbook has a blank row at index 31 and its
totals row at 32; splitting there would strand the totals row in a block of its own, and the sum it is
supposed to reconcile against would be computed without it. So a boundary needs corroboration, the same
discipline the totals-row detector uses: either the next run opens with a row that looks like a HEADER
and contradicts the established column shapes, or it occupies a substantially different set of columns.
A gap alone means nothing.

Rectangularity does double duty. It is worth reporting — a block that is not a clean rectangle is one
the model should read positionally rather than as a table — and it decides whether to search a run's
interior for a header. A rectangular run is already a clean table, so there is nothing inside it to
find; only a ragged one gets the more expensive interior scan. That keeps the cost on the sheets that
need it.

One limit is deliberate. A header needs at least three labels, so on a two-column tab a stacked table
is only found by the disjoint-columns signal. Two cells are too little to tell a header from a note:
"Adjusted | see below" contradicts a period-and-number table exactly as convincingly as a real header
would, and splitting a table in half is the more expensive mistake.

Coverage is the invariant that matters. Every block's range is contiguous and the ranges tile the whole
row set, so no row can fall between two blocks and quietly vanish from the statistics. Reporting a
figure over a subset while implying the whole is the exact failure this package was written to stop.
"""
from __future__ import annotations

from extractor.column_profile import _is_blank, _sample_indices, classify_shapes

_TYPED = ("period", "date", "number")   # kinds a header label would contradict

# A header row needs several labels. One or two stray strings above a table is a note or a merged
# title, and treating it as a header would split a table in half.
_HEADER_MIN_CELLS = 3
_HEADER_TEXT_SHARE = 0.75    # share of a candidate header's filled cells that must be text
_HEADER_DISAGREE = 0.50      # share of the typed columns it fills that must hold text instead
_SPAN_OVERLAP_MIN = 0.50     # below this overlap with the current block, a run is its own table

# How many rows above a candidate interior header to measure the column shapes over. Bounded because
# the shapes that matter are the nearby ones: a table that changes shape 500 rows up is a different
# table anyway. Small enough that the sampling stays cheap when a run has many candidates.
_HEADER_REF_ROWS = 25

# Rectangularity: in a clean table a column is either populated in nearly every row or in almost none.
# Columns sitting in between are what a second table jammed into the same range produces.
_RECT_FILL_HI = 0.80
_RECT_FILL_LO = 0.20
_RECT_SHARE = 0.85           # share of span columns that must be decisively one or the other
_RECT_SAMPLE_ROWS = 400      # rows sampled for the fill measurement, strided across the block

# Bounds. Neither is a tuning knob so much as a statement about what kind of sheet can be stacked: a
# hand-built recon tab has tens of rows and a handful of regions, while a 200k-row JADE export is one
# table by construction. Past these limits the sheet is a ledger and is returned as a single block.
_MAX_SEGMENT_ROWS = 20_000
_MAX_RUNS = 60


def _row_filled(row: list, width: int) -> list[int]:
    """Indices of the cells in `row[:width]` that hold a value."""
    return [i for i in range(min(width, len(row))) if not _is_blank(row[i])]


def _runs(data_rows: list[list], width: int) -> list[tuple[int, int]]:
    """Maximal inclusive ranges of consecutive rows that hold at least one value."""
    out: list[tuple[int, int]] = []
    start = None
    for j, row in enumerate(data_rows):
        blank = not _row_filled(row, width)
        if blank:
            if start is not None:
                out.append((start, j - 1))
                start = None
        elif start is None:
            start = j
    if start is not None:
        out.append((start, len(data_rows) - 1))
    return out


def _span(data_rows: list[list], width: int, lo_row: int, hi_row: int):
    """Populated column span `(lo, hi)` across a row range, or None when every row is blank."""
    lo, hi = None, None
    for j in range(lo_row, hi_row + 1):
        filled = _row_filled(data_rows[j], width)
        if not filled:
            continue
        lo = filled[0] if lo is None else min(lo, filled[0])
        hi = filled[-1] if hi is None else max(hi, filled[-1])
    return None if lo is None else (lo, hi)


def _span_overlap(a, b) -> float:
    """Overlap of two spans as a share of the NARROWER one.

    Measured against the narrower span so that containment counts as full overlap. Sheet1's totals row
    spans columns 49-66 inside a 0-71 table; scoring that against the wider span would make it look
    like a different table and split it off from the rows it totals.
    """
    if a is None or b is None:
        return 0.0
    inter = min(a[1], b[1]) - max(a[0], b[0]) + 1
    if inter <= 0:
        return 0.0
    return inter / min(a[1] - a[0] + 1, b[1] - b[0] + 1)


def _fill_rates(data_rows: list[list], width: int, rows: list[int], span) -> list[float]:
    """Per-column share of `rows` that hold a value, over the span's columns."""
    if not rows or span is None:
        return []
    out = []
    for i in range(span[0], span[1] + 1):
        hit = sum(1 for j in rows
                  if i < len(data_rows[j]) and not _is_blank(data_rows[j][i]))
        out.append(hit / len(rows))
    return out


def _rectangularity(data_rows: list[list], width: int, lo_row: int, hi_row: int, span) -> float:
    """How decisively this range's columns are either always or never populated (1.0 = clean table)."""
    rows = [lo_row + k for k in _sample_indices(hi_row - lo_row + 1, _RECT_SAMPLE_ROWS)]
    rows = [j for j in rows if _row_filled(data_rows[j], width)]
    rates = _fill_rates(data_rows, width, rows, span)
    if not rates:
        return 1.0
    decisive = sum(1 for r in rates if r >= _RECT_FILL_HI or r <= _RECT_FILL_LO)
    return decisive / len(rates)


def _looks_like_labels(row: list, width: int) -> bool:
    """Cheap prefilter: enough cells, and nearly all of them text. True of any header row.

    Also true of plenty of ordinary data rows, so this decides nothing on its own. Its job is to keep
    the expensive check below off the vast majority of rows.
    """
    filled = _row_filled(row, width)
    if len(filled) < _HEADER_MIN_CELLS:
        return False
    return sum(1 for i in filled if isinstance(row[i], str)) / len(filled) >= _HEADER_TEXT_SHARE


def _contradicts_kinds(row: list, width: int, kinds: dict[int, str]) -> bool:
    """True when this row puts TEXT where `kinds` says the table keeps periods, dates or numbers.

    This is the evidence that a new table starts here. A row of mostly text is not remarkable on its
    own — plenty of tables have text-only rows — but a row that contradicts the established shape of
    the columns above it is a header. Where there is nothing to contradict (no typed columns at all)
    this returns False, so an all-text table never splits on this signal. Guessing would be worse than
    leaving it whole.
    """
    filled = _row_filled(row, width)
    typed = [i for i in filled if kinds.get(i) in _TYPED]
    if not typed:
        return False
    contradicts = sum(1 for i in typed if isinstance(row[i], str))
    return contradicts / len(typed) >= _HEADER_DISAGREE


def _is_header_like(row: list, width: int, kinds: dict[int, str]) -> bool:
    """A header row: label-shaped, and contradicting the column shapes it sits above."""
    return _looks_like_labels(row, width) and _contradicts_kinds(row, width, kinds)


def _kinds_for(data_rows: list[list], width: int, lo_row: int, hi_row: int,
               skip: int | None = None) -> dict[int, str]:
    """Column kinds for one block's data rows, excluding its own header row.

    One deliberate difference from `profile_columns`. That function will not call a column typed on the
    evidence of a single value, and it is right not to: a date range or a total claimed from one cell is
    a claim the data does not support. Here the question is only "does the row below disagree with the
    rows above", and for that one typed cell is real evidence. The two mistakes are not symmetric
    either. A column wrongly called text in profiling costs a missing statistic; the same mistake here
    merges two unrelated tables and puts one table's figures under the other's column names.

    So a lone value is taken at its own shape. Two or more that disagree still come back "mixed", since
    a genuinely mixed column is no basis for a boundary.
    """
    rows = [data_rows[j] for j in range(lo_row, hi_row + 1) if j != skip]
    if not rows:
        return {}
    shapes = classify_shapes(rows, width)
    out: dict[int, str] = {}
    for i, kind in shapes.kinds.items():
        vals = shapes.sample_vals.get(i) or []
        if kind not in _TYPED and len(vals) == 1:
            kind = vals[0][1][0]
        out[i] = kind
    return out


def _block(data_rows: list[list], width: int, start: int, end: int,
           header_row: int | None, header: list, fmt) -> dict:
    """Assemble one block descriptor over the inclusive coverage range `start..end`.

    `start`/`end` are the COVERAGE range, which absorbs the blank rows separating this block from the
    next so that the ranges tile the sheet with no gaps. `data_start`/`data_end` are the block's real
    extent, with those blanks and its own header row excluded — that pair is what to report, since a
    range padded out to a separator would overstate how many rows the table has.
    """
    data_lo = start if header_row is None else header_row + 1
    filled = [j for j in range(data_lo, end + 1) if _row_filled(data_rows[j], width)]
    span = _span(data_rows, width, data_lo, end) if data_lo <= end else None
    rect = (_rectangularity(data_rows, width, data_lo, end, span)
            if data_lo <= end and span else 1.0)
    return {
        "start": start,
        "end": end,
        "data_start": filled[0] if filled else None,
        "data_end": filled[-1] if filled else None,
        "header_row": header_row,
        # Blank cells become "" here rather than whatever `fmt` makes of None. A discovered header row
        # is padded out to the sheet's width, so most of its trailing cells are empty; letting them
        # render as "None" would put that word in the prompt as if it were a column name.
        "header": [("" if _is_blank(v) else fmt(v)) for v in (header or [])][:width],
        "n_rows": len(filled),
        "span": list(span) if span else None,
        "rectangularity": round(rect, 3),
        "rectangular": rect >= _RECT_SHARE,
    }


def _interior_splits(data_rows: list[list], width: int, lo: int, hi: int) -> list[int]:
    """Header rows found INSIDE a ragged run — stacked tables with no blank row between them.

    Only reached for a run that failed the rectangularity test, which is both the cost bound and the
    correctness guard: a run whose columns are cleanly all-or-nothing is already one table, and looking
    inside it could only produce a false split.

    Each candidate is measured against the rows immediately ABOVE it, not against the whole run. That
    matters, and it is not a refinement. Over the whole run the candidate header's own text cells are
    part of the sample, so they pull the very columns they should contradict toward "mixed" — in a short
    run, a single text cell among five numbers is enough to do it. The column then reads as untyped,
    there is nothing left to contradict, and the split never happens. Measuring above the candidate
    also asks the right question: does this row contradict the table it would terminate?
    """
    found: list[int] = []
    prev = lo                                  # first row of the table the next candidate would end
    for j in range(lo + 1, hi + 1):
        if not _looks_like_labels(data_rows[j], width):
            continue                           # cheap prefilter keeps the sampling below rare
        kinds = _kinds_for(data_rows, width, max(prev, j - _HEADER_REF_ROWS), j - 1)
        if _contradicts_kinds(data_rows[j], width, kinds):
            found.append(j)
            prev = j + 1                       # the next candidate is judged against THIS table
    return found


def segment(header: list, data_rows: list[list], populated_width: int, fmt=str) -> list[dict]:
    """Split `data_rows` into the tables it contains. Returns one block descriptor per table.

    The blocks' `start..end` ranges are contiguous and tile the whole row set, so every row belongs to
    exactly one block and none can be dropped. `fmt` renders a discovered header row's cells to labels
    (callers pass their own cell formatter so this module has no formatting opinion).

    A single-table sheet — nearly all of them — returns exactly one block covering everything, which is
    what the callers already assumed. Splitting has to be earned.
    """
    n = len(data_rows)
    if populated_width <= 0 or n == 0:
        return []

    whole = [_block(data_rows, populated_width, 0, n - 1, None, header, fmt)]
    runs = _runs(data_rows, populated_width)
    if not runs:
        return whole
    # A ledger is one table by construction, and a sheet with dozens of separated runs is a sparse
    # ledger rather than a stack of tables. Either way, do not segment.
    if n > _MAX_SEGMENT_ROWS or len(runs) > _MAX_RUNS:
        return whole

    # Boundaries are collected as (row_index, header_row_or_None) before any block is built, so the
    # tiling can be laid down in one pass at the end and cannot develop a gap.
    cuts: list[tuple[int, int | None]] = []
    cur_lo, cur_hi = runs[0]
    cur_header_row: int | None = None
    kinds = _kinds_for(data_rows, populated_width, cur_lo, cur_hi)

    def _scan_interior(lo: int, hi: int, hdr: int | None) -> None:
        """Record any header rows inside a ragged run as further boundaries."""
        span = _span(data_rows, populated_width, lo, hi)
        if not span or _rectangularity(data_rows, populated_width, lo, hi, span) >= _RECT_SHARE:
            return
        for j in _interior_splits(data_rows, populated_width, (hdr + 1) if hdr is not None else lo, hi):
            cuts.append((j, j))

    _scan_interior(cur_lo, cur_hi, None)
    for lo, hi in runs[1:]:
        new_header = _is_header_like(data_rows[lo], populated_width, kinds)
        disjoint = (_span_overlap(_span(data_rows, populated_width, lo, hi),
                                 _span(data_rows, populated_width, cur_lo, cur_hi))
                    < _SPAN_OVERLAP_MIN)
        if not (new_header or disjoint):
            cur_hi = hi                     # continuation of the same table across a blank row
            continue
        cuts.append((lo, lo if new_header else None))
        cur_lo, cur_hi = lo, hi
        cur_header_row = lo if new_header else None
        kinds = _kinds_for(data_rows, populated_width, cur_lo, cur_hi, skip=cur_header_row)
        _scan_interior(cur_lo, cur_hi, cur_header_row)

    if not cuts:
        return whole

    # Lay the tiling down: each block runs from its own start to the row before the next block's.
    cuts.sort()
    blocks: list[dict] = []
    bounds = [(0, None)] + cuts
    for k, (start, hdr) in enumerate(bounds):
        end = (bounds[k + 1][0] - 1) if k + 1 < len(bounds) else n - 1
        if end < start:
            continue
        hdr_cells = data_rows[hdr] if hdr is not None else header
        blocks.append(_block(data_rows, populated_width, start, end, hdr, hdr_cells, fmt))
    return blocks or whole


def primary(blocks: list[dict]) -> dict | None:
    """The block that best represents the tab: the most data rows, earliest on a tie."""
    return max(blocks, key=lambda b: (b["n_rows"], -b["start"])) if blocks else None


def block_rows(block: dict, data_rows: list[list]) -> list[list]:
    """A block's data rows — its coverage range with its own header row left out."""
    lo = block["start"] if block["header_row"] is None else block["header_row"] + 1
    return data_rows[lo:block["end"] + 1]


def render_blocks(blocks: list[dict], primary_block: dict | None) -> str:
    """State a multi-table layout for the prompt. Empty string when the tab is one clean table.

    The point is to stop the model reading one table's figures under another's headers, and to be
    explicit that the column profile describes the primary block only — never to imply the profile
    covers rows it did not see.
    """
    if len(blocks) < 2:
        return ""
    lines = [f"SHEET LAYOUT — this tab holds {len(blocks)} separate tables stacked in one sheet, not "
             f"one table. Each has its OWN header, so a value from one table must never be read under "
             f"another's column name."]
    for k, b in enumerate(blocks, start=1):
        tag = " <- the COLUMN PROFILE below describes THIS table only" if b is primary_block else ""
        hdr = [h for h in b["header"] if h][:8]
        # Row numbers are the DATA extent, so a block padded out to a blank separator does not claim
        # rows it has none in. 1-based, to match how the rows are numbered elsewhere in the prompt.
        where = (f"data rows {b['data_start'] + 1}-{b['data_end'] + 1}"
                 if b["data_start"] is not None else "no data rows")
        if b["header_row"] is not None:
            where = f"header on row {b['header_row'] + 1}, " + where
        span = f"cols c{b['span'][0] + 1}-c{b['span'][1] + 1}" if b["span"] else "no populated columns"
        noun = "row" if b["n_rows"] == 1 else "rows"
        lines.append(f"  table {k}: {where} ({b['n_rows']} {noun}, {span}){tag}")
        lines.append(f"     columns: {', '.join(hdr)}"
                     + (" …" if len([h for h in b['header'] if h]) > 8 else ""))
        if not b["rectangular"]:
            lines.append("     this table is NOT a clean rectangle — read its rows from the raw "
                         "positional cells rather than as a column table.")
    return "\n".join(lines)
