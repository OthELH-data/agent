"""Bounded Excel extraction for large audit exhibits.

Auditor exhibits contain raw ledger tabs (e.g. Perdido's `JADE` sheet declares ~983k rows).
Loading such sheets in full is slow and memory-heavy, so we use openpyxl in read_only mode and
stop after a capped number of rows per sheet — enough for an LLM to understand each tab's
structure and sample its data without materializing millions of cells.

Three workbook formats are handled behind one interface: `.xlsx`/`.xlsm` via openpyxl, `.xlsb`
(binary) via pyxlsb, and legacy `.xls` (BIFF) via xlrd — the pyxlsb/xlrd adapters present the
slice of openpyxl's read-only API that `extract_sheet` needs, so all downstream logic is shared.

Column width is reported two ways, and the difference matters. `reported_cols` is what the sheet
DECLARES (openpyxl's `max_column`), which Excel does not shrink when content is deleted — one observed
72-column exhibit declares 16,378. `populated_cols` is the width that actually holds values, and it is
what `header`, `records` and the preview are sized to. Using the declared width shipped tens of
thousands of empty strings into every prompt; on that same sheet the header alone serialized to 66 KB
against 30 KB of real rows. Keep `reported_cols` for provenance, size everything else to
`populated_cols`, and never renumber columns — `raw_grid` addresses cells as `c<idx>` and the
synthesis cites those indices, so an interior gap keeps its slot.

Output shape per sheet:
  {name, reported_rows, reported_cols, populated_cols, detected_type, header_row, header, columns,
   n_data_rows_scanned, truncated, preview_md, raw_grid, rows, records, column_profile, total_rows,
   blocks}
"""
from __future__ import annotations

import datetime as _dt
import itertools
import sys
import zipfile
from pathlib import Path

import openpyxl

from extractor.blocks import block_rows, primary, render_blocks, segment
from extractor.column_profile import profile_columns
from .response_names import is_response_tab  # tab-level twin of rebuttals.drop_operator_responses
from .tab_schemas import (SNIFF_ROWS, detect_tab_type, normalize_wbs_codes, wbs_desc_column,
                          wbs_key_columns, wbs_row_matches)

_MAX_SCAN_ROWS = 300     # data rows serialized per sheet (bounded default)
_FULL_SCAN_ROWS = 200000 # skill runs: pull ALL rows (ceiling far above the largest observed ledger)
_PREVIEW_ROWS = 15       # rows rendered into the markdown preview
_CELL_MAXLEN = 60        # truncate long cell strings in the preview
_RAW_GRID_ROWS = 45      # non-empty rows rendered into the raw positional grid
_RAW_GRID_CHARS = 6000   # hard char cap on the raw positional grid


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        # keep integers clean, money to 2dp; avoid scientific notation noise
        return f"{value:.2f}".rstrip("0").rstrip(".") if value % 1 else f"{value:.0f}"
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()
    return str(value)


def _is_empty_row(cells: list) -> bool:
    return all(c is None or (isinstance(c, str) and not c.strip()) for c in cells)


def _preview_md(header: list[str], rows: list[list], n: int) -> str:
    cols = header or [f"col{i+1}" for i in range(len(rows[0]))] if rows else []
    if not cols:
        return "(empty sheet)"
    def clip(s: str) -> str:
        return s if len(s) <= _CELL_MAXLEN else s[: _CELL_MAXLEN - 1] + "…"
    lines = ["| " + " | ".join(clip(str(c)) for c in cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows[:n]:
        cells = [clip(_fmt(v)) for v in r][: len(cols)]
        cells += [""] * (len(cols) - len(cells))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _raw_grid(data_rows: list[list], max_rows: int = _RAW_GRID_ROWS,
              max_chars: int = _RAW_GRID_CHARS) -> str:
    """Render scanned rows as raw positional cells (`c<idx>=<value>`), skipping empty rows.

    Header-keyed `records` are lossy: duplicate header labels (JADE's Key+Text pairs) collide, and
    a sub-table stacked at a different column offset gets mis-keyed under the primary header. This
    positional view is faithful — it reads each cell by its true column index — so the interpreter
    can pick up stacked/offset blocks (e.g. an invoice#↔amount reconciliation table) that the clean
    header table hides. Bounded by row and character caps so dense ledgers can't blow the prompt.
    """
    lines: list[str] = []
    total = 0
    shown = 0
    for i, row in enumerate(data_rows, start=1):
        cells = [(j + 1, _fmt(v)) for j, v in enumerate(row)]
        nonempty = [(j, v.strip()) for j, v in cells if v and v.strip()]
        if not nonempty:
            continue
        line = f"r{i}: " + "  ".join(f"c{j}={v}" for j, v in nonempty)
        if total + len(line) > max_chars or shown >= max_rows:
            lines.append("…(more rows omitted from raw grid)")
            break
        lines.append(line)
        total += len(line) + 1
        shown += 1
    return "\n".join(lines)


def extract_sheet(ws, max_scan_rows: int = _MAX_SCAN_ROWS, wbs_codes: list[str] | None = None) -> dict:
    reported_rows = int(ws.max_row or 0)
    reported_cols = int(ws.max_column or 0)

    # Sniff the first rows to recognize a known auditor data type (JADE/CJI3/BoM/Recon/iWells) and
    # locate its real header row — many exhibits sit under a title/metadata block, so "first
    # non-empty row" is not always the header.
    row_iter = ws.iter_rows(values_only=True)
    top: list[list] = []
    for cells in row_iter:
        top.append(list(cells))
        if len(top) >= SNIFF_ROWS:
            break
    detected_type, header_idx = detect_tab_type(top)
    stream = itertools.chain(top, row_iter)  # full row stream (buffered prefix + remainder)

    # WBS pre-filter (JADE only): when codes are supplied, stream the WHOLE ledger (no row cap) and
    # keep only rows whose WBS code column matches — so a 100k+/983k-row JADE is ingested in full but
    # reduced to the relevant slice, with bounded memory (only matches are retained).
    norm_codes = normalize_wbs_codes(wbs_codes) if wbs_codes else []
    jade_filter = bool(norm_codes) and detected_type == "JADE"
    key_idxs: list[int] = []

    header: list[str] = []
    data_rows: list[list] = []
    scanned = 0
    for idx, cells in enumerate(stream):
        cells = list(cells)
        if not header:
            if header_idx is not None:
                if idx < header_idx:
                    continue  # title/metadata above the detected header row
                header = [_fmt(c) for c in cells]
            elif _is_empty_row(cells):
                continue  # skip leading title/blank rows to reach the real header
            else:
                header = [_fmt(c) for c in cells]
            if header and jade_filter:
                key_idxs = wbs_key_columns(header)  # resolve WBS code column(s) once
            continue
        if jade_filter:
            if key_idxs and not wbs_row_matches(cells, key_idxs, norm_codes):
                continue  # drop non-matching JADE rows; do NOT count against the cap
            data_rows.append(cells)
            scanned += 1
            continue  # uncapped: examine every row of the ledger
        data_rows.append(cells)
        scanned += 1
        if scanned >= max_scan_rows:
            break

    rows = [_trim_row([_fmt(v) for v in row]) for row in data_rows]

    # Populated width. `_trim_row` has already dropped each row's trailing empties, so its length IS
    # that row's extent — the widest of those (and of the header) is the width that holds values. This
    # costs nothing extra: it reads the trimmed lists, not the cells again. See the module docstring for
    # why the DECLARED width can't be used. Trailing trim only — a column with a header but no data
    # stays, because a labelled empty column is itself a fact about the exhibit, and interior gaps keep
    # their slots so `c<idx>` citations remain valid.
    populated_cols = max([len(_trim_row(header))] + [len(r) for r in rows])
    header = header[:populated_cols]

    records = [
        {header[i] if i < len(header) and header[i] else f"col{i+1}": _fmt(v)
         for i, v in enumerate(row[:populated_cols])}
        for row in data_rows
    ]

    # Which tables this tab actually holds. A hand-made tab can stack several, and profiling them as
    # one mixes unrelated populations: the `Oyler` tab's second table has its own header row, which read
    # as data put the label 'Current POB' in the `TRIP ID` column and left every column "mixed". Nearly
    # every tab is one table and comes back as a single block covering everything.
    blocks = segment(header, data_rows, populated_cols, fmt=_fmt)
    main = primary(blocks)

    # Profile the largest table only, with its own header. For a single-block tab these are the whole
    # sheet and the sheet header, so nothing changes. `render_blocks` tells the prompt when the profile
    # describes one table out of several, so a figure is never presented as covering rows it did not.
    prof_rows = block_rows(main, data_rows) if main else data_rows
    prof_header = header
    if main and main["header_row"] is not None:
        prof_header = [_fmt(v) for v in data_rows[main["header_row"]]][:populated_cols]

    # Profile from the NATIVE values, not the formatted `rows`: a real date cell arrives as a datetime
    # and a money cell as a float, so nothing has to be re-parsed out of a string. Ranges and totals
    # come out the same either way, but native values cannot be misread.
    col_profile, total_rows = profile_columns(prof_header, prof_rows, populated_cols)

    # A totals row's index came out relative to the block it was found in. Shift it back to the sheet's
    # own data-row numbering, because that is what the prompt prints and what a citation has to resolve
    # against. Zero for a single-block tab.
    row_offset = 0 if not main else (main["start"] if main["header_row"] is None
                                    else main["header_row"] + 1)
    if row_offset:
        total_rows = [{**t, "index": t["index"] + row_offset} for t in total_rows]

    return {
        "name": ws.title,
        "reported_rows": reported_rows,
        "reported_cols": reported_cols,
        "populated_cols": populated_cols,
        "detected_type": detected_type,
        "header_row": header_idx,
        "header": header,
        "columns": [h for h in header if h],
        "n_data_rows_scanned": scanned,
        # A WBS-filtered JADE is fully read (every row examined), just reduced to matches — so it is
        # NOT truncated even though the kept count is far below reported_rows.
        "truncated": (False if jade_filter
                      else bool(reported_rows and reported_rows - 1 > scanned)),
        "wbs_filtered": jade_filter,
        "preview_md": _preview_md(header, data_rows, _PREVIEW_ROWS),
        "raw_grid": _raw_grid(data_rows),
        # positional rows (faithful — preserves offset sub-tables and duplicate-header columns that
        # `records` collapses). Trailing empties trimmed to stay compact; the skill JSON uses these.
        "rows": rows,
        "records": records,
        # Per-column kinds, ranges and totals computed over EVERY scanned row. The prompts show a
        # bounded sample of rows but ask for facts about the whole tab, so these carry the facts a
        # sample cannot: see `column_profile` for why they are computed rather than inferred.
        "column_profile": col_profile,
        "total_rows": total_rows,
        # The tables this tab holds, as contiguous ranges that tile every scanned row. One entry for
        # nearly every tab. Where there are several, `column_profile` above describes only the largest,
        # and `blocks` is what says so.
        "blocks": blocks,
    }


def _trim_row(cells: list[str]) -> list[str]:
    """Drop trailing empty cells but keep interior positions intact."""
    last = -1
    for i, c in enumerate(cells):
        if c != "":
            last = i
    return cells[: last + 1]


class _XlsbSheet:
    """Adapt one pyxlsb worksheet to the slice of openpyxl's read_only API that extract_sheet uses.

    Exposes `.title`, `.max_row`, `.max_column`, and `.iter_rows(values_only=True)`. Cells are placed
    at their true column index (pyxlsb rows are sparse), so column alignment — and any stacked/offset
    sub-table — is preserved exactly as for xlsx.
    """

    def __init__(self, wb, name: str):
        self._wb = wb
        self.title = name
        self.max_row = 0
        self.max_column = 0
        try:  # dimension is display-only (+ the `truncated` flag); default to 0 if unavailable
            with wb.get_sheet(name) as sh:
                dim = getattr(sh, "dimension", None)
                self.max_row = int(getattr(dim, "h", 0) or 0)
                self.max_column = int(getattr(dim, "w", 0) or 0)
        except Exception:
            pass

    def iter_rows(self, values_only: bool = True):
        with self._wb.get_sheet(self.title) as sh:
            for row in sh.rows():
                if not row:
                    yield ()
                    continue
                width = row[-1].c + 1  # cells arrive in column order; last cell fixes the width
                out: list = [None] * width
                for cell in row:
                    out[cell.c] = cell.v
                yield tuple(out)


class _XlsbWorkbook:
    """Minimal openpyxl-workbook stand-in (`.worksheets`, `.close()`) backed by pyxlsb."""

    def __init__(self, path: Path):
        from pyxlsb import open_workbook  # optional dep; imported only for .xlsb
        self._wb = open_workbook(str(path))
        self.worksheets = [_XlsbSheet(self._wb, name) for name in self._wb.sheets]

    def close(self):
        try:
            self._wb.close()
        except Exception:
            pass


class _XlsSheet:
    """Adapt one xlrd worksheet (legacy .xls / BIFF) to the read-only API extract_sheet uses."""

    def __init__(self, sheet):
        self._sheet = sheet
        self.title = sheet.name
        self.max_row = int(sheet.nrows or 0)
        self.max_column = int(sheet.ncols or 0)

    def iter_rows(self, values_only: bool = True):
        for r in range(self._sheet.nrows):
            yield tuple(self._sheet.row_values(r))


class _XlsWorkbook:
    """Minimal openpyxl-workbook stand-in (`.worksheets`, `.close()`) backed by xlrd (.xls only)."""

    def __init__(self, path: Path):
        import xlrd  # optional dep; imported only for legacy .xls
        self._book = xlrd.open_workbook(str(path))
        self.worksheets = [_XlsSheet(self._book.sheet_by_index(i))
                           for i in range(self._book.nsheets)]

    def close(self):
        try:
            self._book.release_resources()
        except Exception:
            pass


_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # legacy .xls / OLE2 compound-document header
_READER_LIB = {"xlsb": "pyxlsb", "xls": "xlrd", "xlsx": "openpyxl"}
_READERS = {
    "xlsx": lambda p: openpyxl.load_workbook(p, read_only=True, data_only=True),  # .xlsx/.xlsm
    "xlsb": lambda p: _XlsbWorkbook(p),                                           # binary .xlsb
    "xls": lambda p: _XlsWorkbook(p),                                            # legacy BIFF .xls
}


def _reader_order(path: Path) -> list[str]:
    """Decide which readers to try, best-guess first, from the file's CONTENT — not its extension.

    .xlsx/.xlsm and .xlsb are both ZIP (`PK`) containers and are told apart only by their internal
    parts (`xl/workbook.bin` → xlsb, `xl/workbook.xml` → xlsx); legacy .xls is an OLE2 document. This
    lets a workbook saved under the wrong extension (e.g. an OLE .xls or a binary .xlsb named .xlsx)
    still be parsed. The remaining readers are appended as fallbacks so nothing is skipped outright.
    """
    all_keys = ["xlsx", "xlsb", "xls"]
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
    except OSError:
        head = b""
    order: list[str]
    if head[:8] == _OLE2_MAGIC:
        order = ["xls"]
    elif head[:2] == b"PK":
        is_xlsb = False
        try:
            with zipfile.ZipFile(path) as z:
                is_xlsb = any(n.lower() == "xl/workbook.bin" for n in z.namelist())
        except Exception:
            pass
        order = ["xlsb", "xlsx"] if is_xlsb else ["xlsx", "xlsb"]
    else:
        # Unknown signature — fall back to the extension hint, then try everything.
        ext = {".xlsb": "xlsb", ".xls": "xls"}.get(path.suffix.lower(), "xlsx")
        order = [ext]
    for k in all_keys:  # append the rest as fallbacks (parse, don't skip)
        if k not in order:
            order.append(k)
    return order


class WorkbookOpenError(Exception):
    """Every reader refused the workbook. Carries each engine's own complaint.

    One file can fail three different ways at once, and the three messages together are what tells you
    why. `SEPCO 2022 Historical POB report.xlsx` in the corpus has OLE2 magic bytes, so xlrd is tried
    first and says "Can't find workbook in OLE2 compound document", while openpyxl and pyxlsb both say
    "File is not a zip file" — which reads as an encrypted or rights-managed package. Any single one of
    those messages on its own would be misleading.

    `missing_libs` is kept apart from `errors` because a reader that was never installed did not reject
    the file; it never looked at it. Callers word those two cases differently.
    """

    def __init__(self, path, errors: dict[str, str], missing_libs: tuple[str, ...] = ()):
        self.path = str(path)
        self.errors = dict(errors)
        self.missing_libs = tuple(missing_libs)
        if errors:
            detail = "; ".join(f"{eng}: {msg}" for eng, msg in errors.items())
        elif missing_libs:
            libs = ", ".join(missing_libs)
            detail = f"required reader(s) not installed: {libs} (`pip install {libs}`)"
        else:
            detail = "unreadable workbook"
        super().__init__(f"{Path(path).name}: {detail}")


def open_workbook_any(path: str | Path):
    """Open a workbook with whichever engine can actually read it. Returns `(wb, engine)`.

    The engine order comes from `_reader_order`, which decides from the file's CONTENT rather than its
    extension — so a workbook saved under the wrong suffix still opens. Every remaining engine is tried
    as a fallback, so a file is only declared unreadable once all three have refused it.

    `engine` is one of "xlsx", "xlsb" or "xls" — the reader that succeeded, which is not necessarily
    what the extension claimed. Callers that report provenance should use this, not `path.suffix`.

    Raises `WorkbookOpenError` when nothing opens it. The caller closes the workbook.
    """
    path = Path(path)
    errors: dict[str, str] = {}
    missing: list[str] = []
    for key in _reader_order(path):
        try:
            return _READERS[key](path), key
        except ImportError:
            missing.append(_READER_LIB.get(key, key))
        except Exception as exc:
            errors[key] = f"{type(exc).__name__}: {exc}"
    raise WorkbookOpenError(path, errors, tuple(dict.fromkeys(missing)))


def _tab_title(ws) -> str:
    return getattr(ws, "title", "") or ""


def _usable_worksheets(wb, source: str = "", warn=None) -> tuple[list, list[str]]:
    """Split a workbook's sheets into (sheets to extract, names of response tabs excluded).

    A TAB whose NAME marks it as the operator's is not auditor evidence even inside a workbook that is
    otherwise the auditor's own — and the corpus is full of those. `2023-648_PER002_CATS Labor
    Administrative.xlsx` is the auditor's exception workbook, but its `Response to Auditor` tab carries the
    operator's justifications; `AE 007 JADE and support - Auditor Reply.xlsx` is the auditor's own answer,
    yet one of its six tabs is the operator's `Operator Response`. Excluding the tab here, at extraction,
    is what keeps the operator's argumentation out of the LLM entirely — not as an interpreted attachment,
    not as a skill-scan chunk, not as a WBS candidate — while the rest of the workbook is still mined.

    Guarded against excluding everything: a few workbooks consist ONLY of response tabs (e.g.
    `SIL AE-39.IR-024 JADE.Need Support.xlsx`, whose single tab is `SIL AE-39-SIL.IR-006 RESP`).
    Returning no sheets there would silently turn the exhibit into a read error, so all sheets are kept
    and a warning names the file — a degraded run stays visible.
    """
    warn = warn or (lambda m: print(m, file=sys.stderr))
    sheets = list(wb.worksheets)
    dropped = [ws for ws in sheets if is_response_tab(_tab_title(ws))]
    if not dropped:
        return sheets, []
    kept = [ws for ws in sheets if not is_response_tab(_tab_title(ws))]
    if not kept:
        warn(f"WARN: every tab in {source or 'this workbook'} looks like an operator response "
             f"({', '.join(_tab_title(ws) for ws in dropped)}) — keeping them all, since excluding "
             f"them would leave the workbook with no sheets.")
        return sheets, []
    return kept, [_tab_title(ws) for ws in dropped]


def extract_excel_bounded(path: str | Path, max_scan_rows: int = _MAX_SCAN_ROWS,
                          wbs_codes: list[str] | None = None) -> dict:
    path = Path(path)

    # Empty/missing files are common (auditors leave 0-byte placeholders). openpyxl would raise a
    # scary BadZipFile on a 0-byte .xlsx; surface a clean, benign skip instead.
    try:
        if not path.exists():
            return {"source": str(path), "kind": "excel", "sheets": [], "empty": True,
                    "error": "file not found"}
        if path.stat().st_size == 0:
            return {"source": str(path), "kind": "excel", "sheets": [], "empty": True,
                    "error": "empty file (0 bytes) — nothing to extract"}
    except OSError:
        pass

    # Try readers in content-based order; only fail if EVERY reader fails (parse, don't skip).
    try:
        wb, _engine = open_workbook_any(path)
    except WorkbookOpenError as exc:
        if exc.missing_libs and not exc.errors:
            libs = ", ".join(exc.missing_libs)
            return {"source": str(path), "kind": "excel", "sheets": [],
                    "error": f"required reader(s) not installed: {libs} (`pip install {libs}`)"}
        # Report every engine's complaint, not just the last one — see WorkbookOpenError.
        return {"source": str(path), "kind": "excel", "sheets": [],
                "error": "; ".join(f"{eng}: {msg}" for eng, msg in exc.errors.items())
                         or "unreadable workbook"}
    sheets: list[dict] = []
    try:
        # Operator-response tabs are dropped BEFORE extraction, so their rows are never materialized
        # and never reach the LLM (interpretation, skill scan, or WBS enumeration).
        usable, excluded_tabs = _usable_worksheets(wb, source=path.name)
        for ws in usable:
            try:
                sheets.append(extract_sheet(ws, max_scan_rows, wbs_codes=wbs_codes))
            except Exception as exc:  # keep going if one sheet is malformed
                sheets.append({"name": ws.title, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        wb.close()
    out = {"source": str(path), "kind": "excel", "n_sheets": len(sheets), "sheets": sheets}
    if excluded_tabs:  # surfaced so the exclusion is auditable, never silent
        out["excluded_tabs"] = excluded_tabs
    return out


def extract_excel_full(path: str | Path, wbs_codes: list[str] | None = None) -> dict:
    """Extract a workbook pulling ALL rows (skill runs) — same shape as extract_excel_bounded.

    When `wbs_codes` is given, JADE sheets are streamed UNCAPPED and filtered to rows whose WBS
    code column matches those codes (see extract_sheet); other sheets keep the normal row ceiling.
    """
    return extract_excel_bounded(path, max_scan_rows=_FULL_SCAN_ROWS, wbs_codes=wbs_codes)


def _collect_sheet_wbs(ws, out: dict[str, str]) -> None:
    """Stream one sheet; if it has a label-resolved WBS code column, add {code: description} to `out`.

    Codes come from the '<label> - Key' column; descriptions from its paired plain-label column.
    First-seen non-empty description wins. Only the distinct-code map is retained, so a huge ledger
    costs one cheap streaming pass (no rows materialized).
    """
    row_iter = ws.iter_rows(values_only=True)
    top: list[list] = []
    for cells in row_iter:
        top.append(list(cells))
        if len(top) >= SNIFF_ROWS:
            break
    _detected, header_idx = detect_tab_type(top)
    stream = itertools.chain(top, row_iter)

    header: list[str] = []
    key_idxs: list[int] = []
    desc_for: dict[int, int | None] = {}
    for idx, cells in enumerate(stream):
        cells = list(cells)
        if not header:
            if header_idx is not None:
                if idx < header_idx:
                    continue
                header = [_fmt(c) for c in cells]
            elif _is_empty_row(cells):
                continue
            else:
                header = [_fmt(c) for c in cells]
            key_idxs = wbs_key_columns(header, allow_fallback=False)  # label match only
            if not key_idxs:
                return  # no WBS code column on this sheet — nothing to enumerate
            desc_for = {ki: wbs_desc_column(header, ki) for ki in key_idxs}
            continue
        for ki in key_idxs:
            if ki >= len(cells):
                continue
            code = _fmt(cells[ki]).strip()
            if not code or code == "#":
                continue
            if code not in out:
                di = desc_for.get(ki)
                out[code] = (_fmt(cells[di]).strip()
                             if di is not None and di < len(cells) else "")


def collect_wbs_candidates(path: str | Path) -> dict[str, str]:
    """Return {wbs_code: description} for every distinct WBS/cost-object code across the workbook.

    Reads the FULL ledger (uncapped) but keeps only the small distinct-code map. Reader dispatch is
    content-based (same as extract_excel_bounded), so a mislabeled .xlsb/.xls is still read. Returns
    {} for unreadable/empty files (a benign skip — the caller falls back to no narrowing).
    """
    path = Path(path)
    try:
        if not path.exists() or path.stat().st_size == 0:
            return {}
    except OSError:
        return {}
    try:
        wb, _engine = open_workbook_any(path)
    except WorkbookOpenError:
        return {}
    out: dict[str, str] = {}
    try:
        # Same tab exclusion as extraction: a code seen only on an operator-response tab must not steer
        # the WBS focus pass.
        for ws in _usable_worksheets(wb, source=path.name)[0]:
            try:
                _collect_sheet_wbs(ws, out)
            except Exception:
                continue  # a malformed sheet must not abort enumeration
    finally:
        wb.close()
    return out


def sheets_to_json(book: dict) -> list[dict]:
    """Serialize an extracted workbook to a compact, faithful JSON payload for the skill pass.

    One entry per sheet: tab name, detected type, positional header + all positional rows (no
    filtering, no summing). Uses `rows` (positional) so offset sub-tables and duplicate-header
    columns survive. `truncated` flags a sheet that hit the scan ceiling.

    `header` is already sized to the populated width by `extract_sheet`, so this no longer ships a
    sheet's declared-but-empty columns into every chunk prompt.

    `column_profile` and `total_rows` ride along because a chunk prompt sees only its own slice of
    rows. Without them, a model asked for the tab's date range can only answer from the slice in front
    of it, and will do so confidently.

    `blocks` rides along for the same reason, one step further out: a chunk cannot see that its rows
    belong to the second of two stacked tables, so without this it would read them under the first
    table's column names.
    """
    out: list[dict] = []
    for sh in book.get("sheets", []):
        if sh.get("error"):
            out.append({"tab": sh.get("name"), "error": sh["error"]})
            continue
        out.append({
            "tab": sh.get("name"),
            "detected_type": sh.get("detected_type"),
            "header": sh.get("header", []),
            "populated_cols": sh.get("populated_cols", 0),
            "n_rows": sh.get("n_data_rows_scanned", 0),
            "truncated": bool(sh.get("truncated")),
            "wbs_filtered": bool(sh.get("wbs_filtered")),
            "column_profile": sh.get("column_profile", []),
            "total_rows": sh.get("total_rows", []),
            "blocks": sh.get("blocks", []),
            "rows": sh.get("rows", []),
        })
    return out
