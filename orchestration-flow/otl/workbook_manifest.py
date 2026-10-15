"""Read the shape of a case's workbooks once, from disk, before any agent starts reasoning.

Every code-exec agent used to begin the same way: list the sheets, open a tab, print the first rows,
guess which row is the header, and only then do the real work. Each of those turns is a full LLM call
that re-sends the whole ReAct buffer. Across the traces we measured, 226 of 559 ReAct steps — 40% —
were spent that way.

The shape of a workbook is a fact on disk. Reading it needs no model. So this module reads it once,
deterministically, and hands the answer to the agents through two channels:

  * `build_manifest(...)` builds the record. It is staged into the code sandbox as `manifest.json`,
    where the agent's own Python reads it through the `manifest()` and `find_tab()` helpers. That
    channel costs zero tokens.
  * `manifest_block(...)` renders a short, bounded summary that is appended to the agent's task text,
    so the agent knows the shape before it writes its first line of code.

Nothing here adds a new Excel reader. `extractor/excel_bounded.py` already does content-based reader
dispatch (`open_workbook_any`) and drops operator-response tabs (`_usable_worksheets`), and
`extractor/tab_schemas.py` already recognizes the five auditor data families (`detect_tab_type`) and
picks a header row for the tabs it does not recognize (`choose_header_row`). This module composes them.

Two design points worth knowing:

**Tabs are grouped by shape, and that is where the compression comes from.** Sheet count scales with
the number of people questioned, because each person can get their own data tab. The largest workbook
in the corpus has 63 tabs — and they collapse into 8 distinct shapes, with 50 tabs in the biggest
group. The fingerprint is `(header row index, header labels)`, and the row index has to be part of it:
in that same file the identical 22-column labor header sits at row 0 on 50 tabs and at row 1 on 5 more.

**No row counts.** A true count needs a full stream, and one tab in the corpus is about 100k rows. We
record `ws.max_row` when the writing tool stored a `<dimension/>` tag and `null` otherwise, on disk
only, and never in the prompt — it is the same unreliable number `PROTOCOL` already warns agents about.

Deterministic. Nothing here calls an LLM.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from shared.excel_bounded import (WorkbookOpenError, _usable_worksheets,  # noqa: PLC2701
                                     open_workbook_any)
from shared.tab_schemas import SNIFF_ROWS, choose_header_row, detect_tab_type

# Only these are opened as workbooks. Anything else handed to us is recorded by name and size but not
# parsed — three engine errors on a PDF is noise, not information.
_WORKBOOK_EXTS = {".xlsx", ".xlsm", ".xlsb", ".xls"}

# Prompt-rendering bounds. Every one of these drops WHOLE items and says how many it dropped, the same
# way `_cap_items` in agents/_common.py does — silent truncation reads as "that is everything".
_MAX_TABS_SHOWN = 8        # tab names listed per shape
_MAX_COL_CHARS = 260       # chars of the column-label line per shape
_MAX_DIR_FILES = 30        # filenames listed per directory
_MAX_PERSON_TABS = 4       # candidate tabs listed per person

# Shape profiles keyed on (resolved path, size, mtime_ns). One case is one process, so all five agents
# in a case share this; `run_batch.py` forks per case, so nothing leaks between cases.
_PROFILE_CACHE: dict[tuple, dict] = {}


# --- small helpers ------------------------------------------------------------------------------

def _norm_label(value) -> str:
    """A header label reduced to a comparable form, for fingerprinting."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _label(value) -> str:
    """A header label as we show it: whitespace collapsed, original casing kept."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def _trim(cells: list) -> list:
    """Drop trailing empty cells, keep interior positions."""
    last = -1
    for i, c in enumerate(cells):
        if _norm_label(c):
            last = i
    return cells[: last + 1]


def _human_size(n: int | None) -> str:
    if not n:
        return "0 B"
    step = 1024.0
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < step or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= step
    return f"{size:.1f} GB"


def _stat_key(path: Path) -> tuple | None:
    """Cache key that changes when the file changes. None when the file cannot be stat'd."""
    try:
        st = path.stat()
    except OSError:
        return None
    return (str(path), st.st_size, st.st_mtime_ns)


def _flatten(paths) -> list[str]:
    """Accept a string, a Path, a list, or a nested mix of those; return de-duped strings in order."""
    out: list[str] = []
    seen: set[str] = set()

    def _walk(item) -> None:
        if item is None:
            return
        if isinstance(item, (str, Path)):
            s = str(item).strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
            return
        if isinstance(item, dict):
            for v in item.values():
                _walk(v)
            return
        if isinstance(item, (list, tuple, set)):
            for v in item:
                _walk(v)
            return
        _walk(str(item))

    _walk(paths)
    return out


# --- person → tab resolution --------------------------------------------------------------------

def _name_tokens(text: str) -> list[str]:
    """Alphabetic tokens of a name or tab title, lowercased. Punctuation becomes a separator."""
    return [t for t in re.split(r"[^A-Za-z]+", str(text or "").lower()) if t]


def _person_parts(name: str) -> tuple[str, list[str]]:
    """Split a person's name into (surname token, given-name tokens).

    Handles both orders the corpus uses. "Oyler, Ken" and "Ken Oyler" both give ("oyler", ["ken"]),
    and an initial survives: "K. Oyler" gives ("oyler", ["k"]).
    """
    raw = str(name or "").strip()
    if "," in raw:
        last_part, _, first_part = raw.partition(",")
        last_toks = _name_tokens(last_part)
        return (last_toks[-1] if last_toks else ""), _name_tokens(first_part)
    toks = _name_tokens(raw)
    if not toks:
        return "", []
    return toks[-1], toks[:-1]


def _given_name_agrees(tab_tokens: list[str], surname: str, givens: list[str]) -> bool:
    """True when a tab's leftover tokens agree with the person's given name.

    Per-person labor tabs are named `Adams N` / `Adams S`, so the leftover token is often a single
    initial. An initial on either side is enough to confirm; a full token must match outright.
    """
    others = [t for t in tab_tokens if t != surname]
    if not others or not givens:
        return False
    for other in others:
        for given in givens:
            if other == given:
                return True
            if len(other) == 1 and given.startswith(other):
                return True
            if len(given) == 1 and other.startswith(given):
                return True
    return False


def _names_person(text: str, surname: str, givens: list[str]) -> str:
    """How firmly a tab's extractor notes name this person: `"full"`, `"surname"` or `""`.

    `"full"` means the given name and the surname appear together — "Nancy Adams", "Adams, Nancy",
    "Nancy J Adams", or an initial on either side ("N Adams" for Nancy, "Nancy A"). One token is allowed
    between the two names, which covers a middle name or initial.

    Adjacency is the whole point. A tab's notes often mention several people — a manager, another
    questioned employee — so a bare surname somewhere in a thousand characters of prose is not evidence
    that the tab is about that person. That looser case comes back as `"surname"`, and callers treat it
    as a hint to report, never as a match to rely on.
    """
    if not surname or not text:
        return ""
    norm = " " + re.sub(r"[^a-z]+", " ", str(text).lower()).strip() + " "
    if f" {surname} " not in norm:
        return ""
    for given in givens:
        if not given:
            continue
        # `nancy\w*` also matches a longer form; the bare initial covers notes that abbreviate it.
        head = f"(?:{re.escape(given)}\\w*|{re.escape(given[0])})"
        gap = r"(?:\w+ )?"      # an optional middle name or initial
        if (re.search(rf" {head} {gap}{re.escape(surname)} ", norm)
                or re.search(rf" {re.escape(surname)} {gap}{head} ", norm)):
            return "full"
    return "surname"


def resolve_person_tabs(tab_names, people, tab_hints=None) -> dict:
    """Match each person to the tabs that could be theirs, and say when the match is not certain.

    Two sources, best first.

    1. `tab_hints` — `{tab name: notes}` from `agents/_common.person_tab_hints`, which is what the
       extractor already recorded while reading the workbook. These notes usually name the person
       outright, so they settle cases a tab name cannot: `Adams N` and `Adams S` both carry the surname
       Adams, and only the notes say which one is Nancy.
    2. The tab NAME — surname stem, then the given name or its initial. This is the rule the agent
       prompts already describe, applied once here instead of costing each agent an LLM turn.

    Returns `{person: {"tabs": [...], "match": <how>, "ambiguous": bool}}`, with an entry for EVERY
    person asked about. `match` is one of:

      * `"surname+synthesis"`       — the tab name carried the surname and the extractor's notes named
        the person, which is the strongest signal available.
      * `"surname+synthesis+given"` — both narrowings were needed. This is a real corpus case: the notes
        for `Adams N` discuss Sheila Adams as well as Nancy, so the notes leave two tabs standing and
        the given-name initial in `Adams S` is what settles it.
      * `"surname+given"`           — the tab name carried the surname and the given name confirmed it.
      * `"surname"`                 — the tab name carried the surname and nothing narrowed it further.
      * `"synthesis"`         — NO tab is named for this person, but the extractor's notes name them
        inside a shared tab. Affiliates and contractors land here: they are questioned like anyone else
        and their rows exist, just not under their own tab.
      * `"none"`              — nothing found, so `tabs` is empty.

    A person with no tab is recorded, never dropped. They are still in scope for the exception, and the
    prompt block says so — an agent that skipped them would leave part of the dispute unanswered.

    Ambiguity is REPORTED, never resolved by guessing. Without the notes, `Adams` against
    `["Adams N", "Adams S"]` returns both tabs with `ambiguous: True`, because picking one would hand
    every downstream agent the wrong person's hours with no way to notice.
    """
    tabs = [str(t) for t in (tab_names or [])]
    tokenized = [(t, _name_tokens(t)) for t in tabs]
    hints = {str(k): str(v) for k, v in (tab_hints or {}).items() if str(k) in set(tabs)}
    out: dict[str, dict] = {}
    for person in (people or []):
        name = str(person).strip()
        if not name:
            continue
        surname, givens = _person_parts(name)
        if not surname:
            continue
        hits = [t for t, toks in tokenized if surname in toks]

        if hits:
            match = "surname"
            # The extractor's notes come first: they name the person, so they beat an initial.
            if len(hits) > 1 and givens and hints:
                named = [t for t in hits if _names_person(hints.get(t, ""), surname, givens) == "full"]
                if named:
                    hits, match = named, "surname+synthesis"
            # Then narrow on the tab name, on whatever is still standing. The two checks COMPOSE rather
            # than compete, because one tab's notes can name a second person: `Adams N`'s notes describe
            # Nancy and also discuss Sheila, so for Sheila the notes leave both Adams tabs and the `S` in
            # `Adams S` is the only thing that decides it.
            if len(hits) > 1 and givens:
                toks_of = dict(tokenized)
                narrowed = [t for t in hits if _given_name_agrees(toks_of[t], surname, givens)]
                if narrowed:
                    hits = narrowed
                    match = "surname+synthesis+given" if match == "surname+synthesis" else "surname+given"
            out[name] = {"tabs": hits, "match": match, "ambiguous": len(hits) > 1, "own_tab": True}
            continue

        # No tab is named for them. Ask the extractor's notes which shared tab holds their rows — this
        # is the affiliate / contractor case, and it is the difference between "no data" and "data, just
        # not where you looked".
        named = [t for t in tabs if _names_person(hints.get(t, ""), surname, givens) == "full"]
        if named:
            out[name] = {"tabs": named, "match": "synthesis", "ambiguous": len(named) > 1,
                         "own_tab": False}
            continue
        out[name] = {"tabs": [], "match": "none", "ambiguous": False, "own_tab": False}
    return out


# --- profiling one workbook ---------------------------------------------------------------------

def _sniff(ws) -> list[list]:
    """The first `SNIFF_ROWS` rows of a sheet, as lists of raw cell values."""
    top: list[list] = []
    try:
        for cells in ws.iter_rows(values_only=True):
            top.append(list(cells))
            if len(top) >= SNIFF_ROWS:
                break
    except Exception:      # a corrupt sheet inside an otherwise readable workbook
        pass
    return top


def _sheet_shape(ws) -> dict:
    """The header position and labels of one sheet.

    `detect_tab_type` keeps precedence: when it recognizes one of the five auditor families AND points
    at a header row, that is the answer. `choose_header_row` only runs when it declines, which is the
    case for a POB report, a staff listing or a per-person labor tab.
    """
    top = _sniff(ws)
    detected_type, detected_idx = detect_tab_type(top)
    if detected_idx is not None:
        header_row, confidence = detected_idx, f"detected:{detected_type}"
    else:
        header_row, confidence = choose_header_row(top)

    if header_row is not None and header_row < len(top):
        header = [_label(c) for c in _trim(top[header_row])]
    else:
        header = []
    widest = max((len(_trim(row)) for row in top), default=0)

    reported = getattr(ws, "max_row", None)
    return {
        "header_row": header_row,
        "confidence": confidence,
        "detected_type": detected_type,
        "header": header,
        "n_cols": len(header) or widest,
        # ws.max_row is only as good as the <dimension/> tag the writing tool left behind. Kept on disk
        # for orientation, never rendered into a prompt, never treated as a count.
        "reported_rows_unreliable": int(reported) if reported else None,
    }


def _fingerprint(shape: dict) -> tuple:
    """What makes two tabs "the same shape".

    The header row index is part of the key on purpose. In the 63-tab labor workbook the identical
    22-column header sits at row 0 on 50 tabs and at row 1 on 5 others; collapsing those into one group
    would tell an agent to start reading a row too early on five people.
    """
    labels = tuple(_norm_label(h) for h in shape["header"])
    if not labels:
        return (shape["header_row"], "positional", shape["n_cols"])
    return (shape["header_row"], labels)


def _profile_workbook(path: Path, warnings: list[str]) -> dict:
    """Open one workbook and record its usable tabs, grouped by shape. Cached on (path, size, mtime)."""
    key = _stat_key(path)
    if key is not None and key in _PROFILE_CACHE:
        return dict(_PROFILE_CACHE[key])

    wb, engine = open_workbook_any(path)
    try:
        sheets, dropped = _usable_worksheets(wb, source=path.name, warn=warnings.append)
        tabs: dict[str, dict] = {}
        groups: dict[tuple, dict] = {}
        for ws in sheets:
            name = getattr(ws, "title", "") or ""
            shape = _sheet_shape(ws)
            fp = _fingerprint(shape)
            group = groups.get(fp)
            if group is None:
                group = {"header_row": shape["header_row"], "confidence": shape["confidence"],
                         "detected_type": shape["detected_type"], "header": shape["header"],
                         "n_cols": shape["n_cols"], "tabs": []}
                groups[fp] = group
            group["tabs"].append(name)
            tabs[name] = {"shape": None, "header_row": shape["header_row"],
                          "confidence": shape["confidence"], "detected_type": shape["detected_type"],
                          "reported_rows_unreliable": shape["reported_rows_unreliable"]}
    finally:
        try:
            wb.close()
        except Exception:
            pass

    # Biggest group first, and IDs assigned AFTER the sort — so S1 is always the workbook's dominant
    # shape. Numbering in encounter order instead would name a one-tab outlier S1 and bury the 50-tab
    # group at S3, which is exactly backwards for a reader skimming the summary.
    shapes = sorted(groups.values(), key=lambda g: (-len(g["tabs"]), g["tabs"][0].lower()))
    for i, g in enumerate(shapes, 1):
        g["id"] = f"S{i}"
        g["n_tabs"] = len(g["tabs"])
        for name in g["tabs"]:
            tabs[name]["shape"] = g["id"]
    profile = {
        "path": str(path),
        "name": path.name,
        "engine": engine,                     # the reader that SUCCEEDED, not what the suffix claimed
        "size_bytes": path.stat().st_size if key else None,
        "n_sheets": len(tabs),
        "sheets": list(tabs.keys()),
        "hidden_response_tabs": list(dropped),
        "shapes": shapes,
        "tabs": tabs,
    }
    if key is not None:
        _PROFILE_CACHE[key] = profile
    return dict(profile)


def _index_directory(path: Path) -> dict:
    """Filenames, sizes and extensions of a directory. Names only — no file is opened.

    Agents do list `common_reports_dir` / `otl_information_dir` when a resolved path comes back empty,
    so the listing is worth having. Opening everything in there is not: these directories hold hundreds
    of megabytes.
    """
    files: list[dict] = []
    try:
        for child in sorted(path.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_file() or child.name.startswith("~$"):
                continue
            try:
                size = child.stat().st_size
            except OSError:
                size = None
            files.append({"name": child.name, "size_bytes": size, "ext": child.suffix.lower()})
    except OSError as exc:
        return {"path": str(path), "name": path.name, "n_files": 0, "files": [],
                "error": f"{type(exc).__name__}: {exc}"}
    return {"path": str(path), "name": path.name, "n_files": len(files), "files": files}


# --- the public build ---------------------------------------------------------------------------

def build_manifest(paths, people=None, tab_hints=None) -> dict:
    """Profile every workbook and directory in `paths`, and pre-resolve the person lookup.

    `paths` can be a string, a Path, a list, or any nesting of those — the agents' inputs arrive in all
    of those shapes. Directories are indexed by name only. Files that are not workbooks are recorded by
    name and size without being parsed.

    `tab_hints` is `{file name: {tab name: notes}}` from `agents/_common.person_tab_hints` — what the
    extractor already wrote down about each tab. It is what maps people to tabs; the reads done here are
    for STRUCTURE, which the extractor's notes do not record. Matching is looked up by file name, so a
    workbook the synthesis never mentioned simply falls back to matching on tab names.

    Unreadable workbooks are LISTED WITH THEIR ERRORS rather than dropped. That is not defensive
    padding: `SEPCO 2022 Historical POB report.xlsx` in the corpus fails on all three engines at once
    (OLE2 magic bytes, but xlrd cannot find a workbook in the compound document and both ZIP readers say
    it is not a zip — it reads as an encrypted or rights-managed package). An agent handed a manifest
    that silently omits a file will conclude the file does not exist.

    Building this is cheap: 2.03 seconds for the 5 workbooks of a real case, 137 MB and 71 sheets, in a
    run that takes 10.8 minutes.
    """
    people = [str(p).strip() for p in (people or []) if str(p).strip()]
    # Keyed by base name: the synthesis records a file name, the agents hold an absolute path.
    hints_by_file = {Path(str(k)).name.lower(): (v or {}) for k, v in (tab_hints or {}).items()}
    manifest: dict = {"workbooks": [], "unreadable": [], "directories": [], "other_files": [],
                      "warnings": [], "people": people}

    for raw in _flatten(paths):
        p = Path(raw)
        if p.is_dir():
            manifest["directories"].append(_index_directory(p))
            continue
        if not p.is_file():
            manifest["unreadable"].append({"path": str(p), "name": p.name,
                                           "errors": {"path": "not found"}, "missing_libs": []})
            continue
        if p.suffix.lower() not in _WORKBOOK_EXTS:
            try:
                size = p.stat().st_size
            except OSError:
                size = None
            manifest["other_files"].append({"path": str(p), "name": p.name, "size_bytes": size,
                                            "ext": p.suffix.lower()})
            continue
        try:
            wb_entry = _profile_workbook(p, manifest["warnings"])
        except WorkbookOpenError as exc:
            manifest["unreadable"].append({"path": exc.path, "name": Path(exc.path).name,
                                           "errors": exc.errors,
                                           "missing_libs": list(exc.missing_libs)})
            continue
        except Exception as exc:            # a reader that failed in a way it does not declare
            manifest["unreadable"].append({"path": str(p), "name": p.name,
                                           "errors": {"profile": f"{type(exc).__name__}: {exc}"},
                                           "missing_libs": []})
            continue
        if people:
            wb_entry["person_tabs"] = resolve_person_tabs(
                wb_entry["sheets"], people, tab_hints=hints_by_file.get(p.name.lower()))
        manifest["workbooks"].append(wb_entry)

    return manifest


# --- the prompt rendering -----------------------------------------------------------------------

_HEADER_NOTE = (
    "WORKBOOK MANIFEST — read straight off the files on disk before this loop started; no model was "
    "involved. Use it so you do not spend turns finding out what a file holds. It is a PRIOR, not a "
    "guarantee: if your own read shows a different header row, trust your read. Header row indexes are "
    "0-based, so header row 1 means skip one row first. Tabs are grouped by shape — same header row and "
    "same column labels. A shape is written W1S1: the workbook tag, then the shape. Every workbook "
    "numbers its own shapes from S1, and in code manifest() reports the shape part alone (S1)."
)

_BANNER_NOTE = (
    "NOTE: a row whose non-empty cells all hold the SAME value is a merged banner, not data — and they "
    "recur inside the data too, not just at the top (a repeated region name partway down a POB tab). "
    "Skip all-equal rows when you loop."
)


def _shape_line(group: dict, prefix: str, seen_headers: dict[tuple, str]) -> list[str]:
    """One shape rendered as one or two lines.

    `prefix` is the workbook's tag (`W1`), so shape IDs read `W1S1` and cannot be confused across
    workbooks — every workbook numbers its own shapes from S1.

    `seen_headers` maps an already-printed header to the shape that printed it. Two shapes often share
    the same labels and differ only in where the header sits — the 22-column labor header is at row 0 on
    50 tabs and row 1 on 5 more — so the second one says "same columns as W1S1" instead of repeating 260
    characters. On the 63-tab workbook that alone saves about 640 characters of prompt.
    """
    hrow = group.get("header_row")
    if hrow is None:
        where = f"NO header row — read by column index, about {group.get('n_cols', 0)} columns wide"
    else:
        where = f"header row {hrow}, {group.get('n_cols', 0)} columns"
    dtype = group.get("detected_type")
    if dtype:
        where += f", looks like {dtype}"

    tabs = list(group.get("tabs") or [])
    shown = tabs[:_MAX_TABS_SHOWN]
    tab_text = ", ".join(shown)
    if len(tabs) > len(shown):
        tab_text += f", …(+{len(tabs) - len(shown)} more tabs, same shape)"

    sid = f"{prefix}{group.get('id', 'S?')}"
    lines = [f"   {sid} {where} — {len(tabs)} tab(s): {tab_text}"]
    header = [h for h in (group.get("header") or []) if h]
    if header:
        key = tuple(_norm_label(h) for h in header)
        first = seen_headers.get(key)
        if first:
            lines.append(f"      columns: same as {first}")
        else:
            seen_headers[key] = sid
            cols = " | ".join(header)
            if len(cols) > _MAX_COL_CHARS:
                cols = cols[: _MAX_COL_CHARS - 1] + "…"
            lines.append(f"      columns: {cols}")
    return lines


def _person_lines(manifest: dict) -> list[str]:
    """The pre-resolved person → tab lookup, with ambiguity called out.

    Shapes are named with the same workbook tag the shape lines use (`W1S1`), so a person's tab points
    at exactly one shape line above.

    Every person asked about gets a line, including the ones with no tab. A missing tab is a fact about
    the FILE, not about whether the person is disputed, and an agent that read "not found" as "not my
    problem" would leave part of the exception unanswered.
    """
    lines: list[str] = []
    for i, wb in enumerate(manifest.get("workbooks") or [], 1):
        found = wb.get("person_tabs") or {}
        name = wb.get("name", "")
        for person, hit in found.items():
            tabs = list(hit.get("tabs") or [])
            match = hit.get("match", "")
            if not tabs:
                lines.append(f"   {person} → {name}: NO tab of their own and no tab's notes name them. "
                             f"Still in scope — often an affiliate or a contractor whose rows sit in a "
                             f"shared tab or in another file. Search the shared tabs by name before "
                             f"concluding anything.")
                continue
            shown = tabs[:_MAX_PERSON_TABS]
            detail = []
            for tab in shown:
                info = (wb.get("tabs") or {}).get(tab) or {}
                detail.append(f"{tab} (W{i}{info.get('shape', 'S?')})")
            text = ", ".join(detail)
            if len(tabs) > len(shown):
                text += f", …(+{len(tabs) - len(shown)} more)"
            if match == "synthesis":
                # No tab of their own, but the extractor's notes put them inside one.
                lines.append(f"   {person} → {name}: no tab of their own; named in the notes for "
                             f"{text} — read their rows there")
                continue
            if hit.get("ambiguous"):
                lines.append(f"   {person} → {name}: {text} — AMBIGUOUS, confirm before using")
                continue
            how = (" (the extractor's notes name them on this tab)"
                   if match.startswith("surname+synthesis") else "")
            lines.append(f"   {person} → {name}: {text}{how}")
    return lines


def _anyone_without_a_tab(manifest: dict) -> bool:
    """True when some person was asked about and no tab in any workbook was matched to them."""
    asked: dict[str, bool] = {}
    for wb in manifest.get("workbooks") or []:
        for person, hit in (wb.get("person_tabs") or {}).items():
            asked[person] = asked.get(person, False) or bool(hit.get("tabs"))
    return any(not found for found in asked.values())


def manifest_block(manifest: dict, people=None, max_groups: int = 8) -> str:
    """Render the manifest as a short block to append to an agent's `task` string.

    Append this to `task`, never to the system prompt. `task` is part of the seed `HumanMessage`, and
    the ReAct buffer is append-only, so the block is written into the prompt cache once per agent per
    case and read back at a tenth of the price on every later step. The system prompt is shared across
    cases, so a per-case block there would spoil a cross-case cached prefix.

    Bounded on purpose. At most `max_groups` shapes are shown per workbook, and what is left out is
    stated on its own line — the `_cap_items` habit from `agents/_common.py`.

    Measured on the corpus: the 63-tab exhibit renders to 3,192 characters, about 916 tokens with the
    person lookup included. Listing those 63 sheets one by one, each with its own header row and column
    labels, would be 24,247 characters, about 6,932 tokens. So grouping by shape is roughly 8x. The
    biggest block in the corpus is the `job_title` pair (staff listing + Jobs_Catalogue, 15 shapes
    capped to 8) at 4,556 characters, about 1,306 tokens.
    """
    if not manifest:
        return ""
    workbooks = manifest.get("workbooks") or []
    unreadable = manifest.get("unreadable") or []
    directories = manifest.get("directories") or []
    if not (workbooks or unreadable or directories):
        return ""

    out: list[str] = ["", _HEADER_NOTE, ""]

    for i, wb in enumerate(workbooks, 1):
        hidden = wb.get("hidden_response_tabs") or []
        head = (f"[W{i}] {wb.get('name', '')} — {wb.get('engine', '?')}, "
                f"{_human_size(wb.get('size_bytes'))}, {wb.get('n_sheets', 0)} usable sheet(s)")
        if hidden:
            head += f", {len(hidden)} operator-response tab(s) hidden"
        out.append(head)
        out.append(f"  path: {wb.get('path', '')}")
        shapes = list(wb.get("shapes") or [])
        shown = shapes[:max_groups]
        out.append(f"  {wb.get('n_sheets', 0)} tab(s) in {len(shapes)} shape(s):")
        seen_headers: dict[tuple, str] = {}   # per workbook, so the "same as" back-reference resolves
        for group in shown:
            out.extend(_shape_line(group, f"W{i}", seen_headers))
        if len(shapes) > len(shown):
            dropped_tabs = sum(len(g.get("tabs") or []) for g in shapes[len(shown):])
            out.append(f"   …(+{len(shapes) - len(shown)} more shapes covering {dropped_tabs} tab(s) "
                       f"not listed here — call manifest() in code to see them)")

    for i, d in enumerate(directories, 1):
        files = list(d.get("files") or [])
        shown = [f.get("name", "") for f in files[:_MAX_DIR_FILES]]
        text = ", ".join(shown) if shown else "(empty)"
        if len(files) > len(shown):
            text += f", …(+{len(files) - len(shown)} more files)"
        out.append(f"[D{i}] {d.get('name', '')}/ — {d.get('n_files', 0)} file(s): {text}")
        if d.get("error"):
            out.append(f"  could not be listed: {d['error']}")

    for f in manifest.get("other_files") or []:
        out.append(f"[F] {f.get('name', '')} — {_human_size(f.get('size_bytes'))}, not a workbook "
                   f"(not profiled)")

    for u in unreadable:
        errs = "; ".join(f"{eng}: {msg}" for eng, msg in (u.get("errors") or {}).items())
        out.append(f"UNREADABLE: {u.get('name', '')} — {errs or 'unknown error'}. The file IS there; no "
                   f"reader could open it. Do not treat it as missing.")

    person_lines = _person_lines(manifest)
    if person_lines:
        out.append("PERSON TABS (from the extractor's own tab notes first, then the tab name — surname, "
                   "then given name):")
        out.extend(person_lines)
        # Only spend these lines when someone actually has no tab. This is the gating rule, and it is
        # stated once for the whole block rather than repeated on every person's line.
        if _anyone_without_a_tab(manifest):
            out.append("   A person with no tab of their own is NOT out of scope and NOT unresolvable. "
                       "Analyse them with the same rules and the same gating as everyone else: look for "
                       "their rows in the shared tabs and in the other files you were given. Only if "
                       "that search really comes up empty do you flag them for review, with the reason. "
                       "Never leave a disputed person out of your result object.")

    out.append(_BANNER_NOTE)
    out.append("In code, manifest() returns all of this and find_tab(person) returns a person's "
               "candidate tabs with their shapes. Both read a staged file, so they cost nothing.")
    return "\n".join(out) + "\n"


def manifest_json(manifest: dict) -> str:
    """The manifest as the JSON text staged into the code sandbox."""
    return json.dumps(manifest or {}, default=str)


__all__ = ["build_manifest", "manifest_block", "manifest_json", "resolve_person_tabs"]
