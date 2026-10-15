"""Read the big corpus workbooks once, then ask one med-tier model which names are our people.

Three specialists hunt for people inside very large workbooks: `location` (the POB report and the
helicopter manifest) and `job_title` / `supervision` (the consolidated staff listing). They used to do
the hunting themselves, from inside the ReAct loop: the model wrote a surname screen, streamed the whole
workbook, read the result, wrote the next screen, and streamed it again — once per person. Every one of
those steps re-sent the whole message buffer at the high tier. A stored trace shows the work being done
by hand (`.otl_scratch/GreatWhite_2023_0033/location/result.json`, the `name_screens` field): four
surname screens over a 111,605-row helicopter manifest.

A workbook's name column is a fact on disk, so reading it needs no model. What needs judgement is only
whether `JOHN HARRISON` is the `Harrison, John` we were asked about. So this module splits the work three
ways, and only the middle phase calls a model:

  * `harvest_names` — deterministic. One traversal per workbook, visiting every usable tab, collecting
    the distinct values of each tab's subject-name column and classifying every column it recognizes.
  * `select_names` — one med-tier call per chunk of the FULL distinct-name list, asking which values
    belong to the disputed people. Chunked so every value is seen, never truncated.
  * `pull_rows` — deterministic again. One more traversal, buffering the rows of the selected spellings,
    each row keyed by its column label.

The split is the same one `otl/ledger_people.py` makes between `harvest` and `extract_people`, and for
the same reason: the read should be reproducible, and only the name-identity step is a judgement call.

Two things about this corpus drive the design.

**Column names differ, so they are returned rather than assumed.** The 2021 staff listing is three
schemas in one workbook: six tabs name the subject `Employee Reporting Name` and point at a manager with
`Manager ID`, six name it `Worker` and point at an org with `Supervisory Organization - ID`, and six name
it `Employee Name` and point at a supervisor with `Supervisor Number`. `Job Profile` exists on the 2022
staff tab and is absent from the 2023 and 2024 tabs. So every column this pass used, and every other role
it recognized, comes back as `role -> {label: index}` per tab, and the prompt block states the roles that
are NOT present instead of leaving the agent to discover it by reading.

**"Contains the word name" is not a usable rule.** `Headcount Name`, `Company Code Name`,
`Personnel Area Name`, `Business area name`, `Employee Type`, `Employee Group` and `Employee Subgroup`
all contain `name`, `employee` or `worker`, and none of them is a person. The 2025 staff workbook even
has a `Sheet1` whose detected header row is really data (`['Caleb Wells', 'January', 'Subsea Engineering
Manager', ...]`), so a "contains manager" rule would read a job title as a column role. Classification is
therefore an allow-list of exact normalized labels, and a tab with no subject column — or no header row —
is skipped outright and LISTED as skipped.

Nothing here adds an Excel reader. `extractor/excel_bounded.py` already dispatches the right reader by
file content (`open_workbook_any`) and hides operator-response tabs (`_usable_worksheets`), which is what
keeps this agreeing with the workbook manifest and the code sandbox. Tab shapes come from the manifest the
calling node already built, so no file is sniffed twice.

Nothing is truncated in the read: no row cap, no cap on distinct names. That is the
`ledger_people.harvest_person_values` rule, and the reason is the same — a value left behind is a person
left out of the whole rebuttal.

`OTL_PERSON_INDEX=0` turns the whole thing off: an empty index, no model call, and the specialists fall
back to reading raw, which they can always do anyway.
"""
from __future__ import annotations

import contextvars
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel, Field

from shared.excel_bounded import (WorkbookOpenError, _usable_worksheets,  # noqa: PLC2701
                                     open_workbook_any)

from . import llm_adapter as llm
from .prompts import PERSON_SELECT_SYSTEM
from .workbook_manifest import _flatten, _label, _norm_label, _person_parts  # noqa: PLC2701

# Values that mean "no name here" — the same set `ledger_people` skips.
_PLACEHOLDERS = {"", "-", "--", "n/a", "na", "none", "null", "not assigned", "unassigned",
                 "not applicable", "#n/a", "0"}

# How many characters of harvested names ride in one med-tier request. 1,000,000 is roughly 250k tokens
# at the planning estimate of four characters per token. Actual tokenizer usage varies, and the prompt
# envelope adds overhead; lower it via OTL_NAME_SELECT_CHUNK_CHARS when the selected model or context
# budget needs smaller requests.
_CHUNK_CHARS = 1_000_000

# How many chunk calls run at once inside ONE process. `otl/graph.py` is sequential — one specialist node
# at a time — so this is the whole per-process ceiling and no in-process lock is needed. It does NOT span
# cases: `run_batch.py` forks a subprocess per case, so the batch divides this budget across `--jobs` by
# setting the env var in each child (see run_batch.py).
_SELECT_JOBS = 4

# Rows buffered per selected spelling. The busiest name observed in the corpus has 237 helicopter rows,
# so 400 clears real data; truncation past it is recorded with the true count, never silent.
_MAX_ROWS_PER_NAME = 400

# How many near-miss values are kept for a person nothing was selected for. This is what made the 0033
# trace useful — "199 WILLIAMS-surname values and none was this person" is a finding; "not found" is not.
_MAX_POOL_KEPT = 200

# Prompt-rendering bounds. Each one drops WHOLE items and says how many it dropped — the `_cap_items`
# habit from `agents/_common.py`, because silent truncation reads as "that is everything".
_MAX_ROWS_SHOWN = 3          # rows shown per selected spelling
_MAX_ATTR_CHARS = 170        # chars of one row's attribute line
_MAX_ROLE_CHARS = 300        # chars of the column-role line per tab
_MAX_TABS_SHOWN = 6          # tabs listed per workbook
_MAX_SKIPPED_SHOWN = 8       # skipped tabs listed per source
_MAX_BLOCK_CHARS = 7000      # whole block, as a last resort

# One case is one process, so all three specialists in a case share this — `job_title` and `supervision`
# pay for the staff listing once. `run_batch.py` forks per case, so nothing leaks between cases.
_INDEX_CACHE: dict[tuple, dict] = {}

_SOURCE_TITLES = {"pob": "POB REPORT", "heli": "HELICOPTER MANIFEST", "staff": "STAFF LISTING"}


def _warn(message: str) -> None:
    print(message, file=sys.stderr)


def enabled() -> bool:
    """False when `OTL_PERSON_INDEX` is set to an off value. Off restores the old behaviour exactly."""
    return str(os.getenv("OTL_PERSON_INDEX", "1")).strip().lower() not in ("0", "false", "no", "off")


def _chunk_chars() -> int:
    try:
        n = int(os.getenv("OTL_NAME_SELECT_CHUNK_CHARS", str(_CHUNK_CHARS)))
    except (TypeError, ValueError):
        return _CHUNK_CHARS
    return max(2000, n)


def _select_jobs() -> int:
    try:
        n = int(os.getenv("OTL_NAME_SELECT_JOBS", str(_SELECT_JOBS)))
    except (TypeError, ValueError):
        return _SELECT_JOBS
    return max(1, n)


def _clean(value) -> str:
    """A cell value as we keep it: whitespace collapsed, original casing."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def _norm(value) -> str:
    """A cell value reduced to a comparable form: whitespace collapsed, lowercased."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _usable(value) -> str:
    """A cell value, or `""` when it is blank or one of the "no name here" placeholders."""
    text = _clean(value)
    return "" if _norm(text) in _PLACEHOLDERS else text


# --- which column is which ------------------------------------------------------------------------

# The subject-name column: whose row is this. Precedence in this order, and every label is matched
# EXACTLY after normalization. Exactness is the point — see the module docstring on `Headcount Name`.
_SUBJECT_LABELS = (
    "employee reporting name", "worker", "employee name", "name", "full name",
    "personnel name", "passenger name", "person name", "crew member", "crew member name",
    "pax name", "traveler name", "traveller name", "trip item name", "tripitemname",
)

# The other shape the corpus uses: two columns that only mean a person together. A lone `First Name` is
# not a subject column, so only a PAIR counts.
_FIRST_LABELS = ("first name", "first", "given name")
_LAST_LABELS = ("last name", "last", "surname")

# Every other role worth naming, as `role -> exact normalized labels`. First role to claim a label wins,
# so the order here is the tie-break; there are no overlaps today.
_COLUMN_ROLES: dict[str, tuple[str, ...]] = {
    # The person's own key, and the pointer to whoever they report to. `supervision` walks these, and it
    # is exactly here that the 2021 workbook's three tab families disagree.
    "person_key": ("employee id", "employee number"),
    "reports_to": ("manager id", "supervisor number", "supervisory organization - id"),
    "manager_name": ("manager name", "supervisor name"),
    "position": ("position", "position text", "position id"),
    "job_profile": ("job profile", "us aap job", "job family group", "skillpool job",
                    "skill pool group"),
    # Org units, deliberately NOT people. Named so they are visibly classified rather than reported as
    # unclassified name-ish labels an agent might mistake for a second name column.
    "org": ("sup org name", "supervisory organization", "organizational unit", "business",
            "master cost center name", "cost center"),
    "period": ("month", "start date cost center", "date to position"),
    "location": ("location", "pob location", "current location", "location code", "area", "region",
                 "facility", "installation", "asset", "asset name", "asset description"),
    "company": ("company", "employer", "organization", "vendor"),
    "dates": ("arrival date", "departure date", "arrival", "departure", "start date", "end date",
              "onboard date", "boarding date", "check in", "check out", "start time", "end time",
              "flight date", "trip date", "reservation date"),
    "trip": ("asset", "asset name", "asset description", "service port", "origin", "origin location",
             "from", "from location", "destination", "destination location", "to", "to location",
             "trip id", "flight id", "flight number", "reservation id", "leg",
             "cost code description"),
    "essential": ("essential",),
}

# Roles whose values are person names. `_is_id_label` is applied to these, and only these.
_NAME_ROLES = ("subject", "manager_name")

# A label containing one of these words looks like it might hold a person. When one of them ends up
# with no role, it is COUNTED and reported rather than dropped, so the agent knows to check `manifest()`.
_NAMEISH_WORDS = ("name", "worker", "employee", "personnel", "passenger")

# An id-ish label is never a name column. Lifted from `ledger_people._is_employee_column`, which keeps
# its own copy because it also needs the line-item-text fallback that has no place here.
_ID_WORDS = ("id", "ids", "no", "no.", "num", "number", "code", "key", "#")


def _is_id_label(label: str) -> bool:
    """True when a label names an identifier rather than a person: `Employee ID`, `Employee Number`.

    Exact allow-list matching already keeps `Employee ID` out of `_SUBJECT_LABELS`, so on today's
    corpus this guard never fires. It is what keeps that true if the allow-list ever grows: a label
    added by hand cannot quietly turn an id column into a name column.
    """
    tokens = re.split(r"[^a-z0-9#.]+", _norm(label))
    return any(t in _ID_WORDS for t in tokens if t)


def _first_index(header: list, wanted: tuple[str, ...]) -> tuple[str, int] | None:
    """The first column whose normalized label is one of `wanted`, as `(label as written, index)`."""
    for i, raw in enumerate(header or []):
        if _norm_label(raw) in wanted:
            return _label(raw), i
    return None


def _subject_columns(header: list) -> dict | None:
    """The subject-name column(s) of one header, or None when the tab has none.

    Two kinds:

      * `combined` — one column holding the whole name (`Employee Reporting Name`, `Worker`, `NAME`).
        A tab carrying several of these gets all of them, ordered by the precedence in
        `_SUBJECT_LABELS`, and the first is the primary the prompt names.
      * `first_last` — a `First Name` + `Last Name` PAIR, joined per row as `"LAST, FIRST"`. This is
        what every POB report in the corpus uses.

    A combined column wins over a pair when both are present, because it holds the whole name as the
    file itself writes it. The pair is still recorded, under the `name_parts` role, so the agent can see
    it without re-reading the sheet.
    """
    found: list[tuple[int, str, int]] = []
    for rank, wanted in enumerate(_SUBJECT_LABELS):
        for i, raw in enumerate(header or []):
            label = _label(raw)
            if _norm_label(raw) == wanted and not _is_id_label(label):
                found.append((rank, label, i))
                break
    if found:
        found.sort(key=lambda t: t[0])
        return {"kind": "combined", "labels": [f[1] for f in found], "indices": [f[2] for f in found]}

    first = _first_index(header, _FIRST_LABELS)
    last = _first_index(header, _LAST_LABELS)
    if first and last:
        return {"kind": "first_last", "labels": [last[0], first[0]], "indices": [last[1], first[1]]}
    return None


def classify_columns(header: list) -> dict:
    """Read one header row into `{"subject", "columns", "unclassified_nameish"}`.

    `columns` is `role -> {label as written: column index}` and holds an entry for every role in
    `_COLUMN_ROLES`, empty when the tab does not carry it. Keeping the empty ones is what lets the prompt
    block say "not present: job_profile" instead of leaving the agent to find that out by reading.

    `unclassified_nameish` lists the labels that look like they might hold a person and got no role. They
    are reported, never used — see the module docstring on `Headcount Name`.
    """
    header = list(header or [])
    subject = _subject_columns(header)
    claimed: set[int] = set(subject["indices"]) if subject else set()

    columns: dict[str, dict[str, int]] = {role: {} for role in _COLUMN_ROLES}
    for role, wanted in _COLUMN_ROLES.items():
        for i, raw in enumerate(header):
            if i in claimed or _norm_label(raw) not in wanted:
                continue
            label = _label(raw)
            if role in _NAME_ROLES and _is_id_label(label):
                continue
            columns[role].setdefault(label, i)
            claimed.add(i)

    # The first/last pair, kept for reference when a combined column already won the subject slot.
    if subject and subject["kind"] == "combined":
        parts: dict[str, int] = {}
        for pair in (_first_index(header, _FIRST_LABELS), _first_index(header, _LAST_LABELS)):
            if pair and pair[1] not in claimed:
                parts[pair[0]] = pair[1]
        if parts:
            columns["name_parts"] = parts

    nameish = []
    for i, raw in enumerate(header):
        if i in claimed:
            continue
        norm = _norm_label(raw)
        if norm and any(word in norm for word in _NAMEISH_WORDS):
            nameish.append(_label(raw))
    return {"subject": subject, "columns": columns, "unclassified_nameish": nameish}


# --- phase 1: harvest the names off disk -----------------------------------------------------------

def _is_banner(cells: list) -> bool:
    """True for a merged banner row — every non-empty cell holding the same value, two or more of them.

    These recur PARTWAY DOWN a POB tab, not just at the top (a region name repeated across the row), and
    the workbook manifest already warns agents about them. Skipping them here keeps `Personnel On Board`
    out of the name universe we pay a model to read.
    """
    values = {_norm(c) for c in cells if _norm(c)}
    return len(values) == 1 and sum(1 for c in cells if _norm(c)) > 1


def _subject_values(cells: list, subject: dict) -> list[str]:
    """The name value(s) one row contributes, as the file writes them.

    A `first_last` tab is joined `"LAST, FIRST"`, and degrades gracefully when only one half is filled.
    A tab with several combined columns contributes one value per column.
    """
    def _cell(index: int) -> str:
        return _usable(cells[index]) if 0 <= index < len(cells) else ""

    if subject.get("kind") == "first_last":
        last, first = (subject["indices"] + [None, None])[:2]
        last_v = _cell(last) if last is not None else ""
        first_v = _cell(first) if first is not None else ""
        if last_v and first_v:
            return [f"{last_v}, {first_v}"]
        return [v for v in (last_v, first_v) if v]
    return [v for v in (_cell(i) for i in subject.get("indices") or []) if v]


def _entries_for_workbook(wb_entry: dict, source_role: str = "") -> tuple[list[dict], list[dict]]:
    """The tabs of one manifest workbook worth harvesting, and the ones skipped with the reason.

    One entry per `(tab, subject column set)`. Two hard guards, both drawn from the corpus:

      * No header row → skipped. Roles cannot be assigned positionally, and guessing them is exactly the
        failure this module exists to remove. The agent can still read such a tab raw.
      * No subject column → skipped. That drops the POB `GOM` / `UNDEFINED` / `Summaries` tabs, the 2021
        `Summary` tab, and the 2025 junk `Sheet1` whose "header" is really a person's name.
    """
    entries: list[dict] = []
    skipped: list[dict] = []
    name = wb_entry.get("name", "")
    path = wb_entry.get("path", "")
    for shape in wb_entry.get("shapes") or []:
        tabs = list(shape.get("tabs") or [])
        header_row = shape.get("header_row")
        header = list(shape.get("header") or [])
        if header_row is None or not header:
            for tab in tabs:
                skipped.append({"name": name, "tab": tab, "reason": "no header row"})
            continue
        classified = classify_columns(header)
        if not classified["subject"]:
            for tab in tabs:
                skipped.append({"name": name, "tab": tab, "reason": "no name column"})
            continue
        for tab in tabs:
            entries.append({
                "path": path, "name": name, "tab": tab, "header_row": header_row,
                "header": header, "shape": shape.get("id", ""),
                "subject": classified["subject"], "columns": classified["columns"],
                "unclassified_nameish": list(classified["unclassified_nameish"]),
                "source_role": source_role, "rows": 0, "values": [], "row_data": {},
                "row_counts": {}, "staff_rows": [],
            })
    return entries, skipped


def _read_values(path: str, entries: list[dict], warn) -> None:
    """Fill `values` and `rows` on every entry of ONE workbook, in one traversal of that workbook.

    "Once" means one open and one pass over the sheets we need — not one pass per person, and not one
    open per tab. There is no row cap and no cap on distinct values, on purpose.
    """
    by_tab: dict[str, list[dict]] = {}
    for entry in entries:
        by_tab.setdefault(entry["tab"], []).append(entry)
    file = Path(path)
    try:
        wb, _engine = open_workbook_any(file)
    except Exception as exc:
        warn(f"WARN: could not open {file.name} to index names — {type(exc).__name__}: {exc}")
        return
    try:
        sheets, _dropped = _usable_worksheets(wb, source=file.name, warn=warn)
        for ws in sheets:
            title = getattr(ws, "title", "") or ""
            targets = by_tab.get(title)
            if not targets:
                continue
            seen: dict[int, set[str]] = {id(e): set() for e in targets}
            first_data_row = min(int(e["header_row"]) for e in targets) + 1
            try:
                for i, row in enumerate(ws.iter_rows(values_only=True)):
                    if i < first_data_row:
                        continue
                    cells = list(row)
                    if _is_banner(cells):
                        continue
                    for entry in targets:
                        if i <= int(entry["header_row"]):
                            continue
                        entry["rows"] += 1
                        if entry.get("source_role") == "staff":
                            def _role(role: str, *, exclude_id: bool = False) -> str:
                                for label, column in ((entry.get("columns") or {}).get(role) or {}).items():
                                    if exclude_id and "id" in label.casefold():
                                        continue
                                    value = _clean(cells[column]) if column < len(cells) else ""
                                    if value:
                                        return value
                                return ""

                            names = _subject_values(cells, entry["subject"])
                            name = names[0] if names else ""
                            if name:
                                cost_code = ""
                                cost_name = ""
                                for column, label in enumerate(entry.get("header") or []):
                                    value = _clean(cells[column]) if column < len(cells) else ""
                                    lower = str(label or "").casefold()
                                    if not value or "cost center" not in lower:
                                        continue
                                    if "id" in lower or "key" in lower:
                                        cost_code = cost_code or value
                                    else:
                                        cost_name = cost_name or value
                                entry["staff_rows"].append((
                                    name, _role("person_key"), _role("reports_to"),
                                    _role("manager_name"), _role("position", exclude_id=True),
                                    _role("job_profile"), _role("period"), cost_code, cost_name,
                                    title, i + 1,
                                ))
                        for value in _subject_values(cells, entry["subject"]):
                            key = _norm(value)
                            if key in seen[id(entry)]:
                                continue
                            seen[id(entry)].add(key)
                            entry["values"].append(value)
            except Exception as exc:          # a corrupt sheet inside a readable workbook
                warn(f"WARN: indexing names from {file.name}!{title} stopped early — "
                     f"{type(exc).__name__}: {exc}")
    finally:
        try:
            wb.close()
        except Exception:
            pass


def harvest_names(manifest: dict, paths, warn=None, source_role: str = "") -> tuple[list[dict], list[dict]]:
    """Every name-bearing tab of `paths`, with its distinct name values and its column roles.

    Deterministic — no model is involved. The tab shapes come from `manifest`, which the calling node
    already built, so no file is sniffed twice. Returns `(entries, skipped_tabs)`.
    """
    warn = warn or _warn
    wanted = {str(Path(p)) for p in _flatten(paths)}
    entries: list[dict] = []
    skipped: list[dict] = []
    for wb_entry in (manifest or {}).get("workbooks") or []:
        if str(Path(wb_entry.get("path", ""))) not in wanted:
            continue
        got, missed = _entries_for_workbook(wb_entry, source_role)
        skipped.extend(missed)
        if got:
            _read_values(wb_entry.get("path", ""), got, warn)
            entries.extend(got)
    return entries, skipped


def _attach_selected_staff_rows(entries: list[dict], selected: dict) -> None:
    wanted: dict[str, str] = {}
    for matches in (selected or {}).values():
        for match in matches:
            wanted[_norm(match["name_in_file"])] = match["name_in_file"]
    for entry in entries:
        for row in entry.get("staff_rows") or []:
            name = wanted.get(_norm(row[0]))
            if not name:
                continue
            entry["row_counts"][name] = entry["row_counts"].get(name, 0) + 1
            item = {"_tab": row[9], "_row": row[10] - 1}
            subject_labels = (entry.get("subject") or {}).get("labels") or []
            if subject_labels:
                item[subject_labels[0]] = row[0]
            values_by_role = {
                "person_key": row[1], "reports_to": row[2], "manager_name": row[3],
                "position": row[4], "job_profile": row[5], "period": row[6],
            }
            for role, value in values_by_role.items():
                labels = list(((entry.get("columns") or {}).get(role) or {}).keys())
                for label in labels:
                    if role == "position" and "id" in label.casefold():
                        continue
                    item[label] = value
                    break
            for label in entry.get("header") or []:
                lower = str(label or "").casefold()
                if "cost center" not in lower:
                    continue
                item[str(label)] = row[7] if ("id" in lower or "key" in lower) else row[8]
            entry["row_data"].setdefault(name, []).append(item)


# --- phase 2: which of those names are our people --------------------------------------------------

class _Match(BaseModel):
    """One value from the file judged to be one of the disputed people."""
    person: str = Field(
        description="The disputed person this value belongs to, copied EXACTLY as it appears in the "
                    "list of people you were given.")
    name_in_file: str = Field(
        description="The value from the supplied list, copied EXACTLY and completely as it is written "
                    "there. Never reformat it, never correct its spelling, never invent a value that is "
                    "not in the list.")
    confidence: str = Field(
        default="",
        description="high when the surname and the given name both agree; medium when the given name is "
                    "an initial, a nickname or a shortened form; low when only the surname agrees and "
                    "you are offering the value as a candidate to check.")
    why: str = Field(
        default="", description="One short clause saying what agrees and what does not, e.g. "
                                "'surname and given name match' or 'KEN is short for KENNETH'.")


class _NameSelection(BaseModel):
    """What the med model returns for one chunk of values."""
    matches: list[_Match] = Field(
        default_factory=list,
        description="One entry per (person, value) pair you judge to be the same human being. A person "
                    "may match several values, and most values match nobody. Return an EMPTY list "
                    "rather than guessing when no value in this chunk plausibly names any of the people.")


def _chunks(values: list[str], budget: int) -> list[list[str]]:
    """Split the values into groups that each fit `budget` characters. Every value lands in one group."""
    out: list[list[str]] = []
    current: list[str] = []
    used = 0
    for value in values:
        size = len(value) + 1
        if current and used + size > budget:
            out.append(current)
            current, used = [], 0
        current.append(value)
        used += size
    if current:
        out.append(current)
    return out


def _shape_phrase(entries: list[dict]) -> str:
    """How the values are written, in words, for the selection prompt.

    The model cannot judge `PARK, SUSAN` against `Susan Park` without being told that the file holds a
    surname first. The corpus has three shapes in play — two columns joined, one combined column, and one
    workbook using three different combined column names across its tabs — so this is stated per source.
    """
    kinds = {e["subject"].get("kind") for e in entries}
    labels: list[str] = []
    for entry in entries:
        for label in entry["subject"].get("labels") or []:
            if label not in labels:
                labels.append(label)
    quoted = ", ".join(f"`{lbl}`" for lbl in labels[:6])
    if len(labels) > 6:
        quoted += f", and {len(labels) - 6} more"
    if kinds == {"first_last"}:
        return (f"They come from separate {quoted} columns, joined here as \"LAST, FIRST\" — so the "
                f"surname comes first, before the comma.")
    if "first_last" in kinds:
        return (f"They come from the {quoted} columns of several tabs. Some tabs hold the whole name in "
                f"one column; others hold a first name and a last name separately, joined here as "
                f"\"LAST, FIRST\". So some values put the surname first and others put it last.")
    return (f"They come from the {quoted} column(s). The file is not consistent about word order: a "
            f"value may be written \"Last, First\", \"First Last\", or in all capitals.")


def _select_prompt(chunk: list[str], people: list[str], source_role: str, shape: str) -> str:
    listing = "\n".join(f"- {v}" for v in chunk)
    who = "\n".join(f"- {p}" for p in people)
    where = {"pob": "a personnel-on-board (POB) report",
             "heli": "a helicopter passenger manifest",
             "staff": "a consolidated staff listing"}.get(source_role, "a corpus workbook")
    return (
        f"These are the distinct name values held by {where}. {shape}\n\n"
        f"{listing}\n\n"
        "These are the people under dispute in an audit exception. Their names were written by an "
        "auditor, so they may be spelled differently from the file:\n\n"
        f"{who}\n\n"
        "Which of the values above name these people? Judge each person against every value. Accept a "
        "value when the surname agrees and the given name agrees or is a recognisable form of it — an "
        "initial (K. for Kenneth), a short form (Ken for Kenneth), a nickname (Bob for Robert), a "
        "worker name instead of a legal name, a married or hyphenated surname, or a generational suffix "
        "(Jr., III). Offer a surname-only agreement as a low-confidence candidate. Do NOT accept a value "
        "because the surname alone is close in spelling — PRICHARD is not PRITCHARD. Copy each value "
        "exactly as it is written above; a value that is not in the list will be discarded.")


def _resolve_person(raw: str, people: list[str]) -> str:
    """Map a person name the model echoed back onto the exact name we asked about, or `""`.

    Exact first. Then a unique surname agreement, which covers the model reformatting `Harrison, John`
    as `John Harrison`. An ambiguous surname is refused rather than guessed — two Adamses are two people,
    and picking one would put the wrong person's rows in the rebuttal with no way to notice.
    """
    key = _norm(raw)
    if not key:
        return ""
    by_norm = {_norm(p): p for p in people}
    if key in by_norm:
        return by_norm[key]
    surname, givens = _person_parts(raw)
    if not surname:
        return ""
    hits = []
    for person in people:
        p_surname, p_givens = _person_parts(person)
        if p_surname != surname:
            continue
        if givens and p_givens and not (set(givens) & set(p_givens)):
            continue
        hits.append(person)
    return hits[0] if len(hits) == 1 else ""


def _deterministic_name_matches(person: str, universe: list[str]) -> list[str]:
    """High-precision, order-independent matches resolved before any selector call."""
    surname, givens = _person_parts(person)
    if not surname or not givens:
        return []
    first = givens[0]
    matches: list[str] = []
    for value in universe:
        other_surname, other_givens = _person_parts(value)
        if other_surname != surname or not other_givens:
            continue
        other_first = other_givens[0]
        if first == other_first or (len(first) == 1 and other_first.startswith(first)) or (
            len(other_first) == 1 and first.startswith(other_first)
        ):
            matches.append(value)
    # Multiple source spellings are allowed only when their given-name roots agree. This covers middle
    # names/initials and first-last reversal without admitting same-surname different people.
    roots = {(_person_parts(value)[1] or [""])[0] for value in matches}
    return matches if len(roots) <= 1 else []


def select_names(node: str, source_role: str, entries: list[dict], people: list[str],
                 warn=None) -> tuple[dict, int]:
    """Ask which harvested values name the disputed people.

    All corpus-name selection uses the light tier. The caller's ``node`` is retained as the attribution
    key for the pre-loop matching call.

    The values of EVERY entry in the source are unioned before chunking, so the 2021 staff listing's 18
    tabs and three subject spellings are asked once rather than eighteen times. A value on several tabs is
    asked about once here and mapped back to every tab that holds it in `pull_rows`.

    Returns `({person: [{name_in_file, confidence, why}]}, chunk count)`. Three things are dropped and
    warned rather than trusted: a value the model did not see in its chunk (it may only choose among the
    values it was shown), a person we did not ask about, and a whole chunk whose call failed — a partial
    selection still beats a per-person scan of a 224,000-row workbook.
    """
    warn = warn or _warn
    people = [str(p).strip() for p in (people or []) if str(p).strip()]
    universe: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        for value in entry.get("values") or []:
            key = _norm(value)
            if key in seen:
                continue
            seen.add(key)
            universe.append(value)
    if not (universe and people):
        return {}, 0

    canonical = {_norm(v): v for v in universe}
    selected: dict[str, list[dict]] = {}
    unresolved_people: list[str] = []
    for person in people:
        matches = _deterministic_name_matches(person, universe)
        if matches:
            selected[person] = [
                {"name_in_file": value, "confidence": "high",
                 "why": "deterministic surname and given-name match"}
                for value in matches
            ]
        else:
            unresolved_people.append(person)
    if not unresolved_people:
        return selected, 0

    shape = _shape_phrase(entries)
    claimed_values = {_norm(item["name_in_file"]) for items in selected.values() for item in items}
    unresolved_universe = [value for value in universe if _norm(value) not in claimed_values]
    if not unresolved_universe:
        return selected, 0
    chunks = _chunks(unresolved_universe, _chunk_chars())
    selector_node = "person_index"

    def _one(chunk: list[str]) -> tuple[list, list[str]]:
        """One chunk's matches plus its warnings. Never raises — a dead chunk must not kill the node."""
        notes: list[str] = []
        user = _select_prompt(chunk, unresolved_people, source_role, shape)
        try:
            reply = llm.structured(selector_node, PERSON_SELECT_SYSTEM, user, _NameSelection)
        except Exception as exc:
            notes.append(f"WARN: name selection over {len(chunk)} {source_role} value(s) failed — "
                         f"{type(exc).__name__}: {exc}")
            return [], notes
        out = []
        for match in reply.matches or []:
            value = canonical.get(_norm(match.name_in_file))
            if not value:
                notes.append(f"WARN: dropped {source_role} selection {match.name_in_file!r} — that "
                             f"spelling is not in the file's name list")
                continue
            person = _resolve_person(match.person, unresolved_people)
            if not person:
                notes.append(f"WARN: dropped {source_role} selection {value!r} — it was returned for "
                             f"{match.person!r}, who is not one of the disputed people")
                continue
            out.append((person, value, _clean(match.confidence), _clean(match.why)))
        return out, notes

    jobs = min(_select_jobs(), len(chunks))
    if jobs <= 1:
        results = [_one(chunk) for chunk in chunks]
    else:
        # A fresh `copy_context()` PER TASK, not one shared across them: a Context cannot be entered by
        # two threads at once, and a new thread inherits none of the ContextVars that carry token
        # attribution (`shared.tokens._ATTR`, `otl/token_trace.py`'s `_VISIT`). Without this the
        # `by_node` reconciliation in token_trace.py breaks, and it breaks silently.
        with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="name-select") as pool:
            futures = [pool.submit(contextvars.copy_context().run, _one, chunk) for chunk in chunks]
            results = [f.result() for f in futures]

    taken: set[tuple[str, str]] = {
        (person, _norm(item["name_in_file"]))
        for person, items in selected.items() for item in items
    }
    for matches, notes in results:            # in chunk order, so warnings read in file order
        for note in notes:
            warn(note)
        for person, value, confidence, why in matches:
            key = (person, _norm(value))
            if key in taken:
                continue
            taken.add(key)
            selected.setdefault(person, []).append(
                {"name_in_file": value, "confidence": confidence, "why": why})
    return selected, len(chunks)


def _near_misses(universe: list[str], person: str) -> tuple[list[str], int]:
    """The values sharing a person's surname, bounded, with the true count.

    This is the `WILLIAMS` case from the 0033 trace: 199 values carry that surname and none of them is
    Donell Williams. Reporting the size of that pool is the difference between "absent from the file" and
    "not found", and the first is a finding the drafter can use.
    """
    surname, _givens = _person_parts(person)
    if not surname:
        return [], 0
    hits = [v for v in universe if surname in re.split(r"[^a-z]+", _norm(v))]
    return hits[:_MAX_POOL_KEPT], len(hits)


# --- phase 3: pull the selected people's rows ------------------------------------------------------

def _row_dict(cells: list, header: list, tab: str, index: int) -> dict:
    """One row keyed by its column label, so no index arithmetic is needed to read it.

    EVERY labelled column is emitted, not just the classified ones. A row the agent can read straight is
    the whole point, and it is also what answers "does this year's tab have a Job Profile column" without
    a second read. A repeated label is suffixed with its position so nothing is overwritten.
    """
    out: dict = {"_tab": tab, "_row": index}
    for i, raw in enumerate(header):
        label = _label(raw)
        if not label:
            continue
        value = _clean(cells[i]) if i < len(cells) else ""
        if not value:
            continue
        out[label if label not in out else f"{label} [{i}]"] = value
    return out


def pull_rows(entries: list[dict], selected: dict, warn=None) -> None:
    """Buffer the rows of every selected spelling, filling `row_data` and `row_counts` on each entry.

    Deterministic, and one more traversal per workbook — matching is exact string membership over a set,
    which is why it costs one pass no matter how many people were selected. Past `_MAX_ROWS_PER_NAME` per
    spelling the rows stop being kept but the count keeps rising, so truncation is always reported with
    the true number.
    """
    warn = warn or _warn
    wanted: dict[str, str] = {}
    for matches in (selected or {}).values():
        for match in matches:
            wanted[_norm(match["name_in_file"])] = match["name_in_file"]
    if not wanted:
        return

    by_path: dict[str, list[dict]] = {}
    for entry in entries:
        by_path.setdefault(entry["path"], []).append(entry)

    totals: dict[str, int] = {}
    for path, group in by_path.items():
        file = Path(path)
        try:
            wb, _engine = open_workbook_any(file)
        except Exception as exc:
            warn(f"WARN: could not re-open {file.name} to pull rows — {type(exc).__name__}: {exc}")
            continue
        try:
            sheets, _dropped = _usable_worksheets(wb, source=file.name, warn=warn)
            by_tab: dict[str, list[dict]] = {}
            for entry in group:
                by_tab.setdefault(entry["tab"], []).append(entry)
            for ws in sheets:
                title = getattr(ws, "title", "") or ""
                targets = by_tab.get(title)
                if not targets:
                    continue
                first_data_row = min(int(e["header_row"]) for e in targets) + 1
                try:
                    for i, row in enumerate(ws.iter_rows(values_only=True)):
                        if i < first_data_row:
                            continue
                        cells = list(row)
                        if _is_banner(cells):
                            continue
                        for entry in targets:
                            if i <= int(entry["header_row"]):
                                continue
                            for value in _subject_values(cells, entry["subject"]):
                                name = wanted.get(_norm(value))
                                if not name:
                                    continue
                                totals[name] = totals.get(name, 0) + 1
                                entry["row_counts"][name] = entry["row_counts"].get(name, 0) + 1
                                if totals[name] <= _MAX_ROWS_PER_NAME:
                                    entry["row_data"].setdefault(name, []).append(
                                        _row_dict(cells, entry["header"], title, i))
                except Exception as exc:
                    warn(f"WARN: pulling rows from {file.name}!{title} stopped early — "
                         f"{type(exc).__name__}: {exc}")
        finally:
            try:
                wb.close()
            except Exception:
                pass

    for name, total in totals.items():
        if total > _MAX_ROWS_PER_NAME:
            warn(f"NOTE: {name} has {total} rows; the first {_MAX_ROWS_PER_NAME} are indexed and the "
                 f"true count is recorded")


# --- the public build -----------------------------------------------------------------------------

def _source_key(source_role: str, paths, people: list[str]) -> tuple | None:
    """Cache key that changes when any input file changes. None when a file cannot be stat'd."""
    stamps: list[tuple] = []
    for raw in _flatten(paths):
        p = Path(raw)
        try:
            st = p.stat()
        except OSError:
            return None
        stamps.append((str(p), st.st_size, st.st_mtime_ns))
    return (source_role, tuple(sorted(stamps)), tuple(people))


def build_person_index(sources, people, manifest, warn=None) -> dict:
    """Index the disputed people across one or more corpus sources. Runs phases 1 → 2 → 3.

    `sources` is `[(role, paths), ...]` — `[("pob", pob_reports), ("heli", heli_logs)]` for `location`,
    `[("staff", [staff_listing])]` for `job_title` and `supervision` — which is the shape the nodes
    already pass to `build_manifest`. `manifest` is the manifest that node just built; the tab shapes are
    taken from it, so nothing is sniffed twice.

    Per source, the result records the tabs read with their column roles, the size of the name universe,
    how many model calls it took, which spellings were selected for each person, the near-miss pool for
    each person nothing was selected for, the tabs skipped with the reason, and every unreadable file with
    its error. An unreadable file is LISTED, never dropped: two files in the 2023 corpus folder fail on
    every engine, and an agent handed an index that silently omits one will conclude it does not exist.

    Cached per `(role, file stamps, people)`, so `job_title` and `supervision` share one staff-listing
    pass per case. The cached source dict is returned as-is and is read-only by convention.

    Returns an empty index when `OTL_PERSON_INDEX` is off, with no model call made.
    """
    warn = warn or _warn
    people = [str(p).strip() for p in (people or []) if str(p).strip()]
    index: dict = {"people": people, "sources": [], "unreadable": [], "warnings": [],
                   "enabled": enabled()}
    if not index["enabled"] or not people:
        return index

    for u in (manifest or {}).get("unreadable") or []:
        index["unreadable"].append({"name": u.get("name", ""), "path": u.get("path", ""),
                                    "errors": u.get("errors") or {}})

    for source_role, paths in (sources or []):
        role = str(source_role or "").strip() or "source"
        if not _flatten(paths):
            continue
        key = _source_key(role, paths, people)
        if key is not None and key in _INDEX_CACHE:
            index["sources"].append(_INDEX_CACHE[key])
            continue

        notes: list[str] = []

        def _note(message: str, _notes=notes) -> None:
            _notes.append(message)
            warn(message)

        entries, skipped = harvest_names(manifest, paths, warn=_note, source_role=role)
        universe: list[str] = []
        seen: set[str] = set()
        for entry in entries:
            for value in entry.get("values") or []:
                k = _norm(value)
                if k not in seen:
                    seen.add(k)
                    universe.append(value)

        selected, chunks = select_names("person_index", role, entries, people, warn=_note)
        if role == "staff":
            _attach_selected_staff_rows(entries, selected)
        else:
            pull_rows(entries, selected, warn=_note)

        for person, matches in selected.items():
            for match in matches:
                name = match["name_in_file"]
                match["rows"] = sum(e["row_counts"].get(name, 0) for e in entries)
                match["tabs"] = [e["tab"] for e in entries if e["row_counts"].get(name)]

        unselected: dict[str, dict] = {}
        for person in people:
            if selected.get(person):
                continue
            pool, total = _near_misses(universe, person)
            unselected[person] = {"pool": pool, "pool_total": total}

        source = {"role": role, "entries": entries, "universe": len(universe), "values": universe,
                  "chunks": chunks, "selected": selected, "unselected": unselected,
                  "skipped_tabs": skipped, "warnings": notes}
        if key is not None:
            _INDEX_CACHE[key] = source
        index["sources"].append(source)

    return index


# --- the prompt rendering -------------------------------------------------------------------------

_HEADER_NOTE = (
    "PERSON INDEX — these workbooks were read straight off disk before this loop started, and one "
    "med-tier pass over the FULL list of names in them picked out the disputed people. Do NOT scan a "
    "workbook to find a person: the names and the column roles below are already resolved, and each "
    "selected person's rows are already pulled. It is a PRIOR, not a guarantee — raw reads are still "
    "available, and if what you read disagrees with the index, trust your read and say so. State which "
    "spelling you adopted for each person and why."
)


def _role_line(entry: dict) -> str:
    """The tab's column roles as `role=Label,Label`, plus the roles it does not carry."""
    present = []
    missing = []
    for role, cols in (entry.get("columns") or {}).items():
        if cols:
            present.append(f"{role}={','.join(cols.keys())}")
        else:
            missing.append(role)
    text = " | ".join(present)
    if len(text) > _MAX_ROLE_CHARS:
        text = text[: _MAX_ROLE_CHARS - 1] + "…"
    out = f"    columns: {text}" if text else "    columns: none classified"
    if missing:
        out += f"\n    not present: {', '.join(missing)}"
    nameish = entry.get("unclassified_nameish") or []
    if nameish:
        out += (f"\n    name-like but NOT a person column: {', '.join(nameish[:6])}"
                + (f" (+{len(nameish) - 6} more)" if len(nameish) > 6 else ""))
    return out


def _subject_text(entry: dict) -> str:
    subject = entry.get("subject") or {}
    labels = subject.get("labels") or []
    if subject.get("kind") == "first_last":
        return f"{' + '.join(reversed(labels))} (joined \"LAST, FIRST\")"
    primary = labels[0] if labels else "?"
    if len(labels) > 1:
        return f"{primary} (also {', '.join(labels[1:])})"
    return primary


def _names_shown(people: list[str]) -> int:
    """How many selected spellings to show per person, scaled down as the person count grows.

    The labor pipeline normally examines every ledger person, so a large population must not make this
    prompt block grow without bound. The full selection is in `person_index()` at no token cost, so the
    block gets shorter rather than the index getting smaller.
    """
    n = len(people or [])
    if n <= 10:
        return 4
    if n <= 25:
        return 2
    return 1


def person_index_block(index: dict, people=None) -> str:
    """Render the index as a bounded block to append to an agent's `task` string.

    Append to `task`, never to the system prompt. `task` is part of the seed `HumanMessage` and the ReAct
    buffer is append-only, so the block is written into the prompt cache once per agent per case and read
    back at a tenth of the price on every later step. The system prompt is shared across cases, so a
    per-case block there would spoil a cached prefix.

    Every bound drops whole items and says how many it dropped. Nothing here is the last word: the full
    selection, the full rows and the full column maps are in `person_index()` in the sandbox, which costs
    no tokens at all.
    """
    if not index or not index.get("enabled"):
        return ""
    sources = index.get("sources") or []
    if not (sources or index.get("unreadable")):
        return ""
    people = [str(p).strip() for p in (people or index.get("people") or []) if str(p).strip()]
    per_person = _names_shown(people)

    out: list[str] = ["", _HEADER_NOTE, ""]
    for source in sources:
        entries = source.get("entries") or []
        title = _SOURCE_TITLES.get(source.get("role", ""), str(source.get("role", "")).upper())
        files = []
        for entry in entries:
            if entry["name"] not in files:
                files.append(entry["name"])
        out.append(f"[person index] {title} — {', '.join(files) or '(no readable file)'}")
        out.append(f"  {len(entries)} name-bearing tab(s); {source.get('universe', 0)} distinct name(s) "
                   f"read and put to the selector in {source.get('chunks', 0)} call(s)")

        # Tabs sharing a subject column and a role map are one line, because the 2021 staff listing has
        # 18 of them in three families and listing each would be three quarters of the block.
        families: dict[tuple, list[dict]] = {}
        for entry in entries:
            fam = (entry["header_row"], _subject_text(entry), _role_line(entry))
            families.setdefault(fam, []).append(entry)
        for (header_row, subject, roles), group in families.items():
            tabs = [e["tab"] for e in group]
            shown = tabs[:_MAX_TABS_SHOWN]
            text = ", ".join(shown)
            if len(tabs) > len(shown):
                text += f", …(+{len(tabs) - len(shown)} more tabs, same columns)"
            out.append(f"  tab {text} (header row {header_row}) — subject: {subject}")
            out.append(roles)

        skipped = source.get("skipped_tabs") or []
        if skipped:
            shown = skipped[:_MAX_SKIPPED_SHOWN]
            text = ", ".join(f"{s['tab']} ({s['reason']})" for s in shown)
            if len(skipped) > len(shown):
                text += f", …(+{len(skipped) - len(shown)} more)"
            out.append(f"  skipped tabs: {text}")

        selected = source.get("selected") or {}
        unselected = source.get("unselected") or {}
        for person in people:
            matches = selected.get(person) or []
            if not matches:
                miss = unselected.get(person) or {}
                total = miss.get("pool_total", 0)
                if total:
                    out.append(f"  {person} → no value selected. {total} value(s) share this surname and "
                               f"none was judged to be this person; the full list is in person_index().")
                else:
                    out.append(f"  {person} → no value selected, and NO value in this file shares this "
                               f"surname. Absence is a finding — report it, do not re-scan for it.")
                continue
            shown = matches[:per_person]
            for match in shown:
                rows = match.get("rows", 0)
                head = (f"  {person} → {match['name_in_file']} "
                        f"({match.get('confidence') or 'unrated'} confidence; {rows} row(s))")
                if match.get("why"):
                    head += f" — {match['why']}"
                out.append(head)
                for entry in entries:
                    for row in (entry["row_data"].get(match["name_in_file"]) or [])[:_MAX_ROWS_SHOWN]:
                        attrs = " / ".join(f"{k}={v}" for k, v in row.items()
                                           if not str(k).startswith("_"))
                        if len(attrs) > _MAX_ATTR_CHARS:
                            attrs = attrs[: _MAX_ATTR_CHARS - 1] + "…"
                        out.append(f"      {entry['tab']} row {row['_row']}: {attrs}")
                    kept = len(entry["row_data"].get(match["name_in_file"]) or [])
                    if kept > _MAX_ROWS_SHOWN:
                        out.append(f"      …(+{kept - _MAX_ROWS_SHOWN} more rows on {entry['tab']} — "
                                   f"read them with person_index())")
            if len(matches) > len(shown):
                out.append(f"      …(+{len(matches) - len(shown)} more candidate spelling(s) for "
                           f"{person} — see person_index())")

    for u in index.get("unreadable") or []:
        errs = "; ".join(f"{eng}: {msg}" for eng, msg in (u.get("errors") or {}).items())
        out.append(f"UNREADABLE: {u.get('name', '')} — {errs or 'unknown error'}. The file IS there; no "
                   f"reader could open it. Do not treat it as missing.")

    out.append("In code, person_index() returns all of this: every tab's column map (role → label → "
               "column index), every selected spelling, and every pulled row keyed by column label. It "
               "reads a staged file, so it costs nothing.")
    text = "\n".join(out) + "\n"
    if len(text) > _MAX_BLOCK_CHARS:
        text = (text[: _MAX_BLOCK_CHARS]
                + f"\n…(person index block truncated at {_MAX_BLOCK_CHARS} characters — the whole index "
                  f"is available in code via person_index())\n")
    return text


def index_json(index: dict) -> dict:
    """The payload staged into the code sandbox as `person_index.json`.

    The full name universe is stripped and only the bounded near-miss pool is kept. The universe is up to
    83,700 values for one staff listing, and writing it out per node would cost real disk and time for
    something the agent can re-read off the sheet if it ever needs it. Everything the agent actually
    reasons over — the column maps, the selected spellings and the pulled rows — is kept in full.
    """
    if not index or not index.get("enabled"):
        return {}
    out: dict = {"people": list(index.get("people") or []),
                 "unreadable": list(index.get("unreadable") or []),
                 "warnings": list(index.get("warnings") or []),
                 "sources": []}
    for source in index.get("sources") or []:
        entries = []
        for entry in source.get("entries") or []:
            entries.append({k: v for k, v in entry.items() if k not in ("values", "staff_rows")}
                           | {"n_values": len(entry.get("values") or [])})
        out["sources"].append({k: v for k, v in source.items()
                               if k not in ("entries", "values")} | {"entries": entries})
    return out


__all__ = ["build_person_index", "person_index_block", "index_json", "harvest_names", "select_names",
           "pull_rows", "classify_columns", "enabled"]
