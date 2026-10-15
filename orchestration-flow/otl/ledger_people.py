"""Resolve the people JADE should examine from ledger values, intake leads, and POS suggestions.

Every charge row carries an `Employee` value, and when that cell is blank or "Not assigned" the
person's name sits inside the free-text `Line Item Text Orig` column instead. Intake also contributes
explicitly named people as leads, so the final population does not lose a person merely because a
ledger export abbreviates or omits their name.

So this module does two things, in order:

  * `find_ledger_tabs` + `harvest_person_values` — deterministic. They read the name column off disk and
    return the distinct values, exactly as written. No model is involved and nothing is dropped.
  * `extract_people` — one light-tier model call per chunk of values. The harvested values are not clean
    names ("2023-04 KEN OYLER OFFSHORE SUPPORT 40H"), so a model resolves them alongside intake leads
    and local POS suggestions into the final population.

The read is a fact about the file and should be reproducible; the final person-resolution step is where
the model makes the limited judgment needed to reconcile those facts with intake leads.

Nothing here adds an Excel reader. `extractor/excel_bounded.py` already dispatches the right reader by
file content (`open_workbook_any`) and hides operator-response tabs (`_usable_worksheets`), which is what
keeps this agreeing with the workbook manifest and the code sandbox. The tab shapes come from the
manifest the calling node already built, so no file is sniffed twice.

The values are NOT filtered by labor account code. That filter belongs to the amount math inside the
ReAct loop, not here: a name lost to a header-match miss on the account column is a person left out of
the whole rebuttal, which is a far worse outcome than one extra name that the person cap bounds anyway.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from shared.excel_bounded import (WorkbookOpenError, _usable_worksheets,  # noqa: PLC2701
                                     open_workbook_any)

from . import llm_adapter as llm
from .person_pos_tagger import proper_name_candidates
from .prompts import JADE_PEOPLE_SYSTEM

# Values that mean "no name here". The ledger writes these in the Employee column when the charge is
# not attributed to a person, which is exactly when the name has to come from the line-item text.
_PLACEHOLDERS = {"", "-", "--", "n/a", "na", "none", "null", "not assigned", "unassigned", "unavailable",
                 "not applicable", "#n/a", "0"}

# One light-tier request carries at most this many characters of harvested values. A real ledger has a
# few dozen distinct values, so this is normally one call; the chunking is what guarantees that a long
# one is still covered in full rather than truncated.
_CHUNK_CHARS = 6000


def _warn(message: str) -> None:
    print(message, file=sys.stderr)


def _norm(value) -> str:
    """A label or cell value reduced to a comparable form: collapsed whitespace, lowercased."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _clean(value) -> str:
    """A cell value as we keep it: collapsed whitespace, original casing."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


# --- which column holds the name -----------------------------------------------------------------

# An "Employee ID" column is not a name column, and it sorts before the real one in some exports, so
# the id-ish labels have to be excluded explicitly rather than by ordering.
_ID_WORDS = ("id", "ids", "no", "no.", "num", "number", "code", "key", "#")


def _is_employee_column(label: str) -> bool:
    """True for a person-name column such as `Employee` or `Employee Name`, false for `Employee ID`."""
    norm = _norm(label)
    if "employee" not in norm and "personnel name" not in norm and "worker" not in norm:
        return False
    tokens = re.split(r"[^a-z0-9#.]+", norm)
    return not any(t in _ID_WORDS for t in tokens)


def _is_text_column(label: str) -> bool:
    """True for the free-text line-item column the name falls back to."""
    norm = _norm(label)
    return norm == "text" or "line item text" in norm or "item text" in norm


def _person_columns(header: list) -> dict:
    """`{"employee": idx, "text": idx}` for whichever of the two the header carries.

    Both are returned when both exist, because the choice is per ROW, not per tab: one ledger has an
    Employee value on some rows and "Not assigned" plus a named line-item text on others. The first
    matching column of each kind wins.
    """
    cols: dict[str, int] = {}
    for i, label in enumerate(header or []):
        if "employee" not in cols and _is_employee_column(label):
            cols["employee"] = i
        elif "text" not in cols and _is_text_column(label):
            cols["text"] = i
    return cols


def find_ledger_tabs(manifest: dict) -> list[dict]:
    """The tabs worth harvesting names from, taken from the manifest the calling node already built.

    Returns `[{"path", "name", "tab", "header_row", "cols", "detected_type"}]`, one entry per tab.
    `cols` is the `_person_columns` mapping and is never empty — a tab with no name column has nothing
    to give.

    Two passes, and the first one that finds anything wins:

      1. Tabs the extractor's own signature detection recognized as a JADE ledger
         (`detect_tab_type` → "JADE"). This is the normal case and it is the precise one.
      2. Any other tab carrying a name column. This is the generous fallback, for a hand-compiled
         exhibit whose columns were renamed enough that the signature does not fire.

    The passes are ordered rather than merged so the POB tab that sometimes shares the auditor's
    workbook cannot contribute its passenger list once a real ledger tab has been found.
    """
    detected: list[dict] = []
    fallback: list[dict] = []
    for wb in (manifest or {}).get("workbooks") or []:
        for shape in wb.get("shapes") or []:
            cols = _person_columns(shape.get("header") or [])
            if not cols:
                continue
            for tab in shape.get("tabs") or []:
                entry = {"path": wb.get("path", ""), "name": wb.get("name", ""), "tab": tab,
                         "header_row": shape.get("header_row"), "cols": cols,
                         "header": list(shape.get("header") or []),
                         "detected_type": shape.get("detected_type") or ""}
                (detected if entry["detected_type"] == "JADE" else fallback).append(entry)
    return detected or fallback


# --- reading the values off the sheet -------------------------------------------------------------

def _cell(row, index) -> str:
    if index is None or index >= len(row):
        return ""
    value = _clean(row[index])
    return "" if _norm(value) in _PLACEHOLDERS else value


def harvest_person_values(path, tab: str, header_row, cols: dict, warn=None,
                          access_issues: list[str] | None = None) -> list[str]:
    """The distinct name-column values of one tab, in the order they first appear.

    Per row: the Employee value when it holds a real name, else the line-item text. Blanks and the
    "Not assigned" placeholders are skipped, and near-duplicates are collapsed case-insensitively with
    the first spelling kept.

    There is no row cap and no cap on how many distinct values come back. An auditor-provided JADE
    exhibit is a small hand-built extract — a few dozen rows — so a cap here would only risk losing a
    person while saving nothing.
    """
    warn = warn or _warn
    path = Path(path)
    try:
        wb, _engine = open_workbook_any(path)
    except WorkbookOpenError as exc:
        warn(f"WARN: could not open {path.name} to harvest ledger names — {exc}")
        if access_issues is not None:
            access_issues.append(f"{path.name}!{tab}: {type(exc).__name__}: {exc}")
        return []
    try:
        sheets, _dropped = _usable_worksheets(wb, source=path.name, warn=warn)
        ws = next((s for s in sheets if (getattr(s, "title", "") or "") == tab), None)
        if ws is None:
            if access_issues is not None:
                access_issues.append(f"{path.name}!{tab}: worksheet could not be accessed")
            return []
        first_data_row = (header_row + 1) if isinstance(header_row, int) else 0
        out: list[str] = []
        seen: set[str] = set()
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i < first_data_row:
                continue
            cells = list(row)
            value = _cell(cells, cols.get("employee")) or _cell(cells, cols.get("text"))
            if not value:
                continue
            key = _norm(value)
            if key in seen:
                continue
            seen.add(key)
            out.append(value)
        return out
    except Exception as exc:                       # a corrupt sheet inside a readable workbook
        warn(f"WARN: harvesting ledger names from {path.name}!{tab} failed — "
             f"{type(exc).__name__}: {exc}")
        if access_issues is not None:
            access_issues.append(f"{path.name}!{tab}: {type(exc).__name__}: {exc}")
        return []
    finally:
        try:
            wb.close()
        except Exception:
            pass


def harvest(manifest: dict, warn=None) -> tuple[list[str], list[str]]:
    """Harvest every ledger tab in the manifest. Returns (distinct values, `file!tab` labels read)."""
    values, tabs_read, _access_issues = harvest_with_diagnostics(manifest, warn=warn)
    return values, tabs_read


def harvest_with_diagnostics(manifest: dict, warn=None) -> tuple[list[str], list[str], list[str]]:
    """Harvest ledger values plus concrete file/tab access failures.

    A blank but readable ledger is different from one that cannot be opened. The JADE evidence gate uses
    that distinction to deny unsupported exceptions while deferring inaccessible supplied evidence.
    """
    values: list[str] = []
    seen: set[str] = set()
    tabs_read: list[str] = []
    access_issues: list[str] = []
    for entry in find_ledger_tabs(manifest):
        got = harvest_person_values(entry["path"], entry["tab"], entry["header_row"], entry["cols"],
                                    warn=warn, access_issues=access_issues)
        if not got:
            continue
        tabs_read.append(f"{entry['name']}!{entry['tab']}")
        for value in got:
            key = _norm(value)
            if key in seen:
                continue
            seen.add(key)
            values.append(value)
    return values, tabs_read, access_issues


# --- pulling the names out of the values ----------------------------------------------------------

class _LedgerPeople(BaseModel):
    """What the light model returns. The Field description is part of the schema it reads."""
    people: list[str] = Field(
        default_factory=list,
        description="The final distinct human-person population resolved from the supplied ledger values, "
                    "POS suggestions, and intake lead names. Prefer a matching ledger spelling; retain an "
                    "intake lead's spelling when the ledger does not name that person.")


def _chunks(values: list[str], budget: int = _CHUNK_CHARS) -> list[list[str]]:
    """Split the values into groups that each fit `budget` characters. Every value lands in a group."""
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


def _candidate_digest(candidates: list[dict]) -> list[dict]:
    """Keep the POS prompt compact while retaining the source value and exact suggestion."""
    return [
        {key: item[key] for key in ("candidate_id", "value_id", "text", "start", "end", "score")
         if key in item}
        for item in candidates
    ]


def extract_people(node: str, values: list[str], warn=None, *,
                   intake_people: list[str] | None = None) -> list[str]:
    """Resolve people from harvested ledger values, POS suggestions, and intake leads.

    One call per ledger-value chunk, so a long list is covered in full. Intake leads are supplied to every
    chunk because they can identify a person whose name is abbreviated in one ledger value or absent from
    a particular chunk. Names are deduped across chunks, case-insensitively, with the first spelling kept.
    A chunk whose call fails is reported and skipped rather than failing the node — a partial resolved list
    is retained and traceable.
    """
    warn = warn or _warn
    values = [v for v in (values or []) if str(v).strip()]
    if not values:
        return []
    candidates, _status = proper_name_candidates(values)
    intake_leads = list(dict.fromkeys(
        _clean(person) for person in (intake_people or []) if _clean(person)
    ))
    intake_listing = "\n".join(f"* {person}" for person in intake_leads) or "(none)"
    names: list[str] = []
    seen: set[str] = set()
    first_value_index = 1
    for chunk in _chunks(values):
        value_ids = {f"V{i}" for i in range(first_value_index, first_value_index + len(chunk))}
        local_candidates = [item for item in candidates if item.get("value_id") in value_ids]
        first_value_index += len(chunk)
        listing = "\n".join(f"- {v}" for v in chunk)
        user = (
            "These values come from the name column of a joint-venture accounting ledger. Each line is "
            "one distinct value, exactly as the ledger holds it. A local POS model has marked probable "
            "proper-noun spans, and intake supplied explicitly named people as lead candidates. Reconcile "
            "all three inputs into the final people population. Treat POS spans and intake leads as "
            "high-recall suggestions, not conclusions: reject companies, assets, projects, places, roles, "
            "and activity descriptions. Inspect every original ledger value because the POS model can miss "
            "or split names. Prefer the ledger spelling when it identifies the same person; otherwise, "
            "retain a supported intake lead's spelling. Do not invent or expand a missing token.\n\n"
            f"LEDGER VALUES:\n{listing}\n\n"
            f"LOCAL PROPN CANDIDATES:\n{_candidate_digest(local_candidates)}\n\n"
            f"INTAKE LEAD PEOPLE:\n{intake_listing}\n\n"
            "List the final supported human person names to examine downstream.")
        try:
            ext = llm.structured(node, JADE_PEOPLE_SYSTEM, user, _LedgerPeople)
        except Exception as exc:
            warn(f"WARN: extracting names from {len(chunk)} ledger value(s) failed — "
                 f"{type(exc).__name__}: {exc}")
            continue
        for raw in ext.people or []:
            name = _clean(raw)
            key = _norm(name)
            if not name or key in seen:
                continue
            seen.add(key)
            names.append(name)
    return names


# --- deterministic charge totals for the final resolved population -------------------------------

def _is_cost_plus_overhead_column(label: str) -> bool:
    compact = re.sub(r"[^a-z0-9]+", "", _norm(label))
    return ("cost" in compact and ("overhead" in compact or "oh" in compact)) or compact in {
        "totalcost", "totalchargedamount", "chargeamount",
    }


def _amount_column(header: list) -> int | None:
    """Resolve one loaded-charge column, failing closed when headers are ambiguous."""
    exact = [i for i, label in enumerate(header or []) if _is_cost_plus_overhead_column(label)]
    if len(exact) == 1:
        return exact[0]
    if exact:
        return None
    generic = [i for i, label in enumerate(header or []) if _norm(label) in {"amount", "loaded amount"}]
    return generic[0] if len(generic) == 1 else None


def _number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    raw = str(value).strip()
    if not raw:
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    try:
        value = float(re.sub(r"[^0-9.+-]", "", raw))
    except ValueError:
        return None
    value = -abs(value) if negative else value
    return value if math.isfinite(value) else None


def _name_tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z]+", _norm(value)))


def _row_person(value: str, people: list[str]) -> str | None:
    """Return one final identity whose supported name tokens occur in the ledger value.

    Ambiguous rows are excluded rather than allocating dollars between people by guesswork.
    """
    row_tokens = _name_tokens(value)
    matches = [person for person in people if (tokens := _name_tokens(person)) and tokens <= row_tokens]
    return matches[0] if len(matches) == 1 else None


def charged_amount_by_person(manifest: dict, people: list[str], warn=None) -> dict[str, float]:
    """Sum matched JADE ``Cost + OH`` values for the complete final resolved population.

    The map is a ledger fact: it is calculated after ledger/POS/intake name resolution has settled
    `people`, independently of the ReAct amount-reconciliation conclusion and the later specialist cap.
    It is empty for an ambiguous/missing amount column and skips individual rows that do not map to exactly
    one final person.
    """
    warn = warn or _warn
    totals: dict[str, float] = {}
    selected_people = [str(person).strip() for person in people or [] if str(person).strip()]
    if not selected_people:
        return totals
    for entry in find_ledger_tabs(manifest):
        amount_index = _amount_column(entry.get("header") or [])
        if amount_index is None:
            warn(f"WARN: {entry['name']}!{entry['tab']} has no unambiguous Cost + OH amount column")
            continue
        path = Path(entry["path"])
        try:
            wb, _engine = open_workbook_any(path)
        except WorkbookOpenError as exc:
            warn(f"WARN: could not open {path.name} to total person charges — {exc}")
            continue
        try:
            sheets, _dropped = _usable_worksheets(wb, source=path.name, warn=warn)
            ws = next((sheet for sheet in sheets if (getattr(sheet, "title", "") or "") == entry["tab"]), None)
            if ws is None:
                continue
            first_data_row = (entry["header_row"] + 1 if isinstance(entry.get("header_row"), int) else 0)
            for row_index, row in enumerate(ws.iter_rows(values_only=True)):
                if row_index < first_data_row:
                    continue
                cells = list(row)
                person = _row_person(
                    _cell(cells, entry["cols"].get("employee"))
                    or _cell(cells, entry["cols"].get("text")),
                    selected_people,
                )
                amount = _number(cells[amount_index] if amount_index < len(cells) else None)
                if person is not None and amount is not None:
                    totals[person] = totals.get(person, 0.0) + amount
        except Exception as exc:
            warn(f"WARN: totaling person charges from {path.name}!{entry['tab']} failed — "
                 f"{type(exc).__name__}: {exc}")
        finally:
            try:
                wb.close()
            except Exception:
                pass
    return {person: round(totals[person], 2) for person in selected_people if person in totals}


__all__ = ["charged_amount_by_person", "find_ledger_tabs", "harvest_person_values", "harvest",
           "harvest_with_diagnostics",
           "extract_people"]
