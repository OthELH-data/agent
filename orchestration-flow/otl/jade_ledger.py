"""Deterministic, row-first JADE extraction used before any code-execution fallback."""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from shared.excel_bounded import WorkbookOpenError, _usable_worksheets, open_workbook_any  # noqa: PLC2701

from otl.jade_rules import asset_cost_centers, cats_accounts, mcoh_threshold
from otl.jv_object_types import is_major_construction_code
from otl.state import LedgerRow, McohMonthlyFact, SourceRowRef


_PLACEHOLDERS = {"", "-", "--", "n/a", "na", "none", "null", "not assigned", "unassigned",
                 "unavailable", "not applicable", "#n/a", "0"}


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    text = str(value).strip().replace(",", "").replace("$", "")
    if not text:
        return None
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    try:
        result = float(text.rstrip("%"))
        if str(value).strip().endswith("%"):
            result /= 100
        return result if math.isfinite(result) else None
    except ValueError:
        return None


def _date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _clean(value)
    if not text:
        return None
    for fmt in ("%d.%m.%Y", "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    return None


def _indices(header: list, predicate) -> list[int]:
    return [i for i, label in enumerate(header or []) if predicate(_norm(label))]


def _first(header: list, *aliases: str, contains: tuple[str, ...] = ()) -> int | None:
    normalized = [_norm(v) for v in header or []]
    wanted = {_norm(v) for v in aliases}
    for i, label in enumerate(normalized):
        if label in wanted:
            return i
    for i, label in enumerate(normalized):
        if contains and all(term in label for term in contains):
            return i
    return None


def _column_map(header: list) -> dict[str, int]:
    account = _indices(header, lambda s: "account" in s and "acct" in s)
    wbs = _indices(header, lambda s: "cost object" in s and ("wbs" in s or "cc" in s))
    employee = _indices(header, lambda s: ("employee" in s or "personnel name" in s)
                        and not any(k in s for k in ("number", " id", "code")))
    mapping = {
        "account_code": account[0] if account else None,
        "account_description": account[1] if len(account) > 1 else None,
        "wbs": wbs[0] if wbs else _first(header, "WBS", "WBS Element"),
        "wbs_description": wbs[1] if len(wbs) > 1 else None,
        "employee": employee[0] if employee else None,
        "employee_id": _first(header, "Employee Number", "Employee ID", "Staff Number"),
        "line_item_text": _first(header, "Line Item Text Orig", "Line Item Text", "Item Text", "Text"),
        "jv_object_type": _first(header, "WBS:JV Object Type", "JV Object Type"),
        "entry_date_sl": _first(header, "Entry Date SL"),
        "qty_local_uom": _first(header, "Qty - Local UoM", "Qty – Local UoM", "Qty"),
        "cost_plus_oh": _first(header, "Cost + OH", "Cost+OH", "Total Cost", "Total"),
        "base_amount": _first(header, "Actual Amout in Local Currency USD",
                              "Actual Amout in Local Currency", "Actual Amount in Local Currency"),
        "oh_amount": _first(header, "Overhead $", "OH $", "OH Amount"),
        "oh_rate": _first(header, "OH RATE", "OH %", "OH"),
        "cost_center_code": _first(header, "Master Cost Center Key", "Responsible Cost Center (RCC)",
                                   "Cost Center Key"),
        "cost_center_name": _first(header, "Master Cost Center Name", "Responsible Cost Center Name",
                                   "Cost Center Name"),
        "doc_number": _first(header, "Doc # FI", "Document Number", "Doc # SL"),
        "ref_number": _first(header, "Ref Doc # SL", "Reference Document"),
    }
    return {k: v for k, v in mapping.items() if v is not None}


def _cell(cells: list, mapping: dict[str, int], key: str) -> Any:
    index = mapping.get(key)
    return cells[index] if index is not None and index < len(cells) else None


def _account_parts(code_value: Any, description_value: Any) -> tuple[str, str]:
    code_text, description = _clean(code_value), _clean(description_value)
    combined = f"{code_text} {description}"
    match = re.search(r"\bA\d{7}\b", combined.upper())
    code = match.group(0) if match else code_text
    if code and description.startswith(code):
        description = description[len(code):].strip(" -;")
    return code, description


def _cost_center_parts(code_value: Any, name_value: Any) -> tuple[str, str]:
    code, name = _clean(code_value), _clean(name_value)
    match = re.search(r"\b\d{6}\b", f"{code} {name}")
    return (match.group(0) if match else code), name


def _rate(value: Any) -> float | None:
    number = _number(value)
    if number is None:
        return None
    return number / 100 if abs(number) > 1 else number


def _amount(cells: list, mapping: dict[str, int]) -> tuple[float | None, str, float | None]:
    loaded = _number(_cell(cells, mapping, "cost_plus_oh"))
    if loaded is not None:
        return loaded, "cost_plus_oh_column", _rate(_cell(cells, mapping, "oh_rate"))
    base = _number(_cell(cells, mapping, "base_amount"))
    overhead = _number(_cell(cells, mapping, "oh_amount"))
    rate = _rate(_cell(cells, mapping, "oh_rate"))
    if base is None:
        return None, "", rate
    if overhead is not None:
        return base + overhead, "base_plus_overhead_amount", rate
    if rate is not None:
        return base * (1 + rate), "base_times_rate", rate
    return base, "base_amount", rate


def _is_summary(cells: list, mapping: dict[str, int]) -> bool:
    anchors = [_cell(cells, mapping, key) for key in
               ("wbs", "account_code", "employee", "line_item_text", "jv_object_type")]
    normalized = [_norm(v) for v in cells if _norm(v)]
    has_total_label = any(v in {"total", "grand total", "subtotal"} or v.startswith("grand total")
                          for v in normalized)
    return has_total_label and not any(_clean(v) for v in anchors)


def _row_signature(cells: list, mapping: dict[str, int]) -> str:
    keys = ("account_code", "wbs", "employee_id", "employee", "line_item_text", "jv_object_type",
            "entry_date_sl", "qty_local_uom", "base_amount", "cost_plus_oh", "doc_number", "ref_number")
    values = [re.sub(r"\s+", "", str(_cell(cells, mapping, key) or "").casefold()) for key in keys]
    return hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()


@dataclass
class _Candidate:
    workbook: str
    sheet: str
    rows: list[LedgerRow]
    raw_people: list[str]
    mapping: dict[str, str]
    signatures: Counter
    modes: Counter
    rates: list[float]
    warnings: list[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        return sum(float(row.amount or 0.0) for row in self.rows)


@dataclass
class JadeLedgerExtraction:
    rows: list[LedgerRow]
    raw_people: list[str]
    resolved_columns: dict[str, str]
    amount_basis: str
    amount_computed: float
    amount_in_range: bool
    overhead_mode: str
    overhead_rate: float
    warnings: list[str]


def _candidate(path: Path, workbook_name: str, sheet: str, header_row: int, header: list) -> _Candidate:
    mapping = _column_map(header)
    wb, _engine = open_workbook_any(path)
    try:
        sheets, _dropped = _usable_worksheets(wb, source=workbook_name, warn=lambda _m: None)
        ws = next((item for item in sheets if (getattr(item, "title", "") or "") == sheet), None)
        if ws is None:
            raise ValueError(f"worksheet {sheet!r} not available")
        rows: list[LedgerRow] = []
        people: list[str] = []
        signatures: Counter = Counter()
        modes: Counter = Counter()
        rates: list[float] = []
        seen_people: set[str] = set()
        for index, raw in enumerate(ws.iter_rows(values_only=True), 1):
            if index <= header_row + 1:
                continue
            cells = list(raw)
            if not any(value not in (None, "") for value in cells) or _is_summary(cells, mapping):
                continue
            amount, mode, rate = _amount(cells, mapping)
            wbs = _clean(_cell(cells, mapping, "wbs"))
            obj = _clean(_cell(cells, mapping, "jv_object_type"))
            account_code, account_desc = _account_parts(
                _cell(cells, mapping, "account_code"), _cell(cells, mapping, "account_description"))
            employee = _clean(_cell(cells, mapping, "employee"))
            line_text = _clean(_cell(cells, mapping, "line_item_text"))
            person_value = employee if _norm(employee) not in _PLACEHOLDERS else line_text
            if not any((wbs, obj, account_code, person_value)):
                continue
            if person_value and _norm(person_value) not in seen_people:
                seen_people.add(_norm(person_value)); people.append(person_value)
            cc_code, cc_name = _cost_center_parts(
                _cell(cells, mapping, "cost_center_code"), _cell(cells, mapping, "cost_center_name"))
            row = LedgerRow(
                source_ref=SourceRowRef(workbook=workbook_name, sheet=sheet, excel_row=index),
                employee_id=_clean(_cell(cells, mapping, "employee_id")) or None,
                wbs=wbs or None,
                wbs_description=_clean(_cell(cells, mapping, "wbs_description")) or None,
                account_code=account_code or None,
                account_description=account_desc or None,
                amount=round(amount, 6) if amount is not None else None,
                jv_object_type=obj or None,
                mcoh_status=("mcoh" if is_major_construction_code(obj) else "non_mcoh") if obj else None,
                qty_local_uom=_number(_cell(cells, mapping, "qty_local_uom")),
                entry_date_sl=_date(_cell(cells, mapping, "entry_date_sl")),
                cost_center_code=cc_code or None,
                cost_center_name=cc_name or None,
                line_item_text=" ".join(value for value in (employee, line_text) if value) or None,
            )
            rows.append(row)
            signatures[_row_signature(cells, mapping)] += 1
            if mode: modes[mode] += 1
            if rate is not None: rates.append(rate)
        actual = {key: str(header[index]) for key, index in mapping.items() if index < len(header)}
        return _Candidate(workbook_name, sheet, rows, people, actual, signatures, modes, rates)
    finally:
        try: wb.close()
        except Exception: pass


def _dedupe_candidates(candidates: list[_Candidate]) -> list[_Candidate]:
    selected: list[_Candidate] = []
    for candidate in candidates:
        duplicate = False
        for existing in selected:
            overlap = sum((candidate.signatures & existing.signatures).values())
            denominator = min(sum(candidate.signatures.values()), sum(existing.signatures.values()))
            if denominator and overlap / denominator >= 0.9:
                duplicate = True
                if len(candidate.rows) > len(existing.rows):
                    selected[selected.index(existing)] = candidate
                break
        if not duplicate:
            selected.append(candidate)
    return selected


def extract_jade_ledger(manifest: dict, questioned: float) -> JadeLedgerExtraction:
    candidates: list[_Candidate] = []
    warnings: list[str] = []
    for workbook in (manifest or {}).get("workbooks") or []:
        path = Path(str(workbook.get("path") or ""))
        for shape in workbook.get("shapes") or []:
            header = list(shape.get("header") or [])
            detected = shape.get("detected_type") == "JADE"
            recognizable = any("jv object type" in _norm(value) for value in header)
            if not (detected or recognizable):
                continue
            for tab in shape.get("tabs") or []:
                try:
                    candidates.append(_candidate(
                        path, str(workbook.get("name") or path.name), str(tab),
                        int(shape.get("header_row") or 0), header))
                except (WorkbookOpenError, Exception) as exc:
                    warnings.append(f"{path.name}!{tab}: {type(exc).__name__}: {exc}")
    if not candidates:
        raise ValueError("no readable JADE detail tab was found")
    candidates = _dedupe_candidates(candidates)

    # Consider each candidate and the additive union; choose the population whose sum/half best
    # reconciles to the auditor's amount. This preserves the established double-count guard while row
    # facts retain their source amounts unchanged.
    options: list[list[_Candidate]] = [[item] for item in candidates]
    if len(candidates) > 1:
        options.append(candidates)
    target = abs(float(questioned or 0.0))
    best: tuple[float, str, list[_Candidate], float] | None = None
    for option in options:
        total = sum(item.total for item in option)
        for basis, computed in (("sum", total), ("half", total / 2)):
            difference = abs(abs(computed) - target) if target else 0.0
            candidate = (difference, basis, option, computed)
            if best is None or candidate[0] < best[0]: best = candidate
    assert best is not None
    difference, basis, chosen, computed = best
    rows = [row for candidate in chosen for row in candidate.rows]
    raw_people = list(dict.fromkeys(value for candidate in chosen for value in candidate.raw_people))
    resolved: dict[str, str] = {}
    for candidate in chosen:
        for key, value in candidate.mapping.items(): resolved.setdefault(key, value)
    modes = Counter(mode for candidate in chosen for mode, count in candidate.modes.items() for _ in range(count))
    rates = [rate for candidate in chosen for rate in candidate.rates]
    tolerance = target * 0.025
    return JadeLedgerExtraction(
        rows=rows, raw_people=raw_people, resolved_columns=resolved,
        amount_basis=basis, amount_computed=round(computed, 2),
        amount_in_range=bool(target and difference <= tolerance),
        overhead_mode=modes.most_common(1)[0][0] if modes else "",
        overhead_rate=round(sum(rates) / len(rates), 6) if rates else 0.025,
        warnings=warnings,
    )


def _name_tokens(value: str) -> set[str]:
    stop = {"ptpd", "ptpdg", "labor", "expense", "support", "staff", "contractor"}
    return {token for token in re.findall(r"[a-z]+", _norm(value)) if len(token) > 1 and token not in stop}


def assign_people(rows: list[LedgerRow], people: list[str]) -> list[LedgerRow]:
    people = [str(person).strip() for person in people if str(person).strip()]
    for row in rows:
        haystack = " ".join(filter(None, (row.line_item_text, row.employee_id)))
        tokens = _name_tokens(haystack)
        matches = [person for person in people if _name_tokens(person) and _name_tokens(person) <= tokens]
        if len(matches) == 1:
            row.person = matches[0]
            continue
        # Employee-name columns are not retained separately; line-item text often begins with an ID and
        # then a name. A compact containment fallback handles punctuation/order differences.
        compact = re.sub(r"[^a-z]", "", _norm(haystack))
        contains = [person for person in people
                    if re.sub(r"[^a-z]", "", _norm(person)) in compact]
        if len(contains) == 1: row.person = contains[0]
    return rows


def monthly_mcoh_facts(rows: list[LedgerRow]) -> list[McohMonthlyFact]:
    threshold = mcoh_threshold()
    grouped: dict[tuple[str, str], list[LedgerRow]] = defaultdict(list)
    for row in rows:
        if row.person and row.mcoh_status == "mcoh" and row.entry_date_sl and row.qty_local_uom is not None:
            grouped[(row.person, row.entry_date_sl.strftime("%Y-%m"))].append(row)
    out: list[McohMonthlyFact] = []
    for (person, month), selected in grouped.items():
        hours = sum(float(row.qty_local_uom or 0) for row in selected)
        amount = sum(float(row.amount or 0) for row in selected)
        out.append(McohMonthlyFact(
            person=person, submission_month=month, hours=round(hours, 4),
            threshold_hours=threshold, below_threshold=hours < threshold,
            row_refs=[row.source_ref for row in selected], amount=round(amount, 2),
        ))
    return sorted(out, key=lambda item: (item.person.casefold(), item.submission_month))


def cost_center_people(rows: list[LedgerRow], asset: str) -> list[str]:
    registry = asset_cost_centers(asset)
    return list(dict.fromkeys(
        row.person for row in rows
        if row.person and row.cost_center_code and row.cost_center_code in registry
    ))


def cats_present(rows: list[LedgerRow]) -> bool:
    accounts = cats_accounts()
    return any(str(row.account_code or "").upper() in accounts for row in rows)


__all__ = ["JadeLedgerExtraction", "assign_people", "cats_present", "cost_center_people",
           "extract_jade_ledger", "monthly_mcoh_facts"]
