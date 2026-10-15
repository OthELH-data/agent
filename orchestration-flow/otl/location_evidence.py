"""Deterministic, one-pass POB/heli extraction for the location specialist.

The raw workbooks can contain more than 220,000 rows.  They are streamed once, but only plausible
name candidates and their normalized location events survive into ``location_evidence.json``.  The
light tier still decides identity; deterministic code only narrows the candidate universe.
"""
from __future__ import annotations

import json
import calendar
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field

from shared.excel_bounded import _usable_worksheets, open_workbook_any  # noqa: PLC2701

from . import llm_adapter as llm
from ._bootstrap import ORCH_DIR
from .person_index import classify_columns, select_names
from .state import LocationState

PACKET_MAX_TOKENS = 250_000
PACKET_TARGET_TOKENS = 220_000
_REPRESENTATIVE_ROWS = 3


class LocationDecision(BaseModel):
    result: LocationState = Field(default_factory=LocationState)
    evidence_ids_consumed: list[str] = Field(default_factory=list)
    needs_code_execution: bool = False
    escalation_reasons: list[str] = Field(default_factory=list)
    rationale: str = ""


class LocationShardFinding(BaseModel):
    shard_id: str
    person: str = ""
    evidence_ids_consumed: list[str] = Field(default_factory=list)
    pob_locations: list[str] = Field(default_factory=list)
    heli_assets: list[str] = Field(default_factory=list)
    routes: list[dict] = Field(default_factory=list)
    earliest_date: str = ""
    latest_date: str = ""
    charged_asset_relationship: str = ""
    conflicts: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    needs_raw_review: bool = False
    raw_review_reason: str = ""


@dataclass
class _EventAccumulator:
    event: dict
    duplicate_count: int = 0
    first_source_row: int = 0
    last_source_row: int = 0
    representative_source_rows: list[int] = field(default_factory=list)

    def add(self, row_number: int) -> None:
        self.duplicate_count += 1
        if not self.first_source_row:
            self.first_source_row = row_number
        self.last_source_row = row_number
        if len(self.representative_source_rows) < _REPRESENTATIVE_ROWS:
            self.representative_source_rows.append(row_number)

    def payload(self) -> dict:
        return dict(self.event) | {
            "duplicate_count": self.duplicate_count,
            "first_source_row": self.first_source_row,
            "last_source_row": self.last_source_row,
            "representative_source_rows": list(self.representative_source_rows),
        }


@dataclass
class _CandidateAccumulator:
    raw_name: str
    row_count: int = 0
    events: dict[tuple, _EventAccumulator] = field(default_factory=dict)

    def add(self, event: dict, row_number: int) -> None:
        self.row_count += 1
        signature = tuple((k, str(v)) for k, v in sorted(event.items()))
        acc = self.events.get(signature)
        if acc is None:
            acc = _EventAccumulator(event=event)
            self.events[signature] = acc
        acc.add(row_number)


@dataclass
class ExtractionResult:
    artifact: dict
    artifact_path: str
    unresolved_reasons: list[str]
    rows_scanned: dict[str, int]


def _clean(value) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    return re.sub(r"\s+", " ", str(value).strip())


def _norm(value) -> str:
    text = unicodedata.normalize("NFKD", _clean(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _tokens(value) -> list[str]:
    return [t for t in _norm(value).split() if t and t not in {"jr", "sr", "ii", "iii", "iv"}]


def _person_parts(person: str) -> tuple[str, list[str]]:
    text = _clean(person)
    if "," in text:
        surname_text, given_text = text.split(",", 1)
        surname_tokens, given_tokens = _tokens(surname_text), _tokens(given_text)
        return (surname_tokens[-1], given_tokens) if surname_tokens else ("", given_tokens)
    tokens = _tokens(text)
    return (tokens[-1], tokens[:-1]) if tokens else ("", [])


def _candidate_people(raw_name: str, people: list[str]) -> list[str]:
    """High-recall candidate screen.  Identity remains the light model's decision."""
    raw_tokens = set(_tokens(raw_name))
    out: list[str] = []
    for person in people:
        surname, givens = _person_parts(person)
        if not surname or surname not in raw_tokens:
            continue
        if not givens:
            out.append(person)
            continue
        if any(g in raw_tokens or any(t.startswith(g[:1]) for t in raw_tokens) for g in givens):
            out.append(person)
            continue
        # Same exact surname remains a candidate so nicknames/worker names stay LLM-resolvable.
        out.append(person)
    return out


def _is_banner(cells: list) -> bool:
    values = {_norm(c) for c in cells if _norm(c)}
    return len(values) == 1 and sum(1 for c in cells if _norm(c)) > 1


def _subject_values(cells: list, subject: dict) -> list[str]:
    def cell(i: int) -> str:
        return _clean(cells[i]) if 0 <= i < len(cells) else ""

    if subject.get("kind") == "first_last":
        indices = list(subject.get("indices") or [])
        if len(indices) >= 2:
            last, first = cell(indices[0]), cell(indices[1])
            return [f"{last}, {first}" if last and first else (last or first)] if (last or first) else []
    return [cell(i) for i in subject.get("indices") or [] if cell(i)]


def _header_map(header: list) -> dict[str, int]:
    return {_norm(label): i for i, label in enumerate(header) if _norm(label)}


def _get(cells: list, mapping: dict[str, int], *aliases: str) -> str:
    for alias in aliases:
        i = mapping.get(_norm(alias))
        if i is not None and i < len(cells):
            return _clean(cells[i])
    compact_mapping = {re.sub(r"[^a-z0-9]", "", key): value for key, value in mapping.items()}
    for alias in aliases:
        i = compact_mapping.get(re.sub(r"[^a-z0-9]", "", _norm(alias)))
        if i is not None and i < len(cells):
            return _clean(cells[i])
    return ""


def _date_text(value: str) -> str:
    text = _clean(value)
    if not text:
        return ""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    # POB exports append labels such as "Central Daylight Time" after an otherwise ordinary timestamp.
    match = re.match(r"^(\d{1,2}/\d{1,2}/\d{4})", text)
    if match:
        try:
            return datetime.strptime(match.group(1), "%m/%d/%Y").date().isoformat()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text).date().isoformat()
    except ValueError:
        return text


def _component_date(cells: list, mapping: dict[str, int], prefix: str,
                    inferred_year: int | None = None) -> tuple[str, str]:
    direct = _get(cells, mapping, prefix, prefix.replace(" ", ""), f"{prefix} date")
    if direct:
        return _date_text(direct), "day"
    year = _get(cells, mapping, f"{prefix} - Year")
    month = _get(cells, mapping, f"{prefix} - Month")
    day = _get(cells, mapping, f"{prefix} - Day")
    month_names = {name.lower(): i for i, name in enumerate(
        ("January", "February", "March", "April", "May", "June", "July", "August",
         "September", "October", "November", "December"), 1)}

    def integer(value: str) -> int | None:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    # Some helicopter exports label the column "END TIME - Year" but store the complete timestamp in
    # that cell. Preserve the actual day rather than trying to coerce the timestamp into an integer year.
    year_as_date = _date_text(year)
    if year_as_date and re.fullmatch(r"\d{4}-\d{2}-\d{2}", year_as_date):
        return year_as_date, "day"

    y = integer(year)
    m = integer(month) or month_names.get(month.lower())
    d = integer(day)
    if not any((year, month, day)):
        return "", "unknown"
    if not y and inferred_year and (m or d):
        y = inferred_year
    if y and m and d:
        try:
            return date(y, m, d).isoformat(), "day"
        except ValueError:
            pass
    if y and m:
        return f"{y:04d}-{m:02d}", "month"
    if y:
        return f"{y:04d}", "year"
    if m and d:
        return f"{m:02d}-{d:02d}", "month_day"
    if m:
        return f"{m:02d}", "month"
    return "", "unknown"


def _event_for(role: str, cells: list, header: list, *, workbook: str, sheet: str,
               inferred_year: int | None = None) -> dict:
    mapping = _header_map(header)
    if role == "pob":
        arrival = _get(cells, mapping, "Arrival Date", "Arrival", "Onboard Date", "Boarding Date",
                       "Check In", "Start Date", "Start Time")
        departure = _get(cells, mapping, "Departure Date", "Departure", "Offboard Date",
                         "Check Out", "End Date", "End Time")
        if not arrival:
            arrival, _arrival_precision = _component_date(cells, mapping, "ARRIVAL", inferred_year)
        else:
            arrival = _date_text(arrival)
        if not departure:
            departure, _departure_precision = _component_date(cells, mapping, "DEPARTURE", inferred_year)
        else:
            departure = _date_text(departure)
        return {
            "source_workbook": workbook,
            "source_sheet": sheet,
            "location": _get(cells, mapping, "Location", "POB Location", "Current Location", "Facility",
                             "Installation", "Asset", "Asset Name", "Asset Description"),
            "region": _get(cells, mapping, "Region"),
            "area": _get(cells, mapping, "Area"),
            "company": _get(cells, mapping, "Company", "Employer", "Organization", "Vendor"),
            "position": _get(cells, mapping, "Position", "Position Text", "Job Title", "Role"),
            "arrival_date": arrival,
            "departure_date": departure,
        }
    start_date, start_precision = _component_date(cells, mapping, "START TIME", inferred_year)
    end_date, end_precision = _component_date(cells, mapping, "END TIME", inferred_year)
    if not start_date:
        start_date, start_precision = _component_date(cells, mapping, "DEPARTURE TIME", inferred_year)
    if not end_date:
        end_date, end_precision = _component_date(cells, mapping, "ARRIVAL TIME", inferred_year)
    single_date = _get(cells, mapping, "Flight Date", "Trip Date", "Reservation Date", "Event Date", "Date")
    if single_date and not (start_date or end_date):
        parsed = _date_text(single_date)
        if _date_bounds(parsed)[0]:
            start_date = end_date = parsed
            start_precision = end_precision = ("day" if len(parsed) == 10 else
                                                "month" if len(parsed) == 7 else "year")
    precision = end_precision if end_precision != "unknown" else start_precision
    return {
        "source_workbook": workbook,
        "source_sheet": sheet,
        "asset": _get(cells, mapping, "ASSET", "Asset Name", "Asset Description", "AssetDescription",
                      "Facility", "Installation"),
        "origin": _get(cells, mapping, "ORIGIN", "Origin Location", "From", "From Location",
                       "Departure Location"),
        "destination": _get(cells, mapping, "DESTINATION", "Destination Location", "To", "To Location",
                            "Arrival Location"),
        "trip_id": _get(cells, mapping, "TRIP ID", "TripId", "Flight ID", "Flight Number",
                        "Reservation ID"),
        "company": _get(cells, mapping, "COMPANY", "Employer", "Organization", "Vendor"),
        "start_date": start_date,
        "end_date": end_date,
        "date_precision": precision,
    }


def _manifest_workbook(manifest: dict, path: str) -> dict:
    wanted = str(Path(path))
    for workbook in manifest.get("workbooks") or []:
        if str(Path(workbook.get("path", ""))) == wanted:
            return workbook
    return {}


def _source_year(path: str, audit_year: str) -> int | None:
    years = [int(value) for value in re.findall(r"\b(20\d{2})\b", Path(path).name)]
    if years:
        return years[0]
    try:
        return int(audit_year) - 1
    except (TypeError, ValueError):
        return None


def _date_bounds(value: str) -> tuple[date | None, date | None]:
    text = str(value or "")
    day = _parse_day(text)
    if day:
        return day, day
    match = re.fullmatch(r"(\d{4})-(\d{2})", text)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        try:
            first = date(year, month, 1)
            return first, date(year, month, calendar.monthrange(year, month)[1])
        except ValueError:
            return None, None
    if re.fullmatch(r"\d{4}", text):
        year = int(text)
        return date(year, 1, 1), date(year, 12, 31)
    return None, None


def _scan_source(role: str, path: str, people: list[str], manifest: dict,
                 warnings: list[str], audit_year: str = "") -> tuple[
                     list[dict], dict[str, _CandidateAccumulator], int, dict]:
    wb_info = _manifest_workbook(manifest, path)
    shape_by_tab: dict[str, dict] = {}
    for shape in wb_info.get("shapes") or []:
        for tab in shape.get("tabs") or []:
            shape_by_tab[tab] = shape

    candidates: dict[str, _CandidateAccumulator] = {}
    entries: list[dict] = []
    rows_scanned = 0
    coverage_dates: list[str] = []
    inferred_year = _source_year(path, audit_year)
    workbook, _engine = open_workbook_any(Path(path))
    try:
        sheets, _dropped = _usable_worksheets(workbook, source=Path(path).name, warn=warnings.append)
        for ws in sheets:
            tab = getattr(ws, "title", "") or ""
            shape = shape_by_tab.get(tab) or {}
            header_row = shape.get("header_row")
            header = list(shape.get("header") or [])
            classified = classify_columns(header)
            subject = classified.get("subject")
            if header_row is None or not header or not subject:
                continue
            header_names = set(_header_map(header))
            compact_names = {re.sub(r"[^a-z0-9]", "", value) for value in header_names}
            aliases = (("location", "pob location", "current location", "facility", "installation",
                        "asset", "asset name", "asset description") if role == "pob" else
                       ("asset", "asset name", "asset description", "origin", "origin location",
                        "from", "from location", "destination", "destination location", "to", "to location"))
            if not any(
                _norm(alias) in header_names
                or re.sub(r"[^a-z0-9]", "", _norm(alias)) in compact_names
                for alias in aliases
            ):
                warnings.append(f"skipped {Path(path).name}!{tab}: name column exists but no "
                                f"recognized {role} evidence columns")
                continue
            values: list[str] = []
            seen_values: set[str] = set()
            for i, row in enumerate(ws.iter_rows(values_only=True)):
                if i <= int(header_row):
                    continue
                cells = list(row)
                if _is_banner(cells):
                    continue
                rows_scanned += 1
                row_event = _event_for(
                    role, cells, header, workbook=Path(path).name, sheet=tab,
                    inferred_year=inferred_year,
                )
                date_keys = (("arrival_date", "departure_date") if role == "pob"
                             else ("start_date", "end_date"))
                for key in date_keys:
                    value = str(row_event.get(key) or "")
                    lower, upper = _date_bounds(value)
                    if lower:
                        coverage_dates.append(lower.isoformat())
                    if upper:
                        coverage_dates.append(upper.isoformat())
                for raw_name in _subject_values(cells, subject):
                    if not _candidate_people(raw_name, people):
                        continue
                    key = _norm(raw_name)
                    if key not in seen_values:
                        seen_values.add(key)
                        values.append(raw_name)
                    acc = candidates.setdefault(key, _CandidateAccumulator(raw_name=raw_name))
                    acc.add(row_event, i + 1)
            entries.append({
                "path": path, "name": Path(path).name, "tab": tab, "header_row": header_row,
                "header": header, "subject": subject, "values": values,
            })
    finally:
        try:
            workbook.close()
        except Exception:
            pass
    if not entries:
        raise ValueError(f"no usable name-bearing {role} sheet with recognized evidence columns")
    return entries, candidates, rows_scanned, {
        "start_date": min(coverage_dates) if coverage_dates else "",
        "end_date": max(coverage_dates) if coverage_dates else "",
        "dated_rows": len(coverage_dates),
    }


def _parse_day(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _merge_pob(events: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for event in events:
        key = tuple(event.get(k, "") for k in ("location", "region", "area", "company", "position"))
        groups[key].append(event)
    merged: list[dict] = []
    for group in groups.values():
        group.sort(key=lambda e: (e.get("arrival_date", ""), e.get("departure_date", "")))
        for event in group:
            if not merged or tuple(merged[-1].get(k, "") for k in
                                   ("location", "region", "area", "company", "position")) != tuple(
                                       event.get(k, "") for k in
                                       ("location", "region", "area", "company", "position")):
                merged.append(dict(event))
                continue
            previous = merged[-1]
            prev_end = _parse_day(previous.get("departure_date", ""))
            next_start = _parse_day(event.get("arrival_date", ""))
            if not prev_end or not next_start or next_start > prev_end + timedelta(days=1):
                merged.append(dict(event))
                continue
            next_end = _parse_day(event.get("departure_date", ""))
            if next_end and next_end > prev_end:
                previous["departure_date"] = next_end.isoformat()
            previous["duplicate_count"] = int(previous.get("duplicate_count", 0)) + int(
                event.get("duplicate_count", 0))
            previous["last_source_row"] = event.get("last_source_row", previous.get("last_source_row"))
            reps = list(previous.get("representative_source_rows") or [])
            for row in event.get("representative_source_rows") or []:
                if row not in reps and len(reps) < _REPRESENTATIVE_ROWS:
                    reps.append(row)
            previous["representative_source_rows"] = reps
    return merged


def _route_summaries(events: list[dict]) -> list[dict]:
    routes: dict[tuple, dict] = {}
    for event in events:
        key = tuple(event.get(k, "") for k in ("asset", "origin", "destination", "company"))
        route = routes.setdefault(key, {
            "asset": key[0], "origin": key[1], "destination": key[2], "company": key[3],
            "first_date": "", "last_date": "", "trip_count": 0,
        })
        route["trip_count"] += int(event.get("duplicate_count", 1))
        dates = [d for d in (event.get("start_date", ""), event.get("end_date", "")) if d]
        for value in dates:
            if not route["first_date"] or value < route["first_date"]:
                route["first_date"] = value
            if not route["last_date"] or value > route["last_date"]:
                route["last_date"] = value
    return sorted(routes.values(), key=lambda r: (r["asset"], r["origin"], r["destination"]))


def _scratch_dir(run_id: str) -> Path:
    target = ORCH_DIR / ".otl_scratch" / (run_id or "run") / "location"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _write_artifact(artifact: dict, run_id: str) -> str:
    target = _scratch_dir(run_id) / "location_evidence.json"
    temporary = target.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, default=str, ensure_ascii=False, separators=(",", ":"))
    with temporary.open(encoding="utf-8") as fh:
        json.load(fh)
    temporary.replace(target)
    return str(target)


def extract_location_evidence(*, run_id: str, audit_year: str, charged_asset: str,
                              people: list[str], pob_paths: list[str], heli_paths: list[str],
                              manifest: dict, charge_period_start: str = "",
                              charge_period_end: str = "") -> ExtractionResult:
    warnings: list[str] = []
    scanned: dict[str, int] = {"pob": 0, "heli": 0}
    source_payloads: list[dict] = []
    selected_by_role: dict[str, dict] = {}
    candidate_accumulators: dict[str, dict[str, _CandidateAccumulator]] = {}
    unresolved: list[str] = []
    source_coverage: dict[str, dict] = {
        "pob": {"start_date": "", "end_date": "", "dated_rows": 0, "successful_sources": 0},
        "heli": {"start_date": "", "end_date": "", "dated_rows": 0, "successful_sources": 0},
    }

    if not pob_paths:
        unresolved.append("missing_source:pob")
    if not heli_paths:
        unresolved.append("missing_source:heli")

    for role, paths in (("pob", pob_paths), ("heli", heli_paths)):
        all_entries: list[dict] = []
        all_candidates: dict[str, _CandidateAccumulator] = {}
        for path in paths:
            try:
                entries, candidates, rows, coverage = _scan_source(
                    role, path, people, manifest, warnings, audit_year)
            except Exception as exc:
                warnings.append(f"{role} extraction failed for {Path(path).name}: {type(exc).__name__}: {exc}")
                unresolved.append(f"unsupported_schema:{role}:{Path(path).name}")
                continue
            all_entries.extend(entries)
            scanned[role] += rows
            role_coverage = source_coverage[role]
            role_coverage["successful_sources"] += 1
            role_coverage["dated_rows"] += int(coverage.get("dated_rows") or 0)
            start, end = str(coverage.get("start_date") or ""), str(coverage.get("end_date") or "")
            if start and (not role_coverage["start_date"] or start < role_coverage["start_date"]):
                role_coverage["start_date"] = start
            if end and (not role_coverage["end_date"] or end > role_coverage["end_date"]):
                role_coverage["end_date"] = end
            for key, acc in candidates.items():
                existing = all_candidates.get(key)
                if existing is None:
                    all_candidates[key] = acc
                else:
                    existing.row_count += acc.row_count
                    for signature, event in acc.events.items():
                        if signature not in existing.events:
                            existing.events[signature] = event
                        else:
                            current = existing.events[signature]
                            current.duplicate_count += event.duplicate_count
                            current.last_source_row = event.last_source_row
        candidate_accumulators[role] = all_candidates
        selected, _chunks = select_names("person_index", role, all_entries, people, warn=warnings.append)
        selected_by_role[role] = selected
        for person in people:
            candidates_for_person = [a.raw_name for a in all_candidates.values()
                                     if person in _candidate_people(a.raw_name, [person])]
            matches = selected.get(person) or []
            if candidates_for_person and not matches:
                unresolved.append(f"unresolved_identity:{role}:{person}")
            if any(str(m.get("confidence", "")).lower() == "low" for m in matches):
                unresolved.append(f"low_confidence_identity:{role}:{person}")

    people_payload: dict[str, dict] = {}
    identities: dict[str, dict] = {}
    event_ids: list[str] = []
    for person in people:
        identities[person] = {}
        person_payload = {"pob": {"raw_row_count": 0, "intervals": []},
                          "heli": {"raw_row_count": 0, "events": [], "routes": []}}
        for role in ("pob", "heli"):
            matches = selected_by_role.get(role, {}).get(person) or []
            identities[person][role] = {
                "selected_names": [m.get("name_in_file", "") for m in matches],
                "confidence": ", ".join(str(m.get("confidence", "")) for m in matches if m.get("confidence")),
                "rationale": "; ".join(str(m.get("why", "")) for m in matches if m.get("why")),
            }
            selected_events: list[dict] = []
            raw_count = 0
            for match in matches:
                acc = candidate_accumulators.get(role, {}).get(_norm(match.get("name_in_file", "")))
                if not acc:
                    continue
                raw_count += acc.row_count
                selected_events.extend(event.payload() for event in acc.events.values())
            if role == "pob":
                selected_events = _merge_pob(selected_events)
                for index, event in enumerate(selected_events, 1):
                    event["evidence_id"] = f"pob:{_norm(person).replace(' ', '-')}:{index:04d}"
                    event_ids.append(event["evidence_id"])
                person_payload[role] = {"raw_row_count": raw_count, "intervals": selected_events}
            else:
                selected_events.sort(key=lambda e: (e.get("start_date", ""), e.get("end_date", ""),
                                                    e.get("asset", ""), e.get("origin", "")))
                for index, event in enumerate(selected_events, 1):
                    event["evidence_id"] = f"heli:{_norm(person).replace(' ', '-')}:{index:04d}"
                    event_ids.append(event["evidence_id"])
                person_payload[role] = {"raw_row_count": raw_count, "events": selected_events,
                                        "routes": _route_summaries(selected_events)}
        people_payload[person] = person_payload

    for role, paths in (("pob", pob_paths), ("heli", heli_paths)):
        for path in paths:
            info = _manifest_workbook(manifest, path)
            source_payloads.append({
                "role": role, "path": path, "workbook_name": Path(path).name,
                "sheets": list(info.get("sheets") or []), "rows_scanned": scanned[role],
                "schema_status": "supported" if info else "unknown",
            })

    artifact = {
        "schema_version": 1,
        "case": {"audit_year": audit_year, "charged_asset": charged_asset,
                 "charge_period_start": charge_period_start, "charge_period_end": charge_period_end},
        "sources": source_payloads,
        "identity_selection": identities,
        "people": people_payload,
        "warnings": warnings,
        "unresolved_reasons": sorted(set(unresolved)),
        "source_coverage": source_coverage,
        "evidence_ids": event_ids,
        "extraction_statistics": {
            "rows_scanned": scanned,
            "candidate_names": {role: len(values) for role, values in candidate_accumulators.items()},
            "normalized_events": len(event_ids),
        },
    }
    path = _write_artifact(artifact, run_id)
    return ExtractionResult(artifact=artifact, artifact_path=path,
                            unresolved_reasons=artifact["unresolved_reasons"], rows_scanned=scanned)


def classify_location_statuses(artifact: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Deterministically classify onsite/offsite only when identity and period coverage permit it."""
    case = artifact.get("case") or {}
    period_start = _parse_day(str(case.get("charge_period_start") or ""))
    period_end = _parse_day(str(case.get("charge_period_end") or ""))
    coverage = artifact.get("source_coverage") or {}
    unresolved = list(artifact.get("unresolved_reasons") or [])
    statuses: dict[str, str] = {}
    rationales: dict[str, str] = {}

    def covers(role: str) -> bool:
        row = coverage.get(role) or {}
        start, end = _parse_day(str(row.get("start_date") or "")), _parse_day(str(row.get("end_date") or ""))
        return bool(
            period_start and period_end and start and end
            and int(row.get("successful_sources") or 0) > 0
            and start <= period_start and end >= period_end
        )

    def overlaps(event: dict, role: str) -> bool:
        keys = (("arrival_date", "departure_date") if role == "pob" else ("start_date", "end_date"))
        start_lower, start_upper = _date_bounds(str(event.get(keys[0]) or ""))
        end_lower, end_upper = _date_bounds(str(event.get(keys[1]) or ""))
        start = start_lower or end_lower
        end = end_upper or start_upper or start
        if not (period_start and period_end and start and end):
            return False
        precision = str(event.get("date_precision") or "day")
        if precision in {"month", "year"}:
            # A partial date proves presence only when the entire possible interval falls inside the
            # questioned period. Otherwise the event may have occurred outside a narrower date window.
            return period_start <= start and period_end >= end
        return start <= period_end and end >= period_start

    def partial_date_may_overlap(event: dict, role: str) -> bool:
        if str(event.get("date_precision") or "day") not in {"month", "year"}:
            return False
        keys = (("arrival_date", "departure_date") if role == "pob" else ("start_date", "end_date"))
        lower_a, upper_a = _date_bounds(str(event.get(keys[0]) or ""))
        lower_b, upper_b = _date_bounds(str(event.get(keys[1]) or ""))
        lower, upper = lower_a or lower_b, upper_b or upper_a
        return bool(period_start and period_end and lower and upper
                    and lower <= period_end and upper >= period_start)

    for person, payload in (artifact.get("people") or {}).items():
        identity = (artifact.get("identity_selection") or {}).get(person) or {}
        ambiguous_roles: list[str] = []
        for role in ("pob", "heli"):
            selected = list((identity.get(role) or {}).get("selected_names") or [])
            confidence = str((identity.get(role) or {}).get("confidence") or "").casefold()
            person_reasons = [reason for reason in unresolved if reason.endswith(f":{person}") and f":{role}:" in reason]
            selected_token_sets = {_tokens(value) and frozenset(_tokens(value)) for value in selected if value}
            selected_token_sets.discard(False)
            person_tokens = frozenset(_tokens(person))
            exact_aliases = bool(selected_token_sets) and all(tokens == person_tokens for tokens in selected_token_sets)
            if ((len(selected_token_sets) > 1 and not exact_aliases)
                    or "low" in confidence or person_reasons):
                ambiguous_roles.append(role)
        if ambiguous_roles:
            statuses[person] = "unresolved"
            rationales[person] = (
                "Identity could not be resolved to one definitive source name in "
                + " and ".join(ambiguous_roles) + "."
            )
            continue
        pob_events = list((payload.get("pob") or {}).get("intervals") or [])
        heli_events = list((payload.get("heli") or {}).get("events") or [])
        if any(overlaps(event, "pob") for event in pob_events) or any(
            overlaps(event, "heli") for event in heli_events
        ):
            statuses[person] = "onsite"
            rationales[person] = "A definitive POB or helicopter record overlaps the questioned period."
            continue
        if any(partial_date_may_overlap(event, "pob") for event in pob_events) or any(
            partial_date_may_overlap(event, "heli") for event in heli_events
        ):
            statuses[person] = "unresolved"
            rationales[person] = (
                "A matching location record may overlap the questioned period, but its date precision is "
                "not sufficient for a definitive onsite or offsite conclusion."
            )
            continue
        if not (period_start and period_end):
            statuses[person] = "unresolved"
            rationales[person] = "The questioned period could not be established from the case or ledger."
            continue
        if not (covers("pob") and covers("heli")):
            statuses[person] = "unresolved"
            rationales[person] = "The supplied POB and helicopter records do not both cover the full questioned period."
            continue
        undated_selected = any(
            not (_date_bounds(str(event.get("arrival_date") or ""))[0] and
                 (_date_bounds(str(event.get("departure_date") or ""))[1] or
                  _date_bounds(str(event.get("arrival_date") or ""))[1]))
            for event in pob_events
        ) or any(
            not (_date_bounds(str(event.get("start_date") or ""))[0] or
                 _date_bounds(str(event.get("end_date") or ""))[0])
            for event in heli_events
        )
        if undated_selected:
            statuses[person] = "unresolved"
            rationales[person] = "A selected identity has location records without sufficient date precision."
            continue
        statuses[person] = "offsite"
        rationales[person] = (
            "The person has no definitive POB or helicopter record during the questioned period, and both "
            "source populations cover that period."
        )
    return statuses, rationales


def artifact_prompt_json(artifact: dict) -> str:
    return json.dumps(artifact, default=str, ensure_ascii=False, separators=(",", ":"))


def artifact_tokens(artifact: dict) -> int:
    return llm.estimate_text_tokens(artifact_prompt_json(artifact))


def expected_evidence_ids(artifact: dict) -> list[str]:
    return list(artifact.get("evidence_ids") or [])


def build_shards(artifact: dict, target_tokens: int = PACKET_TARGET_TOKENS,
                 max_tokens: int = PACKET_MAX_TOKENS) -> list[dict]:
    base = {"schema_version": artifact.get("schema_version"), "case": artifact.get("case"),
            "identity_selection": artifact.get("identity_selection"), "warnings": artifact.get("warnings")}
    by_person: dict[str, list[dict]] = {}
    for person, payload in (artifact.get("people") or {}).items():
        items: list[dict] = []
        for event in (payload.get("pob") or {}).get("intervals") or []:
            items.append({"person": person, "role": "pob", "event": event})
        for event in (payload.get("heli") or {}).get("events") or []:
            items.append({"person": person, "role": "heli", "event": event})
        items.sort(key=lambda item: (
            item["role"],
            item["event"].get("arrival_date") or item["event"].get("start_date") or
            item["event"].get("departure_date") or item["event"].get("end_date") or "",
            item["event"].get("asset") or item["event"].get("location") or "",
            item["event"].get("origin", ""), item["event"].get("destination", ""),
        ))
        by_person[person] = items
    shards: list[dict] = []
    for person, items in by_person.items():
        current: list[dict] = []
        person_base = base | {"person": person}
        current_tokens = llm.estimate_text_tokens(artifact_prompt_json(person_base))
        for item in items:
            item_tokens = llm.estimate_text_tokens(artifact_prompt_json(item))
            if current and current_tokens + item_tokens > target_tokens:
                shards.append(person_base | {"shard_id": f"location-{len(shards) + 1:03d}",
                                              "events": current})
                current, current_tokens = [], llm.estimate_text_tokens(artifact_prompt_json(person_base))
            if item_tokens + current_tokens > max_tokens:
                raise ValueError(f"one normalized location event exceeds the {max_tokens}-token shard limit")
            current.append(item)
            current_tokens += item_tokens
        if current:
            shards.append(person_base | {"shard_id": f"location-{len(shards) + 1:03d}",
                                          "events": current})
    if not shards:
        shards.append(base | {"shard_id": "location-001", "person": "", "events": []})
    return shards


def decision_system_prompt() -> str:
    return (
        "You are the location evidence specialist. Analyze the supplied normalized POB and helicopter "
        "evidence. Use every evidence item, preserve date precision, and never infer missing dates. "
        "When deterministic_location_status is supplied, treat it as authoritative: onsite means a "
        "definitive identity/event overlaps the questioned period; offsite means both source populations "
        "cover the period and neither contains the person; unresolved means identity or coverage is not "
        "definitive. Never describe a definitive offsite result as missing evidence. "
        "POB arrival/departure controls start_date/end_date when POB evidence exists; otherwise use the "
        "earliest/latest sufficiently precise heli evidence. Report clearly whether the established "
        "locations clear the charged asset. Set needs_code_execution only when normalized evidence is "
        "insufficient or materially contradictory and raw-row investigation is necessary."
    )


__all__ = [
    "ExtractionResult", "LocationDecision", "LocationShardFinding", "PACKET_MAX_TOKENS",
    "artifact_prompt_json", "artifact_tokens", "build_shards", "classify_location_statuses", "decision_system_prompt",
    "expected_evidence_ids", "extract_location_evidence",
]
