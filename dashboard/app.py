"""Shell JV Audit token-spend dashboard.

The dashboard intentionally uses only the Python standard library at runtime.  It serves a small
JSON API and the static frontend in ``dashboard/static``.  Output discovery follows the same
environment variables as the two pipelines:

* extractor: ``SHAREPOINT_READ_FOLDER`` (or ``SHAREPOINT_FOLDER``) / ``INPUT_BASE``
* inference: ``SHAREPOINT_WRITE_FOLDER`` / ``OUTPUT_BASE``

The home page reads a persistent SQLite metric index. Pipeline runs populate that index before
writing their JSON artifact; an explicit one-time import can seed metrics from older output folders.
"""
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
import sys
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, quote, unquote, urlparse

from shared.token_metrics_cache import (
    OutputScope,
    cache_report_metrics,
    get_metric_record,
    list_metric_records,
    metrics_from_report,
    source_url as cached_source_url,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
ASSET_ROOT = Path(__file__).resolve().parent
CASE_CSV = REPO_ROOT / "datastores" / "labor_classification_test_set_temporal_split.csv"

_OUTPUT_RE = re.compile(r"^rebuttal_(\d+)_(synthesis|otl)\.json$", re.IGNORECASE)
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")


def load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    """Load simple KEY=VALUE entries without overriding explicit process environment values."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def normalized_asset(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def display_exception_id(value: Any) -> str:
    """Normalize CSV ids such as ``3.0`` to the pipeline folder form ``0003``."""
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        decimal = Decimal(text.replace(",", ""))
        if decimal == decimal.to_integral_value():
            integer = str(int(decimal))
            return integer.zfill(4) if len(integer) < 4 else integer
    except (InvalidOperation, ValueError):
        pass
    digits = re.sub(r"\D", "", text)
    return digits.zfill(4) if digits and len(digits) < 4 else (digits or text)


def case_key(year: Any, asset: Any, exception_id: Any) -> str:
    return f"{str(year).strip()}::{normalized_asset(asset)}::{display_exception_id(exception_id)}"


@dataclass
class Metrics:
    artifacts: int = 0
    priced_artifacts: int = 0
    calls: int = 0
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0
    input_cost_usd: float = 0.0
    cache_read_cost_usd: float = 0.0
    cache_write_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    uncached_cost_usd: float = 0.0
    cache_savings_usd: float = 0.0

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "Metrics":
        return cls(**{
            name: values.get(name, 0) or 0
            for name in cls.__dataclass_fields__
        })

    def add(self, other: "Metrics") -> "Metrics":
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))
        return self

    def payload(self) -> dict[str, Any]:
        data = asdict(self)
        for key in list(data):
            if key.endswith("_usd"):
                data[key] = round(float(data[key]), 6)
        return data


def extract_metrics(data: dict[str, Any]) -> Metrics:
    return Metrics.from_mapping(metrics_from_report(data))


# Provider usage keys -> dashboard metric names, per model tier.
TIER_USAGE_FIELDS = {
    "calls": "calls",
    "tokens_in": "input_tokens",
    "cache_read_tokens": "cache_read_tokens",
    "cache_creation_tokens": "cache_write_tokens",
    "tokens_out": "output_tokens",
    "thinking_tokens": "thinking_tokens",
    "tokens_total": "total_tokens",
}
TIER_COST_FIELDS = {
    "input": "input_cost_usd",
    "cache_read": "cache_read_cost_usd",
    "cache_write": "cache_write_cost_usd",
    "output": "output_cost_usd",
    "total": "total_cost_usd",
}


def tier_metrics(token_usage_json: str | None) -> dict[str, dict[str, Any]]:
    """Flatten one cached record's usage/cost `by_tier` blocks into per-tier metric dicts."""
    try:
        usage = json.loads(token_usage_json or "{}")
    except (TypeError, ValueError):
        return {}
    usage_by_tier = usage.get("by_tier") or {}
    cost_by_tier = (usage.get("cost") or {}).get("by_tier") or {}
    tiers: dict[str, dict[str, Any]] = {}
    for name in {*usage_by_tier, *cost_by_tier}:
        u = usage_by_tier.get(name) or {}
        c = cost_by_tier.get(name) or {}
        entry: dict[str, Any] = {"model": u.get("model") or c.get("model") or ""}
        for source, target in TIER_USAGE_FIELDS.items():
            entry[target] = int(u.get(source) or 0)
        for source, target in TIER_COST_FIELDS.items():
            entry[target] = float(c.get(source) or 0.0)
        tiers[str(name)] = entry
    return tiers


def add_tier_metrics(total: dict[str, dict[str, Any]], other: dict[str, dict[str, Any]]) -> None:
    for name, values in other.items():
        entry = total.setdefault(name, {"model": ""})
        entry["model"] = values.get("model") or entry["model"]
        for key, value in values.items():
            if key != "model":
                entry[key] = entry.get(key, 0) + value


@dataclass(frozen=True)
class ArtifactRef:
    section: str
    relative_path: str
    year: str
    asset: str
    exception_id: str
    cycle: int
    kind: str
    source_url: str
    record_id: int = 0
    run_key: str = ""
    run_id: str = ""
    folder_key: str = ""
    target: str = ""
    folder_path: str = ""
    site_url: str = ""

    @property
    def key(self) -> str:
        return case_key(self.year, self.asset, self.exception_id)


def ref_from_path(section: str, relative_path: str, source_url: str) -> ArtifactRef | None:
    parts = [part for part in relative_path.replace("\\", "/").split("/") if part]
    if len(parts) < 4:
        return None
    match = _OUTPUT_RE.match(parts[-1])
    if not match:
        return None
    expected_kind = "synthesis" if section == "extractor" else "otl"
    if match.group(2).lower() != expected_kind:
        return None
    year, asset, exception_id = parts[-4], parts[-3], parts[-2]
    if not _YEAR_RE.match(year):
        return None
    return ArtifactRef(
        section=section,
        relative_path=relative_path.replace("\\", "/"),
        year=year,
        asset=asset,
        exception_id=display_exception_id(exception_id),
        cycle=int(match.group(1)),
        kind=match.group(2).lower(),
        source_url=source_url,
    )


class OutputSource:
    def __init__(self, section: str, target: str, location: str):
        self.section = section
        self.target = target
        self.location = location

    def list_artifacts(self) -> list[ArtifactRef]:
        raise NotImplementedError

    def read_json(self, ref: ArtifactRef) -> dict[str, Any]:
        raise NotImplementedError

    def description(self) -> dict[str, str]:
        return {"target": self.target, "location": self.location}


class LocalOutputSource(OutputSource):
    def __init__(self, section: str, location: str):
        super().__init__(section, "local", location)
        self.root = Path(location).expanduser() if location else Path("/__missing_dashboard_root__")

    def _safe_path(self, relative_path: str) -> Path:
        root = self.root.resolve()
        candidate = (root / relative_path).resolve()
        if root != candidate and root not in candidate.parents:
            raise ValueError("Artifact path escaped its configured output root")
        return candidate

    def list_artifacts(self) -> list[ArtifactRef]:
        if not self.root.is_dir():
            return []
        suffix = "*_synthesis.json" if self.section == "extractor" else "*_otl.json"
        refs: list[ArtifactRef] = []
        for path in self.root.rglob(suffix):
            rel = path.relative_to(self.root).as_posix()
            raw_url = (
                f"/api/artifact?section={quote(self.section, safe='')}"
                f"&path={quote(rel, safe='/')}"
            )
            ref = ref_from_path(self.section, rel, raw_url)
            if ref:
                refs.append(ref)
        return sorted(refs, key=lambda r: (r.year, normalized_asset(r.asset), r.exception_id, r.cycle))

    def read_json(self, ref: ArtifactRef) -> dict[str, Any]:
        return json.loads(self._safe_path(ref.relative_path).read_text(encoding="utf-8"))


class SharePointOutputSource(OutputSource):
    def __init__(self, section: str, site_url: str, folder: str):
        super().__init__(section, "sharepoint", folder)
        self.site_url = site_url.rstrip("/")
        self.folder = folder.strip("/")
        self._writer: Any = None
        self._lock = threading.Lock()

    def _get_writer(self):
        if self._writer is None:
            extractor_path = str(REPO_ROOT / "excel_extractor")
            if extractor_path not in sys.path:
                sys.path.insert(0, extractor_path)
            from shared.sharepoint import SharePointWriter

            self._writer = SharePointWriter(self.site_url, self.folder)
        return self._writer

    def _walk(self, folder_url: str) -> Iterable[tuple[str, str]]:
        from shared.sharepoint import _sp_retry

        writer = self._get_writer()
        ctx = writer.ctx
        folder = ctx.web.get_folder_by_server_relative_url(folder_url)
        folders, files = folder.folders, folder.files
        ctx.load(folders)
        ctx.load(files)
        _sp_retry(lambda: ctx.execute_query())
        for file_obj in files:
            props = file_obj.properties
            name = str(props.get("Name") or "")
            server_url = str(props.get("ServerRelativeUrl") or f"{folder_url}/{name}")
            yield name, server_url
        for child in folders:
            props = child.properties
            name = str(props.get("Name") or "")
            if name.lower() == "forms":
                continue
            server_url = str(props.get("ServerRelativeUrl") or f"{folder_url}/{name}")
            yield from self._walk(server_url)

    def list_artifacts(self) -> list[ArtifactRef]:
        if not self.site_url or not self.folder:
            return []
        writer = self._get_writer()
        refs: list[ArtifactRef] = []
        with self._lock:
            for name, server_url in self._walk(writer.base):
                if not _OUTPUT_RE.match(name):
                    continue
                rel = server_url[len(writer.base):].lstrip("/")
                web_url = self.site_url + "/" + quote(self.folder + "/" + rel, safe="/")
                ref = ref_from_path(self.section, rel, web_url)
                if ref:
                    refs.append(ref)
        return sorted(refs, key=lambda r: (r.year, normalized_asset(r.asset), r.exception_id, r.cycle))

    def read_json(self, ref: ArtifactRef) -> dict[str, Any]:
        from shared.sharepoint import _sp_retry

        writer = self._get_writer()
        file_url = f"{writer.base}/{ref.relative_path.strip('/')}"
        with self._lock:
            binary = writer.ctx.web.get_file_by_server_relative_url(file_url).get_content()
            _sp_retry(lambda: writer.ctx.execute_query())
        raw = binary.value if hasattr(binary, "value") else binary
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    def description(self) -> dict[str, str]:
        data = super().description()
        data["site_url"] = self.site_url
        return data


@dataclass
class CaseRecord:
    key: str
    year: str
    asset: str
    exception_id: str
    category: str
    gross_credit_requested: str = ""
    source_file: str = ""
    description: str = ""
    run_key: str = ""
    run_id: str = ""
    folder_path: str = ""
    folder_key: str = ""
    target: str = ""
    metrics: Metrics = field(default_factory=Metrics)
    tiers: dict[str, dict[str, Any]] = field(default_factory=dict)
    cycles: set[int] = field(default_factory=set)
    refs: list[ArtifactRef] = field(default_factory=list)

    def payload(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "year": self.year,
            "asset": self.asset,
            "exception_id": self.exception_id,
            "category": self.category,
            "gross_credit_requested": self.gross_credit_requested,
            "source_file": self.source_file,
            "description": self.description,
            "run_key": self.run_key,
            "run_id": self.run_id,
            "folder_path": self.folder_path,
            "folder_key": self.folder_key,
            "target": self.target,
            "cycles": sorted(self.cycles),
            "latest_cycle": max(self.cycles) if self.cycles else None,
            "metrics": self.metrics.payload(),
            "tiers": {
                name: {
                    key: round(value, 6) if key.endswith("_usd") else value
                    for key, value in values.items()
                }
                for name, values in self.tiers.items()
            },
        }


def load_catalog(path: Path = CASE_CSV) -> dict[str, CaseRecord]:
    catalog: dict[str, CaseRecord] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            year = str(row.get("year") or "").strip()
            asset = str(row.get("site") or row.get("asset") or "").strip()
            exception_id = display_exception_id(row.get("exception_ID") or row.get("exception_id"))
            category = str(
                row.get("type") or row.get("classification_category") or row.get("category") or "Uncategorized"
            ).strip()
            key = case_key(year, asset, exception_id)
            catalog[key] = CaseRecord(
                key=key,
                year=year,
                asset=asset,
                exception_id=exception_id,
                category=category or "Uncategorized",
                gross_credit_requested=str(row.get("gross_credit_requested") or "").strip(),
                source_file=str(row.get("source_file") or "").strip(),
                description=str(row.get("description") or "").strip(),
            )
    return catalog


class DashboardData:
    """Cache-backed dashboard model. Normal reads never touch SharePoint or output JSONs."""

    def __init__(self, sources: dict[str, OutputSource], csv_path: Path = CASE_CSV,
                 cache_db_path: str | Path | None = None):
        self.sources = sources
        self.csv_path = csv_path
        self.cache_db_path = cache_db_path
        self._lock = threading.Lock()
        self._summary: dict[str, Any] | None = None
        self._cases_by_section: dict[str, dict[str, CaseRecord]] = {}
        self._records_by_id: dict[int, dict[str, Any]] = {}

    @staticmethod
    def _run_key(folder_key: str, run_id: str) -> str:
        return f"{folder_key}::{run_id}"

    @staticmethod
    def _scope_for_source(section: str, source: OutputSource) -> OutputScope:
        description = source.description()
        return OutputScope(
            pipeline=section,
            target=source.target,
            folder_path=source.location,
            site_url=description.get("site_url", ""),
        )

    @staticmethod
    def _resolve_record_key(catalog: dict[str, CaseRecord], row: dict[str, Any]) -> str | None:
        exact = case_key(row.get("year"), row.get("asset"), row.get("exception_id"))
        if exact in catalog:
            return exact
        year = str(row.get("year") or "")
        wanted_id = display_exception_id(row.get("exception_id"))
        candidates = [
            key for key, record in catalog.items()
            if record.year == year and record.exception_id == wanted_id
        ]
        if len(candidates) == 1:
            return candidates[0]
        wanted_asset = normalized_asset(row.get("asset"))
        close = [
            key for key in candidates
            if wanted_asset in normalized_asset(catalog[key].asset)
            or normalized_asset(catalog[key].asset) in wanted_asset
        ]
        return close[0] if len(close) == 1 else None

    @staticmethod
    def _metric_from_row(row: dict[str, Any]) -> Metrics:
        values = dict(row)
        values["artifacts"] = 1
        return Metrics.from_mapping(values)

    def reindex_legacy_outputs(self) -> dict[str, Any]:
        """Explicit one-time backfill for artifacts created before pipeline cache writes existed."""
        imported: dict[str, int] = {}
        errors: dict[str, list[str]] = {}
        for section, source in self.sources.items():
            scope = self._scope_for_source(section, source)
            run_id = f"legacy-{Path(scope.folder_path).name or section}-{scope.folder_key[:8]}"
            imported[section] = 0
            errors[section] = []
            try:
                refs = source.list_artifacts()
            except Exception as exc:
                errors[section].append(f"{type(exc).__name__}: {exc}")
                continue
            for ref in refs:
                try:
                    data = source.read_json(ref)
                    cache_report_metrics(
                        pipeline=section, data=data,
                        year=str(data.get("year") or ref.year),
                        asset=str(data.get("asset") or ref.asset),
                        exception_id=display_exception_id(
                            data.get("exception_code") or data.get("exc_id") or ref.exception_id
                        ),
                        rebuttal_cycle=int(data.get("rebuttal_cycle") or ref.cycle),
                        relative_path=ref.relative_path, artifact_kind=ref.kind,
                        run_id=run_id, scope=scope, status="imported",
                        path=self.cache_db_path,
                    )
                    imported[section] += 1
                except Exception as exc:
                    errors[section].append(
                        f"{ref.relative_path}: {type(exc).__name__}: {exc}"
                    )
        self._summary = None
        return {"imported": imported, "errors": errors, "summary": self.refresh()}

    def refresh(self) -> dict[str, Any]:
        """Rebuild the in-memory view from SQLite only; this performs no output-folder I/O."""
        with self._lock:
            records = list_metric_records(path=self.cache_db_path)
            self._records_by_id = {int(row["record_id"]): row for row in records}
            sections: dict[str, Any] = {}
            cases_by_section: dict[str, dict[str, CaseRecord]] = {}
            categories: set[str] = set()

            for section, source in self.sources.items():
                section_records = [row for row in records if row.get("pipeline") == section]
                grouped: dict[str, list[dict[str, Any]]] = {}
                for row in section_records:
                    grouped.setdefault(
                        self._run_key(str(row["folder_key"]), str(row["run_id"])), []
                    ).append(row)

                run_meta: list[dict[str, Any]] = []
                flattened_cases: list[CaseRecord] = []
                section_case_map: dict[str, CaseRecord] = {}
                aggregate = Metrics()

                for run_key, run_rows in grouped.items():
                    first = run_rows[0]
                    catalog = load_catalog(self.csv_path)
                    for item in catalog.values():
                        item.run_key = run_key
                        item.run_id = str(first["run_id"])
                        item.folder_path = str(first["folder_path"])
                        item.folder_key = str(first["folder_key"])
                        item.target = str(first["target"])

                    for row in run_rows:
                        key = self._resolve_record_key(catalog, row)
                        if key is None:
                            key = case_key(row["year"], row["asset"], row["exception_id"])
                            catalog[key] = CaseRecord(
                                key=key,
                                year=str(row["year"]),
                                asset=str(row["asset"]),
                                exception_id=display_exception_id(row["exception_id"]),
                                category=str(row.get("category") or "Unmapped output"),
                                run_key=run_key,
                                run_id=str(row["run_id"]),
                                folder_path=str(row["folder_path"]),
                                folder_key=str(row["folder_key"]),
                                target=str(row["target"]),
                            )
                        case = catalog[key]
                        metric = self._metric_from_row(row)
                        case.metrics.add(metric)
                        add_tier_metrics(case.tiers, tier_metrics(row.get("token_usage_json")))
                        case.cycles.add(int(row["rebuttal_cycle"] or 0))
                        scope = OutputScope(
                            pipeline=section, target=str(row["target"]),
                            folder_path=str(row["folder_path"]), site_url=str(row["site_url"]),
                        )
                        output_location = str(row.get("output_location") or "")
                        fell_back_local = bool(
                            output_location and Path(output_location).is_absolute()
                        )
                        case.refs.append(ArtifactRef(
                            section=section,
                            relative_path=str(row["relative_path"]),
                            year=str(row["year"]), asset=str(row["asset"]),
                            exception_id=display_exception_id(row["exception_id"]),
                            cycle=int(row["rebuttal_cycle"] or 0),
                            kind=str(row["artifact_kind"]),
                            source_url=(
                                f"/api/cached-artifact?record_id={int(row['record_id'])}"
                                if fell_back_local else
                                cached_source_url(scope, str(row["relative_path"]),
                                                  int(row["record_id"]))
                            ),
                            record_id=int(row["record_id"]), run_key=run_key,
                            run_id=str(row["run_id"]), folder_key=str(row["folder_key"]),
                            target=str(row["target"]), folder_path=str(row["folder_path"]),
                            site_url=str(row["site_url"]),
                        ))
                        aggregate.add(metric)

                    updated_at = max(str(row["updated_at"]) for row in run_rows)
                    status_counts: dict[str, int] = {}
                    for row in run_rows:
                        status = str(row.get("write_status") or "cached")
                        status_counts[status] = status_counts.get(status, 0) + 1
                    run_meta.append({
                        "run_key": run_key,
                        "run_id": str(first["run_id"]),
                        "folder_key": str(first["folder_key"]),
                        "folder_path": str(first["folder_path"]),
                        "target": str(first["target"]),
                        "site_url": str(first["site_url"]),
                        "updated_at": updated_at,
                        "artifact_count": len(run_rows),
                        "status_counts": status_counts,
                        "cached": True,
                    })
                    for key, case in catalog.items():
                        flattened_cases.append(case)
                        section_case_map[f"{run_key}::{key}"] = case

                current_scope = self._scope_for_source(section, source)
                run_meta.sort(key=lambda item: item["updated_at"], reverse=True)
                matching = [item for item in run_meta if item["folder_key"] == current_scope.folder_key]
                default_run_key = (matching[0] if matching else (run_meta[0] if run_meta else None))
                default_run_key = default_run_key["run_key"] if default_run_key else ""

                if not run_meta:
                    virtual_run_key = f"current::{current_scope.folder_key}"
                    catalog = load_catalog(self.csv_path)
                    for key, case in catalog.items():
                        case.run_key = virtual_run_key
                        case.run_id = "No cached run"
                        case.folder_path = current_scope.folder_path
                        case.folder_key = current_scope.folder_key
                        case.target = current_scope.target
                        flattened_cases.append(case)
                        section_case_map[f"{virtual_run_key}::{key}"] = case
                    run_meta.append({
                        "run_key": virtual_run_key,
                        "run_id": "No cached run",
                        "folder_key": current_scope.folder_key,
                        "folder_path": current_scope.folder_path,
                        "target": current_scope.target,
                        "site_url": current_scope.site_url,
                        "updated_at": "",
                        "artifact_count": 0,
                        "status_counts": {},
                        "cached": False,
                    })
                    default_run_key = virtual_run_key

                flattened_cases.sort(
                    key=lambda row: (row.run_key, row.year, normalized_asset(row.asset), row.exception_id)
                )
                categories.update(row.category for row in flattened_cases)
                cases_by_section[section] = section_case_map
                sections[section] = {
                    "label": "Excel Extractor" if section == "extractor" else "Labor Inference",
                    "source": source.description(),
                    "aggregate": aggregate.payload(),
                    "cases": [
                        row.payload() for row in flattened_cases
                        if row.run_key == default_run_key
                    ],
                    "runs": run_meta,
                    "default_run_key": default_run_key,
                    "errors": [],
                    "error_count": 0,
                }

            self._cases_by_section = cases_by_section
            self._summary = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "cache_only": True,
                "record_count": len(records),
                "categories": sorted(categories),
                "sections": sections,
            }
            return self._summary

    def run_cases(self, section: str, run_key: str) -> dict[str, Any]:
        self.summary()
        rows = [
            case.payload() for case in self._cases_by_section.get(section, {}).values()
            if case.run_key == run_key
        ]
        if not rows:
            raise KeyError("Cached run not found")
        rows.sort(key=lambda row: (row["year"], normalized_asset(row["asset"]), row["exception_id"]))
        return {"section": section, "run_key": run_key, "cases": rows}

    def summary(self, force: bool = False) -> dict[str, Any]:
        if force or self._summary is None:
            return self.refresh()
        return self._summary

    def case_details(self, section: str, run_key: str, year: str, asset: str,
                     exception_id: str) -> dict[str, Any]:
        self.summary()
        key = f"{run_key}::{case_key(year, asset, exception_id)}"
        record = self._cases_by_section.get(section, {}).get(key)
        if record is None:
            candidates = [
                item for item in self._cases_by_section.get(section, {}).values()
                if item.run_key == run_key and item.year == str(year)
                and item.exception_id == display_exception_id(exception_id)
            ]
            if len(candidates) == 1:
                record = candidates[0]
        if record is None:
            raise KeyError("Cached case not found")

        artifacts: list[dict[str, Any]] = []
        for ref in sorted(record.refs, key=lambda item: (item.cycle, item.relative_path)):
            row = self._records_by_id.get(ref.record_id)
            if row is None:
                continue
            usage = json.loads(row.get("token_usage_json") or "{}")
            cost = usage.get("cost") or {}
            artifacts.append({
                "cycle": ref.cycle,
                "file_name": Path(ref.relative_path).name,
                "relative_path": ref.relative_path,
                "source_url": ref.source_url,
                "metrics": self._metric_from_row(row).payload(),
                "usage_by_tier": usage.get("by_tier") or {},
                "cost_by_tier": cost.get("by_tier") or {},
                "cost_metadata": {
                    "priced": bool(cost.get("priced")),
                    "currency": cost.get("currency") or "USD",
                    "rates_source": cost.get("rates_source") or "",
                    "as_of": cost.get("as_of") or "",
                    "cache_ttl": cost.get("cache_ttl") or "",
                    "unpriced_tiers": cost.get("unpriced_tiers") or [],
                    "model_mismatch": cost.get("model_mismatch") or [],
                    "mixed_model_tiers": cost.get("mixed_model_tiers") or [],
                },
                "estimator": row.get("estimator") or "",
                "elapsed_minutes": row.get("elapsed_minutes"),
                "write_status": row.get("write_status") or "cached",
                "result": json.loads(row.get("result_excerpt_json") or "{}"),
            })
        return {
            "section": section,
            "label": "Excel Extractor" if section == "extractor" else "Labor Inference",
            "case": record.payload(),
            "artifacts": artifacts,
        }

    def cached_artifact_json(self, record_id: int) -> dict[str, Any]:
        row = get_metric_record(record_id, path=self.cache_db_path)
        if row is None:
            raise KeyError("Cached artifact record not found")
        output_location = str(row.get("output_location") or "")
        if output_location and Path(output_location).is_absolute():
            output_path = Path(output_location)
            if not output_path.is_file():
                raise KeyError("Cached artifact's local output file no longer exists")
            return json.loads(output_path.read_text(encoding="utf-8"))
        section = str(row["pipeline"])
        if str(row["target"]) == "sharepoint":
            source: OutputSource = SharePointOutputSource(
                section, str(row["site_url"]), str(row["folder_path"])
            )
        else:
            source = LocalOutputSource(section, str(row["folder_path"]))
        ref = ArtifactRef(
            section=section, relative_path=str(row["relative_path"]),
            year=str(row["year"]), asset=str(row["asset"]),
            exception_id=display_exception_id(row["exception_id"]),
            cycle=int(row["rebuttal_cycle"] or 0), kind=str(row["artifact_kind"]),
            source_url="", record_id=int(row["record_id"]),
        )
        return source.read_json(ref)

def build_sources() -> dict[str, OutputSource]:
    target = (os.getenv("OUTPUT_TARGET") or "local").strip().lower()
    if target not in {"local", "sharepoint"}:
        raise ValueError(f"Unknown OUTPUT_TARGET={target!r}; expected local or sharepoint")
    if target == "sharepoint":
        site = os.getenv("SHAREPOINT_URL") or ""
        return {
            "extractor": SharePointOutputSource(
                "extractor", site,
                os.getenv("SHAREPOINT_READ_FOLDER") or os.getenv("SHAREPOINT_FOLDER") or "",
            ),
            "inference": SharePointOutputSource(
                "inference", site, os.getenv("SHAREPOINT_WRITE_FOLDER") or "",
            ),
        }
    return {
        "extractor": LocalOutputSource("extractor", os.getenv("INPUT_BASE") or ""),
        "inference": LocalOutputSource("inference", os.getenv("OUTPUT_BASE") or ""),
    }


class DashboardHandler(BaseHTTPRequestHandler):
    data: DashboardData

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, exc: Exception, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        self._json({"error": str(exc), "type": type(exc).__name__}, status)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/refresh":
            try:
                self._json(self.data.refresh())
            except Exception as exc:
                self._error(exc, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/health":
                self._json({"status": "ok"})
                return
            if parsed.path == "/api/summary":
                self._json(self.data.summary(force=query.get("refresh") == ["1"]))
                return
            if parsed.path == "/api/case":
                required = {name: unquote(query.get(name, [""])[0])
                            for name in ("section", "run_key", "year", "asset", "exception_id")}
                if not all(required.values()):
                    raise ValueError("section, run_key, year, asset, and exception_id are required")
                self._json(self.data.case_details(**required))
                return
            if parsed.path == "/api/run":
                section = unquote(query.get("section", [""])[0])
                run_key = unquote(query.get("run_key", [""])[0])
                if not section or not run_key:
                    raise ValueError("section and run_key are required")
                self._json(self.data.run_cases(section, run_key))
                return
            if parsed.path == "/api/cached-artifact":
                record_id = int(query.get("record_id", ["0"])[0])
                if record_id <= 0:
                    raise ValueError("record_id is required")
                self._json(self.data.cached_artifact_json(record_id))
                return
        except KeyError as exc:
            self._error(exc, HTTPStatus.NOT_FOUND)
            return
        except Exception as exc:
            self._error(exc, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._serve_static(parsed.path)

    def _serve_static(self, request_path: str) -> None:
        if request_path == "/shell_logo.svg":
            root = ASSET_ROOT.resolve()
            path = (root / "shell_logo.svg").resolve()
        else:
            relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
            root = STATIC_ROOT.resolve()
            path = (root / relative).resolve()
        if root != path and root not in path.parents:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Shell JV Audit token-spend dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--warm", action="store_true",
        help="load the local SQLite metric cache before opening the server",
    )
    parser.add_argument(
        "--legacy-backfill", action="store_true",
        help="explicitly scan the currently configured output folders once to seed pre-cache runs",
    )
    parser.add_argument(
        "--legacy-backfill-only", action="store_true",
        help="perform the explicit pre-cache output backfill and exit",
    )
    args = parser.parse_args()

    load_dotenv()
    data = DashboardData(build_sources())
    if args.legacy_backfill or args.legacy_backfill_only:
        result = data.reindex_legacy_outputs()
        print("Legacy backfill:", result["imported"])
        for section, errors in result["errors"].items():
            if errors:
                print(f"  {section}: {len(errors)} error(s); first: {errors[0]}")
        if args.legacy_backfill_only:
            return
    if args.warm:
        summary = data.refresh()
        print(f"Loaded {summary['record_count']} cached metric record(s)")

    handler = type("ConfiguredDashboardHandler", (DashboardHandler,), {"data": data})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Shell JV Audit dashboard: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
