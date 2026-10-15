"""Read and normalize the classified-exceptions catalog used as OTL's baseline case input."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from shared.case_keys import _norm, pad_exc

from shared.pipeline_config import configured_path, input_source_settings


_REBUTTAL_RE = re.compile(r"rebuttal\s+(\d+)", re.I)


def classified_exceptions_path() -> Path:
    settings = input_source_settings("classified_exceptions", default_enabled=True)
    value = settings.get("path") or "./datastores/classified_exceptions.json"
    return configured_path(str(value))


@lru_cache(maxsize=8)
def _load_catalog(path_text: str, _modified_ns: int, _size: int) -> dict[str, Any]:
    path = Path(path_text)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid classified exceptions JSON at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"invalid classified exceptions JSON at {path}: top level must be a mapping")
    return data


def classified_exceptions_catalog() -> dict[str, Any]:
    settings = input_source_settings("classified_exceptions", default_enabled=True)
    if not settings.get("enabled"):
        return {}
    path = classified_exceptions_path()
    if not path.is_file():
        raise FileNotFoundError(f"classified exceptions JSON not found at {path}")
    stat = path.stat()
    return _load_catalog(str(path), stat.st_mtime_ns, stat.st_size)


def _rebuttal_cycle(value: Any) -> int:
    match = _REBUTTAL_RE.search(str(value or ""))
    return int(match.group(1)) if match else 0


def _rebuttal_source_information(record: dict[str, Any], requested_cycle: int) -> dict[str, str]:
    rebuttals = [r for r in (record.get("rebuttals", []) or []) if isinstance(r, dict) and _rebuttal_cycle(r.get("rebuttal")) == requested_cycle]
    unique_source_info = {rebuttal.get("source_type") for rebuttal in rebuttals}
    source_info = {"source_file": "", "source_type": ""}

    for rebuttal in rebuttals:
        if len(unique_source_info) == 1:
            pass
        elif rebuttal.get("source_type", "").lower() == "operator response":
            continue
        
        source_info = {"source_file": rebuttal.get("source_file", ""), "source_type": rebuttal.get("source_type", "")}
        return source_info

    return source_info


def _requested_auditor_rebuttal(
    record: dict[str, Any], requested_cycle: int,
) -> list[dict[str, Any]]:
    """Parse source file path & auditor replies from the requested Rebuttal N sub-JSON.

    Later classified rebuttal entries carry the cumulative exchange. Operator replies are intentionally
    ignored here: prior Shell responses come from the pipeline's own markdown history instead.
    Duplicate Auditor Report / Operator Response copies are collapsed by date + text.
    """
    rebuttals = [r for r in (record.get("rebuttals", []) or []) if isinstance(r, dict) and _rebuttal_cycle(r.get("rebuttal")) == requested_cycle]
    unique_source_info = {rebuttal.get("source_type") for rebuttal in rebuttals}
    if not rebuttals:
        if requested_cycle == 0:
            return []
        raise KeyError(f"classified exception has no Rebuttal {requested_cycle}")
    replies: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for rebuttal in rebuttals:
        if len(unique_source_info) == 1:
            pass
        elif rebuttal.get("source_type", "").lower() == "operator response":
            continue

        current_cycle = 1
        for reply in rebuttal.get("replies", []) or []:
            if not isinstance(reply, dict) or str(reply.get("role") or "").strip().lower() != "auditor":
                continue
            date = str(reply.get("date") or "").strip()
            text = str(reply.get("text") or "").strip()
            key = (date.casefold(), " ".join(text.split()).casefold())
            if not text or key in seen:
                current_cycle += 1
                continue
            seen.add(key)
            replies.append({
                "rebuttal_cycle": current_cycle,
                "role": "Auditor",
                "date": date,
                "text": text,
            })
            current_cycle += 1
    return replies


def find_classified_exception(
    year: str, asset: str, exc_id: str, rebuttal_cycle: int = 0,
) -> dict[str, Any] | None:
    """Find one record with tolerant asset spelling and exception-id padding."""
    catalog = classified_exceptions_catalog()
    year_records = catalog.get(str(year), {})
    if not isinstance(year_records, dict):
        return None
    wanted_asset = _norm(asset)
    wanted_exc = pad_exc(exc_id)
    for asset_name, exceptions in year_records.items():
        if _norm(asset_name) != wanted_asset or not isinstance(exceptions, dict):
            continue
        for record_id, record in exceptions.items():
            if pad_exc(str(record_id)) != wanted_exc or not isinstance(record, dict):
                continue
            auditor_rebuttals = _requested_auditor_rebuttal(record, rebuttal_cycle)
            source_info = _rebuttal_source_information(record, requested_cycle=rebuttal_cycle)
            classification = record.get("classification") or {}
            return {
                "year": str(year),
                "asset": str(asset_name),
                "exception_id": str(record.get("exception_id") or record_id),
                "error_label": str(record.get("error_label") or ""),
                "error_type": str(record.get("error_type") or ""),
                "error_major": str(record.get("error_major") or ""),
                "matched_category": str(record.get("matched_category") or ""),
                "is_target_category": bool(record.get("is_target_category", False)),
                "short_description": str(record.get("short_description") or ""),
                "gross_credit_requested": str(record.get("gross_credit_requested") or ""),
                "description": str(record.get("description") or ""),
                "table_references": list(record.get("table_references") or []),
                "classification": dict(classification) if isinstance(classification, dict) else {},
                "rebuttal_cycle": rebuttal_cycle,
                "auditor_rebuttals": auditor_rebuttals,
                "latest_auditor_reply": auditor_rebuttals[-1] if auditor_rebuttals else {},
                "source_file": str(source_info.get("source_file") or ""),
                "source_type": str(source_info.get("source_type") or ""),
                "source": str(classified_exceptions_path()),
            }
    return None


def available_rebuttal_cycles(year: str, asset: str, exc_id: str) -> list[int]:
    """Return every classified Rebuttal N for one exception, sorted and deduplicated."""
    catalog = classified_exceptions_catalog()
    year_records = catalog.get(str(year), {})
    wanted_asset, wanted_exc = _norm(asset), pad_exc(exc_id)
    for asset_name, exceptions in (year_records.items() if isinstance(year_records, dict) else []):
        if _norm(asset_name) != wanted_asset or not isinstance(exceptions, dict):
            continue
        for record_id, record in exceptions.items():
            if pad_exc(str(record_id)) == wanted_exc and isinstance(record, dict):
                return sorted({
                    _rebuttal_cycle(item.get("rebuttal"))
                    for item in record.get("rebuttals", []) or [] if isinstance(item, dict)
                    if item.get("source_type").lower() != "operator response"
                })
    return []


def clear_classified_exceptions_cache() -> None:
    _load_catalog.cache_clear()


__all__ = [
    "classified_exceptions_catalog",
    "classified_exceptions_path",
    "clear_classified_exceptions_cache",
    "find_classified_exception",
    "available_rebuttal_cycles",
]
