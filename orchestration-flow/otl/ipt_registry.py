"""Static audit-year/asset IPT membership registry."""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


_ROOT = Path(__file__).resolve().parents[2]
_PATH = _ROOT / "configs" / "ipt_registry.yaml"
_ASSETS = ("Vito", "Sparta", "Whale")


def _tokens(value: str) -> tuple[str, ...]:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    return tuple(sorted(token for token in re.findall(r"[a-z]+", text.casefold())
                        if token not in {"jr", "sr", "ii", "iii", "iv"}))


def _employee(value: Any) -> str:
    return re.sub(r"\D", "", str(value or ""))


def _asset(value: str) -> str:
    normalized = re.sub(r"[^a-z]", "", str(value or "").casefold())
    for asset in _ASSETS:
        if re.sub(r"[^a-z]", "", asset.casefold()) in normalized:
            return asset
    return ""


@lru_cache(maxsize=1)
def registry() -> dict[str, Any]:
    raw = yaml.safe_load(_PATH.read_text(encoding="utf-8")) or {}
    if raw.get("schema_version") != 1 or not isinstance(raw.get("audit_years"), dict):
        raise ValueError(f"invalid IPT registry: {_PATH}")
    return raw


@lru_cache(maxsize=32)
def _entries(year: str, asset: str) -> tuple[dict[str, Any], ...]:
    canonical = _asset(asset)
    if not canonical:
        return ()
    values = (((registry().get("audit_years") or {}).get(str(year)) or {}).get(canonical) or {}).get(
        "people") or []
    return tuple(value for value in values if value.get("review_status") == "confirmed" and value.get("ipt"))


def lookup(*, audit_year: str, asset: str, person: str, employee_id: str = "") -> dict[str, Any] | None:
    entries = _entries(str(audit_year), asset)
    wanted_id = _employee(employee_id)
    if wanted_id:
        matched = [row for row in entries if _employee(row.get("employee_id")) == wanted_id]
        if len(matched) == 1:
            return dict(matched[0])
    wanted = _tokens(person)
    if not wanted:
        return None
    matched = []
    for row in entries:
        names = [row.get("name", ""), *(row.get("aliases") or [])]
        if any(_tokens(value) == wanted for value in names):
            matched.append(row)
    return dict(matched[0]) if len(matched) == 1 else None


def status(*, audit_year: str, asset: str, person: str, employee_id: str = "") -> tuple[str, dict | None]:
    match = lookup(audit_year=audit_year, asset=asset, person=person, employee_id=employee_id)
    return ("ipt", match) if match else ("non_ipt", None)


def clear_cache() -> None:
    registry.cache_clear()
    _entries.cache_clear()


__all__ = ["clear_cache", "lookup", "registry", "status"]
