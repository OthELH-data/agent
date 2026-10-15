"""Validated JADE-domain configuration used by deterministic row analysis."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


_ROOT = Path(__file__).resolve().parents[2]
_PATH = _ROOT / "configs" / "otl_jade_rules.yaml"


@lru_cache(maxsize=1)
def jade_rules() -> dict[str, Any]:
    raw = yaml.safe_load(_PATH.read_text(encoding="utf-8")) or {}
    if raw.get("schema_version") != 1:
        raise ValueError(f"unsupported JADE-rules schema in {_PATH}")
    return raw


def cats_accounts() -> dict[str, str]:
    return {str(k).strip().upper(): str(v) for k, v in (jade_rules().get("cats_accounts") or {}).items()}


def mcoh_threshold() -> float:
    return float(jade_rules().get("mcoh_monthly_threshold_hours") or 8.0)


def asset_cost_centers(asset: str) -> dict[str, str]:
    wanted = " ".join(str(asset or "").casefold().split())
    for name, values in (jade_rules().get("asset_cost_centers") or {}).items():
        norm = " ".join(str(name).casefold().split())
        if norm == wanted or norm in wanted or wanted in norm:
            return {str(k).strip(): str(v) for k, v in (values or {}).items()}
    return {}


def exempt_terms() -> list[str]:
    return [str(v).casefold() for v in jade_rules().get("exempt_roles") or []]


def ipt_terms() -> list[str]:
    return [str(v).casefold() for v in jade_rules().get("ipt_terms") or []]


__all__ = ["asset_cost_centers", "cats_accounts", "exempt_terms", "ipt_terms", "jade_rules",
           "mcoh_threshold"]
