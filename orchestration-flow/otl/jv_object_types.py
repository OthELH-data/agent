"""Major-construction JV object-type code registry.

JADE has no "JV Object Type Description" text column. The only field available per charge row is a
coded "WBS:JV Object Type" (equivalently "JV Object Type") value — a short code such as "C142" or
"O143". This module loads the reference table (`configs/major_construction_jv_object_types.yaml`) that
code is looked up against: membership in this table, not a text pattern, is what makes a charge row
major construction overhead (MCOH).

`jade_analyzer`'s ReAct loop is handed `major_construction_jv_object_types()` as an INPUT so its own
sandboxed code does the lookup — an exact dict membership check — rather than asking the model to
memorize or reason about which codes qualify.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_REGISTRY = _REPO_ROOT / "configs" / "major_construction_jv_object_types.yaml"


def registry_path() -> Path:
    """Resolved registry path; the environment override supports controlled registry updates."""
    return Path(
        os.getenv("OTL_MAJOR_CONSTRUCTION_REGISTRY") or _DEFAULT_REGISTRY
    ).expanduser().resolve()


def _normalize_code(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().upper())


def normalized_code_candidates(value: str) -> list[str]:
    """Possible registry keys for a JADE object type, preserving exact-match semantics.

    Some exports prefix the registry code with a site such as ``US01/C116``.  The registry contains
    the actual object-type stem (``C116``), so the full normalized cell and its final slash-delimited
    component are the only accepted candidates; no description/text inference is performed.
    """
    full = _normalize_code(value)
    candidates = [full] if full else []
    tail = full.rsplit("/", 1)[-1] if "/" in full else ""
    if tail and tail not in candidates:
        candidates.append(tail)
    return candidates


@lru_cache(maxsize=1)
def major_construction_jv_object_types() -> dict[str, str]:
    """code -> description, keyed by the normalized code (upper-cased, whitespace stripped)."""
    path = registry_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"major-construction JV object-type registry not found at {path}. Set "
            "OTL_MAJOR_CONSTRUCTION_REGISTRY to its path."
        )
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    codes = data.get("codes")
    if not isinstance(codes, list):
        raise ValueError(f"invalid major-construction JV object-type registry: {path}")
    return {
        _normalize_code(row["code"]): str(row.get("description") or "")
        for row in codes
        if isinstance(row, dict) and row.get("code")
    }


def is_major_construction_code(value: str) -> bool:
    """True when `value` (whatever whitespace/case JADE wrote it in) is a registered MCOH code."""
    registry = major_construction_jv_object_types()
    return any(code in registry for code in normalized_code_candidates(value))


__all__ = [
    "is_major_construction_code", "major_construction_jv_object_types", "normalized_code_candidates",
    "registry_path",
]
