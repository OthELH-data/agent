"""Codified authority registry for Chevron GOM OTL decisions.

The LLM does not invent agreement names, section numbers, or legal propositions. This module loads the
pinpointed settlement registry, exposes the exact provisions used by matching/publication, and keeps a
path to the underlying agreement for the audit trail. Role-specific CFR ranges remain evidence: the
matching agent retrieves them from the CFR-mapping workbook and the drafter explains their application.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REGISTRY = _REPO_ROOT / "configs" / "otl_settlement_sections.yaml"
_DEFAULT_AGREEMENT = (
    _REPO_ROOT / "Data Files" / "OTL Information" / "Settlement Documents"
    / "C00025092-001 _ 2019 CVX GOM SETTLEMENT AGREEMENT.pdf"
)

_CATEGORY_RULE_IDS = {
    "Regulatory Activity": "otl-cvx-regulatory-chargeability-attachment-1-3a",
    "Feasibility and Project Studies": "otl-cvx-project-studies-attachment-1-3b",
    "Planners & Schedulers, maintenance and turnaround, rig administration": (
        "otl-cvx-planners-schedulers-attachment-1-3c"
    ),
}
_REGULATORY_DOCUMENTATION_RULE = "otl-cvx-regulatory-documentation-attachment-1-3a"
_TECHNICAL_RULE = "otl-cvx-technical-employees-attachment-1-2"
_TECHNICAL_EXCLUSIONS_RULE = "otl-cvx-technical-employees-exclusions-attachment-1-2"
_OFFSITE_RULE = "otl-cvx-definition-attachment-1-1"
_AFFILIATE_RULE = "otl-cvx-affiliate-attachment-1-5"
_MAJOR_CONSTRUCTION_RULE = "otl-coua-major-construction-exhibit-c-3-2b"


def registry_path() -> Path:
    """Resolved registry path; the environment override supports controlled registry updates."""
    return Path(os.getenv("OTL_SETTLEMENT_REGISTRY") or _DEFAULT_REGISTRY).expanduser().resolve()


def agreement_path() -> Path:
    """Resolved source-agreement path retained for reviewers and future source verification."""
    return Path(os.getenv("OTL_SETTLEMENT_AGREEMENT") or _DEFAULT_AGREEMENT).expanduser().resolve()


@lru_cache(maxsize=1)
def settlement_registry() -> dict[str, Any]:
    path = registry_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"OTL settlement registry not found at {path}. Set OTL_SETTLEMENT_REGISTRY to its path."
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not data.get("instrument") or not isinstance(data.get("sections"), list):
        raise ValueError(f"invalid OTL settlement registry: {path}")
    return data


@lru_cache(maxsize=1)
def _rules_by_id() -> dict[str, dict[str, Any]]:
    return {
        str(rule.get("rule_id")): rule
        for rule in settlement_registry().get("sections", [])
        if isinstance(rule, dict) and rule.get("rule_id")
    }


def provision(rule_id: str) -> dict[str, Any]:
    rule = _rules_by_id().get(rule_id)
    if not rule:
        raise KeyError(f"settlement provision {rule_id!r} is absent from {registry_path()}")
    return dict(rule)


def category_provision(category: str) -> dict[str, Any]:
    rule_id = _CATEGORY_RULE_IDS.get(category)
    return provision(rule_id) if rule_id else {}


def _normalize_asset(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def agreement_applies_to_asset(asset: str) -> bool:
    """True only when the case asset is within the registry's stated agreement scope."""
    target = _normalize_asset(asset)
    if not target:
        return False
    aliases = {_normalize_asset(value) for value in settlement_registry().get("asset_scope", [])}
    return target in aliases or any(
        len(alias) >= 5 and (alias in target or target in alias) for alias in aliases
    )


OTL_GOVERNING_INSTRUMENT = str(settlement_registry()["instrument"])
OTL_GOVERNING_SOURCE_DOCUMENT = str(settlement_registry().get("source_doc") or "")
OTL_TECHNICAL_EMPLOYEE_PROVISION = provision(_TECHNICAL_RULE)
OTL_TECHNICAL_EXCLUSIONS_PROVISION = provision(_TECHNICAL_EXCLUSIONS_RULE)
OTL_OFFSITE_PROVISION = provision(_OFFSITE_RULE)
OTL_AFFILIATE_PROVISION = provision(_AFFILIATE_RULE)
# Own source_doc/instrument (a COUA, not the GOM Settlement Agreement) — the drafter must cite THIS
# provision's instrument/section, not the case's governing_instrument, when it quotes this sentence.
OTL_MAJOR_CONSTRUCTION_PROVISION = provision(_MAJOR_CONSTRUCTION_RULE)
OTL_TECHNICAL_EMPLOYEE_SECTION = str(OTL_TECHNICAL_EMPLOYEE_PROVISION["section"])
OTL_CATEGORY_PROVISIONS = {
    category: category_provision(category) for category in _CATEGORY_RULE_IDS
}
OTL_CATEGORY_SECTIONS = {
    category: str(rule.get("section") or "") for category, rule in OTL_CATEGORY_PROVISIONS.items()
}
OTL_CATEGORY_APPLICATIONS = {
    category: str(rule.get("statement") or rule.get("application_summary") or "")
    for category, rule in OTL_CATEGORY_PROVISIONS.items()
}
OTL_CATEGORY_CLAUSES = {
    category: str(rule.get("quote") or "") for category, rule in OTL_CATEGORY_PROVISIONS.items()
}
# §3.A contains a separate documentation paragraph in the source; retain both exact excerpts.
OTL_CATEGORY_CLAUSES["Regulatory Activity"] = "\n\n".join(filter(None, (
    OTL_CATEGORY_CLAUSES["Regulatory Activity"],
    str(provision(_REGULATORY_DOCUMENTATION_RULE).get("quote") or ""),
)))


__all__ = [
    "OTL_AFFILIATE_PROVISION", "OTL_CATEGORY_APPLICATIONS", "OTL_CATEGORY_CLAUSES",
    "OTL_CATEGORY_PROVISIONS", "OTL_CATEGORY_SECTIONS", "OTL_GOVERNING_INSTRUMENT",
    "OTL_GOVERNING_SOURCE_DOCUMENT", "OTL_MAJOR_CONSTRUCTION_PROVISION", "OTL_OFFSITE_PROVISION",
    "OTL_TECHNICAL_EMPLOYEE_PROVISION", "OTL_TECHNICAL_EMPLOYEE_SECTION",
    "OTL_TECHNICAL_EXCLUSIONS_PROVISION", "agreement_applies_to_asset", "agreement_path",
    "category_provision", "provision", "registry_path", "settlement_registry",
]
