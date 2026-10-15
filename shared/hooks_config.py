"""Validated, module-scoped runtime-hook configuration.

The default file is ``configs/hooks.yaml`` and may be overridden with
``HOOKS_CONFIG_FILE``.  Modules own independent hook namespaces, for example
``extractor.hooks.budget`` and ``labor.hooks.verify_evidence``. Traditional runtime hooks use an
``enabled/intercepts/parameters`` object; graph-node decision hooks contain ED rules directly.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "hooks.yaml"
_SUPPORTED_SCHEMA_VERSIONS = {1}


class HookDefinition(BaseModel):
    """One named hook and the call surfaces it intercepts."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    intercepts: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)


class EvidenceRuleDefinition(BaseModel):
    """One deterministic evidence rule configured directly under a graph-node hook."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    detector: Literal[
        "labor_ledger", "location_logs", "shell_invoice_prerequisite", "hierarchy_support",
        "jade_amount_reconciliation",
    ]
    evidence_labels: dict[str, str] = Field(default_factory=dict)
    reason: str = ""
    outcomes: dict[Literal["missing", "unreadable", "mismatch"], Literal[
        "deny_missing_evidence", "needs_evidence_unreadable", "deny_jade_amount_mismatch",
    ]]

    @model_validator(mode="after")
    def _validate_rule(self) -> "EvidenceRuleDefinition":
        if self.detector == "jade_amount_reconciliation":
            if set(self.outcomes) != {"mismatch"}:
                raise ValueError("jade reconciliation hook `outcomes` must define exactly mismatch")
            if not self.reason.strip():
                raise ValueError("jade reconciliation hook `reason` must not be empty")
            return self
        if not self.evidence_labels:
            raise ValueError("evidence rule `evidence_labels` must not be empty")
        for key, value in self.evidence_labels.items():
            if not str(key or "").strip() or not str(value or "").strip():
                raise ValueError("evidence rule labels must use non-empty keys and values")
        if set(self.outcomes) != {"missing", "unreadable"}:
            raise ValueError("evidence rule `outcomes` must define exactly missing and unreadable")
        return self


class ChargeabilityRuleDefinition(BaseModel):
    """One supplied GR/SE rule. Nodes provide facts; the rule supplies outcomes only."""

    model_config = ConfigDict(extra="forbid")

    all: dict[str, list[Any]] = Field(default_factory=dict)
    any: list[dict[str, list[Any]]] = Field(default_factory=list)
    decision: str
    disposition: Literal["deny", "accept", "partial", "needs_evidence"]
    outcome: str

    @model_validator(mode="after")
    def _validate_conditions(self) -> "ChargeabilityRuleDefinition":
        if not self.all and not self.any:
            raise ValueError("chargeability rule must define at least one condition")
        if not self.decision.strip() or not self.outcome.strip():
            raise ValueError("chargeability decision/outcome must not be empty")
        return self


class ChargeabilityHookDefinition(BaseModel):
    """Declarative GR/SE rules configured under ``labor.hooks.chargeability``."""

    model_config = ConfigDict(extra="forbid")
    rules: dict[str, ChargeabilityRuleDefinition]

    @model_validator(mode="after")
    def _validate_rule_ids(self) -> "ChargeabilityHookDefinition":
        invalid = [key for key in self.rules if not re.fullmatch(r"(?:GR|SE)-\d{2}", str(key))]
        if invalid:
            raise ValueError("chargeability rule IDs must match GR-## or SE-##: " + ", ".join(invalid))
        return self


class NodeHookDefinitions(RootModel[dict[str, EvidenceRuleDefinition]]):
    """Evidence rules keyed directly by stable IDs such as ``ED-01``."""

    @model_validator(mode="after")
    def _validate_gate_ids(self) -> "NodeHookDefinitions":
        invalid = [key for key in self.root if not re.fullmatch(r"ED-\d{2}", str(key))]
        if invalid:
            raise ValueError(
                "node hook rule IDs must match ED-##; invalid: " + ", ".join(sorted(invalid))
            )
        return self


class ModuleHooks(BaseModel):
    """Hook definitions belonging to one application module."""

    model_config = ConfigDict(extra="forbid")

    hooks: dict[str, HookDefinition | NodeHookDefinitions | ChargeabilityHookDefinition] = Field(
        default_factory=dict)


def hooks_config_path() -> Path:
    """Return the active hooks YAML path."""
    return Path(os.getenv("HOOKS_CONFIG_FILE") or _DEFAULT_CONFIG).expanduser().resolve()


@lru_cache(maxsize=8)
def _load_hooks_config(path_text: str, _modified_ns: int, _size: int) -> dict[str, ModuleHooks]:
    path = Path(path_text)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid hooks YAML at {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"invalid hooks config at {path}: top level must be a mapping")
    version = raw.get("schema_version")
    if version not in _SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(str(v) for v in sorted(_SUPPORTED_SCHEMA_VERSIONS))
        raise ValueError(
            f"invalid hooks config at {path}: schema_version must be one of {supported}"
        )

    modules: dict[str, ModuleHooks] = {}
    for name, value in raw.items():
        if name == "schema_version":
            continue
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"invalid hooks config at {path}: module names must be non-empty strings")
        modules[name] = ModuleHooks.model_validate(value)
    return modules


def hooks_config() -> dict[str, ModuleHooks]:
    """Load and validate the active hook configuration."""
    path = hooks_config_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"hooks config not found at {path}. Set HOOKS_CONFIG_FILE to override it."
        )
    stat = path.stat()
    return _load_hooks_config(str(path), stat.st_mtime_ns, stat.st_size)


def hook_definition(module: str, hook: str) -> HookDefinition | None:
    """Return one configured hook, or ``None`` when the module/hook is absent."""
    module_config = hooks_config().get(module)
    definition = module_config.hooks.get(hook) if module_config else None
    return definition if isinstance(definition, HookDefinition) else None


def node_hook_definitions(module: str, node: str) -> dict[str, EvidenceRuleDefinition]:
    """Return the evidence rules configured directly under ``module.hooks.<node>``."""
    module_config = hooks_config().get(module)
    definition = module_config.hooks.get(node) if module_config else None
    return dict(definition.root) if isinstance(definition, NodeHookDefinitions) else {}


def chargeability_hook_definitions(module: str = "labor", hook: str = "chargeability") -> dict[
    str, ChargeabilityRuleDefinition
]:
    module_config = hooks_config().get(module)
    definition = module_config.hooks.get(hook) if module_config else None
    return dict(definition.rules) if isinstance(definition, ChargeabilityHookDefinition) else {}


def clear_hooks_config_cache() -> None:
    """Clear cached YAML content after an in-process config change (primarily tests)."""
    _load_hooks_config.cache_clear()


__all__ = [
    "HookDefinition",
    "EvidenceRuleDefinition",
    "ChargeabilityHookDefinition",
    "ChargeabilityRuleDefinition",
    "ModuleHooks",
    "NodeHookDefinitions",
    "clear_hooks_config_cache",
    "chargeability_hook_definitions",
    "hook_definition",
    "hooks_config",
    "hooks_config_path",
    "node_hook_definitions",
]
