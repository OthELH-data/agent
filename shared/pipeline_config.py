"""Versioned YAML configuration for OTL input sources and optional node gates.

The configuration is intentionally node-agnostic: graph code asks whether a named gate is enabled
and owns the routing semantics for that gate. This lets future gates reuse the same validated YAML
shape without coupling this module to the current graph topology.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from shared.extractor_settings import resolve_extractor_excel_parsing_enabled


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "pipeline.yaml"
_SUPPORTED_SCHEMA_VERSIONS = {1}


def _validate_switch_map(value: Any, *, section: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"invalid OTL pipeline config at {path}: {section} must be a mapping")
    for name, entry in value.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"invalid OTL pipeline config at {path}: {section} names must be non-empty strings"
            )
        if isinstance(entry, bool):
            continue
        if not isinstance(entry, dict):
            raise ValueError(
                f"invalid OTL pipeline config at {path}: {section}.{name} must be a boolean or mapping"
            )
        enabled = entry.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError(
                f"invalid OTL pipeline config at {path}: {section}.{name}.enabled must be true or false"
            )
    return value


def pipeline_config_path() -> Path:
    """Return the active config path, with an override for deployments and tests."""
    return Path(
        os.getenv("OTL_PIPELINE_CONFIG") or _DEFAULT_CONFIG
    ).expanduser().resolve()


def _validate_config(data: Any, path: Path) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"invalid OTL pipeline config at {path}: top level must be a mapping")

    version = data.get("schema_version")
    if version not in _SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(str(v) for v in sorted(_SUPPORTED_SCHEMA_VERSIONS))
        raise ValueError(
            f"invalid OTL pipeline config at {path}: schema_version must be one of {supported}"
        )

    sources = _validate_switch_map(data.get("input_sources", {}), section="input_sources", path=path)
    gates = _validate_switch_map(data.get("node_gates", {}), section="node_gates", path=path)

    classified = sources.get("classified_exceptions", {})
    if isinstance(classified, dict) and classified.get("enabled", False):
        source_path = classified.get("path")
        if not isinstance(source_path, str) or not source_path.strip():
            raise ValueError(
                f"invalid OTL pipeline config at {path}: "
                "input_sources.classified_exceptions.path must be a non-empty string"
            )

    extractor = sources.get("extractor_outputs", {})
    if isinstance(extractor, dict):
        if "excel_parsing" in extractor:
            resolve_extractor_excel_parsing_enabled(extractor, source=path)
        if extractor.get("enabled", False):
            for field, default in (("directory_env", "INPUT_BASE"),
                                   ("read_folder_env", "SHAREPOINT_READ_FOLDER")):
                env_name = extractor.get(field, default)
                if not isinstance(env_name, str) or not env_name.strip():
                    raise ValueError(
                        f"invalid OTL pipeline config at {path}: "
                        f"input_sources.extractor_outputs.{field} must be a non-empty string"
                    )

    return data


@lru_cache(maxsize=8)
def _load_pipeline_config(path_text: str, _modified_ns: int, _size: int) -> dict[str, Any]:
    """Load one file revision; the stat fields make edits visible without restarting the process."""
    path = Path(path_text)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in OTL pipeline config at {path}: {exc}") from exc
    return _validate_config(data, path)


def pipeline_config() -> dict[str, Any]:
    """Load and validate the active pipeline configuration."""
    path = pipeline_config_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"OTL pipeline config not found at {path}. Set OTL_PIPELINE_CONFIG to its path."
        )
    stat = path.stat()
    return _load_pipeline_config(str(path), stat.st_mtime_ns, stat.st_size)


def node_gate_enabled(node_name: str, *, default: bool = False) -> bool:
    """Return whether the named gate should execute.

    Entries support either concise boolean syntax::

        node_gates:
          verify_evidence: false

    or an expanded mapping that can carry future gate metadata::

        node_gates:
          verify_evidence:
            enabled: false
            description: Optional operator-facing note

    Gates are opt-in by default: an absent entry is disabled unless the caller explicitly supplies a
    different ``default``. Unknown mapping fields are retained for forward-compatible consumers.
    """
    gates = pipeline_config().get("node_gates", {})
    entry = gates.get(node_name)
    if entry is None:
        return bool(default)
    if isinstance(entry, bool):
        return entry
    return entry.get("enabled", bool(default))


def input_source_settings(source_name: str, *, default_enabled: bool = False) -> dict[str, Any]:
    """Return one normalized input-source entry while preserving future source-specific fields."""
    entry = pipeline_config().get("input_sources", {}).get(source_name)
    if entry is None:
        return {"enabled": bool(default_enabled)}
    if isinstance(entry, bool):
        return {"enabled": entry}
    return {**entry, "enabled": entry.get("enabled", bool(default_enabled))}


def configured_path(value: str) -> Path:
    """Resolve a config path; relative values are rooted at ``orchestration-flow``."""
    path = Path(str(value or "")).expanduser()
    return (path if path.is_absolute() else _REPO_ROOT / path).resolve()


def clear_pipeline_config_cache() -> None:
    """Clear cached YAML content after an in-process config change (primarily tests/tools)."""
    _load_pipeline_config.cache_clear()


__all__ = [
    "clear_pipeline_config_cache",
    "configured_path",
    "input_source_settings",
    "node_gate_enabled",
    "pipeline_config",
    "pipeline_config_path",
]
