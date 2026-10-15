"""Shared configuration lookup for extractor behavior used by both pipelines.

The Excel extractor is runnable on its own, while OTL consumes the JSON artifacts it produces. Both
must therefore read the same switch without either package importing the other one's runtime stack.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "pipeline.yaml"
_SUPPORTED_SCHEMA_VERSIONS = {1}


def extractor_config_path() -> Path:
    """Return the shared pipeline config path, honoring the deployment/test override."""
    return Path(os.getenv("OTL_PIPELINE_CONFIG") or _DEFAULT_CONFIG).expanduser().resolve()


def resolve_extractor_excel_parsing_enabled(
    settings: dict[str, Any], *, source: str | Path = "OTL pipeline config"
) -> bool:
    """Resolve the nested Excel switch from normalized ``extractor_outputs`` settings.

    Missing values default to enabled so configurations written before the switch was introduced keep
    their existing workbook behavior. Both boolean and expanded mapping forms are accepted.
    """
    entry = settings.get("excel_parsing")
    if entry is None:
        return True
    if isinstance(entry, bool):
        return entry
    section = "input_sources.extractor_outputs.excel_parsing"
    if not isinstance(entry, dict):
        raise ValueError(
            f"invalid OTL pipeline config at {source}: {section} must be a boolean or mapping"
        )
    enabled = entry.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(
            f"invalid OTL pipeline config at {source}: {section}.enabled must be true or false"
        )
    return enabled


@lru_cache(maxsize=8)
def _load_excel_setting(path_text: str, _modified_ns: int, _size: int) -> bool:
    path = Path(path_text)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in OTL pipeline config at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"invalid OTL pipeline config at {path}: top level must be a mapping")
    version = data.get("schema_version")
    if version not in _SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(str(v) for v in sorted(_SUPPORTED_SCHEMA_VERSIONS))
        raise ValueError(
            f"invalid OTL pipeline config at {path}: schema_version must be one of {supported}"
        )
    sources = data.get("input_sources", {})
    if not isinstance(sources, dict):
        raise ValueError(f"invalid OTL pipeline config at {path}: input_sources must be a mapping")
    extractor = sources.get("extractor_outputs", {})
    if isinstance(extractor, bool):
        extractor = {"enabled": extractor}
    if not isinstance(extractor, dict):
        raise ValueError(
            f"invalid OTL pipeline config at {path}: "
            "input_sources.extractor_outputs must be a boolean or mapping"
        )
    return resolve_extractor_excel_parsing_enabled(extractor, source=path)


def extractor_excel_parsing_enabled() -> bool:
    """Read whether standalone extractor runs should include Excel workbook parsing."""
    path = extractor_config_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"OTL pipeline config not found at {path}. Set OTL_PIPELINE_CONFIG to its path."
        )
    stat = path.stat()
    return _load_excel_setting(str(path), stat.st_mtime_ns, stat.st_size)


def clear_extractor_settings_cache() -> None:
    """Clear cached YAML content after an in-process config change (primarily tests/tools)."""
    _load_excel_setting.cache_clear()


__all__ = [
    "clear_extractor_settings_cache",
    "extractor_config_path",
    "extractor_excel_parsing_enabled",
    "resolve_extractor_excel_parsing_enabled",
]
