"""Tracked manifest for the confidential agreement corpus.

The PDFs live outside Git under ``Data Files``.  This module keeps only stable, reviewable metadata in
the repository: asset aliases, source paths, party scopes, and the legal track each instrument belongs
to.  Runtime retrieval never guesses an asset from a filename and never narrows Great White merely
because the exception folder says "BP only" or "Chevron only".

Effective dates.  Each document may carry ``active_start_date`` / ``active_end_date`` written in its
``date_format`` (default ``MM/DD/YYYY``; ``DD/MM/YYYY`` and ``YYYY-MM-DD`` style tokens are also
accepted).  A blank value or ``N/A`` means the period is open on that side.  An unparseable date, or an
end before the start, fails manifest loading with the document_id and field named.  For an audit year
the retriever keeps only documents whose active period overlaps that calendar year (see
``documents_for_case``); with no usable year nothing is filtered.

Index fingerprint.  ``manifest.fingerprint`` is a SHA-256 over a canonical JSON of only the fields that
are copied into stored chunks or that select the source PDF: per document (sorted by document_id)
``document_id``, ``relative_path``, ``document_name``, ``document_type``, ``legal_track``, sorted
``asset_scopes`` and ``party_scope`` (order kept).  Editing any of those marks the index stale and needs
a rebuild.  Dates, ``date_format``, asset aliases, comments and YAML formatting are not fingerprinted:
date edits take effect at runtime (on the next manifest load) without a rebuild.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from otl._bootstrap import DATA_ROOT, REPO_ROOT


_DEFAULT_MANIFEST = REPO_ROOT / "configs" / "agreement_corpus.yaml"


class AssetSpec(BaseModel):
    aliases: list[str] = Field(default_factory=list)


DEFAULT_DATE_FORMAT = "MM/DD/YYYY"
_DATE_TOKENS = {"YYYY": "%Y", "MM": "%m", "DD": "%d"}
_DATE_FORMAT_PATTERN = re.compile(r"YYYY|MM|DD|[/\-. ]")
_OPEN_DATE_VALUES = {"", "n/a"}
_DATE_FIELDS = ("active_start_date", "active_end_date")
# Manifest fields that are copied into stored chunks (record.ChunkRecord via chunk.py and
# cache.sync_document_metadata) or that select the source PDF. Only these drive the index fingerprint.
INDEX_FIELDS = (
    "document_id", "relative_path", "document_name", "document_type", "legal_track", "asset_scopes",
    "party_scope",
)


def strptime_format(date_format: str) -> str:
    """Translate a manifest ``date_format`` such as ``MM/DD/YYYY`` into a ``strptime`` pattern."""
    text = str(date_format or "").strip()
    tokens = _DATE_FORMAT_PATTERN.findall(text)
    if not text or "".join(tokens) != text or sorted(t for t in tokens if t in _DATE_TOKENS) != [
        "DD", "MM", "YYYY",
    ]:
        raise ValueError(f"unsupported date_format {date_format!r}; use tokens YYYY, MM, DD")
    return "".join(_DATE_TOKENS.get(token, token) for token in tokens)


def parse_manifest_date(value: Any, date_format: str = DEFAULT_DATE_FORMAT) -> date | None:
    """Parse one manifest date; blank, None or ``N/A`` (any case) mean open-ended and return None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value  # YAML already turned an ISO literal such as 2022-02-01 into a date
    text = str(value).strip()
    if text.casefold() in _OPEN_DATE_VALUES:
        return None
    return datetime.strptime(text, strptime_format(date_format)).date()


class AgreementDocumentSpec(BaseModel):
    document_id: str
    relative_path: str
    document_name: str
    document_type: str
    legal_track: str
    asset_scopes: list[str]
    party_scope: list[str] = Field(default_factory=list)
    date_format: str = DEFAULT_DATE_FORMAT
    active_start_date: date | None = None
    active_end_date: date | None = None

    @model_validator(mode="before")
    @classmethod
    def _parse_dates(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        values = dict(data)
        document_id = values.get("document_id", "<unknown>")
        date_format = values.get("date_format")
        if date_format is None or not str(date_format).strip():
            date_format = DEFAULT_DATE_FORMAT
        values["date_format"] = date_format
        for name in _DATE_FIELDS:
            try:
                values[name] = parse_manifest_date(values.get(name), date_format)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"agreement {document_id}: invalid {name} {values.get(name)!r} for "
                    f"date_format {date_format!r}: {exc}"
                ) from None
        return values

    @model_validator(mode="after")
    def _check_period(self) -> "AgreementDocumentSpec":
        start, end = self.active_start_date, self.active_end_date
        if start and end and end < start:
            raise ValueError(
                f"agreement {self.document_id}: active_end_date {end.isoformat()} is before "
                f"active_start_date {start.isoformat()}"
            )
        return self

    def index_definition(self) -> dict[str, Any]:
        """The index-relevant fields (``INDEX_FIELDS``), as stored in each document cache."""
        return {
            "document_id": self.document_id,
            "relative_path": self.relative_path,
            "document_name": self.document_name,
            "document_type": self.document_type,
            "legal_track": self.legal_track,
            "asset_scopes": list(self.asset_scopes),
            "party_scope": list(self.party_scope),
        }

    def active_in_year(self, year: int) -> bool:
        """True when the active period overlaps calendar ``year``; open sides always overlap."""
        start, end = self.active_start_date, self.active_end_date
        return (start is None or start <= date(year, 12, 31)) and (end is None or end >= date(year, 1, 1))


class AgreementCorpusManifest(BaseModel):
    version: int = 1
    assets: dict[str, AssetSpec]
    documents: list[AgreementDocumentSpec]
    fingerprint: str = ""

    @model_validator(mode="after")
    def _validate_references(self) -> "AgreementCorpusManifest":
        known_assets = set(self.assets)
        document_ids: set[str] = set()
        for doc in self.documents:
            if doc.document_id in document_ids:
                raise ValueError(f"duplicate agreement document_id: {doc.document_id}")
            document_ids.add(doc.document_id)
            unknown = set(doc.asset_scopes) - known_assets
            if unknown:
                raise ValueError(
                    f"agreement {doc.document_id} references unknown assets: {sorted(unknown)}"
                )
        return self


def compute_fingerprint(documents: list[AgreementDocumentSpec]) -> str:
    """SHA-256 of the canonical JSON of each document's ``INDEX_FIELDS`` (sorted by document_id).

    ``asset_scopes`` is sorted (a set of scopes); ``party_scope`` keeps its order because it is stored
    in chunk rows as written.
    """
    payload = []
    for document in sorted(documents, key=lambda doc: doc.document_id):
        definition = document.index_definition()
        definition["asset_scopes"] = sorted(definition["asset_scopes"])
        payload.append(definition)
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def manifest_path() -> Path:
    return Path(os.getenv("OTL_AGREEMENT_MANIFEST") or _DEFAULT_MANIFEST).expanduser().resolve()


def data_root() -> Path:
    return Path(REPO_ROOT / (os.getenv("OTL_AGREEMENT_DATA_ROOT") or DATA_ROOT)).expanduser().resolve()


def index_dir() -> Path:
    return Path(
        os.getenv("OTL_AGREEMENT_INDEX_DIR") or (data_root() / ".agreement_rag")
    ).expanduser().resolve()


def normalize_asset(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())


@lru_cache(maxsize=4)
def _load_manifest(path_text: str) -> AgreementCorpusManifest:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"agreement corpus manifest not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    manifest = AgreementCorpusManifest(**data)
    manifest.fingerprint = compute_fingerprint(manifest.documents)
    return manifest


def load_manifest(path: str | Path | None = None) -> AgreementCorpusManifest:
    return _load_manifest(str(Path(path).expanduser().resolve() if path else manifest_path()))


def _alias_map(manifest: AgreementCorpusManifest) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for canonical, spec in manifest.assets.items():
        for value in [canonical, *spec.aliases]:
            key = normalize_asset(value)
            if key:
                aliases[key] = canonical
    return aliases


def canonical_asset(value: str, manifest: AgreementCorpusManifest | None = None) -> str:
    manifest = manifest or load_manifest()
    target = normalize_asset(value)
    if not target:
        return ""
    aliases = _alias_map(manifest)
    if target in aliases:
        return aliases[target]
    # Folder-derived case IDs occasionally append year/exception tokens.  Permit a unique contained
    # alias, preferring the longest match so "greatwhite" wins over the shorter "gw" alias.
    matches = [
        (len(alias), canonical)
        for alias, canonical in aliases.items()
        if len(alias) >= 5 and (alias in target or target in alias)
    ]
    if not matches:
        return ""
    matches.sort(reverse=True)
    best_length = matches[0][0]
    best = {canonical for length, canonical in matches if length == best_length}
    return next(iter(best)) if len(best) == 1 else ""


def parse_audit_year(value: Any) -> int | None:
    """An audit year as an int, or None when ``value`` is empty or not a plain 4-digit year."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 1 <= value <= 9999 else None
    text = str(value).strip()
    if re.fullmatch(r"\d{4}(\.0+)?", text):
        return int(text.split(".", 1)[0])
    return None


def documents_for_case(
    asset: str, year: Any = None, manifest: AgreementCorpusManifest | None = None,
) -> tuple[list[AgreementDocumentSpec], list[AgreementDocumentSpec]]:
    """``(included, excluded)`` documents for an asset and audit year.

    A document is included when its asset scope covers the asset and, if ``year`` parses to an int,
    its active period overlaps that calendar year:
    ``(start is None or start <= Dec 31 of year) and (end is None or end >= Jan 1 of year)``.
    ``excluded`` lists the asset's documents removed by that date rule (empty when no year applies).
    """
    manifest = manifest or load_manifest()
    canonical = canonical_asset(asset, manifest)
    if not canonical:
        return [], []
    scoped = [doc for doc in manifest.documents if canonical in doc.asset_scopes]
    audit_year = parse_audit_year(year)
    if audit_year is None:
        return scoped, []
    included = [doc for doc in scoped if doc.active_in_year(audit_year)]
    excluded = [doc for doc in scoped if not doc.active_in_year(audit_year)]
    return included, excluded


def documents_for_asset(
    asset: str, manifest: AgreementCorpusManifest | None = None, year: Any = None,
) -> list[AgreementDocumentSpec]:
    """The asset's documents, date-filtered by ``year`` when it parses (see ``documents_for_case``)."""
    return documents_for_case(asset, year, manifest)[0]


def date_filter_provenance(year: Any, excluded: list[AgreementDocumentSpec]) -> dict[str, Any]:
    """The ``date_filter`` provenance block recorded by the retriever and the offline tools."""
    audit_year = parse_audit_year(year)
    return {
        "year": audit_year,
        "applied": audit_year is not None,
        "excluded_documents": [
            {
                "document_id": doc.document_id,
                "active_start_date": doc.active_start_date.isoformat() if doc.active_start_date else None,
                "active_end_date": doc.active_end_date.isoformat() if doc.active_end_date else None,
            }
            for doc in excluded
        ],
    }


def source_path(document: AgreementDocumentSpec, root: str | Path | None = None) -> Path:
    base = Path(root).expanduser().resolve() if root else data_root()
    return (base / document.relative_path).resolve()


__all__ = [
    "AgreementCorpusManifest", "AgreementDocumentSpec", "AssetSpec", "DEFAULT_DATE_FORMAT",
    "INDEX_FIELDS", "canonical_asset", "compute_fingerprint", "data_root", "date_filter_provenance",
    "documents_for_asset", "documents_for_case", "index_dir", "load_manifest", "manifest_path",
    "normalize_asset", "parse_audit_year", "parse_manifest_date", "source_path", "strptime_format",
]
