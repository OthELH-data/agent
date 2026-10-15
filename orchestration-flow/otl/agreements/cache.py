"""The per-document JSON cache under ``<index_dir>/documents/<document_id>.json``.

Each cache holds one document's extracted pages, its chunks and their vectors. It is keyed by the source
PDF's SHA-256 and ``EXTRACTION_VERSION``, so an unchanged 100+ page PDF is never OCR'd again, and a
chunk is re-embedded only when its ``content_sha256``, the model or the dimensions change. The SQLite
index is rebuilt from these caches on every build.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from .corpus import AgreementDocumentSpec, source_path
from .embedding import Embedder, encode_vector
from .store import EXTRACTION_VERSION

from .chunk import chunk_document
from .extract import OCR_DPI, extract_pages

BATCH_SIZE = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def read_cache(path: Path) -> dict[str, Any]:
    """Return the cache at ``path``, or ``{}`` when it is missing or unreadable."""
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def sync_document_metadata(cache: dict[str, Any], document: AgreementDocumentSpec) -> bool:
    """Update manifest-owned fields without invalidating the expensive OCR/chunk text.

    The cached ``document`` holds only the index-relevant fields (``corpus.INDEX_FIELDS``); effective
    dates are runtime-only filters and are never written into caches or chunk rows.
    """
    definition = document.index_definition()
    changed = cache.get("document") != definition
    if changed:
        cache["document"] = definition
    expected = {
        "document_id": document.document_id,
        "document_name": document.document_name,
        "document_type": document.document_type,
        "legal_track": document.legal_track,
        "asset_scopes": list(document.asset_scopes),
        "party_scope": list(document.party_scope),
    }
    for chunk in cache.get("chunks") or []:
        if any(chunk.get(key) != value for key, value in expected.items()):
            chunk.update(expected)
            changed = True
        # These names belonged to the original manifest contract. Keep generated caches unambiguous
        # after a metadata-only refresh rather than carrying two names for the same values.
        for obsolete_key in ("instrument_name", "instrument_type"):
            if obsolete_key in chunk:
                chunk.pop(obsolete_key, None)
                changed = True
    return changed


def refresh_document_cache(
    document: AgreementDocumentSpec, cache_path: Path, *, force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> str:
    """Bring one document's cache up to date; return ``"reused"`` or ``"rebuilt"``.

    Reuse when the PDF hash and extraction version match; re-chunk from cached pages when only the
    extraction version changed; otherwise extract (text layer / OCR) from the PDF.
    """
    path = source_path(document)
    if not path.is_file():
        raise FileNotFoundError(f"agreement source PDF not found: {path}")
    source_sha = sha256_file(path)
    previous = read_cache(cache_path)
    # --force discards cached OCR, but the old vectors can still be carried onto unchanged text.
    cached = {} if force else previous
    if (
        cached.get("source_sha256") == source_sha
        and cached.get("extraction_version") == EXTRACTION_VERSION
    ):
        # A manifest edit may change an instrument's asset/party/legal-track metadata without
        # changing its PDF. Keep the expensive OCR/chunks, but persist the current tracked document
        # definition before the SQLite rows are rebuilt.
        if sync_document_metadata(cached, document):
            write_json_atomic(cache_path, cached)
        if progress:
            progress(f"reuse {document.document_id}")
        return "reused"
    if (
        cached.get("source_sha256") == source_sha
        and int(cached.get("ocr_dpi") or 0) == OCR_DPI
        and cached.get("pages")
    ):
        if progress:
            progress(f"re-chunk {document.document_id} from cached {OCR_DPI}-DPI OCR")
        earlier = {"embedding": cached.get("embedding"), "chunks": cached.get("chunks") or []}
        cached["document"] = document.index_definition()
        cached["extraction_version"] = EXTRACTION_VERSION
        cached["chunks"] = chunk_document(document, list(cached["pages"]))
        carry_over_embeddings(earlier, cached)
        write_json_atomic(cache_path, cached)
        return "rebuilt"
    if progress:
        progress(f"OCR {document.document_id}: {path.name}")
    pages, methods = extract_pages(path, document.document_id, progress=progress)
    extracted = {
        "document": document.index_definition(),
        "source_path": str(path),
        "source_sha256": source_sha,
        "extraction_version": EXTRACTION_VERSION,
        "ocr_dpi": OCR_DPI,
        "methods": methods,
        "pages": pages,
        "chunks": chunk_document(document, pages),
    }
    carry_over_embeddings(previous, extracted)
    write_json_atomic(cache_path, extracted)
    return "rebuilt"


def embed_cache(cache: dict[str, Any], embedder: Embedder) -> int:
    """Embed every chunk in one document cache that lacks a current vector; return how many."""
    expected = {"model": embedder.model, "dimensions": embedder.dimensions}
    chunks = cache.get("chunks") or []
    if cache.get("embedding") != expected:
        # A model or dimension change invalidates every vector; never mix two vector spaces.
        for chunk in chunks:
            chunk.pop("embedding_b64", None)
        cache.pop("embedding", None)
    pending = [chunk for chunk in chunks if not chunk.get("embedding_b64")]
    encoded: list[str] = []
    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start:start + BATCH_SIZE]
        vectors = embedder.embed([chunk["verbatim_text"] for chunk in batch])
        if len(vectors) != len(batch):
            raise RuntimeError(f"embedding deployment returned {len(vectors)} vectors for {len(batch)} inputs")
        for vector in vectors:
            if len(vector) != embedder.dimensions:
                raise RuntimeError(
                    f"embedding deployment returned a {len(vector)}-dim vector; expected "
                    f"{embedder.dimensions}. Check AZURE_EMBEDDING_DEPLOYMENT and AZURE_EMBEDDING_DIMENSIONS."
                )
            encoded.append(encode_vector(vector))
    # Assign only after every batch succeeded, so a failed call leaves the cache as it was.
    for chunk, value in zip(pending, encoded):
        chunk["embedding_b64"] = value
    cache["embedding"] = expected
    return len(pending)


def carry_over_embeddings(previous: dict[str, Any], current: dict[str, Any]) -> None:
    """Copy vectors from an older cache onto chunks whose text did not change."""
    if not previous.get("embedding"):
        return
    by_content = {
        chunk["content_sha256"]: chunk["embedding_b64"]
        for chunk in previous.get("chunks") or [] if chunk.get("embedding_b64")
    }
    for chunk in current.get("chunks") or []:
        vector = by_content.get(chunk["content_sha256"])
        if vector:
            chunk["embedding_b64"] = vector
    current["embedding"] = dict(previous["embedding"])


__all__ = [
    "BATCH_SIZE", "carry_over_embeddings", "embed_cache", "read_cache", "refresh_document_cache",
    "sha256_file", "sync_document_metadata", "write_json_atomic",
]
