"""CLI for the explicit, incremental agreement-index build.

Refresh each document cache, embed new chunk text, write SQLite.

Ingestion is an explicit offline operation; no chat LLM is involved.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .corpus import AgreementCorpusManifest, index_dir as default_index_dir, load_manifest
from .embedding import AzureEmbedder, Embedder, EmbeddingSettings
from .store import EXTRACTION_VERSION, SCHEMA_VERSION, write_index
from .cache import embed_cache, refresh_document_cache, write_json_atomic


def build_index(
    *, manifest: AgreementCorpusManifest | None = None, output_dir: str | Path | None = None,
    document_ids: Iterable[str] | None = None, force: bool = False,
    progress: Callable[[str], None] | None = None, embedder: Embedder | None = None,
) -> dict[str, Any]:
    """Incrementally OCR and embed the corpus, then atomically rebuild the SQLite search index."""
    manifest = manifest or load_manifest()
    # Resolve the Azure settings before any OCR so a missing .env value fails in seconds.
    embedder = embedder or AzureEmbedder(EmbeddingSettings.from_env())
    root = Path(output_dir).expanduser().resolve() if output_dir else default_index_dir()
    cache_dir = root / "documents"
    cache_dir.mkdir(parents=True, exist_ok=True)
    requested = set(document_ids or [])
    unknown = requested - {doc.document_id for doc in manifest.documents}
    if unknown:
        raise ValueError(f"unknown agreement document IDs: {sorted(unknown)}")

    rebuilt: list[str] = []
    reused: list[str] = []
    for document in manifest.documents:
        cache_path = cache_dir / f"{document.document_id}.json"
        should_process = not requested or document.document_id in requested or not cache_path.is_file()
        if not should_process:
            continue
        outcome = refresh_document_cache(document, cache_path, force=force, progress=progress)
        (reused if outcome == "reused" else rebuilt).append(document.document_id)

    caches: list[dict[str, Any]] = []
    missing_caches: list[str] = []
    embedded_chunks = 0
    for document in manifest.documents:
        cache_path = cache_dir / f"{document.document_id}.json"
        if not cache_path.is_file():
            missing_caches.append(document.document_id)
            continue
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        before = cache.get("embedding")
        count = embed_cache(cache, embedder)
        if count or before != cache.get("embedding"):
            # Persist per document, so an Azure failure on a later document keeps this one's vectors.
            write_json_atomic(cache_path, cache)
            if progress:
                progress(f"embed {document.document_id}: {count} chunks")
        embedded_chunks += count
        caches.append(cache)
    if missing_caches:
        raise RuntimeError(f"agreement index caches missing after build: {missing_caches}")

    embedding = {"model": embedder.model, "dimensions": embedder.dimensions}
    write_index(root, caches, manifest, embedding=embedding)

    build_manifest = {
        "schema_version": SCHEMA_VERSION,
        "extraction_version": EXTRACTION_VERSION,
        "manifest_fingerprint": manifest.fingerprint,
        "embedding_model": embedder.model,
        "embedding_dimensions": embedder.dimensions,
        "rebuilt_documents": rebuilt,
        "reused_documents": reused,
        "document_count": len(caches),
        "page_count": sum(len(cache.get("pages") or []) for cache in caches),
        "chunk_count": sum(len(cache.get("chunks") or []) for cache in caches),
        "embedded_chunks": embedded_chunks,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json_atomic(root / "build_manifest.json", build_manifest)
    if progress:
        progress(
            f"index ready: {build_manifest['document_count']} documents, "
            f"{build_manifest['page_count']} pages, {build_manifest['chunk_count']} chunks, "
            f"{embedded_chunks} newly embedded"
        )
    return build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR, embed and index the configured agreement corpus")
    parser.add_argument("--document", action="append", default=[], help="document_id to rebuild")
    parser.add_argument("--force", action="store_true", help="OCR selected documents even if hashes match")
    parser.add_argument("--output-dir", default=None, help="override OTL_AGREEMENT_INDEX_DIR")
    args = parser.parse_args()
    result = build_index(
        output_dir=args.output_dir,
        document_ids=args.document or None,
        force=args.force,
        progress=print,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
