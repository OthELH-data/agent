"""SQLite storage for the agreement index: schema, the build-time writer, and the runtime reader.

Writer and reader live together so the table layout is defined once; swapping SQLite for Databricks
Vector Search means replacing this module behind the same ``AgreementIndex`` interface. ``chunks`` is
the single source table (one row per ``ChunkRecord``); ``chunk_fts`` (FTS5) and ``chunk_vec``
(sqlite-vec) are derived indexes over it; ``metadata`` holds the build facts ``ready_for()`` checks.
The runtime reader uses plain ``sqlite3``, except ``AgreementIndex.vector_ranking()``, which loads
sqlite-vec on a read-only connection to run scoped KNN search.
"""
from __future__ import annotations

import contextlib
import os
import re
import sqlite3
import sys
import tempfile
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .corpus import AgreementCorpusManifest, AgreementDocumentSpec, index_dir as default_index_dir
from .embedding import vector_bytes
from .record import ChunkRecord, decode_row


SCHEMA_VERSION = 3
# Identifies the extraction + chunking output. Ingestion stamps it on each document cache; the index
# records it in ``metadata`` so ``ready_for()`` rejects an index built from different text.
EXTRACTION_VERSION = "agreement-ocr-v6-major-boundaries-deduped"
INDEX_FILENAME = "index.sqlite3"

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'&/-]*")
_STOPWORDS = {
    "about", "after", "against", "also", "among", "and", "are", "because", "been", "before",
    "being", "between", "both", "but", "can", "could", "does", "each", "from", "had", "has",
    "have", "into", "its", "may", "more", "not", "only", "other", "our", "should", "such",
    "than", "that", "the", "their", "them", "there", "these", "they", "this", "those", "through",
    "under", "was", "were", "what", "when", "where", "which", "while", "will", "with", "would",
}


def connect_with_vec(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    if not hasattr(connection, "enable_load_extension"):
        connection.close()
        raise RuntimeError(
            "this Python's sqlite3 cannot load extensions (needed for sqlite-vec); "
            "build the agreement index with the repo .venv (Homebrew/uv Python)"
        )
    # Imported here, not at module top: only the build's writer needs sqlite-vec; the runtime reader
    # (AgreementIndex) uses plain sqlite3 and must not require the extension.
    import sqlite_vec

    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)
    return connection


def _connect_vec_readonly(path: Path) -> sqlite3.Connection:
    """Open ``path`` read-only with sqlite-vec loaded, for ``vector_ranking()`` only."""
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    if not hasattr(connection, "enable_load_extension"):
        connection.close()
        raise RuntimeError(
            "this Python's sqlite3 cannot load extensions (needed for sqlite-vec); "
            "build the agreement index with the repo .venv (Homebrew/uv Python)"
        )
    import sqlite_vec

    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)
    return connection


def _vector_bytes(vector: Sequence[float]) -> bytes:
    """Little-endian float32 bytes for a query vector, the layout sqlite-vec MATCH expects."""
    values = array("f", vector)
    if sys.byteorder != "little":  # pragma: no cover - every supported host is little-endian
        values.byteswap()
    return values.tobytes()


def create_database(
    path: Path, caches: list[dict[str, Any]], manifest: AgreementCorpusManifest,
    embedding: dict[str, Any] | None = None,
) -> None:
    """Write chunks, the FTS5 keyword index and, when ``embedding`` is given, the sqlite-vec index."""
    connection = connect_with_vec(path) if embedding else sqlite3.connect(path)
    try:
        connection.executescript(ChunkRecord.sqlite_table_sql() + ";" + """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE VIRTUAL TABLE chunk_fts USING fts5(
                chunk_id UNINDEXED,
                verbatim_text,
                heading_path,
                document_name,
                tokenize='unicode61 remove_diacritics 2'
            );
        """)
        metadata = {
            "schema_version": str(SCHEMA_VERSION),
            "extraction_version": EXTRACTION_VERSION,
            "manifest_fingerprint": manifest.fingerprint,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        dimensions = 0
        if embedding:
            dimensions = int(embedding["dimensions"])
            connection.execute(f"""
                CREATE VIRTUAL TABLE chunk_vec USING vec0(
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT partition key,
                    legal_track TEXT,
                    embedding float[{dimensions}] distance_metric=cosine
                )
            """)
            metadata["embedding_model"] = str(embedding["model"])
            metadata["embedding_dimensions"] = str(dimensions)
        connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items())
        columns = ChunkRecord.columns()
        insert_chunk = (
            f"INSERT INTO chunks ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})"
        )
        for cache in caches:
            if embedding and cache.get("embedding") != embedding:
                raise ValueError(
                    f"agreement cache {cache['document']['document_id']} has no "
                    f"{embedding['model']}/{dimensions} embeddings"
                )
            for chunk in cache.get("chunks") or []:
                record = ChunkRecord.model_validate(chunk)
                connection.execute(insert_chunk, record.to_row())
                connection.execute(
                    "INSERT INTO chunk_fts VALUES (?, ?, ?, ?)",
                    (record.chunk_id, record.verbatim_text, record.heading_path, record.document_name),
                )
                if embedding:
                    connection.execute(
                        "INSERT INTO chunk_vec(chunk_id, document_id, legal_track, embedding) "
                        "VALUES (?, ?, ?, ?)",
                        (record.chunk_id, record.document_id, record.legal_track,
                         vector_bytes(chunk, dimensions)),
                    )
        connection.commit()
    finally:
        connection.close()


def write_index(
    root: Path, caches: list[dict[str, Any]], manifest: AgreementCorpusManifest,
    embedding: dict[str, Any] | None = None,
) -> Path:
    """Build the database in a temporary file and atomically replace ``<root>/index.sqlite3``."""
    root.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix="agreement-index-", suffix=".sqlite3", dir=root)
    os.close(fd)
    temporary = Path(temporary_name)
    target = root / INDEX_FILENAME
    try:
        create_database(temporary, caches, manifest, embedding=embedding)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _fts_query(text: str, limit: int = 80) -> str:
    terms: list[str] = []
    seen: set[str] = set()
    for raw in _WORD.findall(str(text or "")):
        term = raw.casefold().strip("-/")
        if len(term) < 3 or term in _STOPWORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term.replace('"', '""'))
        if len(terms) >= limit:
            break
    return " OR ".join(f'"{term}"' for term in terms)


class AgreementIndex:
    """Read-only runtime access to the generated FTS5/sqlite-vec agreement index.

    Every method here uses plain ``sqlite3``, except ``vector_ranking()``, which opens a separate
    read-only connection with sqlite-vec loaded for the duration of the call.
    """

    def __init__(self, path: str | Path | None = None):
        root = default_index_dir() if path is None else Path(path).expanduser().resolve()
        self.path = root if root.suffix == ".sqlite3" else root / INDEX_FILENAME

    def exists(self) -> bool:
        return self.path.is_file()

    def metadata(self) -> dict[str, str]:
        if not self.exists():
            return {}
        with sqlite3.connect(self.path) as connection:
            return dict(connection.execute("SELECT key, value FROM metadata").fetchall())

    def ready_for(self, manifest: AgreementCorpusManifest) -> bool:
        metadata = self.metadata()
        return (
            metadata.get("schema_version") == str(SCHEMA_VERSION)
            and metadata.get("extraction_version") == EXTRACTION_VERSION
            and metadata.get("manifest_fingerprint") == manifest.fingerprint
        )

    def get_chunks(self, chunk_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        ids = list(dict.fromkeys(str(value) for value in chunk_ids if str(value)))
        if not ids or not self.exists():
            return {}
        placeholders = ",".join("?" for _ in ids)
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", ids
            ).fetchall()
        decoded: dict[str, dict[str, Any]] = {}
        for row in rows:
            chunk = decode_row(dict(row))
            decoded[chunk["chunk_id"]] = chunk
        return decoded

    def lexical_ranking(
        self, query: str, documents: list[AgreementDocumentSpec], depth: int = 100,
    ) -> list[dict[str, Any]]:
        """BM25 over all applicable documents as one list: no legal-track split, no quote_safe filter.

        Rows are decoded chunks plus ``bm25_score`` (lower is better) and a 1-based ``lexical_rank``,
        ordered by ``(bm25_score, chunk_id)``. The FTS expression is built by ``_fts_query``.
        """
        match = _fts_query(query)
        if not match or not documents or depth <= 0 or not self.exists():
            return []
        document_ids = list(dict.fromkeys(doc.document_id for doc in documents))
        placeholders = ",".join("?" for _ in document_ids)
        sql = f"""
            SELECT c.*, bm25(chunk_fts, 1.0, 2.5, 1.4) AS bm25_score
            FROM chunk_fts
            JOIN chunks c ON c.chunk_id = chunk_fts.chunk_id
            WHERE chunk_fts MATCH ? AND c.document_id IN ({placeholders})
            ORDER BY bm25_score ASC, c.chunk_id ASC
            LIMIT ?
        """
        with contextlib.closing(sqlite3.connect(self.path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(sql, [match, *document_ids, depth]).fetchall()
        ranked: list[dict[str, Any]] = []
        for rank, row in enumerate(rows, start=1):
            candidate = decode_row(dict(row))
            candidate["bm25_score"] = float(candidate["bm25_score"])
            candidate["lexical_rank"] = rank
            ranked.append(candidate)
        return ranked

    def vector_ranking(
        self, query_vector: Sequence[float], documents: list[AgreementDocumentSpec], depth: int = 100,
    ) -> list[dict[str, Any]]:
        """Cosine KNN over all applicable documents as one list.

        KNN runs per document (partition-scoped by ``document_id``), hits are merged, sorted by
        ``(vector_distance, chunk_id)`` and cut to ``depth``. Rows are decoded chunks plus
        ``vector_distance`` and a 1-based ``vector_rank``. Raises ``ValueError`` on a dimension mismatch.
        """
        if not documents or depth <= 0 or not self.has_vectors():
            return []
        dimensions = int(self.metadata()["embedding_dimensions"])
        if len(query_vector) != dimensions:
            raise ValueError(f"query vector has {len(query_vector)} dims; index expects {dimensions}")
        document_ids = list(dict.fromkeys(doc.document_id for doc in documents))
        applicable_ids = set(document_ids)
        query_bytes = _vector_bytes(query_vector)
        hits: dict[str, float] = {}
        with contextlib.closing(_connect_vec_readonly(self.path)) as connection:
            for document_id in document_ids:
                rows = connection.execute(
                    "SELECT chunk_id, distance FROM chunk_vec WHERE embedding MATCH ? "
                    "AND document_id = ? AND k = ? ORDER BY distance",
                    (query_bytes, document_id, depth),
                ).fetchall()
                for chunk_id, distance in rows:
                    hits[chunk_id] = float(distance)
        best = sorted(hits.items(), key=lambda item: (item[1], item[0]))[:depth]
        chunks = self.get_chunks(chunk_id for chunk_id, _ in best)
        ranked: list[dict[str, Any]] = []
        for chunk_id, distance in best:
            chunk = chunks.get(chunk_id)
            if chunk is None or chunk["document_id"] not in applicable_ids:
                continue
            candidate = dict(chunk)
            candidate["vector_distance"] = distance
            candidate["vector_rank"] = len(ranked) + 1
            ranked.append(candidate)
        return ranked

    def chunks_for_documents(self, documents: list[AgreementDocumentSpec]) -> list[dict[str, Any]]:
        """Every chunk of the applicable documents, decoded, ordered (document_id, pdf_page_start, chunk_id)."""
        document_ids = list(dict.fromkeys(doc.document_id for doc in documents))
        if not document_ids or not self.exists():
            return []
        placeholders = ",".join("?" for _ in document_ids)
        with contextlib.closing(sqlite3.connect(self.path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                f"SELECT * FROM chunks WHERE document_id IN ({placeholders}) "
                "ORDER BY document_id ASC, pdf_page_start ASC, chunk_id ASC",
                document_ids,
            ).fetchall()
        return [decode_row(dict(row)) for row in rows]

    def has_vectors(self) -> bool:
        if not self.exists():
            return False
        metadata = self.metadata()
        if not metadata.get("embedding_model") or not metadata.get("embedding_dimensions"):
            return False
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chunk_vec'"
            ).fetchone()
        return row is not None


__all__ = [
    "AgreementIndex", "EXTRACTION_VERSION", "INDEX_FILENAME", "SCHEMA_VERSION", "connect_with_vec",
    "create_database", "write_index",
]
