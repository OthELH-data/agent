"""Pure reciprocal rank fusion (RRF) for agreement retrieval.

``rrf_lists`` fuses any number of named ranked lists by 1-based position: ``score = sum(1 / (k + rank))``
over the lists that contain a chunk, with ``k = RRF_K`` (60). The agreement retriever uses it twice:
within one search, to fuse its BM25 and vector lists when both are non-empty, and across searches, to
fuse the auditor and Shell findings lists when both returned rows. A single non-empty list is never
passed through RRF; the retriever keeps its own order instead.

This module holds no I/O: no SQLite, no Azure, no imports from `store.py`. Inputs are plain dicts and
are never mutated; every returned row is a new dict.
"""
from __future__ import annotations

from typing import Any

RRF_K = 60

_SCORE_FIELDS = ("bm25_score", "lexical_rank", "vector_distance", "vector_rank")


def rrf_lists(ranked_lists: dict[str, list[dict[str, Any]]], k: int = RRF_K) -> list[dict[str, Any]]:
    """Reciprocal rank fusion across any number of named ranked lists.

    A chunk's rank in a list is its 1-based position there (a repeated ``chunk_id`` counts only at its
    first position), and ``fusion_score(c) = sum(1 / (k + rank))`` over the lists containing ``c``.
    Each output row is a new dict holding the chunk fields from its first appearance (in input-dict
    order) minus the list-specific score fields (``bm25_score``, ``lexical_rank``, ``vector_distance``,
    ``vector_rank``), plus ``fusion_score``, a 1-based ``fusion_rank`` and ``retrieval_lists``:
    ``[{"list": name, "rank": r}, ...]`` in input-dict order. Rows are sorted by ``fusion_score``
    descending, then ``chunk_id`` ascending. The inputs are never mutated.
    """
    records: dict[str, dict[str, Any]] = {}
    for name, rows in ranked_lists.items():
        seen: set[str] = set()
        for position, row in enumerate(rows, start=1):
            chunk_id = row["chunk_id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            record = records.get(chunk_id)
            if record is None:
                record = {key: value for key, value in row.items() if key not in _SCORE_FIELDS}
                record["fusion_score"] = 0.0
                record["retrieval_lists"] = []
                records[chunk_id] = record
            record["fusion_score"] += 1.0 / (k + position)
            record["retrieval_lists"].append({"list": name, "rank": position})

    fused = sorted(records.values(), key=lambda row: (-row["fusion_score"], row["chunk_id"]))
    for rank, record in enumerate(fused, start=1):
        record["fusion_rank"] = rank
    return fused


__all__ = ["RRF_K", "rrf_lists"]
