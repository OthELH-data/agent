"""Single definition of one stored agreement chunk.

``ChunkRecord`` is the storage row. The JSON caches under ``.agreement_rag/documents``, the SQLite
``chunks`` table and the future Databricks Delta table all carry exactly these columns. The downstream
contract ``AgreementChunk`` in ``otl/state.py`` is built from a stored row and is a separate model.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

# List columns are JSON-encoded in SQLite. Whether Databricks Vector Search can filter array columns is
# still unverified, so this encoding lives here and nowhere else.
JSON_LIST_COLUMNS = ("asset_scopes", "party_scope", "printed_page_labels", "extraction_methods")

_SQLITE_TYPES = {str: "TEXT", int: "INTEGER", float: "REAL", bool: "INTEGER"}


class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    document_type: str
    legal_track: str
    asset_scopes: list[str] = Field(default_factory=list)
    party_scope: list[str] = Field(default_factory=list)
    heading_path: str = ""
    pdf_page_start: int
    pdf_page_end: int
    printed_page_labels: list[str] = Field(default_factory=list)
    extraction_methods: list[str] = Field(default_factory=list)
    verbatim_text: str
    ocr_confidence: float
    quote_safe: bool
    content_sha256: str
    token_count: int

    @classmethod
    def columns(cls) -> list[str]:
        return list(cls.model_fields)

    @classmethod
    def sqlite_table_sql(cls, table: str = "chunks") -> str:
        definitions = []
        for name, field in cls.model_fields.items():
            kind = "TEXT" if name in JSON_LIST_COLUMNS else _SQLITE_TYPES[field.annotation]
            constraint = "PRIMARY KEY" if name == "chunk_id" else "NOT NULL"
            definitions.append(f"{name} {kind} {constraint}")
        return f"CREATE TABLE {table} (\n    " + ",\n    ".join(definitions) + "\n)"

    def to_row(self) -> tuple[Any, ...]:
        values = self.model_dump()
        for name in JSON_LIST_COLUMNS:
            values[name] = json.dumps(values[name])
        values["quote_safe"] = int(values["quote_safe"])
        return tuple(values[name] for name in self.columns())


def decode_row(row: dict[str, Any]) -> dict[str, Any]:
    """Turn a SQLite ``chunks`` row (plus any extra SELECT columns) back into Python values."""
    value = dict(row)
    for name in JSON_LIST_COLUMNS:
        value[name] = json.loads(value[name])
    value["quote_safe"] = bool(value["quote_safe"])
    return value


__all__ = ["ChunkRecord", "JSON_LIST_COLUMNS", "decode_row"]
