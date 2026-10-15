"""Azure AI Foundry embedding settings, client and vector encoding for agreement chunks.

Shared by ingestion (chunk embeddings, using ``AzureEmbedder``'s default retry/timeout) and runtime
retrieval (query embeddings), which must use the same model, deployment and dimensions so both live
in one vector space. Runtime callers construct ``AzureEmbedder`` with ``RUNTIME_MAX_RETRIES`` and
``RUNTIME_TIMEOUT_SECONDS`` so an unavailable embedding service fails fast instead of holding an
audit run for ingestion's much longer timeout and retry budget. ``bound_embedding_input()`` keeps a
runtime query under the model's input token limit before it reaches the client, and
``validate_vectors()`` checks the shape of whatever the deployment returns. Settings come from the
repo-root ``.env`` (loaded by ``otl._bootstrap``). Vectors are stored as base64 little-endian
float32, the layout sqlite-vec expects for a ``float[]`` column.
"""
from __future__ import annotations

import base64
import math
import os
import sys
from array import array
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from .tokens import cl100k_encoding

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIMENSIONS = 1536
DEFAULT_API_VERSION = "2024-10-21"
RUNTIME_MAX_RETRIES = 1
RUNTIME_TIMEOUT_SECONDS = 15.0
# text-embedding-3-small accepts up to 8,192 input tokens; keep a safety margin below that.
MAX_INPUT_TOKENS = 8000
_REQUIRED = ("AZURE_EMBEDDING_ENDPOINT", "AZURE_EMBEDDING_API_KEY", "AZURE_EMBEDDING_DEPLOYMENT")


class EmbeddingConfigError(RuntimeError):
    """Required Azure embedding settings are missing or invalid."""


@dataclass(frozen=True)
class EmbeddingSettings:
    endpoint: str
    api_key: str = field(repr=False)
    api_version: str
    deployment: str
    model: str
    dimensions: int

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "EmbeddingSettings":
        env = os.environ if environ is None else environ
        missing = [name for name in _REQUIRED if not (env.get(name) or "").strip()]
        if missing:
            raise EmbeddingConfigError(
                "missing Azure embedding settings in the repo-root .env: " + ", ".join(missing)
            )
        raw_dimensions = (env.get("AZURE_EMBEDDING_DIMENSIONS") or str(DEFAULT_DIMENSIONS)).strip()
        try:
            dimensions = int(raw_dimensions)
        except ValueError:
            dimensions = 0
        if dimensions <= 0:
            raise EmbeddingConfigError(
                f"AZURE_EMBEDDING_DIMENSIONS must be a positive integer, got {raw_dimensions!r}"
            )
        return cls(
            endpoint=env["AZURE_EMBEDDING_ENDPOINT"].strip().rstrip("/"),
            api_key=env["AZURE_EMBEDDING_API_KEY"].strip(),
            api_version=(env.get("AZURE_EMBEDDING_API_VERSION") or DEFAULT_API_VERSION).strip(),
            deployment=env["AZURE_EMBEDDING_DEPLOYMENT"].strip(),
            model=(env.get("AZURE_EMBEDDING_MODEL") or DEFAULT_MODEL).strip(),
            dimensions=dimensions,
        )


class Embedder(Protocol):
    model: str
    dimensions: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class AzureEmbedder:
    """Calls one Azure AI Foundry embedding deployment."""

    def __init__(self, settings: EmbeddingSettings, *, max_retries: int = 5, timeout: float = 120.0):
        # Imported here, not at module top: store.py imports this module for vector_bytes, and readers of
        # the index should not pay for loading the OpenAI SDK unless they actually call Azure.
        from openai import AzureOpenAI

        self.model = settings.model
        self.dimensions = settings.dimensions
        self._deployment = settings.deployment
        self._client = AzureOpenAI(
            azure_endpoint=settings.endpoint, api_key=settings.api_key,
            api_version=settings.api_version, max_retries=max_retries, timeout=timeout,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self._deployment, input=list(texts), dimensions=self.dimensions,
        )
        return [list(item.embedding) for item in sorted(response.data, key=lambda item: item.index)]


class EmbeddingResponseError(ValueError):
    """The embedding deployment returned a response that does not match what was requested."""


def bound_embedding_input(
    text: str, max_tokens: int = MAX_INPUT_TOKENS, *, encoding: Any = None,
) -> tuple[str, bool]:
    """Keep at most the first ``max_tokens`` tokens of ``text``; report whether it was shortened.

    Tokenizes with cl100k_base, the encoding ``text-embedding-3-small`` uses, loaded through
    ``tokens.cl100k_encoding()``, the loader chunking also uses; ``encoding`` lets
    tests inject a tokenizer. If no tokenizer is available, falls back to a conservative
    ``max_tokens * 3`` character cap.
    """
    tokenizer = encoding if encoding is not None else cl100k_encoding()
    if tokenizer is None:
        limit = max_tokens * 3
        return (text, False) if len(text) <= limit else (text[:limit], True)
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text, False
    keep = max_tokens
    shortened = tokenizer.decode(tokens[:keep])
    # Re-encoding a decoded prefix can come out a token over at the cut point; trim until it fits.
    while keep > 0 and len(tokenizer.encode(shortened)) > max_tokens:
        keep -= 1
        shortened = tokenizer.decode(tokens[:keep])
    return shortened, True


def validate_vectors(
    vectors: Sequence[Sequence[float]], expected_count: int, dimensions: int,
) -> list[list[float]]:
    """Validate embedding vectors returned by the deployment; raise EmbeddingResponseError on mismatch.

    Checks the vector count, each vector's length, and that every value is finite (rejects NaN and
    +/-inf, which a misconfigured or failing deployment can return). Error messages report only
    counts and dimensions, never vector contents or request/response credentials.
    """
    count = len(vectors)
    if count != expected_count:
        raise EmbeddingResponseError(
            f"embedding deployment returned {count} vectors; expected {expected_count}"
        )
    validated: list[list[float]] = []
    for index, vector in enumerate(vectors):
        values = [float(value) for value in vector]
        if len(values) != dimensions:
            raise EmbeddingResponseError(
                f"embedding vector {index} has {len(values)} dimensions; expected {dimensions}"
            )
        if not all(math.isfinite(value) for value in values):
            raise EmbeddingResponseError(f"embedding vector {index} contains a non-finite value")
        validated.append(values)
    return validated


def encode_vector(vector: Sequence[float]) -> str:
    values = array("f", vector)
    if sys.byteorder != "little":  # pragma: no cover - every supported host is little-endian
        values.byteswap()
    return base64.b64encode(values.tobytes()).decode("ascii")


def decode_vector(text: str) -> list[float]:
    values = array("f")
    values.frombytes(base64.b64decode(text))
    if sys.byteorder != "little":  # pragma: no cover
        values.byteswap()
    return values.tolist()


def vector_bytes(chunk: dict[str, Any], dimensions: int) -> bytes:
    """Raw little-endian float32 bytes, the layout sqlite-vec expects for a float[] column."""
    raw = base64.b64decode(chunk.get("embedding_b64") or "")
    if len(raw) != 4 * dimensions:
        raise ValueError(
            f"chunk {chunk.get('chunk_id')} has a {len(raw) // 4}-dim vector; expected {dimensions}"
        )
    return raw


__all__ = [
    "AzureEmbedder", "DEFAULT_DIMENSIONS", "DEFAULT_MODEL", "Embedder", "EmbeddingConfigError",
    "EmbeddingResponseError", "EmbeddingSettings", "MAX_INPUT_TOKENS", "RUNTIME_MAX_RETRIES",
    "RUNTIME_TIMEOUT_SECONDS", "bound_embedding_input", "decode_vector", "encode_vector",
    "validate_vectors", "vector_bytes",
]
