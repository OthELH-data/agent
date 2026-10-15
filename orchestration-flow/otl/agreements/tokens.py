"""Token counting with the cl100k_base encoding, shared by chunking and query-embedding bounding.

``cl100k_base`` is the encoding ``text-embedding-3-small`` uses. ``tiktoken`` is imported lazily, so
importing this module stays cheap.
"""
from __future__ import annotations

import re
from functools import lru_cache

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'&/-]*")


@lru_cache(maxsize=1)
def cl100k_encoding():
    """The cl100k_base tiktoken encoding, or None if it cannot be loaded (the result is cached)."""
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - only used in minimal/offline installs
        return None


def token_count(text: str) -> int:
    encoding = cl100k_encoding()
    if encoding is not None:
        return len(encoding.encode(text))
    return max(1, int(len(_WORD.findall(text)) * 1.35))


__all__ = ["cl100k_encoding", "token_count"]
