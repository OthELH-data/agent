"""Local proper-name candidate generation for JADE population discovery.

The configured model is a token-classification part-of-speech model, not a PERSON NER model.  Its
``PROPN`` tags are therefore high-recall suggestions for the grounded person review in
``ledger_people.py``; they are never treated as final identities on their own.

The optional model is lazily loaded on CPU and cached once per process.  It reads only the local Hugging
Face cache unless ``JV_AUDIT_HF_ALLOW_DOWNLOAD`` is explicitly enabled.  A missing dependency or model
therefore leaves population discovery on its existing LLM-only path immediately rather than failing or
delaying the JADE node.
"""
from __future__ import annotations

import functools
import os
import re
import threading
from pathlib import Path
from typing import Any

MODEL_ID = "GozdeA/finetuned-bertweet-poskd3"
_PIPELINE = None
_LOAD_ERROR = ""
_LOCK = threading.RLock()


def enabled() -> bool:
    """Return whether local POS candidate generation is enabled."""
    return str(os.getenv("OTL_NAME_ENTITY_RECOGNITION", "1")).strip().lower() not in {
        "0", "false", "no", "off",
    }


def _cache_dir() -> Path:
    configured = os.getenv("JV_AUDIT_HF_CACHE") or os.getenv("HF_HOME")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1] / ".cache" / "huggingface"


def _model_id() -> str:
    return str(os.getenv("JV_AUDIT_POS_MODEL") or MODEL_ID).strip()


def _allow_download() -> bool:
    return str(os.getenv("JV_AUDIT_HF_ALLOW_DOWNLOAD", "0")).strip().lower() in {
        "1", "true", "yes", "on",
    }


def _load_pipeline():
    global _PIPELINE, _LOAD_ERROR
    if _PIPELINE is not None:
        return _PIPELINE
    if _LOAD_ERROR or not enabled():
        return None
    with _LOCK:
        if _PIPELINE is not None:
            return _PIPELINE
        if _LOAD_ERROR:
            return None
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

            cache = str(_cache_dir())
            model_id = _model_id()
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, cache_dir=cache, local_files_only=not _allow_download(),
            )
            model = AutoModelForTokenClassification.from_pretrained(
                model_id, cache_dir=cache, local_files_only=not _allow_download(),
            )
            _PIPELINE = pipeline(
                "token-classification", model=model, tokenizer=tokenizer,
                aggregation_strategy="simple", device=-1,
            )
        except Exception as exc:  # Optional local capability; the LLM-only review remains available.
            _LOAD_ERROR = f"{type(exc).__name__}: {exc}"
            return None
    return _PIPELINE


def model_status() -> dict[str, Any]:
    """Expose safe provenance for trace/debug output without forcing a model load."""
    return {
        "enabled": enabled(),
        "model": _model_id(),
        "loaded": _PIPELINE is not None,
        "error": _LOAD_ERROR,
        "cache_dir": str(_cache_dir()),
        "allow_download": _allow_download(),
    }


def _windows(text: str, max_chars: int = 1200) -> list[tuple[int, str]]:
    """Cover long values in whitespace-aligned chunks while preserving original offsets."""
    if len(text) <= max_chars:
        return [(0, text)]
    out: list[tuple[int, str]] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        out.append((start, text[start:end]))
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    return out


def _candidate(text: str, start: int, end: int, scores: list[float]) -> dict[str, Any] | None:
    surface = text[start:end].strip(" \t\r\n,;:()[]{}")
    if not surface or not re.search(r"[A-Za-z]", surface):
        return None
    return {
        "text": surface,
        "start": start,
        "end": end,
        "score": round(sum(scores) / max(1, len(scores)), 6),
        "method": "bertweet_pos_propn",
    }


@functools.lru_cache(maxsize=4096)
def _tag_text(text: str) -> tuple[dict[str, Any], ...]:
    tagger = _load_pipeline()
    if tagger is None:
        return ()
    spans: list[dict[str, Any]] = []
    with _LOCK:
        for base, window in _windows(text):
            proper = [
                result for result in (tagger(window) or [])
                if str(result.get("entity_group") or result.get("entity") or "").upper() == "PROPN"
                and int(result.get("end") or 0) > int(result.get("start") or 0)
            ]
            if not proper:
                continue
            current_start = int(proper[0]["start"])
            current_end = int(proper[0]["end"])
            scores = [float(proper[0].get("score") or 0.0)]
            for result in proper[1:]:
                start, end = int(result["start"]), int(result["end"])
                if re.fullmatch(r"[\s,.'’\-]*", window[current_end:start] or ""):
                    current_end = end
                    scores.append(float(result.get("score") or 0.0))
                    continue
                candidate = _candidate(text, base + current_start, base + current_end, scores)
                if candidate:
                    spans.append(candidate)
                current_start, current_end = start, end
                scores = [float(result.get("score") or 0.0)]
            candidate = _candidate(text, base + current_start, base + current_end, scores)
            if candidate:
                spans.append(candidate)

    unique: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for item in spans:
        key = (int(item["start"]), int(item["end"]), str(item["text"]).casefold())
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return tuple(unique)


def proper_name_candidates(values: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return source-linked proper-noun suggestions for every supplied ledger value."""
    candidates: list[dict[str, Any]] = []
    if enabled():
        _load_pipeline()
    for value_index, raw in enumerate(values or [], start=1):
        for candidate_index, item in enumerate(_tag_text(str(raw or "")), start=1):
            candidates.append({
                "candidate_id": f"V{value_index}C{candidate_index}",
                "value_id": f"V{value_index}",
                **dict(item),
            })
    status = model_status()
    status.update(values_checked=len(values or []), candidates_found=len(candidates))
    status["mode"] = "bertweet_pos_propn" if status["loaded"] else "llm_fallback"
    return candidates, status


__all__ = ["MODEL_ID", "enabled", "model_status", "proper_name_candidates"]
