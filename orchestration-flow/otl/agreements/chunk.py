"""Legal-aware chunking of extracted agreement pages into ``ChunkRecord`` rows.

Pages are split into paragraph units (heading-like lines become their own units), packed into chunks of
about ``TARGET_CHUNK_TOKENS`` (hard cap ``MAX_CHUNK_TOKENS``) with ~``OVERLAP_TOKENS`` of overlap, and a
major legal heading always starts a clean chunk. Chunk IDs hash the document ID, page range and text, so
unchanged text keeps its ID (and its cached embedding) across builds.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from .corpus import AgreementDocumentSpec
from .record import ChunkRecord
from .tokens import token_count


TARGET_CHUNK_TOKENS = 750
MAX_CHUNK_TOKENS = 1100
OVERLAP_TOKENS = 120
QUOTE_SAFE_CONFIDENCE = 0.80

_HEADING = re.compile(
    r"^(?:ARTICLE|SECTION|EXHIBIT|ATTACHMENT|SCHEDULE|APPENDIX|PART)\b|"
    r"^(?:\d+(?:\.\d+){0,4}|[A-Z])(?:[.)]|\s+-)?\s+[A-Z]",
    re.IGNORECASE,
)
_MAJOR_BOUNDARY = re.compile(
    r"^(?:ARTICLE|SECTION|EXHIBIT|ATTACHMENT|SCHEDULE|APPENDIX|PART)\b", re.IGNORECASE
)


@dataclass
class _Segment:
    text: str
    page: int
    printed_page_label: str
    heading_path: str
    confidence: float
    method: str
    break_before: bool = False


def _looks_like_heading(paragraph: str) -> bool:
    first_line = paragraph.splitlines()[0].strip()
    if not first_line or len(first_line) > 180:
        return False
    if _HEADING.search(first_line):
        return True
    letters = [char for char in first_line if char.isalpha()]
    return bool(letters) and len(first_line) <= 100 and sum(c.isupper() for c in letters) / len(letters) >= 0.75


def _paragraph_units(text: str) -> list[str]:
    """Recover clause units even when OCR emits a page with no blank paragraph separators."""
    units: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            units.append("\n".join(current).strip())
            current.clear()

    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if _looks_like_heading(line):
            flush()
            units.append(line)
            continue
        current.append(line)
    flush()
    return [unit for unit in units if unit]


def _split_words(text: str, maximum: int) -> list[str]:
    words = text.split()
    if token_count(text) <= maximum:
        return [text]
    output: list[str] = []
    current: list[str] = []
    for word in words:
        proposal = " ".join([*current, word])
        if current and token_count(proposal) > maximum:
            output.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        output.append(" ".join(current))
    return output


def _tail_text(text: str, maximum: int) -> str:
    words = text.split()
    selected: list[str] = []
    for word in reversed(words):
        proposal_words = [word, *selected]
        proposal = " ".join(proposal_words)
        if selected and token_count(proposal) > maximum:
            break
        selected = proposal_words
    return " ".join(selected)


def _overlap_segments(segments: list[_Segment]) -> list[_Segment]:
    output: list[_Segment] = []
    used = 0
    for segment in reversed(segments):
        remaining = OVERLAP_TOKENS - used
        if remaining <= 0:
            break
        count = token_count(segment.text)
        if count <= remaining + 40:
            output.insert(0, segment)
            used += count
        else:
            tail = _tail_text(segment.text, remaining)
            if tail:
                output.insert(0, _Segment(
                    text=tail,
                    page=segment.page,
                    printed_page_label=segment.printed_page_label,
                    heading_path=segment.heading_path,
                    confidence=segment.confidence,
                    method=segment.method,
                    break_before=False,
                ))
            break
    return output


def _build_chunk(document: AgreementDocumentSpec, segments: list[_Segment]) -> ChunkRecord:
    text = "\n\n".join(segment.text for segment in segments).strip()
    pages_seen = [segment.page for segment in segments]
    printed = list(dict.fromkeys(
        segment.printed_page_label for segment in segments if segment.printed_page_label
    ))
    headings = list(dict.fromkeys(segment.heading_path for segment in segments if segment.heading_path))
    methods = list(dict.fromkeys(segment.method for segment in segments if segment.method))
    confidence = sum(segment.confidence for segment in segments) / len(segments)
    identity = "|".join((document.document_id, str(min(pages_seen)), str(max(pages_seen)), text))
    return ChunkRecord(
        chunk_id=f"agr-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:20]}",
        document_id=document.document_id,
        document_name=document.document_name,
        document_type=document.document_type,
        legal_track=document.legal_track,
        asset_scopes=list(document.asset_scopes),
        party_scope=list(document.party_scope),
        heading_path=" > ".join(headings[-3:]),
        pdf_page_start=min(pages_seen),
        pdf_page_end=max(pages_seen),
        printed_page_labels=printed,
        extraction_methods=methods,
        verbatim_text=text,
        ocr_confidence=round(confidence, 4),
        quote_safe=confidence >= QUOTE_SAFE_CONFIDENCE,
        content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        token_count=token_count(text),
    )


def chunk_document(document: AgreementDocumentSpec, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[_Segment] = []
    current_heading = ""
    for page in pages:
        text = str(page.get("text") or "").strip()
        for paragraph in _paragraph_units(text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            is_heading = _looks_like_heading(paragraph)
            if is_heading:
                current_heading = paragraph.splitlines()[0][:240]
            for piece_index, piece in enumerate(_split_words(paragraph, MAX_CHUNK_TOKENS)):
                segments.append(_Segment(
                    text=piece,
                    page=int(page["page"]),
                    printed_page_label=str(page.get("printed_page_label") or ""),
                    heading_path=current_heading,
                    confidence=float(page.get("confidence") or 0.0),
                    method=str(page.get("method") or "unknown"),
                    break_before=is_heading and piece_index == 0,
                ))

    chunks: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()
    current: list[_Segment] = []
    current_tokens = 0
    new_content_since_emit = False

    def emit(*, keep_overlap: bool = True) -> None:
        nonlocal current, current_tokens, new_content_since_emit
        text = "\n\n".join(segment.text for segment in current).strip()
        if not text:
            current, current_tokens = [], 0
            return
        record = _build_chunk(document, current)
        if record.chunk_id not in seen_chunk_ids:
            chunks.append(record.model_dump())
            seen_chunk_ids.add(record.chunk_id)
        current = _overlap_segments(current) if keep_overlap else []
        current_tokens = sum(token_count(segment.text) for segment in current)
        new_content_since_emit = False

    for segment in segments:
        segment_tokens = token_count(segment.text)
        if (
            segment.break_before
            and current
            and new_content_since_emit
            and (current_tokens >= 180 or _MAJOR_BOUNDARY.search(segment.text.strip()))
        ):
            # A major legal heading starts a new citation unit.  Do not carry text from a signature
            # page or preceding clause into the new section merely to satisfy sliding-window overlap.
            emit(keep_overlap=False)
        if current and current_tokens + segment_tokens > MAX_CHUNK_TOKENS:
            emit()
        current.append(segment)
        current_tokens += segment_tokens
        new_content_since_emit = True
        if current_tokens >= TARGET_CHUNK_TOKENS:
            emit()
    if current and new_content_since_emit:
        # Final tail: unlike ``emit`` there is no reason to retain an overlap afterward.
        text = "\n\n".join(segment.text for segment in current).strip()
        if text:
            record = _build_chunk(document, current)
            duplicate_tail = bool(chunks) and record.content_sha256 == chunks[-1]["content_sha256"]
            if record.chunk_id not in seen_chunk_ids and not duplicate_tail:
                chunks.append(record.model_dump())
                seen_chunk_ids.add(record.chunk_id)
    return chunks


__all__ = [
    "MAX_CHUNK_TOKENS", "OVERLAP_TOKENS", "QUOTE_SAFE_CONFIDENCE", "TARGET_CHUNK_TOKENS",
    "chunk_document", "token_count",
]
