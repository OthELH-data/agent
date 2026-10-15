"""Per-page text extraction for agreement PDFs: the PDF text layer first, Tesseract OCR as the fallback.

A page whose text layer yields at least 40 characters is read directly (confidence 1.0); any other page
is rendered at ``OCR_DPI`` and OCR'd, recording the mean Tesseract word confidence.
"""
from __future__ import annotations

import io
import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Callable

import pymupdf
import pytesseract
from PIL import Image
from pytesseract import Output

# The corpus is scan-only.  300 DPI materially improves legal headings and clause spelling over the
# faster 220-DPI pass, and ingestion is offline/hash-cached rather than paid on every audit run.
OCR_DPI = 300

_EXPLICIT_PAGE = re.compile(r"\bPage\s+(\d+)\s+of\s+\d+\b", re.IGNORECASE)


def normalize_lines(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u00ad", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    output: list[str] = []
    previous_blank = True
    for line in lines:
        if line:
            output.append(line)
            previous_blank = False
        elif not previous_blank:
            output.append("")
            previous_blank = True
    return "\n".join(output).strip()


def ocr_page(page: Any) -> tuple[str, float, str]:
    """Return reconstructed page text, mean word confidence, and extraction method."""
    text_layer = normalize_lines(page.get_text("text"))
    if len(text_layer) >= 40:
        return text_layer, 1.0, "text"

    scale = OCR_DPI / 72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False, colorspace=pymupdf.csGRAY)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    data = pytesseract.image_to_data(image, config="--psm 3", output_type=Output.DICT)

    paragraphs: OrderedDict[tuple[int, int, int], list[str]] = OrderedDict()
    line_confidences: list[float] = []
    for index, raw_word in enumerate(data.get("text", [])):
        word = str(raw_word or "").strip()
        if not word:
            continue
        key = (
            int(data["block_num"][index]),
            int(data["par_num"][index]),
            int(data["line_num"][index]),
        )
        paragraphs.setdefault(key, []).append(word)
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            confidence = -1.0
        if confidence >= 0:
            line_confidences.append(confidence / 100.0)

    grouped: list[str] = []
    previous_paragraph: tuple[int, int] | None = None
    for (block_num, paragraph_num, _line_num), words in paragraphs.items():
        paragraph_key = (block_num, paragraph_num)
        if previous_paragraph is not None and paragraph_key != previous_paragraph:
            grouped.append("")
        grouped.append(" ".join(words))
        previous_paragraph = paragraph_key
    text = normalize_lines("\n".join(grouped))
    confidence = sum(line_confidences) / len(line_confidences) if line_confidences else 0.0
    return text, round(confidence, 4), "ocr"


def printed_page_label(text: str) -> str:
    matches = _EXPLICIT_PAGE.findall(text)
    return matches[-1] if matches else ""


def extract_pages(
    path: Path, document_id: str, progress: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Read every page of one PDF; return the page records and a count per extraction method."""
    pages: list[dict[str, Any]] = []
    methods: defaultdict[str, int] = defaultdict(int)
    with pymupdf.open(path) as pdf:
        for number, page in enumerate(pdf, 1):
            text, confidence, method = ocr_page(page)
            methods[method] += 1
            pages.append({
                "page": number,
                "printed_page_label": printed_page_label(text),
                "text": text,
                "confidence": confidence,
                "method": method,
            })
            if progress and (number == 1 or number % 10 == 0 or number == len(pdf)):
                progress(f"  {document_id}: page {number}/{len(pdf)}")
    return pages, dict(methods)


__all__ = ["OCR_DPI", "extract_pages", "normalize_lines", "ocr_page", "printed_page_label"]
