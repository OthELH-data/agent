"""Minimal PDF text extraction (PyMuPDF) with an optional OCR fallback.

Standalone copy for the exhibit explainer. OCR is used only if pytesseract + the tesseract
binary happen to be installed; otherwise pages with no text layer come back empty. The audit
reports in scope are text-based, so OCR is rarely needed.
"""
from __future__ import annotations

import io
from pathlib import Path

import fitz  # PyMuPDF

try:
    import pytesseract
    from PIL import Image

    _OCR_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _OCR_AVAILABLE = False

_MIN_TEXT_CHARS = 20
_OCR_ZOOM = 2.0

# Some audit-report PDFs render EVERY inter-word gap as a non-breaking space (U+00A0) and use other
# Unicode space variants. Left as-is, "EXCEPTION<nbsp>DETAILS" won't match the "EXCEPTION DETAILS"
# marker, code lines like "17<nbsp>" fail the whitespace-anchored code regex, and the prose handed to
# the LLM is nbsp-joined. Fold these to a regular space (line/para separators to newline, zero-width
# chars dropped) at the source so every downstream consumer sees clean, space-delimited text.
_WS_NORMALIZE = {
    0x00A0: " ",   # no-break space
    0x2007: " ",   # figure space
    0x202F: " ",   # narrow no-break space
    0x2009: " ",   # thin space
    0x200A: " ",   # hair space
    0x2002: " ",   # en space
    0x2003: " ",   # em space
    0x2028: "\n",  # line separator
    0x2029: "\n",  # paragraph separator
    0x200B: "",    # zero-width space
    0xFEFF: "",    # zero-width no-break space / BOM
}


def _normalize_ws(text: str) -> str:
    """Fold non-breaking / Unicode space variants to plain spaces so text anchors match."""
    return text.translate(_WS_NORMALIZE)


def classify_method(pages: list[dict]) -> str:
    """Classify how a document's text was obtained, from per-page {text, ocr_used} dicts.

    Returns one of: "text" (all content from the PDF text layer), "ocr" (all content OCR'd),
    "mixed" (some pages text, some OCR), or "empty" (no extractable text at all — typically a
    scanned PDF with OCR unavailable).
    """
    with_text = [p for p in pages if (p.get("text") or "").strip()]
    if not with_text:
        return "empty"
    ocr = any(p.get("ocr_used") for p in with_text)
    txt = any(not p.get("ocr_used") for p in with_text)
    if ocr and txt:
        return "mixed"
    return "ocr" if ocr else "text"


def _ocr_page(page) -> str:
    if not _OCR_AVAILABLE:
        return ""
    pix = page.get_pixmap(matrix=fitz.Matrix(_OCR_ZOOM, _OCR_ZOOM))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    try:
        return pytesseract.image_to_string(img).strip()
    except Exception:
        return ""


def extract_pdf(path: str | Path) -> dict:
    """Extract text from a PDF, OCR'ing scanned pages when possible.

    Returns per-page text plus a joined `text` field with `[page N]` markers (the description
    slicer strips those markers). Non-breaking / Unicode space variants are normalized to plain
    spaces so text-anchored parsing (exception markers, code lines) and the LLM prompts are clean.
    """
    path = Path(path)
    doc = fitz.open(path)
    pages: list[dict] = []
    for i, page in enumerate(doc):
        text = _normalize_ws(page.get_text("text")).strip()
        ocr_used = False
        if len(text) < _MIN_TEXT_CHARS and page.get_images(full=True):
            ocr_text = _normalize_ws(_ocr_page(page)).strip()
            if ocr_text:
                text, ocr_used = ocr_text, True
        pages.append({"page": i + 1, "text": text, "ocr_used": ocr_used})
    doc.close()
    full_text = "\n\n".join(f"[page {p['page']}]\n{p['text']}" for p in pages)
    return {"source": str(path), "kind": "pdf", "n_pages": len(pages),
            "pages": pages, "text": full_text, "ocr_available": _OCR_AVAILABLE}
