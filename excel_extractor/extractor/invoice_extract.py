"""Extract vendor-invoice PDFs that accompany an exception's Excel exhibits.

Auditor exhibits frequently include vendor invoices as PDFs sitting in the same exception folder
as the `.xlsx` workpapers (e.g. `H & P Inv 2020023346.pdf` next to `AE01.xlsx`). Some are
born-digital (a real text layer) and some are scanned images (no text — OCR'd via
pdf_extract when tesseract is installed).

Two entry points:
  * extract_invoice(path)  — pull an invoice's text (+ OCR fallback), classify how it was
    obtained, and best-effort parse a few header fields (number, vendor, dates, total).
  * discover_invoices(...) — find sibling invoice PDFs in the exhibit folders, excluding the
    audit report and other non-invoice documents.

Nothing here is invented: parsed fields are only ever lifted verbatim from the extracted text;
when a field can't be found it is left None.
"""
from __future__ import annotations

import re
from pathlib import Path

from extractor.pdf_extract import classify_method, extract_pdf

# Invoices can run several pages; cap the text handed to the LLM so a long attachment can't blow
# the light-tier context. ~12k chars ≈ 3k tokens — plenty once the legal T&C pages are stripped
# (see _is_boilerplate_page), leaving the value-bearing content (coversheet + invoice + milestone /
# proof pages).
_MAX_INVOICE_CHARS = 12000

# --- legal-boilerplate page detection --------------------------------------------------------
# Vendor invoices frequently append 1-2 pages of "STANDARD TERMS & CONDITIONS" legalese (~17k chars
# each) that carries no reconcilable value and, left in, drowns the billing detail. We drop such
# pages before the text is joined/capped. Detection is conservative: a page is boilerplate only when
# it shows clear legal-clause structure AND lacks billing signal, and we never drop every page.
_TC_MARKERS = (
    "terms & conditions", "terms and conditions", "limitation of liability",
    "limitation of remedy", "force majeure", "indemnification", "governing law",
    "consequential damages", "patent infringement", "warranty", "risk of loss",
    "standard product and incidental services",
)
# Numbered legal clauses like "1. PRODUCTS:" / "14. INDEMNIFICATION:" — a dense run of these is the
# signature of a T&C page.
_CLAUSE_RE = re.compile(r"(?m)^\s*\d{1,2}\.\s+[A-Z][A-Z /]{2,}")
# A "$1,234.56"-style money amount; billing pages carry several, T&C pages essentially none.
_DOLLAR_RE = re.compile(r"\$\s?\d")


def _is_boilerplate_page(text: str) -> bool:
    """True when a page is clearly vendor legal T&C (no reconcilable billing content).

    Requires either several distinct T&C markers or a dense run of numbered legal clauses, AND
    very few dollar amounts — so a real billing page that merely cites a clause is never dropped.
    """
    if not text or len(text) < 400:
        return False  # short pages (cover notes, screenshots) are cheap to keep
    low = text.lower()
    markers = sum(1 for m in _TC_MARKERS if m in low)
    clauses = len(_CLAUSE_RE.findall(text))
    dollars = len(_DOLLAR_RE.findall(text))
    return (markers >= 2 or clauses >= 5) and dollars < 4


def _strip_boilerplate(pages: list[dict]) -> tuple[list[dict], int]:
    """Drop legal-T&C pages, but never drop them all (return the original set if nothing survives)."""
    kept = [p for p in pages if not _is_boilerplate_page(p.get("text") or "")]
    if not kept:
        return pages, 0
    return kept, len(pages) - len(kept)

# Auto-discovery deliberately keeps supporting PDFs alongside vendor invoices. POB records,
# helicopter logs, IR support, JIBs, and similar documents can carry material exception evidence.
# Only the audit report itself and rebuttal/correspondence documents are excluded by filename.
_SKIP_RE = re.compile(
    r"audit(?:or)?s?\s*report|external\s*report|final\s*\[|"
    r"rebuttal|response|reply",
    re.I,
)

# --- lightweight header-field parsing --------------------------------------------------------

_NUM_RE = re.compile(
    r"(?im)\b(?:invoice|inv|document)\s*(?:no\.?|number|num|#)?\s*[:#]\s*([A-Z0-9][A-Z0-9\-/]{3,})\b"
    r"|\b(?:invoice|inv)\s*(?:no\.?|number|num|#)\s+([A-Z0-9][A-Z0-9\-/]{3,})\b"
)
# vendor lines usually carry a corporate suffix; matching these avoids grabbing prose/labels.
_VENDOR_RE = re.compile(
    r"(?im)^\s*([A-Z0-9&.,'\- ]{3,60}?\b(?:INC|L\.?L\.?C|CORP(?:ORATION)?|LTD|L\.?P|CO|COMPANY|"
    r"SERVICES|GULF|ENGINEERING|TECHNOLOGIES|OFFSHORE|ENERGY|SOLUTIONS|SYSTEMS)\.?)\s*$"
)
_PAGE_MARKER_RE = re.compile(r"\[page \d+\]")
_PO_RE = re.compile(
    r"(?im)\b(?:p\.?o\.?|purchase\s*order)\s*(?:no\.?|number|#|:)?\s*[:#]?\s*(\d[\d\-]{3,})\b"
)
_DATE_RE = re.compile(
    r"(?im)\b(?:invoice\s*date|date)\s*[:#]?\s*"
    r"(\d{1,2}[/-][A-Za-z0-9]{1,4}[/-]\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})"
)
# money like 1,234.56 or 1234.56, optionally $-prefixed
_MONEY_RE = re.compile(r"\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})|\d+\.\d{2})")
_TOTAL_LINE_RE = re.compile(
    r"(?im)^.*\b(?:total|amount\s*due|invoice\s*total|grand\s*total|net\s*amount)\b.*$"
)


def _to_amount(s: str) -> float | None:
    try:
        return float(s.replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def _parse_fields(text: str) -> dict:
    """Best-effort extraction of invoice header fields from raw text (never invents values)."""
    fields: dict[str, object] = {
        "invoice_number": None, "vendor": None, "invoice_date": None,
        "po_number": None, "currency": None, "total_amount": None,
    }
    if not text:
        return fields
    # Drop the [page N] markers extract_pdf() injects so they can't leak into parsed fields.
    text = _PAGE_MARKER_RE.sub(" ", text)

    # invoice_number: first labelled match whose captured token actually contains a digit
    # (avoids capturing the label word "Number"/"No").
    for m in _NUM_RE.finditer(text):
        cand = (m.group(1) or m.group(2) or "").strip()
        if cand and re.search(r"\d", cand):
            fields["invoice_number"] = cand
            break
    if (m := _PO_RE.search(text)):
        fields["po_number"] = m.group(1).strip()
    if (m := _DATE_RE.search(text)):
        fields["invoice_date"] = m.group(1).strip()

    # Vendor: first line carrying a corporate suffix; left None if none surfaces (never guessed).
    if (m := _VENDOR_RE.search(text)):
        fields["vendor"] = re.sub(r"\s{2,}", " ", m.group(1).strip())[:80]

    if "$" in text or re.search(r"\bUSD\b", text):
        fields["currency"] = "USD"

    # total_amount: prefer a $ figure on a line mentioning a total; else the largest money value.
    total = None
    for line in _TOTAL_LINE_RE.findall(text):
        for mm in _MONEY_RE.findall(line):
            amt = _to_amount(mm)
            if amt is not None:
                total = amt if total is None else max(total, amt)
    if total is None:
        amounts = [a for a in (_to_amount(x) for x in _MONEY_RE.findall(text)) if a is not None]
        total = max(amounts) if amounts else None
    fields["total_amount"] = total
    return fields


def extract_invoice(path: str | Path) -> dict:
    """Extract one invoice PDF's contents, OCR'ing scanned pages when tesseract is available.

    Returns a dict consumed by explain.build_invoice_evidence:
      {source, file_name, kind, n_pages, extraction_method, ocr_pages, ocr_available,
       dropped_boilerplate_pages, parsed, text}
    On a read error, returns the same shape with an `error` key set and empty content.
    """
    path = Path(path)
    try:
        doc = extract_pdf(path)
    except Exception as exc:  # keep the run alive if one invoice is unreadable
        return {"source": str(path), "file_name": path.name, "kind": "invoice_pdf",
                "n_pages": 0, "extraction_method": "empty", "ocr_pages": 0,
                "ocr_available": False, "dropped_boilerplate_pages": 0,
                "parsed": _parse_fields(""), "text": "",
                "error": f"{type(exc).__name__}: {exc}"}

    all_pages = doc.get("pages", [])
    # Drop legal-T&C pages before joining/capping so the budget is spent on billing content.
    pages, dropped = _strip_boilerplate(all_pages)
    method = classify_method(pages)
    ocr_pages = sum(1 for p in pages if p.get("ocr_used"))
    # Rebuild the joined text from the kept pages (mirrors extract_pdf's "[page N]" markers).
    text = "\n\n".join(f"[page {p['page']}]\n{p['text']}" for p in pages)
    truncated = len(text) > _MAX_INVOICE_CHARS
    if truncated:
        text = text[:_MAX_INVOICE_CHARS] + "\n…(invoice text truncated)"

    return {
        "source": str(path),
        "file_name": path.name,
        "kind": "invoice_pdf",
        "n_pages": doc.get("n_pages", len(all_pages)),
        "extraction_method": method,
        "ocr_pages": ocr_pages,
        "ocr_available": bool(doc.get("ocr_available")),
        "dropped_boilerplate_pages": dropped,
        "parsed": _parse_fields(text),
        "text": text,
    }


def discover_invoices(exhibit_paths, report_path=None) -> list[Path]:
    """Find sibling invoice PDFs in the folders that hold the Excel exhibits.

    For each exhibit's parent directory, glob `*.pdf`, drop the audit report and anything whose
    name matches the non-invoice skip pattern, and de-dupe by filename (rebuttal folders often
    duplicate the same invoice). Returns the discovered paths, sorted.
    """
    report_name = Path(report_path).name if report_path else None
    seen: set[str] = set()
    found: list[Path] = []
    dirs = {Path(e).parent for e in exhibit_paths}
    for d in dirs:
        if not d.exists():
            continue
        for pdf in sorted(d.glob("*.pdf")):
            if pdf.name == report_name or _SKIP_RE.search(pdf.name):
                continue
            if pdf.name in seen:
                continue
            seen.add(pdf.name)
            found.append(pdf)
    return found
