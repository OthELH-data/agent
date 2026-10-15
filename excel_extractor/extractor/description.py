"""Slice a single exception's narrative Description out of the auditor report text.

These COPAS-style audit reports render as: a summary table of all exceptions, then one
`EXCEPTION DETAILS` block per exception. Each block carries the venture metadata, a one-row
detail line (code / title / amounts / type), and a narrative `Description:` paragraph. We want
just that paragraph for the target code.

Note: the summary table has a `Description` COLUMN header (no colon); the narrative uses
`Description:` (with colon), so matching the colon form avoids the table header.
"""
from __future__ import annotations

import re

_BLOCK_MARKER = "EXCEPTION DETAILS"
_DESC_LABEL = "Description:"
_SUMMARY_HEADER_RE = re.compile(r"(?i)summary\s+of\s+audit\s+(?:claims|exceptions)")


def _extract_from_summary_table(report_text: str, variants: list[str]) -> str:
    """Parse a one-line description from a summary table (no EXCEPTION DETAILS blocks).

    Summary tables look like:
        8
        Material Reconciliation- SA006 (3.1)
        $TBD

    The code appears on its own line, the next non-blank line(s) are the description, and it ends
    at the next exception code line or a dollar amount on its own line.
    """
    # Only try this if there's a recognizable summary header
    if not _SUMMARY_HEADER_RE.search(report_text):
        return ""
    code_line = _code_line_re(variants)
    for m in code_line.finditer(report_text):
        after = report_text[m.end():]
        # Collect lines until we hit the next exception code or end of section
        lines: list[str] = []
        for raw_line in after.split("\n"):
            stripped = raw_line.strip()
            if not stripped:
                if lines:
                    break
                continue
            # Stop at next exception number (1-4 digit code on its own line)
            if re.match(r"^\d{1,4}$", stripped):
                break
            # Stop at dollar amounts or "TOTAL" markers
            if re.match(r"^\$", stripped) or stripped.upper().startswith("TOTAL"):
                break
            lines.append(stripped)
        if lines:
            desc = " ".join(lines)
            # Strip a trailing dollar amount that PDF sometimes joins onto the same line
            desc = re.sub(r"\s+\$?[\d,]+\.?\d*\s*(\+\s*\$?TBD)?$", "", desc)
            return desc.strip()

# Later-cycle audit reports embed the running rebuttal dialogue per exception, as dated entries
# ("Operator's Reply on April 03, 2023", "Auditor's Reply on June 29, 2023", …). Some PDFs instead
# render the same labels with a colon ("Operator Reply: 08/13/2024"). This marks where that dialogue
# begins — used to (a) trim it off the concise Description and (b) extract it on its own for the
# synthesis and output artifact.
_DIALOGUE_RE = re.compile(
    r"(?i)(?:auditor|operator)(?:'?s)?\s+(?:reply|response)(?:\s+on\s+|:\s*)"
)
_MAX_DIALOGUE_CHARS = 12000


def _collapse(text: str) -> str:
    """Fold the per-visual-line breaks of a PDF paragraph into clean prose."""
    text = re.sub(r"\[page \d+\]", " ", text)  # extract_pdf() page markers
    text = re.sub(r"[ \t]*\r?\n[ \t]*", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _blocks(full: str) -> list[str]:
    """Split the report into per-exception EXCEPTION DETAILS blocks."""
    starts = [m.start() for m in re.finditer(re.escape(_BLOCK_MARKER), full)]
    if not starts:
        return []
    bounds = starts + [len(full)]
    return [full[bounds[i] : bounds[i + 1]] for i in range(len(starts))]


def _code_variants(code: str) -> list[str]:
    """Generate matching variants: as-is, stripped of leading zeros, zero-padded to 2/3/4 digits.

    The 2-digit pad matters: exc ids arrive here as 4-digit ("0002") but COPAS reports print the
    code 2-digit ("02") on its own line. Without regenerating "02", a 4-digit id falls back to the
    bare stripped form ("2"), which can match a stray digit inside a *different* exception's block
    (e.g. "0002" matching a line in exception "01a"). Emitting "02" makes 4-digit and 2-digit inputs
    resolve to the same variant set, so both match the correct block.
    """
    variants = {code}
    stripped = code.lstrip("0") or "0"
    variants.add(stripped)
    variants.add(stripped.zfill(2))
    variants.add(stripped.zfill(3))
    variants.add(stripped.zfill(4))
    return list(variants)


def _code_line_re(variants: list[str]) -> re.Pattern:
    """Match any of `variants` alone on a line — how COPAS reports print an exception's code."""
    return re.compile(rf"(?m)^[ \t]*(?:{'|'.join(re.escape(v) for v in variants)})[ \t]*$")


def report_lists_exception(report_text: str, exception_code: str) -> bool:
    """True if this report has an entry for *exception_code* — its code alone on a line, anywhere.

    Whether the report covers the exception at all, asked without committing to a description. Both
    the summary table and each EXCEPTION DETAILS block print the code on its own line, so one
    own-line match anywhere means the exception is in this report; no match means it is not.

    Callers use this to avoid the loose end of `extract_exception_description`. Its window fallback
    matches the code as a bare SUBSTRING, which in a report that never mentions the exception finds
    an audit number (`10` inside `BP Audit No.: 2022-101`) or another exception's row (`002` inside
    `Material Recon - SA002`) and returns whatever `Description:` paragraph sits nearby — a
    neighbouring exception's text, presented as this one's. Late-cycle letters carry only the
    still-open exceptions, so any run asking about a closed one lands in exactly that state.
    """
    return bool(_code_line_re(_code_variants(exception_code.strip())).search(report_text))


def extract_exception_description(report_text: str, exception_code: str,
                                  title: str = "") -> str:
    """Return the narrative Description paragraph for *exception_code*.

    Primary path: find the EXCEPTION DETAILS block, locate the exception's code on its own
    line (trying zero-padded variants), then take the text after the nearest `Description:`
    label. Handles both one-block-per-exception and all-in-one-block report formats.
    """
    code = exception_code.strip()
    variants = _code_variants(code)
    code_line = _code_line_re(variants)

    for block in _blocks(report_text):
        if _DESC_LABEL not in block:
            continue
        # Find all positions where the code appears on its own line within this block
        matches = list(code_line.finditer(block))
        if not matches:
            continue

        # The report may have the code in a summary table AND later in the detail section.
        # We want the occurrence that is immediately followed by a Description: label.
        for match in matches:
            after_code = block[match.end():]
            # Find the next Description: after this code occurrence
            desc_pos = after_code.find(_DESC_LABEL)
            if desc_pos == -1:
                continue
            # Make sure there isn't another exception code line between this match and
            # the Description: (which would mean this Description: belongs to a different one)
            between = after_code[:desc_pos]
            if code_line.search(between):
                continue
            tail = after_code[desc_pos + len(_DESC_LABEL):]
            # Stop at the next exception's code line or a next-section label
            next_exc = code_line.search(tail)
            # Also look for any 3-4 digit code on its own line (next exception)
            next_any_code = re.search(r"(?m)^[ \t]*\d{3,4}[ \t]*$", tail)
            # Find the nearest Description: that belongs to the NEXT exception
            next_desc = tail.find(_DESC_LABEL)
            # Cut at the earliest next-exception boundary
            cuts = [len(tail)]
            if next_any_code:
                cuts.append(next_any_code.start())
            if next_desc > 0:
                cuts.append(next_desc)
            tail = tail[:min(cuts)]
            tail = re.split(r"(?m)^[ \t]*(?:ID|Venture Property:)\s*$", tail)[0]
            desc = _collapse(tail)
            desc = _DIALOGUE_RE.split(desc, 1)[0].strip()  # keep the concise description only
            if desc:
                return desc

    # Fallback: window around the code near a Description: label.
    for variant in variants:
        for m in re.finditer(re.escape(variant), report_text):
            window = report_text[m.start(): m.start() + 2000]
            if _DESC_LABEL in window:
                tail = window.split(_DESC_LABEL, 1)[1]
                tail = re.split(re.escape(_BLOCK_MARKER), tail)[0]
                # Cut at next exception code
                next_code = re.search(r"(?m)^[ \t]*\d{3,4}[ \t]*$", tail)
                if next_code:
                    tail = tail[:next_code.start()]
                desc = _collapse(tail)
                desc = _DIALOGUE_RE.split(desc, 1)[0].strip()
                if len(desc) > 40:
                    return desc

    # Summary-only format: some reports (e.g. 2021 Silvertip) have no EXCEPTION DETAILS blocks —
    # just a cover letter + a summary table with "Number / Description / Amount" rows. Extract the
    # one-line description from that table (code on its own line, followed by text, then $amount).
    summary_desc = _extract_from_summary_table(report_text, variants)
    if summary_desc:
        return summary_desc

    # Last resort: return the title if provided — the actual description may be inside the Excel
    # exhibits rather than the PDF report. Let the pipeline proceed; the LLM will derive context
    # from the workbook tabs.
    if title and title.strip():
        return title.strip()

    raise ValueError(f"Could not locate a Description for exception {code!r} in the report text")


def extract_rebuttal_dialogue(report_text: str, exception_code: str, title: str = "") -> str:
    """Return the operator/auditor rebuttal dialogue for *exception_code*, from the audit report.

    Later-cycle reports are running documents that embed the full back-and-forth per exception
    (dated 'Operator's Reply on …' / 'Auditor's Reply on …' entries). This returns that chronology
    so the synthesis can consider the operator's prior responses without re-reading the operator
    folder. Returns '' for an initial issuance that carries no dialogue yet.
    """
    code = exception_code.strip()
    variants = _code_variants(code)
    code_line = _code_line_re(variants)

    def _dialogue_from(region: str) -> str:
        collapsed = _collapse(region)
        m = _DIALOGUE_RE.search(collapsed)
        return collapsed[m.start():][:_MAX_DIALOGUE_CHARS] if m else ""

    # Primary: within the exception's EXCEPTION DETAILS block, from its code line to the next code.
    for block in _blocks(report_text):
        if _DESC_LABEL not in block:
            continue
        m = code_line.search(block)
        if not m:
            continue
        region = block[m.end():]
        nxt = re.search(r"(?m)^[ \t]*\d{3,4}[ \t]*$", region)
        if nxt:
            region = region[:nxt.start()]
        dialogue = _dialogue_from(region)
        if dialogue:
            return dialogue

    # Fallback: a window after the code occurrence, bounded by the next exception code.
    for variant in variants:
        for m in re.finditer(re.escape(variant), report_text):
            window = report_text[m.start(): m.start() + 15000]
            nxt = re.search(r"(?m)^[ \t]*\d{3,4}[ \t]*$", window[len(variant):])
            if nxt:
                window = window[:len(variant) + nxt.start()]
            dialogue = _dialogue_from(window)
            if dialogue:
                return dialogue
    return ""


def split_rebuttal_dialogue(dialogue: str) -> list[dict[str, str]]:
    """Split extracted dialogue into its ordered operator and auditor entries.

    The report's PDF text frequently collapses visual lines, so entries are bounded by the next
    reply/response header rather than by a newline. The date remains verbatim at the beginning of
    ``reply`` because reports use several incompatible date formats.
    """
    matches = list(_DIALOGUE_RE.finditer(dialogue or ""))
    entries: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        reply_end = matches[index + 1].start() if index + 1 < len(matches) else len(dialogue)
        label = match.group(0).strip()
        party = "auditor" if label.casefold().startswith("auditor") else "operator"
        reply = dialogue[match.end():reply_end].strip()
        if reply:
            entries.append({"party": party, "reply": reply})
    return entries
