"""One predicate: does this NAME belong to the operator's reply, or to the auditor's own work?

The pipeline mines the AUDITOR side only. Uncited operator argumentation must not reach the LLM. But the
corpus files the two sides together routinely, and it names them inconsistently, so the ordinary
separation is made by name at two levels: whole workbooks (`rebuttals.drop_operator_responses`) and
individual tabs inside an otherwise auditor-owned workbook (`excel_bounded._usable_worksheets`). Both
levels use the SAME vocabulary, so both call the one predicate here rather than carrying near-duplicate
patterns that drift apart. A workbook explicitly named in the auditor's exception description is a
deliberate exception at the evidence-union layer; this predicate still reports what its filename implies.

`orchestration-flow` imports this module too (`otl/response_filter.py`), which is why it deliberately has
no intra-package imports: `rebuttals` would drag in `invoice_extract -> pdf_extract`, and the OTL code
sandbox needs the patterns without the PDF dependency chain.

THE ONE SUBTLETY — "reply" and "rebuttal" cut both ways. A rebuttal cycle has two sides writing about
each other, so those words alone don't say whose document this is: `AE 007 JADE and support - Auditor
Reply.xlsx` is the AUDITOR's answer (parse it), while `Operator's Reply` is the operator's (drop it).
Names that attribute the reply/comments to the auditor are therefore auditor-side and must be parsed —
and a workbook carrying BOTH signals ("Operator response … with auditor comments") is still the
operator's, because the operator's own signal is unconditional. Hence two tiers, not one pattern:

  * UNCONDITIONAL signals settle ownership on their own and are tested against the raw name.
  * QUALIFIED signals are tested only AFTER auditor-owned phrases are stripped out.

The tiering is not cosmetic. A single strip-then-test collapses on
`AE 003 Support and Analysis to auditor - Rebuttal 2.xlsx`: stripping `auditor - Rebuttal` also destroys
the `to auditor` signal, and the operator's returned package silently becomes auditor evidence.

Matching is always on the BASENAME (or the bare tab title), never a path — every cycle directory in the
corpus is literally named `Rebuttal N`, so matching a path would exclude the entire corpus.

The three patterns are public because OTL's code sandbox has to enforce the same rule inside a child
process that CANNOT import this module (it runs with the scratch dir as cwd), so it injects these pattern
strings and recompiles them — see `otl/code_exec.py`. Injecting them verbatim is what keeps that copy from
becoming a fourth definition of the rule.
"""
from __future__ import annotations

import re

# Word edges, hand-rolled instead of `\b`: `_` is a word character to `re`, so `\b` does NOT see an edge
# in `Back_to_Auditor` or `IR14_Resp` — and the corpus separates these words with underscores as readily
# as with spaces or hyphens. `_S`/`_E` treat every non-alphanumeric as a separator, which is also why
# "correspondence" and "responsible" still cannot match (the edge is checked, just not via `\b`).
_S = r"(?<![A-Za-z0-9])"   # start of a word
_E = r"(?![A-Za-z0-9])"    # end of a word
_GAP = r"[\W_]*"           # any run of separators between two words

# Signals that identify the operator outright, whoever else the name mentions.
#   * `resp` / `response` / `responses` — "…EXP 86 - Response", "…-Jai Resp rev", "IR010 OP Response".
#   * `to auditor(s)` — the operator's returned package: "back to auditor", "Back_to_Auditor".
#   * `operator('s) comments` — the operator's annotations: "EX-04 workings and operator comments".
UNCONDITIONAL_RE = re.compile(
    rf"""{_S}resp(?:onses?)?{_E}                    # Resp / Response / Responses (whole word)
       | {_S}to{_GAP}auditors?{_E}                  # (back) to auditor — the returned package
       | {_S}operator'?s?{_GAP}comments?{_E}        # operator comments / operator's comment
       | {_S}shell{_GAP}findings?{_E}               # Shell Finding(s) workbook/material
    """,
    re.I | re.X,
)

# Signals that only identify the operator when the name doesn't hand them to the auditor.
QUALIFIED_RE = re.compile(
    rf"""{_S}repl(?:y|ies){_E}      # "Operator's Reply"  vs  "Auditor Reply"
       | {_S}rebuttals?{_E}         # "Op Rebuttal"       vs  "Auditor's Rebuttal"
    """,
    re.I | re.X,
)

# Phrases attributing a reply/comment to the AUDITOR. Removed before the qualified test so the auditor's
# own answer, and a response the auditor then annotated, are not mistaken for the operator's.
AUDITOR_OWNED_RE = re.compile(
    rf"{_S}auditor'?s?{_GAP}(?:repl(?:y|ies)|rebuttals?|comments?){_E}", re.I)

# Deliberately NOT a signal: "JL FINAL". Those initials mark a reviewer's final copy, not a response, and
# 5 workbooks in the corpus are genuine auditor evidence named only that (e.g. "IR 03 Affiliate Cost
# Certificate JL FINAL.xlsx"). Every actual response file carrying "JL FINAL" also carries a
# `resp`/`to auditor` signal, so it is already covered.


def is_response_name(name: str) -> bool:
    """True if this workbook filename or tab title marks the content as the OPERATOR's, not the auditor's.

    >>> is_response_name("Operator response to EXC 41 with auditor comments.xlsx")   # both -> operator
    True
    >>> is_response_name("EXC 41 with auditor comments.xlsx")                        # auditor-side
    False
    >>> is_response_name("AE 007 JADE and support - Auditor Reply.xlsx")             # auditor's answer
    False
    >>> is_response_name("AE 003 Support and Analysis to auditor - Rebuttal 2.xlsx")  # returned package
    True
    """
    n = name or ""
    return bool(UNCONDITIONAL_RE.search(n)) or bool(
        QUALIFIED_RE.search(AUDITOR_OWNED_RE.sub(" ", n)))


# Two names for one test, so each call site reads in its own terms.
is_operator_response = is_response_name  # workbook filenames  (rebuttals.py)
is_response_tab = is_response_name       # sheet titles        (excel_bounded.py)

__all__ = ["is_response_name", "is_operator_response", "is_response_tab",
           "UNCONDITIONAL_RE", "QUALIFIED_RE", "AUDITOR_OWNED_RE"]
