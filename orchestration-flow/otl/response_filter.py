"""OTL's one import surface for the operator/auditor name test — plus the list-level guard around it.

The rule itself lives in the extractor (`extractor/response_names.py`) and is NOT restated here: both
tools must exclude exactly the same names, and the extractor is where the corpus evidence for each signal
is documented. This module adds only what OTL needs on top.

WHY OTL NEEDS ITS OWN FILTERING AT ALL — it has two input paths and they inherit differently:

  * The five code-exec specialists get `exception_exhibits`, which `case_resolver` builds by re-scanning
    disk through the extractor's own `find_exception_evidence`. That path already gets FILE-level
    filtering plus the explicit auditor-citation override for free, but nothing filters the TABS inside
    a workbook the agent then opens itself.
  * The spine (intake / category_check / verify_evidence) never touches disk. It
    reads the STAGED synthesis JSON, which was written by an earlier extractor run — so it inherits
    whatever filtering existed the day that artifact was produced, which for the current staging is
    none. The spine filters at read time and honors a current manifest's cited-workbook override
    (`agents/_common.filtered_attachments`).

The extractor filters worksheet OBJECTS (`excel_bounded._usable_worksheets`); OTL filters NAMES — out of
JSON, and out of `wb.sheetnames` inside the sandbox — hence the name-level helper below.
"""
from __future__ import annotations

import sys

from shared.response_names import (AUDITOR_OWNED_RE, QUALIFIED_RE, UNCONDITIONAL_RE,
                                      is_operator_response, is_response_name, is_response_tab)


def drop_responses(names, kind: str = "item", source: str = "", warn=None, key=None) -> tuple[list, list]:
    """Split ``names`` into safe and excluded operator-response items.

    `kind` and `source` only shape the warning text ("workbook"/"tab", and the case or file they came
    from). Items are tested through `key` (default `str`), so a caller holding full PATHS must pass
    `key=lambda p: Path(p).name` — the test is defined on BASENAMES and every corpus cycle directory is
    literally named `Rebuttal N`, so testing a path would exclude the entire corpus.

    When every item is operator-side, return an empty safe list. Re-admitting the items would leak the
    operator's argumentation into auditor-evidence analysis.
    """
    warn = warn or (lambda m: print(m, file=sys.stderr))
    key = key or str
    items = list(names or [])
    dropped = [n for n in items if is_response_name(key(n))]
    if not dropped:
        return items, []
    kept = [n for n in items if not is_response_name(key(n))]
    if not kept:
        warn(f"WARN: every {kind} in {source or 'this set'} looks like an operator response "
             f"({', '.join(key(n) for n in dropped)}) — excluding all of them; no safe auditor-side "
             "evidence remains.")
        return [], dropped
    return kept, dropped


__all__ = ["is_response_name", "is_operator_response", "is_response_tab", "drop_responses",
           "UNCONDITIONAL_RE", "QUALIFIED_RE", "AUDITOR_OWNED_RE"]
