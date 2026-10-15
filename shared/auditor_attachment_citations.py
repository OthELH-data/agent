"""Schema and prompt for attachment names explicitly cited in an auditor's description.

The caller supplies *only* the description.  Keeping candidate filenames out of the prompt means the
light-tier model extracts what the auditor actually named rather than selecting from a discovered list.
Path resolution and ownership filtering remain deterministic in :mod:`shared.rebuttals`.
"""
from __future__ import annotations

SYSTEM = """\
You extract attachment filenames explicitly cited by the auditor in one exception description.

Return only filenames the description itself names. Preserve the printed filename and extension. Do not
infer a filename from a document type, project, ledger, invoice number, or surrounding context. Do not
use any outside attachment list, and return an empty list when the description names no attachment.
"""


def user_prompt(description: str) -> str:
    """Build the deliberately description-only prompt for the light-tier extraction pass."""
    return "AUDITOR EXCEPTION DESCRIPTION:\n" + str(description or "").strip()


__all__ = ["SYSTEM", "user_prompt"]
