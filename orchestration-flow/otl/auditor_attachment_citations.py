"""Light-tier attachment-citation extraction used only when OTL has no extractor manifest."""
from __future__ import annotations

from shared.auditor_attachment_citations import (
    SYSTEM, user_prompt,
)


NODE = "auditor_attachment_citations"


def extract_auditor_cited_attachment_names(description: str) -> list[str]:
    """Extract auditor-cited filenames from the current exception description only."""
    # Keep the case resolver importable for file-discovery-only callers and tests that do not exercise
    # the no-manifest fallback. The LangChain-backed adapter is needed only at the actual light-tier call.
    from otl import llm_adapter as llm
    from extractor.schemas import AuditorCitedAttachmentNames

    result = llm.structured(NODE, SYSTEM, user_prompt(description), AuditorCitedAttachmentNames)
    return list(result.attachment_names or [])


__all__ = ["NODE", "extract_auditor_cited_attachment_names"]
