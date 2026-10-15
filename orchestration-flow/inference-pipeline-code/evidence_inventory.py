"""Deterministic inventory and coverage helpers for case evidence.

The reasoning agents should never have to infer whether an auditor artifact was
attached from a truncated workbook summary.  This module keeps a compact,
first-class inventory in ``AuditState.collected_evidence`` and relates the
planner's evidence needs to artifacts already in the record.

The inventory records *availability*, not truth.  An attached invoice can still
be inadequate or contradictory; that sufficiency decision belongs to evidence
synthesis.  The distinction prevents "attached but unpersuasive" from being
silently rewritten as "not provided".
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable


_SPACE_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^a-z0-9]+")
_SPREADSHEET_KINDS = {"xlsx", "xlsm", "xlsb", "xls", "csv"}
_STATUS_RANK = {"listed": 0, "attached": 1, "extracted": 2, "analyzed": 3}


def _clean(value: Any, limit: int = 1200) -> str:
    text = _SPACE_RE.sub(" ", str(value or "")).strip()
    return text[:limit]


def _dedupe(values: Iterable[Any], *, limit: int = 24, item_limit: int = 800) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean(value, item_limit)
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        output.append(text)
        if len(output) >= limit:
            break
    return output


def _normalized_name(value: Any) -> str:
    return _NON_WORD_RE.sub(" ", Path(str(value or "unnamed")).name.lower()).strip()


def _artifact_type(attachment: dict[str, Any]) -> str:
    kind = str(attachment.get("kind") or "").lower().lstrip(".")
    if not kind:
        kind = Path(str(attachment.get("name") or attachment.get("path") or "")).suffix.lower().lstrip(".")
    if kind in _SPREADSHEET_KINDS:
        return "spreadsheet"
    if kind in {"pdf", "doc", "docx", "txt", "rtf", "msg", "eml"}:
        return "document"
    if kind in {"png", "jpg", "jpeg", "tif", "tiff", "bmp"}:
        return "image"
    return kind or "artifact"


def evidence_id(source_party: str, artifact_name: str) -> str:
    """Return a stable, path-independent artifact identifier."""
    identity = f"{source_party.casefold()}|{_normalized_name(artifact_name)}"
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
    return f"evidence-{digest}"


def inventory_from_case(case: dict[str, Any]) -> list[dict[str, Any]]:
    """List artifacts supplied with the case before any model is called."""
    items: list[dict[str, Any]] = []
    name_counts: dict[tuple[str, str], int] = {}
    for index, attachment in enumerate(case.get("attachments", []) or []):
        if not isinstance(attachment, dict):
            continue
        source_party = _clean(attachment.get("source") or "auditor", 40).lower()
        name = _clean(
            attachment.get("name") or Path(str(attachment.get("path") or "")).name
            or f"attachment-{index + 1}",
            240,
        )
        key = (source_party, _normalized_name(name))
        occurrence = name_counts.get(key, 0)
        name_counts[key] = occurrence + 1
        stable_name = name if occurrence == 0 else f"{name} #{occurrence + 1}"
        extracted_text = _clean(attachment.get("text"), 900)
        items.append({
            "evidence_id": evidence_id(source_party, stable_name),
            "source_party": source_party,
            "artifact_name": name,
            "artifact_type": _artifact_type(attachment),
            "status": "extracted" if extracted_text else "attached",
            "description": extracted_text or f"{_artifact_type(attachment).title()} attached to the case.",
            "supports": [],
            "locations": [],
        })
    return items


def merge_collected_evidence(
    existing: Iterable[dict[str, Any]] | None,
    additions: Iterable[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Merge inventory enrichments without duplicating the same artifact."""
    merged: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    by_name: dict[tuple[str, str], dict[str, Any]] = {}

    for raw in list(existing or []) + list(additions or []):
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        source = _clean(item.get("source_party") or "auditor", 40).lower()
        name = _clean(item.get("artifact_name") or "unnamed artifact", 240)
        has_explicit_id = bool(item.get("evidence_id"))
        item_id = str(item.get("evidence_id") or evidence_id(source, name))
        name_key = (source, _normalized_name(name))
        current = by_id.get(item_id)
        if current is None and not has_explicit_id:
            current = by_name.get(name_key)
        if current is None:
            current = {
                "evidence_id": item_id,
                "source_party": source,
                "artifact_name": name,
                "artifact_type": _clean(item.get("artifact_type") or "artifact", 60),
                "status": _clean(item.get("status") or "listed", 40),
                "description": _clean(item.get("description"), 1200),
                "supports": _dedupe(item.get("supports", []) or []),
                "locations": _dedupe(item.get("locations", []) or []),
                "provenance_role": _clean(
                    item.get("provenance_role") or "source_fact", 60
                ),
            }
            merged.append(current)
            by_id[item_id] = current
            by_name.setdefault(name_key, current)
            continue

        incoming_description = _clean(item.get("description"), 1200)
        if len(incoming_description) > len(str(current.get("description") or "")):
            current["description"] = incoming_description
        current["supports"] = _dedupe(
            list(current.get("supports", []) or []) + list(item.get("supports", []) or [])
        )
        current["locations"] = _dedupe(
            list(current.get("locations", []) or []) + list(item.get("locations", []) or [])
        )
        incoming_status = _clean(item.get("status") or "listed", 40)
        if _STATUS_RANK.get(incoming_status, 0) > _STATUS_RANK.get(str(current.get("status")), 0):
            current["status"] = incoming_status
        if current.get("artifact_type") == "artifact" and item.get("artifact_type"):
            current["artifact_type"] = _clean(item["artifact_type"], 60)
        if (
            current.get("provenance_role") in {None, "", "source_fact"}
            and item.get("provenance_role")
        ):
            current["provenance_role"] = _clean(item["provenance_role"], 60)
    return merged


def state2_evidence_item(attachment: dict[str, Any], state2: dict[str, Any]) -> dict[str, Any]:
    """Convert the structural Excel synthesis into an inventory enrichment."""
    source = _clean(attachment.get("source") or "auditor", 40).lower()
    name = _clean(attachment.get("name") or Path(str(attachment.get("path") or "workbook.xlsx")).name, 240)
    return {
        "evidence_id": evidence_id(source, name),
        "source_party": source,
        "artifact_name": name,
        "artifact_type": "spreadsheet",
        "status": "analyzed",
        "description": _clean(state2.get("calculation_summary"), 1200),
        "supports": _dedupe([state2.get("calculation_reasoning")]),
        "locations": _dedupe(state2.get("keywords", []) or []),
    }


def finding_evidence_items(
    findings: Iterable[dict[str, Any]],
    *,
    default_source: str = "auditor",
) -> list[dict[str, Any]]:
    """Convert directed workbook findings into inventory enrichments."""
    items: list[dict[str, Any]] = []
    for finding in findings or []:
        if not isinstance(finding, dict):
            continue
        name = _clean(finding.get("file") or "auditor workbook", 240)
        source = default_source.lower()
        proves = finding.get("proves")
        proves_values = proves if isinstance(proves, list) else [proves]
        row_locations = []
        for row in finding.get("evidence_rows", []) or []:
            if isinstance(row, dict):
                row_locations.append(row.get("why") or row.get("location"))
        items.append({
            "evidence_id": evidence_id(source, name),
            "source_party": source,
            "artifact_name": name,
            "artifact_type": "spreadsheet",
            "status": "analyzed",
            "description": _clean(finding.get("reasoning") or proves, 1200),
            "supports": _dedupe(proves_values),
            "locations": _dedupe(
                list(finding.get("supporting_fields", []) or []) + row_locations
            ),
        })
    return items


def compact_inventory(items: Iterable[dict[str, Any]] | None, limit: int = 20) -> list[dict[str, Any]]:
    """Return the prompt-safe portion of the evidence inventory."""
    compact: list[dict[str, Any]] = []
    for item in list(items or [])[:limit]:
        if not isinstance(item, dict):
            continue
        compact.append({
            "evidence_id": item.get("evidence_id"),
            "source_party": item.get("source_party"),
            "artifact_name": item.get("artifact_name"),
            "artifact_type": item.get("artifact_type"),
            "status": item.get("status"),
            "description": _clean(item.get("description"), 700),
            "supports": _dedupe(item.get("supports", []) or [], limit=8, item_limit=500),
            "locations": _dedupe(item.get("locations", []) or [], limit=12, item_limit=200),
            "provenance_role": item.get("provenance_role", "source_fact"),
        })
    return compact


def _item_search_text(item: dict[str, Any]) -> str:
    return " ".join(
        [
            str(item.get("artifact_name") or ""),
            str(item.get("artifact_type") or ""),
            str(item.get("description") or ""),
            " ".join(str(value) for value in item.get("supports", []) or []),
            " ".join(str(value) for value in item.get("locations", []) or []),
        ]
    ).lower()


def find_evidence_request_conflicts(
    requests: Iterable[Any] | None,
    inventory: Iterable[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Flag generic requests for artifact types that are already attached.

    A conflict does not prove the attached artifact is sufficient.  It means the
    request must identify the missing field, period, signature, or tie-out rather
    than incorrectly saying the entire artifact was not provided.
    """
    items = list(inventory or [])
    searchable = [(item, _item_search_text(item)) for item in items]
    cues = {
        "invoice": ("invoice", "vendor invoice"),
        "workbook": ("workbook", "excel", "spreadsheet", "workpaper"),
        "iwells": ("iwells", "iwell", "installation record"),
        "cja": ("cji3",),
        "jade": ("jade", "ledger extract"),
        "bom": ("bill of materials", "bom"),
        "timesheet": ("timesheet", "time sheet", "timecard"),
        "manifest": ("manifest",),
        "certificate": ("certificate", "attestation"),
        "comments": ("auditor comments", "ir workbook", "information request"),
    }
    specificity_terms = (
        "field", "column", "row", "line", "period", "date", "signature",
        "approval", "tie-out", "tie out", "reconcile", "page", "amount",
        "rate", "basis", "status", "serial", "quantity", "section",
    )
    conflicts: list[dict[str, Any]] = []
    for raw_request in requests or []:
        request = _clean(raw_request, 1000)
        request_lower = request.lower()
        if not request:
            continue
        # A precise request for missing content within an attached artifact is
        # valid.  Only quarantine generic re-requests for the artifact itself.
        if any(term in request_lower for term in specificity_terms):
            continue
        matched_ids: list[str] = []
        matched_cues: list[str] = []
        for label, aliases in cues.items():
            if not any(alias in request_lower for alias in aliases):
                continue
            for item, item_text in searchable:
                artifact_match = (
                    item.get("artifact_type") == "spreadsheet" and label == "workbook"
                ) or any(alias in item_text for alias in aliases)
                if artifact_match:
                    matched_ids.append(str(item.get("evidence_id")))
                    matched_cues.append(label)
        if matched_ids:
            conflicts.append({
                "request": request,
                "evidence_ids": _dedupe(matched_ids, limit=12, item_limit=80),
                "matched_artifact_types": _dedupe(matched_cues, limit=12, item_limit=80),
                "reason": (
                    "A potentially responsive artifact is already attached; identify the "
                    "specific missing content or failed tie-out instead of requesting the artifact again."
                ),
            })
    return conflicts


def structure_evidence_requests(
    requests: Iterable[Any] | None,
    *,
    blocking: bool = False,
) -> list[dict[str, Any]]:
    """Make request ownership and the precise missing content machine-readable."""
    artifacts = (
        ("iwells", ("iwells", "iwell", "installation record")),
        ("ledger_extract", ("cji3", "jade", "ledger", "sap")),
        ("reconciliation_workbook", ("workbook", "spreadsheet", "reconciliation")),
        ("invoice_or_po", ("invoice", "purchase order", "po chain")),
        ("return_or_transfer_record", ("manifest", "goods return", "transfer")),
        ("activity_support", ("timesheet", "work order", "timewriting", "project document")),
        ("agreement_support", ("agreement", "contract", "clause")),
    )
    details: list[dict[str, Any]] = []
    for raw in requests or []:
        request = _clean(raw, 1000)
        if not request:
            continue
        lower = request.lower()
        if "auditor" in lower:
            owner = "auditor"
        elif any(term in lower for term in ("operator", "shell", "internal")):
            owner = "operator"
        else:
            owner = "case_owner"
        artifact = next(
            (
                label for label, aliases in artifacts
                if any(alias in lower for alias in aliases)
            ),
            "case_specific_support",
        )
        details.append({
            "request": request,
            "owner": owner,
            "artifact_or_system": artifact,
            "missing_content": request,
            "blocking": bool(blocking),
        })
    return details


def _matching_inventory(need: dict[str, Any], inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source = str(need.get("source_system") or "").lower()
    lookup = need.get("lookup_key") or {}
    text_by_item = [(item, _item_search_text(item)) for item in inventory]
    if source == "workbook":
        return [item for item, _ in text_by_item if item.get("artifact_type") == "spreadsheet"]
    if source == "materials":
        cues = ("material", "iwells", "iwell", "cji3", "bom", "octg", "recon", "casing")
        return [item for item, text in text_by_item
                if item.get("artifact_type") == "spreadsheet" and any(cue in text for cue in cues)]
    if source == "jade":
        cues = ("jade", "cji3", "ledger", "sap", "line item", "line-item")
        return [item for item, text in text_by_item if any(cue in text for cue in cues)]
    if source == "external":
        document = str(lookup.get("document") or "").replace("_", " ").lower()
        cues = [part for part in document.split() if len(part) >= 4]
        if not cues:
            need_text = str(need.get("need") or "").lower()
            known = ("invoice", "certificate", "attestation", "timesheet", "manifest", "contract", "white paper")
            cues = [cue for cue in known if cue in need_text]
        return [item for item, text in text_by_item if cues and any(cue in text for cue in cues)]
    return []


def assess_evidence_coverage(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Relate planned proof needs to evidence already available in state.

    ``available_for_review`` intentionally does not mean ``satisfied``.  It tells
    synthesis that the artifact exists and must be evaluated before requesting it.
    """
    plan = state.get("evidence_plan", {}) or {}
    inventory = list(state.get("collected_evidence", []) or [])
    try:
        from graph import state_helpers as state_helpers
        selected = state_helpers.get_selected(state)
    except Exception:
        selected = []
    verified_authority = [section for section in selected if section.get("verified")]

    coverage: list[dict[str, Any]] = []
    for index, need in enumerate(plan.get("needs", []) or []):
        if not isinstance(need, dict):
            continue
        source = str(need.get("source_system") or "")
        matches = _matching_inventory(need, inventory)
        source_status = str(need.get("source_status") or "planned")
        if source_status == "resolved":
            status = "satisfied"
            rationale = "The planner resolved this need against its source system."
        elif source in {"agreement", "copas"} and verified_authority:
            status = "satisfied"
            rationale = "Verified governing authority is already selected in state."
        elif matches:
            status = "available_for_review"
            rationale = "Potentially responsive evidence is already attached; assess sufficiency before requesting more."
        elif source_status == "needs_human" or source == "external":
            status = "missing"
            rationale = "No potentially responsive artifact is present in the collected-evidence inventory."
        else:
            status = "unresolved"
            rationale = "The need has not been resolved, but absence from the auditor packet is not established."
        coverage.append({
            "requirement_id": f"requirement-{index + 1}",
            "requirement": _clean(need.get("need"), 700),
            "source_system": source,
            "planner_status": source_status,
            "coverage_status": status,
            "evidence_ids": [str(item.get("evidence_id")) for item in matches[:8]],
            "rationale": rationale,
        })
    return coverage
