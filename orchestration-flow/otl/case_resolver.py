"""Resolve one CSV row (a `Case`) to the concrete on-disk artifacts the OTL graph needs.

This composes the extractor's evidence-discovery contract — including its final auditor-citation
union — and adds the OTL-specific locations the specialist agents read:

  * `common_reports_dir`  — the "Common Reports for all assets" folder under `<year>` (POB + heli
                            logs for the location agent),
  * `otl_information_dir` — `Data Files/OTL Information` (staff listing + Jobs_Catalogue for
                            job_title / matching),
  * `sandbox_read_roots`  — the concrete dirs the code-exec sandbox may read.

It returns a plain dict that run.py folds straight into the initial `OTLAuditState` (the keys match
the master-state resolver-path fields).

The extractor manifest is authoritative when present. When it is absent, this resolver performs the
same fallback: find safe auditor-side files, extract filenames named in the current description on the
light tier, and union exact resolved matches into the discovery result. Only Excel lock files are
removed after that convergence step.
"""
from __future__ import annotations

from pathlib import Path

from otl._bootstrap import DATA_ROOT
from otl.auditor_attachment_citations import extract_auditor_cited_attachment_names
from shared.case_resolver import Case
from shared.case_keys import _norm
from shared.output_paths import infer_year_asset, synthesis_rel_path
from shared.rebuttals import (Cycle, EvidenceDiscovery, find_exception_evidence,
                              union_auditor_cited_attachments)
from extractor.description import extract_exception_description, report_lists_exception
from extractor.pdf_extract import extract_pdf

_COMMON_REPORTS_KEY = "commonreportsforallassets"
_OTL_INFO_DIRNAME = "OTL Information"


def _is_lock_file(p: Path) -> bool:
    """Excel/office temp lock file, e.g. `~$Workbook.xlsx` — never a real exhibit."""
    return p.name.startswith("~$")


def _attachment_names(paths: list[Path]) -> list[str]:
    """Stable basename union for the case/state attachment view.

    Resolver paths remain available separately in ``exception_exhibits`` and ``supporting_pdfs``. The
    case record carries names because that is the catalog's established representation, but its contents
    must now be derived from those final manifest-backed path lists rather than from catalog metadata.
    """
    names: list[str] = []
    seen: set[str] = set()
    for path in paths:
        name = path.name.strip()
        key = name.casefold()
        if name and key not in seen:
            seen.add(key)
            names.append(name)
    return names


def _find_common_reports_dir(year_dir: Path) -> Path | None:
    """The 'Common Reports for all assets' sibling folder under `<year>` (name varies slightly)."""
    if not year_dir.exists():
        return None
    for d in sorted(year_dir.iterdir()):
        if d.is_dir() and _norm(d.name) == _COMMON_REPORTS_KEY:
            return d
    return None


def _fallback_auditor_cited_attachments(case: Case, discovery: EvidenceDiscovery) -> EvidenceDiscovery:
    """Mirror extractor citation union only when no extractor manifest was supplied."""
    report = discovery.auditor_report
    if not report or not report.is_file():
        return discovery
    try:
        report_text = extract_pdf(report)["text"]
        if (not discovery.workbooks and not discovery.supporting_pdfs
                and not report_lists_exception(report_text, case.exc_id)):
            return discovery
        description = extract_exception_description(report_text, case.exc_id, case.exception_type)
    except Exception:
        # Keep ordinary discovery available when the optional fallback report cannot be read.
        return discovery
    if description.strip():
        union_auditor_cited_attachments(
            discovery, extract_auditor_cited_attachment_names(description)
        )
    return discovery


def resolve_otl_case(case: Case, cycles: list[Cycle], *, evidence_manifest: dict | None = None) -> dict:
    """Resolve a `Case` to the OTL graph's resolver-path fields (a partial `OTLAuditState`).

    Uses the extractor's cycle-specific evidence manifest when present; otherwise discovers evidence
    locally for the exact requested cycle. No later-cycle fallback is permitted.
    """
    entity_dir = Path(case.entity_dir)
    year, asset = infer_year_asset(entity_dir)
    year_dir = entity_dir.parent
    chosen = next((cycle for cycle in cycles if cycle.number == case.rebuttal_cycle), None)

    discovery = EvidenceDiscovery.from_manifest(evidence_manifest or {}, entity_dir)
    if discovery and discovery.cycle_number != case.rebuttal_cycle:
        discovery = None
    if discovery is None and chosen:
        discovery = find_exception_evidence(chosen, case.exc_id)
        discovery = _fallback_auditor_cited_attachments(case, discovery)
    if discovery is None:
        discovery = EvidenceDiscovery(cycle_number=int(case.rebuttal_cycle or 0))

    exhibits = [path for path in discovery.workbooks if not _is_lock_file(path)]
    supporting_pdfs = list(discovery.supporting_pdfs)
    final_attachment_names = _attachment_names([*exhibits, *supporting_pdfs])
    auditor_dir = discovery.auditor_dir or (chosen.auditor_dir if chosen else None)
    auditor_report = discovery.auditor_report or (chosen.auditor_report if chosen else None)

    rebuttal_dir = str(auditor_dir) if auditor_dir else ""
    auditor_report_pdf = str(auditor_report) if auditor_report else ""

    common_reports = _find_common_reports_dir(year_dir)
    otl_information = DATA_ROOT / _OTL_INFO_DIRNAME

    # The specific dirs the code-exec sandbox is told about. code_exec also always allows DATA_ROOT,
    # so this is the informative shortlist, not the whole permission surface.
    read_roots: list[str] = []
    for candidate in (auditor_dir, common_reports, otl_information):
        if candidate and Path(candidate).exists():
            read_roots.append(str(candidate))

    return {
        "rebuttal_dir": rebuttal_dir,
        "auditor_report_pdf": auditor_report_pdf,
        "exception_exhibits": [str(p) for p in exhibits],
        "supporting_pdfs": [str(p) for p in supporting_pdfs],
        "final_attachment_names": final_attachment_names,
        "all_workbooks_operator_responses": discovery.all_workbooks_operator_responses,
        "evidence_exclusion_notice": discovery.evidence_exclusion_notice,
        "excluded_operator_workbooks": [str(path) for path in discovery.excluded_operator_workbooks],
        "excluded_operator_pdfs": [str(path) for path in discovery.excluded_operator_pdfs],
        "common_reports_dir": str(common_reports) if common_reports else "",
        "otl_information_dir": str(otl_information) if otl_information.exists() else "",
        "sandbox_read_roots": read_roots,
        # convenience header fields (run.py also sets these on `case`)
        "_year": year,
        "_asset": asset,
        "_synthesis_rel_path": synthesis_rel_path(year, asset, case.exc_id),
    }


__all__ = ["resolve_otl_case"]
