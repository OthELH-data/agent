"""Assemble classified baseline data and the optional local extractor-synthesis overlay."""
from __future__ import annotations

import os
from pathlib import Path

from shared.case_resolver import Case
from shared.output_paths import infer_year_asset
from shared.extractor_settings import resolve_extractor_excel_parsing_enabled

from shared.classified_exceptions import available_rebuttal_cycles, find_classified_exception
from shared.pipeline_config import input_source_settings, pipeline_config_path
from otl.sharepoint_reader import SharePointReader


def extractor_synthesis(reader: SharePointReader, year: str, asset: str, exc_id: str,
                        rebuttal_cycle: int) -> dict:
    """Read the enabled extractor output matching this run's exact rebuttal cycle."""
    settings = input_source_settings("extractor_outputs")
    if not settings.get("enabled"):
        return {}

    if reader.target == "sharepoint":
        env_name = str(settings.get("read_folder_env") or "SHAREPOINT_READ_FOLDER")
        read_folder = os.getenv(env_name) or ""
        if not read_folder:
            return {}
        reader.read_folder = read_folder
        return reader.sharepoint_synthesis(
            year, asset, exc_id, rebuttal_cycle
        ) or {}

    env_name = str(settings.get("directory_env") or "INPUT_BASE")
    input_base = os.getenv(env_name) or ""
    if not input_base or not Path(input_base).expanduser().is_dir():
        return {}
    reader.input_base = str(Path(input_base).expanduser().resolve())
    return reader.local_synthesis(year, asset, exc_id, rebuttal_cycle) or {}


def extractor_output_required() -> bool:
    """Whether a case may run only when its extractor synthesis is available.

    ``extractor_outputs.enabled`` is deliberately a hard dependency switch, rather than merely a
    permission to read an optional overlay. This keeps an OTL run from silently mixing cases that
    did and did not receive the intended Excel-extractor pre-analysis.
    """
    return bool(input_source_settings("extractor_outputs").get("enabled"))


def extractor_excel_parsing_enabled() -> bool:
    """Whether extractor runs should include Excel workbook parsing.

    The setting is independent of ``extractor_outputs.enabled`` so callers can distinguish disabling
    the extractor integration entirely from running the extractor over non-workbook evidence only.
    Missing settings default to enabled to preserve the behavior of existing deployment configs.
    Both ``excel_parsing: false`` and ``excel_parsing: {enabled: false}`` are supported.
    """
    return resolve_extractor_excel_parsing_enabled(
        input_source_settings("extractor_outputs"), source=pipeline_config_path()
    )


def missing_extractor_synthesis_reason(reader: SharePointReader, year: str, asset: str,
                                       exc_id: str, rebuttal_cycle: int) -> str:
    """Return the user-facing reason for a required-but-missing extractor artifact."""
    settings = input_source_settings("extractor_outputs")
    if reader.target == "sharepoint":
        env_name = str(settings.get("read_folder_env") or "SHAREPOINT_READ_FOLDER")
    else:
        env_name = str(settings.get("directory_env") or "INPUT_BASE")
    return (
        "required Excel Extractor synthesis is missing for "
        f"{year}/{asset}/{exc_id} (OUTPUT_TARGET={reader.target!r} requires "
        f"${env_name}/<year>/<asset>/<exception>/"
        f"rebuttal_{rebuttal_cycle}_synthesis.json)"
    )


def classified_case_state(case: Case, year: str, asset: str) -> tuple[dict, dict]:
    """Return the classified baseline plus the enriched master ``case`` mapping."""
    record = find_classified_exception(year, asset, case.exc_id, case.rebuttal_cycle)
    if record is None:
        # A temporal-split evaluation row is still a valid runnable case even
        # when it has not yet been added to the classified-exceptions catalog.
        # It will progress with CSV facts and naturally evidence-gate when the
        # corresponding attachments are absent.
        record = {"year": year, "asset": asset, "exception_id": case.exc_id,
                  "classification": {}, "rebuttal_cycle": 0}
    else:
        record = dict(record)
    raw_classification = record.get("classification")
    classification = {
        k: v for k, v in raw_classification.items()
        if k not in ("category_id", "model")
    } if isinstance(raw_classification, dict) else {}
    # Return the normalized form to every downstream consumer, not just the case-state projection.
    record["classification"] = classification

    case_state = {
        "case_id": case.case_id,
        "exc_id": case.exc_id,
        "rebuttal_cycle": case.rebuttal_cycle,
        "year": year,
        "asset": asset,
        "exception_type": case.exception_type,
        "amount": record.get("gross_credit_requested", ""),
        "description": record.get("description", ""),
        "short_description": record.get("short_description", ""),
        "error_label": record.get("error_label", ""),
        "error_type": record.get("error_type", ""),
        "error_major": record.get("error_major", ""),
        "matched_category": record.get("matched_category", ""),
        "is_target_category": bool(record.get("is_target_category", False)),
        "classification": classification,
        "table_references": record.get("table_references", []),
        "source_file": record.get("source_file", ""),
        "source_type": record.get("source_type", ""),
    }
    return record, case_state


def graph_input_state(*, case_state: dict, classified: dict, resolved: dict, synthesis: dict,
                      run_id: str, trace_path: str, rebuttal_cycle: int) -> dict:
    """Build the graph seed while enforcing extractor-output isolation when absent/disabled."""
    # The catalog retains its raw attachment metadata, but the labor graph gets one converged attachment
    # list derived from the final EvidenceDiscovery manifest (safe workbooks ∪ supporting PDFs).
    final_attachments = resolved.get("final_attachment_names")
    if final_attachments is None:
        # Compatibility for callers/tests building a resolver dict directly: derive the same basename
        # union from its two final path lists instead of falling back to catalog attachment metadata.
        final_attachments = list(dict.fromkeys(
            Path(str(path or "")).name
            for path in [*(resolved.get("exception_exhibits") or []),
                         *(resolved.get("supporting_pdfs") or [])]
            if Path(str(path or "")).name
        ))
    else:
        final_attachments = list(final_attachments)
    # ``classified_case_state`` already emits this shape, but normalize again at the graph boundary so
    # direct callers and future catalog variants without a ``classification`` block remain safe.
    resolved_case = dict(case_state or {})
    if not isinstance(resolved_case.get("classification"), dict):
        resolved_case["classification"] = {}
    resolved_case["attachments"] = final_attachments
    state = {
        "run_id": run_id,
        "case": resolved_case,
        "final_attachments": final_attachments,
        "trace_path": trace_path,
        # Category routing (and several downstream specialists) consumes the active-cycle narrative
        # from this top-level channel. The case mapping retains it for provenance, but is not a
        # substitute: those agents deliberately read ``auditor_narrative``.
        "auditor_narrative": str(
            classified.get("description") or resolved_case.get("description") or ""
        ).strip(),
        "latest_auditor_reply": dict(classified.get("latest_auditor_reply") or {}),
        "extractor_synthesis_loaded": bool(synthesis),
        "rebuttal_dir": resolved["rebuttal_dir"],
        "auditor_report_pdf": resolved["auditor_report_pdf"],
        "exception_exhibits": resolved["exception_exhibits"],
        "supporting_pdfs": list(resolved.get("supporting_pdfs") or []),
        "all_workbooks_operator_responses": bool(
            resolved.get("all_workbooks_operator_responses")
        ),
        "evidence_exclusion_notice": str(resolved.get("evidence_exclusion_notice") or ""),
        "excluded_operator_workbooks": list(resolved.get("excluded_operator_workbooks") or []),
        "excluded_operator_pdfs": list(resolved.get("excluded_operator_pdfs") or []),
        "common_reports_dir": resolved["common_reports_dir"],
        "otl_information_dir": resolved["otl_information_dir"],
        "sandbox_read_roots": resolved["sandbox_read_roots"],
        "open_questions": [],
        "next_agent": "",
        "final_mode": "",
        "orchestrator_hops": 0,
        "agent_run_counts": {},
        "agent_instructions": {},
        "agent_status": {},
    }
    if synthesis:
        state["synthesis"] = synthesis
        state["open_questions"] = list(synthesis.get("open_questions") or [])
    return state


def expand_rebuttal_cases(cases: list[Case], max_rebuttal_cycles: int | None) -> list[Case]:
    """Expand each selected exception into chronological prerequisite rebuttal cycles.

    With no explicit cycle override, the runner expands every cycle listed for the exception in
    classified_exceptions.json, including cycle 0. An explicit Rebuttal N override runs every available
    cycle through N. Duplicate case rows collapse before expansion.
    """
    grouped: dict[str, list[Case]] = {}
    for case in cases:
        grouped.setdefault(case.case_id, []).append(case)
    expanded: list[Case] = []
    for group in grouped.values():
        base = group[0]
        year, asset = infer_year_asset(base.entity_dir)
        available = available_rebuttal_cycles(year, asset, base.exc_id)
        ceiling = int(max_rebuttal_cycles) if max_rebuttal_cycles else (max(available) if available else 0)
        cycles = [cycle for cycle in available if 0 <= cycle <= ceiling]
        if 0 not in cycles:
            cycles.insert(0, 0)
        for cycle in dict.fromkeys(cycles):
            expanded.append(Case(
                entity_dir=base.entity_dir, exc_id=base.exc_id, exception_type=base.exception_type,
                case_id=base.case_id, rebuttal_cycle=cycle
            ))
    return expanded


def limit_exceptions(cases: list[Case], limit: int | None) -> list[Case]:
    """Keep every row for the first N unique exceptions, preserving their first-seen order."""
    if limit is None or int(limit) <= 0:
        return list(cases)
    allowed: list[str] = []
    for case in cases:
        if case.case_id not in allowed:
            if len(allowed) >= int(limit):
                break
            allowed.append(case.case_id)
    allowed_set = set(allowed)
    return [case for case in cases if case.case_id in allowed_set]


__all__ = [
    "classified_case_state", "expand_rebuttal_cases", "extractor_synthesis", "graph_input_state",
    "extractor_excel_parsing_enabled", "extractor_output_required", "limit_exceptions",
    "missing_extractor_synthesis_reason",
]
