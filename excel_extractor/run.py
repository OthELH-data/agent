"""Explain what an auditor's Excel exhibits demonstrate for an audit exception.

Reads an audit report PDF + one or more Excel exhibits, uses Python (PyMuPDF/openpyxl) to
extract their content, then an LLM (via the IBM ICA gateway) to explain what each tab
demonstrates relative to the exception and pinpoint the figures the auditor likely used to
reach the reasoning and the questioned amount. Writes a JSON report.

One case per run, named by its case id (`<Asset>_<year>_<exc>`). The id is all that's needed: the
entity folder is matched off disk and the topic comes from datastores/clustered_exception_tables.csv,
so no CSV row has to exist. Every rebuttal cycle the exception appears in is processed (one JSON per
cycle). `run_batch.py` is the parallel driver — it runs one of these per case.

Usage:
  python run.py --case-id GreatWhite_2024_0040
  python run.py --case-id GreatWhite_2024_0040 --dry-run     # resolve and exit; no LLM spend
  python run.py --case-id Silvertip_2024_0019 --topic "BOM & MDM Services"   # topic not in the catalog
  python run.py --case-id GreatWhite_2024_0040 --no-skill    # skip skill resolution

  # Check the LLM setup only — config offline, or a live round-trip per tier:
  python run.py --llm-config
  python run.py --provider azure --llm-config
  python run.py --smoke-test
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

_EXTRACTOR_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXTRACTOR_DIR.parent
for _path in (str(_EXTRACTOR_DIR), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from shared.case_keys import _norm, pad_exc  # noqa: E402
from shared.case_resolver import (DEFAULT_CASES_CSV, default_corpus_base, lookup_exception_type,
                                  resolve_case_id)  # noqa: E402
from shared.classified_exceptions import (classified_exceptions_path,
                                          find_classified_exception)  # noqa: E402
from extractor.description import (  # noqa: E402
    extract_exception_description, extract_rebuttal_dialogue, report_lists_exception,
    split_rebuttal_dialogue,
)
from shared.excel_bounded import (  # noqa: E402
    collect_wbs_candidates, extract_excel_bounded, extract_excel_full, sheets_to_json,
)
from extractor.explain import (  # noqa: E402
    build_invoice_evidence, extract_auditor_cited_attachment_names, interpret_attachment,
    select_wbs_focus, summarize_rebuttals, synthesize,
)
from extractor.invoice_extract import discover_invoices, extract_invoice  # noqa: E402
from extractor.llm_adapter import HOOK_MODULE as EXTRACTOR_HOOK_MODULE  # noqa: E402
from extractor.skill_engine import apply_skill  # noqa: E402
from extractor.skills import resolve_skill  # noqa: E402
from shared.output_paths import (  # noqa: E402
    infer_year_asset, make_sink, synthesis_rel_path,
)
from extractor.pdf_extract import extract_pdf  # noqa: E402
from shared.rebuttals import (  # noqa: E402
    EvidenceDiscovery, build_previous_rebuttals, discover_rebuttal_cycles,
    exception_folder_is_empty, find_exception_evidence, union_auditor_cited_attachments,
)
from extractor.schemas import (AttachmentInterpretation, ExhibitEvidenceReport, HookExecutionStatus,
                               InvoiceEvidence, RebuttalDialogueEntry)  # noqa: E402
from shared.hooks import BudgetHookBlocked, HOOKS  # noqa: E402
from shared.token_schemas import TokenUsageReport  # noqa: E402
from shared.tokens import USAGE  # noqa: E402
from shared.extractor_settings import extractor_excel_parsing_enabled  # noqa: E402
from shared.token_metrics_cache import (cache_report_metrics, ensure_run_id,  # noqa: E402
                                        mark_metric_write)

def _claimed_amount(description: str) -> float | None:
    m = re.search(r"\$([\d,]+\.\d{2})", description)
    return float(m.group(1).replace(",", "")) if m else None


def _latest_catalog_auditor_reply(year: str, asset: str, exception: str,
                                  cycle_number: int) -> dict[str, str | int]:
    """Return the latest cataloged Auditor reply available as of this rebuttal cycle.

    Catalog rebuttal records may be absent for a given cycle, so walk backward rather than leak a
    later auditor position into an earlier cycle. The catalog has already removed Operator entries
    when it derives ``latest_auditor_reply``.
    """
    for requested_cycle in range(cycle_number, -1, -1):
        try:
            record = find_classified_exception(year, asset, exception, requested_cycle)
        except KeyError:
            continue
        except FileNotFoundError:
            return {}
        if not record:
            return {}
        reply = record.get("latest_auditor_reply") or {}
        text = str(reply.get("text") or "").strip()
        if text:
            return {
                "rebuttal_cycle": int(reply.get("rebuttal_cycle") or requested_cycle),
                "date": str(reply.get("date") or "").strip(),
                "text": text,
            }
    return {}


def _latest_pdf_auditor_reply(entries: list[dict[str, str]],
                              cycle_number: int) -> dict[str, str | int]:
    """Return the final Auditor-labeled reply from the existing cycle PDF dialogue."""
    if cycle_number < 1:
        return {}
    for entry in reversed(entries):
        if str(entry.get("party") or "").strip().lower() != "auditor":
            continue
        text = str(entry.get("reply") or "").strip()
        if text:
            return {
                "rebuttal_cycle": cycle_number,
                "date": "",
                "text": text,
                "source": "audit_report_pdf",
            }
    return {}


def _with_latest_auditor_reply(description: str, latest_auditor_reply: dict | None) -> str:
    """Add the selected auditor reply to prompts used before final synthesis."""
    reply = latest_auditor_reply or {}
    text = str(reply.get("text") or "").strip()
    if not text:
        return description
    cycle = reply.get("rebuttal_cycle")
    date = str(reply.get("date") or "").strip()
    label = f"Rebuttal {cycle}" if cycle is not None else "Auditor rebuttal"
    if date:
        label += f" — {date}"
    return (
        f"{description}\n\n"
        "LATEST AUDITOR REPLY (auditor-only context; do not treat it as an operator response):\n"
        f"[{label}]\n{text}"
    )


@dataclass(frozen=True)
class _ExtractorCaseMetadata:
    """The report-header facts supplied by a present classified-exceptions catalog."""

    year: str
    asset: str
    exception_category: str
    case_id: str
    title: str
    auditor_description: str
    gross_credit_requested: str


def _classified_case_metadata(entity_dir: Path, exception: str, fallback_case_id: str,
                              fallback_title: str) -> _ExtractorCaseMetadata | None:
    """Return catalog-backed extractor facts, or ``None`` when no catalog file exists.

    A missing catalog is intentionally a true no-op: callers keep their existing folder/PDF/CSV path.
    When the catalog exists but its historical records lack ``classification``, the designated labor
    classification CSV supplies both category and title.
    """
    catalog_path = classified_exceptions_path()
    if not catalog_path.is_file():
        return None

    source_year, source_asset = infer_year_asset(entity_dir)
    record = find_classified_exception(source_year, source_asset, exception)
    if not record:
        return None

    year = str(record.get("year") or source_year)
    asset = str(record.get("asset") or source_asset)
    exception_id = pad_exc(str(record.get("exception_id") or exception))
    classification = record.get("classification")
    classification = classification if isinstance(classification, dict) else {}
    category = str(
        classification.get("exception_category") or classification.get("category") or ""
    ).strip()
    title = str(
        classification.get("title") or classification.get("exception_title") or ""
    ).strip()
    if not category or not title:
        csv_category = lookup_exception_type(year, asset, exception_id, [DEFAULT_CASES_CSV])
        category = category or csv_category
        title = title or csv_category

    canonical_case_id = f"{re.sub(r'[^A-Za-z0-9]', '', _norm(asset))}_{year}_{exception_id}"
    return _ExtractorCaseMetadata(
        year=year,
        asset=asset,
        exception_category=category or fallback_title,
        case_id=str(record.get("case_id") or canonical_case_id or fallback_case_id),
        title=title or fallback_title,
        auditor_description=str(record.get("description") or ""),
        gross_credit_requested=str(record.get("gross_credit_requested") or ""),
    )


def _coerce_evidence_discovery(value, cycle) -> EvidenceDiscovery:
    """Keep older test hooks returning ``(workbooks, pdfs)`` compatible during the API migration."""
    if isinstance(value, EvidenceDiscovery):
        return value
    workbooks, supporting_pdfs = value
    return EvidenceDiscovery(
        cycle_number=cycle.number,
        auditor_dir=cycle.auditor_dir,
        auditor_report=cycle.auditor_report,
        operator_dir=cycle.operator_dir,
        operator_report=cycle.operator_report,
        workbooks=list(workbooks),
        supporting_pdfs=list(supporting_pdfs),
    )


_PARTIAL_REPORT: ContextVar[ExhibitEvidenceReport | None] = ContextVar(
    "extractor_partial_report", default=None
)


def _build_report(report_path: Path, exhibit_paths: list[Path], exception: str,
                 title: str, case_id: str, invoice_paths: list[Path] | None = None,
                 auto_invoices: bool = True, reset_usage: bool = True,
                 previous_rebuttals: dict[str, str] | None = None,
                 year: str = "", asset: str = "", category: str = "",
                 skill_name: str | None = None, use_skills: bool = True,
                 report_text: str | None = None,
                 auditor_description: str | None = None,
                 latest_auditor_reply: dict | None = None,
                 rebuttal_cycle: int = 0,
                 excel_parsing_enabled: bool = True,
                 evidence_discovery: dict | None = None):
    print(f"# Exhibit-evidence explainer — {case_id}: exception {exception} ({title})")
    started = time.monotonic()

    # Fresh token counts for this exception so the JSON reports only its own run. In rebuttal-cycle
    # mode the caller manages the USAGE window (to also capture the running-summary call), so it
    # passes reset_usage=False.
    if reset_usage:
        USAGE.reset()

    # `report_text` lets a caller that already read this PDF (the cycle loop, for its own guards) hand
    # the text over rather than have it parsed a second time — which for a scanned report means OCR
    # twice. Absent it, read the PDF here.
    print(f"  reading report: {report_path.name}")
    if report_text is None:
        report_text = extract_pdf(report_path)["text"]
    description = (str(auditor_description) if auditor_description is not None
                   else extract_exception_description(report_text, exception, title))
    claimed = _claimed_amount(description)
    print(f"  description: {len(description)} chars; claimed amount = {claimed}")

    # The audit report is a running document — later cycles embed the full operator/auditor rebuttal
    # dialogue per exception. Pull it straight from here (no operator-folder reads) for the synthesis.
    operator_dialogue = extract_rebuttal_dialogue(report_text, exception, title)
    rebuttal_dialogue_entries = split_rebuttal_dialogue(operator_dialogue)
    if operator_dialogue:
        print(f"  prior-round rebuttal dialogue: {len(operator_dialogue)} chars / "
              f"{len(rebuttal_dialogue_entries)} reply entries (from the report)")
    active_auditor_reply = latest_auditor_reply or _latest_pdf_auditor_reply(
        rebuttal_dialogue_entries, rebuttal_cycle,
    )
    llm_description = _with_latest_auditor_reply(description, active_auditor_reply)
    if active_auditor_reply:
        source = "catalog" if active_auditor_reply.get("source") != "audit_report_pdf" else "PDF"
        print(f"  {source} auditor reply: rebuttal {active_auditor_reply.get('rebuttal_cycle')} "
              f"({len(str(active_auditor_reply.get('text') or ''))} chars)")

    previous_rebuttals = previous_rebuttals or {}
    result = ExhibitEvidenceReport(
        year=year,
        asset=asset,
        exception_category=category or title,
        case_id=case_id,
        exception_code=exception,
        title=title,
        auditor_description=description,
        excel_parsing_enabled=excel_parsing_enabled,
        evidence_discovery=dict(evidence_discovery or {}),
        previous_rebuttals=previous_rebuttals,
        rebuttal_dialogue=operator_dialogue,
        rebuttal_dialogue_entries=[RebuttalDialogueEntry(**entry)
                                  for entry in rebuttal_dialogue_entries],
    )
    _PARTIAL_REPORT.set(result)

    # Resolve a domain skill for this exception type (the topic — from the catalog, or --topic). When
    # one exists we pull ALL rows (full extraction) so the skill pass can scan everything; otherwise
    # the bounded default.
    skill = (resolve_skill(skill_name or category or title, description=llm_description)
             if use_skills and excel_parsing_enabled else None)
    if skill:
        print(f"  skill: {skill.name} (skill_{skill.name}.md) — full-value extraction + skill scan")
    elif not excel_parsing_enabled:
        print(f"  Excel parsing: disabled by pipeline config — skipping {len(exhibit_paths)} "
              f"workbook(s), WBS selection, tab interpretation, and skill scanning")

    # WBS pre-filter (skill runs): discover the distinct WBS/cost-object codes present in the
    # evidence (with their descriptions), let a light pass pick the code(s) of interest, then filter
    # the JADE ledger to those codes before it is chunked. On `unsure`/no candidates → no narrowing.
    wbs_codes: list[str] | None = None
    focus_facts: dict[str, str] = {}
    if skill:
        wbs_map: dict[str, str] = {}
        for path in exhibit_paths:
            wbs_map.update(collect_wbs_candidates(path))
        focus = select_wbs_focus(llm_description, wbs_map, skill=skill)
        focus_facts = focus.focus
        if focus.unsure or not focus.wbs_codes:
            print(f"  WBS focus: {len(wbs_map)} candidate code(s); of interest: none "
                  f"(unsure={focus.unsure}) — analyzing the full ledger | focus={focus_facts}")
        else:
            wbs_codes = focus.wbs_codes
            print(f"  WBS focus: {len(wbs_map)} candidate code(s); of interest: {wbs_codes} | "
                  f"focus={focus_facts}")

    attachments = result.attachments
    skill_sheets_json: list[dict] = []
    tab_profiles: dict[str, dict] = {}   # "<file>::<tab>" -> computed column profile for the synthesis
    for path in exhibit_paths if excel_parsing_enabled else []:
        print(f"\n  exhibit: {path.name}")
        book = extract_excel_full(path, wbs_codes=wbs_codes) if skill else extract_excel_bounded(path)
        if book.get("error"):
            label = "empty file" if book.get("empty") else "read error"
            print(f"    SKIP ({label}): {book['error']}")
            continue

        # Operator-reply tabs were dropped at extraction and never reach the LLM — say which, so the
        # exclusion is auditable rather than a silently shorter tab list.
        if book.get("excluded_tabs"):
            print(f"    excluded operator-response tab(s): {book['excluded_tabs']}")

        def _log(tab_name, ti):
            print(f"    tab {tab_name!r}: ties={ti.ties_to_amount}, "
                  f"figures={len(ti.auditor_figures_used)}")

        # Report how far the WBS pre-filter narrowed each JADE ledger (no silent truncation).
        if wbs_codes:
            for sh in book["sheets"]:
                if sh.get("wbs_filtered"):
                    print(f"    JADE {sh.get('name')!r}: kept {sh.get('n_data_rows_scanned')} of "
                          f"~{sh.get('reported_rows')} reported rows for WBS {wbs_codes}")

        attachment = AttachmentInterpretation(
            file_name=path.name, role="", overall_summary="", tabs=[]
        )
        attachments.append(attachment)
        interpret_attachment(
            llm_description, path.name, book["sheets"], on_tab=_log, target=attachment,
        )
        # Carry each tab's computed profile through to the synthesis. The synthesis states the
        # exhibit's date range and its total, but it sees only the tab INTERPRETATIONS — so without
        # this it can only repeat whatever range the interpreter happened to write down, which is how
        # a March-to-August window got reported for charges running to December.
        for sh in book["sheets"]:
            if sh.get("column_profile"):
                tab_profiles[f"{path.name}::{sh.get('name')}"] = {
                    "column_profile": sh["column_profile"],
                    "total_rows": sh.get("total_rows") or [],
                    # Which table the profile describes, on a tab that holds more than one.
                    "blocks": sh.get("blocks") or [],
                }
        if skill:
            for tab in sheets_to_json(book):
                tab["file"] = path.name
                skill_sheets_json.append(tab)

    if excel_parsing_enabled and not attachments:
        print("  NOTE: no auditor Excel evidence found — the synthesis will record that the "
              "auditor provided no supporting evidence (insufficient information to assess the claim).")

    # --- skill pass: chunked, LLM-driven scan over ALL values (no deterministic filtering) ---
    skill_analysis = None
    if skill and skill_sheets_json:
        print(f"\n  applying skill {skill.name} over {len(skill_sheets_json)} tab(s)…")
        skill_analysis = apply_skill(
            skill, skill_sheets_json, llm_description,
            meta={"title": title, "disputed_amount": claimed, "focus": focus_facts})
        print(f"    scanned {skill_analysis['chunks_scanned']}/{skill_analysis['total_chunks']} chunk(s); "
              f"findings={len(skill_analysis['findings'])}; sufficient={skill_analysis['sufficient']}; "
              f"gaps={len(skill_analysis['gaps'])}")

    # --- invoices: explicit --invoice paths + (optionally) sibling PDFs in the exhibit folders ---
    invoice_candidates: list[Path] = list(invoice_paths or [])
    if auto_invoices:
        discovered = discover_invoices(exhibit_paths, report_path)
        known = {p.name for p in invoice_candidates}
        for p in discovered:
            if p.name not in known:
                invoice_candidates.append(p)
                known.add(p.name)

    invoices = result.invoices
    if invoice_candidates:
        print(f"\n  invoices: {len(invoice_candidates)} found")
        for path in invoice_candidates:
            if not path.exists():
                print(f"    SKIP (missing): {path}")
                continue
            data = extract_invoice(path)
            if data.get("error"):
                print(f"    SKIP (read error) {path.name}: {data['error']}")
                continue
            invoice_context = llm_description
            if not excel_parsing_enabled and operator_dialogue:
                invoice_context += ("\n\nPRIOR-ROUND REBUTTAL DIALOGUE:\n" + operator_dialogue)
            ev = InvoiceEvidence(
                file_name=data.get("file_name", path.name),
                source_path=data.get("source", str(path)),
                n_pages=data.get("n_pages", 0),
                extraction_method=data.get("extraction_method", "text"),
                ocr_pages=data.get("ocr_pages", 0),
                ocr_available=bool(data.get("ocr_available")),
                parsed=data.get("parsed") or {},
            )
            invoices.append(ev)
            build_invoice_evidence(
                invoice_context, data, attachments=attachments,
                excel_parsing_enabled=excel_parsing_enabled,
                target=ev,
            )
            dropped = data.get("dropped_boilerplate_pages", 0)
            print(f"    invoice {path.name!r}: method={ev.extraction_method}, "
                  f"pages={ev.n_pages}" + (f" (−{dropped} T&C)" if dropped else "") + ", "
                  f"line_items={len(ev.extraction.line_items)}, "
                  f"ties={ev.ties_to_amount}")

    print("\n  synthesizing across exhibits…")
    if previous_rebuttals:
        print(f"    (incorporating {len(previous_rebuttals)} prior-rebuttal context entr"
              f"{'y' if len(previous_rebuttals) == 1 else 'ies'} into synthesis)")
    meta = {"case_id": case_id, "exception_code": exception, "title": title,
            "disputed_amount": claimed, "currency": "USD",
            "year": year, "asset": asset, "exception_category": category or title}
    result = synthesize(description, attachments, meta, invoices=invoices,
                        previous_rebuttals=previous_rebuttals,
                        operator_dialogue=operator_dialogue,
                        rebuttal_dialogue_entries=rebuttal_dialogue_entries,
                        latest_auditor_reply=active_auditor_reply,
                        skill=skill, skill_analysis=skill_analysis,
                        tab_profiles=tab_profiles,
                        excel_parsing_enabled=excel_parsing_enabled)

    # Attach the token totals (all tiers — provider-reported on both ica and azure, tiktoken only
    # for a reply that carried no usage block) + wall-clock minutes at the bottom of the report.
    result.token_usage = TokenUsageReport.model_validate(USAGE.report())
    result.elapsed_minutes = round((time.monotonic() - started) / 60, 2)
    return result


def _placeholder_sources(report: ExhibitEvidenceReport, exhibit_paths: list[Path],
                         invoice_paths: list[Path]) -> None:
    """Retain the identity of discovered evidence that a hook prevented from being processed."""
    attached = {item.file_name for item in report.attachments}
    for path in exhibit_paths if report.excel_parsing_enabled else []:
        if path.name not in attached:
            report.attachments.append(AttachmentInterpretation(
                file_name=path.name,
                role="",
                overall_summary="",
                tabs=[],
            ))
    invoiced = {item.file_name for item in report.invoices}
    for path in invoice_paths:
        if path.name not in invoiced:
            report.invoices.append(InvoiceEvidence(file_name=path.name, source_path=str(path)))


def _attach_hook_status(report: ExhibitEvidenceReport) -> None:
    usage_report = USAGE.report()
    snapshot = HOOKS.budget_snapshot(
        module=EXTRACTOR_HOOK_MODULE,
        usage_report=usage_report,
        usage_generation=USAGE.generation(),
    )
    report.processing.hooks["budget"] = HookExecutionStatus.model_validate(snapshot)
    report.token_usage = TokenUsageReport.model_validate(usage_report)


def _synthesize_after_budget_stop(report: ExhibitEvidenceReport,
                                  latest_auditor_reply: dict | None = None) -> None:
    """Fill terminal synthesis fields from evidence collected before a budget stop.

    This is intentionally the sole extraction call allowed to exceed the configured budget.  It
    goes through the normal tracked LLM path, so provider token and cost data are included in the
    final report instead of being treated as an unreported recovery expense.
    """
    meta = {
        "case_id": report.case_id,
        "exception_code": report.exception_code,
        "title": report.title,
        "disputed_amount": _claimed_amount(report.auditor_description),
        "currency": "USD",
        "year": report.year,
        "asset": report.asset,
        "exception_category": report.exception_category,
    }
    final = synthesize(
        report.auditor_description,
        report.attachments,
        meta,
        invoices=report.invoices,
        previous_rebuttals=report.previous_rebuttals,
        operator_dialogue=report.rebuttal_dialogue,
        rebuttal_dialogue_entries=[entry.model_dump() for entry in report.rebuttal_dialogue_entries],
        latest_auditor_reply=latest_auditor_reply,
        excel_parsing_enabled=report.excel_parsing_enabled,
        allow_budget_overrun=True,
    )
    report.likely_basis_for_amount = final.likely_basis_for_amount
    report.synthesis = final.synthesis
    report.open_questions = final.open_questions
    report.operator_dialogue_considered = final.operator_dialogue_considered


def build_report(report_path: Path, exhibit_paths: list[Path], exception: str,
                 title: str, case_id: str, invoice_paths: list[Path] | None = None,
                 auto_invoices: bool = True, reset_usage: bool = True,
                 previous_rebuttals: dict[str, str] | None = None,
                 year: str = "", asset: str = "", category: str = "",
                 skill_name: str | None = None, use_skills: bool = True,
                 report_text: str | None = None,
                 auditor_description: str | None = None,
                 latest_auditor_reply: dict | None = None,
                 rebuttal_cycle: int = 0,
                 excel_parsing_enabled: bool = True,
                 evidence_discovery: dict | None = None):
    """Build a complete report, or finalize the in-progress report after a hook stop."""
    started = time.monotonic()
    token = _PARTIAL_REPORT.set(None)
    try:
        try:
            result = _build_report(
                report_path, exhibit_paths, exception, title, case_id,
                invoice_paths=invoice_paths, auto_invoices=auto_invoices,
                reset_usage=reset_usage, previous_rebuttals=previous_rebuttals,
                year=year, asset=asset, category=category, skill_name=skill_name,
                use_skills=use_skills, report_text=report_text,
                auditor_description=auditor_description,
                latest_auditor_reply=latest_auditor_reply,
                rebuttal_cycle=rebuttal_cycle,
                excel_parsing_enabled=excel_parsing_enabled,
                evidence_discovery=evidence_discovery,
            )
        except BudgetHookBlocked as exc:
            result = _PARTIAL_REPORT.get()
            if result is None:
                raise
            result.processing.status = "partial"
            result.processing.stopped_stage = exc.stage
            result.processing.stop_reason = exc.reason
            print("  synthesizing from evidence collected before the budget stop…")
            recovery_reply = latest_auditor_reply or _latest_pdf_auditor_reply(
                [entry.model_dump() for entry in result.rebuttal_dialogue_entries], rebuttal_cycle,
            )
            _synthesize_after_budget_stop(result, recovery_reply)
            result.open_questions.append(
                f"Processing stopped before {exc.stage} because the configured extractor budget "
                f"hook blocked the next LLM call ({exc.reason}); synthesis uses the evidence collected "
                "before that stop."
            )
            candidates = list(invoice_paths or [])
            if auto_invoices:
                known = {path.name for path in candidates}
                for path in discover_invoices(exhibit_paths, report_path):
                    if path.name not in known:
                        candidates.append(path)
                        known.add(path.name)
            _placeholder_sources(result, exhibit_paths, candidates)
            print(f"  !! {exc}; finalizing partial cycle output", file=sys.stderr)

        if isinstance(result, ExhibitEvidenceReport):
            _attach_hook_status(result)
            result.elapsed_minutes = round((time.monotonic() - started) / 60, 2)
            # Validate the fully defaulted shape before either output sink sees it, while preserving
            # object identity for callers that keep a reference to the completed report.
            ExhibitEvidenceReport.model_validate(result.model_dump(mode="python"))
        return result
    finally:
        _PARTIAL_REPORT.reset(token)


def _print_result(result) -> None:
    """Human-readable summary of one finished cycle: synthesis, the amount's basis, invoices, tokens."""
    print("\n--- SYNTHESIS ---")
    print(result.synthesis)
    print("\n--- LIKELY BASIS FOR AMOUNT ---")
    for f in result.likely_basis_for_amount:
        print(f"  • {f.value}  @ {f.location}\n      {f.why_relevant}")

    if result.invoices:
        print("\n--- INVOICES ---")
        for iv in result.invoices:
            ext = iv.extraction
            print(f"  • {iv.file_name}  [{iv.ties_to_amount}; {iv.extraction_method}"
                  f"{f', ocr_pages={iv.ocr_pages}' if iv.ocr_pages else ''}]")
            print(f"      vendor={ext.vendor or iv.parsed.vendor} "
                  f"number={ext.invoice_number or iv.parsed.invoice_number} "
                  f"total={ext.total_amount or iv.parsed.total_amount} "
                  f"line_items={len(ext.line_items)}")
            print(f"      {iv.demonstrates}")
            if iv.ties_to_exhibit:
                print(f"      reconciliation: {iv.ties_to_exhibit}")

    tu = result.token_usage
    if tu:
        # "gateway" = the provider's own usage block, which both ica and azure return; "tiktoken" =
        # estimated locally; "mixed" = some tiers each way, i.e. a tier whose replies carried no
        # usage at all fell back to the estimate.
        label = {"gateway": "TOKENS (gateway-reported)",
                 "mixed": "TOKENS (mixed: gateway + tiktoken estimate)"}.get(
                     tu.estimator, "ESTIMATED TOKENS (tiktoken)")
        print(f"\n--- {label} ---")
        # `in=` is the whole prompt, so the columns still read as "sent" and "returned". The cache
        # line below splits that prompt into the three classes it was actually billed as.
        for tier, t in tu.by_tier.items():
            print(f"  [{tier:>5}] {t.model:<18} in={t.prompt_tokens:>7} out={t.tokens_out:>7} "
                  f"({t.calls} calls)")
        print(f"  {'TOTAL':>7}      in={tu.total_prompt_tokens:>7} out={tu.total_tokens_out:>7} "
              f"→ {tu.total_tokens} tokens")
        if tu.total_cache_read_tokens or tu.total_cache_creation_tokens:
            print(f"  {'cache':>7}      read={tu.total_cache_read_tokens} "
                  f"write={tu.total_cache_creation_tokens} new_in={tu.total_tokens_in}")
        if tu.total_thinking_tokens:
            print(f"  {'think':>7}      {tu.total_thinking_tokens} of the output tokens")
        if tu.cost.total_usd is not None:
            saved = tu.cost.cache_savings_usd or 0.0
            print(f"  {'cost':>7}      {tu.cost.total_usd:.4f} {tu.cost.currency}"
                  + (f"  (caching saved {saved:.4f})" if saved else "")
                  + (f"  [unpriced: {', '.join(tu.cost.unpriced_tiers)}]"
                     if tu.cost.unpriced_tiers else ""))
    if result.elapsed_minutes is not None:
        print(f"  elapsed: {result.elapsed_minutes} min")


def run_rebuttal_cycles(entity_dir: Path, exception: str, title: str, case_id: str,
                        sink, auto_invoices: bool = True, category: str = "",
                        skill_name: str | None = None, use_skills: bool = True,
                        excel_parsing_enabled: bool = True, year: str = "", asset: str = "",
                        auditor_description: str | None = None, force: bool = False) -> list:
    """Process every rebuttal cycle the exception appears in, writing one JSON per cycle.

    A cycle is skipped when it has nothing to say about the exception: no auditor report PDF, a
    per-exception exhibit folder that exists and is empty (`exception_folder_is_empty`), or a report
    with no entry for the exception and no exhibits (`report_lists_exception`). Both of the latter two
    are late-cycle reports about the exceptions still open that round, where the exception in hand has
    already closed.

    Each remaining cycle runs the full exhibit+invoice+synthesis pipeline. Cycles are chained via
    `previous_rebuttals`: the oldest appearances are compressed into a running summary (rolled
    forward every 3rd appearance) and the most recent (count mod 3) stay verbatim.

    Outputs go to <year>/<asset>/<exception>/rebuttal_<n>_synthesis.json via `sink` (local or
    SharePoint).
    """
    cycles = discover_rebuttal_cycles(entity_dir)
    if not cycles:
        sys.exit(f"No rebuttal cycles found under {entity_dir}")
    print(f"# Rebuttal-cycle run — {case_id}: exception {exception} ({title})")
    print(f"  entity: {entity_dir}")
    print(f"  discovered cycles: {[c.number for c in cycles]}")

    # Infer year/asset from the entity_dir path (e.g. .../2022/Great White (BP Only)/...)
    inferred_year, inferred_asset = infer_year_asset(entity_dir)
    year = year or inferred_year
    asset = asset or inferred_asset

    # Skip if already processed unless the operator explicitly asks to overwrite it.
    exc_folder = f"{year}/{asset}/{exception}"
    if sink.folder_exists(exc_folder) and not force:
        print(f"  SKIP (already processed): {exc_folder}/ exists with synthesis output")
        return []
    if force and sink.folder_exists(exc_folder):
        print(f"  FORCE: re-running and overwriting synthesis output under {exc_folder}/")

    prior_reports: list = []
    running_summary = ""
    summarized_upto = 0  # how many prior appearances are already folded into running_summary
    written: list[str] = []

    for cyc in cycles:
        # Evidence (exhibits + invoices) is scoped to this exception on the AUDITOR side only.
        # Citation extraction below can promote a response-named workbook when the auditor expressly
        # cited it. Suppress the preliminary filename-filter warning so the log reflects the final
        # evidence set after that promotion, not an intermediate state.
        evidence = _coerce_evidence_discovery(
            find_exception_evidence(cyc, exception, warn=lambda _message: None), cyc
        )
        xlsx = evidence.workbooks
        exc_invoices = evidence.supporting_pdfs
        # Auditor side ONLY — never the operator-response folder. Two conditions skip a cycle: no
        # auditor report PDF to read the exception description from, or an exhibit folder that exists
        # and is empty. Any other cycle that HAS the report but no exhibits is still processed (the
        # synthesis records that no auditor evidence was given).
        report = cyc.auditor_report
        if not report:
            print(f"\n  -- cycle {cyc.number}: exception {exception} — no auditor report PDF; "
                  f"skipping")
            continue
        # An empty per-exception folder means the auditor filed the slot and attached nothing —
        # normally because the exception closed that round. The report then carries no description
        # block for it either, so synthesizing would rest on whatever the description fallback
        # scrapes out of a neighbouring exception. Skip the cycle rather than write that.
        if exception_folder_is_empty(cyc, exception):
            print(f"\n  -- cycle {cyc.number}: exception {exception} — exhibit folder exists but is "
                  f"empty (no evidence filed this cycle); skipping")
            continue

        # Read the report once here: the guard below needs its text, and build_report takes it from us
        # instead of re-parsing (and possibly re-OCR'ing) the same PDF.
        try:
            report_text = extract_pdf(report)["text"]
        except Exception as exc:
            print(f"\n  -- cycle {cyc.number}: exception {exception} — cannot read "
                  f"{report.name} ({type(exc).__name__}: {exc}); skipping")
            continue

        # A late-cycle letter carries only the exceptions still open that round. When this report has
        # no entry for the exception AND the cycle filed no exhibits, there is nothing here about it:
        # the description would come from the fallback matching the code inside an audit number or
        # another exception's row, i.e. a neighbouring exception's paragraph under this code. A report
        # that DOES list it with no exhibits is the real 'auditor claimed it and filed nothing'
        # finding, and still runs.
        if not xlsx and not exc_invoices and not report_lists_exception(report_text, exception):
            print(f"\n  -- cycle {cyc.number}: exception {exception} — not listed in "
                  f"{report.name} and no exhibits filed; skipping")
            continue

        # Extract only the filenames the auditor names in THIS exception description. The light-tier
        # call never receives the discovered list; deterministic evidence discovery resolves its result
        # to safe exact matches and unions them into the extractor/OTL handoff manifest.
        cycle_auditor_reply = _latest_catalog_auditor_reply(year, asset, exception, cyc.number)
        if not cycle_auditor_reply and cyc.number >= 1:
            pdf_dialogue = extract_rebuttal_dialogue(report_text, exception, title)
            cycle_auditor_reply = _latest_pdf_auditor_reply(
                split_rebuttal_dialogue(pdf_dialogue), cyc.number,
            )
        current_description = (str(auditor_description) if auditor_description is not None
                               else extract_exception_description(report_text, exception, title))
        try:
            cited_attachment_names = extract_auditor_cited_attachment_names(
                _with_latest_auditor_reply(current_description, cycle_auditor_reply)
            )
        except BudgetHookBlocked as exc:
            # This description-only citation pass happens immediately before build_report(), so it
            # previously had no in-progress ExhibitEvidenceReport for build_report's partial-output
            # handler to finalize. Skip the optional citation refinement; the first budget-aware call
            # inside build_report will now convert the cycle into a valid partial report and write it.
            cited_attachment_names = []
            print(f"  !! {exc}; skipping optional attachment-citation refinement and "
                  "finalizing partial cycle output", file=sys.stderr)
        evidence = union_auditor_cited_attachments(evidence, cited_attachment_names)
        xlsx = evidence.workbooks
        exc_invoices = evidence.supporting_pdfs
        if evidence.evidence_exclusion_notice:
            print(f"\n  -- cycle {cyc.number}: exception {exception} — "
                  f"{evidence.evidence_exclusion_notice}")
        if cited_attachment_names:
            print(f"  auditor-cited attachment names: {evidence.auditor_cited_attachment_names}")
        if evidence.cited_attachment_workbooks or evidence.cited_attachment_pdfs:
            print("  citation-resolved files: "
                  f"workbooks={[path.name for path in evidence.cited_attachment_workbooks]}, "
                  f"supporting_pdfs={[path.name for path in evidence.cited_attachment_pdfs]}")

        if excel_parsing_enabled and not xlsx:
            print(f"\n  -- cycle {cyc.number}: exception {exception} — auditor provided no Excel "
                  f"evidence; recording 'insufficient evidence'")

        rid = f"{case_id}__r{cyc.number}"
        print(f"\n  == cycle {cyc.number} → {rid} ==")

        # Open this cycle's token window, then (if a window of 3 priors just completed) fold it into
        # the running summary FIRST so the high-tier summary cost lands in this cycle's token_usage.
        USAGE.reset()
        while len(prior_reports) >= summarized_upto + 3:
            window = prior_reports[summarized_upto:summarized_upto + 3]
            print(f"     folding rebuttal window [{summarized_upto + 1}-{summarized_upto + 3}] "
                  f"into running summary (high tier)…")
            try:
                running_summary = summarize_rebuttals(
                    running_summary, [r.synthesis for r in window]
                )
            except BudgetHookBlocked as exc:
                print(f"     {exc}; preserving the existing running summary", file=sys.stderr)
                break
            summarized_upto += 3

        # Build the prior-cycle chain BEFORE synthesizing so this cycle's synthesis actually
        # incorporates the earlier cycles' outcomes (verbatim tail + running summary). The operator
        # responses from prior rounds are read from this cycle's audit report inside build_report
        # (running document), so no operator-folder handling is needed here.
        previous_rebuttals = build_previous_rebuttals(prior_reports, running_summary)

        try:
            # Invoices are already scoped to this exception (works for flat folders too, where a
            # blind sibling-glob would wrongly pull every exception's PDFs), so pass them explicitly
            # and disable the loose auto-discovery.
            report = build_report(report, xlsx, exception, title, case_id,
                                  invoice_paths=(exc_invoices if auto_invoices else []),
                                  auto_invoices=False, reset_usage=False,
                                  previous_rebuttals=previous_rebuttals,
                                  year=year, asset=asset, category=category,
                                  skill_name=skill_name, use_skills=use_skills,
                                  report_text=report_text,
                                  auditor_description=auditor_description,
                                  latest_auditor_reply=cycle_auditor_reply,
                                  rebuttal_cycle=cyc.number,
                                  excel_parsing_enabled=excel_parsing_enabled,
                                  evidence_discovery=evidence.to_manifest(entity_dir))
        except SystemExit as exc:
            print(f"     SKIP cycle {cyc.number}: {exc}")
            continue
        except Exception as exc:  # a bad report format shouldn't kill the whole chain
            print(f"     SKIP cycle {cyc.number} ({type(exc).__name__}: {exc})")
            continue

        report.rebuttal_cycle = cyc.number
        report.rebuttal_id = rid

        rel = synthesis_rel_path(year, asset, exception, cyc.number)
        payload = report.model_dump(
            mode="json", exclude_none=False, exclude_unset=False, exclude_defaults=False,
        )
        metric_identity = cache_report_metrics(
            pipeline="extractor", data=payload, year=year, asset=asset,
            exception_id=exception, rebuttal_cycle=cyc.number, relative_path=rel,
            artifact_kind="synthesis",
        )
        try:
            location = sink.write(rel, payload)
        except Exception as exc:
            mark_metric_write(metric_identity, error=f"{type(exc).__name__}: {exc}")
            raise
        mark_metric_write(metric_identity, location=location)
        written.append(location)
        pr = report.previous_rebuttals
        chain = (f"summarized + {len(pr) - 1} verbatim") if "summarized_rebuttals" in pr \
            else f"{len(pr)} verbatim"
        print(f"     wrote {location}  (previous_rebuttals: {chain})")
        _print_result(report)

        prior_reports.append(report)

    print(f"\n=== processed {len(prior_reports)} cycle(s); wrote {len(written)} JSON(s)")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case-id", default="",
                    help="the case to run, '<Asset>_<year>_<exc>' e.g. GreatWhite_2024_0040. Needs no "
                         "CSV row: the entity folder is matched off disk and the topic comes from "
                         "datastores/clustered_exception_tables.csv. Padding-insensitive (40 == 0040)")
    ap.add_argument("--topic", default="",
                    help="exception topic/category, overriding the catalog lookup. Selects the "
                         "extraction skill and lands in the JSON as exception_category — pass it when "
                         "the catalog has no row for this case")
    ap.add_argument("--skill", default=None,
                    help="force a specific skill (resolves skills/skill_<name>.md); "
                         "default: auto-resolve from the topic")
    ap.add_argument("--no-skill", action="store_true",
                    help="disable skill resolution and run the plain pipeline")
    ap.add_argument("--provider", choices=("azure", "ica"), default=None,
                    help="LLM backend for this run (default: LLM_PROVIDER in .env)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the case and exit — no LLM calls, nothing written")
    ap.add_argument("--force", action="store_true",
                    help="re-run and overwrite synthesis output when this exception already has results")
    ap.add_argument("--llm-config", action="store_true",
                    help="print the resolved LLM configuration (provider, per-tier model/deployment/"
                         "endpoint/api-version, auth mode — never the key) and exit; makes no network "
                         "call, so it is the cheap way to check --provider azure before a batch")
    ap.add_argument("--smoke-test", action="store_true",
                    help="probe the active provider (LLM_PROVIDER: ica or azure) with one round-trip "
                         "per model tier and exit")
    ap.add_argument("--token-map", action="store_true",
                    help="ping the gateway, list its models and the fallback estimator encoding for "
                         "each (only used when the provider reports no usage), then exit")
    ap.add_argument("--run-id", default=None,
                    help="stable token-metrics run id; generated automatically when omitted")
    args = ap.parse_args()

    # Before any diagnostic block, so each reports the switched provider. LLM_PROVIDER is frozen at
    # import (shared.llm is already loaded via extractor.explain), so this needs the setter, not env.
    if args.provider:
        from shared.llm import set_provider
        print(f"# LLM_PROVIDER -> {set_provider(args.provider)}")

    if args.llm_config:
        from shared.llm import llm_config
        print(llm_config())
        return

    if args.smoke_test:
        # Config first: a failure below is nearly always a misconfiguration, and this way the reply
        # and the settings that produced it are in the same block of output.
        from shared.llm import llm_config, smoke_test
        print(llm_config(), end="\n\n")
        for tier, reply in smoke_test(hook_module=EXTRACTOR_HOOK_MODULE).items():
            print(f"[{tier:>5}] {reply!r}")
        return

    if args.token_map:
        from shared.llm import discover_models
        from shared.tokens import build_model_map
        models = discover_models()
        print(f"# {len(models)} models on the ICA gateway → fallback estimator encoding "
              "(--provider ica records the gateway's own usage instead)")
        for mid, enc in build_model_map(models).items():
            print(f"  {mid:<52} {enc}")
        return

    if not args.case_id:
        ap.error("--case-id is required, e.g. --case-id GreatWhite_2024_0040")

    # The id carries year/asset/exception, so this is the whole job spec: entity folder off disk,
    # topic from --topic or the master catalog.
    base = default_corpus_base()
    case = resolve_case_id(base, args.case_id, csv_path=None, exception_type=args.topic)
    if not case:
        sys.exit(f"could not resolve case id {args.case_id!r} under {base}")
    fallback_title = str(getattr(case, "exception_type", getattr(case, "title", "")) or "")
    metadata = _classified_case_metadata(
        case.entity_dir, case.exc_id, case.case_id, fallback_title,
    )
    effective_title = metadata.title if metadata else fallback_title
    resolved_year, resolved_asset = (
        (metadata.year, metadata.asset) if metadata else infer_year_asset(case.entity_dir)
    )
    display_case_id = metadata.case_id if metadata else case.case_id
    credit = metadata.gross_credit_requested if metadata else ""
    print(
        "# case metadata — "
        f"case_id={display_case_id}; year={resolved_year}; asset={resolved_asset}; "
        f"exception={case.exc_id}; category={effective_title or '(unclassified)'}; "
        f"credit={credit or '(not supplied)'}"
    )
    print(f"  source: {case.entity_dir}")
    if not effective_title:
        # The topic becomes the category downstream, which selects the extraction skill; blank means the
        # generic path, so say so rather than let it look like a normal run.
        print(f"WARN: no topic resolved for {case.case_id} — running with an empty category, so skill "
              f"selection falls back to the generic path. Pass --topic '<Exception Topic>' to set it.",
              file=sys.stderr)

    if args.dry_run:
        print(f"  {display_case_id} | exc {case.exc_id} | "
              f"{effective_title} -> {case.entity_dir}")
        return

    metric_run_id = ensure_run_id("extractor", args.run_id)
    print(f"# token-metrics run id: {metric_run_id}")
    parse_excel = extractor_excel_parsing_enabled()
    run_rebuttal_cycles(case.entity_dir, case.exc_id, effective_title,
                        display_case_id, make_sink(),
                        category=metadata.exception_category if metadata else fallback_title,
                        skill_name=args.skill,
                        use_skills=not args.no_skill,
                        excel_parsing_enabled=parse_excel,
                        year=metadata.year if metadata else "",
                        asset=metadata.asset if metadata else "",
                        auditor_description=metadata.auditor_description if metadata else None,
                        force=args.force)


if __name__ == "__main__":
    main()
