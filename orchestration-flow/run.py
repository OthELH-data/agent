"""OTL orchestration batch runner — drives OTL/affiliate/payroll labor exceptions through the OTL graph.

No-flags default: resolve every **Labor (OTL, Affiliates, Payroll)** row from
`datastores/labor_classification_test_set_temporal_split.csv` (the type
filter is hard-coded — this pipeline only handles this OTL labor category) and run each through the graph
in-process, one at a time. Each case gets a fresh token counter (`USAGE.reset()`), its own
checkpoint thread (`thread_id=case_id`), a live-tailable `rebuttal_<N>_otl_trace.jsonl`, and a
cycle-indexed `rebuttal_<N>_otl.json` report plus `rebuttal_<N>_analysis.md` agent memory. Classified
exceptions are the baseline input; extractor synthesis is an optional YAML-controlled local overlay.

Optional flags narrow the batch without changing the hard-coded type:
  --year 2023            only this year (comma-separated for several)
  --exc 0003             only this exception id (padding-insensitive: 3 == 0003)
  --case-id GreatWhite_2023_0003
                         needs NO CSV at all — the id is self-describing, so the entity folder is matched
                         off disk and the exception type pulled from the master catalog. The run CSV is skipped
                         entirely (so --csv/--year/--exc don't apply), and the OTL labor type filter
                         is BYPASSED with a warning, so a case of any exception type can be forced through.
  --exception-type "Labor (OTL, Affiliates, Payroll)" exception type/category for a case no CSV lists; overrides the catalog lookup
  --topic "Labor (OTL, Affiliates, Payroll)" legacy alias for --exception-type
  --limit N              first N resolved cases
  --dry-run              list resolved cases and exit
  --force                re-run cases whose OTL report already exists at the output target
  --csv NAME             a different cases CSV; a bare filename resolves under datastores/, and a missing
                         file warns and falls back to the default (the corpus root is not configurable)
  --provider P           LLM backend for this run: azure | ica (overrides LLM_PROVIDER in .env)
  --code-exec-tier T     set every code-exec specialist (jade_analyzer, location, job_title, matching,
                         supervision) to tier T (high|med|light); orchestrator + drafter are unaffected
  --max-people N         optionally cap how many disputed people each specialist examines
                         (omitted or 0 = no cap)
  --llm-config           print the resolved provider/tier/deployment config and the node→tier map, then
                         exit. No network call — check `--provider azure` here, not 40 minutes into a batch

Env: importing `otl` loads the repo-root `.env` (see `otl._bootstrap`); extractor modules load the
same file again on import, which is a no-op since load_dotenv doesn't override already-set values.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import otl  # noqa: F401  — bootstraps env + sys.path BEFORE any extractor import
from otl.case_resolver import resolve_otl_case
from otl.budget import invoke_with_budget_recovery
from otl.graph import build_otl_graph
from otl.input_sources import (classified_case_state, expand_rebuttal_cases, extractor_synthesis,
                               extractor_output_required, graph_input_state, limit_exceptions,
                               missing_extractor_synthesis_reason)
from otl.agents._common import max_disputed_people, set_max_disputed_people
from otl.llm_adapter import set_code_exec_tier
from otl.output_writer import (assemble_report, local_trace_path, make_otl_sink,
                               otl_report_rel_path, write_report, write_trace)
from otl.sharepoint_reader import SharePointReader
from otl.trace import TraceWriter

from shared.case_resolver import (Case, default_corpus_base, pad_exc,  # noqa: E402
                                     resolve_cases, resolve_cases_csv, select_case_id,
                                     warn_exception_type_mismatch)
from shared.output_paths import infer_year_asset  # noqa: E402
from shared.rebuttals import Cycle, discover_rebuttal_cycles
from shared.tokens import USAGE  # noqa: E402
from shared.rebuttal_history import rebuttal_analysis_exists
from shared.token_metrics_cache import (cache_report_metrics, ensure_run_id,
                                        mark_metric_write)

TYPE_FILTER = "Labor (OTL, Affiliates, Payroll)"
DEFAULT_OTL_CASES_CSV = "labor_classification_test_set_temporal_split.csv"


def run_case(case: Case, graph, reader: SharePointReader, sink, cycles: list[Cycle]) -> dict:
    """Run one exception end-to-end; return a compact result row (also writes the JSON + trace)."""
    year, asset = infer_year_asset(case.entity_dir)
    USAGE.reset()
    started = time.monotonic()

    classified, case_state = classified_case_state(case, year, asset)
    rebuttal_cycle = case.rebuttal_cycle

    # An enabled extractor input source is a cycle-specific precondition, not a best-effort overlay.
    # Return a structured skip before resolving evidence, creating traces, or invoking the graph.
    synthesis = extractor_synthesis(
        reader, year, asset, case.exc_id, rebuttal_cycle
    )
    if extractor_output_required() and not synthesis:
        return {
            "case_id": case.case_id, "exc_id": case.exc_id, "year": year, "asset": asset,
            "skipped": True,
            "skip_reason": missing_extractor_synthesis_reason(
                reader, year, asset, case.exc_id, rebuttal_cycle
            ),
            "elapsed_min": (time.monotonic() - started) / 60.0,
        }

    resolved = resolve_otl_case(
        case,
        cycles,
        evidence_manifest=(synthesis.get("evidence_discovery") or {}),
    )
    trace_path = local_trace_path(year, asset, case.exc_id, rebuttal_cycle)
    TraceWriter(trace_path).reset()

    init_state = graph_input_state(
        case_state=case_state, classified=classified, resolved=resolved, synthesis=synthesis,
        run_id=case.case_id, trace_path=str(trace_path), rebuttal_cycle=rebuttal_cycle,
    )

    # recursion_limit is the engine-level seatbelt only — the orchestrator's own hop/run-count guard
    # (MAX_HOPS/MAX_AGENT_RUNS, tripping well under 150) is the primary graceful terminator.
    graph_config = {"configurable": {
        "thread_id": f"{case.case_id}:rebuttal:{rebuttal_cycle}"
    }, "recursion_limit": 150}
    final = invoke_with_budget_recovery(graph, init_state, graph_config)
    elapsed_min = (time.monotonic() - started) / 60.0

    events = TraceWriter(trace_path).read_events()
    report = assemble_report(final, run_id=case.case_id, elapsed_minutes=elapsed_min,
                             token_usage=USAGE.report(), trace_events=events)
    rel = otl_report_rel_path(year, asset, case.exc_id, rebuttal_cycle)
    metric_identity = cache_report_metrics(
        pipeline="inference", data=report.model_dump(), year=year, asset=asset,
        exception_id=case.exc_id, rebuttal_cycle=rebuttal_cycle, relative_path=rel,
        artifact_kind="otl",
    )
    try:
        location = write_report(
            report, sink=sink, year=year, asset=asset, exc_id=case.exc_id, cycle=rebuttal_cycle,
        )
    except Exception as exc:
        mark_metric_write(metric_identity, error=f"{type(exc).__name__}: {exc}")
        raise
    mark_metric_write(metric_identity, location=location)
    trace_location = write_trace(sink, trace_path=trace_path, year=year, asset=asset,
                                 exc_id=case.exc_id, cycle=rebuttal_cycle)

    return {
        "case_id": case.case_id, "exc_id": case.exc_id, "year": year, "asset": asset,
        "disposition": report.disposition, "hitl": report.hitl_review,
        "hitl_stage": report.hitl_stage, "cfr_category": report.identified_cfr_category,
        "elapsed_min": elapsed_min, "location": location, "trace_path": str(trace_path),
        "trace_location": trace_location,
        "rebuttal_cycle": rebuttal_cycle,
        "rebuttal_analysis_path": final.get("rebuttal_analysis_path", ""),
        # None when no price map is configured, which is not the same as a run that cost nothing.
        "cost_usd": report.estimated_cost_usd,
        "currency": report.token_usage.cost.currency,
        "tokens": report.token_usage.total_tokens,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=f"Run OTL '{TYPE_FILTER}' exceptions through the graph.")
    ap.add_argument("--year", default=None, help="only this year, e.g. 2023 (comma-separate several)")
    ap.add_argument("--exc", default=None, help="only this exception id, e.g. 0003 (padding-insensitive)")
    ap.add_argument("--case-id", default=None,
                    help="run just this case, e.g. GreatWhite_2024_0040. With --csv, matching row-level "
                         "context (credit amount, classification, description, source) is retained; when "
                         "no row matches, the entity resolves off disk and the catalog is used as fallback.")
    ap.add_argument("--exception-type", "--topic", dest="exception_type", default=None,
                    help="exception type/category for a --case-id no CSV lists; overrides the catalog "
                         "lookup (--topic is a legacy alias)")
    ap.add_argument("--rebuttal-cycle", type=int, default=None, metavar="N",
                    help="run every available classified cycle through Rebuttal N (inclusive)")
    ap.add_argument("--limit", type=int, default=None,
                    help="run the first N unique exceptions; rebuttal cycles do not count toward N")
    ap.add_argument("--csv", default=None,
                    help="cases CSV: a bare filename resolves under datastores/ (default: "
                         "labor_classification_test_set_temporal_split.csv); a missing file warns and "
                         "falls back to the default. "
                         "Ignored entirely when --case-id is given.")
    ap.add_argument("--dry-run", action="store_true", help="list resolved cases and exit")
    ap.add_argument("--force", action="store_true",
                    help="re-run even cases whose OTL report already exists at the output target")
    ap.add_argument("--provider", choices=("azure", "ica"), default=None,
                    help="LLM backend for this run: azure | ica (default: LLM_PROVIDER in .env)")
    ap.add_argument("--code-exec-tier", choices=("high", "med", "light"), default=None,
                    help="set every code-exec specialist (jade_analyzer, location, job_title, matching, "
                         "supervision) to this tier; orchestrator + drafter stay on their own tiers")
    ap.add_argument("--max-people", type=int, default=None, metavar="N",
                    help="optionally cap disputed people each specialist examines; omitted or 0 or less "
                         "examines everyone (the budget hook remains the run stop condition)")
    ap.add_argument("--llm-config", action="store_true",
                    help="print the resolved LLM configuration (provider, per-tier model/deployment/"
                         "endpoint/api-version, auth mode — never the key) plus the node->tier map, then "
                         "exit; makes no network call, so it is the cheap pre-flight for --provider azure")
    ap.add_argument("--run-id", default=None,
                    help="stable token-metrics run id; generated automatically when omitted")
    args = ap.parse_args()

    if args.provider:
        # First, so both --code-exec-tier's map and --llm-config below report the switched provider.
        # `shared.llm` is already imported (via otl.llm_adapter), so the env var alone is too late —
        # set_provider reassigns the module globals and re-exports LLM_PROVIDER.
        from shared.llm import set_provider  # noqa: E402
        print(f"# LLM_PROVIDER -> {set_provider(args.provider)}")

    if args.code_exec_tier:
        set_code_exec_tier(args.code_exec_tier)
        print(f"# code-exec specialists -> tier '{args.code_exec_tier}'")

    if args.max_people is not None:
        set_max_disputed_people(args.max_people)

    if args.llm_config:
        # Placed after --code-exec-tier so the map printed is the one this run would actually use.
        from shared.llm import llm_config  # noqa: E402
        from otl.llm_adapter import node_tier_map
        print(llm_config())
        print("# OTL node -> tier")
        for node, tier in node_tier_map().items():
            print(f"  {node:<24} {tier}")
        return

    base = default_corpus_base()
    csv_path = resolve_cases_csv(args.csv or DEFAULT_OTL_CASES_CSV)
    max_rebuttal_cycles = None
    if args.case_id:
        # Resolve the CSV first so singular runs receive the same context as a batch worker. A case
        # absent from that CSV still falls back to the self-sufficient case-id resolver.
        cases = select_case_id(resolve_cases(base, csv_path), args.case_id, base,
                               csv_path=csv_path, exception_type=args.exception_type)
        if args.exception_type:
            cases = [replace(c, exception_type=args.exception_type) for c in cases]
        warn_exception_type_mismatch(cases, TYPE_FILTER)
    else:
        cases = resolve_cases(base, csv_path,
                              type_filter=TYPE_FILTER, year_filter=args.year)
        if args.exc:
            cases = [c for c in cases if pad_exc(c.exc_id) == pad_exc(args.exc)]
    if args.rebuttal_cycle is not None:
        max_rebuttal_cycles = max(0, args.rebuttal_cycle)
    cases = limit_exceptions(cases, args.limit)
    cases = expand_rebuttal_cases(cases, max_rebuttal_cycles)

    cycles_by_entity = {}
    for case in cases:
        entity_dir = Path(case.entity_dir)
        if entity_dir not in cycles_by_entity:
            cycles_by_entity[entity_dir] = discover_rebuttal_cycles(entity_dir)

    cap = max_disputed_people()
    print(f"# OTL runner — type={TYPE_FILTER!r}; {len({c.case_id for c in cases})} unique exception(s), "
          f"{len(cases)} rebuttal cycle run(s) resolved; "
          f"people examined per case: {'no cap' if cap <= 0 else cap}")
    for c in cases:
        cycle = f"rebuttal {c.rebuttal_cycle}" if c.rebuttal_cycle is not None else "latest rebuttal"
        print(f"  - {c.case_id}  exc {c.exc_id}  ({c.exception_type}); {cycle}  <- {c.entity_dir}")
    if args.dry_run:
        return
    if not cases:
        print("# nothing to run.")
        return

    metric_run_id = ensure_run_id("inference", args.run_id)
    print(f"# token-metrics run id: {metric_run_id}")

    reader = SharePointReader()
    sink = make_otl_sink()
    graph = build_otl_graph()

    batch_start = time.monotonic()
    results: list[dict] = []
    skipped_existing = 0
    skipped_missing_synthesis = 0
    for i, case in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] {case.case_id} — exc {case.exc_id} ({case.exception_type})")
        if not args.force:
            year, asset = infer_year_asset(case.entity_dir)
            staged = extractor_synthesis(
                reader, year, asset, case.exc_id, case.rebuttal_cycle
            )
            if extractor_output_required() and not staged:
                print(
                    "   -> SKIPPED: "
                    f"{missing_extractor_synthesis_reason(reader, year, asset, case.exc_id, case.rebuttal_cycle)}"
                )
                skipped_missing_synthesis += 1
                continue
            rel = otl_report_rel_path(year, asset, case.exc_id, case.rebuttal_cycle)
            history_case = {"year": year, "asset": asset, "exc_id": case.exc_id}
            if sink.exists(rel) and rebuttal_analysis_exists(history_case, case.rebuttal_cycle):
                print(f"   -> already written ({rel}); skipping. Use --force to re-run.")
                skipped_existing += 1
                continue
        try:
            res = run_case(case, graph, reader, sink, cycles_by_entity[Path(case.entity_dir)])
            if res.get("skipped"):
                print(f"   -> SKIPPED: {res['skip_reason']}")
                skipped_missing_synthesis += 1
                continue
            tag = (f"HITL@{res['hitl_stage']}" if res["hitl"] else f"disposition={res['disposition']}")
            cat = f", cfr={res['cfr_category']}" if res["cfr_category"] else ""
            spend = (f", {res['cost_usd']:.4f} {res['currency']}"
                     if res["cost_usd"] is not None else "")
            print(f"   -> {tag}{cat}  ({res['elapsed_min']:.2f} min, "
                  f"{res['tokens']:,} tokens{spend})  ->  {res['location']}")
            results.append(res)
        except Exception:  # one bad case shouldn't abort the batch
            print(f"   !! FAILED:\n{traceback.format_exc()}", file=sys.stderr)

    total_min = (time.monotonic() - batch_start) / 60.0
    ok = len(results)
    # Unsupported auditor evidence is a terminal denial. `needs_evidence` is reserved for HITL/deferred
    # cases, so every completed case belongs in exactly one of these two buckets.
    hitl_n = sum(1 for r in results if r["hitl"])
    term_n = ok - hitl_n
    skip_notes = []
    if skipped_existing:
        skip_notes.append(f"{skipped_existing} skipped (already written)")
    if skipped_missing_synthesis:
        skip_notes.append(f"{skipped_missing_synthesis} skipped (missing required extractor synthesis)")
    skip_note = f", {'; '.join(skip_notes)}" if skip_notes else ""
    print(f"\n# Done: {ok}/{len(cases)} case(s) in {total_min:.2f} min "
          f"({term_n} terminal, {hitl_n} HITL{skip_note}).")
    # Priced cases only. An unpriced case contributes no dollars, so it is counted out loud rather
    # than folded into a total that would look complete.
    priced = [r for r in results if r["cost_usd"] is not None]
    if results:
        spend = (f", {sum(r['cost_usd'] for r in priced):.4f} {priced[0]['currency']}"
                 f" over {len(priced)} priced case(s)" if priced else "")
        print(f"#   {sum(r['tokens'] for r in results):,} tokens{spend}.")


if __name__ == "__main__":
    main()
