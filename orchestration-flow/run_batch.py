#!/usr/bin/env python3
"""Parallel OTL batch runner: run many OTL/affiliate/payroll labor exceptions concurrently, one process per case.

Each exception is independent (its own folder/synthesis/report), so cases run concurrently for a large
speedup on the LLM-latency-bound graph. Parallelism is at the PROCESS level, not thread: each case is a
separate `run.py --case-id <id> --force` subprocess with its own `USAGE` token counter, checkpoint
thread, and output sink — so nothing shared is mutated and per-exception `token_usage` stays correct.
(Thread-level parallelism is unsafe here: `shared.tokens.USAGE` is a process-global counter that
`run.py` resets/reports per case.) The driver only supervises those subprocesses; its worker threads
just block on `subprocess.run`.

The parent owns the skip decision (checks the output sink once) so already-written cases don't pay a
subprocess-startup cost; `--force` re-runs them. The subprocess is always passed `--force` because the
parent has already decided it should run.

`--case-id` needs no CSV row: the id is self-describing, so the entity folder is matched off disk and
the topic pulled from the master catalog (`--exception-type` overrides). An explicit `--case-id` also bypasses
the topic filter, with a warning naming the actual topic.

Without `--type` the batch runs every case in the CSV whatever its topic. Pass
`--type "Labor (OTL, Affiliates, Payroll)"`
to keep to the topic this pipeline's prompts and skills were written for.

Usage:
    ../.venv/bin/python3 run_batch.py [--jobs N] [--year Y] [--exc ID] [--dry-run] [--force]
    ../.venv/bin/python3 run_batch.py --type "Labor (OTL, Affiliates, Payroll)"  # only that topic
    ../.venv/bin/python3 run_batch.py --case-id GreatWhite_2024_0040
    ../.venv/bin/python3 run_batch.py --provider azure --llm-config    # pre-flight, no case run
    ../.venv/bin/python3 run_batch.py --max-people 3   # optionally cap each case at 3 people
    OUTPUT_TARGET=local ../.venv/bin/python3 run_batch.py --jobs 4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import zipfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import otl  # noqa: F401  — bootstraps env + sys.path BEFORE any extractor import (mirrors run.py)
from otl.agents._common import max_disputed_people, set_max_disputed_people
from shared.classified_exceptions import find_classified_exception
from otl.input_sources import expand_rebuttal_cases, limit_exceptions
from otl.output_writer import make_otl_sink, otl_report_rel_path

from shared.case_resolver import (Case, default_corpus_base, pad_exc,  # noqa: E402
                                     resolve_cases, resolve_cases_csv, select_case_id, slugify,
                                     warn_exception_type_mismatch)
from shared.output_paths import infer_year_asset  # noqa: E402
from shared.rebuttal_history import rebuttal_analysis_exists
from shared.token_metrics_cache import ensure_run_id

HERE = Path(__file__).resolve().parent
TYPE_FILTER = "Labor (OTL, Affiliates, Payroll)"  # the topic this pipeline's prompts/skills target
DEFAULT_OTL_CASES_CSV = "labor_classification_test_set_temporal_split.csv"
# How many name-selection calls this whole BATCH may have in flight at once (otl/person_index.py chunks
# a workbook's full name list and calls the med tier once per chunk). The pool that issues them lives
# inside each `run.py` child, so its `max_workers` bounds one case and nothing more — see
# `_divide_name_select_budget`.
NAME_SELECT_BUDGET = 4


def _extract_sibling_zips(entity_dirs: list[Path]) -> None:
    """Idempotently extract any `Exception-*.zip` under each entity dir (mirrors `unzip -n`).

    Done serially in the parent — several cases can share one entity dir (and its zips), so extracting
    inside concurrent workers would race on the same archive.
    """
    for d in entity_dirs:
        for zpath in sorted(d.rglob("Exception-*.zip")):
            try:
                with zipfile.ZipFile(zpath) as zf:
                    dest = zpath.parent
                    for member in zf.namelist():
                        target = dest / member
                        if not target.exists():  # -n: never overwrite
                            zf.extract(member, dest)
            except Exception as exc:  # a bad zip shouldn't abort the batch
                print(f"  WARN: could not extract {zpath.name}: {exc}", file=sys.stderr)


def _divide_name_select_budget(jobs: int) -> str | None:
    """Split `NAME_SELECT_BUDGET` across `jobs` concurrent cases, in the env the children inherit.

    Each case is its own `run.py` subprocess (see `_run_case`), so the ThreadPoolExecutor inside
    `otl/person_index.py` caps one case and knows nothing about the other three: four children each
    running a pool of 4 puts 16 calls in flight, every one carrying ~10k tokens of names against a single
    deployment's quota. No in-process object can bound that, because no object crosses the fork.

    The environment does cross it — `_run_case` passes `env=os.environ` — the same reason `--max-people`
    had to travel on the command line instead. So the parent divides the budget here and each child reads
    its share from `OTL_NAME_SELECT_JOBS`. `--jobs 4` gives each child 1; `--jobs 1` gives it the full 4;
    the batch-wide total is the budget either way. An operator who set the variable themselves keeps it.

    Returns the value applied, or None when the operator had already set one.
    """
    if os.getenv("OTL_NAME_SELECT_JOBS"):
        return None
    share = str(max(1, NAME_SELECT_BUDGET // max(1, jobs)))
    os.environ["OTL_NAME_SELECT_JOBS"] = share
    return share


def _run_case(case: Case, index: int, total: int, csv_path: Path,
              code_exec_tier: str | None = None, max_people: int | None = None) -> dict:
    """Run one exception as an isolated `run.py` subprocess; capture its output and status.

    The resolved CSV path follows the case into the child. This preserves temporal-split row context
    (credit amount, classification, description, and provenance) in addition to its structural ID.
    """
    cmd = [sys.executable, "run.py", "--case-id", case.case_id, "--csv", str(csv_path), "--force"]
    if case.rebuttal_cycle is not None:
        cmd += ["--rebuttal-cycle", str(case.rebuttal_cycle)]
    # Hand the child the topic the parent already resolved, so a case absent from the CSV doesn't
    # repeat the catalog lookup (and any --exception-type override reaches the worker).
    if case.exception_type:
        cmd += ["--exception-type", case.exception_type]
    if code_exec_tier:
        cmd += ["--code-exec-tier", code_exec_tier]
    if max_people is not None:
        # Forwarded on the command line, not via env: the cap is a process global in `_common`, so it
        # does not survive the fork into this worker the way LLM_PROVIDER (an env var) does.
        cmd += ["--max-people", str(max_people)]
    start = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=str(HERE), env=os.environ,
                              capture_output=True, text=True)
        out, rc = proc.stdout + proc.stderr, proc.returncode
    except Exception as exc:
        out, rc = f"{type(exc).__name__}: {exc}", 1
    return {"case": case, "index": index, "total": total, "rc": rc,
            "output": out, "elapsed": time.monotonic() - start}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run OTL exceptions in parallel, one process per case. Runs every case in the "
                    "cases CSV unless --type narrows it to one exception topic.")
    ap.add_argument("--jobs", type=int, default=4, help="max concurrent cases (default: 4)")
    ap.add_argument("--year", default=None, help="only this year, e.g. 2023 (comma-separate several)")
    ap.add_argument("--exc", default=None, help="only this exception id, e.g. 0003 (padding-insensitive)")
    ap.add_argument("--type", default=None, metavar="TOPIC",
                    help="only cases whose exception topic matches this, compared as a slug so spacing "
                         f"and case don't matter (e.g. {TYPE_FILTER!r}). Omit to run every case in the "
                         "CSV whatever its topic")
    ap.add_argument("--case-id", default=None,
                    help="run just this case, e.g. GreatWhite_2024_0040 — no CSV row needed (resolved "
                         "off disk) and any --type filter is bypassed with a warning")
    ap.add_argument("--exception-type", "--topic", dest="exception_type", default=None,
                    help="exception type/category for a --case-id no CSV lists; overrides the catalog "
                         "lookup (--topic is a legacy alias)")
    ap.add_argument("--rebuttal-cycle", type=int, default=None, metavar="N",
                    help="run every available classified cycle through Rebuttal N (inclusive)")
    ap.add_argument("--limit", type=int, default=0,
                    help="run first N unique exceptions; rebuttal cycles do not count (0 = all)")
    ap.add_argument("--csv", default=None,
                    help="cases CSV: a bare filename resolves under datastores/ (default: "
                         "labor_classification_test_set_temporal_split.csv); a missing file warns and "
                         "falls back to the default")
    ap.add_argument("--dry-run", action="store_true", help="list the resolved cases and exit")
    ap.add_argument("--force", action="store_true",
                    help="re-run even cases whose OTL report already exists at the output target")
    ap.add_argument("--provider", choices=("azure", "ica"), default=None,
                    help="LLM backend for this batch: azure | ica (default: LLM_PROVIDER in .env). Set "
                         "here in the parent, so every case subprocess inherits it via env.")
    ap.add_argument("--code-exec-tier", choices=("high", "med", "light"), default=None,
                    help="set every code-exec specialist (jade_analyzer, location, job_title, matching, "
                         "supervision) to this tier for all cases; forwarded to each run.py worker")
    ap.add_argument("--max-people", type=int, default=None, metavar="N",
                    help="optionally cap disputed people each specialist examines; omitted or 0 or less "
                         "examines everyone; forwarded to each run.py worker")
    ap.add_argument("--llm-config", action="store_true",
                    help="print the resolved LLM configuration + node->tier map and exit (no network "
                         "call, no case run) — the pre-flight for ./run_otl_cases.sh --provider azure")
    ap.add_argument("--run-id", default=None,
                    help="token-metrics run id shared by every case subprocess (generated when omitted)")
    args = ap.parse_args()

    if args.provider:
        # Before --llm-config so it reports the switched provider. set_provider also writes
        # os.environ["LLM_PROVIDER"], which every worker subprocess inherits (env=os.environ).
        from shared.llm import set_provider  # noqa: E402
        print(f"# LLM_PROVIDER -> {set_provider(args.provider)}")

    if args.max_people is not None:
        # This parent runs no case itself; applying the flag here only makes the header line below report
        # the value the workers will use (each gets it on its own command line, see `_run_case`).
        set_max_disputed_people(args.max_people)

    if args.llm_config:
        # Answered here rather than forwarded: every worker inherits this process's env, so the config
        # this parent resolves is the config the whole batch would run under.
        from shared.llm import llm_config  # noqa: E402
        from otl.llm_adapter import node_tier_map, set_code_exec_tier
        if args.code_exec_tier:
            # This parent normally just forwards the flag to each worker; apply it locally so the map
            # printed matches what the workers will build.
            set_code_exec_tier(args.code_exec_tier)
        print(llm_config())
        print("# OTL node -> tier")
        for node, tier in node_tier_map().items():
            print(f"  {node:<24} {tier}")
        return

    # Resolve, then filter: topic (only when --type is given), then year/exc/case-id/limit.
    base, csv_path = default_corpus_base(), resolve_cases_csv(args.csv or DEFAULT_OTL_CASES_CSV)
    cases = resolve_cases(base, csv_path, type_filter=args.type, year_filter=args.year)
    max_rebuttal_cycles = None
    if args.exc:
        cases = [c for c in cases if pad_exc(c.exc_id) == pad_exc(args.exc)]
    if args.case_id:
        # Self-sufficient: falls back to resolving the id off disk when the CSV never listed it (or its
        # type/year filters dropped it), then says so loudly if the topic is outside this pipeline.
        cases = select_case_id(cases, args.case_id, base, csv_path=csv_path,
                               exception_type=args.exception_type)
        warn_exception_type_mismatch(cases, TYPE_FILTER)
    if args.rebuttal_cycle is not None:
        max_rebuttal_cycles = max(0, args.rebuttal_cycle)
    cases = limit_exceptions(cases, args.limit)
    cases = expand_rebuttal_cases(cases, max_rebuttal_cycles)

    if args.year:
        print(f"# filtering to year: {args.year}")
    if args.code_exec_tier:
        print(f"# code-exec specialists -> tier '{args.code_exec_tier}' (all cases)")
    cap = max_disputed_people()
    type_note = repr(args.type) if args.type else "all topics"
    unique_count = len({case.case_id for case in cases})
    print(f"# OTL batch — type={type_note}; {unique_count} unique exception(s), "
          f"{len(cases)} rebuttal cycle run(s) resolved from {csv_path}; "
          f"people examined per case: {'no cap' if cap <= 0 else cap}")
    if not args.type:
        # Running every topic is now the default, so say how many cases fall outside the topic these
        # prompts/skills were written for — those results are expected to be weaker.
        native = slugify(TYPE_FILTER)
        off = [c for c in cases if slugify(c.exception_type) != native]
        if off:
            print(f"# note: {len(off)} of {len(cases)} case(s) are not {TYPE_FILTER!r} — this pipeline's "
                  f"prompts are built for that topic, so expect degraded results. Pass "
                  f"--type '{TYPE_FILTER}' to run only that topic.")

    if args.dry_run:
        for c in cases:
            cycle = f"rebuttal {c.rebuttal_cycle}" if c.rebuttal_cycle is not None else "latest rebuttal"
            print(f"  {c.case_id} | exc {c.exc_id} | {cycle} | {c.exception_type} -> {c.entity_dir}")
        return
    if not cases:
        print("# nothing to run.")
        return

    metric_run_id = ensure_run_id("inference", args.run_id)
    print(f"# token-metrics run id: {metric_run_id}")

    # Pre-flight (serial, in the parent): extract sibling zips once; build the sink once (this also
    # warms SharePoint auth so workers reuse the cached token) and let the parent own the skip decision.
    _extract_sibling_zips(sorted({c.entity_dir for c in cases}))
    sink = make_otl_sink()
    if (os.getenv("OUTPUT_TARGET") or "local").strip().lower() == "sharepoint":
        print("# SharePoint sink built (token cache populated for workers)")

    to_run: list[Case] = []
    skipped = 0
    for c in cases:
        if not args.force:
            year, asset = infer_year_asset(c.entity_dir)
            classified = find_classified_exception(year, asset, c.exc_id, c.rebuttal_cycle) or {}
            cycle = c.rebuttal_cycle if c.rebuttal_cycle is not None else int(
                classified.get("rebuttal_cycle") or 0
            )
            history_case = {"year": year, "asset": asset, "exc_id": c.exc_id}
            if sink.exists(otl_report_rel_path(year, asset, c.exc_id, cycle)) and rebuttal_analysis_exists(
                history_case, cycle
            ):
                print(f"  - skip {c.case_id} rebuttal {cycle} "
                      "(report already written; use --force to re-run)")
                skipped += 1
                continue
        to_run.append(c)

    if not to_run:
        print(f"# nothing to run ({skipped} already written).")
        return

    jobs = max(1, args.jobs)
    share = _divide_name_select_budget(jobs)
    if share:
        print(f"# name-selection calls: {share} concurrent per case "
              f"({NAME_SELECT_BUDGET} across the batch)")
    print(f"# running {len(to_run)} rebuttal cycle(s) across "
          f"{len({case.case_id for case in to_run})} exception(s), up to {jobs} exceptions at a time"
          f"{f' ({skipped} skipped)' if skipped else ''}\n")
    started = time.monotonic()
    ok, failed = 0, []
    done = 0

    groups: OrderedDict[str, list[Case]] = OrderedDict()
    for case in to_run:
        groups.setdefault(case.case_id, []).append(case)

    def _run_sequence(sequence: list[Case]) -> list[dict]:
        results: list[dict] = []
        for case in sequence:
            result = _run_case(case, 0, len(to_run), csv_path, args.code_exec_tier, args.max_people)
            results.append(result)
            if result["rc"] != 0:
                break
        return results

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_run_sequence, sequence) for sequence in groups.values()]
        for fut in as_completed(futures):
            for r in fut.result():
                done += 1
                c = r["case"]
                status = "ok" if r["rc"] == 0 else "FAILED"
                if r["rc"] == 0:
                    ok += 1
                else:
                    failed.append(f"{c.case_id}/rebuttal_{c.rebuttal_cycle}")
                print(f"=== [{done}/{len(to_run)}] {c.case_id} — rebuttal {c.rebuttal_cycle} — "
                      f"exc {c.exc_id} ({c.exception_type}) — {status} in {r['elapsed']:.0f}s ===")
                print(r["output"].rstrip())
                print()

    elapsed = time.monotonic() - started
    skip_note = f", {skipped} skipped (already written)" if skipped else ""
    print(f"=== ALL DONE: {ok} ok, {len(failed)} failed, {len(to_run)} run in {elapsed:.0f}s "
          f"({jobs} concurrent){skip_note} ===")
    if failed:
        print("   failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
