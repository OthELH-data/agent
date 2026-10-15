#!/usr/bin/env python3
"""Parallel batch runner: run many exceptions concurrently, each in its own isolated process.

Each exception is independent (its own folder/files), so cases run concurrently for a large speedup
on the LLM-latency-bound pipeline. Parallelism is at the PROCESS level, not thread: each case is a
separate `run.py` subprocess with its own `USAGE` token counter and output sink — so nothing shared
is mutated and per-exception `token_usage` stays correct. The driver here only supervises those
subprocesses (its worker threads just block on `subprocess.run`).

`--case-id` needs no CSV row: the id is self-describing (`<Asset>_<year>_<exc>`), so the entity folder is
matched off disk and the topic pulled from the master catalog (`--topic` overrides). It also bypasses
`--type`, since naming one case outright is a deliberate override.

Usage:
    ../.venv/bin/python3 run_batch.py [--type <slug>] [--jobs N] [--dry-run]
    ../.venv/bin/python3 run_batch.py --case-id GreatWhite_2024_0040
    ../.venv/bin/python3 run_batch.py --provider azure --llm-config    # pre-flight, no case run
    OUTPUT_TARGET=local ../.venv/bin/python3 run_batch.py --type material_reconciliation --jobs 4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_EXTRACTOR_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXTRACTOR_DIR.parent
for _path in (str(_EXTRACTOR_DIR), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from shared.case_resolver import (Case, default_corpus_base, pad_exc,  # noqa: E402
                                     resolve_cases, resolve_cases_csv, select_case_id,
                                     warn_exception_type_mismatch)
from shared.token_metrics_cache import ensure_run_id  # noqa: E402

HERE = Path(__file__).resolve().parent


def _case_title(case) -> str:
    """Compatibility view of a shared ``Case`` topic for command/display use."""
    return str(getattr(case, "exception_type", getattr(case, "title", "")) or "")


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


def _warm_auth() -> None:
    """If output goes to SharePoint, authenticate once in the parent so workers reuse the token cache."""
    if (os.getenv("OUTPUT_TARGET") or "local").strip().lower() != "sharepoint":
        return
    try:
        from shared.output_paths import make_sink
        make_sink()  # triggers device-code auth once here; workers then read the cached token silently
        print("# SharePoint auth warmed (token cache populated for workers)")
    except Exception as exc:
        print(f"# WARN: SharePoint auth warm-up failed ({exc}); workers will each attempt auth",
              file=sys.stderr)


def _run_case(case: Case, index: int, total: int, *, no_skill: bool = False,
              stream_output: bool = False, force: bool = False) -> dict:
    """Run one exception as an isolated `run.py` subprocess.

    The case id is the whole contract: it carries asset/year/exception, so the child re-resolves the
    entity folder off disk itself. Only the topic has to travel — and passing it means the child never
    has to read the master catalog, so a CSV row the catalog doesn't list still runs.

    A single case (or serial batch) streams the child directly to the terminal. Parallel batches keep
    each child buffered so their logs remain contiguous rather than interleaved.
    """
    cmd = [sys.executable, "run.py", "--case-id", case.case_id]
    if force:
        cmd += ["--force"]
    if _case_title(case):
        cmd += ["--topic", _case_title(case)]
    if no_skill:
        cmd += ["--no-skill"]
    start = time.monotonic()
    try:
        if stream_output:
            print(f"=== START [{index}/{total}] {case.case_id} — exc {case.exc_id} "
                  f"({_case_title(case)}) ===", flush=True)
            proc = subprocess.run(cmd, cwd=str(HERE), env=os.environ)
            out, rc = "", proc.returncode
        else:
            proc = subprocess.run(cmd, cwd=str(HERE), env=os.environ,
                                  capture_output=True, text=True)
            out, rc = proc.stdout + proc.stderr, proc.returncode
    except Exception as exc:
        out, rc = f"{type(exc).__name__}: {exc}", 1
    return {"case": case, "index": index, "total": total, "rc": rc,
            "output": out, "elapsed": time.monotonic() - start}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run batch exceptions in parallel, one process per case.")
    ap.add_argument("--jobs", type=int, default=4, help="max concurrent cases (default: 4)")
    ap.add_argument("--type", default=None, help="only run cases whose topic slugifies to this "
                                                 "(e.g. material_reconciliation)")
    ap.add_argument("--year", default=None, help="only run cases from this year, e.g. 2024 "
                                                 "(comma-separate for several, e.g. 2023,2024)")
    ap.add_argument("--exc", default=None, help="only this exception id, e.g. 0040 (padding-insensitive)")
    ap.add_argument("--case-id", default=None,
                    help="run just this case, e.g. GreatWhite_2024_0040 — no CSV row needed (the entity "
                         "folder is resolved off disk) and --type is bypassed with a warning")
    ap.add_argument("--topic", default=None,
                    help="topic/category for a --case-id no CSV lists; overrides the catalog lookup")
    ap.add_argument("--csv", default=None,
                    help="cases CSV: a bare filename resolves under datastores/ (default: "
                         "test_cases_46.csv); a missing file warns and falls back to the default")
    ap.add_argument("--no-skill", action="store_true",
                    help="disable the skills layer for every case (forwarded to each run.py)")
    ap.add_argument("--dry-run", action="store_true", help="list the resolved cases and exit")
    ap.add_argument("--force", action="store_true",
                    help="re-run cases even when the output target already has synthesis JSONs")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N resolved cases (0 = all)")
    ap.add_argument("--provider", choices=("azure", "ica"), default=None,
                    help="LLM backend for this batch: azure | ica (default: LLM_PROVIDER in .env). Set "
                         "here in the parent, so every case subprocess inherits it via env.")
    ap.add_argument("--llm-config", action="store_true",
                    help="print the resolved LLM configuration and exit (no network call, no case run) — "
                         "the pre-flight for --provider azure")
    ap.add_argument("--run-id", default=None,
                    help="token-metrics run id shared by every case subprocess (generated when omitted)")
    args = ap.parse_args()

    if args.provider:
        # Before --llm-config so it reports the switched provider. set_provider also writes
        # os.environ["LLM_PROVIDER"], which every worker subprocess inherits (env=os.environ).
        from shared.llm import set_provider
        print(f"# LLM_PROVIDER -> {set_provider(args.provider)}")

    if args.llm_config:
        # Answered in the parent: every worker subprocess inherits this env, so what resolves here is
        # what the whole batch would run under.
        from shared.llm import llm_config
        print(llm_config())
        return

    base, csv_path = default_corpus_base(), resolve_cases_csv(args.csv)

    cases = resolve_cases(base, csv_path, args.type, args.year)
    if args.exc:
        cases = [c for c in cases if pad_exc(c.exc_id) == pad_exc(args.exc)]
    if args.case_id:
        # Self-sufficient: falls back to resolving the id off disk when the CSV never listed it (or its
        # type/year filters dropped it).
        cases = select_case_id(cases, args.case_id, base, csv_path=csv_path,
                               exception_type=args.topic)
        if args.type:
            warn_exception_type_mismatch(cases, args.type)
        for c in cases:
            if not _case_title(c):
                # `title` becomes the child's `--topic`, which selects the extraction skill; blank means
                # the generic path, so say so rather than let it look like a normal run.
                print(f"WARN: no topic resolved for {c.case_id} — running with no topic, so skill "
                      f"selection falls back to the generic path. Pass --topic '<Exception Topic>' "
                      f"to set it.", file=sys.stderr)
    if args.limit > 0:
        cases = cases[: args.limit]
    if args.type:
        print(f"# filtering to exception type: {args.type}")
    if args.year:
        print(f"# filtering to year: {args.year}")
    print(f"# {len(cases)} resolvable case(s) from {csv_path}")

    if args.dry_run:
        for c in cases:
            print(f"  {c.case_id} | exc {c.exc_id} | {_case_title(c)} -> {c.entity_dir}")
        return
    if not cases:
        return

    metric_run_id = ensure_run_id("extractor", args.run_id)
    print(f"# token-metrics run id: {metric_run_id}")

    # Pre-flight (serial, in the parent): extract sibling zips once, warm SharePoint auth once.
    _extract_sibling_zips(sorted({c.entity_dir for c in cases}))
    _warm_auth()

    jobs = max(1, args.jobs)
    stream_output = jobs == 1 or len(cases) == 1
    if stream_output:
        print("# live worker output enabled")
    print(f"# running {len(cases)} case(s), up to {jobs} at a time\n")
    started = time.monotonic()
    ok, failed = 0, []
    done = 0

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_run_case, c, i + 1, len(cases), no_skill=args.no_skill,
                               stream_output=stream_output, force=args.force)
                   for i, c in enumerate(cases)]
        for fut in as_completed(futures):
            r = fut.result()
            done += 1
            c = r["case"]
            status = "ok" if r["rc"] == 0 else "FAILED"
            if r["rc"] == 0:
                ok += 1
            else:
                failed.append(c.case_id)
            # Captured parallel-worker output stays contiguous; a live worker already wrote its own log.
            print(f"=== [{done}/{len(cases)}] {c.case_id} — exc {c.exc_id} ({_case_title(c)}) — "
                  f"{status} in {r['elapsed']:.0f}s ===")
            if r["output"]:
                print(r["output"].rstrip())
            print()

    elapsed = time.monotonic() - started
    print(f"=== ALL DONE: {ok} ok, {len(failed)} failed, {len(cases)} total in {elapsed:.0f}s "
          f"({jobs} concurrent) ===")
    if failed:
        print("   failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
