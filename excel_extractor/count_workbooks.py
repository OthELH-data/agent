#!/usr/bin/env python3
"""Count how many Excel workbooks were analyzed across all test-case synthesis JSONs.

Each synthesis JSON (rebuttal_<n>_synthesis.json) carries an `attachments` list — one entry per
Excel workbook the pipeline read and interpreted for that rebuttal cycle. So the workbooks analyzed
for a case = the sum of `len(attachments)` over ALL of that case's cycle JSONs (rebuttal_0,
rebuttal_1, …). This script resolves every case in the batch CSV exactly the way the pipeline does
(case_resolver + infer_year_asset → `<year>/<asset>/<exc>/`), enumerates each case's cycle JSONs
from the configured backend, and tallies the workbook count.

Backend follows OUTPUT_TARGET in .env (sharepoint by default here), same as the extractor:
  * sharepoint → reads Run folder on the SHAREPOINT_URL site (device-code login on first use)
  * local      → walks the OUTPUT_BASE_EXTRACTOR tree
Force local with --local.

Usage:
    ../.venv/bin/python3 count_workbooks.py                 # all cases, backend from .env
    ../.venv/bin/python3 count_workbooks.py --local         # read the local extractor-output tree
    ../.venv/bin/python3 count_workbooks.py --by-case        # add a per-case breakdown table
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for path in (str(REPO_ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)
from shared.case_resolver import Case, resolve_cases  # noqa: E402
from shared.output_paths import extractor_output_base, infer_year_asset  # noqa: E402

DEFAULT_BASE = ("/Users/harshakolachina/Documents/Projects/Shell JV Audit/"
                "Data Files/Exception Reports and Rebuttals by year")
DEFAULT_CSV = str(HERE.parent / "datastores" / "test_cases_46.csv")


# --- backend readers: each maps a case's `<year>/<asset>/<exc>/` folder to its cycle JSONs --------

class LocalReader:
    """Read cycle synthesis JSONs from the local extractor-output tree."""

    def __init__(self, base: Path | None = None):
        self.base = base or extractor_output_base()
        print(f"# reading local synthesis JSONs under: {self.base}")

    def cycle_jsons(self, rel_folder: str) -> list[tuple[str, dict]]:
        folder = self.base / rel_folder
        if not folder.is_dir():
            return []
        out: list[tuple[str, dict]] = []
        for fp in sorted(folder.glob("rebuttal_*_synthesis.json")):
            try:
                out.append((fp.name, json.loads(fp.read_text(encoding="utf-8"))))
            except Exception as exc:  # a corrupt file shouldn't abort the tally
                print(f"  WARN: could not read {fp}: {exc}", file=sys.stderr)
        return out


class SharePointReader:
    """Read cycle synthesis JSONs from the SharePoint Run folder (reuses SharePointWriter auth)."""

    def __init__(self, site_url: str, folder_path: str):
        from shared.sharepoint import SharePointWriter

        self.w = SharePointWriter(site_url, folder_path)  # device-code auth happens here (once)
        print(f"# reading SharePoint synthesis JSONs under: {self.w.base}")

    def cycle_jsons(self, rel_folder: str) -> list[tuple[str, dict]]:
        from office365.runtime.client_request_exception import ClientRequestException
        from office365.sharepoint.files.file import File

        from shared.sharepoint import _sp_retry

        folder_url = f"{self.w.base}/{rel_folder.strip('/')}"
        try:
            folder = self.w.ctx.web.get_folder_by_server_relative_url(folder_url)
            files = folder.files
            self.w.ctx.load(files)
            _sp_retry(lambda: self.w.ctx.execute_query())
        except ClientRequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 404:      # case never produced output (folder absent)
                return []
            raise

        out: list[tuple[str, dict]] = []
        for f in files:
            name = f.properties.get("Name", "")
            if not (name.startswith("rebuttal_") and name.endswith("_synthesis.json")):
                continue
            server_rel = f.properties.get("ServerRelativeUrl") or f"{folder_url}/{name}"
            resp = _sp_retry(lambda url=server_rel: File.open_binary(self.w.ctx, url))
            try:
                out.append((name, json.loads(resp.content)))
            except Exception as exc:
                print(f"  WARN: could not parse {name}: {exc}", file=sys.stderr)
        return sorted(out, key=lambda t: t[0])


def make_reader(force_local: bool):
    """Build the reader for the backend selected by OUTPUT_TARGET (or forced local)."""
    target = "local" if force_local else (os.getenv("OUTPUT_TARGET") or "local").strip().lower()
    if target == "sharepoint":
        url, folder = os.getenv("SHAREPOINT_URL"), os.getenv("SHAREPOINT_FOLDER")
        if not (url and folder):
            sys.exit("OUTPUT_TARGET=sharepoint requires SHAREPOINT_URL and SHAREPOINT_FOLDER in .env")
        return SharePointReader(url, folder)
    if target != "local":
        sys.exit(f"Unknown OUTPUT_TARGET={target!r} (expected 'local' or 'sharepoint')")
    return LocalReader()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=DEFAULT_CSV, help="cases CSV (default: datastores/test_cases_46.csv)")
    ap.add_argument("--base", default=DEFAULT_BASE, help="corpus root the CSV assets resolve against")
    ap.add_argument("--local", action="store_true", help="force reading the local OUTPUT_BASE tree")
    ap.add_argument("--type", default=None, help="only count cases whose topic slugifies to this")
    ap.add_argument("--year", default=None, help="only count cases from this year (comma-separate ok)")
    ap.add_argument("--by-case", action="store_true", help="print a per-case breakdown table")
    args = ap.parse_args()

    if not Path(args.csv).is_file():
        sys.exit(f"CSV not found: {args.csv}")

    cases = resolve_cases(args.base, args.csv, args.type, args.year)
    print(f"# {len(cases)} resolvable case(s) from {args.csv}\n")
    if not cases:
        return

    reader = make_reader(args.local)

    total_workbooks = 0        # sum of attachments across every cycle JSON (a workbook re-analyzed
                               # in cycle N is counted again — it was analyzed again)
    total_distinct = 0         # distinct workbook filenames per case (de-duped across that case's cycles)
    total_jsons = 0            # cycle synthesis JSONs found
    cases_with_output = 0
    rows: list[tuple[Case, int, int, int]] = []  # (case, n_jsons, workbooks, distinct)

    for case in cases:
        year, asset = infer_year_asset(case.entity_dir)
        rel_folder = f"{year}/{asset}/{case.exc_id}"
        cycles = reader.cycle_jsons(rel_folder)

        case_workbooks = 0
        distinct_names: set[str] = set()
        for _name, data in cycles:
            attachments = data.get("attachments") or []
            case_workbooks += len(attachments)
            distinct_names.update(a.get("file_name", "") for a in attachments)
        distinct_names.discard("")

        total_jsons += len(cycles)
        total_workbooks += case_workbooks
        total_distinct += len(distinct_names)
        if cycles:
            cases_with_output += 1
        rows.append((case, len(cycles), case_workbooks, len(distinct_names)))

    if args.by_case:
        print(f"{'case_id':<28} {'exc':>5} {'cycles':>6} {'workbooks':>9} {'distinct':>8}")
        print("-" * 62)
        for case, n_json, wb, dist in rows:
            print(f"{case.case_id:<28} {case.exc_id:>5} {n_json:>6} {wb:>9} {dist:>8}")
        print()

    print("=" * 62)
    print(f"  cases resolved ............... {len(cases)}")
    print(f"  cases with synthesis output .. {cases_with_output}")
    print(f"  cycle synthesis JSONs read ... {total_jsons}")
    print(f"  WORKBOOKS ANALYZED (total) ... {total_workbooks}   "
          f"(sum of attachments over all cycles)")
    print(f"  distinct workbooks (per-case) {total_distinct}   "
          f"(de-duped within each case's cycles)")
    print("=" * 62)


if __name__ == "__main__":
    main()
