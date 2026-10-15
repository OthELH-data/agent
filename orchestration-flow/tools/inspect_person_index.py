"""Build the person index over the real corpus and print what it read, tab by tab.

The verification harness for `otl/person_index.py`. It resolves each year's POB report, helicopter
manifest and consolidated staff listing exactly the way the `location` / `job_title` / `supervision`
nodes do — `pick_latest_by_keyword` and `resolve_staff_listing`, no second copy of that logic — then
prints, per source and per tab: the header row, the subject column and its kind, the full column-role
map, the roles the tab does NOT carry, the name-like labels left unclassified, rows read, distinct
names, the union across tabs, how many model calls that union needs, and the wall time.

`--no-llm` stops after phase 1, so the deterministic half of the module is checkable at zero spend.
That is the default. Pass `--llm` to run the med-tier selection and see the chosen spellings, the
near-miss pools, and the size of the prompt block each node would receive.

Run it as:

    cd orchestration-flow
    ../.venv/bin/python tools/inspect_person_index.py                        # every year, no spend
    ../.venv/bin/python tools/inspect_person_index.py --year 2021            # the 3-schema workbook
    ../.venv/bin/python tools/inspect_person_index.py --year 2023 --all-versions
    ../.venv/bin/python tools/inspect_person_index.py --year 2023 --llm     # costs med-tier tokens

`--all-versions` indexes every version of a workbook rather than just the latest, which is how the
2023 folder's two structurally unreadable files get exercised: production picks their `_v2` siblings,
so only this flag puts the broken ones in front of the index.

Three things to confirm on this corpus, from the plan:
  1. Every year resolves the tabs the profiling table records, and the 2021 staff listing yields 18
     name-bearing tabs in three families with three different `reports_to` labels.
  2. The 2023 folder reports its two unreadable files with their errors, while still indexing the
     readable `_v2` siblings (`--all-versions`).
  3. The 2021 union size — the number the up-front profiling could not pin down, and what decides
     whether the default chunk budget needs raising.

Exits 0 always: this reports, it does not assert. Read the output.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import otl  # noqa: F401,E402  — bootstraps env + sys.path before any extractor import
from otl._bootstrap import DATA_ROOT  # noqa: E402
from otl.agents._common import pick_latest_by_keyword, resolve_staff_listing  # noqa: E402
from otl.person_index import (build_person_index, harvest_names, person_index_block,  # noqa: E402
                              index_json)
# Deliberately private: this script exists to show phase 1 on its own, and the union + chunk count are
# computed inside `build_person_index` between phases 1 and 2. Borrowing the module's own helpers keeps
# the numbers printed here identical to the ones it would use, rather than a lookalike.
from otl.person_index import _chunk_chars, _chunks, _norm  # noqa: E402

from shared.case_resolver import _norm as _norm_dirname  # noqa: E402

YEARS = ("2021", "2022", "2023", "2024", "2025")
CORPUS = DATA_ROOT / "Exception Reports and Rebuttals by year"
# The keyword sets the three nodes use. `staff` is handled by `resolve_staff_listing` (it carries an env
# override), so its keywords are only used to enumerate the other versions under `--all-versions`.
ROLE_KEYWORDS = {"pob": ["pob"], "heli": ["heli"], "staff": ["consolidated", "staff", "listing"]}
# The four disputed people from the stored 2023 exc 0033 trace, so a `--llm` run can be read against a
# known result: JOHN HARRISON on MARS, the other three absent.
DEFAULT_PEOPLE = ["Foultz, Ryan", "Harrison, John", "Pritchard, Joshua", "Williams, Donell"]


def _common_reports_dir(year: str) -> Path | None:
    """The year's 'Common Reports for all assets' folder (the name varies slightly by year)."""
    year_dir = CORPUS / year
    if not year_dir.is_dir():
        return None
    for d in sorted(year_dir.iterdir()):
        if d.is_dir() and _norm_dirname(d.name) == "commonreportsforallassets":
            return d
    return None


def _all_versions(directory: Path, keywords: list[str]) -> list[str]:
    """Every workbook in `directory` matching all `keywords` — the same filter as
    `pick_latest_by_keyword`, without the latest-version pick."""
    kws = [k.lower() for k in keywords]
    return sorted(
        str(p) for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in (".xlsx", ".xlsb", ".xls")
        and not p.name.startswith("~$") and all(k in p.name.lower() for k in kws)
    )


def _resolve(crd: Path, role: str, all_versions: bool) -> tuple[list[str], list[str]]:
    """`(paths_to_index, other_versions_not_indexed)` for one role in one year's folder."""
    every = _all_versions(crd, ROLE_KEYWORDS[role])
    if all_versions:
        return every, []
    if role == "staff":
        picked = resolve_staff_listing(str(crd))
    else:
        picked = pick_latest_by_keyword(str(crd), ROLE_KEYWORDS[role])
    if not picked:
        return [], every
    return [picked], [p for p in every if p != picked]


def _fmt(n) -> str:
    return f"{n:,}" if isinstance(n, int) else str(n)


def _print_entry(entry: dict, show_values: int) -> None:
    subject = entry.get("subject") or {}
    labels = ", ".join(subject.get("labels") or [])
    print(f"    tab {entry['tab']!r} (header row {entry['header_row']}, shape {entry.get('shape', '?')})")
    print(f"      subject: {labels}  [{subject.get('kind', '?')}]")
    present, absent = [], []
    for role, cols in (entry.get("columns") or {}).items():
        (present if cols else absent).append(
            f"{role}={','.join(cols.keys())}" if cols else role)
    if present:
        print(f"      roles: {' | '.join(present)}")
    if absent:
        print(f"      absent: {', '.join(absent)}")
    if entry.get("unclassified_nameish"):
        print(f"      name-like but unclassified: {', '.join(entry['unclassified_nameish'])}")
    print(f"      rows {_fmt(entry.get('rows', 0))} — distinct names "
          f"{_fmt(len(entry.get('values') or []))}")
    for value in (entry.get("values") or [])[:show_values]:
        print(f"        · {value}")


def _print_skipped(skipped: list[dict]) -> None:
    if not skipped:
        return
    by_reason: dict[str, list[str]] = {}
    for s in skipped:
        by_reason.setdefault(s.get("reason", "?"), []).append(s.get("tab", "?"))
    for reason, tabs in by_reason.items():
        print(f"    skipped ({reason}): {', '.join(tabs)}")


def _print_unreadable(unreadable: list[dict]) -> None:
    for u in unreadable or []:
        print(f"    UNREADABLE {u.get('name', '?')}")
        for engine, err in (u.get("errors") or {}).items():
            print(f"      {engine}: {str(err)[:160]}")


def _union(entries: list[dict]) -> list[str]:
    """The distinct name universe across every entry, first spelling kept — as phase 2 unions it."""
    out, seen = [], set()
    for entry in entries:
        for value in entry.get("values") or []:
            k = _norm(value)
            if k not in seen:
                seen.add(k)
                out.append(value)
    return out


def _harvest_only(paths: list[str], manifest: dict, show_values: int) -> None:
    """Phase 1, with the union and the chunk count phase 2 would need. No model call."""
    started = time.monotonic()
    entries, skipped = harvest_names(manifest, paths, warn=lambda m: print(f"    {m}"))
    elapsed = time.monotonic() - started
    for entry in entries:
        _print_entry(entry, show_values)
    _print_skipped(skipped)
    _print_unreadable(manifest.get("unreadable") or [])
    universe = _union(entries)
    budget = _chunk_chars()
    chunks = _chunks(universe, budget) if universe else []
    chars = sum(len(v) + 1 for v in universe)
    print(f"    UNION across {len(entries)} tab(s): {_fmt(len(universe))} distinct names, "
          f"{_fmt(chars)} chars → {len(chunks)} chunk(s) at {_fmt(budget)} chars each "
          f"(~{_fmt(chars // 4)} prompt tokens)")
    print(f"    phase 1 wall time: {elapsed:.1f}s")


def _with_llm(role: str, paths: list[str], manifest: dict, people: list[str],
              show_values: int) -> None:
    """All three phases, then the block each node would be handed."""
    started = time.monotonic()
    index = build_person_index([(role, paths)], people, manifest,
                               warn=lambda m: print(f"    {m}"))
    elapsed = time.monotonic() - started
    for source in index.get("sources") or []:
        for entry in source.get("entries") or []:
            _print_entry(entry, show_values)
        _print_skipped(source.get("skipped_tabs") or [])
        print(f"    UNION: {_fmt(source.get('universe', 0))} distinct names in "
              f"{source.get('chunks', 0)} model call(s)")
        for person, matches in (source.get("selected") or {}).items():
            for m in matches:
                print(f"    SELECTED {person} -> {m['name_in_file']!r} "
                      f"({m.get('confidence', '?')}, {_fmt(m.get('rows', 0))} rows, "
                      f"tabs {m.get('tabs') or []}) — {m.get('why', '')}")
        for person, miss in (source.get("unselected") or {}).items():
            total = miss.get("pool_total", 0)
            pool = list(miss.get("pool") or [])
            # Slice the LIST, not the joined string — cutting mid-value reads like a data error.
            shown = "; ".join(pool[:6]) + (f" … (+{len(pool) - 6} more shown to the selector)"
                                           if len(pool) > 6 else "")
            tail = f" — e.g. {shown}" if pool else " — nothing in the file shares it"
            print(f"    NOT SELECTED {person} — {_fmt(total)} value(s) share the surname{tail}")
    _print_unreadable(index.get("unreadable") or [])
    block = person_index_block(index, people)
    staged = index_json(index)
    rows = sum(len(e.get("row_data") or {}) for s in staged.get("sources") or []
               for e in s.get("entries") or [])
    print(f"    prompt block: {_fmt(len(block))} chars — staged payload holds {_fmt(rows)} "
          f"name(s) with rows")
    print(f"    all three phases, wall time: {elapsed:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build the person index over the real corpus and print what it read.")
    ap.add_argument("--year", default=None,
                    help=f"only these years, comma-separated (default: {', '.join(YEARS)})")
    ap.add_argument("--source", default=None,
                    help="only these roles, comma-separated: pob, heli, staff (default: all three)")
    ap.add_argument("--llm", action="store_true",
                    help="run the med-tier selection pass too (COSTS TOKENS). Default is phase 1 only.")
    ap.add_argument("--all-versions", action="store_true",
                    help="index every version of each workbook, not just the latest — how the 2023 "
                         "folder's two unreadable files get exercised")
    ap.add_argument("--people", nargs="*", default=None,
                    help=f"disputed people for the selection pass (default: the 2023 exc 0033 four — "
                         f"{'; '.join(DEFAULT_PEOPLE)})")
    ap.add_argument("--values", type=int, default=0, metavar="N",
                    help="also print the first N distinct names of each tab (default 0)")
    args = ap.parse_args()

    years = [y.strip() for y in (args.year or ",".join(YEARS)).split(",") if y.strip()]
    roles = [r.strip().lower() for r in (args.source or "pob,heli,staff").split(",") if r.strip()]
    bad = [r for r in roles if r not in ROLE_KEYWORDS]
    if bad:
        sys.exit(f"unknown source role(s): {', '.join(bad)} (choose from pob, heli, staff)")
    people = list(args.people) if args.people else list(DEFAULT_PEOPLE)

    print(f"# corpus: {CORPUS}")
    print(f"# years: {', '.join(years)} | sources: {', '.join(roles)} | "
          f"phase 2/3: {'ON (spending tokens)' if args.llm else 'off (--llm to enable)'}")
    if args.llm:
        print(f"# people: {'; '.join(people)}")

    # Imported here so a --no-llm run does not need the manifest module's import cost decided up front.
    from otl.workbook_manifest import build_manifest

    for year in years:
        crd = _common_reports_dir(year)
        print("\n" + "=" * 96)
        print(f"{year} — {crd if crd else 'NO Common Reports folder found'}")
        print("=" * 96)
        if not crd:
            continue
        for role in roles:
            paths, others = _resolve(crd, role, args.all_versions)
            print(f"\n  [{role}] {len(paths)} file(s) to index")
            for p in paths:
                print(f"    · {Path(p).name}")
            for p in others:
                print(f"    (not indexed, older version: {Path(p).name})")
            if not paths:
                print("    nothing matched — check the folder's naming")
                continue
            started = time.monotonic()
            manifest = build_manifest(paths)
            print(f"    manifest: {len(manifest.get('workbooks') or [])} workbook(s), "
                  f"{sum(len(w.get('shapes') or []) for w in manifest.get('workbooks') or [])} "
                  f"shape(s), {len(manifest.get('unreadable') or [])} unreadable, "
                  f"built in {time.monotonic() - started:.1f}s")
            if args.llm:
                _with_llm(role, paths, manifest, people, args.values)
            else:
                _harvest_only(paths, manifest, args.values)


if __name__ == "__main__":
    main()
