"""Resolve CSV rows — or a bare case id — to concrete extractor cases.

The batch CSV (e.g. datastores/test_cases_46.csv) lists one exception per row (year, asset, exc_id,
topic). `resolve_cases` maps each row to a runnable case — matching the CSV asset name to the real
entity folder on disk and building a stable case id — so both the parallel driver (run_batch.py) and
the shell runner share one resolver instead of duplicating the logic.

A case id is self-describing (`<Asset>_<year>_<exc>`), so it does NOT need a CSV row: `resolve_case_id`
rebuilds the whole Case from the id alone — entity folder off disk via `find_classified_exception`, exception type from the
master catalog (`clustered_exception_tables.csv`, every year/site) or an explicit override. That is what
lets `--case-id GreatWhite_2024_0040` run an exception no run-format CSV happens to list.
`select_case_id` is the entry point the runners use: filter the CSV-resolved list, else synthesize.

Exception ids are canonicalized to the corpus's 4-digit form (`pad_exc`) because the CSVs disagree
(labor_*.csv carry bare `40`, test_cases_*.csv carry `0040`) while `exc_id` lands verbatim in the output
tree — one exception must resolve to one case id and one output path whichever CSV named it. Rebuttal
cycles are not CSV fields; OTL cycle expansion comes from the classified-exceptions catalog, with an
explicit `--rebuttal-cycle` command-line override for targeted runs.
"""
from __future__ import annotations

import csv
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from shared.case_keys import _norm, pad_exc
from shared.classified_exceptions import find_classified_exception


@dataclass
class Case:
    entity_dir: Path
    exc_id: str
    exception_type: str      # the CSV topic — travels to the child as --exception-type (exception type AND category downstream)
    case_id: str
    rebuttal_cycle: int = 0  # explicit OTL cycle selector, normally set by the runner (default 0)


def slugify(s: str) -> str:
    """Normalize an exception type/topic to a slug (mirrors extractor.skills.slugify).

    "Material Reconciliation" -> "material_reconciliation".
    """
    return re.sub(r"[^a-z0-9]+", "_", (s or "").strip().lower()).strip("_")


# Each runnable field maps to the header names it may appear under, matched case-insensitively after
# stripping. This lets one resolver read both CSV schemas in datastores/: the test_cases_*.csv form
# (asset / exc_id / exception_type) and the clustered_exception_tables.csv form (Site / Exception ID /
# cluster_topic).
_COLUMN_ALIASES = {
    "year": ("year",),
    "asset": ("asset", "site"),
    "exc_id": ("exc_id", "exc", "exception id", "exception_id"),
    "exception_type": ("topic", "cluster_topic", "classification_category", "classification"),
    "amount": ("gross_credit_requested", "credit_amount", "credit amount"),
}


def _field(row_norm: dict[str, str], field: str) -> str:
    """First non-empty value among a field's accepted header aliases (keys already lowercased)."""
    for alias in _COLUMN_ALIASES[field]:
        val = (row_norm.get(alias) or "").strip()
        if val:
            return val
    return ""


def _asset_from_source_file(row_norm: dict[str, str]) -> str:
    """Recover a missing CSV site from a corpus-relative source-file path.

    The temporal split contains four 2025 Great White rows with a blank
    ``site`` but a source path such as ``2025/GreatWhite - Chevron only/...``.
    The first path component after the year is the entity label; its suffix is
    a participant qualifier rather than part of the on-disk asset name.
    """
    source = (row_norm.get("source_file") or "").strip().replace("\\", "/")
    match = re.match(r"^\d{4}/([^/]+)/", source)
    if not match:
        return ""
    return match.group(1).strip()


def parse_case_id(case_id: str) -> tuple[str, str, str] | None:
    """Split '<Asset>_<year>_<exc>' back into (asset, year, padded exc); None if it isn't that shape.

    Safe to split from the right: `resolve_cases` builds the asset segment by stripping every
    non-alphanumeric from the entity folder name, so it can never contain '_' — the last two
    underscore-separated fields are always the year and the exception.
    """
    parts = (case_id or "").strip().rsplit("_", 2)
    if len(parts) != 3:
        return None
    asset, year, exc = (p.strip() for p in parts)
    if not (asset and exc and year.isdigit() and len(year) == 4):
        return None
    return asset, year, pad_exc(exc)


def case_id_key(case_id: str) -> str:
    """Canonical comparison key for a case id, so 'gw_2024_44' matches 'GreatWhite_2024_0044'.

    Normalizes the asset through `_norm` (case, punctuation and the 'GW' shorthand) and the exception
    through `pad_exc`. An unparseable id degrades to a plain lowercase compare.
    """
    parsed = parse_case_id(case_id)
    if not parsed:
        return (case_id or "").strip().lower()
    asset, year, exc = parsed
    return f"{_norm(asset)}_{year}_{exc}"


# The master catalog: every year/site/exception with its exception type, including the ones no run-format CSV
# lists. It is the exception type source for a case id resolved without a CSV row.
CATALOG_CSV = Path(__file__).resolve().parents[1] / "datastores" / "labor_classification_test_set_temporal_split.csv"


def catalog_csv() -> Path:
    """Path to the master exception catalog (env `CASES_CATALOG_CSV` overrides)."""
    return Path(os.getenv("CASES_CATALOG_CSV") or CATALOG_CSV)


# The corpus root every case resolves against, and the default run-format cases CSV. Both live here so
# the four runners share one definition instead of the four DEFAULT_BASE / six DEFAULT_CSV copies they
# used to carry (one of which was an absolute /Users/... path that only worked on one machine).
DATASTORES = Path(__file__).resolve().parents[1] / "datastores"
DEFAULT_CASES_CSV = DATASTORES / "labor_classification_test_set_temporal_split.csv"
CORPUS_BASE = Path(__file__).resolve().parents[2] / "Data Files" / \
    "Exception Reports and Rebuttals by year"


def default_corpus_base() -> Path:
    """The corpus root holding the per-year exception folders.

    Deliberately NOT configurable — the corpus location does not change, and having it in one place is
    what lets `--case-id` alone identify a case. Derived from this file's location, so it is the same
    path `otl/_bootstrap.py` computes as DATA_ROOT rather than a second hardcoded absolute.
    """
    return CORPUS_BASE


def resolve_cases_csv(arg: str | Path | None, warn=None) -> Path:
    """Resolve a `--csv` argument to a readable cases CSV, falling back to the default.

    A bare filename ("test_cases_34.csv") resolves under datastores/, so callers name a file rather than
    a path; anything with a separator, or absolute, is taken as given. A value that doesn't exist warns
    and falls back to the default rather than failing the run — the flag is a convenience, and a typo
    shouldn't cost a batch. Only a missing default is fatal, and it exits cleanly instead of surfacing
    the bare `open()` traceback resolve_cases used to raise.
    """
    warn = warn or (lambda m: print(m, file=sys.stderr))
    if arg:
        p = Path(arg)
        p = p if (p.is_absolute() or len(p.parts) > 1) else DATASTORES / p.name
        if p.is_file():
            return p
        warn(f"WARN: cases CSV not found: {p} — falling back to {DEFAULT_CASES_CSV.name}")
    if not DEFAULT_CASES_CSV.is_file():
        sys.exit(f"cases CSV not found: {DEFAULT_CASES_CSV}")
    return DEFAULT_CASES_CSV


def lookup_exception_type(year: str, asset: str, exc_id: str, csv_paths=None) -> str:
    """This exception's type, from the given CSVs in order (first hit wins); '' if none has it.

    Defaults to the master catalog. Matching is tolerant on both keys the CSVs disagree about — asset
    spelling via `_norm`, exception padding via `pad_exc` — so a row reading `2024,GreatWhite,…,40`
    still answers a lookup for '0040'.
    """
    paths = [Path(p) for p in (csv_paths if csv_paths is not None else [catalog_csv()]) if p]
    want_year, want_asset, want_exc = str(year).strip(), _norm(asset), pad_exc(exc_id)
    for path in paths:
        if not path.is_file():
            continue
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                row = {(k or "").strip().lower(): (v or "") for k, v in r.items()}
                if (_field(row, "year").strip() == want_year
                        and _norm(_field(row, "asset")) == want_asset
                        and pad_exc(_field(row, "exc_id")) == want_exc):
                    exception_type = _field(row, "exception_type")
                    if exception_type:
                        return exception_type
    return ""


def resolve_cases(base: str | Path, csv_path: str | Path,
                  type_filter: str | None = None, year_filter: str | None = None,
                  warn=None) -> list[Case]:
    """Resolve every runnable case from the CSV, optionally filtered to one exception-type slug
    and/or one or more years.

    `type_filter` is compared against each row's slugified `exception_type`; `year_filter` against each row's
    `year` (a single year like "2024", or a comma-separated set like "2023,2024"). Both filters
    compose. Unresolvable rows (no matching entity folder on disk) are skipped; `warn(msg)` is called
    for each (defaults to stderr).
    """
    base = Path(base)
    warn = warn or (lambda m: print(m, file=sys.stderr))
    tf = slugify(type_filter) if type_filter else ""
    years = {y.strip() for y in (year_filter or "").split(",") if y.strip()}
    cases: list[Case] = []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            # Normalize header keys (lowercase + strip) so both CSV schemas resolve via _field aliases.
            row = {(k or "").strip().lower(): (v or "") for k, v in r.items()}
            year, asset, exc_id = _field(row, "year"), _field(row, "asset"), _field(row, "exc_id")
            if not asset:
                asset = _asset_from_source_file(row)
            exception_type = _field(row, "exception_type")
            if not (year and asset and exc_id):
                continue  # not a runnable row (blank line or missing the key columns)
            exc_id = pad_exc(exc_id)  # canonical 4-digit id whichever CSV convention this row uses
            if tf and slugify(exception_type) != tf:
                continue
            if years and year.strip() not in years:
                continue
            report = find_classified_exception(year, asset, exc_id)
            source_file = report.get("source_file", "") if report else None
            if not source_file:
                warn(f"WARN: no entity folder for {year} / {asset!r} (exc {exc_id})")
                continue
            ent = base / "/".join(source_file.split("/")[0:2])
            case_id = f"{re.sub(r'[^A-Za-z0-9]', '', _norm(asset))}_{year}_{exc_id}"
            cases.append(Case(entity_dir=ent, exc_id=exc_id, exception_type=exception_type, case_id=case_id))
    return cases


def resolve_case_id(base: str | Path, case_id: str, *, csv_path: str | Path | None = None,
                    exception_type: str | None = None, warn=None) -> Case | None:
    """Build one Case straight from a case id — no CSV row required.

    The id carries everything structural (`<Asset>_<year>_<exc>`), so the entity folder is matched off
    disk by `find_classified_exception` and only the exception_type has to be looked up: `exception_type` when given, else `csv_path`
    then the master catalog. Returns None when the id is malformed or no entity folder matches. A
    missing exception_type only warns — the caller decides whether a blank category is fatal (it matters to the
    extractor, where the exception_type selects the extraction skill).
    """
    warn = warn or (lambda m: print(m, file=sys.stderr))
    parsed = parse_case_id(case_id)
    if not parsed:
        warn(f"WARN: cannot parse case id {case_id!r} — expected '<Asset>_<year>_<exc>', "
             f"e.g. GreatWhite_2024_0040")
        return None
    asset, year, exc_id = parsed
    report = find_classified_exception(year, asset, exc_id)
    source_file = report.get("source_file", "") if report else None
    if not source_file:
        warn(f"WARN: no entity folder for {year} / {asset!r} (exc {exc_id})")
        return None
    ent = base / "/".join(source_file.split("/")[0:2])
    exception_type = (exception_type or "").strip() or lookup_exception_type(
        year, asset, exc_id, [p for p in (csv_path, catalog_csv()) if p])
    if not exception_type:
        warn(f"WARN: no exception type found for {year}/{asset}/{exc_id} in the cases CSV or "
             f"{catalog_csv().name} — pass --exception-type '<Exception Type>' to set it explicitly.")
    canonical = f"{re.sub(r'[^A-Za-z0-9]', '', _norm(asset))}_{year}_{exc_id}"
    return Case(entity_dir=ent, exc_id=exc_id, exception_type=exception_type, case_id=canonical)


def select_case_id(cases: list[Case], case_id: str, base: str | Path, *,
                   csv_path: str | Path | None = None, exception_type: str | None = None,
                   warn=None) -> list[Case]:
    """Narrow `cases` to one `--case-id`, synthesizing it from disk when the CSV doesn't list it.

    Matching goes through `case_id_key`, so the id you type resolves regardless of asset spelling or
    exception padding. An id the CSV never had — or one its type/year filters removed — falls through to
    `resolve_case_id`; that fallback is what makes `--case-id` self-sufficient. Every runner calls this
    one function so the behaviour can't drift between them.
    """
    want = case_id_key(case_id)
    hit = [c for c in cases if case_id_key(c.case_id) == want]
    if hit:
        return hit
    syn = resolve_case_id(base, case_id, csv_path=csv_path, exception_type=exception_type, warn=warn)
    return [syn] if syn else []


def warn_exception_type_mismatch(cases: list[Case], type_filter: str, warn=None) -> None:
    """Warn for each case whose exception type isn't `type_filter` — the explicit-`--case-id` bypass is loud.

    A case id names one exception outright, so the runners honour it even when its exception type is outside the
    scope they filter to (`GreatWhite_2024_0040` is "Unsupported Charges", not "Labor Issues"). Saying so
    up front is the difference between a deliberate override and a confusing downstream result.
    """
    warn = warn or (lambda m: print(m, file=sys.stderr))
    tf = slugify(type_filter)
    for c in cases:
        if tf and slugify(c.exception_type) != tf:
            warn(f"WARN: {c.case_id} exception type is {c.exception_type or '(unknown)'!r}, not {type_filter!r} — "
                 f"running anyway because --case-id names it explicitly. The prompts/skills here are "
                 f"built for {type_filter!r}, so expect a degraded result.")
