"""Discover and chain an exception's rebuttal cycles.

An audit exception evolves over successive rebuttal cycles. In the corpus each cycle is a
`Rebuttal N/` folder under the entity (N = 0,1,2,…); a few entities instead use a bare
`Audit Report/` + `Operator Response/` pair (a single cycle 0).

Evidence comes from the AUDITOR side only (the `Audit Report` folder): its report PDF carries the
auditor's exception description, questioned amount and dispute, and its Excel files are the
auditor's evidence. The operator-response folder is never mined for exhibits or content — the only
thing it is used for is a fallback location of the report PDF when the auditor folder has none, and
even then only the auditor's description/amount is read out of it (see run.py + description.py).
Within the auditor folder, a workbook explicitly named by the auditor is evidence even when its
filename otherwise resembles an operator response; the citation is the auditor's affirmative use of
that workbook in the exception.

The auditor side organizes exhibits two ways, both handled here:
  * numbered exhibit folders — `<code>/` (e.g. `0027/`) holding that exception's files, or
  * a flat folder — files named by exception (`EX03-…`, `Ex 10 - …`, `Exception 0019 …`).

This module locates the cycles, finds a given exception's evidence (exhibits + invoices) within
each, and builds the compact `previous_rebuttals` chain (rolling window of 3 with a verbatim tail
— see build_previous_rebuttals).
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from extractor.invoice_extract import _SKIP_RE  # non-invoice PDF filter (reports/correspondence/etc.)
# The operator/auditor name test, shared with the tab-level filter in excel_bounded (and with OTL).
from shared.llm import discover_models
from shared.response_names import is_operator_response

_CYCLE_RE = re.compile(r"^rebuttal\s*(\d+)$", re.I)
_AUDITOR_RE = re.compile(r"audit(?:or)?s?\s*report", re.I)
_OPERATOR_RE = re.compile(r"operator'?s?\s*(?:response|reply)", re.I)
_REPORT_NAME_RE = re.compile(r"audit(?:or)? report|external report|final\s*\[", re.I)
_NUMBERED_RE = re.compile(r"^0*\d{1,4}$")
# Version suffix at the end of a filename stem.  Matches v1 / v_1 / v.1 / v 1 / (1) / (12) — but
# caps the (N) form at 3 digits so 9-digit SharePoint document IDs (e.g. "(129174001)") are not
# mistaken for version markers.
_VER_SUFFIX_RE = re.compile(r"[\s._-]*(?:v[\s._]?(\d{1,3})|\((\d{1,3})\))\s*$", re.I)

_SHEET_EXTS = (".xlsx", ".xls", ".xlsm", ".xlsb")
# Flat-layout filename prefix carrying the exception number: "EX03-", "Ex 10 - ", "EXC0012",
# "Exception 0019 ". The leading digits (zero-stripped) are the exception code.
_NAME_CODE_RE = re.compile(r"^\s*ex(?:c|ception)?\s*0*(\d{1,4})", re.I)


@dataclass
class Cycle:
    """One rebuttal cycle for an entity."""
    number: int
    auditor_dir: Path | None = None
    auditor_report: Path | None = None
    operator_dir: Path | None = None
    operator_report: Path | None = None


@dataclass
class EvidenceDiscovery:
    """Auditor-side evidence found for one exception in one rebuttal cycle.

    ``workbooks`` contains ordinary safe auditor-side material plus any workbook explicitly cited in
    the auditor's description. Uncited workbooks identified as operator responses remain visible only
    under ``excluded_operator_workbooks`` for reviewer provenance.
    """
    cycle_number: int
    auditor_dir: Path | None = None
    auditor_report: Path | None = None
    operator_dir: Path | None = None
    operator_report: Path | None = None
    workbooks: list[Path] = field(default_factory=list)
    supporting_pdfs: list[Path] = field(default_factory=list)
    excluded_operator_workbooks: list[Path] = field(default_factory=list)
    excluded_operator_pdfs: list[Path] = field(default_factory=list)
    # Names come from a light-tier, description-only extraction pass. The names are resolved beneath
    # ``auditor_dir`` rather than treated as paths supplied by the model. An explicit workbook citation
    # can override the ordinary filename ownership filter. These fields make the extractor → OTL
    # attachment handoff auditable and reproducible.
    auditor_cited_attachment_names: list[str] = field(default_factory=list)
    cited_attachment_workbooks: list[Path] = field(default_factory=list)
    cited_attachment_pdfs: list[Path] = field(default_factory=list)

    @property
    def all_workbooks_operator_responses(self) -> bool:
        """Whether every discovered workbook was excluded as operator-side material."""
        return not self.workbooks and bool(self.excluded_operator_workbooks)

    @property
    def evidence_exclusion_notice(self) -> str:
        """Reviewer-facing explanation when no safe workbook survives the ownership filter."""
        if not self.all_workbooks_operator_responses:
            return ""
        names = ", ".join(path.name for path in self.excluded_operator_workbooks)
        return ("No auditor-side workbook evidence was available for this rebuttal cycle because all "
                f"candidate workbooks were excluded as operator-response material: {names}.")

    @staticmethod
    def _relative(path: Path | None, entity_dir: Path) -> str:
        if not path:
            return ""
        try:
            return str(path.resolve().relative_to(entity_dir.resolve()))
        except ValueError:
            return ""

    def to_manifest(self, entity_dir: str | Path) -> dict:
        """Serialize stable entity-relative paths for the extractor → OTL handoff."""
        root = Path(entity_dir)

        def relative_paths(paths: list[Path]) -> list[str]:
            return [value for path in paths if (value := self._relative(path, root))]

        return {
            "path_base": "entity_dir",
            "cycle_number": self.cycle_number,
            "auditor_dir": self._relative(self.auditor_dir, root),
            "auditor_report": self._relative(self.auditor_report, root),
            "operator_dir": self._relative(self.operator_dir, root),
            "operator_report": self._relative(self.operator_report, root),
            "workbooks": relative_paths(self.workbooks),
            "supporting_pdfs": relative_paths(self.supporting_pdfs),
            "excluded_operator_workbooks": relative_paths(self.excluded_operator_workbooks),
            "excluded_operator_pdfs": relative_paths(self.excluded_operator_pdfs),
            "auditor_cited_attachment_names": list(self.auditor_cited_attachment_names),
            "cited_attachment_workbooks": relative_paths(self.cited_attachment_workbooks),
            "cited_attachment_pdfs": relative_paths(self.cited_attachment_pdfs),
        }

    @staticmethod
    def _manifest_path(entity_dir: Path, value: object) -> Path | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts:
            return None
        path = (entity_dir / relative).resolve()
        try:
            path.relative_to(entity_dir.resolve())
        except ValueError:
            return None
        return path

    @classmethod
    def from_manifest(cls, manifest: dict, entity_dir: str | Path) -> "EvidenceDiscovery | None":
        """Rehydrate an extractor evidence manifest without re-discovering the directory."""
        if not isinstance(manifest, dict) or manifest.get("path_base") != "entity_dir":
            return None
        try:
            cycle_number = int(manifest.get("cycle_number"))
        except (TypeError, ValueError):
            return None
        root = Path(entity_dir)

        def paths(key: str, *, require_file: bool = True) -> list[Path]:
            out: list[Path] = []
            for raw in manifest.get(key, []) or []:
                path = cls._manifest_path(root, raw)
                if path and (not require_file or path.is_file()):
                    out.append(path)
            return out

        return cls(
            cycle_number=cycle_number,
            auditor_dir=cls._manifest_path(root, manifest.get("auditor_dir")),
            auditor_report=cls._manifest_path(root, manifest.get("auditor_report")),
            operator_dir=cls._manifest_path(root, manifest.get("operator_dir")),
            operator_report=cls._manifest_path(root, manifest.get("operator_report")),
            workbooks=paths("workbooks"),
            supporting_pdfs=paths("supporting_pdfs"),
            # Preserve exclusion provenance even when the raw workbook is not mounted locally.
            excluded_operator_workbooks=paths("excluded_operator_workbooks", require_file=False),
            excluded_operator_pdfs=paths("excluded_operator_pdfs", require_file=False),
            auditor_cited_attachment_names=_clean_attachment_names(
                manifest.get("auditor_cited_attachment_names") or []
            ),
            cited_attachment_workbooks=paths("cited_attachment_workbooks"),
            cited_attachment_pdfs=paths("cited_attachment_pdfs"),
        )


def _pick_report(folder: Path | None) -> Path | None:
    """The report PDF/DOCX directly inside `folder`: prefer a report-named one, else the sole PDF."""
    if not folder or not folder.exists():
        return None
    pdfs = sorted(folder.glob("*.pdf"))
    docx = sorted(folder.glob("*.docx"))
    if not pdfs and not docx:
        return None
    named = [p for p in pdfs if _REPORT_NAME_RE.search(p.name)]
    if not named:
        named = [p for p in docx if _REPORT_NAME_RE.search(p.name)]

    if named:
        return named[0]

    return None


def _find_side(cycle_dir: Path, pattern: re.Pattern) -> Path | None:
    """First immediate subfolder of cycle_dir whose name matches `pattern`."""
    for d in sorted(cycle_dir.iterdir()):
        if d.is_dir() and pattern.search(d.name):
            return d
    return None


def _build_cycle(number: int, cycle_dir: Path) -> Cycle:
    auditor = _find_side(cycle_dir, _AUDITOR_RE)
    operator = _find_side(cycle_dir, _OPERATOR_RE)
    return Cycle(
        number=number,
        auditor_dir=auditor,
        auditor_report=_pick_report(auditor),
        operator_dir=operator,
        operator_report=_pick_report(operator),
    )


def discover_rebuttal_cycles(entity_dir: str | Path) -> list[Cycle]:
    """Return the entity's rebuttal cycles, sorted ascending by cycle number.

    Handles the `Rebuttal N/` layout and the bare `Audit Report/` + `Operator Response/` pair
    (treated as a single cycle 0).
    """
    entity_dir = Path(entity_dir)
    cycles: list[Cycle] = []
    for d in entity_dir.iterdir():
        if not d.is_dir():
            continue
        m = _CYCLE_RE.match(d.name)
        if m:
            cycles.append(_build_cycle(int(m.group(1)), d))
    if cycles:
        return sorted(cycles, key=lambda c: c.number)

    # Fallback: no "Rebuttal N" wrappers — the entity dir itself is a single cycle 0.
    if _find_side(entity_dir, _AUDITOR_RE) or _find_side(entity_dir, _OPERATOR_RE):
        return [_build_cycle(0, entity_dir)]
    return []


def _code_matches(folder_name: str, code: str) -> bool:
    """True if a numbered exhibit folder name equals the exception code, ignoring zero-padding."""
    if not _NUMBERED_RE.match(folder_name):
        return False
    try:
        return int(folder_name) == int(code)
    except ValueError:
        return False


def _name_code_matches(filename: str, code: str) -> bool:
    """True if a flat filename's leading exception-number prefix equals `code` (padding-agnostic)."""
    m = _NAME_CODE_RE.match(filename)
    if not m:
        return False
    try:
        return int(m.group(1)) == int(code)
    except ValueError:
        return False


def _is_lock_file(path: Path) -> bool:
    """Whether ``path`` is an Office temporary lock/owner file, not a real workbook."""
    return path.name.startswith("~$")


def _spreadsheets(paths) -> list[Path]:
    return sorted(
        p for p in paths
        if p.is_file() and p.suffix.lower() in _SHEET_EXTS and not _is_lock_file(p)
    )


def _clean_attachment_names(names) -> list[str]:
    """Normalize untrusted LLM-returned filenames without treating them as paths.

    The description-only model is allowed to name an attachment, not to select a directory.  Matching
    against a basename also makes the later lookup work when a report printed a relative path.
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in names or []:
        name = Path(str(value or "").strip().strip("`'\" ")).name
        key = name.casefold()
        if not name or key in seen:
            continue
        seen.add(key)
        cleaned.append(name)
    return cleaned


def _unique_paths(paths: list[Path]) -> list[Path]:
    """Deterministically deduplicate resolved paths while preserving sorted discovery order."""
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(path)
    return sorted(out, key=lambda path: str(path).casefold())


def _attachment_key(value: str) -> str:
    """Case/punctuation-insensitive key used only for auditor-citation filename resolution."""
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _cited_workbook_match_rank(path: Path, cited_name: str) -> int | None:
    """Rank a workbook filename against the name printed in the audit description.

    Audit descriptions commonly omit ``.xlsx`` and lightly reformat separators. A small trailing
    year/version is also common in the stored upload name. Prefer exact matches, then normalized stem
    matches, then a normalized stem followed only by such a suffix.
    """
    cited = Path(cited_name)
    if path.name.casefold() == cited.name.casefold():
        return 0

    cited_has_sheet_extension = cited.suffix.lower() in _SHEET_EXTS
    cited_base = cited.stem if cited_has_sheet_extension else cited.name
    actual_key = _attachment_key(path.stem)
    cited_key = _attachment_key(cited_base)
    if not cited_key:
        return None
    if actual_key == cited_key:
        return 1
    if not cited_has_sheet_extension and actual_key.startswith(cited_key):
        suffix = actual_key[len(cited_key):]
        if re.fullmatch(r"(?:19|20)\d{2}|v\d{1,3}|\d{1,3}", suffix):
            return 2
    return None


def _resolve_cited_workbooks(all_files: list[Path], cited_names: list[str]) -> list[Path]:
    """Resolve each auditor-printed workbook name to its best on-disk match(es)."""
    candidates = [
        path for path in all_files
        if path.suffix.lower() in _SHEET_EXTS and not _is_lock_file(path)
    ]
    resolved: list[Path] = []
    for cited_name in cited_names:
        ranked = [
            (rank, path)
            for path in candidates
            if (rank := _cited_workbook_match_rank(path, cited_name)) is not None
        ]
        if not ranked:
            continue
        best_rank = min(rank for rank, _path in ranked)
        resolved.extend(path for rank, path in ranked if rank == best_rank)
    return _unique_paths(resolved)


def _safe_cited_paths(auditor_dir: Path | None, cited_names: list[str]) -> tuple[
    list[Path], list[Path], list[Path],
]:
    """Resolve cited basenames under the auditor directory with explicit-workbook precedence.

    A cited filename can live outside the exception-numbered folder (for example, a common attachment
    referenced in the exception description). We therefore search the auditor-side tree. Workbooks
    explicitly named by the auditor override the operator-response filename filter and join the manifest;
    supporting PDFs retain the ordinary ownership filter.
    """
    if not auditor_dir or not auditor_dir.exists() or not cited_names:
        return [], [], []
    wanted = {name.casefold() for name in cited_names}
    all_files = sorted(path for path in auditor_dir.rglob("*") if path.is_file())
    workbooks = _resolve_cited_workbooks(all_files, cited_names)
    pdf_candidates = [path for path in all_files
                      if path.suffix.lower() == ".pdf"
                      and not _SKIP_RE.search(path.name)
                      and path.name.casefold() in wanted]
    pdfs, excluded_pdfs = _split_operator_responses(pdf_candidates, kind="supporting PDF")
    return workbooks, pdfs, excluded_pdfs


def union_auditor_cited_attachments(discovery: EvidenceDiscovery, attachment_names) -> EvidenceDiscovery:
    """Union auditor-cited files into already-filtered discovery evidence.

    ``find_exception_evidence`` first establishes the ordinary exception-scoped safe list.  This second
    step is deliberately additive: an auditor can explicitly cite an attachment stored outside that
    folder or lacking the flat-layout exception prefix. An explicitly cited workbook overrides the
    operator-response filename filter, because the auditor has made it part of the exception's evidence.
    Names still must resolve deterministically beneath the same auditor-side directory.
    """
    cited_names = _clean_attachment_names(attachment_names)
    discovery.auditor_cited_attachment_names = cited_names
    cited_books, cited_pdfs, excluded_pdfs = _safe_cited_paths(
        discovery.auditor_dir, cited_names
    )
    discovery.cited_attachment_workbooks = cited_books
    discovery.cited_attachment_pdfs = cited_pdfs
    discovery.workbooks = _unique_paths([*discovery.workbooks, *cited_books])
    discovery.supporting_pdfs = _unique_paths([*discovery.supporting_pdfs, *cited_pdfs])
    promoted_books = {path.resolve() for path in cited_books}
    discovery.excluded_operator_workbooks = _unique_paths(
        [path for path in discovery.excluded_operator_workbooks
         if path.resolve() not in promoted_books]
    )
    discovery.excluded_operator_pdfs = _unique_paths(
        [*discovery.excluded_operator_pdfs, *excluded_pdfs]
    )
    return discovery


def _ver_key(stem: str) -> int:
    """Version number from a filename stem: 0 for the base (no suffix), N for vN or (N)."""
    m = _VER_SUFFIX_RE.search(stem)
    return int(m.group(1) or m.group(2)) if m else 0


def _strip_ver(stem: str) -> str:
    """Stem with the version suffix removed, for grouping files that are the same exhibit."""
    return _VER_SUFFIX_RE.sub("", stem).rstrip(" ._-")


def dedupe_versions(paths: list[Path], warn=None) -> list[Path]:
    """Keep only the latest version of each workbook when an exhibit is filed multiple times.

    Groups files in the same folder by their base stem (version suffix stripped) + extension. Within
    each group of >1, the file with the highest version number wins (base < v1/(1) < v2/(2) …); name
    sorts as a tiebreaker so the result is deterministic. Single-file groups pass through unchanged.

    Only numeric suffixes ≤ 999 count as version markers — large identifiers such as (129174001) are
    not version numbers and leave their file ungrouped.
    """
    warn = warn or (lambda m: print(m, file=sys.stderr))
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for p in paths:
        key = (str(p.parent), _strip_ver(p.stem), p.suffix.lower())
        groups[key].append(p)
    result: list[Path] = []
    for group in groups.values():
        if len(group) == 1:
            result.append(group[0])
        else:
            winner = max(group, key=lambda p: (_ver_key(p.stem), p.name))
            for p in group:
                if p != winner:
                    warn(f"  skipping older version: {p.name} (keeping {winner.name})")
            result.append(winner)
    return sorted(result)


def _split_operator_responses(paths: list[Path], warn=None, kind: str = "workbook") -> tuple[list[Path], list[Path]]:
    """Filter operator-response files out of an auditor-side evidence list.

    A workbook whose FILENAME marks it as the OPERATOR's is not auditor evidence even when it sits in the
    auditor folder — the corpus misfiles them routinely (`Response to 2023SOV_E102 GW Ex 07` filed next to
    the auditor's own `IR 32 … JADE`). Which names those are is decided by `response_names`, shared with
    the tab-level counterpart (`excel_bounded._usable_worksheets`); note that a name handing the reply to
    the auditor (`… - Auditor Reply.xlsx`) is auditor-side and is KEPT.

    Returning no safe paths is intentional when every ordinary candidate is operator-side. A later,
    explicit citation by the auditor may promote a workbook through
    ``union_auditor_cited_attachments``.
    """
    warn = warn or (lambda m: print(m, file=sys.stderr))
    dropped = [p for p in paths if is_operator_response(p.name)]
    if not dropped:
        return paths, []
    kept = [p for p in paths if not is_operator_response(p.name)]
    if not kept:
        warn(f"WARN: every auditor-side {kind} here looks like an operator response "
             f"({', '.join(p.name for p in dropped)}) — excluding all of them; no auditor-side "
             f"{kind} evidence is safe to analyze.")
        return [], dropped
    for p in dropped:
        warn(f"  skipping operator-response {kind}: {p.name}")
    return kept, dropped


def drop_operator_responses(paths: list[Path], warn=None) -> list[Path]:
    """Return only safe auditor-side workbooks; see ``_split_operator_responses`` for provenance."""
    kept, _dropped = _split_operator_responses(paths, warn=warn)
    return kept


def _exception_folder(base: Path, code: str) -> Path | None:
    """The exhibit subfolder for `code`, if the auditor side uses per-exception folders.

    Matches both a pure-numeric folder (`0027`, `032`) and a prefixed one (`Ex 03`, `ex 02`).
    """
    for d in sorted(base.iterdir()):
        if d.is_dir() and (_code_matches(d.name, code) or _name_code_matches(d.name, code)):
            return d
    return None


def exception_folder_is_empty(cycle: Cycle, code: str) -> bool:
    """True when this cycle filed a per-exception folder for `code` that holds no files at all.

    An empty folder is a deliberate filing, not a gap: the auditor kept the exception's slot in the
    cycle but attached nothing — typically because the exception was CLOSED in that round (e.g.
    Silvertip 2022 exception 02 at Rebuttal 1, denied in full with no new workpapers). Such a cycle
    carries no description block in the report either, so running it produces a synthesis built on
    whatever the description fallbacks scrape up. Callers skip the cycle instead.

    Only the per-exception-folder layout can be "empty" in this sense. False when the auditor side
    is missing, when the layout is flat (no folder to be empty), and when the folder holds files but
    none of them are usable evidence — that last case still runs and records 'insufficient
    evidence'. Dotfiles (`.DS_Store`) and empty nested subfolders do not count as content.
    """
    base = cycle.auditor_dir
    if not base or not base.exists():
        return False
    folder = _exception_folder(base, code)
    if folder is None:
        return False
    return not any(p.is_file() and not p.name.startswith(".") for p in folder.rglob("*"))


def find_exception_evidence(cycle: Cycle, code: str, *, warn=None) -> EvidenceDiscovery:
    """Return structured auditor-side evidence for ``code`` in one cycle.

    Handles every auditor-side layout and scopes BOTH kinds of file to the one exception:
      * per-exception folder — `0027/`, `032/`, `Ex 03/`, `ex 02/` (searched recursively, since a
        few nest one level, e.g. `008/008/`); everything inside belongs to this exception, or
      * flat auditor folder — files whose name prefix encodes `code` (`EX03-`, `Ex 10 - `, …),
        which is essential because a flat folder mixes every exception's files together.

    Whichever layout, operator-response workbooks misfiled on the auditor side are filtered out by
    filename. An all-operator workbook set returns an empty ``workbooks`` list plus the excluded
    names, rather than allowing those workbooks back into the evidence stream. ``warn`` optionally
    receives preliminary filtering diagnostics; callers that will run
    ``union_auditor_cited_attachments`` can defer those messages until citation promotion is complete.
    """
    base = cycle.auditor_dir
    common = {
        "cycle_number": cycle.number,
        "auditor_dir": cycle.auditor_dir,
        "auditor_report": cycle.auditor_report,
        "operator_dir": cycle.operator_dir,
        "operator_report": cycle.operator_report,
    }
    if not base or not base.exists():
        return EvidenceDiscovery(**common)

    folder = _exception_folder(base, code)
    if folder is not None:  # per-exception folder: recurse — all of its contents are this exception
        kept, excluded = _split_operator_responses(_spreadsheets(folder.rglob("*")), warn=warn)
        xlsx = dedupe_versions(kept)
        pdf_candidates = [p for p in sorted(folder.rglob("*.pdf")) if not _SKIP_RE.search(p.name)]
        pdfs, excluded_pdfs = _split_operator_responses(pdf_candidates, warn=warn,
                                                         kind="supporting PDF")
        return EvidenceDiscovery(workbooks=xlsx, supporting_pdfs=pdfs,
                                 excluded_operator_workbooks=excluded,
                                 excluded_operator_pdfs=excluded_pdfs, **common)

    # flat layout: pick only the top-level files whose leading prefix is this exception's number
    candidates = [p for p in _spreadsheets(base.iterdir()) if _name_code_matches(p.name, code)]
    kept, excluded = _split_operator_responses(candidates, warn=warn)
    xlsx = dedupe_versions(kept)
    pdf_candidates = [p for p in sorted(base.glob("*.pdf"))
                      if _name_code_matches(p.name, code) and not _SKIP_RE.search(p.name)]
    pdfs, excluded_pdfs = _split_operator_responses(pdf_candidates, warn=warn,
                                                     kind="supporting PDF")
    return EvidenceDiscovery(workbooks=xlsx, supporting_pdfs=pdfs,
                             excluded_operator_workbooks=excluded,
                             excluded_operator_pdfs=excluded_pdfs, **common)


def build_previous_rebuttals(prior_reports: list, running_summary: str) -> dict[str, str]:
    """Compress the chain: oldest 3·⌊k/3⌋ under 'summarized_rebuttals', trailing k mod 3 verbatim.

    `prior_reports` is the list of already-processed cycle reports (ExhibitEvidenceReport), in
    appearance order, excluding the current cycle. Each must have `.rebuttal_id` and `.synthesis`.
    """
    k = len(prior_reports)
    if k == 0:
        return {}
    if k < 3:
        return {r.rebuttal_id: r.synthesis for r in prior_reports}

    tail = k % 3
    out: dict[str, str] = {"summarized_rebuttals": running_summary}
    if tail:
        for r in prior_reports[-tail:]:
            out[r.rebuttal_id] = r.synthesis
    return out
