"""Local code-execution sandbox for the data-exploration agents (jade_analyzer, location, job_title).

These agents cannot fit raw workbook rows in context, so they instead *write Python* that opens the
workbook, computes a summary, and prints/saves only the result. This executor runs that code in a
child `python` (the same interpreter/venv, so `openpyxl`/`pandas`/`pyxlsb`/`xlrd` are already
available) with:

* **Read scope** rooted at `DATA_ROOT` = two levels above `orchestration-flow`
  (".../Shell JV Audit/Data Files"), env-overridable via `OTL_DATA_ROOT`, plus any explicit
  `read_roots` the case resolver hands in. A preamble helper `open_data(path)` resolves the real path
  and asserts it lives under an allowed root — a soft convenience guard for READS (a stray read path
  raises a clean, in-loop error the ReAct loop can react to).
* **Read roots are WRITE-PROTECTED deterministically.** The corpus is read-only, so the preamble
  intercepts the standard write/mutation entry points (`builtins.open` in a write mode, `os.open` with
  a write flag, and `os.remove`/`unlink`/`rename`/`replace`/`truncate`/`mkdir`/`rmdir`) BEFORE any
  agent code runs and refuses any target that resolves inside a read root — even if the agent bypasses
  `open_data`. Concurrent cases can therefore read the shared workbooks (POB/heli, staff listing, jobs
  catalogue) safely: reads never mutate, so no lock is needed.
* **Operator-response TABS are structurally absent.** The same reasoning as the write guard: the agent
  enumerates sheets itself, so `openpyxl.load_workbook`, `pandas.read_excel` and `pandas.ExcelFile`
  (`.sheet_names` / `.parse`) are patched in the preamble to hide any sheet whose name marks it as the
  operator's reply — the auditor's own workbook often carries exactly one such tab. Naming one raises a
  clean `PermissionError` the ReAct loop can react to; a sheet INDEX addresses the readable sheets only.
  `list_sheets(path)` is the convenience view. A workbook that is *entirely* response tabs keeps them all
  with a WARN, since an unreadable exhibit is worse than a degraded one.
* **Write side** confined to `orchestration-flow/.otl_scratch/<run_id>/<agent>/` (or `$TMPDIR` if that
  isn't writable): the child's cwd IS that scratch dir, and writes there (and anywhere outside the read
  roots — `$TMPDIR`, the pip target) are always allowed. `script.py`, `inputs.json`, `result.json`
  live there.

This is a trusted-local sandbox (the design requires local `pip install`), not an OS-hardened jail:
no network isolation, run via `sys.executable` with `shell=False` and a hard per-run `timeout`. The
write guard blocks the accidental-write vectors an LLM emits (a stray `openpyxl` `.save()`,
`DataFrame.to_excel(<source>)`, `open(path, "w")`, `os.remove`/`rename`), not a determined in-process
escape.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from otl._bootstrap import DATA_ROOT, ORCH_DIR
from otl.response_filter import AUDITOR_OWNED_RE, QUALIFIED_RE, UNCONDITIONAL_RE


def _data_root() -> Path:
    """The read-scope root: OTL_DATA_ROOT override, else two levels above orchestration-flow."""
    override = os.getenv("OTL_DATA_ROOT")
    return Path(override) if override else DATA_ROOT


# The operator/auditor tab test, shipped to the child as pattern SOURCE because the child cannot import
# the extractor (its cwd is the scratch dir). Injecting the patterns keeps the sandbox's copy of the rule
# from becoming an independent definition of it — see extractor/response_names.py.
_RESPONSE_PATTERNS = {
    "unconditional": UNCONDITIONAL_RE.pattern,
    "qualified": QUALIFIED_RE.pattern,
    "auditor_owned": AUDITOR_OWNED_RE.pattern,
}

# Preamble prepended to every agent snippet. It loads INPUTS, defines the read guard, and offers a
# save_result() convenience. `_READ_ROOTS_RAW` (a JSON list of allowed dirs) and `_RESPONSE_PATTERNS` are
# prepended per run — NOT via str.format, since the preamble's own f-strings contain literal braces.
_PREAMBLE = '''\
import json, os, sys
from pathlib import Path

with open("inputs.json", "r", encoding="utf-8") as _f:
    INPUTS = json.load(_f)

_READ_ROOTS = [Path(p).resolve() for p in _READ_ROOTS_RAW]

def _under_roots(p: Path) -> bool:
    rp = p.resolve()
    for root in _READ_ROOTS:
        try:
            rp.relative_to(root)
            return True
        except ValueError:
            continue
    return False

def open_data(path):
    """Resolve a data path and assert it is inside an allowed read root (soft guard).

    Returns a pathlib.Path you can pass to pandas/openpyxl. Raises PermissionError with a clear
    message if the path escapes the read window, so the agent can correct course next step.
    """
    p = Path(path)
    if not _under_roots(p):
        raise PermissionError(
            f"path escapes the allowed read window: {p} (allowed roots: "
            + ", ".join(str(r) for r in _READ_ROOTS) + ")")
    return p

def save_result(obj):
    """Persist a JSON-serializable result the parent reads back as result['result']."""
    with open("result.json", "w", encoding="utf-8") as _rf:
        json.dump(obj, _rf, default=str)

# ---- deterministic write-protection for the read roots ----------------------
# The data corpus is READ-ONLY. Since the agent writes its own code, a soft convention is not enough:
# intercept the standard write/mutation entry points HERE (before any agent code runs) and refuse any
# operation whose target resolves inside a read root — even if the agent bypasses open_data(). Reads
# are untouched; writes to the scratch dir (this process's cwd) and anywhere outside the read roots
# ($TMPDIR, the pip target) are untouched.
import builtins as _bi

_SCRATCH = Path.cwd().resolve()

def _deny_write(target):
    """Raise PermissionError if `target` resolves inside a read root (scratch dir always allowed)."""
    try:
        raw = target if isinstance(target, (bytes, bytearray)) else os.fspath(target)
    except TypeError:
        return  # not a path (e.g. an int file descriptor) — nothing to guard
    if isinstance(raw, (bytes, bytearray)):
        raw = os.fsdecode(raw)
    p = Path(raw).resolve()
    try:
        p.relative_to(_SCRATCH)   # writing inside the scratch dir is always allowed
        return
    except ValueError:
        pass
    if _under_roots(p):
        raise PermissionError(
            f"write blocked: {p} is inside a read-only data root. The data corpus is read-only; "
            "write results to the scratch dir (e.g. save_result(...)), never under the data roots.")

_real_open = _bi.open
def open(file, mode="r", *args, **kwargs):   # noqa: A001 — shadow builtin on purpose (sandbox)
    _writing = isinstance(mode, str) and any(c in mode for c in ("w", "a", "x", "+"))
    if _writing and isinstance(file, (str, bytes, bytearray, os.PathLike)):
        _deny_write(file)
    return _real_open(file, mode, *args, **kwargs)
_bi.open = open   # so pandas/openpyxl/csv/zipfile (all funnel through builtins.open) are covered too

_real_os_open = os.open
_O_WRITE = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
def _guarded_os_open(path, flags, *args, **kwargs):
    if flags & _O_WRITE:
        _deny_write(path)
    return _real_os_open(path, flags, *args, **kwargs)
os.open = _guarded_os_open

def _wrap1(_name):   # single-path mutators: remove/unlink/truncate/rmdir/mkdir
    _real = getattr(os, _name)
    def _w(path, *a, **k):
        _deny_write(path)
        return _real(path, *a, **k)
    setattr(os, _name, _w)

def _wrap2(_name):   # two-path mutators: rename/replace (src is removed, dst is created)
    _real = getattr(os, _name)
    def _w(src, dst, *a, **k):
        _deny_write(src)
        _deny_write(dst)
        return _real(src, dst, *a, **k)
    setattr(os, _name, _w)

for _fn in ("remove", "unlink", "truncate", "rmdir", "mkdir"):
    _wrap1(_fn)
for _fn in ("rename", "replace"):
    _wrap2(_fn)

# ---- deterministic exclusion of operator-response TABS ----------------------
# Same argument as the write guard above, applied to sheet access. The auditor's own workbook routinely
# carries one tab of the OPERATOR's argumentation (`AE 007 JADE and support - Auditor Reply.xlsx` has six
# tabs, one of them `Operator Response`), and that content must never reach the model. Since the agent
# writes its own code — it enumerates `wb.sheetnames` and calls `pd.read_excel` itself — a prompt
# instruction is not enough: patch the three access paths HERE, before any agent code runs, so a response
# tab is structurally absent rather than merely discouraged.
#
# `_RESPONSE_PATTERNS` is injected per run from extractor/response_names.py (this child cannot import the
# extractor — its cwd is the scratch dir), so the rule is not restated here, only re-applied.
import re as _re

_RESP_UNCOND = _re.compile(_RESPONSE_PATTERNS["unconditional"], _re.I | _re.X)
_RESP_QUALIFIED = _re.compile(_RESPONSE_PATTERNS["qualified"], _re.I | _re.X)
_RESP_AUDITOR_OWNED = _re.compile(_RESPONSE_PATTERNS["auditor_owned"], _re.I)

def _is_response_tab(name) -> bool:
    """True if this sheet name is the OPERATOR's content. Auditor-attributed names are auditor-side."""
    n = str(name or "")
    return bool(_RESP_UNCOND.search(n)) or bool(
        _RESP_QUALIFIED.search(_RESP_AUDITOR_OWNED.sub(" ", n)))

_TAB_NOTED = set()

def _allowed_tabs(names, source=""):
    """(allowed, excluded) sheet names. Never excludes every tab; explains itself once per workbook."""
    names = [str(n) for n in names]
    excluded = [n for n in names if _is_response_tab(n)]
    if not excluded:
        return names, []
    kept = [n for n in names if not _is_response_tab(n)]
    first_time = source not in _TAB_NOTED
    _TAB_NOTED.add(source)
    if not kept:
        # Same guard as the extractor: a workbook that is ALL response tabs would otherwise become
        # unreadable, which is worse than reading it. Keep everything and say so.
        if first_time:
            print(f"# WARN: every tab in {source or 'this workbook'} is operator-response content "
                  f"({', '.join(excluded)}) — keeping them all, since excluding them would leave no "
                  f"sheets to read.")
        return names, []
    if first_time:
        print(f"# excluded operator-response tab(s) from {source or 'this workbook'}: "
              f"{', '.join(excluded)} — the operator's argumentation is out of scope. Their absence is "
              f"intentional, not a bad path; readable tabs: {', '.join(kept)}")
    return kept, excluded

def _tab_denied(names, allowed, source=""):
    return PermissionError(
        f"tab(s) {', '.join(str(n) for n in names)} in {source or 'this workbook'} are excluded as "
        "operator-response content (the operator's reply is out of scope for this analysis). Readable "
        "tabs: " + (", ".join(allowed) if allowed else "(none listed)"))

try:
    import openpyxl as _oxl

    _real_load_workbook = _oxl.load_workbook

    def load_workbook(*args, **kwargs):
        """openpyxl.load_workbook with operator-response sheets removed from the returned workbook.

        Removing the sheet objects (rather than filtering a name list) is what makes the exclusion hold
        for every view at once: `wb.sheetnames`, `wb.worksheets` and `wb[name]` all agree the tab does
        not exist. Works under read_only=True too — both views derive from `Workbook._sheets`.
        """
        wb = _real_load_workbook(*args, **kwargs)
        src = kwargs.get("filename", args[0] if args else "")
        try:
            source = Path(os.fspath(src)).name
        except TypeError:
            source = ""
        allowed, excluded = _allowed_tabs(wb.sheetnames, source)
        if excluded:
            keep = set(allowed)
            for _ws in [s for s in list(wb.worksheets) if s.title not in keep]:
                try:
                    wb.remove(_ws)
                except (ValueError, KeyError):
                    pass
        return wb

    _oxl.load_workbook = load_workbook
    try:   # the same function is re-exported here; agents occasionally import it from the reader
        import openpyxl.reader.excel as _oxl_reader
        _oxl_reader.load_workbook = load_workbook
    except ImportError:
        pass
except ImportError:
    pass

try:
    import pandas as _pd

    def _pd_source(io):
        for cand in (io, getattr(io, "io", None), getattr(io, "_io", None)):
            try:
                return Path(os.fspath(cand)).name
            except TypeError:
                continue
        return ""

    _real_sheet_names = _pd.ExcelFile.sheet_names   # the real property, before patching
    _SHEET_VIEWS = {}

    def _sheet_view(io):
        """(all names, allowed names) for `io` — None when they can't be resolved (buffer/stream)."""
        if isinstance(io, _pd.ExcelFile):
            names = list(_real_sheet_names.fget(io))
            return names, _allowed_tabs(names, _pd_source(io))[0]
        try:
            key = str(Path(os.fspath(io)).resolve())
        except TypeError:
            return None
        if key not in _SHEET_VIEWS:
            try:
                _xl = _pd.ExcelFile(io)
            except Exception:
                return None
            try:
                names = list(_real_sheet_names.fget(_xl))
            finally:
                try:
                    _xl.close()
                except Exception:
                    pass
            _SHEET_VIEWS[key] = (names, _allowed_tabs(names, Path(key).name)[0])
        return _SHEET_VIEWS[key]

    def _guarded_sheet_names(self):
        return _allowed_tabs(_real_sheet_names.fget(self), _pd_source(self))[0]
    _pd.ExcelFile.sheet_names = property(_guarded_sheet_names)

    def _resolve_sheet(sheet_name, allowed, source):
        """Map a caller's sheet selector onto the allowed sheets, or refuse it.

        A NAME that is a response tab is refused — unless it survived the all-response guard and is in
        `allowed`, in which case refusing it would contradict the decision to keep it. An INDEX is
        resolved against the ALLOWED list rather than the real one — that is what a hidden sheet means:
        `sheet_name=1` should be the second readable sheet, not silently the excluded one.
        """
        if isinstance(sheet_name, str):
            if _is_response_tab(sheet_name) and sheet_name not in (allowed or ()):
                raise _tab_denied([sheet_name], allowed, source)
            return sheet_name
        if isinstance(sheet_name, bool) or not isinstance(sheet_name, int) or allowed is None:
            return sheet_name
        if sheet_name < len(allowed):
            return allowed[sheet_name]
        raise IndexError(
            f"sheet index {sheet_name} is out of range for {source or 'this workbook'}: only "
            f"{len(allowed)} readable sheet(s) ({', '.join(allowed)}) — operator-response tabs are "
            "excluded, so indices refer to the readable sheets only.")

    def _guard_selector(sheet_name, allowed, source):
        if isinstance(sheet_name, (list, tuple)):
            return type(sheet_name)(_resolve_sheet(s, allowed, source) for s in sheet_name)
        return _resolve_sheet(sheet_name, allowed, source)

    def _drop_response_frames(frames, allowed, source):
        """sheet_name=None returns {tab: DataFrame} — drop the excluded keys from it."""
        if not isinstance(frames, dict):
            return frames
        keep = set(allowed if allowed is not None else _allowed_tabs(list(frames), source)[0])
        return {k: v for k, v in frames.items() if str(k) in keep}

    _real_read_excel = _pd.read_excel

    def read_excel(io, sheet_name=0, **kwargs):
        """pandas.read_excel that cannot return operator-response sheets."""
        view = _sheet_view(io)
        allowed = view[1] if view else None
        source = _pd_source(io)
        if sheet_name is None:
            return _drop_response_frames(_real_read_excel(io, None, **kwargs), allowed, source)
        return _real_read_excel(io, _guard_selector(sheet_name, allowed, source), **kwargs)

    _pd.read_excel = read_excel

    _real_parse = _pd.ExcelFile.parse

    def _guarded_parse(self, sheet_name=0, *args, **kwargs):
        allowed = _allowed_tabs(_real_sheet_names.fget(self), _pd_source(self))[0]
        source = _pd_source(self)
        if sheet_name is None:
            return _drop_response_frames(_real_parse(self, None, *args, **kwargs), allowed, source)
        return _real_parse(self, _guard_selector(sheet_name, allowed, source), *args, **kwargs)
    _pd.ExcelFile.parse = _guarded_parse
except ImportError:
    pass

def list_sheets(path):
    """The readable sheet names of a workbook — operator-response tabs already excluded.

    Use this instead of hand-rolling `load_workbook(path).sheetnames`; it is the cheapest way to see what
    a workbook offers, and it prints why any tab is missing.
    """
    p = open_data(path)
    try:
        _wb = _real_load_workbook(p, read_only=True, data_only=True)
    except NameError:   # openpyxl unavailable
        return list(_pd.ExcelFile(p).sheet_names)
    try:
        return _allowed_tabs(_wb.sheetnames, Path(p).name)[0]
    finally:
        _wb.close()


# The workbook manifest, read off disk before this loop began (see otl/workbook_manifest.py) and staged
# here as manifest.json. Reading it costs nothing: it never passes through the model.
_MANIFEST_CACHE = None


def manifest(path=None):
    """The workbook manifest, or one workbook's entry when `path` is given.

    A workbook entry holds: path, engine, size_bytes, n_sheets, sheets, hidden_response_tabs, shapes
    (tabs grouped by header row + column labels) and tabs (each tab's shape id, header_row and
    confidence). The whole manifest also holds directories, unreadable and warnings.

    Returns {} when no manifest was staged for this node, so code written against it still runs.
    """
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is None:
        try:
            with open("manifest.json", encoding="utf-8") as fh:
                _MANIFEST_CACHE = json.load(fh)
        except Exception:
            _MANIFEST_CACHE = {}
    if path is None:
        return _MANIFEST_CACHE
    want = str(path)
    base = Path(want).name.lower()
    for _wb in (_MANIFEST_CACHE.get("workbooks") or []):
        if _wb.get("path") == want or Path(_wb.get("path", "")).name.lower() == base:
            return _wb
    return {}


def find_tab(person, path=None):
    """The tabs that could belong to `person`, each with its shape. Restrict to one workbook with `path`.

    Each hit is {"workbook", "path", "tab", "shape", "header_row", "confidence", "header", "ambiguous"}.
    `header_row` is 0-based, and None means the tab has no header row — read it by column index.
    `ambiguous` is True when the surname matched more than one tab (e.g. "Adams" against "Adams N" and
    "Adams S"); confirm which one before you use it rather than taking the first.

    Falls back to a surname match over the manifest's tab names when the person was not in the list the
    manifest was built with. Returns [] when no manifest was staged.
    """
    _m = manifest()
    base = Path(str(path)).name.lower() if path else ""
    _toks = [t for t in _re.split(r"[^A-Za-z]+", str(person or "").lower()) if t]
    _surname = _toks[-1] if _toks else ""
    hits = []
    for _wb in (_m.get("workbooks") or []):
        if base and Path(_wb.get("path", "")).name.lower() != base:
            continue
        _shapes = {s.get("id"): s for s in (_wb.get("shapes") or [])}
        _found, _ambiguous = None, False
        for _name, _hit in (_wb.get("person_tabs") or {}).items():
            if str(_name).strip().lower() == str(person or "").strip().lower():
                _found, _ambiguous = list(_hit.get("tabs") or []), bool(_hit.get("ambiguous"))
                break
        if _found is None:      # not pre-resolved — match the surname against the tab names
            _found = [t for t in (_wb.get("tabs") or {})
                      if _surname and _surname in _re.split(r"[^A-Za-z]+", t.lower())]
            _ambiguous = len(_found) > 1
        for _tab in _found:
            _info = (_wb.get("tabs") or {}).get(_tab) or {}
            _shape = _shapes.get(_info.get("shape")) or {}
            hits.append({"workbook": _wb.get("name", ""), "path": _wb.get("path", ""), "tab": _tab,
                         "shape": _info.get("shape"), "header_row": _info.get("header_row"),
                         "confidence": _info.get("confidence"), "header": _shape.get("header") or [],
                         "ambiguous": _ambiguous})
    return hits


# The person index, built off disk before this loop began (see otl/person_index.py) and staged here as
# person_index.json. Reading it costs nothing: it never passes through the model.
_PERSON_INDEX_CACHE = None


def person_index(person=None, role=None):
    """The names the disputed people go by in these workbooks, their rows, and the column map.

    The whole index holds `people`, `sources` and `unreadable`. One source is one kind of file — "pob",
    "heli" or "staff" — and holds:

      * `entries`  — one per name-bearing tab: path, name, tab, header_row, header, subject
        ({kind, labels, indices}; kind is "combined" or "first_last"), columns (role -> {label: column
        index}), unclassified_nameish, rows, row_data (name in file -> [row as {label: value}]) and
        row_counts (the TRUE count, which is larger than row_data when a name was truncated).
      * `selected` — person -> [{name_in_file, confidence, why, rows, tabs}]. These spellings were read
        off the file, so they are exact.
      * `unselected` — person -> {pool, pool_total} for a person nothing was selected for: the values
        sharing their surname, so "absent" can be told apart from "not looked for".
      * `skipped_tabs` — [{name, tab, reason}]. A tab with no name column or no header row.

    `person` filters `selected` / `row_data` down to one person; `role` filters to one source. Returns {}
    when no index was staged, so code written against it still runs.
    """
    global _PERSON_INDEX_CACHE
    if _PERSON_INDEX_CACHE is None:
        try:
            with open("person_index.json", encoding="utf-8") as fh:
                _PERSON_INDEX_CACHE = json.load(fh)
        except Exception:
            _PERSON_INDEX_CACHE = {}
    _idx = _PERSON_INDEX_CACHE
    if not _idx:
        return {}
    _sources = [s for s in (_idx.get("sources") or [])
                if not role or str(s.get("role", "")) == str(role)]
    if person is None:
        return dict(_idx, sources=_sources) if role else _idx

    _want = str(person).strip().lower()
    _out = []
    for _s in _sources:
        _matches = []
        for _p, _ms in (_s.get("selected") or {}).items():
            if str(_p).strip().lower() == _want:
                _matches = list(_ms)
                break
        _names = {m.get("name_in_file") for m in _matches}
        _entries = []
        for _e in (_s.get("entries") or []):
            _rows = {n: r for n, r in (_e.get("row_data") or {}).items() if n in _names}
            _entries.append(dict(_e, row_data=_rows,
                                 row_counts={n: c for n, c in (_e.get("row_counts") or {}).items()
                                             if n in _names}))
        _miss = (_s.get("unselected") or {}).get(person) or {}
        _out.append(dict(_s, selected={person: _matches} if _matches else {},
                         unselected={person: _miss} if _miss else {}, entries=_entries))
    return {"people": _idx.get("people") or [], "sources": _out,
            "unreadable": _idx.get("unreadable") or []}

# ----- agent code below -----------------------------------------------------
'''


class CodeExecutor:
    """Runs agent-generated Python in a read-scoped, scratch-confined child process."""

    def __init__(self, run_id: str, agent: str, read_roots: list[str | Path] | None = None,
                 scratch_base: Path | None = None, manifest: dict | None = None,
                 person_index: dict | None = None):
        self.run_id = run_id or "run"
        self.agent = agent
        # The workbook manifest (otl/workbook_manifest.build_manifest), staged for the child as
        # manifest.json. It never passes through the model, so this channel costs no tokens; the
        # `manifest()` / `find_tab()` helpers in _PREAMBLE read it. None means the node passes none, and
        # those helpers then return {} / [] instead of failing.
        self.manifest = manifest or None
        # The person index (otl/person_index.index_json), staged as person_index.json through the same
        # zero-token channel and read by the `person_index()` helper. This is the one the agent reaches
        # for most: it holds each disputed person's spelling in the file, their rows keyed by column
        # label, and the column-role map for the tab those rows came from.
        self.person_index = person_index or None
        roots = list(read_roots or [])
        roots.append(_data_root())          # always allow the global data window
        # de-dup while preserving order
        seen: set[str] = set()
        self.read_roots: list[str] = []
        for r in roots:
            s = str(Path(r).resolve())
            if s not in seen:
                seen.add(s)
                self.read_roots.append(s)
        self.scratch = self._make_scratch(scratch_base)

    def _make_scratch(self, scratch_base: Path | None) -> Path:
        base = scratch_base or (ORCH_DIR / ".otl_scratch")
        target = base / self.run_id / self.agent
        try:
            target.mkdir(parents=True, exist_ok=True)
            return target
        except OSError:  # ORCH_DIR not writable — fall back to the system temp dir
            tmp = Path(os.getenv("TMPDIR", "/tmp")) / "otl_scratch" / self.run_id / self.agent
            tmp.mkdir(parents=True, exist_ok=True)
            return tmp

    def run(self, code: str, inputs: dict | None = None, timeout: int = 150) -> dict:
        """Execute `code` with `inputs` available as INPUTS; return a structured result dict.

        Returns:
          {ok: bool, stdout: str, stderr: str, result: <save_result payload | None>,
           error: str | None, timed_out: bool}
        `ok` is True only when the process exits 0 within the timeout.
        """
        (self.scratch / "inputs.json").write_text(json.dumps(inputs or {}, default=str),
                                                  encoding="utf-8")
        # The manifest is staged the same way, and rewritten each run so an edited workbook cannot leave
        # a stale shape behind. A node that passes no manifest leaves no file, which is what the
        # `manifest()` helper treats as "nothing staged".
        manifest_path = self.scratch / "manifest.json"
        if self.manifest:
            manifest_path.write_text(json.dumps(self.manifest, default=str), encoding="utf-8")
        elif manifest_path.exists():
            manifest_path.unlink()
        # Same rule for the person index: rewritten when present, removed when not, so a node that runs
        # with OTL_PERSON_INDEX off cannot read a previous run's selection.
        index_path = self.scratch / "person_index.json"
        if self.person_index:
            index_path.write_text(json.dumps(self.person_index, default=str), encoding="utf-8")
        elif index_path.exists():
            index_path.unlink()
        # A fresh result.json each run so a previous step's payload can't be mistaken for this one's.
        result_path = self.scratch / "result.json"
        if result_path.exists():
            result_path.unlink()
        header = (f"_READ_ROOTS_RAW = {json.dumps(self.read_roots)}\n"
                  f"_RESPONSE_PATTERNS = {json.dumps(_RESPONSE_PATTERNS)}\n")
        script = header + _PREAMBLE + "\n" + code
        (self.scratch / "script.py").write_text(script, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, "script.py"],
                cwd=str(self.scratch),
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return {"ok": False, "stdout": exc.stdout or "", "stderr": exc.stderr or "",
                    "result": None, "error": f"timed out after {timeout}s", "timed_out": True}

        result = None
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                return {"ok": False, "stdout": proc.stdout, "stderr": proc.stderr,
                        "result": None, "error": f"result.json not valid JSON: {exc}",
                        "timed_out": False}

        ok = proc.returncode == 0
        return {"ok": ok, "stdout": proc.stdout, "stderr": proc.stderr, "result": result,
                "error": None if ok else f"exit {proc.returncode}", "timed_out": False}

    def pip_install(self, pkgs: list[str], timeout: int = 300) -> dict:
        """Best-effort `pip install` into the running interpreter (needed only on the user's machine;
        network egress may be blocked in some sandboxes)."""
        if not pkgs:
            return {"ok": True, "stdout": "", "stderr": "", "error": None, "timed_out": False}
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *pkgs],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "stderr": "", "error": f"pip timed out after {timeout}s",
                    "timed_out": True}
        ok = proc.returncode == 0
        return {"ok": ok, "stdout": proc.stdout, "stderr": proc.stderr,
                "error": None if ok else f"pip exit {proc.returncode}", "timed_out": False}


__all__ = ["CodeExecutor"]
