"""Deterministic column profiling for extracted sheets — computed in code, over every row.

Why this exists. Prompts show the model a bounded SAMPLE of a tab's rows, but ask it for facts about
the whole population: what period range the exhibit covers, what the charges total. On
GreatWhite_2023_0003 the sample stopped early and the synthesis reported the charge window as March to
August 2022 against a real span of March to December, and $27,499.67 against a real total of
$47,365.25. Both answers looked complete. Neither was.

A range or a total is arithmetic, so no model should be inferring it from a sample. This module
computes those facts in code over every row and hands them to the prompt as ground truth, leaving the
sampled rows to do what they are actually good at — showing what a row looks like.

Schema-agnostic by design. Nothing here matches a column NAME. A column's kind comes from the shape of
its values, so an unstandardized sheet with invented headers profiles as well as a JADE export. The one
piece of domain knowledge is a shape, not a name: SAP writes fiscal periods as `PPP.YYYY` (`003.2022` =
period 3 of 2022, running to 016 for special periods), which shares a separator with the `DD.MM.YYYY`
dates in the same exports. Part count separates them — two dot-parts is a period, three is a date — so
`003.2022` is never read as a day in March.

Total rows are detected and EXCLUDED from sums, then reported alongside them. A trailing summary row is
ordinary in an auditor exhibit, and summing straight through it doubles every figure: on that same
sheet a naive `Cost + OH` sum returns $94,730.50. Reporting the computed sum and the sheet's own stated
total side by side also turns the reconciliation into something the model can check rather than assert.

Cost. A skill run pulls up to 200,000 rows, so this streams. Each column's KIND is decided from a
strided sample (a few hundred rows spanning the whole sheet), and the statistics then come from one
pass that runs a single parser per cell and accumulates into fixed-size state — values are never
materialized. Profiling all of a 200k x 72 ledger costs a few seconds rather than the ~40 a
classify-everything-twice implementation took.
"""
from __future__ import annotations

import datetime as _dt
import re
import statistics

# A fiscal period (2 dot-parts, leader 001-016) — tested BEFORE any date shape, since `003.2022` and
# `31.03.2022` share a separator and only the part count tells them apart.
_RE_PERIOD = re.compile(r"^(0\d{2}|1[0-6])\.((?:19|20)\d{2})$")
_RE_DOTTED = re.compile(r"^(\d{1,2})\.(\d{1,2})\.((?:19|20)\d{2})$")        # SAP text export, DD.MM.YYYY
_RE_ISO = re.compile(r"^((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})(?:[T ]|$)")
_RE_SLASH = re.compile(r"^(\d{1,2})/(\d{1,2})/((?:19|20)\d{2})(?:[T ]|$)")  # US convention, MM/DD/YYYY
# A number that arrived as text: "1,224.00", "$ 1,224", "(1,224.00)" for a credit.
_RE_NUMERIC_TEXT = re.compile(r"^\(?\s*[-+]?[$€£]?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*\)?$|"
                              r"^\(?\s*[-+]?[$€£]?\s*\d+(?:\.\d+)?\s*\)?$")

_MIN_VALUES_TO_CLASSIFY = 2   # with the 90% rule below, 2 values must agree unanimously
_KIND_AGREEMENT = 0.90        # share of sampled non-empty values that must share a shape
_KIND_SAMPLE_ROWS = 600       # rows sampled to decide kinds — strided, so it spans the whole sheet
_ENUMERATE_MAX = 12           # text columns with at most this many distinct values are listed in full
_PERIOD_ENUMERATE_MAX = 24    # periods are the reason this module exists — list more of them
_DISTINCT_CAP = 5000          # ceiling on tracked distinct values, so a 200k-row column stays bounded
_TOTAL_ROW_CANDIDATES = 4     # how many trailing rows may be considered a totals row
_MIN_ROWS_ABOVE_TOTAL = 2     # unlabelled arithmetic needs several rows to be summing anything
# Three tolerances, because finding a totals row and reporting on it want opposite things.
# DETECT is loose enough to absorb the exhibit's own rounding. NEAR is far looser, and exists because a
# trailing row whose figure is 6% off the rows above it is still a totals row — one with a WRONG total,
# which is worth surfacing. Failing to recognize it is the worse error: we would sum straight through it
# and report double. REPORT is a cent, because once the row is found, "matches" has to mean matches —
# 0.5% of a $47,000 total is $237, ample room for a real coding error to hide in.
_TOTAL_DETECT_REL_TOL = 0.005
_TOTAL_DETECT_ABS_TOL = 0.01
_TOTAL_NEAR_REL_TOL = 0.20
_TOTAL_REPORT_TOL = 0.01
# Value-based, not header-based: this reads the row's own cells, so it stays schema-agnostic. It is only
# ever a corroborating signal, never sufficient on its own, so missing a label costs nothing.
_SUMMARY_LABELS = ("total", "sum", "sums")

_FULL_PASS_KINDS = ("period", "date", "number")   # kinds whose stats must cover every row


def _is_blank(v) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def _parse_period(v):
    """`'003.2022'` -> `(2022, 3)`; None if not a fiscal period."""
    if not isinstance(v, str):
        return None
    m = _RE_PERIOD.match(v.strip())
    return (int(m.group(2)), int(m.group(1))) if m else None


def _parse_date(v):
    """Parse a date from a native object or any of the shapes above. Returns a `date`, or None.

    Dot-separated forms are read as DD.MM.YYYY (the SAP text-export convention) and slash-separated
    forms as MM/DD/YYYY (the US convention in the auditor's own exhibits). When the primary reading
    yields an impossible calendar date the other order is tried, so `03.31.2022` still parses rather
    than being dropped as junk.
    """
    if isinstance(v, _dt.datetime):
        return v.date()
    if isinstance(v, _dt.date):
        return v
    if not isinstance(v, str):
        return None
    s = v.strip()
    m = _RE_ISO.match(s)
    if m:
        return _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    for rx, day_first in ((_RE_DOTTED, True), (_RE_SLASH, False)):
        m = rx.match(s)
        if not m:
            continue
        a, b, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        d, mo = (a, b) if day_first else (b, a)
        return _safe_date(year, mo, d) or _safe_date(year, d, mo)
    return None


def _safe_date(y: int, m: int, d: int):
    try:
        return _dt.date(y, m, d)
    except ValueError:
        return None


def _parse_number(v):
    """Return `(value, was_text)`, or None. Bools are not numbers here even though Python says so."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return (float(v), False)
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s or not _RE_NUMERIC_TEXT.match(s):
        return None
    neg = s.startswith("(") and s.endswith(")")          # accounting notation for a credit
    body = s.strip("()").replace(",", "").replace("$", "").replace("€", "").replace("£", "").strip()
    try:
        n = float(body)
    except ValueError:
        return None
    return (-n if neg else n, True)


_PARSERS = {"period": _parse_period, "date": _parse_date, "number": _parse_number}


def _classify(v) -> tuple[str, object] | None:
    """Classify ONE value by shape. Returns `(kind, parsed)`, or None for a blank.

    Order matters: period before date (shared separator), and number last so a bare year like `2022`
    in a date column does not win the column for `number`.
    """
    if _is_blank(v):
        return None
    p = _parse_period(v)
    if p is not None:
        return ("period", p)
    d = _parse_date(v)
    if d is not None:
        return ("date", d)
    n = _parse_number(v)
    if n is not None:
        return ("number", n)
    return ("text", str(v).strip())


def _cell(row: list, i: int):
    return row[i] if i < len(row) else None


def _fmt_num(x: float) -> str:
    """Money keeps both decimals — `6,666.60`, not `6,666.6`. Whole numbers stay whole."""
    x = round(x + 0.0, 2)
    return f"{x:,.2f}" if x % 1 else f"{x:,.0f}"


def _fmt_period(p: tuple[int, int]) -> str:
    return f"{p[1]:03d}.{p[0]}"


def _display(kind: str, parsed) -> str:
    """Render a parsed value back to a readable string, for the enumerated/example lists."""
    if kind == "number":
        return _fmt_num(parsed[0])
    if kind == "period":
        return _fmt_period(parsed)
    if kind == "date":
        return parsed.isoformat()
    return str(parsed)


def _close(a: float, b: float) -> bool:
    """Loose comparison, for DECIDING whether a trailing row is a totals row."""
    return abs(a - b) <= max(_TOTAL_DETECT_ABS_TOL, _TOTAL_DETECT_REL_TOL * max(abs(a), abs(b)))


def _near(a: float, b: float) -> bool:
    """Same order of magnitude as the sum above — a wrong total, not an unrelated stray figure."""
    return abs(a - b) <= _TOTAL_NEAR_REL_TOL * max(abs(a), abs(b), 1.0)


def _has_summary_label(row: list, width: int) -> bool:
    """True when some text cell in the row reads like a summary label ("Total", "Sum")."""
    for i in range(min(width, len(row))):
        v = row[i]
        if isinstance(v, str):
            s = v.strip().lower().rstrip(":").strip()
            if "total" in s or s in _SUMMARY_LABELS:
                return True
    return False


def _sample_indices(n: int, cap: int) -> list[int]:
    """Evenly strided row indices spanning the WHOLE sheet — never a head slice.

    A head slice is biased in exactly the way that produced the original bug: on a date-ordered ledger
    the first N rows share the earliest periods, so a column of late-year dates could be classified
    from values that all sit in one quarter.
    """
    if n <= cap:
        return list(range(n))
    step = n / cap
    return sorted({int(i * step) for i in range(cap)})


class _Acc:
    """Fixed-size accumulator for one column, so a 200k-row ledger never materializes its values.

    Values are only ever added, never removed: rows in the trailing totals-row window are held aside
    until their status is decided, then the ones that turned out to be ordinary data are added. That
    keeps min/max exact, which a subtract-afterwards approach could not.
    """

    __slots__ = ("kind", "count", "total", "lo", "hi", "tally", "capped", "text_formatted")

    def __init__(self, kind: str):
        self.kind = kind
        self.count = 0
        self.total = 0.0
        self.lo = None
        self.hi = None
        self.tally: dict = {}
        self.capped = False
        self.text_formatted = False

    def add(self, parsed) -> None:
        self.count += 1
        if self.kind == "number":
            val, was_text = parsed
            self.total += val
            if was_text:
                self.text_formatted = True
        else:
            val = parsed
        if self.lo is None or val < self.lo:
            self.lo = val
        if self.hi is None or val > self.hi:
            self.hi = val
        if len(self.tally) < _DISTINCT_CAP or val in self.tally:
            self.tally[val] = self.tally.get(val, 0) + 1
        else:
            self.capped = True


class Shapes:
    """What a strided sample says about a set of rows: each column's kind, plus the typical density.

    Shared so that block segmentation and column profiling decide "what shape is this column" the same
    way. If they drifted apart, a column could be a date to one and text to the other, and the two
    would disagree about where a table ends.
    """

    __slots__ = ("kinds", "sample_vals", "sample_idx", "median_pop", "sampled")

    def __init__(self, kinds, sample_vals, sample_idx, median_pop, sampled):
        self.kinds = kinds              # {col -> "period"|"date"|"number"|"mixed"|"text"|"empty"}
        self.sample_vals = sample_vals  # {col -> [(row_idx, (kind, parsed)), ...]}
        self.sample_idx = sample_idx    # the rows the sample visited
        self.median_pop = median_pop    # median cells filled per non-empty row
        self.sampled = sampled          # True when the sample is narrower than the row set


def classify_shapes(data_rows: list[list], populated_width: int,
                    cap: int = _KIND_SAMPLE_ROWS) -> Shapes:
    """Decide every column's kind from a strided sample, and measure the typical row density.

    Sampling is what keeps this affordable on a 200k-row ledger: what a column IS does not need every
    row, only a spread of them. The stride spans the whole row set, never a head slice — see
    `_sample_indices` for why that distinction caused the bug this module exists for.
    """
    n = len(data_rows)
    if populated_width <= 0 or n == 0:
        return Shapes({}, {}, [], 0, False)
    sample_idx = _sample_indices(n, cap)
    kinds: dict[int, str] = {}
    sample_vals: dict[int, list] = {}
    for i in range(populated_width):
        # Row index is kept alongside each classified value so callers can drop the totals row from
        # text-column statistics too — otherwise a summary row's "Total" label lands in the column's
        # distinct-value list and its row count disagrees with the typed columns'.
        vals = [(j, c) for j in sample_idx for c in (_classify(_cell(data_rows[j], i)),)
                if c is not None]
        sample_vals[i] = vals
        if len(vals) < _MIN_VALUES_TO_CLASSIFY:
            kinds[i] = "text" if vals else "empty"
            continue
        tally: dict[str, int] = {}
        for _j, (k, _p) in vals:
            tally[k] = tally.get(k, 0) + 1
        best, hits = max(tally.items(), key=lambda kv: kv[1])
        kinds[i] = best if hits / len(vals) >= _KIND_AGREEMENT else "mixed"
    pop_counts = [sum(1 for i in range(populated_width) if not _is_blank(_cell(data_rows[j], i)))
                  for j in sample_idx]
    nonzero = [c for c in pop_counts if c]
    return Shapes(kinds, sample_vals, sample_idx,
                  statistics.median(nonzero) if nonzero else 0, len(sample_idx) < n)


def profile_columns(header: list[str], data_rows: list[list], populated_width: int
                    ) -> tuple[list[dict], list[dict]]:
    """Profile every populated column. Returns `(profiles, total_rows)`.

    Four stages. Kinds come from a strided sample, because what a column IS does not need every row.
    Statistics then stream over every row for the kinds that carry ranges and totals. The trailing rows
    that might be a totals row are held aside during that pass, judged against the accumulated sums,
    and either recorded as totals or folded in as ordinary data.
    """
    n = len(data_rows)
    if populated_width <= 0 or n == 0:
        return [], []

    # -- stage 1: kinds, and the sheet's typical row density, from a strided sample ------------------
    shapes = classify_shapes(data_rows, populated_width)
    sample_idx, sampled = shapes.sample_idx, shapes.sampled
    kinds, sample_vals, median_pop = shapes.kinds, shapes.sample_vals, shapes.median_pop

    # -- stage 2: one streaming pass for the kinds whose stats must cover every row ------------------
    accs = {i: _Acc(kinds[i]) for i in range(populated_width) if kinds[i] in _FULL_PASS_KINDS}
    plan = [(i, _PARSERS[a.kind], a) for i, a in accs.items()]   # hoisted out of the row loop
    window_start = max(0, n - _TOTAL_ROW_CANDIDATES)
    for j in range(window_start):
        row = data_rows[j]
        for i, parse, acc in plan:
            v = row[i] if i < len(row) else None
            if v is None or (isinstance(v, str) and not v.strip()):
                continue
            p = parse(v)
            if p is not None:
                acc.add(p)

    # -- stage 3: decide which trailing rows are totals rows -----------------------------------------
    window = list(range(window_start, n))
    wcount = {j: sum(1 for i in range(populated_width) if not _is_blank(_cell(data_rows[j], i)))
              for j in window}
    wvals = {j: {i: p for i, parse, _a in plan
                 for p in (parse(_cell(data_rows[j], i)),) if p is not None}
             for j in window}
    numeric_cols = [i for i, a in accs.items() if a.kind == "number"]
    confirmed: dict[int, dict] = {}
    # OLDEST-first. A "subtotal then grand total" tail only resolves in this order: the subtotal is
    # confirmed first, and is then left out of the sum the grand total is compared against. Working
    # newest-first counts the subtotal in as data, so the grand total misses by exactly the subtotal
    # and neither row is ever found.
    for j in window:
        if not wcount[j]:
            continue
        # How many data rows the candidate is being compared against. A totals row is a row whose value
        # equals the sum of SEVERAL rows. With one row above, "the sum above" IS that row, so equality
        # only means the candidate repeats it — the opposite of a totals row. The `Oyler` trip table
        # shows why this has to be checked: three trips sharing an asset id, call sign and trip id made
        # the second trip match the first on eight columns and read as a high-confidence totals row,
        # which then excluded a real trip from every figure. Column agreement is no corroboration when
        # the agreement comes from repetition. A summary label still is, so a labelled row is exempt.
        rows_above = window_start + sum(1 for k in window if k < j and k not in confirmed and wcount[k])
        if (rows_above < _MIN_ROWS_ABOVE_TOTAL
                and not _has_summary_label(data_rows[j], populated_width)):
            continue
        matched, near = [], []
        for c in numeric_cols:
            cand = wvals[j].get(c)
            if cand is None:
                continue
            above = accs[c].total + sum(wvals[k][c][0] for k in window
                                       if k < j and k not in confirmed and c in wvals[k])
            if _close(cand[0], above):
                matched.append(c)
            elif _near(cand[0], above):
                near.append(c)
        if not matched and not near:
            continue
        # Arithmetic alone is not proof. On a narrow sheet an ordinary final row can equal the sum above
        # by chance — 1, 2, 3 is followed by 6 — and wrongly excluding a data row understates the total.
        # So require a second signal: more than one column agreeing, a row that is thinner than the
        # sheet's typical row, or a summary label. The exception is a one-column sheet, where a totals
        # row has nowhere to look different and the arithmetic is the only signal there can be.
        sparse = wcount[j] < median_pop
        if not (len(matched) > 1 or sparse or median_pop <= 1
                or _has_summary_label(data_rows[j], populated_width)):
            continue
        confirmed[j] = {"index": j, "matched_cols": matched, "near_cols": near,
                        "confidence": "high" if len(matched) > 1 else "probable"}
    for j in window:                              # ordinary trailing rows join the statistics
        if j in confirmed:
            continue
        for i, p in wvals[j].items():
            accs[i].add(p)
    total_rows = [confirmed[j] for j in sorted(confirmed)]
    body_rows = n - len(confirmed)

    # -- stage 4: assemble ---------------------------------------------------------------------------
    profiles: list[dict] = []
    for i in range(populated_width):
        kind = kinds[i]
        if kind == "empty":
            continue
        prof = {
            "index": i,
            "col": i + 1,                                   # 1-based, matching raw_grid's c<idx>
            "header": str(header[i]) if i < len(header) else "",
            "kind": kind,
        }
        if kind in _FULL_PASS_KINDS:
            acc = accs[i]
            prof.update(count=acc.count, of_rows=body_rows, basis="all rows",
                        distinct=len(acc.tally) + (1 if acc.capped else 0))
            if acc.capped:
                prof["distinct_capped"] = True
            if acc.count == 0:
                prof["kind"] = kind = "text"
                prof.update(distinct=0)
            elif kind == "period":
                prof.update(min=_fmt_period(acc.lo), max=_fmt_period(acc.hi))
                if len(acc.tally) <= _PERIOD_ENUMERATE_MAX and not acc.capped:
                    prof["values"] = [_fmt_period(k) for k in sorted(acc.tally)]
            elif kind == "date":
                # min/max come from parsed keys, never lexical order: "31.03.2022" sorts before
                # "30.09.2021" as text, which would invert the range this module exists to fix.
                prof.update(min=acc.lo.isoformat(), max=acc.hi.isoformat())
            else:
                prof.update(sum=round(acc.total, 2), min=acc.lo, max=acc.hi)
                if acc.text_formatted:
                    prof["note"] = "values arrived as formatted text; parsed to numbers"
                stated = [wvals[t["index"]].get(i) for t in total_rows]
                stated = [s[0] for s in stated if s is not None]
                if stated:
                    st = round(stated[-1], 2)
                    prof["stated_total"] = st
                    prof["total_diff"] = round(st - acc.total, 2)
                    prof["reconciles"] = abs(st - acc.total) <= _TOTAL_REPORT_TOL
        else:
            vals = [(k, p) for j, (k, p) in sample_vals[i] if j not in confirmed]
            seen = len([j for j in sample_idx if j not in confirmed])
            prof.update(count=len(vals), of_rows=seen,
                        basis=(f"a {seen}-row sample spanning the sheet" if sampled
                               else "all rows"))
            distinct = sorted({_display(k, p) for k, p in vals})
            prof["distinct"] = len(distinct)
            if len(distinct) <= _ENUMERATE_MAX:
                prof["values"] = distinct
            elif vals:
                # Start, middle and end — a top-N sample is biased the same way a row prefix is.
                picks = [vals[0], vals[len(vals) // 2], vals[-1]]
                prof["examples"] = [_display(k, p) for k, p in picks]
        profiles.append(prof)
    return profiles, total_rows


def detect_total_rows(header: list[str], data_rows: list[list], populated_width: int) -> list[dict]:
    """Just the totals-row detection, for callers that do not want the whole profile."""
    return profile_columns(header, data_rows, populated_width)[1]


def _profile_line(p: dict) -> str:
    head = f"c{p['col']} {p['header']!r} {p['kind']} {p['count']}/{p['of_rows']}"
    if p.get("basis") and p["basis"] != "all rows":
        head += f" [from {p['basis']}]"
    k = p["kind"]
    if k == "period":
        s = f"{head}, {p['distinct']} distinct: {p['min']} -> {p['max']}  (PPP.YYYY = period.year)"
        if p.get("values"):
            s += f"\n     all periods: {', '.join(p['values'])}"
        return s
    if k == "date":
        return f"{head}, {p['distinct']} distinct: {p['min']} -> {p['max']}"
    if k == "number":
        s = f"{head}, sum={_fmt_num(p['sum'])} min={_fmt_num(p['min'])} max={_fmt_num(p['max'])}"
        if "stated_total" in p:
            s += f"; the sheet's own stated total={_fmt_num(p['stated_total'])}"
            if p.get("reconciles"):
                s += " (MATCHES the computed sum exactly)"
            else:
                s += (f" (does NOT match: the sheet's figure is "
                      f"{_fmt_num(abs(p['total_diff']))} "
                      f"{'higher' if p['total_diff'] > 0 else 'lower'} than the rows above it sum to)")
        if p.get("note"):
            s += f"; {p['note']}"
        return s
    if k == "mixed":
        s = (f"{head} — values in this column DISAGREE on shape, so it probably spans more than one "
             f"table; {p.get('distinct', 0)} distinct")
    else:
        s = f"{head}, {p.get('distinct', 0)} distinct"
    if p.get("values"):
        s += ": " + ", ".join(p["values"])
    elif p.get("examples"):
        s += "; e.g. " + " | ".join(p["examples"])
    return s


# Order for rendering: the kinds that carry ranges and totals first, since those are the facts a
# sampled table cannot supply. Never withhold a period, date or number column.
_KIND_ORDER = {"period": 0, "date": 1, "number": 2, "mixed": 3, "text": 4}
_ALWAYS_KEEP = ("period", "date", "number")
_MIXED_TAB_SHARE = 0.40   # this share of columns disagreeing on shape means stacked tables


def _sort_key(p: dict):
    """Ranges first, then the numeric columns that reconcile to a stated total, then the rest.

    A ledger can have a dozen numeric columns — document numbers, cost centres, rates — and only one
    or two the exhibit itself totals. Those are the reconciliation, so they lead.
    """
    return (_KIND_ORDER.get(p["kind"], 9), 0 if "stated_total" in p else 1, p["col"])


def render_profile(profiles: list[dict], total_rows: list[dict], max_chars: int = 4000) -> str:
    """Render the profile for a prompt. Empty string when there is nothing worth stating."""
    if not profiles:
        return ""
    ordered = sorted(profiles, key=_sort_key)
    n_rows = max((p.get("of_rows", 0) for p in profiles if p.get("basis") == "all rows"), default=0) \
        or profiles[0].get("of_rows", 0)
    head = (f"COLUMN PROFILE — computed in code over ALL {n_rows} data rows, not a sample. Ranges, "
            f"minima, maxima, counts and totals must be read from here, never inferred from the "
            f"sampled rows below.")
    # A tab whose columns mostly disagree on shape is not one table. Say so, rather than let per-column
    # ranges that straddle two unrelated blocks read as facts about a single population.
    mixed = sum(1 for p in profiles if p["kind"] == "mixed")
    if mixed >= max(2, _MIXED_TAB_SHARE * len(profiles)):
        head += ("\nWARNING: most columns on this tab hold values of disagreeing shapes, so the tab "
                 "almost certainly contains SEVERAL stacked tables rather than one. Treat the "
                 "per-column figures below as spanning unrelated blocks, and read the rows directly.")
    for t in total_rows:
        head += (f"\nData row {t['index'] + 1} is a TOTALS row ({t['confidence']} confidence) and is "
                 f"EXCLUDED from every sum below.")
        if not t.get("matched_cols"):
            head += (" Its own figures do NOT agree with the rows above it — see the per-column "
                     "comparison, and treat the discrepancy as a finding rather than a rounding note.")
    lines, used, held = [], 0, 0
    for p in ordered:
        line = "  " + _profile_line(p)
        if used + len(line) > max_chars and p["kind"] not in _ALWAYS_KEEP:
            held += 1
            continue
        lines.append(line)
        used += len(line) + 1
    if held:
        lines.append(f"  …({held} further low-signal text column(s) not profiled here)")
    return head + "\n" + "\n".join(lines)
