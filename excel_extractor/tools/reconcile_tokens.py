"""Check that the three views of a run's token usage agree.

Reads one OTL report plus its trace JSONL and asserts the invariants the accounting promises:

  1. Every trace record's `tokens` block for node N sums to `by_tier[*].by_node[N]`.
  2. Each tier's by_node entries sum to the tier's own counters.
  3. The per-tier counters sum to the run totals.
  4. Within every block, tier and node: the three input classes add up to `prompt_tokens`,
     `tokens_total` is `prompt_tokens + tokens_out`, and `thinking_tokens` is not more than
     `tokens_out`.
  5. The embedded `trace` array matches the standalone trace JSONL record for record.
  6. The cost block adds up, and the report's `estimated_cost_usd` mirrors it.

Run it as:

    .venv/bin/python excel_extractor/tools/reconcile_tokens.py <report.json> [trace.jsonl]

The trace path defaults to the report path with `_otl.json` swapped for `_otl_trace.jsonl`.
Exits non-zero on the first failure, with the two numbers that disagreed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# The counters that must add up. `prompt_tokens` and `tokens_total` are derived from these, and are
# checked separately by _check_derived.
SUM_FIELDS = ("calls", "tokens_in", "cache_read_tokens", "cache_creation_tokens",
              "tokens_out", "thinking_tokens")
DERIVED = ("prompt_tokens", "tokens_total")

USD_TOLERANCE = 1e-6


class Failure(Exception):
    pass


def _eq(what: str, got, want) -> None:
    if got != want:
        raise Failure(f"{what}: got {got}, expected {want}")


def _close(what: str, got: float, want: float) -> None:
    if abs(float(got) - float(want)) > USD_TOLERANCE:
        raise Failure(f"{what}: got {got}, expected {want}")


def _sum_blocks(blocks: list[dict]) -> dict:
    return {f: sum(int(b.get(f, 0)) for b in blocks) for f in SUM_FIELDS}


def _check_derived(what: str, d: dict) -> None:
    """The two derived fields, plus the thinking subset rule."""
    prompt = int(d["tokens_in"]) + int(d["cache_read_tokens"]) + int(d["cache_creation_tokens"])
    _eq(f"{what}.prompt_tokens", int(d["prompt_tokens"]), prompt)
    _eq(f"{what}.tokens_total", int(d["tokens_total"]), prompt + int(d["tokens_out"]))
    if int(d["thinking_tokens"]) > int(d["tokens_out"]):
        raise Failure(f"{what}: thinking_tokens {d['thinking_tokens']} exceeds "
                      f"tokens_out {d['tokens_out']}")


def trace_blocks_by_node(trace: list[dict]) -> dict[str, list[dict]]:
    """Every `tokens` block in the trace, grouped by the node it belongs to."""
    by_node: dict[str, list[dict]] = {}
    for record in trace:
        block = record.get("tokens")
        if not block:
            continue
        node = record.get("node") or record.get("agent") or ""
        by_node.setdefault(node, []).append(block)
    return by_node


def reconcile(report: dict, trace_file: list[dict]) -> list[str]:
    notes: list[str] = []
    usage = report["token_usage"]
    trace = report.get("trace") or []

    # 5. the embedded array and the standalone file are the same records
    _eq("trace record count (embedded vs file)", len(trace), len(trace_file))
    for i, (a, b) in enumerate(zip(trace, trace_file)):
        if a != b:
            raise Failure(f"trace record {i} differs between the report and the JSONL")
    notes.append(f"trace: {len(trace)} records, embedded array matches the JSONL exactly")

    # every trace block is internally consistent
    for record in trace:
        block = record.get("tokens")
        if block:
            _check_derived(f"trace[{record.get('event', '?')}].tokens", block)

    grouped = trace_blocks_by_node(trace)
    llm_calls = 0
    tiers = usage["by_tier"]

    for tier, t in tiers.items():
        _check_derived(f"by_tier[{tier}]", t)
        node_totals = {f: 0 for f in SUM_FIELDS}
        for node, n in t["by_node"].items():
            _check_derived(f"by_node[{tier}][{node}]", n)
            # 1. the node's trace blocks sum to its by_node entry
            summed = _sum_blocks(grouped.get(node, []))
            for f in SUM_FIELDS:
                _eq(f"trace sum vs by_node[{tier}][{node}].{f}", summed[f], int(n[f]))
            for f in SUM_FIELDS:
                node_totals[f] += int(n[f])
            llm_calls += int(n["calls"])
        # 2. the by_node entries sum to the tier
        for f in SUM_FIELDS:
            _eq(f"by_node sum vs by_tier[{tier}].{f}", node_totals[f], int(t[f]))
        notes.append(f"{tier:>5}: {t['calls']:>3} calls over {len(t['by_node'])} node(s), "
                     f"new_in={t['tokens_in']:,} read={t['cache_read_tokens']:,} "
                     f"write={t['cache_creation_tokens']:,} prompt={t['prompt_tokens']:,} "
                     f"out={t['tokens_out']:,} (thinking {t['thinking_tokens']:,})")

    # 3. the tiers sum to the run totals
    totals = {
        "total_tokens_in": "tokens_in",
        "total_cache_read_tokens": "cache_read_tokens",
        "total_cache_creation_tokens": "cache_creation_tokens",
        "total_prompt_tokens": "prompt_tokens",
        "total_tokens_out": "tokens_out",
        "total_thinking_tokens": "thinking_tokens",
    }
    for total_key, field in totals.items():
        _eq(f"tier sum vs {total_key}", sum(int(t[field]) for t in tiers.values()),
            int(usage[total_key]))
    _eq("total_prompt_tokens", int(usage["total_prompt_tokens"]),
        int(usage["total_tokens_in"]) + int(usage["total_cache_read_tokens"])
        + int(usage["total_cache_creation_tokens"]))
    _eq("total_tokens", int(usage["total_tokens"]),
        int(usage["total_prompt_tokens"]) + int(usage["total_tokens_out"]))
    notes.append(f"TOTAL: {llm_calls} LLM calls, prompt={usage['total_prompt_tokens']:,} "
                 f"(new {usage['total_tokens_in']:,} / read "
                 f"{usage['total_cache_read_tokens']:,} / write "
                 f"{usage['total_cache_creation_tokens']:,}), out={usage['total_tokens_out']:,}, "
                 f"tokens_total={usage['total_tokens']:,}")

    # 6. cost
    cost = usage.get("cost") or {}
    if cost.get("priced"):
        for tier, c in cost["by_tier"].items():
            _close(f"cost.by_tier[{tier}].total",
                   c["input"] + c["cache_write"] + c["cache_read"] + c["output"], c["total"])
            _close(f"cost.by_tier[{tier}].cache_savings", c["cache_savings"],
                   c["uncached_equivalent"] - c["total"])
        _close("cost.total_usd", sum(c["total"] for c in cost["by_tier"].values()),
               cost["total_usd"])
        _close("cost.cache_savings_usd", cost["cache_savings_usd"],
               cost["uncached_equivalent_usd"] - cost["total_usd"])
        _close("estimated_cost_usd", report.get("estimated_cost_usd"), cost["total_usd"])
        notes.append(f" COST: {cost['total_usd']:.4f} {cost['currency']} "
                     f"(uncached would be {cost['uncached_equivalent_usd']:.4f}, "
                     f"caching saved {cost['cache_savings_usd']:.4f}) "
                     f"[rates: {cost['rates_source'] or 'unnamed'}]")
        if cost.get("unpriced_tiers"):
            notes.append(f"        unpriced tiers EXCLUDED: {', '.join(cost['unpriced_tiers'])}")
    else:
        if report.get("estimated_cost_usd") is not None:
            raise Failure("estimated_cost_usd is set while cost.priced is false")
        if any(k.endswith("_usd") and cost.get(k) is not None for k in cost):
            raise Failure("cost carries dollar figures while priced is false")
        notes.append(" COST: not priced (no rates configured) — no dollar figures emitted")
    return notes


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    report_path = Path(sys.argv[1])
    trace_path = (Path(sys.argv[2]) if len(sys.argv) > 2
                  else report_path.with_name(report_path.name.replace("_otl.json",
                                                                      "_otl_trace.jsonl")))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    trace_file = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
    print(f"report: {report_path}")
    print(f"trace : {trace_path}\n")
    try:
        for note in reconcile(report, trace_file):
            print(f"  {note}")
    except Failure as exc:
        print(f"\nFAILED — {exc}", file=sys.stderr)
        return 1
    print("\nOK — all three views agree, and every derived field checks out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
