"""Per-node / per-step token attribution for the trace (the OTL half of token accounting).

`shared.tokens.USAGE` is the single funnel every LLM call passes through; it buckets each call by
tier AND by the node in the active `USAGE.attribute(...)` scope, and ledgers it as a `CallRecord`.
This module is the OTL-side consumer of that ledger: it opens the scopes and turns the ledger slices
into the compact `tokens` blocks that ride on the trace records.

  * `node_visit(node)`   — entered once per graph node execution (graph.py wraps every node with it).
                           Registers the visit, opens the `node=` attribution scope, and anchors a
                           ledger cursor for the visit.
  * `step_scope(step)`   — the ReAct loop adds just `step=` around each turn's LLM call.
  * `step_tokens(...)` / `node_tokens(...)` — the block for one trace record.

Disjointness is enforced by CONSUMPTION, not by arithmetic: a call may be claimed by exactly one
trace record (the first one whose predicate matches it), and the visit remembers which `seq`s are
already spoken for. So a ReAct turn is counted by its `react_step:*` record and NOT again by the
node's own event, while a `structured_call` parse retry — which is billed, and therefore ledgered —
is still counted somewhere. The invariant a reader can rely on:

    sum of every trace record's `tokens` block for node N  ==  by_node[N] in the report footer

A block for a record that covers no call is emitted with `calls: 0` and zeros rather than omitted, so
that sum is uniform to write. Outside a `node_visit` (a node function called directly in a unit test)
every block is zeroed — attribution is a reporting concern and must never break the caller.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

from shared.tokens import USAGE, CallRecord, source_label

from otl.llm_adapter import tier_for

# The active node visit: {"node": str, "mark": int, "consumed": set[int]}. None outside a graph node.
_VISIT: ContextVar[dict | None] = ContextVar("otl_token_visit", default=None)

# Numeric fields a `tokens` block carries — the ones that must add up. `prompt_tokens` and
# `tokens_total` are left out because they are sums of these, and are derived in `_block` instead.
_SUM_FIELDS = ("tokens_in", "cache_read_tokens", "cache_creation_tokens", "tokens_out",
               "thinking_tokens")


@contextmanager
def node_visit(node: str):
    """Scope one execution of `node`: count the visit, attribute its calls, anchor the ledger."""
    tier = tier_for(node)
    USAGE.note_visit(tier, node)
    token = _VISIT.set({"node": node, "mark": USAGE.mark(), "consumed": set()})
    try:
        with USAGE.attribute(node=node):
            yield
    finally:
        _VISIT.reset(token)


@contextmanager
def step_scope(step: int):
    """Attribute the LLM call inside this block to ReAct step `step` of the current node."""
    with USAGE.attribute(step=step):
        yield


def step_tokens(node: str, step: int) -> dict:
    """The token block for one ReAct step — the calls made inside its `step_scope`."""
    return _consume(node, lambda c: c.node == node and c.step == step)


def node_tokens(node: str) -> dict:
    """The token block for a node's own event: the calls it made DIRECTLY this visit.

    "Directly" = outside any ReAct step (`step is None`) and not yet claimed by an earlier record of
    this visit — i.e. its `structured`/`text` calls plus any parse retry. A pure-ReAct specialist
    reports `calls: 0` here because its spend is itemized on its `react_step:*` records.
    """
    return _consume(node, lambda c: c.node == node and c.step is None)


def _consume(node: str, predicate) -> dict:
    """Claim this visit's unconsumed ledger entries matching `predicate` and summarize them."""
    visit = _VISIT.get()
    if visit is None:                      # called outside the graph (unit test) — nothing to claim
        return _block([], tier=tier_for(node))
    consumed = visit["consumed"]
    picked = [c for c in USAGE.calls_since(visit["mark"])
              if c.seq not in consumed and predicate(c)]
    consumed.update(c.seq for c in picked)
    return _block(picked, tier=tier_for(node))


def _block(calls: list[CallRecord], *, tier: str) -> dict:
    """A JSON-ready token block. `tier` is named so it reconciles by eye against by_tier/by_node.

    Field order and meaning match `by_node` exactly, so the two can be compared line by line:
    `tokens_in` is new input only, the two cache counts hold the rest of the prompt, and the three
    add up to `prompt_tokens`. `thinking_tokens` is a subset of `tokens_out`.
    """
    totals = {f: sum(getattr(c, f) for c in calls) for f in _SUM_FIELDS}
    prompt = totals["tokens_in"] + totals["cache_read_tokens"] + totals["cache_creation_tokens"]
    block = {
        "tier": tier,
        "model": calls[-1].model if calls else "",
        "source": source_label({c.source for c in calls}) if calls else "",
        "calls": len(calls),
        "tokens_in": totals["tokens_in"],
        "cache_read_tokens": totals["cache_read_tokens"],
        "cache_creation_tokens": totals["cache_creation_tokens"],
        "prompt_tokens": prompt,
        "tokens_out": totals["tokens_out"],
        "thinking_tokens": totals["thinking_tokens"],
        "tokens_total": prompt + totals["tokens_out"],
    }
    if len(calls) > 1:
        # More than one call behind a single record — today only a structured_call parse retry, whose
        # first attempt was billed too. Itemized so the wasted attempt stays visible.
        block["detail"] = [
            {"seq": c.seq, "kind": c.kind, "source": c.source,
             **{f: getattr(c, f) for f in _SUM_FIELDS}}
            for c in calls
        ]
    return block


__all__ = ["node_visit", "step_scope", "step_tokens", "node_tokens"]
