"""Per-tier token usage — provider-reported where possible, tiktoken-estimated as a last resort.

BOTH providers report their own counts, so an estimate is the exception rather than the rule:

  * `--provider ica` goes to the ICA gateway (LiteLLM, which speaks both dialects) over the
    Anthropic-native `/v1/messages` route, and gets an Anthropic `usage` block — see
    docs/GATEWAY_FINDINGS.md.
  * `--provider azure` goes to Azure OpenAI and gets an OpenAI `usage` block, reassembled out of the
    LangChain reply by `llm._langchain_usage`.

The two do not always report the same things. A Claude tier reports cache reads AND cache writes plus
a thinking subset of its output. A GPT tier always reports cache reads when prompt caching hits and
may report a cache-write counter when Azure or its gateway exposes a separately billable write class;
otherwise `cache_creation_tokens` remains 0. GPT reasoning remains a subset of billed output.

tiktoken is the fallback estimator only: offline, or when a reply reports nothing at all. It ships
true encodings for OpenAI models only, so for non-OpenAI families (Claude, Gemini, Llama, Granite, …)
it picks the closest public encoding — measurably wrong for Claude, which it undercounts by ~15–20%
on prose and more on code. Treat an estimated figure as a budgeting aid, never as billing.

Three things live here:
  * MODEL_ENCODINGS — the model-id → tiktoken-encoding mapping (with prefix fallbacks), which also
    feeds the `encoding` field every artifact's token report is validated against.
  * count_tokens / count_messages — the fallback estimator itself.
  * USAGE — a process-wide TokenUsage tracker that llm.record_reply_usage feeds, and run.py reads at
    the end of each exception to embed the totals at the bottom of the JSON. Each tier carries how
    its numbers were obtained (`source`), plus the cost of the run if a price map is configured.

Input is reported as THREE DISJOINT classes, because they bill at different rates: `tokens_in` is NEW
(uncached) input only, `cache_read_tokens` is served from the prompt cache (about 0.1x on Claude,
about 0.1x on GPT too), and `cache_creation_tokens` is written to it when the provider reports a
separately billable creation class. They add up to `prompt_tokens`, the full prompt size — which is what occupies the context
window, and what `tokens_in` alone used to mean. So a warm run and a cold one are told apart by where
the input sits, not by the total.

Attribution (who spent it): `record()` is the ONE funnel every LLM call in either pipeline passes
through, so it is also where a call is attributed to a caller. A caller opens `USAGE.attribute(...)`
(a contextvar scope naming e.g. the graph node and ReAct step) and every call made inside it is
additionally bucketed under that node (`by_node`, nested inside its tier in `report()`) and appended
to a per-call ledger. Because the tier counters, the node buckets and the ledger are all updated by
that single statement, the three views can never disagree: per node, the ledger sums to `by_node`,
and a tier's `by_node` entries sum to the tier itself. Calls made with no scope (the extractor's own,
which has no node concept) bucket under `_unattributed`, so that identity holds unconditionally.
The orchestration-flow layer consumes the ledger in `otl/token_trace.py` to stamp per-step token
blocks onto its trace records.

Offline safety: tiktoken lazily downloads its vocab (.tiktoken) files on first use. Behind a
proxy that blocks that download the encoding can't load, so count_tokens falls back to a
~4-chars-per-token heuristic rather than crashing the whole extraction just to count tokens.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from .pricing import estimate_cost
from .token_accounting import cache_counters, output_counters

try:
    import tiktoken
except ModuleNotFoundError:  # pragma: no cover - tiktoken is a hard dep, but degrade anyway
    tiktoken = None

# --- Model → tiktoken encoding mapping -------------------------------------------------------
#
# o200k_base  — modern OpenAI BPE (gpt-4o, gpt-5.x). Also used as the estimator for Gemini.
# cl100k_base — prior-gen OpenAI BPE (gpt-4, gpt-3.5). Used as the *approximation* for Claude and
#               the other non-OpenAI families, which have proprietary tokenizers tiktoken can't
#               reproduce exactly. cl100k_base is the conventional public proxy for Claude.
#
# Exact ids discovered on the ICA gateway (GET /models) are listed explicitly; the prefix rules
# in encoding_name_for() cover future/renamed variants.
MODEL_ENCODINGS: dict[str, str] = {
    # OpenAI GPT
    "gpt-4o": "o200k_base",
    "gpt-5.1-chat-gus": "o200k_base",
    "gpt-5.4-gus": "o200k_base",
    # Anthropic Claude (approximation — Claude uses a proprietary tokenizer)
    "claude-haiku-4-5": "cl100k_base",
    "claude-sonnet-4-5": "cl100k_base",
    "claude-sonnet-4-6": "cl100k_base",
    "claude-sonnet-5": "cl100k_base",
    "claude-opus-5": "cl100k_base",
    "claude-opus-4-6": "cl100k_base",
    "claude-opus-4-7": "cl100k_base",
    "claude-opus-4-8": "cl100k_base",
    # Google Gemini (approximation)
    "gemini-3.1-pro-preview": "o200k_base",
    "gemini-3.5-flash": "o200k_base",
    # Other open families (approximation)
    "gemma-4-26b-a4b-it": "cl100k_base",
    "ibm/granite-4-h-small": "cl100k_base",
    "meta-llama/llama-4-maverick-17b-128e-instruct-fp8": "cl100k_base",
}

_DEFAULT_ENCODING = "cl100k_base"


def encoding_name_for(model: str) -> str:
    """tiktoken encoding to estimate `model` with. Exact map first, then family prefixes."""
    if model in MODEL_ENCODINGS:
        return MODEL_ENCODINGS[model]
    m = model.lower()
    # strip any provider namespace, e.g. "meta-llama/llama-4..." or "ibm/granite..."
    tail = m.rsplit("/", 1)[-1]
    if tail.startswith(("gpt-4o", "gpt-5", "gpt-4.1", "o1", "o3", "o4", "chatgpt-4o", "gemini")):
        return "o200k_base"
    if tail.startswith(("gpt-4", "gpt-3.5", "gpt-35")):
        return "cl100k_base"
    if tail.startswith(("claude", "gemma", "granite", "llama", "mistral", "mixtral")):
        return "cl100k_base"
    return _DEFAULT_ENCODING


# Cache encoding objects (and remember which failed to load, so we don't retry the blocked
# download on every single call).
_ENC_CACHE: dict[str, object] = {}
_ENC_FAILED: set[str] = set()
_ENC_LOCK = threading.Lock()


def _get_encoding(name: str):
    if tiktoken is None or name in _ENC_FAILED:
        return None
    enc = _ENC_CACHE.get(name)
    if enc is not None:
        return enc
    with _ENC_LOCK:
        if name in _ENC_CACHE:
            return _ENC_CACHE[name]
        if name in _ENC_FAILED:
            return None
        try:
            enc = tiktoken.get_encoding(name)
            _ENC_CACHE[name] = enc
            return enc
        except Exception:  # download blocked / unknown encoding → fall back to heuristic
            _ENC_FAILED.add(name)
            return None


def count_tokens(text: str, model: str) -> int:
    """Estimate the token count of `text` for `model` via tiktoken (heuristic if unavailable)."""
    if not text:
        return 0
    enc = _get_encoding(encoding_name_for(model))
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    # Offline fallback: ~4 characters per token is the widely used rough estimate.
    return max(1, (len(text) + 3) // 4)


# OpenAI's chat format adds a few tokens of framing per message plus a fixed reply priming; we
# fold in a small, documented overhead so the estimate tracks real usage a little more closely.
_PER_MESSAGE_OVERHEAD = 4
_REPLY_PRIMING = 3


def count_messages(messages, model: str) -> int:
    """Estimate input tokens for a list of chat messages (content + per-message framing)."""
    total = _REPLY_PRIMING
    for msg in messages:
        content = getattr(msg, "content", msg if isinstance(msg, str) else "")
        if not isinstance(content, str):
            content = str(content)
        total += _PER_MESSAGE_OVERHEAD + count_tokens(content, model)
    return total


# --- Usage tracking --------------------------------------------------------------------------

# Bucket for calls made outside any `USAGE.attribute(...)` scope — i.e. every extractor-side call
# (the extractor has no node concept). Keeps `sum(by_node) == the tier's own counters` true always.
UNATTRIBUTED = "_unattributed"

# The current attribution scope: {"node": str, "step": int | None, "kind": str}. A contextvar (not a
# thread-local) so it survives across await points and is inherited by threads spawned from a
# scope — and so nesting restores the outer scope exactly on exit.
_ATTR: ContextVar[dict] = ContextVar("token_attribution", default={})


def source_label(sources: set) -> str:
    """Collapse the set of observed sources into one label ("mixed" when a bucket saw both)."""
    if not sources:
        return "local-estimate"
    if len(sources) == 1:
        return next(iter(sources))
    return "mixed"


@dataclass
class _NodeUsage:
    """One caller's slice of a tier (an OTL graph node). Same fields as the tier, plus `visits`."""
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    calls: int = 0
    visits: int = 0                 # times the node ran (bumped by note_visit, not by record)
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    thinking_tokens: int = 0
    sources: set = field(default_factory=set)

    @property
    def source(self) -> str:
        return source_label(self.sources)

    @property
    def prompt_tokens(self) -> int:
        return self.tokens_in + self.cache_read_tokens + self.cache_creation_tokens


@dataclass
class _TierUsage:
    model: str = ""
    # NEW (uncached) input only. The two cache buckets below are the other two input classes;
    # all three are disjoint and add up to `prompt_tokens`, the full prompt size.
    tokens_in: int = 0
    tokens_out: int = 0
    calls: int = 0
    # Reported by whoever answered. Claude reports both; GPT reports reads only, because OpenAI
    # caching is automatic and a write is not billed as its own class. Both stay 0 on an estimate,
    # where tokens_in is therefore the whole prompt on its own.
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    # The part of tokens_out spent before the answer: extended thinking on gen-5 Claude, reasoning on
    # gpt-5. A SUBSET of tokens_out, never an addition to it.
    thinking_tokens: int = 0
    # Every source seen for this tier — "gateway" and/or "local-estimate". Kept as a set so a tier
    # that fell back mid-run reports "mixed" instead of silently claiming reported accuracy.
    sources: set = field(default_factory=set)
    # Every model id seen for this tier. Normally one. Two means the Azure→ICA fallback latched
    # part-way through, so this bucket's tokens were billed at two different price lists — which
    # `pricing.estimate_cost` cannot express, and therefore flags rather than averaging away.
    models: set = field(default_factory=set)
    # Per-caller breakdown of the very same numbers, keyed by node name (see UNATTRIBUTED).
    nodes: "OrderedDict[str, _NodeUsage]" = field(default_factory=OrderedDict)

    @property
    def source(self) -> str:
        return source_label(self.sources)

    @property
    def prompt_tokens(self) -> int:
        return self.tokens_in + self.cache_read_tokens + self.cache_creation_tokens


@dataclass
class CallRecord:
    """One LLM call, as attributed at the moment it was recorded.

    `seq` is monotonic within a run (1-based), so a consumer can slice "calls since <mark>". A parse
    retry inside `llm.structured_call` produces its own record — retried attempts ARE billed, so they
    are ledgered like any other call.

    `tokens_in` is NEW input only, matching the tier and node counters. Use `prompt_tokens` for the
    whole prompt.
    """
    seq: int
    tier: str
    model: str
    node: str
    step: int | None
    kind: str
    tokens_in: int
    tokens_out: int
    thinking_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    source: str

    @property
    def prompt_tokens(self) -> int:
        return self.tokens_in + self.cache_read_tokens + self.cache_creation_tokens


class TokenUsage:
    """Process-wide accumulator of tokens in/out, grouped by model tier (and, within a tier, by the
    caller that spent them — see `attribute`).

    Thread-safe: interpret_attachment could be parallelized later. Reset per exception so each
    JSON artifact reports only its own run.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tiers: "OrderedDict[str, _TierUsage]" = OrderedDict()
        self._calls: list[CallRecord] = []
        self._seq = 0
        self._generation = 0

    def reset(self) -> None:
        with self._lock:
            self._tiers.clear()
            self._calls.clear()
            self._seq = 0
            self._generation += 1

    def generation(self) -> int:
        """Monotonic reset generation used by cycle-scoped runtime hooks."""
        with self._lock:
            return self._generation

    # --- attribution ------------------------------------------------------------------------
    @contextmanager
    def attribute(self, **fields):
        """Attribute every call made inside this block to `fields` (node / step / kind).

        Merges OVER the current scope — an explicitly passed field wins, anything not passed (or
        passed as None) is inherited. So the ReAct loop can add just `step=` inside the graph node's
        `node=` scope and both land on the call. Re-entrant and exception-safe.
        """
        merged = {**_ATTR.get(), **{k: v for k, v in fields.items() if v is not None}}
        token = _ATTR.set(merged)
        try:
            yield merged
        finally:
            _ATTR.reset(token)

    def note_visit(self, tier: str, node: str) -> None:
        """Register that `node` ran, so it appears in `by_node` even if it made no LLM call."""
        with self._lock:
            self._node_entry(tier, node).visits += 1

    def mark(self) -> int:
        """Current ledger cursor — pass to `calls_since` to read the calls made after this point."""
        with self._lock:
            return self._seq

    def calls_since(self, mark: int) -> list[CallRecord]:
        """The calls recorded after `mark`, in order. A snapshot copy — safe to hold and filter."""
        with self._lock:
            return [c for c in self._calls if c.seq > mark]

    # --- recording --------------------------------------------------------------------------
    def _node_entry(self, tier: str, node: str) -> _NodeUsage:
        """The (tier, node) bucket, created on first use. Caller must hold the lock."""
        entry = self._tiers.get(tier)
        if entry is None:
            entry = _TierUsage()
            self._tiers[tier] = entry
        node_entry = entry.nodes.get(node)
        if node_entry is None:
            node_entry = _NodeUsage()
            entry.nodes[node] = node_entry
        return node_entry

    def record(self, tier: str, model: str, tokens_in: int, tokens_out: int,
               source: str = "local-estimate", usage: dict | None = None) -> None:
        """Add one call's tokens to a tier (and to the current attribution scope's node + ledger).

        `tokens_in` is NEW (uncached) input only. The cached part of the prompt is counted in the two
        cache buckets instead, because the three classes bill at three different rates. The caller
        (`llm.record_reply_usage`) does that split with `token_accounting.input_buckets`; the buckets
        themselves are re-read from `usage` here, through the same parser, so the nested
        `cache_creation` TTL split and the OpenAI-shaped `prompt_tokens_details` fallback are handled
        in one place. Nothing is lost: the three add back up to the full prompt, reported as
        `prompt_tokens`.

        `tokens_out` is billed output, extended thinking included. The thinking part is read out of
        `usage` and reported alongside as a labelled subset, never as an addition.

        `source` / `usage` default so any un-migrated caller keeps working, recorded as an estimate.
        A caller with no `usage` block therefore reports all of its input as new, which is right: a
        provider that reports no cache counters is not caching.
        """
        read, creation = cache_counters(usage) if usage else (0, 0)
        thinking = output_counters(usage)[1] if usage else 0
        source = source or "local-estimate"
        attr = _ATTR.get()
        node = attr.get("node") or UNATTRIBUTED
        step, kind = attr.get("step"), attr.get("kind") or ""
        with self._lock:
            entry = self._tiers.get(tier)
            if entry is None:
                entry = _TierUsage(model=model)
                self._tiers[tier] = entry
            entry.model = model
            if model:
                entry.models.add(model)
            entry.tokens_in += tokens_in
            entry.tokens_out += tokens_out
            entry.thinking_tokens += thinking
            entry.cache_read_tokens += read
            entry.cache_creation_tokens += creation
            entry.calls += 1
            entry.sources.add(source)

            # Same numbers, second view. Updated under the same lock acquisition as the tier's, so
            # the per-node roll-up can never drift from the tier total it belongs to.
            node_entry = self._node_entry(tier, node)
            node_entry.model = model
            node_entry.tokens_in += tokens_in
            node_entry.tokens_out += tokens_out
            node_entry.thinking_tokens += thinking
            node_entry.cache_read_tokens += read
            node_entry.cache_creation_tokens += creation
            node_entry.calls += 1
            node_entry.sources.add(source)

            self._seq += 1
            self._calls.append(CallRecord(
                seq=self._seq, tier=tier, model=model, node=node, step=step, kind=kind,
                tokens_in=tokens_in, tokens_out=tokens_out, thinking_tokens=thinking,
                cache_read_tokens=read, cache_creation_tokens=creation, source=source))

    def report(self) -> dict:
        """A JSON-serializable summary: per-tier detail (with its per-node split) plus overall totals."""
        with self._lock:
            by_tier = {
                tier: {
                    "model": e.model,
                    # Normally the same single id as `model`. A second entry means this tier ran two
                    # models (the Azure→ICA latch), so its tokens span two price lists — see
                    # `cost.mixed_model_tiers`, which says so rather than quietly picking one.
                    "models_seen": sorted(e.models),
                    # Which tiktoken encoding WOULD estimate this model. Still emitted for every
                    # tier (including provider-reported ones) because it is a required field on
                    # schemas.TierTokenUsage — dropping it fails validation on every run.
                    "encoding": encoding_name_for(e.model),
                    "source": e.source,
                    "calls": e.calls,
                    # New input only. The two cache buckets below hold the rest of the prompt, and
                    # all three add up to prompt_tokens.
                    "tokens_in": e.tokens_in,
                    "cache_read_tokens": e.cache_read_tokens,
                    "cache_creation_tokens": e.cache_creation_tokens,
                    "prompt_tokens": e.prompt_tokens,
                    "tokens_out": e.tokens_out,
                    # A subset of tokens_out, not an addition to it.
                    "thinking_tokens": e.thinking_tokens,
                    "tokens_total": e.prompt_tokens + e.tokens_out,
                    # Who spent it. Every field above except `encoding` is broken down here, so each
                    # tier's by_node entries sum back to the tier. Empty of interest for the
                    # extractor (one `_unattributed` bucket); populated per graph node under OTL.
                    "by_node": {
                        node: {
                            "model": n.model,
                            "source": n.source,
                            "visits": n.visits,
                            "calls": n.calls,
                            "tokens_in": n.tokens_in,
                            "cache_read_tokens": n.cache_read_tokens,
                            "cache_creation_tokens": n.cache_creation_tokens,
                            "prompt_tokens": n.prompt_tokens,
                            "tokens_out": n.tokens_out,
                            "thinking_tokens": n.thinking_tokens,
                            "tokens_total": n.prompt_tokens + n.tokens_out,
                        }
                        for node, n in e.nodes.items()
                    },
                }
                for tier, e in self._tiers.items()
            }
            total_in = sum(e.tokens_in for e in self._tiers.values())
            total_out = sum(e.tokens_out for e in self._tiers.values())
            total_read = sum(e.cache_read_tokens for e in self._tiers.values())
            total_creation = sum(e.cache_creation_tokens for e in self._tiers.values())
            total_prompt = total_in + total_read + total_creation
            total_thinking = sum(e.thinking_tokens for e in self._tiers.values())
            # Only tiers that actually called something get a vote: a tier registered by note_visit
            # alone has seen no source and would otherwise drag an all-gateway run to "mixed".
            sources = {e.source for e in self._tiers.values() if e.calls}
        # Shared by all three notes: the field conventions a reader needs to add the columns up.
        convention = (
            "tokens_in is NEW (uncached) input only. cache_read_tokens and cache_creation_tokens "
            "hold the rest of the prompt, and the three add up to prompt_tokens (the full prompt). "
            "thinking_tokens is a subset of tokens_out. tokens_total is prompt_tokens + tokens_out. "
            "Which classes a provider reports differs: Claude reports cache reads and writes plus a "
            "thinking subset, GPT reports cache reads plus a reasoning subset and has no cache-write "
            "class at all, so a 0 there means the class does not exist rather than nothing cached.")
        estimate_caveat = (
            "Estimated input and output are counted with the closest public tiktoken encoding "
            "rather than the model's native tokenizer, which undercounts Claude by roughly 15-20% "
            "on prose and more on code. Output is undercounted further for a thinking model: the "
            "estimate only sees the reply text, so extended-thinking tokens are missed entirely.")
        if sources == {"gateway"}:
            estimator, note = "gateway", (
                "Provider-reported usage — the `usage` block on each reply (Anthropic /v1/messages "
                "on ica, Azure OpenAI on azure). " + convention)
        elif "gateway" in sources or "mixed" in sources:
            estimator, note = "mixed", (
                "Some tiers are provider-reported and some are tiktoken estimates (see each tier's "
                "`source`). " + convention + " " + estimate_caveat)
        else:
            estimator, note = "tiktoken", ("Estimated, not reported. " + estimate_caveat + " "
                                           + convention)
        return {
            "estimator": estimator,
            "note": note,
            "by_tier": by_tier,
            "total_tokens_in": total_in,
            "total_cache_read_tokens": total_read,
            "total_cache_creation_tokens": total_creation,
            "total_prompt_tokens": total_prompt,
            "total_tokens_out": total_out,
            "total_thinking_tokens": total_thinking,
            "total_tokens": total_prompt + total_out,
            # Dollars, from the per-tier aggregates above and a configured price map. Reports
            # `priced: false` rather than a guess when no rates are available.
            "cost": estimate_cost(by_tier),
        }


# Singleton the whole pipeline shares.
USAGE = TokenUsage()


def build_model_map(model_ids) -> "OrderedDict[str, str]":
    """Map discovered gateway model ids → the tiktoken encoding each would FALL BACK to.

    Only relevant when a provider reports no usage of its own (Azure, or offline); the native ICA
    path records the gateway's counts instead.
    """
    return OrderedDict((mid, encoding_name_for(mid)) for mid in sorted(model_ids))
