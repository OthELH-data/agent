"""LLM access with model tiering, over one of two switchable providers.

`LLM_PROVIDER` selects the backend (default `ica`):
  * `ica`   — the IBM ICA gateway via the NATIVE Anthropic SDK (`anthropic.Anthropic` pointed at
              ICA_BASE_URL, POST /messages); reads ICA_API_KEY / ICA_BASE_URL. The gateway is
              LiteLLM and speaks BOTH dialects — the Anthropic-native routes are mounted alongside
              the OpenAI-compat ones — and the native route is the one used here, for two measured
              reasons (docs/GATEWAY_FINDINGS.md): (1) prompt caching only works on it (the same
              `cache_control` prefix reports cache_creation=0/cached=0 on /chat/completions, i.e.
              full price every call, but writes then reads ~13K tokens on /messages), and (2) it
              reports a real Anthropic `usage` block, so token accounting stops being a tiktoken
              guess. Auth is the SDK default `x-api-key` header (same key value the OpenAI-compat
              path sent as `Authorization: Bearer`).
  * `azure` — an Azure AI Foundry project endpoint through the OpenAI-compatible `ChatOpenAI`
              client. One unified AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY covers all three
              tiers; the tiers differ only by AZURE_MODEL_{HIGH,MED,LIGHT} (default gpt-5.4-pro /
              gpt-5.4 / gpt-5.4-mini). The project endpoint is extended with `/openai/v1/`, and all
              Azure tiers use the Responses API.

`llm_config()` renders the whole resolved configuration (provider, Foundry v1 endpoint, model,
auth mode — never the credential itself) without making a network call, which is what
`run.py --llm-config` prints; use it to confirm a provider switch before a batch spends an hour
discovering the misconfiguration.

`get_llm` returns whichever client backs the active provider — `ChatOpenAI` on Azure Foundry, the
`_AnthropicTierClient` wrapper on ica — and both expose `.invoke(messages)` returning an `LLMReply`
(text + the gateway's raw usage dict), so `structured_call` / `text_call` / token accounting stay
provider-agnostic. Model ids per tier come from <PROVIDER>_MODEL_* (with provider-aware defaults).
The repo-root `.env` is loaded if python-dotenv is installed. `structured_call`
instructs the model to emit JSON, extracts the first balanced JSON object, and validates it against
a Pydantic schema (retrying once on failure).

Token accounting: `record_reply_usage` records the PROVIDER'S OWN counts whenever the reply carried a
usage block, which both providers do — Anthropic names on ica, OpenAI names on azure (see
`_langchain_usage`) — and falls back to the tiktoken estimate only when a reply reports nothing.
`tokens_in` counts NEW input only; the cached part of the prompt is reported in its own buckets,
because the three input classes bill at different rates — see `record_reply_usage`. The two
providers do not report the same counters, and neither is normalised into the other: OpenAI has no
cache-WRITE class (its caching is automatic and writes are unbilled) and reports reasoning rather
than thinking tokens. On the native ica path each call is also pre-flighted against the tier's input
limit (`check_threshold`), and a reactive context-overflow is trimmed and retried instead of aborting
the case.

Prompt caching has two targets. `_system_blocks` marks a large system prompt, and `_to_anthropic`
marks a REPEATING conversation buffer — which in practice means the ReAct loop, the only caller that
re-sends a growing history. Both are gated on size, and the conversation one is also gated on the
buffer actually repeating: marking a one-off prompt costs extra with no later read to earn that back.
`MODEL_CACHE_SYSTEM_PREFIX=0` and `MODEL_CACHE_CONVERSATION=0` turn them off independently, and
`MODEL_CACHE_TTL` picks how long an entry lives — `1h` (the default) or `5m`.

Azure configuration is deliberately minimal: one project endpoint, one API key, and a model id for
each tier. When Azure fails and ICA credentials are configured, the run latches onto ICA and retries
the failed request there for the remainder of that process.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# langchain-openai backs the AZURE path only — the ica path talks to the gateway through the
# official `anthropic` SDK (imported lazily in _get_ica_anthropic, so an Azure-only install that
# never pip-installs it still works).
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from .hooks import HOOKS
from .token_accounting import (CACHE_WRITE_MULTIPLIER, MODEL_INPUT_LIMITS, DEFAULT_INPUT_LIMIT,
                               UsageBaseline, cache_ttl, capture_usage, check_threshold,
                               estimate_message_tokens, identity, input_buckets,
                               is_context_overflow_error, normalise_usage, output_counters)
from .token_accounting import logger as _accounting_logger
from .tokens import USAGE, count_messages, count_tokens

# Load the repo-root .env (one level up from this package) if python-dotenv is present. There is
# no package-local .env — excel_extractor and orchestration-flow both read this single file.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ModuleNotFoundError:  # pragma: no cover
    pass

# Which backend get_llm() builds. "ica" (default) = IBM ICA gateway; "azure" = native Azure OpenAI.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ica").strip().lower()

ICA_API_KEY = os.getenv("ICA_API_KEY", "")
ICA_BASE_URL = os.getenv("ICA_BASE_URL", "https://api.servicesessentials.ibm.com")

# The Anthropic SDK appends `/v1/messages` to its base URL. ICA_BASE_URL is now configured without
# that version segment, so it is passed through unchanged. ICA_ANTHROPIC_BASE_URL remains an optional
# override for deployments that expose the native Anthropic surface at a different base path.


def ica_anthropic_base_url() -> str:
    """The base_url handed to `anthropic.Anthropic` for the ica provider."""
    return ICA_BASE_URL.strip().rstrip("/")

# Azure AI Foundry project endpoint. The client uses the OpenAI-compatible
# `/openai/v1/` route, which intentionally has no dated `api-version` query parameter.
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://shell-goa-openai.openai.azure.com")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")

# Azure calls use the Responses API for every tier. The API version must support that route.

# Per-provider, per-tier model ids (also the request-body `model` and Azure deployment names). For each
# provider a tier resolves as <PROVIDER>_MODEL_<TIER> → provider-aware hardcoded default.
# ICA defaults are the gen-5 ids confirmed present on the gateway's /v1/models. NOTE the resolution
# order below: a `.env` that still pins ICA_MODEL_HIGH=claude-opus-4-8 overrides these defaults.
_TIER_DEFAULTS = {
    "azure": {"high": "gpt-5.6-sol", "med": "gpt-5.6-terra", "light": "gpt-5.6-luna"},
    "ica": {"high": "claude-opus-5", "med": "claude-sonnet-5", "light": "claude-haiku-4-5"},
}


def _tier_model(provider: str, tier: str) -> str:
    """Model id for a provider+tier: <PROVIDER>_MODEL_<TIER> → legacy MODEL_<TIER> → default."""
    up = tier.upper()
    prefix = "AZURE" if provider == "azure" else "ICA"
    return (os.getenv(f"{prefix}_MODEL_{up}") or _TIER_DEFAULTS[provider][tier])


_TIER_MODELS = {
    provider: {tier: _tier_model(provider, tier) for tier in ("high", "med", "light")}
    for provider in ("azure", "ica")
}

# Back-compat module-level names reflect the PRIMARY provider (the one LLM_PROVIDER selects at import).
_PRIMARY = "azure" if LLM_PROVIDER == "azure" else "ica"
_TIERS = _TIER_MODELS[_PRIMARY]
MODEL_HIGH, MODEL_MED, MODEL_LIGHT = _TIERS["high"], _TIERS["med"], _TIERS["light"]

# Azure→ICA fallback is automatic and process-local. A case subprocess that starts on Azure switches to
# ICA only after an Azure failure; subsequent calls use ICA so the case can finish consistently.
_fallback_latched = False


def _fallback_available() -> bool:
    return LLM_PROVIDER == "azure" and bool(ICA_API_KEY)

def active_provider() -> str:
    """The provider handling calls now — ICA after an Azure fallback latch."""
    return "ica" if (_fallback_latched and LLM_PROVIDER == "azure") else _PRIMARY


def set_provider(provider: str) -> str:
    """Switch the LLM backend for this process; returns the normalized provider.

    Backs the `--provider {azure,ica}` flag on every runner. `LLM_PROVIDER` is read once at import
    (see the top of this module), so by the time a `main()` parses argv this module is long loaded and
    setting os.environ alone would change nothing — the switch has to reassign the derived globals here.

    Flipping post-import is safe because nothing needs re-reading or invalidating:
      * `_TIER_MODELS` already holds the tier map for BOTH providers, and the ICA credentials are
        provider-independent — so no env re-read.
      * Nothing outside this module reads `MODEL_HIGH|MED|LIGHT` / `_TIERS` / `LLM_PROVIDER`, so no
        `from .llm import MODEL_X` binding can go stale.
      * `_build_llm`'s lru_cache is keyed on provider, and active_provider/model_for_tier/get_llm/
        llm_config all read these globals lazily.

    Also exports LLM_PROVIDER so the batch drivers' `run.py` subprocesses (which inherit env=os.environ)
    pick the switch up for free, exactly as the shell wrappers do by exporting it before Python starts.
    """
    global LLM_PROVIDER, _PRIMARY, _TIERS, MODEL_HIGH, MODEL_MED, MODEL_LIGHT, _fallback_latched
    p = (provider or "").strip().lower()
    if p not in ("azure", "ica"):
        raise ValueError(f"unknown LLM provider {provider!r} — expected 'azure' or 'ica'")
    LLM_PROVIDER = _PRIMARY = p
    _TIERS = _TIER_MODELS[p]
    MODEL_HIGH, MODEL_MED, MODEL_LIGHT = _TIERS["high"], _TIERS["med"], _TIERS["light"]
    _fallback_latched = False
    os.environ["LLM_PROVIDER"] = p
    return p

# Cap on output tokens per tier. Without an explicit cap the gateway applies a low default (~4096)
# and long structured replies (e.g. a synthesis JSON) get truncated mid-object → "unbalanced JSON
# object". Default to each model's real max output so a full reply is never cut off: Opus 4.8 (high)
# and Sonnet 4.6 (med) = 128K, Haiku 4.5 (light) = 64K. Override any tier via MODEL_MAX_TOKENS_<TIER>.
_MAX_TOKENS_BY_TIER = {
    "high": int(os.getenv("MODEL_MAX_TOKENS_HIGH", "128000")),
    "med": int(os.getenv("MODEL_MAX_TOKENS_MED", "128000")),
    "light": int(os.getenv("MODEL_MAX_TOKENS_LIGHT", "64000")),
}
_DEFAULT_MAX_TOKENS = 64000  # fallback for an unknown tier

# Some models on the ICA/Bedrock gateway reject the `temperature` param outright — the gen-5 Claude
# family (claude-opus-5, claude-sonnet-5, …) runs with extended thinking on and accepts only
# temperature=1 ("does not support temperature=0.0"), and Opus 4.8 rejects it too. For any such model
# we OMIT the param and let the gateway default apply (gen-5 ignores temperature=0 anyway, so nothing
# is lost). `_omits_temperature` decides by major-version so a `.env` model swap (MODEL_MED=claude-
# sonnet-5) doesn't silently reintroduce the bug; MODEL_NO_TEMPERATURE (comma-separated) is an extra
# manual escape hatch.
_NO_TEMPERATURE_EXPLICIT = {"claude-opus-4-8"} | {
    m.strip() for m in os.getenv("MODEL_NO_TEMPERATURE", "").split(",") if m.strip()
}
# Anchored on the major-version token, so claude-haiku-4-5 / claude-sonnet-4-6 (4.x, which DO honor
# temperature=0) are not matched, while claude-opus-5 / claude-sonnet-5 (major >= 5) are.
_MODEL_GEN_RE = re.compile(r"claude-(?:opus|sonnet|haiku)-(\d+)")
# OpenAI reasoning models (o-series o1/o3/o4…, and the gpt-5 family incl. gpt-5.4) accept only the
# default temperature and reject an explicit temperature=0. Anchored on a boundary so gpt-4o (which DOES
# honor temperature) is not matched.
_OPENAI_REASONING_RE = re.compile(r"(?:^|[-_/])(?:o[1-9]|gpt-5)", re.IGNORECASE)


def _omits_temperature(model: str) -> bool:
    """True if `temperature` must be omitted for this model (Claude gen-5+, OpenAI reasoning, or list)."""
    if model in _NO_TEMPERATURE_EXPLICIT:
        return True
    m = _MODEL_GEN_RE.search(model)
    if m and int(m.group(1)) >= 5:
        return True
    return bool(_OPENAI_REASONING_RE.search(model or ""))

# Per-request timeout (seconds). A large opus synthesis (128K max output, big context) can legitimately
# run past the old 180s; 180s then killed a call that would have succeeded. Raise the default and let
# it be tuned via MODEL_TIMEOUT.
_REQUEST_TIMEOUT = int(os.getenv("MODEL_TIMEOUT", "300"))
# Extra whole-call retries on transient transport failures (timeout/connection/rate-limit/5xx) beyond
# the openai client's own max_retries. These are raised straight out of invoke() — structured_call's
# parse-retry loop never sees them — so without this a single slow/throttled gateway response aborts
# (and SKIPs) an entire rebuttal cycle. Tune via MODEL_TRANSIENT_RETRIES.
_TRANSIENT_RETRIES = int(os.getenv("MODEL_TRANSIENT_RETRIES", "3"))

# --- native-path context accounting knobs (ICA only) -----------------------------------------
#
# INPUT limit per tier, for the proactive 80%-of-context check. GET /v1/model/info reports
# max_input_tokens=1000000 for the Claude models (GET /v1/models/{id} is proxied but returns null,
# so it can't be used); Probe E corroborated it by accepting a 576,198-token prompt with no
# truncation. Override with MODEL_MAX_INPUT_TOKENS_<TIER> → MODEL_MAX_INPUT_TOKENS → the per-model
# table → 200K for an unknown model.
def _max_input_tokens(tier: str, model: str) -> int:
    for name in (f"MODEL_MAX_INPUT_TOKENS_{tier.upper()}", "MODEL_MAX_INPUT_TOKENS"):
        raw = (os.getenv(name) or "").strip()
        if raw:
            try:
                return int(raw)
            except ValueError:
                print(f"    ignoring non-numeric {name}={raw!r}", file=sys.stderr)
    return MODEL_INPUT_LIMITS.get(model, DEFAULT_INPUT_LIMIT)


# Smallest system prompt worth marking with cache_control. Anthropic documents a 1024-token minimum,
# but cache WRITES were not observed landing until ~2048 on this gateway, so the default sits
# comfortably above that — a marked block below the real minimum is just a silently wasted breakpoint.
_CACHE_MIN_TOKENS = int(os.getenv("MODEL_CACHE_MIN_TOKENS", "2500"))
_CACHE_SYSTEM_PREFIX = os.getenv("MODEL_CACHE_SYSTEM_PREFIX", "1").strip().lower() not in (
    "0", "false", "no", "n", "f")
# The other cache target: the ReAct loop's growing message buffer, which is re-sent in full on every
# step and is identical up to its tail. That is where the tokens actually are — a system prompt of a
# few hundred tokens never reaches _CACHE_MIN_TOKENS, so before this the cache never fired at all.
_CACHE_CONVERSATION = os.getenv("MODEL_CACHE_CONVERSATION", "1").strip().lower() not in (
    "0", "false", "no", "n", "f")

# Whether the buffer being sent is a REPEATING one, i.e. worth a cache breakpoint. Set by
# `_invoke_checked` from `bool(conv_key)` and read by `_to_anthropic`.
#
# It has to be a side channel rather than an argument because `_invoke(llm, messages)` is deliberately
# provider-blind: it hands the message list to whichever client it was given, and only the Anthropic
# client knows about content blocks. A contextvar keeps that signature untouched, survives await
# points, and restores the outer value on exit. `tokens.py` uses one the same way for attribution.
#
# The gate matters both ways. Marking a prompt that never repeats moves its input tokens from 1x to
# about 1.25x with no later read to earn that back, so `structured_call` and `text_call` — which build
# stateless two-message prompts — must NOT be marked.
_CACHE_REPEATING: ContextVar[bool] = ContextVar("llm_cache_repeating", default=False)


def _cache_control() -> dict:
    """The `cache_control` value to stamp on a breakpoint, honouring MODEL_CACHE_TTL.

    Anthropic's default lifetime is five minutes and is what a bare `{"type": "ephemeral"}` asks
    for; a one-hour entry adds an explicit `ttl`. Both are generally available — no beta header is
    involved either way. The 5-minute branch deliberately stays a bare block rather than sending
    `ttl: "5m"`, so turning the TTL back down reproduces byte-for-byte the request shape this
    gateway was already verified against.

    The default is one hour (see `token_accounting.cache_ttl`): a write then costs about 2x the
    input rate instead of 1.25x, so it needs three reads rather than two to pay for itself, but the
    ReAct loop re-sends its buffer far more often than that and a single slow `run_python` step can
    open a gap wide enough to expire a 5-minute entry.
    """
    ttl = cache_ttl()
    return {"type": "ephemeral"} if ttl == "5m" else {"type": "ephemeral", "ttl": ttl}

# How many trailing turns a trim keeps (plus the system message and the first user turn, which carry
# the task). Only the ReAct loop's growing buffer is ever long enough for this to bite.
_TRIM_KEEP_RECENT = max(2, int(os.getenv("MODEL_TRIM_KEEP_RECENT", "6")))

# The token_accounting module logs one INFO record per threshold check (estimate/limit/threshold/
# ratio/source/semantics/reason/raw_usage). Nothing in this repo configures logging, so wire that
# logger to stderr once here or the line would never appear. TOKEN_ACCOUNTING_DEBUG=0 silences it.
_TOKEN_DEBUG = os.getenv("TOKEN_ACCOUNTING_DEBUG", "1").strip().lower() not in (
    "0", "false", "no", "n", "f")
if _TOKEN_DEBUG and not any(
        isinstance(h, logging.StreamHandler) for h in _accounting_logger.handlers):
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("    %(message)s"))
    _accounting_logger.addHandler(_handler)
    _accounting_logger.setLevel(logging.INFO)
    _accounting_logger.propagate = False  # ours alone; don't double-print via a root handler


def model_for_tier(tier: str) -> str:
    """The model id backing a tier under the active provider (falls back to mid for unknown tiers)."""
    models = _TIER_MODELS[active_provider()]
    return models.get(tier, models["med"])


@dataclass(frozen=True)
class LLMReply:
    """One model reply: the flattened text plus whatever usage the provider reported.

    `usage` is the provider's own usage mapping, in the provider's OWN FIELD NAMES. On the native ica
    path it is the Anthropic block unmodified — including the nested `cache_creation` TTL split and,
    on claude-opus-5, `output_tokens_details.thinking_tokens`. On the Azure path it is the OpenAI
    block (`prompt_tokens`, `prompt_tokens_details.cached_tokens`,
    `completion_tokens_details.reasoning_tokens`), reassembled by `_langchain_usage`.

    Keeping each provider's own names is deliberate, not laziness: `token_accounting` decides whether
    the input counter INCLUDES the cached part by looking at which names it sees. Anthropic's
    `input_tokens` excludes it, OpenAI's `prompt_tokens` includes it, and translating one into the
    other's names would silently pick the wrong rule and either double-count or lose the cache.

    A `0` means "reported as zero" and a missing key means "never reported"; the two are never
    collapsed. `{}` means the provider reported nothing at all, which is what makes
    `record_reply_usage` fall back to the tiktoken estimate.
    """
    text: str
    usage: dict = field(default_factory=dict)
    raw: object = None


def _message_role(msg: Any) -> str:
    """Map a LangChain message (or a plain dict/str) onto an Anthropic role."""
    if isinstance(msg, SystemMessage):
        return "system"
    if isinstance(msg, AIMessage):
        return "assistant"
    if isinstance(msg, dict):
        role = str(msg.get("role") or msg.get("type") or "user").lower()
        return {"ai": "assistant", "human": "user"}.get(role, role)
    kind = str(getattr(msg, "type", "") or "").lower()
    return {"ai": "assistant", "system": "system", "human": "user"}.get(kind, "user")


def _message_text(msg: Any) -> str:
    """The message's text content, flattening a block list the way AIMessage.text does."""
    content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(getattr(block, "text", "") or ""))
        return "".join(parts)
    return "" if content is None else str(content)


def _system_blocks(text: str) -> list[dict] | None:
    """Render the system prompt as Anthropic content blocks, cache-marking it when it is big enough.

    THE point of the native path: `structured_call`'s system message (prose + the full
    `model_json_schema()`) and each OTL node's system prompt are byte-identical across calls and
    cases, so one ephemeral breakpoint on this block turns every repeat into a cache READ. Only one
    breakpoint is set, and only above `_CACHE_MIN_TOKENS` — a marked block under the gateway's real
    minimum is a silently wasted breakpoint, not an error.
    """
    if not text:
        return None
    block: dict = {"type": "text", "text": text}
    if _CACHE_SYSTEM_PREFIX and estimate_message_tokens([], system=text) >= _CACHE_MIN_TOKENS:
        block["cache_control"] = _cache_control()
    return [block]


def _cache_breakpoints(turns: list[dict]) -> list[int]:
    """Which turn indices to put a cache breakpoint on — at most two, both on user turns.

    A breakpoint caches everything BEFORE and including it, so the useful place is as late in the
    buffer as possible. The last turn is always a user turn at call time, so marking it caches the
    whole prompt for the next step to read.

    A second, older breakpoint is added when the buffer is long enough to have one. A cache entry
    expires (after an hour by default, five minutes if MODEL_CACHE_TTL says so) and a write does not
    become readable instantly, so a single rolling breakpoint can leave a step with no hit at all.
    The older one is the fallback that still matches.
    Two here plus the one in `_system_blocks` uses three of Anthropic's four.
    """
    user_turns = [i for i, t in enumerate(turns) if t["role"] == "user"]
    if not user_turns:
        return []
    marks = [user_turns[-1]]
    if len(turns) >= 3 and len(user_turns) >= 2:
        marks.insert(0, user_turns[-2])
    return marks


def _to_anthropic(messages, cache_conversation: bool | None = None
                  ) -> tuple[list[dict] | None, list[dict]]:
    """Translate a LangChain message list into Anthropic (system, messages).

    The pipeline speaks SystemMessage/HumanMessage/AIMessage; `/v1/messages` wants `system` hoisted
    out and only user/assistant turns, alternating. Consecutive same-role turns are MERGED because
    both callers produce them: `structured_call` appends a HumanMessage nudge after a failed parse
    (two user turns in a row once the assistant reply is not in the list), and `react_loop` appends
    an OBSERVATION plus, on a malformed turn, a protocol reminder.

    When the buffer is a repeating one it also gets cache breakpoints, which is what turns the ReAct
    loop's re-sent history into cache reads. `cache_conversation` defaults to the `_CACHE_REPEATING`
    contextvar; pass it explicitly to test either behaviour. Marking happens LAST, after the merge and
    the role fixups, so a breakpoint always lands on a final turn rather than one that then gets
    merged away. A marked turn's content becomes a one-element block list, because `cache_control`
    lives on a content block and a plain string has nowhere to carry it.
    """
    system_parts: list[str] = []
    turns: list[dict] = []
    for msg in messages:
        role = _message_role(msg)
        text = _message_text(msg)
        if role == "system":
            system_parts.append(text)
            continue
        if role != "assistant":
            role = "user"
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] = f"{turns[-1]['content']}\n\n{text}"
        else:
            turns.append({"role": role, "content": text})
    if not turns:
        # The API rejects an empty messages array; a trimmed/odd prompt must not become a hard error.
        turns.append({"role": "user", "content": "(no user content)"})
    if turns[0]["role"] == "assistant":
        turns.insert(0, {"role": "user", "content": "(continue)"})
    if turns[-1]["role"] == "assistant":
        # Trailing whitespace on a final assistant turn is rejected outright.
        turns[-1]["content"] = turns[-1]["content"].rstrip()
    marked = _CACHE_REPEATING.get() if cache_conversation is None else cache_conversation
    # The size gate is the same one `_system_blocks` uses: below the gateway's real cache minimum a
    # breakpoint writes nothing and is silently wasted.
    if (_CACHE_CONVERSATION and marked
            and estimate_message_tokens(turns) >= _CACHE_MIN_TOKENS):
        control = _cache_control()
        for index in _cache_breakpoints(turns):
            turns[index]["content"] = [{"type": "text", "text": turns[index]["content"],
                                        "cache_control": control}]
    return _system_blocks("\n\n".join(p for p in system_parts if p)), turns


def _anthropic_reply(msg: Any) -> LLMReply:
    """Wrap an anthropic Message: concatenated text blocks + the raw usage mapping."""
    parts = []
    for block in getattr(msg, "content", None) or []:
        if getattr(block, "type", None) == "text" or (
                isinstance(block, dict) and block.get("type") == "text"):
            parts.append(block["text"] if isinstance(block, dict) else block.text)
    usage = getattr(msg, "usage", None)
    return LLMReply(text="".join(parts), usage=normalise_usage(usage) if usage else {}, raw=msg)


# The two OpenAI-dialect sub-counters we care about, and the `usage_metadata` names they arrive under.
# `cached_tokens` is the discounted part of the prompt; `reasoning_tokens` is the part of the
# completion the gpt-5 models spent thinking. Both are the GPT counterparts of what the Anthropic path
# reports as `cache_read_input_tokens` and `output_tokens_details.thinking_tokens`.
_LC_CACHE_READ_KEYS = ("cache_read", "cache_read_tokens", "cache_read_input_tokens", "cached",
                       "cached_tokens")
_LC_CACHE_WRITE_KEYS = ("cache_write", "cache_write_tokens", "cache_creation",
                        "cache_creation_tokens", "cache_creation_input_tokens")
_LC_REASONING_KEYS = ("reasoning", "reasoning_tokens", "thinking", "thinking_tokens")


def _detail(details: Any, names: tuple[str, ...]) -> int:
    """First present key of `names` in a LangChain token-details mapping, as an int."""
    if not isinstance(details, dict):
        return 0
    for name in names:
        if name in details:
            try:
                return max(0, int(details.get(name) or 0))
            except (TypeError, ValueError):
                return 0
    return 0


def _langchain_usage(reply: Any) -> dict:
    """The usage block from a LangChain reply, expressed in OPENAI field names.

    Azure's numbers used to be thrown away here on purpose. They are kept now, and the awkward part is
    that LangChain exposes them in two places that do not agree on shape:

      * `usage_metadata` — LangChain's own normalised summary. It is the one to prefer, because it
        reads the same whether the tier went out over Chat Completions or the Responses API (which
        names its counters `input_tokens` / `input_tokens_details.cached_tokens` instead). It carries
        cache reads and, when Azure/the gateway reports them, cache writes.
      * `response_metadata["token_usage"]` — the provider block verbatim. Used only as a fallback
        when it is recognisably either a Chat Completions (`prompt_tokens`) or Responses (`input_tokens`)
        usage block. `token_accounting` reads the detail fields for both dialects.

    Everything is emitted under OpenAI names because that is how `token_accounting.detect_semantics`
    learns these counters are INCLUSIVE — `prompt_tokens` already contains `cached_tokens`, so the
    cache buckets get carved back out of it rather than added to it. Emitting an Anthropic name here
    would flip that rule and roughly double the reported prompt.

    A cache WRITE counter remains optional: public OpenAI responses commonly report only cache reads,
    but Azure or a gateway can expose a separately billable write counter. When present, preserve it so
    the accounting and pricing layers can split it from new input without double counting the prompt.
    """
    meta = getattr(reply, "usage_metadata", None)
    if isinstance(meta, dict) and (meta.get("input_tokens") or meta.get("output_tokens")):
        return {
            "prompt_tokens": max(0, int(meta.get("input_tokens") or 0)),
            "completion_tokens": max(0, int(meta.get("output_tokens") or 0)),
            "prompt_tokens_details": {
                "cached_tokens": _detail(meta.get("input_token_details"), _LC_CACHE_READ_KEYS),
                "cache_write_tokens": _detail(meta.get("input_token_details"), _LC_CACHE_WRITE_KEYS),
            },
            "completion_tokens_details": {
                "reasoning_tokens": _detail(meta.get("output_token_details"), _LC_REASONING_KEYS)},
        }
    raw = (getattr(reply, "response_metadata", None) or {}).get("token_usage")
    if isinstance(raw, dict) and ("prompt_tokens" in raw or "input_tokens" in raw):
        return dict(raw)
    return {}


@dataclass(frozen=True)
class _AnthropicTierClient:
    """`.invoke(messages) -> LLMReply` over `anthropic.Anthropic`, so `_invoke` stays provider-blind.

    Mirrors the surface `_invoke` used against the LangChain clients (one `invoke(messages)` call)
    while keeping every knob the OpenAI-compat client honoured: `max_tokens` from
    `_MAX_TOKENS_BY_TIER`, `temperature` OMITTED entirely when `_omits_temperature(model)` (gen-5
    Claude rejects temperature=0 outright), and the SDK's own timeout/max_retries.
    """
    client: Any
    model: str
    max_tokens: int
    temperature: float | None
    base_url: str

    def invoke(self, messages) -> LLMReply:
        system, turns = _to_anthropic(messages)
        kwargs: dict = {"model": self.model, "max_tokens": self.max_tokens, "messages": turns}
        if system is not None:
            kwargs["system"] = system
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        return _anthropic_reply(self.client.messages.create(**kwargs))


def _get_ica_anthropic(model: str, temperature: float, mt: int) -> _AnthropicTierClient:
    """The native Anthropic client for a tier — `/v1/messages` on the ICA gateway, no OpenAI shim."""
    if not ICA_API_KEY:
        raise RuntimeError("ICA_API_KEY is not set — fill the repo-root .env")
    try:
        import anthropic
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "LLM_PROVIDER=ica needs the `anthropic` package (the native /v1/messages path) — "
            "pip install -r excel_extractor/requirements.txt") from exc
    base_url = ica_anthropic_base_url()
    client = anthropic.Anthropic(
        base_url=base_url,
        api_key=ICA_API_KEY,          # sent as the SDK-default `x-api-key` header, not Bearer
        # An explicit timeout also suppresses the SDK's "streaming is strongly recommended for
        # operations that may take longer than 10 minutes" ValueError on a large max_tokens.
        timeout=_REQUEST_TIMEOUT,
        max_retries=2,
    )
    return _AnthropicTierClient(
        client=client, model=model, max_tokens=mt, base_url=base_url,
        temperature=None if _omits_temperature(model) else temperature)


def azure_foundry_v1_base_url() -> str:
    """The OpenAI-compatible v1 base route for the configured Foundry project endpoint."""
    endpoint = AZURE_OPENAI_ENDPOINT.rstrip("/")
    return endpoint + "/" if endpoint.endswith("/openai/v1") else endpoint + "/openai/v1/"


def _get_azure_llm(tier: str, model: str, temperature: float, mt: int) -> ChatOpenAI:
    if not AZURE_OPENAI_ENDPOINT:
        raise RuntimeError("No Azure endpoint configured — set AZURE_OPENAI_ENDPOINT "
                           "in the repo-root .env")
    # Foundry project endpoints require the `/openai/v1/` route rather than a legacy
    # `/openai/deployments/<deployment>?api-version=...` Azure route. Every tier uses Responses API.
    kwargs = dict(
        model=model,
        base_url=azure_foundry_v1_base_url(),
        use_responses_api=True,
        timeout=_REQUEST_TIMEOUT, max_retries=2, max_completion_tokens=mt,
    )
    if AZURE_OPENAI_API_KEY:
        kwargs["api_key"] = AZURE_OPENAI_API_KEY
    else:
        raise RuntimeError("No Azure credential configured — set AZURE_OPENAI_API_KEY "
                           "in the repo-root .env")
    if not _omits_temperature(model):
        kwargs["temperature"] = temperature
    return ChatOpenAI(**kwargs)


@lru_cache(maxsize=None)
def _build_llm(provider: str, tier: str, model: str, temperature: float,
               mt: int) -> ChatOpenAI | _AnthropicTierClient:
    """Build and cache a client for an explicit provider and tier."""
    if provider == "azure":
        return _get_azure_llm(tier, model, temperature, mt)
    return _get_ica_anthropic(model, temperature, mt)


def get_llm(tier: str = "med", temperature: float = 0.0,
            max_tokens: int | None = None) -> ChatOpenAI | _AnthropicTierClient:
    """The LLM client for a tier under the configured provider."""
    provider = active_provider()
    model = model_for_tier(tier)
    mt = max_tokens if max_tokens is not None else _MAX_TOKENS_BY_TIER.get(tier, _DEFAULT_MAX_TOKENS)
    return _build_llm(provider, tier, model, temperature, mt)


# A backslash inside a JSON string is only legal before one of these. Models frequently emit a lone
# backslash in verbatim findings — an inch mark (5.5\ "), a weight, or a Windows path (C:\Users) — which
# is an invalid JSON escape that aborts json.loads. `_repair_escapes` doubles any such stray backslash
# so the string parses as the literal text the model intended.
_VALID_ESCAPE_CHARS = set('\\"/bfnrtu')


def _repair_escapes(s: str) -> str:
    """Double stray backslashes (those not starting a valid JSON escape) so json.loads accepts them.

    Scans pairwise so a legitimate ``\\\\`` is consumed as a unit and never corrupted — only a
    backslash that is NOT the start of a valid escape (``\\"`` ``\\\\`` ``\\/`` ``\\b`` ``\\f`` ``\\n``
    ``\\r`` ``\\t`` ``\\uXXXX``) is doubled.
    """
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch == "\\":
            nxt = s[i + 1] if i + 1 < n else ""
            if nxt in _VALID_ESCAPE_CHARS:
                out.append(s[i : i + 2])  # keep the valid escape as a unit
                i += 2
                continue
            out.append("\\\\")  # stray backslash -> escape it
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _loads(candidate: str) -> dict:
    """json.loads, but retry once with stray backslashes repaired on an 'Invalid \\escape' error."""
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        if "escape" not in str(exc).lower():
            raise
        return json.loads(_repair_escapes(candidate))


def _transient_errors() -> tuple[type[BaseException], ...]:
    """The transport-failure classes worth retrying, across both SDKs.

    Both are imported defensively: an Azure-only install may not have `anthropic`, and vice versa —
    a missing package must degrade to "nothing to retry on" rather than crash at import.
    """
    classes: list[type[BaseException]] = []
    for module in ("openai", "anthropic"):
        try:
            sdk = __import__(module)
            classes += [sdk.APITimeoutError, sdk.APIConnectionError,
                        sdk.RateLimitError, sdk.InternalServerError]
        except (ImportError, AttributeError):
            continue
    return tuple(classes)


def _invoke(llm, messages) -> LLMReply:
    """One model call as an `LLMReply`, retrying transient transport errors with exponential backoff.

    Timeouts / dropped connections / throttling / 5xx are raised straight out of invoke() (the
    structured_call parse-retry loop never catches them), so without this a single slow or throttled
    gateway response would abort — and SKIP — a whole rebuttal cycle. This is orthogonal to the
    parse/validation retry, which stays in structured_call.

    Returning the reply OBJECT rather than a bare str is what makes real token accounting possible:
    the usage block used to be discarded here, one line after the gateway reported it.
    """
    transient = _transient_errors()
    last_exc: Exception | None = None
    for attempt in range(_TRANSIENT_RETRIES + 1):
        try:
            reply = llm.invoke(messages)
            if isinstance(reply, LLMReply):     # native Anthropic client — already normalised
                return reply
            content = reply.content             # LangChain AIMessage — behaviour preserved exactly
            # Responses API (gpt-5.4-pro / any Azure Responses-routed tier) returns a list of
            # content blocks instead of a plain str; .text concatenates the text-type blocks.
            text = content if isinstance(content, str) else str(reply.text)
            # Azure reports its own counters too (in OpenAI field names), so they are carried across
            # rather than estimated — see `_langchain_usage`. An empty dict means the reply genuinely
            # carried nothing, and only then does the tiktoken fallback apply.
            return LLMReply(text=text, usage=_langchain_usage(reply), raw=reply)
        except transient as exc:
            last_exc = exc
            if attempt == _TRANSIENT_RETRIES:
                break
            wait = 2 ** attempt * 5  # 5s, 10s, 20s, …
            print(f"    LLM {type(exc).__name__}; retrying in {wait}s "
                  f"(attempt {attempt + 1}/{_TRANSIENT_RETRIES})", file=sys.stderr)
            time.sleep(wait)
    raise last_exc  # type: ignore[misc]


# Per-conversation usage anchors for the proactive context check, keyed
# f"{conv_key}|{identity(base_url, model)}" so a stale anchor can never be applied to a different
# endpoint/model. Only the ReAct loop passes a conv_key (it is the one genuinely growing buffer);
# structured_call/text_call build stateless 2-message prompts and pre-flight with no baseline, i.e.
# the conservative 1.25× local estimate.
_BASELINES: dict[str, UsageBaseline] = {}


def _trim_messages(messages, keep_recent: int | None = None) -> list:
    """Drop middle turns, keeping the system prompt, the first user turn (the task) and the tail.

    An elision marker replaces what was cut so the model is told history is missing rather than
    silently seeing a jump. A short prompt (nothing but system + task + a turn or two) is returned
    unchanged — there is nothing safe to remove.
    """
    keep = _TRIM_KEEP_RECENT if keep_recent is None else max(1, keep_recent)
    head_system = [m for m in messages[:1] if _message_role(m) == "system"]
    body = list(messages[len(head_system):])
    if len(body) <= keep + 1:
        return list(messages)
    first, tail = body[:1], body[-keep:]
    elided = len(body) - len(first) - len(tail)
    marker = HumanMessage(content=f"[{elided} earlier turn(s) elided to stay within the model's "
                                  "input limit — work from the task and the recent turns below.]")
    return [*head_system, *first, marker, *tail]


def _invoke_checked(tier: str, messages, temperature: float, conv_key: str) -> LLMReply:
    """Pre-flight the prompt against the tier's input limit, invoke, and anchor the next check.

    ICA-native only, so Azure calls pass straight through untouched. Not for want of numbers — Azure
    reports usage too now — but because everything below is Anthropic-shaped: the cache breakpoints
    `_to_anthropic` writes, the per-model input limits, and a trim that rewrites an Anthropic turn
    list. Extending it to Azure would change the Azure REQUEST, which this change deliberately does
    not touch. Over 80% of the limit the prompt is trimmed BEFORE sending (the primary defence); a
    context-overflow error coming back is a backstop that trims harder and retries once.

    This is also where prompt caching is decided, for the same reason the baseline is: `conv_key` is
    the existing signal that the caller is accumulating one growing buffer across calls, so it is the
    signal that the prompt will repeat and is worth caching. It is published on `_CACHE_REPEATING` for
    `_to_anthropic` to read.
    """
    llm = get_llm(tier, temperature)
    if not isinstance(llm, _AnthropicTierClient):
        return _invoke(llm, messages)

    # Reset on the way out, so one ReAct call cannot leave the flag set for the next stateless
    # structured_call made in the same context.
    cache_token = _CACHE_REPEATING.set(bool(conv_key))
    try:
        model, base_url = llm.model, llm.base_url
        key = f"{conv_key}|{identity(base_url, model)}"
        decision = check_threshold(messages, _BASELINES.get(key), model, base_url,
                                   limit=_max_input_tokens(tier, model))
        outgoing = list(messages)
        if decision.should_compact:
            outgoing = _trim_messages(outgoing)
            print(f"    context {decision.estimate.tokens}/{decision.limit} tokens "
                  f"({decision.ratio:.0%}, {decision.estimate.source}) — trimmed "
                  f"{len(messages)} turns to {len(outgoing)} before sending", file=sys.stderr)
        elif decision.should_warn:
            print(f"    context {decision.estimate.tokens}/{decision.limit} tokens "
                  f"({decision.ratio:.0%}, {decision.estimate.source})", file=sys.stderr)

        try:
            reply = _invoke(llm, outgoing)
        except Exception as exc:
            if not is_context_overflow_error(exc):
                raise
            harder = _trim_messages(outgoing, keep_recent=2)
            if len(harder) >= len(outgoing):
                raise                   # nothing left to cut — the prompt is irreducibly too large
            print(f"    context overflow reported ({type(exc).__name__}); retrying with "
                  f"{len(harder)} of {len(outgoing)} turns", file=sys.stderr)
            outgoing, reply = harder, _invoke(llm, harder)

        # Anchor the next turn's estimate — but only when the buffer we sent IS the caller's buffer.
        # After a trim the reported size describes a prompt the caller is not accumulating from, so
        # the `history_len` cursor would point into the wrong list; dropping the anchor makes the next
        # check fall back to a conservative local estimate instead of a confidently wrong one.
        if conv_key and reply.usage and len(outgoing) == len(messages):
            _BASELINES[key] = capture_usage(reply.usage, len(outgoing), model, base_url)
        return reply
    finally:
        _CACHE_REPEATING.reset(cache_token)


def _invoke_tier(tier: str, messages, temperature: float = 0.0,
                 conv_key: str = "") -> tuple[LLMReply, str]:
    """Invoke a tier, falling back from Azure to ICA once when Azure fails."""
    global _fallback_latched
    try:
        return _invoke_checked(tier, messages, temperature, conv_key), model_for_tier(tier)
    except Exception as exc:
        if is_context_overflow_error(exc) or not (_fallback_available() and not _fallback_latched):
            raise
        print(f"    Azure call failed ({type(exc).__name__}: {exc}); falling back to ICA for the rest "
              "of this run.", file=sys.stderr)
        _fallback_latched = True
        return _invoke_checked(tier, messages, temperature, conv_key), model_for_tier(tier)


def record_reply_usage(tier: str, model: str, reply: LLMReply, messages) -> None:
    """Record one call's tokens: provider-reported when available, tiktoken-estimated otherwise.

    Provider-blind on purpose. Both providers report, in their own field names, and this function
    reads neither name directly — it asks `token_accounting`, which is where the one rule that differs
    between them lives. Anthropic's `input_tokens` EXCLUDES the cached part of the prompt; OpenAI's
    `prompt_tokens` INCLUDES it. So the same three numbers mean different things depending on who sent
    them, and getting it backwards either doubles the prompt or loses the cache entirely.

    tokens_in convention: NEW (uncached) input only, semantics-corrected. `input_buckets()` applies
    that inclusive/exclusive check and returns the three disjoint input classes, so the cached part of
    the prompt lands in its own buckets rather than in `tokens_in`. They bill at different rates, which
    is the whole reason they are kept apart; nothing is lost, because the three add back up to the full
    prompt (reported as `prompt_tokens`).

    Never read the raw counters directly. Under exclusive semantics `input_tokens` alone reported 12
    tokens for a 12,955-token prompt, and under inclusive semantics summing all three roughly doubles
    the number. `input_buckets` is the one place that check lives.

    Not every class is reported by every provider, and that is fine rather than a gap to paper over.
    A GPT tier reports cache READS and no cache writes, because OpenAI's caching is automatic and a
    write is not billed as its own class; `cache_creation_tokens` therefore stays 0 there and the write
    rate multiplies nothing.

    tokens_out comes from `output_counters()`, which reads the counter by key presence rather than
    with an `or` chain — a reported 0 and a missing key mean different things here (see `LLMReply`).
    It also returns the reasoning/thinking subset, which is why a call can bill hundreds of output
    tokens while `reply.text` is empty: `_anthropic_reply` keeps only `text` blocks. Claude reports it
    as `thinking_tokens` and GPT as `reasoning_tokens`; both land in the same field, because both are
    output tokens spent before the answer.

    The local fallback is weaker on both sides. It estimates input with a foreign tokenizer, and it
    counts output from `reply.text` alone, so thinking tokens are missed entirely. That caveat is
    carried in the report's `note`.
    """
    usage = normalise_usage(reply.usage) if reply.usage else {}
    if any(usage.get(k) for k in ("input_tokens", "cache_read_input_tokens",
                                  "cache_creation_input_tokens", "prompt_tokens")):
        USAGE.record(tier, model, input_buckets(usage)[0], output_counters(usage)[0],
                     source="gateway", usage=usage)
    else:
        USAGE.record(tier, model, count_messages(messages, model),
                     count_tokens(reply.text, model), source="local-estimate")


def _extract_json(text: str) -> dict:
    """Pull the first balanced top-level JSON object out of a model reply."""
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in reply")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return _loads(text[start : i + 1])
    raise ValueError("unbalanced JSON object in reply")


def structured_call(tier: str, system: str, user: str, schema: type[BaseModel],
                    retries: int = 1, temperature: float = 0.0,
                    *, hook_module: str, hook_stage: str = "",
                    allow_budget_overrun: bool = False) -> BaseModel:
    """Call a tier and return a validated instance of `schema`.

    ``allow_budget_overrun`` is reserved for a terminal recovery call after a budget hook has
    stopped earlier pipeline work.  The reply remains fully accounted for and reported to the
    hook after it completes; only the pre-call block is bypassed.
    """
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    sys_msg = (
        f"{system}\n\n"
        "Respond with ONE JSON object and nothing else — no markdown fences, no prose.\n"
        f"It must validate against this JSON schema:\n{schema_json}"
    )
    messages = [SystemMessage(content=sys_msg), HumanMessage(content=user)]
    last_err = None
    stage = hook_stage or schema.__name__
    recovery_calls = {("extractor", "final_synthesis"), ("labor", "drafter")}
    if allow_budget_overrun and (hook_module, stage) not in recovery_calls:
        raise ValueError(
            "allow_budget_overrun is restricted to terminal extractor/labor recovery calls"
        )
    for _ in range(retries + 1):
        if not allow_budget_overrun:
            HOOKS.before_call(
                module=hook_module, call_type="structured_call", stage=stage,
                usage_report=USAGE.report(), usage_generation=USAGE.generation(),
            )
        reply, model = _invoke_tier(tier, messages, temperature)
        record_reply_usage(tier, model, reply, messages)
        HOOKS.after_call(
            module=hook_module, call_type="structured_call", stage=stage,
            usage_report=USAGE.report(), usage_generation=USAGE.generation(),
        )
        try:
            return schema.model_validate(_extract_json(reply.text))
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            last_err = exc
            messages.append(HumanMessage(
                content=f"Your previous reply was invalid ({exc}) — it was likely truncated or not "
                        "pure JSON. Re-emit ONE complete, valid JSON object matching the schema and "
                        "nothing else (no markdown fences, no prose); keep string values concise so "
                        "the whole object fits."))
    raise RuntimeError(f"structured_call failed after retries: {last_err}")


def text_call(tier: str, system: str, user: str, temperature: float = 0.0,
              *, hook_module: str, hook_stage: str = "text_call") -> str:
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    HOOKS.before_call(
        module=hook_module, call_type="text_call", stage=hook_stage,
        usage_report=USAGE.report(), usage_generation=USAGE.generation(),
    )
    reply, model = _invoke_tier(tier, messages, temperature)
    record_reply_usage(tier, model, reply, messages)
    HOOKS.after_call(
        module=hook_module, call_type="text_call", stage=hook_stage,
        usage_report=USAGE.report(), usage_generation=USAGE.generation(),
    )
    return reply.text


def discover_models() -> list[str]:
    """List model ids the ICA gateway exposes (GET /models). ICA provider only."""
    import httpx

    if LLM_PROVIDER == "azure":
        raise RuntimeError(
            "discover_models() is ICA-only; on Azure the available models are the configured "
            f"tier models: {sorted(set(_TIER_MODELS['azure'].values()))}")
    if not ICA_API_KEY:
        raise RuntimeError("ICA_API_KEY is not set — fill the repo-root .env")
    resp = httpx.get(
        f"{ICA_BASE_URL.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {ICA_API_KEY}"},
        timeout=30,
    )
    resp.raise_for_status()
    return [m["id"] for m in resp.json().get("data", [])]


def _auth_mode() -> str:
    """Describe the Azure credential without exposing its value.

    The length is there to catch a truncated paste (the usual cause of a 401 that looks like a
    misconfigured endpoint); the value itself must never reach a log or a terminal.
    """
    if AZURE_OPENAI_API_KEY:
        return f"api-key via AZURE_OPENAI_API_KEY ({len(AZURE_OPENAI_API_KEY)} chars)"
    return "NONE — no credential configured, every call will fail"


def llm_config() -> str:
    """The fully resolved LLM configuration as printable text — no network calls, no secret values.

    This is the cheap pre-flight for a provider switch: it shows the endpoint, tier model/deployment,
    API version, and credential mode without spending a token or leaking a key.
    """
    provider = active_provider()
    out = [f"# LLM configuration — LLM_PROVIDER={LLM_PROVIDER!r}, active provider {provider!r}"]
    if provider != _PRIMARY:
        out.append(f"#   Azure->ICA fallback latched (primary {_PRIMARY!r})")
    elif _PRIMARY == "azure":
        out.append("#   Azure->ICA fallback: "
                   + ("armed" if _fallback_available() else "unavailable (ICA_API_KEY is not set)"))
    for tier in ("high", "med", "light"):
        model = _TIER_MODELS[provider][tier]
        mt = _MAX_TOKENS_BY_TIER.get(tier, _DEFAULT_MAX_TOKENS)
        out.append(f"[{tier:>5}] model={model}  max_tokens={mt}  "
                   f"temperature={'omitted' if _omits_temperature(model) else 'sent'}")
        input_limit = _max_input_tokens(tier, model)
        out.append(f"        max_input_tokens={input_limit}  "
                   f"(compact at {int(input_limit * 0.80)})")
        if provider != "azure":
            continue
        out.append(f"        endpoint={azure_foundry_v1_base_url() if AZURE_OPENAI_ENDPOINT else '(unset)'}")
        out.append(f"        model={model}  api_version=not used (Foundry v1)")
        out.append(f"        auth={_auth_mode()}")
    if provider == "azure":
        out.append("# responses API: forced for every Azure tier")
        # Both providers report their own usage now, so say so here too. Which route a tier takes
        # does not change that: `_langchain_usage` reads LangChain's normalised `usage_metadata`,
        # which carries the cached and reasoning sub-counters over Chat Completions AND Responses.
        out.append("#   prompt caching: automatic and server-side (nothing is requested; a cache "
                   "write is not a billed class on OpenAI)")
        out.append("#   token accounting: provider-reported usage in OpenAI field names — "
                   "prompt_tokens INCLUDES cached_tokens, and reasoning_tokens is reported as the "
                   "thinking subset of the output")
    else:
        out.append("# ICA client=anthropic.Anthropic (native POST /v1/messages, auth header x-api-key)")
        out.append(f"#   base_url={ica_anthropic_base_url()}"
                   + ("  (ICA_ANTHROPIC_BASE_URL override)"
                      if os.getenv("ICA_ANTHROPIC_BASE_URL") else ""))
        out.append(f"#   api_key="
                   f"{'set (' + str(len(ICA_API_KEY)) + ' chars)' if ICA_API_KEY else 'NOT SET'}")
        out.append("#   system-prefix caching: "
                   + (f"armed (cache_control on system prompts >= {_CACHE_MIN_TOKENS} est. tokens)"
                      if _CACHE_SYSTEM_PREFIX else "off (MODEL_CACHE_SYSTEM_PREFIX=0)"))
        out.append("#   conversation caching: "
                   + (f"armed (2 rolling breakpoints on repeating buffers >= {_CACHE_MIN_TOKENS} "
                      "est. tokens; ReAct only)"
                      if _CACHE_CONVERSATION else "off (MODEL_CACHE_CONVERSATION=0)"))
        _ttl = cache_ttl()
        out.append(f"#   cache lifetime: {_ttl} (MODEL_CACHE_TTL; writes bill at about "
                   f"{CACHE_WRITE_MULTIPLIER[_ttl]}x the input rate)")
        out.append(f"#   token accounting: gateway-reported usage; debug line "
                   f"{'on' if _TOKEN_DEBUG else 'off (TOKEN_ACCOUNTING_DEBUG=0)'}")
    return "\n".join(out)


def smoke_test(*, hook_module: str) -> dict:
    """One round-trip per tier; returns {tier: reply or error}."""
    out = {}
    for tier in _TIERS:
        try:
            out[tier] = text_call(tier, "You are a connectivity probe.",
                                  "Reply with the single word: pong",
                                  hook_module=hook_module, hook_stage="smoke_test")
        except Exception as exc:
            out[tier] = f"ERROR: {type(exc).__name__}: {exc}"
    return out


if __name__ == "__main__":
    # `python -m shared.llm` prints the resolved config, then probes each tier. `--config` stops
    # after the config so it works with no network (and no spend) — the fast way to check a provider
    # switch. Both runners expose the same two things as --llm-config / --smoke-test.
    import sys

    print(llm_config())
    if "--config" not in sys.argv[1:]:
        _module = (os.getenv("LLM_HOOK_MODULE") or "").strip()
        if not _module:
            sys.exit("Set LLM_HOOK_MODULE when running shared.llm smoke tests directly")
        print()
        for tier, reply in smoke_test(hook_module=_module).items():
            print(f"[{tier:>5}] {model_for_tier(tier)}: {reply!r}")
