"""Gateway-reported context accounting for the Anthropic-shaped LiteLLM proxy.

This backs the native Anthropic path in `llm.py` — `LLM_PROVIDER=ica` talking to
`ICA_BASE_URL` through the official `anthropic` SDK rather than the OpenAI-compat shim.
See docs/GATEWAY_FINDINGS.md
for the full Probe A--F record.

Probe A--F established that this gateway reports Anthropic usage field names
(``input_tokens``, ``output_tokens``, ``cache_read_input_tokens``, and
``cache_creation_input_tokens``), with cache creation sometimes additionally
broken down under ``cache_creation``.  Cache counters are *exclusive* on the
measured gateway: its input counter does not include cache read/creation
input.  That is detected from every captured usage record rather than assumed,
because another proxy can report inclusive or OpenAI-shaped counters — and the
probes ran against a SIBLING LiteLLM deployment (same proxy software, same three
model ids) since the project key was revoked, so ICA itself may differ.  Runtime
detection is what makes that harmless.  The OpenAI-compat route on this very
gateway reports inclusive counters with caching disabled, and Azure is a third
case, so nothing here is hardcoded either way.  Detection reads the usage
DIALECT first (which field names the provider chose) and only falls back to
comparing magnitudes, because the magnitude rule alone cannot tell a small cache
read behind a large uncached tail from an inclusive counter.  Probe limits reported a
1,000,000-token input limit for the Claude models.  The error probes also found
LiteLLM double-encodes upstream errors and appends a model-group trailer, so
overflow detection searches flattened error text.

No tokenizer dependency is used here.  When reported usage is unavailable,
the local fallback is deliberately a conservative content-aware character
estimate, not an OpenAI tokenizer approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import os
import time
from typing import Any, Literal


# Measured through GET /v1/model/info, which reports max_input_tokens=1000000 for
# the Claude models on this gateway.  (GET /v1/models/{id} IS proxied but returns
# max_input_tokens=null, so it is useless for this.)  Corroborated by Probe E: a
# 576,198-token prompt was accepted with no error and a needle test proved there is
# no silent truncation.  llm.py can override per tier via MODEL_MAX_INPUT_TOKENS[_<TIER>]
# — see `check_threshold(limit=...)`.  Unknown models keep the conservative default.
MODEL_INPUT_LIMITS: dict[str, int] = {
    # Azure GPT-5.6 tier map: high / med / light
    "gpt-5.6-sol": 1_050_000,
    "gpt-5.6-terra": 1_050_000,
    "gpt-5.6-luna": 1_050_000,
    # gen-5 (the current ICA tier map: high / med / light)
    "claude-opus-5": 1_000_000,
    "claude-sonnet-5": 1_000_000,
    "claude-haiku-4-5": 1_000_000,
    # gen-4 ids still selectable via ICA_MODEL_<TIER>
    "claude-opus-4-8": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
}
DEFAULT_INPUT_LIMIT: int = 200_000
COMPACT_THRESHOLD = 0.80
WARN_THRESHOLD = 0.60

PER_MESSAGE_OVERHEAD_TOKENS = 4
PER_TOOL_BLOCK_OVERHEAD_TOKENS = 12
PER_IMAGE_TOKENS = 1_600
HUGE_STRING_CHAR_CUTOFF = 200_000
CHARS_PER_TOKEN = 3.5
BASELINE_MARGIN = 1.05
LOCAL_ESTIMATE_MARGIN = 1.25

# How long a prompt-cache entry lives.  Anthropic offers two lifetimes, and the choice is a BILLING
# choice as much as a timing one, so it is resolved here rather than in `llm.py`: the TTL sets both
# what a cache write costs and which `cache_creation` sub-bucket the gateway reports the tokens in.
#
#   5m   write costs about 1.25x the input rate.  Pays for itself from the 2nd read on.
#   1h   write costs about 2x the input rate.  Needs 3 reads to pay for itself, but survives a
#        gap between steps that would let a 5-minute entry expire.
#
# 1h is the default because this pipeline's callers re-send the same buffer many times per node —
# well past 3 reads — and a slow `run_python` step (the timeout alone is 150s) can easily open a gap
# wider than five minutes.  Both `llm.py` (which writes the breakpoints) and `pricing.py` (which
# picks the matching write rate) read this, so the two can never disagree.
CACHE_TTLS = ("5m", "1h")
DEFAULT_CACHE_TTL = "1h"
# The write-rate multiple over the plain input rate, per TTL.  Documentation only: real rates come
# from the price-map config file, never from a multiplier applied to `in`.
CACHE_WRITE_MULTIPLIER = {"5m": 1.25, "1h": 2.0}

logger = logging.getLogger(__name__)
# A library must not configure the application's root logger.  This handler
# prevents "No handler" warnings while still allowing application handlers.
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

_ttl_warned: set[str] = set()


def cache_ttl() -> str:
    """The configured prompt-cache lifetime — ``"5m"`` or ``"1h"``, from ``MODEL_CACHE_TTL``.

    Read fresh on every call instead of frozen at import, so setting the variable in a test (or in a
    `.env` file loaded after this module) reaches both readers.  An unrecognised value falls back to
    the default and warns once per bad value: a typo that silently halved the cache lifetime would
    otherwise show up only as a mysteriously worse cost estimate.
    """
    raw = (os.getenv("MODEL_CACHE_TTL") or "").strip().lower()
    if not raw:
        return DEFAULT_CACHE_TTL
    if raw in CACHE_TTLS:
        return raw
    if raw not in _ttl_warned:
        _ttl_warned.add(raw)
        logger.warning("ignoring MODEL_CACHE_TTL=%r (expected one of %s); using %s",
                       raw, ", ".join(CACHE_TTLS), DEFAULT_CACHE_TTL)
    return DEFAULT_CACHE_TTL


@dataclass(frozen=True)
class UsageBaseline:
    usage: dict
    history_len: int
    identity: str
    reported_prompt_size: int
    semantics: str
    captured_at: float


@dataclass(frozen=True)
class EffectiveEstimate:
    tokens: int
    source: Literal["baseline", "local-estimate"]
    semantics: str
    reason: str
    raw_usage: dict
    margin_applied: float


@dataclass(frozen=True)
class ThresholdDecision:
    should_compact: bool
    should_warn: bool
    estimate: EffectiveEstimate
    limit: int
    threshold: int
    ratio: float


def identity(base_url: str, model: str) -> str:
    """Return the endpoint/model key to which a usage baseline belongs."""
    return f"{base_url}::{model}"


def normalise_usage(response_or_usage: Any) -> dict:
    """Convert common SDK response containers to a usage mapping unchanged."""
    if isinstance(response_or_usage, dict):
        candidate: Any = response_or_usage
    else:
        candidate = response_or_usage
        for method_name in ("model_dump", "to_dict"):
            method = getattr(candidate, method_name, None)
            if callable(method):
                candidate = method()
                break
        else:
            try:
                candidate = vars(candidate)
            except TypeError:
                candidate = {}
    if not isinstance(candidate, dict):
        return {}
    usage = candidate.get("usage")
    return dict(usage) if isinstance(usage, dict) else dict(candidate)


def _number(value: Any) -> int:
    """Coerce reported counters without allowing malformed values to crash flow."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _cache_creation_tokens(usage: dict) -> int:
    flat = usage.get("cache_creation_input_tokens")
    if flat is not None:
        return _number(flat)
    nested = usage.get("cache_creation")
    if isinstance(nested, dict) and nested:
        return sum(_number(value) for value in nested.values())
    for parent in ("prompt_tokens_details", "input_tokens_details"):
        value = _sub_counter(usage, parent, "cache_write_tokens", "cache_creation_tokens",
                             "cache_creation_input_tokens", "cache_write", "cache_creation")
        if value:
            return value
    return 0


def _sub_counter(usage: dict, parent: str, *names: str) -> int:
    """Read a nested detail counter, e.g. ``output_tokens_details.thinking_tokens``."""
    details = usage.get(parent, {})
    if not isinstance(details, dict):
        return 0
    for name in names:
        if name in details:
            return _number(details.get(name))
    return 0


def _input_and_cache(usage: dict) -> tuple[int, int, int]:
    input_tokens = _number(usage.get("input_tokens", usage.get("prompt_tokens", 0)))
    if "cache_read_input_tokens" in usage:
        read = _number(usage.get("cache_read_input_tokens"))
    else:
        read = 0
        for parent in ("prompt_tokens_details", "input_tokens_details"):
            read = _sub_counter(usage, parent, "cached_tokens", "cache_read_tokens",
                                "cache_read_input_tokens", "cache_read", "cached")
            if read:
                break
    return input_tokens, read, _cache_creation_tokens(usage)


def cache_counters(usage: Any) -> tuple[int, int]:
    """Return (cache_read_input_tokens, cache_creation_input_tokens) from any usage shape.

    Public so the per-tier aggregator in `tokens.py` reads the cache buckets through the
    same parser the semantics detection uses — including the nested ``cache_creation``
    TTL breakdown and the OpenAI-shaped ``prompt_tokens_details.cached_tokens`` fallback.
    """
    _, read, creation = _input_and_cache(normalise_usage(usage))
    return read, creation


# Field names that identify which dialect a usage block is written in.  The dialect
# determines the cache semantics on its own, so it is checked BEFORE the magnitude
# heuristic below.
_ANTHROPIC_KEYS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
                   "cache_creation")
_OPENAI_KEYS = ("prompt_tokens", "prompt_tokens_details", "input_tokens_details")


def detect_semantics(usage: dict) -> str:
    """Classify input counter semantics: by usage dialect first, then by magnitude.

    Anthropic's ``/v1/messages`` never double-counts — ``input_tokens`` excludes both cache
    buckets — while OpenAI-compat ``prompt_tokens`` always includes ``cached_tokens``.  So
    the field names the provider chose are stronger evidence than comparing the numbers, and
    the dialect check runs first.

    The magnitude rule (``input_tokens >= read + creation`` means inclusive) is kept as the
    fallback for an unrecognised dialect, but it cannot decide an Anthropic block on its own:
    a small cache read behind a large uncached tail (input=5000, read=200) satisfies it and
    would be misread as inclusive, under-reporting the prompt by the cached 200.  That case is
    only reachable once caching is actually enabled, which is why the dialect check exists.

    Note that the magnitude rule is currently UNREACHABLE, and is retained only as a safe
    default: a nonzero cache counter can only be parsed out of a key that is itself a dialect
    marker (see ``_input_and_cache``), so any block that gets this far has already matched one
    of the two lists.  It would come back into play if a third dialect were added to the
    parser without being added to the lists.
    """
    normalised = normalise_usage(usage)
    input_tokens, read, creation = _input_and_cache(normalised)
    if input_tokens <= 0 and read <= 0 and creation <= 0:
        return "unreported"
    if read + creation == 0:
        return "no-cache"
    if any(key in normalised for key in _OPENAI_KEYS):
        return "inclusive"
    if any(key in normalised for key in _ANTHROPIC_KEYS):
        return "exclusive"
    if input_tokens >= read + creation:
        return "inclusive"
    return "exclusive"


def reported_prompt_size(usage: dict) -> int:
    """Return total current prompt tokens according to detected counter semantics.

    This is the FULL prompt — cached input included — because that is what occupies the
    model's context window, which is the question `check_threshold` asks.  For the billing
    question ("how much of this was new?") use `input_buckets`, whose three disjoint classes
    always add back up to this number.
    """
    normalised = normalise_usage(usage)
    input_tokens, read, creation = _input_and_cache(normalised)
    if detect_semantics(normalised) == "exclusive":
        return input_tokens + read + creation
    return input_tokens


def input_buckets(usage: Any) -> tuple[int, int, int]:
    """Split the prompt into (new_input, cache_read, cache_creation) — three DISJOINT classes.

    These are the three rates a prompt is billed at: new input at 1x, a cache read at about 0.1x,
    and a cache write at a multiple that depends on the lifetime asked for — about 1.25x for a
    5-minute entry, about 2x for a 1-hour one (see `cache_ttl`).  `tokens.py` records them
    separately so a warm run is priceable, and `sum(input_buckets(u)) == reported_prompt_size(u)`
    always holds, so the full prompt size is never lost.

    Under exclusive semantics the counters are already disjoint and are returned as reported.
    Under inclusive semantics the cache buckets are carved back out of the input counter.
    """
    normalised = normalise_usage(usage)
    input_tokens, read, creation = _input_and_cache(normalised)
    if detect_semantics(normalised) == "inclusive":
        return max(0, input_tokens - read - creation), read, creation
    return input_tokens, read, creation


def output_counters(usage: Any) -> tuple[int, int]:
    """Return (output_tokens, thinking_tokens) — the second is a SUBSET of the first.

    Extended thinking is on for gen-5 Claude and its tokens are billed as output, so they are
    already inside ``output_tokens``.  They are reported separately because `llm._anthropic_reply`
    only concatenates ``text`` blocks: a thinking-heavy reply can bill hundreds of output tokens
    while `LLMReply.text` is empty, and without the split that reads as a bug.

    Read by key PRESENCE, not by an ``or`` chain, so a reported 0 is not collapsed with a
    missing key (the `LLMReply` contract).
    """
    normalised = normalise_usage(usage)
    output = 0
    for name in ("output_tokens", "completion_tokens"):
        if name in normalised:
            output = _number(normalised.get(name))
            break
    thinking = _sub_counter(normalised, "output_tokens_details", "thinking_tokens",
                            "reasoning_tokens")
    if not thinking:
        thinking = _sub_counter(normalised, "completion_tokens_details", "reasoning_tokens",
                                "thinking_tokens")
    return output, min(thinking, output) if output else thinking


def capture_usage(response: Any, history_len: int, model: str, base_url: str) -> UsageBaseline:
    """Capture gateway usage immediately after a request for use by the next turn."""
    usage = normalise_usage(response)
    return UsageBaseline(
        usage=usage,
        history_len=history_len,
        identity=identity(base_url, model),
        reported_prompt_size=reported_prompt_size(usage),
        semantics=detect_semantics(usage),
        captured_at=time.time(),
    )


def _string_tokens(text: str) -> int:
    # BPE regexes can backtrack catastrophically on repetitive huge inputs.
    if len(text) > HUGE_STRING_CHAR_CUTOFF:
        return math.ceil(len(text) / CHARS_PER_TOKEN)
    # A 3.5-character prose heuristic, rounded up, is intentionally cautious.
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def _is_image_block(block: dict) -> bool:
    source = block.get("source")
    return block.get("type") == "image" or (
        isinstance(source, dict) and ("data" in source or "base64" in source)
    )


def _content_tokens(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return _string_tokens(content)
    if isinstance(content, (int, float, bool)):
        return _string_tokens(str(content))
    if isinstance(content, list):
        return sum(_content_tokens(item) for item in content)
    if isinstance(content, dict):
        if _is_image_block(content):
            # Images are billed by pixels; tokenizing 1 MB base64 would invent
            # roughly 250K phantom tokens, so payload is always skipped.
            return PER_IMAGE_TOKENS
        text = content.get("text")
        if isinstance(text, str):
            return _string_tokens(text)
        try:
            return _string_tokens(json.dumps(content, sort_keys=True, default=str))
        except (TypeError, ValueError):
            return _string_tokens(str(content))
    return _string_tokens(str(content))


def estimate_message_tokens(messages: list[Any], system: Any = None, tools: Any = None) -> int:
    """Estimate message, system, and tool-schema input without tokenizing images."""
    total = 0
    if system is not None:
        total += PER_MESSAGE_OVERHEAD_TOKENS + _content_tokens(system)
    for message in messages:
        total += PER_MESSAGE_OVERHEAD_TOKENS
        if isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = getattr(message, "content", message)
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in {"tool_use", "tool_result"}:
                    total += PER_TOOL_BLOCK_OVERHEAD_TOKENS
                total += _content_tokens(block)
        else:
            total += _content_tokens(content)
    if tools is not None:
        try:
            schema = json.dumps(tools, sort_keys=True, default=str)
        except (TypeError, ValueError):
            schema = str(tools)
        total += PER_TOOL_BLOCK_OVERHEAD_TOKENS + _string_tokens(schema)
    return total


def effective_input_tokens(
    messages: list[Any], baseline: UsageBaseline | None, model: str, base_url: str,
    system: Any = None, tools: Any = None,
) -> EffectiveEstimate:
    """Use compatible gateway usage plus appended history, or conservatively estimate all."""
    fallback_reason: str | None = None
    if baseline is None:
        fallback_reason = "no usage baseline"
    elif identity(base_url, model) != baseline.identity:
        fallback_reason = "baseline identity does not match endpoint/model"
    elif len(messages) < baseline.history_len:
        fallback_reason = "history was rewritten or truncated since baseline"
    else:
        semantics = detect_semantics(baseline.usage)
        if semantics == "unreported":
            fallback_reason = "unreported/zero input-side usage counters in baseline"
        else:
            # history_len is captured before the assistant response is appended:
            # that response is in this delta, so never add output_tokens again.
            delta = estimate_message_tokens(messages[baseline.history_len:])
            raw = reported_prompt_size(baseline.usage) + delta
            return EffectiveEstimate(
                tokens=math.ceil(raw * BASELINE_MARGIN),
                source="baseline",
                semantics=semantics,
                reason="reported baseline plus appended-message delta",
                raw_usage=baseline.usage,
                margin_applied=BASELINE_MARGIN,
            )
    # Local estimates lack gateway truth and deserve a larger safety margin.
    raw = estimate_message_tokens(messages, system=system, tools=tools)
    return EffectiveEstimate(
        tokens=math.ceil(raw * LOCAL_ESTIMATE_MARGIN),
        source="local-estimate",
        semantics=detect_semantics(baseline.usage) if baseline else "unreported",
        reason=fallback_reason or "local estimate required",
        raw_usage=baseline.usage if baseline else {},
        margin_applied=LOCAL_ESTIMATE_MARGIN,
    )


def check_threshold(
    messages: list[Any], baseline: UsageBaseline | None, model: str, base_url: str,
    system: Any = None, tools: Any = None, limit: int | None = None,
) -> ThresholdDecision:
    """Calculate proactive compact/warn decisions and emit one diagnostic record.

    `limit` overrides the MODEL_INPUT_LIMITS lookup, so the caller can honour an env
    override (llm.py resolves MODEL_MAX_INPUT_TOKENS[_<TIER>]) without this module
    reading the environment.
    """
    estimate = effective_input_tokens(messages, baseline, model, base_url, system, tools)
    limit = limit or MODEL_INPUT_LIMITS.get(model, DEFAULT_INPUT_LIMIT)
    threshold = int(limit * COMPACT_THRESHOLD)
    ratio = estimate.tokens / limit if limit else 0.0
    decision = ThresholdDecision(
        should_compact=estimate.tokens >= threshold,
        should_warn=estimate.tokens >= int(limit * WARN_THRESHOLD),
        estimate=estimate,
        limit=limit,
        threshold=threshold,
        ratio=ratio,
    )
    logger.info(
        "token_accounting estimate=%s limit=%s threshold=%s ratio=%.6f source=%s "
        "semantics=%s reason=%s raw_usage=%r",
        estimate.tokens, limit, threshold, ratio, estimate.source,
        estimate.semantics, estimate.reason, estimate.raw_usage,
    )
    return decision


_OVERFLOW_PHRASES = (
    "prompt is too long", "context window", "context_length_exceeded", "too many tokens",
    "input length", "exceeds the model limit", "maximum context",
)


def _flatten_error(value: Any, parts: list[str]) -> None:
    if isinstance(value, str):
        parts.append(value)
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        _flatten_error(parsed, parts)
    elif isinstance(value, dict):
        for item in value.values():
            _flatten_error(item, parts)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _flatten_error(item, parts)


def is_context_overflow_error(exc: BaseException) -> bool:
    """Broadly identify proxy-wrapped context overflows for reactive recovery."""
    # A genuine overflow was never reproduced (576,198 tokens was accepted;
    # actual Sonnet limit is 1M), so this is broad; 80% compaction is defence.
    parts: list[str] = []
    _flatten_error(str(exc), parts)
    _flatten_error(getattr(exc, "body", None), parts)
    _flatten_error(getattr(exc, "message", None), parts)
    text = " ".join(parts).lower()
    if any(phrase in text for phrase in _OVERFLOW_PHRASES):
        return True
    status = getattr(exc, "status_code", None)
    return status in {400, 413} and any(
        phrase in text for phrase in ("token", "context", "input", "length", "size", "limit")
    )


def _smoke_demo() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base_url = "https://gateway.example/v1"
    model = "claude-sonnet-4-6"
    messages = [{"role": "user", "content": "hello"}]
    baseline = capture_usage({"usage": {"input_tokens": 100, "cache_read_input_tokens": 900}}, 1, model, base_url)
    check_threshold(messages + [{"role": "assistant", "content": "A brief reply."}], baseline, model, base_url)
    check_threshold(messages, None, model, base_url, system="You are helpful.")


if __name__ == "__main__":
    _smoke_demo()
