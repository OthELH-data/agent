"""Model pricing, loaded from a config file — no rates are hardcoded in this repo.

A run's dollar cost is a contract question, not a code question: list prices change, and a
gateway (ICA here) can bill differently from the model vendor. So this module ships NO numbers.
It reads a price map from `configs/model_pricing.json` (override with `MODEL_PRICING_FILE`) and,
when that file is absent or a tier is unpriced, says so explicitly instead of guessing. See
`configs/model_pricing.example.json` for the shape.

Rates are looked up by MODEL first and by TIER second, and within either by TOKEN CLASS — the four
different rates a single call is billed at:

    in            new (uncached) input tokens
    cache_write   input tokens written into the prompt cache   (rate depends on the TTL, below)
    cache_read    input tokens served from the prompt cache    (about 0.1x `in`)
    out           output tokens, extended thinking included

Model-first is what lets one file price both providers. A tier name (`high` / `med` / `light`, per
`otl/llm_adapter._NODE_TIER`) says how important a call was, not what it cost: `high` is
claude-opus-5 on ica and a gpt-5 model on azure, at different rates. So the file's `models` object,
keyed by the exact model id the run recorded, wins; its `tiers` object is the fallback for a model
that has no entry of its own, which keeps a tier-only price map working unchanged.

Not every provider bills every class, and that is expressed rather than fudged. OpenAI's prompt
caching is automatic and a cache WRITE is not billed as a separate class, so a `models` entry can
declare `"cache_write_billed": false` and satisfy the all-four-rates rule below with an explicit zero
instead of an invented number. It costs nothing either way — a GPT reply reports no write tokens, so
the write rate multiplies zero — but "declared free" and "rate unknown" must not look alike.

A cache WRITE is the one class whose rate depends on how long the entry is asked to live: about
1.25x `in` for a 5-minute entry, about 2x for a 1-hour one. So the config file carries both columns
(`cache_write_5m` and `cache_write_1h`) and this module prices whichever one `MODEL_CACHE_TTL`
selected for the run, reporting the choice as `cache_ttl`. Reading the TTL here rather than fixing
one write rate in the file is what stops a flip of that variable from silently mispricing every run
afterwards. A file that carries only a plain `cache_write` still works: it is used for either TTL.

`estimate_cost` multiplies those four rates by the four matching aggregates `tokens.report()`
already produces per tier. It is the FINAL aggregate that gets priced — nothing is priced per
call — so this stays four multiplications per tier no matter how long the run was.

Following `otl/authority.py`: an env-overridable path, a cached loader, and validation that
complains rather than silently substituting a default. Unlike that module it never raises. Pricing
is a reporting nicety appended at the end of every case, and it must not be able to fail a run
that has already done its work.
"""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from shared.token_accounting import CACHE_TTLS, cache_ttl

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PRICING = _REPO_ROOT / "configs" / "model_pricing.json"

# The four token classes a tier is priced per, in report order.
RATE_CLASSES = ("in", "cache_write", "cache_read", "out")

# `cache_write` is the one class whose rate depends on the cache lifetime, so the config carries one
# column per TTL under these keys and `estimate_cost` picks the matching one.
WRITE_RATE_KEYS = {ttl: f"cache_write_{ttl}" for ttl in CACHE_TTLS}

# Which aggregate from `tokens.report()` each rate class multiplies.
_CLASS_TOKENS = {
    "in": "tokens_in",                      # new input only, once B lands
    "cache_write": "cache_creation_tokens",
    "cache_read": "cache_read_tokens",
    "out": "tokens_out",
}

# Rates are quoted per this many tokens.
TOKENS_PER_UNIT = 1_000_000

# Sub-cent precision. Every dollar figure is rounded to this many places, and the totals are
# summed from the rounded parts, so a reader can add the columns up by hand and get the total.
_USD_PLACES = 6


def pricing_path() -> Path:
    """Resolved price-map path. The env override keeps contract rates out of the repo."""
    return Path(os.getenv("MODEL_PRICING_FILE") or _DEFAULT_PRICING).expanduser().resolve()


def _rate(value: Any) -> float | None:
    """A rate is a non-negative number, or None meaning "not priced". Anything else is None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value >= 0 else None


def _entry(spec: dict) -> dict:
    """Parse one rate set — a `tiers` entry or a `models` one; the keys are identical."""
    entry: dict[str, Any] = {"model": str(spec.get("model") or "")}
    entry.update({cls: _rate(spec.get(cls)) for cls in RATE_CLASSES})
    # A provider whose cache writes are not billed as their own class (OpenAI) declares that here,
    # and both write columns become an explicit 0. Without this the entry would need a made-up write
    # rate to clear the all-four-rates rule, or would be dropped as unpriced and silently contribute
    # nothing — two ways of being wrong about a number that is genuinely zero.
    billed = spec.get("cache_write_billed")
    entry["cache_write_billed"] = billed is not False
    if billed is False:
        for ttl_key in WRITE_RATE_KEYS.values():
            entry[ttl_key] = 0.0
        entry["cache_write"] = 0.0
        return entry
    # Both write columns are kept, so the run's TTL decides which one is used. A file that
    # predates the split and carries only `cache_write` supplies both, which keeps an older
    # hand-written price map working instead of quietly dropping the tier as unpriced.
    for ttl_key in WRITE_RATE_KEYS.values():
        rate = _rate(spec.get(ttl_key))
        entry[ttl_key] = rate if rate is not None else entry["cache_write"]
    return entry


@lru_cache(maxsize=1)
def price_map() -> dict:
    """The parsed price map, or `{}` when there is no usable one.

    `{}` is the honest answer for "no pricing available" and is what makes the report emit
    `priced: false`. A malformed file complains on stderr and is then treated as absent — a bad
    JSON file must not take down a completed extraction.

    Either top-level object is enough on its own: `models` alone is a perfectly good price map, and so
    is `tiers` alone (what every file written before the two providers diverged looks like).
    """
    path = pricing_path()
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"    ignoring unreadable model pricing file {path}: {exc}", file=sys.stderr)
        return {}
    if not isinstance(raw, dict) or not any(
            isinstance(raw.get(section), dict) for section in ("models", "tiers")):
        print(f"    ignoring model pricing file {path}: expected an object with a `models` and/or "
              f"`tiers` object", file=sys.stderr)
        return {}
    parsed = {
        section: {str(key): _entry(spec)
                  for key, spec in (raw.get(section) or {}).items() if isinstance(spec, dict)}
        for section in ("models", "tiers")
    }
    return {
        "rates_source": str(raw.get("rates_source") or ""),
        "as_of": str(raw.get("as_of") or ""),
        "currency": str(raw.get("currency") or "USD"),
        **parsed,
    }


def _amount(tokens: int, rate: float) -> float:
    return round(tokens / TOKENS_PER_UNIT * rate, _USD_PLACES)


def _resolve(prices: dict, tier: str, model: str) -> tuple[dict, str, str]:
    """The rate set for a tier that ran `model`, plus how it was found: (spec, basis, key).

    The recorded model id wins, because that is the thing that actually has a price. The tier is the
    fallback, and `basis` records which one answered so the cost block can be audited without
    re-reading the file — a `tier` basis on a two-provider setup is the shape of a mispricing.
    """
    by_model = prices.get("models") or {}
    if model and model in by_model:
        return by_model[model], "model", model
    return (prices.get("tiers") or {}).get(tier) or {}, "tier", tier


def estimate_cost(by_tier: dict[str, dict]) -> dict:
    """Price a finished run from its per-tier token aggregates.

    `by_tier` is `tokens.report()`'s own `by_tier` mapping — each entry needs `model`,
    `tokens_in`, `cache_creation_tokens`, `cache_read_tokens`, `tokens_out` and `prompt_tokens`,
    and optionally `models_seen`.

    Rates come from the recorded model id where the file has an entry for it, and from the tier
    otherwise — see `_resolve`. Each priced tier records which of the two answered (`rates_basis`)
    and under what key (`rates_key`).

    A tier is priced only when ALL FOUR of its rates are numbers. A partial rate set would
    produce a total that looks complete but silently omits a class, so such a tier is named in
    `unpriced_tiers` and contributes nothing. A provider with no cache-write class declares
    `cache_write_billed: false` and gets an explicit 0 for that column, which is a number and prices
    normally.

    Two things are named instead of being smoothed over, because both make a total less trustworthy
    than it looks:

      * `model_mismatch` — the tier fell back to TIER rates and its recorded model is not the one
        those rates were written for. Still priced, since the rate is per tier, but a silent tier
        remap becomes visible. Not raised when a `models` entry answered: there the id matched by
        definition.
      * `mixed_model_tiers` — the tier ran more than one model (the Azure→ICA fallback latching
        part-way through), so its tokens were billed against two price lists while this function can
        only apply one. It prices at the last model's rates and says so.

    `uncached_equivalent` prices the WHOLE prompt at the plain `in` rate and adds the same output
    cost: what this run would have cost with prompt caching off. Output is included so the
    subtraction against `total` is apples-to-apples, and `cache_savings` is that difference.
    """
    prices = price_map()
    ttl = cache_ttl()
    priced: dict[str, dict] = {}
    unpriced: list[str] = []
    mismatch: list[dict] = []
    mixed_models: list[dict] = []

    for tier, agg in by_tier.items():
        model = str(agg.get("model") or "")
        spec, basis, key = _resolve(prices, tier, model)
        rates = {cls: spec.get(cls) for cls in RATE_CLASSES}
        # The write rate is whichever column matches the lifetime this run actually asked for.
        rates["cache_write"] = spec.get(WRITE_RATE_KEYS[ttl])
        if any(rates[cls] is None for cls in RATE_CLASSES):
            unpriced.append(tier)
            continue
        if basis == "tier" and spec.get("model") and model and spec["model"] != model:
            mismatch.append({"tier": tier, "recorded": model, "priced_for": spec["model"]})
        seen = [str(m) for m in (agg.get("models_seen") or []) if m]
        if len(seen) > 1:
            mixed_models.append({"tier": tier, "models": seen, "priced_as": model})
        amounts = {cls: _amount(int(agg.get(_CLASS_TOKENS[cls], 0) or 0), rates[cls])
                   for cls in RATE_CLASSES}
        total = round(sum(amounts.values()), _USD_PLACES)
        uncached = _amount(int(agg.get("prompt_tokens", 0) or 0), rates["in"]) \
            + amounts["out"]
        priced[tier] = {
            "model": model,
            # How this tier's rates were found, so a reader can tell a model-exact price from a
            # tier-shaped guess without opening the price map.
            "rates_basis": basis,
            "rates_key": key,
            # False means this model's provider does not bill cache writes as their own class, so the
            # $0 on the cache_write line is a fact about the price list rather than a missing rate.
            "cache_write_billed": bool(spec.get("cache_write_billed", True)),
            "input": amounts["in"],
            "cache_write": amounts["cache_write"],
            "cache_read": amounts["cache_read"],
            "output": amounts["out"],
            "total": total,
            "uncached_equivalent": round(uncached, _USD_PLACES),
            "cache_savings": round(uncached - total, _USD_PLACES),
        }

    block: dict[str, Any] = {
        "priced": bool(priced),
        "currency": prices.get("currency", "USD"),
        "rates_source": prices.get("rates_source", ""),
        "as_of": prices.get("as_of", ""),
        "rates_file": str(pricing_path()) if prices else "",
        # Which cache lifetime the run asked for, and therefore which write column was priced.
        "cache_ttl": ttl,
        "unpriced_tiers": unpriced,
        "model_mismatch": mismatch,
        "mixed_model_tiers": mixed_models,
        "by_tier": priced,
    }
    if not priced:
        # No dollar figures at all rather than a confident $0.00.
        block["note"] = (
            f"No usable rates for any tier — cost not estimated. Provide a price map at "
            f"{pricing_path()} (or set MODEL_PRICING_FILE); see "
            f"configs/model_pricing.example.json for the shape.")
        return block
    total = round(sum(t["total"] for t in priced.values()), _USD_PLACES)
    uncached = round(sum(t["uncached_equivalent"] for t in priced.values()), _USD_PLACES)
    block["total_usd"] = total
    block["uncached_equivalent_usd"] = uncached
    block["cache_savings_usd"] = round(uncached - total, _USD_PLACES)
    # Six decimal places look authoritative, so say plainly what the number is and is not. The rates
    # are whatever the config file holds, and the gateway in front of the model may bill differently.
    parts = [f"Estimate only, from the rates in {pricing_path().name} "
             f"({block['rates_source'] or 'source not named'}). Not a bill: the gateway may charge "
             f"differently.",
             f"Cache writes are priced at the {ttl} rate, the lifetime this run asked for."]
    free_writes = [t for t, p in priced.items() if not p["cache_write_billed"]]
    if free_writes:
        parts.append(f"Tier(s) {', '.join(free_writes)} run a model whose provider does not bill "
                     f"cache writes as a separate class, so their cache_write line is 0 by "
                     f"declaration, not for want of a rate.")
    if unpriced:
        parts.append(f"Rates missing for tier(s) {', '.join(unpriced)} — their tokens are EXCLUDED "
                     f"from total_usd.")
    if mismatch:
        parts.append("A tier fell back to tier rates written for a different model (see "
                     "model_mismatch); priced anyway, because the rate is per tier.")
    if mixed_models:
        parts.append("A tier ran more than one model (see mixed_model_tiers) and is priced entirely "
                     "at the last one's rates, so its total is wrong by the rate difference on the "
                     "earlier model's share.")
    block["note"] = " ".join(parts)
    return block


__all__ = ["price_map", "pricing_path", "estimate_cost", "RATE_CLASSES", "TOKENS_PER_UNIT"]
