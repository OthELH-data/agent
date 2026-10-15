"""Regression coverage for the Azure/OpenAI side of token accounting and pricing.

Three things are checked, and they are the three places the Azure path could go wrong quietly:

  1. The translation off a LangChain reply (`llm._langchain_usage`). It must emit OPENAI field names,
     because that is what tells `token_accounting` the counters are inclusive. Emitting an Anthropic
     name would roughly double every reported Azure prompt, and nothing downstream would complain.
  2. Recording (`llm.record_reply_usage` → `tokens.USAGE`). The cached part must be carved OUT of
     `prompt_tokens` into `cache_read_tokens`, not added on top, and `reasoning_tokens` must land in
     `thinking_tokens` as a subset of output.
  3. Pricing (`pricing.estimate_cost`). Rates must resolve by model id ahead of tier, cache reads and
     reported cache writes must bill independently, and a tier that ran two models must say so.

Everything here is offline: fake reply objects, a temporary price map, no network and no API key.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Same path insert as test_token_accounting: excel_extractor/ (the parent of tests/) on sys.path so
# `extractor` resolves however discovery was started.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import llm, pricing
from shared import token_accounting as accounting
from shared.token_schemas import TokenUsageReport
from shared.tokens import USAGE


class FakeReply:
    """The bits of a LangChain `AIMessage` that `_langchain_usage` reads. Nothing else is needed."""

    def __init__(self, usage_metadata=None, response_metadata=None, content="hello"):
        self.content = content
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata or {}


# A Chat Completions reply as LangChain normalises it: 13,000 prompt tokens of which 12,000 came out
# of the cache, and 700 of the 900 output tokens were spent reasoning.
AZURE_META = {
    "input_tokens": 13_000,
    "output_tokens": 900,
    "total_tokens": 13_900,
    "input_token_details": {"cache_read": 12_000},
    "output_token_details": {"reasoning": 700},
}

AZURE_META_WITH_WRITE = {
    "input_tokens": 13_000,
    "output_tokens": 900,
    "total_tokens": 13_900,
    "input_token_details": {"cache_read": 12_000, "cache_write": 500},
    "output_token_details": {"reasoning": 700},
}


class LangChainUsageTests(unittest.TestCase):
    """`_langchain_usage` — the one function that decides what the Azure numbers look like."""

    def test_usage_metadata_becomes_an_openai_block(self):
        usage = llm._langchain_usage(FakeReply(usage_metadata=AZURE_META))

        self.assertEqual(13_000, usage["prompt_tokens"])
        self.assertEqual(900, usage["completion_tokens"])
        self.assertEqual(12_000, usage["prompt_tokens_details"]["cached_tokens"])
        self.assertEqual(0, usage["prompt_tokens_details"]["cache_write_tokens"])
        self.assertEqual(700, usage["completion_tokens_details"]["reasoning_tokens"])
        # The names are the whole point: they are what make the block inclusive rather than exclusive.
        self.assertEqual("inclusive", accounting.detect_semantics(usage))
        self.assertNotIn("input_tokens", usage)

    def test_prompt_is_not_double_counted(self):
        """The cached part is already inside prompt_tokens, so the full size stays 13,000."""
        usage = llm._langchain_usage(FakeReply(usage_metadata=AZURE_META))

        self.assertEqual(13_000, accounting.reported_prompt_size(usage))
        new_in, read, creation = accounting.input_buckets(usage)
        self.assertEqual((1_000, 12_000, 0), (new_in, read, creation))
        # The invariant that ties the two readers together.
        self.assertEqual(accounting.reported_prompt_size(usage), new_in + read + creation)

    def test_missing_cache_write_counter_stays_zero(self):
        """A response that omits cache writes remains compatible with read-only usage."""
        usage = llm._langchain_usage(FakeReply(usage_metadata=AZURE_META))

        self.assertEqual(0, usage["prompt_tokens_details"]["cache_write_tokens"])
        self.assertEqual(0, accounting.input_buckets(usage)[2])

    def test_usage_metadata_preserves_a_reported_cache_write(self):
        usage = llm._langchain_usage(FakeReply(usage_metadata=AZURE_META_WITH_WRITE))

        self.assertEqual(500, usage["prompt_tokens_details"]["cache_write_tokens"])
        new_in, read, creation = accounting.input_buckets(usage)
        self.assertEqual((500, 12_000, 500), (new_in, read, creation))
        self.assertEqual(13_000, new_in + read + creation)

    def test_alternate_detail_key_spellings_are_read(self):
        """LangChain has used more than one name for these sub-counters across versions."""
        usage = llm._langchain_usage(FakeReply(usage_metadata={
            "input_tokens": 500, "output_tokens": 40,
            "input_token_details": {"cached_tokens": 300},
            "output_token_details": {"reasoning_tokens": 10},
        }))

        self.assertEqual(300, usage["prompt_tokens_details"]["cached_tokens"])
        self.assertEqual(10, usage["completion_tokens_details"]["reasoning_tokens"])

    def test_raw_chat_completions_block_is_the_fallback(self):
        """No usage_metadata, but a recognisable Chat Completions block: use it verbatim."""
        raw = {"prompt_tokens": 2_000, "completion_tokens": 50,
               "prompt_tokens_details": {"cached_tokens": 1_800}}
        usage = llm._langchain_usage(FakeReply(response_metadata={"token_usage": raw}))

        self.assertEqual(raw, usage)
        self.assertEqual(1_800, accounting.input_buckets(usage)[1])

    def test_responses_shaped_raw_block_is_accounted(self):
        """Responses usage supplies its cache details beneath input_tokens_details."""
        usage = llm._langchain_usage(FakeReply(response_metadata={"token_usage": {
            "input_tokens": 9_000, "output_tokens": 100,
            "input_tokens_details": {"cached_tokens": 8_000, "cache_write_tokens": 500},
        }}))

        self.assertEqual((500, 8_000, 500), accounting.input_buckets(usage))

    def test_a_reply_with_no_usage_reports_nothing(self):
        self.assertEqual({}, llm._langchain_usage(FakeReply()))
        self.assertEqual({}, llm._langchain_usage(FakeReply(usage_metadata={})))
        # All-zero counters are treated as "reported nothing" too, so the estimate takes over rather
        # than a run silently recording 0 tokens for every call.
        self.assertEqual({}, llm._langchain_usage(
            FakeReply(usage_metadata={"input_tokens": 0, "output_tokens": 0})))


class RecordAzureUsageTests(unittest.TestCase):
    """`record_reply_usage` on an Azure reply, through the real `USAGE` singleton."""

    def setUp(self):
        USAGE.reset()
        self.addCleanup(USAGE.reset)

    def record(self, reply_obj, tier="high", model="gpt-5.4-pro"):
        reply = llm.LLMReply(text=reply_obj.content, usage=llm._langchain_usage(reply_obj),
                            raw=reply_obj)
        llm.record_reply_usage(tier, model, reply, [])
        return USAGE.report()

    def test_azure_reply_is_recorded_as_provider_reported(self):
        report = self.record(FakeReply(usage_metadata=AZURE_META))
        tier = report["by_tier"]["high"]

        self.assertEqual("gateway", report["estimator"])
        self.assertEqual("gateway", tier["source"])
        self.assertEqual(1_000, tier["tokens_in"])          # new input only
        self.assertEqual(12_000, tier["cache_read_tokens"])
        self.assertEqual(0, tier["cache_creation_tokens"])  # response omitted a write counter
        self.assertEqual(13_000, tier["prompt_tokens"])
        self.assertEqual(900, tier["tokens_out"])
        self.assertEqual(700, tier["thinking_tokens"])      # reasoning, reported as the same subset
        self.assertEqual(13_900, tier["tokens_total"])

    def test_reported_cache_write_reaches_the_usage_metrics(self):
        report = self.record(FakeReply(usage_metadata=AZURE_META_WITH_WRITE), model="gpt-5.6-terra")
        tier = report["by_tier"]["high"]

        self.assertEqual(500, tier["tokens_in"])
        self.assertEqual(12_000, tier["cache_read_tokens"])
        self.assertEqual(500, tier["cache_creation_tokens"])
        self.assertEqual(13_000, tier["prompt_tokens"])
        self.assertEqual(500, report["total_cache_creation_tokens"])

    def test_totals_and_node_rollup_agree_with_the_tier(self):
        with USAGE.attribute(node="drafter", step=1, kind="draft"):
            report = self.record(FakeReply(usage_metadata=AZURE_META))
        tier = report["by_tier"]["high"]
        node = tier["by_node"]["drafter"]

        for field in ("tokens_in", "cache_read_tokens", "cache_creation_tokens", "prompt_tokens",
                      "tokens_out", "thinking_tokens", "tokens_total"):
            self.assertEqual(tier[field], node[field], field)
        self.assertEqual(1_000, report["total_tokens_in"])
        self.assertEqual(12_000, report["total_cache_read_tokens"])
        self.assertEqual(13_000, report["total_prompt_tokens"])
        self.assertEqual(700, report["total_thinking_tokens"])
        self.assertEqual(13_900, report["total_tokens"])

    def test_the_per_call_ledger_carries_the_same_call(self):
        mark = USAGE.mark()
        with USAGE.attribute(node="drafter", step=2, kind="draft"):
            self.record(FakeReply(usage_metadata=AZURE_META))
        calls = USAGE.calls_since(mark)

        self.assertEqual(1, len(calls))
        call = calls[0]
        self.assertEqual(("drafter", 2, "gateway"), (call.node, call.step, call.source))
        self.assertEqual((1_000, 12_000, 0), (call.tokens_in, call.cache_read_tokens,
                                              call.cache_creation_tokens))
        self.assertEqual((900, 700), (call.tokens_out, call.thinking_tokens))

    def test_a_reply_with_no_usage_falls_back_to_the_estimate(self):
        report = self.record(FakeReply(content="a short answer"))

        self.assertEqual("tiktoken", report["estimator"])
        self.assertEqual("local-estimate", report["by_tier"]["high"]["source"])
        self.assertEqual(0, report["by_tier"]["high"]["cache_read_tokens"])

    def test_models_seen_records_every_model_on_a_tier(self):
        """A tier that latched from Azure to ICA mid-run spans two price lists, and says so."""
        self.record(FakeReply(usage_metadata=AZURE_META), model="gpt-5.4-pro")
        report = self.record(FakeReply(usage_metadata=AZURE_META), model="claude-opus-5")

        self.assertEqual(["claude-opus-5", "gpt-5.4-pro"], report["by_tier"]["high"]["models_seen"])
        self.assertEqual("claude-opus-5", report["by_tier"]["high"]["model"])

    def test_the_report_still_validates_against_the_schema(self):
        self.record(FakeReply(usage_metadata=AZURE_META))
        validated = TokenUsageReport.model_validate(USAGE.report())

        self.assertEqual(12_000, validated.total_cache_read_tokens)
        self.assertEqual(["gpt-5.4-pro"], validated.by_tier["high"].models_seen)
        self.assertEqual(700, validated.by_tier["high"].thinking_tokens)


# A two-provider price map: Claude priced by tier (as every older file is), GPT priced by model id
# with no cache-write class. Round numbers so the expected dollars can be read off by hand.
PRICE_MAP = {
    "rates_source": "unit test",
    "as_of": "2026-08-14",
    "currency": "USD",
    "models": {
        "gpt-5.4-pro": {"cache_write_billed": False, "in": 10.0, "cache_read": 1.0, "out": 100.0},
        "gpt-5.6-terra": {"in": 2.0, "cache_write_5m": 2.5, "cache_write_1h": 2.5,
                           "cache_read": 0.2, "out": 12.0},
    },
    "tiers": {
        "high": {"model": "claude-opus-5", "in": 5.0, "cache_write_5m": 6.25,
                 "cache_write_1h": 10.0, "cache_read": 0.5, "out": 25.0},
        "med": {"model": "claude-sonnet-5", "in": 2.0, "cache_read": 0.2, "out": 10.0},
    },
}

# One tier's aggregates, in the shape `tokens.report()` emits: 1M new input, 1M cache reads, and 1M
# output, so each rate reads straight through as dollars.
AGG = {"tokens_in": 1_000_000, "cache_read_tokens": 1_000_000, "cache_creation_tokens": 0,
       "prompt_tokens": 2_000_000, "tokens_out": 1_000_000}


class ModelKeyedPricingTests(unittest.TestCase):
    """`pricing.estimate_cost` against a file that prices both providers."""

    def setUp(self):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(PRICE_MAP, tmp)
        tmp.close()
        self.path = Path(tmp.name)
        os.environ["MODEL_PRICING_FILE"] = str(self.path)
        os.environ["MODEL_CACHE_TTL"] = "1h"
        # `price_map` is lru_cached, so a test that swaps the file has to drop the cache.
        # `cache_ttl` deliberately is not cached — it re-reads the env var on every call.
        pricing.price_map.cache_clear()

        def restore():
            os.environ.pop("MODEL_PRICING_FILE", None)
            os.environ.pop("MODEL_CACHE_TTL", None)
            pricing.price_map.cache_clear()
            self.path.unlink(missing_ok=True)

        self.addCleanup(restore)

    def test_a_gpt_tier_prices_off_its_model_entry(self):
        cost = pricing.estimate_cost({"high": {"model": "gpt-5.4-pro", "models_seen": ["gpt-5.4-pro"],
                                               **AGG}})
        tier = cost["by_tier"]["high"]

        self.assertTrue(cost["priced"])
        self.assertEqual(("model", "gpt-5.4-pro"), (tier["rates_basis"], tier["rates_key"]))
        self.assertEqual(10.0, tier["input"])         # GPT rates, not the tier's Claude rates
        self.assertEqual(1.0, tier["cache_read"])
        self.assertEqual(100.0, tier["output"])
        self.assertEqual(111.0, tier["total"])
        # The tier entry for `high` would have priced this at 5 + 0.5 + 25; the model entry won.
        self.assertEqual([], cost["model_mismatch"])

    def test_no_cache_write_class_prices_as_an_explicit_zero(self):
        """`cache_write_billed: false` clears the all-four-rates gate without inventing a rate."""
        cost = pricing.estimate_cost({"high": {"model": "gpt-5.4-pro", **AGG}})
        tier = cost["by_tier"]["high"]

        self.assertEqual([], cost["unpriced_tiers"])
        self.assertFalse(tier["cache_write_billed"])
        self.assertEqual(0.0, tier["cache_write"])
        # And the note says the $0 is a declaration, so it cannot read as a rate nobody filled in.
        self.assertIn("by declaration", cost["note"])

    def test_a_gpt_56_cache_write_is_priced_when_the_gateway_reports_one(self):
        agg = dict(AGG, cache_creation_tokens=1_000_000, prompt_tokens=3_000_000)
        cost = pricing.estimate_cost({"med": {"model": "gpt-5.6-terra", **agg}})
        tier = cost["by_tier"]["med"]

        self.assertEqual(("model", "gpt-5.6-terra"), (tier["rates_basis"], tier["rates_key"]))
        self.assertEqual(2.0, tier["input"])
        self.assertEqual(0.2, tier["cache_read"])
        self.assertEqual(2.5, tier["cache_write"])
        self.assertEqual(12.0, tier["output"])
        self.assertEqual(16.7, tier["total"])

    def test_a_claude_tier_still_prices_off_the_tier_section(self):
        cost = pricing.estimate_cost({"high": {"model": "claude-opus-5", **AGG}})
        tier = cost["by_tier"]["high"]

        self.assertEqual(("tier", "high"), (tier["rates_basis"], tier["rates_key"]))
        self.assertTrue(tier["cache_write_billed"])
        self.assertEqual(5.0, tier["input"])
        self.assertEqual(0.5, tier["cache_read"])
        self.assertEqual(25.0, tier["output"])

    def test_a_tier_fallback_for_the_wrong_model_is_flagged(self):
        """No `models` entry for this id, so the tier answered — and the mismatch is named."""
        cost = pricing.estimate_cost({"high": {"model": "gpt-5.4-mini", **AGG}})

        self.assertEqual([{"tier": "high", "recorded": "gpt-5.4-mini",
                           "priced_for": "claude-opus-5"}], cost["model_mismatch"])
        self.assertEqual("tier", cost["by_tier"]["high"]["rates_basis"])

    def test_a_partial_rate_set_is_excluded_rather_than_half_priced(self):
        """`med` has no write columns and does not declare writes unbilled, so it stays unpriced."""
        cost = pricing.estimate_cost({"med": {"model": "claude-sonnet-5", **AGG}})

        self.assertEqual(["med"], cost["unpriced_tiers"])
        self.assertFalse(cost["priced"])

    def test_a_tier_that_ran_two_models_is_flagged(self):
        cost = pricing.estimate_cost({"high": {
            "model": "gpt-5.4-pro", "models_seen": ["claude-opus-5", "gpt-5.4-pro"], **AGG}})

        self.assertEqual([{"tier": "high", "models": ["claude-opus-5", "gpt-5.4-pro"],
                           "priced_as": "gpt-5.4-pro"}], cost["mixed_model_tiers"])
        self.assertIn("mixed_model_tiers", cost["note"])
        # Still priced — a wrong-by-a-known-amount total that says so beats no total at all.
        self.assertTrue(cost["priced"])

    def test_the_ttl_decides_which_write_column_a_claude_tier_pays(self):
        agg = dict(AGG, cache_creation_tokens=1_000_000, prompt_tokens=3_000_000)
        cost_1h = pricing.estimate_cost({"high": {"model": "claude-opus-5", **agg}})

        os.environ["MODEL_CACHE_TTL"] = "5m"
        cost_5m = pricing.estimate_cost({"high": {"model": "claude-opus-5", **agg}})

        self.assertEqual(10.0, cost_1h["by_tier"]["high"]["cache_write"])
        self.assertEqual(6.25, cost_5m["by_tier"]["high"]["cache_write"])


if __name__ == "__main__":
    unittest.main()
