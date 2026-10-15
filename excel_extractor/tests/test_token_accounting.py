"""Regression coverage for gateway-reported token accounting semantics."""

from __future__ import annotations

import logging
import math
import os
import sys
import unittest

# Put excel_extractor/ (the parent of tests/) on sys.path so `extractor` resolves however
# discovery was started — `python3 -m unittest discover` from excel_extractor/ or -s tests -t .
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import token_accounting as accounting


BASE_URL = "https://gateway.example/v1"
MODEL = "claude-sonnet-4-6"


class TokenAccountingTests(unittest.TestCase):
    """Exercise the public API using the Probe A--F usage shapes."""

    def capture(self, usage: dict, history_len: int = 0, *, model: str = MODEL,
                base_url: str = BASE_URL) -> accounting.UsageBaseline:
        return accounting.capture_usage({"usage": usage}, history_len, model, base_url)

    def effective(self, usage: dict, messages: list[dict] | None = None, *,
                  history_len: int = 0, model: str = MODEL, base_url: str = BASE_URL):
        return accounting.effective_input_tokens(
            messages or [], self.capture(usage, history_len, model=model, base_url=base_url),
            model, base_url,
        )

    def test_inclusive_usage_does_not_sum_cache_buckets(self):
        """Inclusive counters retain prompt_tokens rather than double-counting cached input.

        The inclusive dialect is OpenAI's, so this uses OpenAI field names. An Anthropic-named
        block of the same shape is exclusive no matter how the numbers compare — see
        test_anthropic_names_are_exclusive_whatever_the_magnitudes.
        """
        usage = {"prompt_tokens": 13_000, "prompt_tokens_details": {"cached_tokens": 12_000}}
        result = self.effective(usage)

        self.assertEqual("inclusive", accounting.detect_semantics(usage))
        self.assertEqual(13_000, accounting.reported_prompt_size(usage))
        self.assertEqual("baseline", result.source)
        self.assertEqual(13_650, result.tokens)
        naive_sum = usage["prompt_tokens"] + usage["prompt_tokens_details"]["cached_tokens"]
        self.assertGreater(naive_sum / accounting.reported_prompt_size(usage), 1.9)

    def test_anthropic_names_are_exclusive_whatever_the_magnitudes(self):
        """Dialect decides, not size: Anthropic /v1/messages never double-counts.

        The older rule ("input_tokens >= read + creation means inclusive") got this wrong, and
        under-reported the prompt by the whole cached part. It only became reachable once prompt
        caching was switched on, because before that read + creation was always 0.
        """
        usage = {"input_tokens": 13_000, "cache_read_input_tokens": 12_000,
                 "cache_creation_input_tokens": 0}

        self.assertEqual("exclusive", accounting.detect_semantics(usage))
        self.assertEqual(25_000, accounting.reported_prompt_size(usage))
        # A small read behind a large uncached tail is the shape that used to be misread.
        small_read = {"input_tokens": 5_000, "cache_read_input_tokens": 200,
                      "cache_creation_input_tokens": 100}
        self.assertEqual("exclusive", accounting.detect_semantics(small_read))
        self.assertEqual(5_300, accounting.reported_prompt_size(small_read))
        self.assertEqual((5_000, 200, 100), accounting.input_buckets(small_read))

    def test_exclusive_usage_includes_cache_buckets(self):
        """Measured exclusive cache hits include read/creation counters in prompt size."""
        usage = {"input_tokens": 12, "cache_read_input_tokens": 12_943,
                 "cache_creation_input_tokens": 0}
        result = self.effective(usage)

        self.assertEqual("exclusive", accounting.detect_semantics(usage))
        self.assertEqual(12_955, accounting.reported_prompt_size(usage))
        self.assertEqual("baseline", result.source)
        self.assertEqual(13_603, result.tokens)
        self.assertGreater(accounting.reported_prompt_size(usage) / usage["input_tokens"], 1_000)

    def test_no_cache_has_coincident_inclusive_and_exclusive_formulas(self):
        """With zero cache buckets, both semantic formulas reduce to input_tokens."""
        usage = {"input_tokens": 200, "cache_read_input_tokens": 0,
                 "cache_creation_input_tokens": 0}
        result = self.effective(usage)

        self.assertEqual("no-cache", accounting.detect_semantics(usage))
        self.assertEqual(200, accounting.reported_prompt_size(usage))
        self.assertEqual(usage["input_tokens"], usage["input_tokens"] + 0 + 0)
        self.assertEqual("baseline", result.source)
        self.assertEqual(210, result.tokens)

    def test_all_zero_usage_falls_back_to_local_estimate(self):
        """All-zero reported counters are unreported usage, not a trustworthy baseline."""
        usage = {"input_tokens": 0, "cache_read_input_tokens": 0,
                 "cache_creation_input_tokens": 0}
        messages = [{"role": "user", "content": "hello"}]
        result = self.effective(usage, messages, history_len=1)

        self.assertEqual("unreported", accounting.detect_semantics(usage))
        self.assertEqual("local-estimate", result.source)
        self.assertEqual(8, result.tokens)
        self.assertRegex(result.reason, r"unreported|zero")

    def test_identity_change_falls_back_to_local_estimate(self):
        """A baseline captured for another endpoint/model identity is never reused."""
        messages = [{"role": "user", "content": "hello"}]
        baseline = self.capture({"input_tokens": 100}, history_len=1)
        result = accounting.effective_input_tokens(messages, baseline, MODEL, "https://other.gateway/v1")

        self.assertEqual("local-estimate", result.source)
        self.assertEqual(8, result.tokens)
        self.assertIn("identity", result.reason)

    def test_history_shorter_than_cursor_falls_back_to_local_estimate(self):
        """A rewritten/truncated history cannot safely apply the stored baseline cursor."""
        messages = [{"role": "user", "content": "hello"}]
        result = self.effective({"input_tokens": 100}, messages, history_len=2)

        self.assertEqual("local-estimate", result.source)
        self.assertEqual(8, result.tokens)
        self.assertIn("rewritten or truncated", result.reason)

    def test_exclusive_undercount_is_now_eliminated_not_merely_delayed(self):
        """The shape the magnitude heuristic used to undercount is now counted in full.

        This used to classify as inclusive and report 2,000 for a prompt that really was 3,200 —
        safe only because the next call re-anchored the baseline. Deciding by dialect removes the
        undercount instead of deferring it, which is the direction that matters: the context guard
        must never think the window is emptier than it is.
        """
        usage = {"input_tokens": 2_000, "cache_read_input_tokens": 1_200,
                 "cache_creation_input_tokens": 0}
        result = self.effective(usage)
        true_exclusive_total = usage["input_tokens"] + usage["cache_read_input_tokens"]

        self.assertEqual("exclusive", accounting.detect_semantics(usage))
        self.assertEqual(true_exclusive_total, accounting.reported_prompt_size(usage))
        self.assertEqual("baseline", result.source)
        self.assertEqual(3_360, result.tokens)
        self.assertGreaterEqual(accounting.reported_prompt_size(usage), usage["input_tokens"])

    def test_nested_cache_creation_tokens_are_counted(self):
        """TTL-bucket cache_creation sub-objects work when the flat counter is absent."""
        usage = {"input_tokens": 12, "cache_read_input_tokens": 0,
                 "cache_creation": {"ephemeral_5m_input_tokens": 300,
                                    "ephemeral_1h_input_tokens": 40}}
        result = self.effective(usage)

        self.assertEqual("exclusive", accounting.detect_semantics(usage))
        self.assertEqual(352, accounting.reported_prompt_size(usage))
        self.assertEqual("baseline", result.source)
        self.assertEqual(370, result.tokens)

    def test_openai_style_usage_names_are_parsed(self):
        """OpenAI prompt_tokens and nested cached_tokens remain compatible fallback inputs."""
        usage = {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 90}}
        result = self.effective(usage)

        self.assertEqual("inclusive", accounting.detect_semantics(usage))
        self.assertEqual(100, accounting.reported_prompt_size(usage))
        self.assertEqual("baseline", result.source)
        self.assertEqual(105, result.tokens)

    def test_input_buckets_are_disjoint_and_sum_to_the_reported_prompt(self):
        """The three billing classes always add back up to the full prompt, in every dialect.

        `reported_prompt_size` answers "how much of the context window is used" and `input_buckets`
        answers "what was each part billed at". They must describe the same prompt, or the cost
        estimate and the context guard would disagree about the same call.
        """
        shapes = [
            {"input_tokens": 500, "cache_read_input_tokens": 15_000,
             "cache_creation_input_tokens": 800},                      # warm exclusive
            {"input_tokens": 5_000, "cache_read_input_tokens": 200,
             "cache_creation_input_tokens": 100},                      # small read, large tail
            {"input_tokens": 2, "cache_creation": {"ephemeral_5m_input_tokens": 8_813,
                                                   "ephemeral_1h_input_tokens": 0}},   # nested TTL
            {"input_tokens": 200, "cache_read_input_tokens": 0,
             "cache_creation_input_tokens": 0},                        # no cache
            {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 90}},     # inclusive
            {},                                                        # unreported
        ]
        for usage in shapes:
            with self.subTest(usage=usage):
                buckets = accounting.input_buckets(usage)
                self.assertEqual(3, len(buckets))
                self.assertTrue(all(b >= 0 for b in buckets))
                self.assertEqual(accounting.reported_prompt_size(usage), sum(buckets))

    def test_input_buckets_report_only_new_tokens_as_new(self):
        """A warm call's `tokens_in` is the uncached remainder, not the whole prompt."""
        usage = {"input_tokens": 1, "cache_read_input_tokens": 9_820,
                 "cache_creation_input_tokens": 739}

        self.assertEqual((1, 9_820, 739), accounting.input_buckets(usage))
        self.assertEqual(10_560, accounting.reported_prompt_size(usage))
        # An inclusive block reports the whole prompt in one number, so the new part is what is
        # left after the cached part is taken out.
        inclusive = {"prompt_tokens": 10_560, "prompt_tokens_details": {"cached_tokens": 9_820}}
        self.assertEqual((740, 9_820, 0), accounting.input_buckets(inclusive))

    def test_output_counters_separate_thinking_without_double_counting(self):
        """Thinking tokens are a SUBSET of billed output, and a reported 0 is not a missing key."""
        usage = {"output_tokens": 1_246, "output_tokens_details": {"thinking_tokens": 1_005}}
        output, thinking = accounting.output_counters(usage)
        self.assertEqual(1_246, output)
        self.assertEqual(1_005, thinking)
        self.assertLessEqual(thinking, output)

        # OpenAI dialect names the same thing differently.
        self.assertEqual((900, 640), accounting.output_counters(
            {"completion_tokens": 900, "completion_tokens_details": {"reasoning_tokens": 640}}))
        # No thinking reported is 0 thinking, not "unknown".
        self.assertEqual((72, 0), accounting.output_counters({"output_tokens": 72}))
        # A genuinely zero output stays 0 rather than being read as absent.
        self.assertEqual((0, 0), accounting.output_counters({"output_tokens": 0}))
        self.assertEqual((0, 0), accounting.output_counters({}))

    def test_image_base64_is_block_aware_not_character_tokenized(self):
        """A ~1 MB base64 image avoids the historical ~250K phantom-token estimate."""
        b64 = "a" * 1_000_000
        messages = [{"role": "user", "content": [
            {"type": "image", "source": {"media_type": "image/png", "data": b64}},
        ]}]
        estimate = accounting.estimate_message_tokens(messages)

        self.assertLess(estimate, accounting.PER_IMAGE_TOKENS * 2)
        self.assertLess(estimate, len(b64) // 100)
        self.assertGreaterEqual(estimate, accounting.PER_IMAGE_TOKENS)

    def test_huge_repetitive_string_short_circuits_to_character_heuristic(self):
        """Strings over the cutoff estimate safely at roughly chars/CHARS_PER_TOKEN."""
        text = "x" * (accounting.HUGE_STRING_CHAR_CUTOFF + 1)
        estimate = accounting.estimate_message_tokens([{"role": "user", "content": text}])
        expected = accounting.PER_MESSAGE_OVERHEAD_TOKENS + math.ceil(
            len(text) / accounting.CHARS_PER_TOKEN
        )

        self.assertEqual(expected, estimate)
        self.assertAlmostEqual(estimate, len(text) / accounting.CHARS_PER_TOKEN,
                               delta=accounting.PER_MESSAGE_OVERHEAD_TOKENS + 1)

    def test_local_estimate_margin_exceeds_baseline_margin(self):
        """Fallback estimates carry a deliberately larger safety margin than reported usage."""
        messages = [{"role": "user", "content": "hello"}]
        baseline = self.capture({"input_tokens": 100}, history_len=1)
        baseline_result = accounting.effective_input_tokens(messages, baseline, MODEL, BASE_URL)
        local_result = accounting.effective_input_tokens(messages, baseline, MODEL, "https://other.gateway/v1")

        self.assertEqual("baseline", baseline_result.source)
        self.assertEqual(105, baseline_result.tokens)
        self.assertEqual("local-estimate", local_result.source)
        self.assertEqual(8, local_result.tokens)
        self.assertGreater(local_result.margin_applied, baseline_result.margin_applied)

    def test_check_threshold_logs_usage_and_separates_warn_from_compact(self):
        """Compaction starts at 80%; warning fires below it and every check logs source/raw usage."""
        compact_usage = {"input_tokens": 761_905}
        warn_usage = {"input_tokens": 571_429}
        # Logger name follows the module's package path ("shared.token_accounting"), so read it
        # off the module rather than hardcoding it.
        with self.assertLogs(accounting.logger.name, level="INFO") as captured:
            compact = accounting.check_threshold([], self.capture(compact_usage), MODEL, BASE_URL)
            warn = accounting.check_threshold([], self.capture(warn_usage), MODEL, BASE_URL)

        self.assertEqual("baseline", compact.estimate.source)
        self.assertEqual(800_001, compact.estimate.tokens)
        self.assertTrue(compact.should_compact)
        self.assertTrue(compact.should_warn)
        self.assertEqual("baseline", warn.estimate.source)
        self.assertEqual(600_001, warn.estimate.tokens)
        self.assertFalse(warn.should_compact)
        self.assertTrue(warn.should_warn)
        log_text = "\n".join(captured.output)
        self.assertIn("source=baseline", log_text)
        self.assertIn(repr(compact_usage), log_text)

    def test_context_overflow_error_matches_litellm_and_first_party_only(self):
        """Flatten double-encoded LiteLLM errors, accept first-party wording, reject unrelated 403s."""
        class GatewayError(Exception):
            def __init__(self, message: str, status_code: int, body: dict):
                super().__init__(message)
                self.status_code = status_code
                self.body = body
                self.message = body["error"]["message"]

        litellm = GatewayError(
            "gateway rejected request", 400,
            {"error": {"type": None, "message":
                '{"message":"The maximum tokens you requested exceeds the model limit of 128000"}. '
                "Received Model Group=claude-sonnet-4-6\nAvailable Model Group Fallbacks=None"}},
        )
        first_party = GatewayError("prompt is too long", 400,
                                   {"error": {"type": "invalid_request_error", "message": "prompt is too long"}})
        denied = GatewayError("denied", 403,
                              {"error": {"type": "team_model_access_denied", "message": "model access denied"}})

        self.assertTrue(accounting.is_context_overflow_error(litellm))
        self.assertTrue(accounting.is_context_overflow_error(first_party))
        self.assertFalse(accounting.is_context_overflow_error(denied))


class CacheTTLTests(unittest.TestCase):
    """The cache lifetime is a billing choice, so it has to reach the price map, not just the wire."""

    def setUp(self) -> None:
        self.previous = os.environ.get("MODEL_CACHE_TTL")
        accounting._ttl_warned.clear()

    def tearDown(self) -> None:
        if self.previous is None:
            os.environ.pop("MODEL_CACHE_TTL", None)
        else:
            os.environ["MODEL_CACHE_TTL"] = self.previous
        accounting._ttl_warned.clear()

    def set_ttl(self, value: str | None) -> None:
        if value is None:
            os.environ.pop("MODEL_CACHE_TTL", None)
        else:
            os.environ["MODEL_CACHE_TTL"] = value

    def test_default_is_one_hour_and_only_5m_or_1h_are_accepted(self):
        self.set_ttl(None)
        self.assertEqual(accounting.cache_ttl(), "1h")
        self.set_ttl(" 5M ")                       # trimmed and lowercased
        self.assertEqual(accounting.cache_ttl(), "5m")
        # A typo must not silently pick a different lifetime — it warns and uses the default.
        self.set_ttl("90m")
        with self.assertLogs(accounting.logger, level="WARNING") as logs:
            self.assertEqual(accounting.cache_ttl(), "1h")
        self.assertIn("90m", logs.output[0])

    def test_one_hour_sends_an_explicit_ttl_and_five_minutes_stays_a_bare_block(self):
        from shared import llm

        self.set_ttl(None)
        self.assertEqual(llm._cache_control(), {"type": "ephemeral", "ttl": "1h"})
        # The 5m branch must stay byte-identical to the shape the gateway was verified against.
        self.set_ttl("5m")
        self.assertEqual(llm._cache_control(), {"type": "ephemeral"})

    def test_the_write_rate_follows_the_configured_lifetime(self):
        from shared import pricing

        # 1M cache-write tokens only, so the reported cost IS the write rate.
        aggregates = {"high": {"model": "claude-opus-5", "tokens_in": 0, "tokens_out": 0,
                               "cache_creation_tokens": 1_000_000, "cache_read_tokens": 0,
                               "prompt_tokens": 1_000_000}}
        rates = pricing.price_map().get("tiers", {}).get("high") or {}
        if any(rates.get(key) is None for key in ("in", "cache_read", "out")) or not all(
                rates.get(pricing.WRITE_RATE_KEYS[ttl]) for ttl in ("5m", "1h")):
            self.skipTest("no local price map with both cache_write columns")
        for ttl in ("5m", "1h"):
            with self.subTest(ttl=ttl):
                self.set_ttl(ttl)
                block = pricing.estimate_cost(aggregates)
                self.assertEqual(block["cache_ttl"], ttl)
                self.assertEqual(block["by_tier"]["high"]["cache_write"],
                                 rates[pricing.WRITE_RATE_KEYS[ttl]])
        # And the 1-hour write is the more expensive of the two, per Anthropic's 2x vs 1.25x.
        self.assertGreater(rates[pricing.WRITE_RATE_KEYS["1h"]],
                           rates[pricing.WRITE_RATE_KEYS["5m"]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
