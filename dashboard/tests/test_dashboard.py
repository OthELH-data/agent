from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from dashboard.app import (
    DashboardData,
    LocalOutputSource,
    display_exception_id,
    extract_metrics,
    ref_from_path,
)


def sample_report(*, cost: float = 1.25, savings: float = 0.5) -> dict:
    return {
        "year": "2024",
        "asset": "Great White",
        "exception_code": "0026",
        "rebuttal_cycle": 0,
        "token_usage": {
            "by_tier": {
                "med": {
                    "model": "gpt-test",
                    "calls": 3,
                    "tokens_in": 100,
                    "cache_read_tokens": 50,
                    "cache_creation_tokens": 10,
                    "prompt_tokens": 160,
                    "tokens_out": 40,
                    "thinking_tokens": 5,
                    "tokens_total": 200,
                }
            },
            "total_tokens_in": 100,
            "total_cache_read_tokens": 50,
            "total_cache_creation_tokens": 10,
            "total_prompt_tokens": 160,
            "total_tokens_out": 40,
            "total_thinking_tokens": 5,
            "total_tokens": 200,
            "cost": {
                "priced": True,
                "total_usd": cost,
                "uncached_equivalent_usd": cost + savings,
                "cache_savings_usd": savings,
                "by_tier": {
                    "med": {
                        "input": 0.2,
                        "cache_read": 0.01,
                        "cache_write": 0.04,
                        "output": 1.0,
                        "total": cost,
                        "cache_savings": savings,
                    }
                },
            },
        },
    }


class DashboardTests(unittest.TestCase):
    def test_exception_id_normalization(self) -> None:
        self.assertEqual(display_exception_id("3.0"), "0003")
        self.assertEqual(display_exception_id(26), "0026")
        self.assertEqual(display_exception_id("AB-12"), "0012")

    def test_extracts_all_token_and_cost_classes(self) -> None:
        metrics = extract_metrics(sample_report())
        self.assertEqual(metrics.calls, 3)
        self.assertEqual(metrics.input_tokens, 100)
        self.assertEqual(metrics.cache_read_tokens, 50)
        self.assertEqual(metrics.cache_write_tokens, 10)
        self.assertEqual(metrics.output_tokens, 40)
        self.assertEqual(metrics.thinking_tokens, 5)
        self.assertEqual(metrics.total_tokens, 200)
        self.assertAlmostEqual(metrics.total_cost_usd, 1.25)
        self.assertAlmostEqual(metrics.cache_savings_usd, 0.5)
        self.assertAlmostEqual(metrics.output_cost_usd, 1.0)

    def test_ref_uses_last_four_path_segments(self) -> None:
        ref = ref_from_path(
            "extractor",
            "archived/run-a/2024/Great White/26/rebuttal_2_synthesis.json",
            "/source",
        )
        self.assertIsNotNone(ref)
        assert ref is not None
        self.assertEqual(ref.year, "2024")
        self.assertEqual(ref.asset, "Great White")
        self.assertEqual(ref.exception_id, "0026")
        self.assertEqual(ref.cycle, 2)

    def test_local_index_joins_csv_and_aggregates_rebuttals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extractor = root / "extractor"
            inference = root / "inference"
            csv_path = root / "cases.csv"
            folder = extractor / "2024" / "GreatWhite" / "0026"
            folder.mkdir(parents=True)
            inference.mkdir()
            (folder / "rebuttal_0_synthesis.json").write_text(
                json.dumps(sample_report(cost=1.25, savings=0.5)), encoding="utf-8"
            )
            second = sample_report(cost=2.0, savings=0.75)
            second["rebuttal_cycle"] = 1
            (folder / "rebuttal_1_synthesis.json").write_text(
                json.dumps(second), encoding="utf-8"
            )
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["year", "site", "exception_ID", "type", "gross_credit_requested"],
                )
                writer.writeheader()
                writer.writerow({
                    "year": "2024",
                    "site": "Great White",
                    "exception_ID": "26.0",
                    "type": "Labor",
                    "gross_credit_requested": "1000",
                })

            data = DashboardData(
                {
                    "extractor": LocalOutputSource("extractor", str(extractor)),
                    "inference": LocalOutputSource("inference", str(inference)),
                },
                csv_path=csv_path,
                cache_db_path=root / "metrics.sqlite3",
            )
            # Explicit legacy backfill is the only operation that touches the output folder.
            data.reindex_legacy_outputs()
            summary = data.refresh()  # cache-only
            row = summary["sections"]["extractor"]["cases"][0]
            self.assertEqual(row["category"], "Labor")
            self.assertEqual(row["cycles"], [0, 1])
            self.assertEqual(row["metrics"]["artifacts"], 2)
            self.assertAlmostEqual(row["metrics"]["total_cost_usd"], 3.25)
            self.assertAlmostEqual(row["metrics"]["cache_savings_usd"], 1.25)

            details = data.case_details(
                "extractor", row["run_key"], "2024", "Great White", "0026"
            )
            self.assertEqual([item["cycle"] for item in details["artifacts"]], [0, 1])
            self.assertEqual(details["artifacts"][0]["usage_by_tier"]["med"]["calls"], 3)


if __name__ == "__main__":
    unittest.main()
