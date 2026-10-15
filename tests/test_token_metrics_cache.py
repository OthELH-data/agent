from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from shared.token_metrics_cache import (
    OutputScope,
    cache_report_metrics,
    list_metric_records,
    mark_metric_write,
)


def report(cost: float = 1.0) -> dict:
    return {
        "case_id": "GreatWhite_2024_0026",
        "year": "2024",
        "asset": "Great White",
        "exception_code": "0026",
        "exception_category": "Labor",
        "rebuttal_cycle": 0,
        "token_usage": {
            "by_tier": {"med": {"calls": 2}},
            "total_tokens_in": 100,
            "total_cache_read_tokens": 50,
            "total_cache_creation_tokens": 10,
            "total_prompt_tokens": 160,
            "total_tokens_out": 40,
            "total_tokens": 200,
            "cost": {
                "priced": True,
                "total_usd": cost,
                "uncached_equivalent_usd": cost + 0.25,
                "cache_savings_usd": 0.25,
                "by_tier": {
                    "med": {
                        "input": 0.2,
                        "cache_read": 0.01,
                        "cache_write": 0.04,
                        "output": 0.75,
                    }
                },
            },
        },
    }


class TokenMetricCacheTests(unittest.TestCase):
    def test_same_exception_is_isolated_by_run_and_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "metrics.sqlite3"
            folder_a = OutputScope("extractor", "sharepoint", "Documents/Run 1", "https://site")
            folder_b = OutputScope("extractor", "sharepoint", "Documents/Run 2", "https://site")
            kwargs = dict(
                pipeline="extractor", data=report(), year="2024", asset="Great White",
                exception_id="0026", rebuttal_cycle=0,
                relative_path="2024/Great White/0026/rebuttal_0_synthesis.json",
                artifact_kind="synthesis", path=db,
            )
            cache_report_metrics(**kwargs, scope=folder_a, run_id="run-a")
            cache_report_metrics(**kwargs, scope=folder_a, run_id="run-b")
            cache_report_metrics(**kwargs, scope=folder_b, run_id="run-a")

            rows = list_metric_records(path=db)
            self.assertEqual(len(rows), 3)
            self.assertEqual({row["run_id"] for row in rows}, {"run-a", "run-b"})
            self.assertEqual({row["folder_key"] for row in rows}, {
                folder_a.folder_key, folder_b.folder_key,
            })

    def test_same_run_artifact_upserts_and_tracks_output_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "metrics.sqlite3"
            scope = OutputScope("inference", "local", str(Path(tmp) / "output"))
            kwargs = dict(
                pipeline="inference", year="2024", asset="Great White",
                exception_id="0026", rebuttal_cycle=0,
                relative_path="2024/Great White/0026/rebuttal_0_otl.json",
                artifact_kind="otl", scope=scope, run_id="run-1", path=db,
            )
            identity = cache_report_metrics(data=report(1.0), **kwargs)
            cache_report_metrics(data=report(2.0), **kwargs)
            rows = list_metric_records(path=db)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["total_cost_usd"], 2.0)
            self.assertEqual(rows[0]["write_status"], "cached")

            mark_metric_write(identity, location="/output/report.json", path=db)
            row = list_metric_records(path=db)[0]
            self.assertEqual(row["write_status"], "written")
            self.assertEqual(row["output_location"], "/output/report.json")


if __name__ == "__main__":
    unittest.main()
