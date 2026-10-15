"""Persistent, run-scoped token metric index shared by both pipelines and the dashboard.

The JSON reports remain the source artifacts, but they are deliberately poor dashboard indexes:
they are large, remote when SharePoint is active, and several runs may contain the same exception.
This module maintains a small local SQLite database whose identity includes the pipeline, configured
output folder, run id, and relative artifact path.  A repeated exception therefore never overwrites
or merges with the same exception from another run.

Pipeline writers call :func:`cache_report_metrics` *before* persisting the JSON and then call
:func:`mark_metric_write` with the outcome.  SQLite WAL mode and a busy timeout make writes safe
across the batch runners' concurrent subprocesses.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_PATH = REPO_ROOT / ".cache" / "token_metrics.sqlite3"
RUN_ID_ENV = "JV_AUDIT_RUN_ID"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cache_path(path: str | Path | None = None) -> Path:
    configured = path or os.getenv("TOKEN_METRICS_CACHE_PATH") or DEFAULT_CACHE_PATH
    return Path(configured).expanduser().resolve()


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    return cleaned[:100] or "run"


def new_run_id(pipeline: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{_safe_component(pipeline)}-{stamp}-{secrets.token_hex(4)}"


def ensure_run_id(pipeline: str, requested: str | None = None) -> str:
    """Return one run id for the current process tree and export it for child processes."""
    value = (requested or os.getenv(RUN_ID_ENV) or "").strip()
    if not value:
        value = new_run_id(pipeline)
    value = _safe_component(value)
    os.environ[RUN_ID_ENV] = value
    return value


@dataclass(frozen=True)
class OutputScope:
    pipeline: str
    target: str
    folder_path: str
    site_url: str = ""

    @property
    def canonical(self) -> str:
        return "|".join((self.pipeline, self.target, self.site_url.rstrip("/"), self.folder_path.strip("/")))

    @property
    def folder_key(self) -> str:
        return hashlib.sha256(self.canonical.encode("utf-8")).hexdigest()[:20]


def configured_output_scope(pipeline: str, *, target: str | None = None,
                            folder_path: str | None = None,
                            site_url: str | None = None) -> OutputScope:
    pipeline = pipeline.strip().lower()
    resolved_target = (target or os.getenv("OUTPUT_TARGET") or "local").strip().lower()
    if folder_path is None:
        if resolved_target == "sharepoint":
            if pipeline == "extractor":
                folder_path = (
                    os.getenv("SHAREPOINT_FOLDER")
                    or os.getenv("SHAREPOINT_READ_FOLDER")
                    or ""
                )
            else:
                folder_path = os.getenv("SHAREPOINT_WRITE_FOLDER") or ""
        else:
            # This is the actual write root used by LocalSink for both pipelines.
            folder_path = os.getenv("OUTPUT_BASE") or ""
    if resolved_target == "local" and folder_path:
        folder_path = str(Path(folder_path).expanduser().resolve())
    return OutputScope(
        pipeline=pipeline,
        target=resolved_target,
        folder_path=str(folder_path or ""),
        site_url=str(site_url if site_url is not None else os.getenv("SHAREPOINT_URL") or ""),
    )


def source_url(scope: OutputScope, relative_path: str, record_id: int | None = None) -> str:
    if scope.target == "sharepoint" and scope.site_url:
        return scope.site_url.rstrip("/") + "/" + quote(
            scope.folder_path.strip("/") + "/" + relative_path.strip("/"), safe="/"
        )
    if record_id is not None:
        return f"/api/cached-artifact?record_id={record_id}"
    return ""


def metrics_from_report(data: dict[str, Any]) -> dict[str, Any]:
    usage = data.get("token_usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    cost = usage.get("cost") or {}
    if not isinstance(cost, dict):
        cost = {}
    by_tier_usage = usage.get("by_tier") or {}
    by_tier_cost = cost.get("by_tier") or {}
    if not isinstance(by_tier_usage, dict):
        by_tier_usage = {}
    if not isinstance(by_tier_cost, dict):
        by_tier_cost = {}

    def as_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def as_float(value: Any) -> float:
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0

    metrics: dict[str, Any] = {
        "artifacts": 1,
        "priced_artifacts": 1 if cost.get("priced") else 0,
        "calls": sum(as_int(item.get("calls")) for item in by_tier_usage.values()
                     if isinstance(item, dict)),
        "input_tokens": as_int(usage.get("total_tokens_in")),
        "cache_read_tokens": as_int(usage.get("total_cache_read_tokens")),
        "cache_write_tokens": as_int(usage.get("total_cache_creation_tokens")),
        "prompt_tokens": as_int(usage.get("total_prompt_tokens")),
        "output_tokens": as_int(usage.get("total_tokens_out")),
        "thinking_tokens": as_int(usage.get("total_thinking_tokens")),
        "total_tokens": as_int(usage.get("total_tokens")),
        "input_cost_usd": 0.0,
        "cache_read_cost_usd": 0.0,
        "cache_write_cost_usd": 0.0,
        "output_cost_usd": 0.0,
        "total_cost_usd": as_float(cost.get("total_usd")),
        "uncached_cost_usd": as_float(cost.get("uncached_equivalent_usd")),
        "cache_savings_usd": as_float(cost.get("cache_savings_usd")),
    }
    for tier in by_tier_cost.values():
        if not isinstance(tier, dict):
            continue
        metrics["input_cost_usd"] += as_float(tier.get("input"))
        metrics["cache_read_cost_usd"] += as_float(tier.get("cache_read"))
        metrics["cache_write_cost_usd"] += as_float(tier.get("cache_write"))
        metrics["output_cost_usd"] += as_float(tier.get("output"))
    if not metrics["prompt_tokens"]:
        metrics["prompt_tokens"] = (
            metrics["input_tokens"] + metrics["cache_read_tokens"] + metrics["cache_write_tokens"]
        )
    if not metrics["total_tokens"]:
        metrics["total_tokens"] = metrics["prompt_tokens"] + metrics["output_tokens"]
    if not metrics["uncached_cost_usd"] and metrics["total_cost_usd"]:
        metrics["uncached_cost_usd"] = metrics["total_cost_usd"] + metrics["cache_savings_usd"]
    return metrics


def result_excerpt(pipeline: str, data: dict[str, Any]) -> dict[str, Any]:
    def clipped(value: Any, limit: int = 2000) -> str:
        text = str(value or "")
        return text if len(text) <= limit else text[:limit] + "…"

    if pipeline == "extractor":
        return {
            "title": clipped(data.get("title"), 300),
            "synthesis": clipped(data.get("synthesis")),
            "open_questions": [clipped(item, 500) for item in (data.get("open_questions") or [])[:10]],
            "operator_dialogue_considered": bool(data.get("operator_dialogue_considered")),
            "previous_rebuttal_count": len(data.get("previous_rebuttals") or {}),
        }
    return {
        "title": clipped(data.get("title"), 300),
        "disposition": clipped(data.get("disposition"), 300),
        "identified_cfr_category": clipped(data.get("identified_cfr_category"), 300),
        "cfr_clause": clipped(data.get("cfr_clause"), 500),
        "draft_response": clipped(data.get("draft_response")),
        "hitl_review": bool(data.get("hitl_review")),
        "hitl_reason": clipped(data.get("hitl_reason"), 500),
    }


SCHEMA = """
CREATE TABLE IF NOT EXISTS token_metric_records (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline TEXT NOT NULL,
    target TEXT NOT NULL,
    site_url TEXT NOT NULL DEFAULT '',
    folder_path TEXT NOT NULL,
    folder_key TEXT NOT NULL,
    run_id TEXT NOT NULL,
    case_id TEXT NOT NULL DEFAULT '',
    year TEXT NOT NULL,
    asset TEXT NOT NULL,
    asset_key TEXT NOT NULL,
    exception_id TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT '',
    rebuttal_cycle INTEGER NOT NULL DEFAULT 0,
    artifact_kind TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    output_location TEXT NOT NULL DEFAULT '',
    write_status TEXT NOT NULL DEFAULT 'cached',
    write_error TEXT NOT NULL DEFAULT '',
    estimator TEXT NOT NULL DEFAULT '',
    elapsed_minutes REAL,
    calls INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    thinking_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    priced_artifacts INTEGER NOT NULL DEFAULT 0,
    input_cost_usd REAL NOT NULL DEFAULT 0,
    cache_read_cost_usd REAL NOT NULL DEFAULT 0,
    cache_write_cost_usd REAL NOT NULL DEFAULT 0,
    output_cost_usd REAL NOT NULL DEFAULT 0,
    total_cost_usd REAL NOT NULL DEFAULT 0,
    uncached_cost_usd REAL NOT NULL DEFAULT 0,
    cache_savings_usd REAL NOT NULL DEFAULT 0,
    token_usage_json TEXT NOT NULL DEFAULT '{}',
    result_excerpt_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (pipeline, folder_key, run_id, relative_path)
);
CREATE INDEX IF NOT EXISTS idx_token_metrics_run
    ON token_metric_records (pipeline, folder_key, run_id);
CREATE INDEX IF NOT EXISTS idx_token_metrics_case
    ON token_metric_records (pipeline, folder_key, run_id, year, asset_key, exception_id);
CREATE TABLE IF NOT EXISTS token_metric_runs (
    pipeline TEXT NOT NULL,
    target TEXT NOT NULL,
    site_url TEXT NOT NULL DEFAULT '',
    folder_path TEXT NOT NULL,
    folder_key TEXT NOT NULL,
    run_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (pipeline, folder_key, run_id)
);
"""


def _connect(path: str | Path | None = None) -> sqlite3.Connection:
    resolved = cache_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(resolved, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


@dataclass(frozen=True)
class MetricIdentity:
    pipeline: str
    folder_key: str
    run_id: str
    relative_path: str


def cache_report_metrics(*, pipeline: str, data: dict[str, Any], year: str, asset: str,
                         exception_id: str, rebuttal_cycle: int, relative_path: str,
                         artifact_kind: str, run_id: str | None = None,
                         scope: OutputScope | None = None, status: str = "cached",
                         path: str | Path | None = None) -> MetricIdentity:
    """Upsert one artifact's lightweight metrics and return its stable cache identity."""
    scope = scope or configured_output_scope(pipeline)
    run_id = ensure_run_id(pipeline, run_id)
    metrics = metrics_from_report(data)
    usage = data.get("token_usage") or {}
    now = _now()
    asset_key = re.sub(r"[^a-z0-9]+", "", asset.lower())
    identity = MetricIdentity(pipeline, scope.folder_key, run_id, relative_path.replace("\\", "/"))
    values = {
        "pipeline": pipeline,
        "target": scope.target,
        "site_url": scope.site_url,
        "folder_path": scope.folder_path,
        "folder_key": scope.folder_key,
        "run_id": run_id,
        "case_id": str(data.get("case_id") or data.get("run_id") or ""),
        "year": str(year),
        "asset": str(asset),
        "asset_key": asset_key,
        "exception_id": str(exception_id),
        "category": str(data.get("exception_category") or data.get("title") or ""),
        "rebuttal_cycle": int(rebuttal_cycle),
        "artifact_kind": artifact_kind,
        "relative_path": identity.relative_path,
        "write_status": status,
        "estimator": str(usage.get("estimator") or ""),
        "elapsed_minutes": data.get("elapsed_minutes"),
        "token_usage_json": json.dumps(usage, separators=(",", ":"), ensure_ascii=False),
        "result_excerpt_json": json.dumps(
            result_excerpt(pipeline, data), separators=(",", ":"), ensure_ascii=False
        ),
        "created_at": now,
        "updated_at": now,
        **{key: metrics[key] for key in (
            "calls", "input_tokens", "cache_read_tokens", "cache_write_tokens", "prompt_tokens",
            "output_tokens", "thinking_tokens", "total_tokens", "priced_artifacts",
            "input_cost_usd", "cache_read_cost_usd", "cache_write_cost_usd", "output_cost_usd",
            "total_cost_usd", "uncached_cost_usd", "cache_savings_usd",
        )},
    }
    columns = list(values)
    placeholders = ", ".join("?" for _ in columns)
    updates = ", ".join(
        f"{column}=excluded.{column}" for column in columns
        if column not in {"pipeline", "folder_key", "run_id", "relative_path", "created_at"}
    )
    with closing(_connect(path)) as conn, conn:
        conn.execute(
            """INSERT INTO token_metric_runs
               (pipeline, target, site_url, folder_path, folder_key, run_id, started_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(pipeline, folder_key, run_id) DO UPDATE SET
                 target=excluded.target, site_url=excluded.site_url,
                 folder_path=excluded.folder_path, updated_at=excluded.updated_at""",
            (pipeline, scope.target, scope.site_url, scope.folder_path, scope.folder_key,
             run_id, now, now),
        )
        conn.execute(
            f"INSERT INTO token_metric_records ({', '.join(columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT(pipeline, folder_key, run_id, relative_path) DO UPDATE SET {updates}",
            [values[column] for column in columns],
        )
    return identity


def mark_metric_write(identity: MetricIdentity, *, location: str = "", error: str = "",
                      path: str | Path | None = None) -> None:
    status = "failed" if error else "written"
    with closing(_connect(path)) as conn, conn:
        conn.execute(
            """UPDATE token_metric_records
               SET output_location=?, write_status=?, write_error=?, updated_at=?
               WHERE pipeline=? AND folder_key=? AND run_id=? AND relative_path=?""",
            (location, status, error, _now(), identity.pipeline, identity.folder_key,
             identity.run_id, identity.relative_path),
        )


def list_metric_records(*, pipeline: str | None = None,
                        path: str | Path | None = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM token_metric_records"
    params: tuple[Any, ...] = ()
    if pipeline:
        sql += " WHERE pipeline=?"
        params = (pipeline,)
    sql += " ORDER BY updated_at, record_id"
    with closing(_connect(path)) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def get_metric_record(record_id: int, *, path: str | Path | None = None) -> dict[str, Any] | None:
    with closing(_connect(path)) as conn:
        row = conn.execute(
            "SELECT * FROM token_metric_records WHERE record_id=?", (int(record_id),)
        ).fetchone()
    return dict(row) if row else None


def delete_cache(path: str | Path | None = None) -> None:
    """Test helper. Production code is append-only and never calls this function."""
    resolved = cache_path(path)
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(resolved) + suffix)
        if candidate.exists():
            candidate.unlink()


__all__ = [
    "MetricIdentity", "OutputScope", "cache_path", "cache_report_metrics",
    "configured_output_scope", "delete_cache", "ensure_run_id", "get_metric_record",
    "list_metric_records", "mark_metric_write", "metrics_from_report", "new_run_id",
    "result_excerpt", "source_url",
]
