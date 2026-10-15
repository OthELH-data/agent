"""The permanent, UI-readable trace (design doc lines 82-84, decision #7).

Two message channels exist in this system:

* **Working memory** — the ephemeral in-node ReAct buffer (`react_loop.py`), resent to the model each
  turn and discarded when the node returns. It never touches disk.
* **Permanent trace** — THIS module. Every node emits a compact structured event AND a one-line
  message, the event carrying what the node spent on LLM calls (`tokens`, from token_trace.py — the
  same blocks the ReAct steps carry, which is what makes the trace add up to the report's `by_node`).
  `trace_event` does two things at once:
    1. appends the event as a JSON line to the case's `*_otl_trace.jsonl` file immediately, so a UI can
       tail it live while the graph runs, and
    2. returns a partial state update ({"trace": [...], "messages": [...], "nl_summaries_by_node": {...}}) the
       node folds into what it returns, so the event also accumulates in graph state (via the
       `operator.add` / `add_messages` / `merge_dict` reducers) and ends up embedded in the final JSON.

State stays checkpoint-safe: it carries only the trace file PATH (a string), never a live handle.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from otl.token_trace import node_tokens


def _now_iso() -> str:
    """UTC timestamp, seconds precision — stable and sortable in the JSONL."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_jsonl(path: str | None, rec: dict) -> None:
    """Append one event as a JSON line. No-op when no path is set (e.g. unit tests)."""
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")


def _digest(value: Any, limit: int = 400) -> str:
    """A compact, single-line string form of key outputs for the message log."""
    try:
        s = json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        s = str(value)
    s = " ".join(s.split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def trace_event(state: dict, agent: str, action: str, *,
                key_outputs: dict | None = None, inputs_digest: str = "",
                nl_summary: str | None = None, message: str | None = None) -> dict:
    """Record one node step: flush a JSONL line now, and return the partial state update to merge.

    Args:
      state:         the current graph state (read for run_id / case_id / trace_path).
      agent:         node name, e.g. "jade_analyzer".
      action:        short verb phrase, e.g. "reconstructed amount".
      key_outputs:   the salient typed outputs of this step (kept small — no raw workbook rows).
      inputs_digest: a short description of what the node consumed.
      nl_summary:    the agent's bounded narrative summary (also flows into `nl_summaries_by_node`).
      message:       an explicit message-log line; defaults to a digest of key_outputs.

    Returns:
      {"trace": [rec], "messages": [AIMessage], "nl_summaries_by_node": {agent: nl_summary}?} — a partial
      update the node spreads into its return value.
    """
    case = state.get("case") or {}
    rec = {
        "ts": _now_iso(),
        "run_id": state.get("run_id", ""),
        "case_id": case.get("case_id", ""),
        "agent": agent,
        "action": action,
        "inputs_digest": inputs_digest,
        "key_outputs": key_outputs or {},
        # What this node spent DIRECTLY on this visit (its structured/text calls + any parse retry).
        # A ReAct specialist reports calls: 0 here — its spend is itemized on its react_step records,
        # and no call is ever claimed twice. See token_trace.py.
        "tokens": node_tokens(agent),
    }
    _append_jsonl(state.get("trace_path"), rec)

    body = message if message is not None else _digest(key_outputs or {})
    update: dict[str, Any] = {
        "trace": [rec],
        "messages": [AIMessage(content=f"[{agent}] {action}: {body}")],
    }
    if nl_summary is not None:
        update["nl_summaries_by_node"] = {agent: nl_summary}
    return update


class TraceWriter:
    """run.py-side helper to initialize a case's trace file and read it back for embedding."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def reset(self) -> "TraceWriter":
        """Start a fresh (empty) trace file for this case — call once before graph.invoke."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        return self

    def read_events(self) -> list[dict]:
        """Load every event line back (for embedding the full trace array in the final report)."""
        if not self.path.exists():
            return []
        out: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out


__all__ = ["trace_event", "TraceWriter"]
