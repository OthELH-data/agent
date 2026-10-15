"""The in-node ReAct harness (design doc line 89, decisions #5 & #7).

`run_agent_loop` is the reason→act→observe loop the data-exploration agents (jade_analyzer, location,
job_title) run *inside* a single graph node. It is the concrete realization of "in-context working
memory, no read_file, no disk-for-memory":

  * A fresh `list[BaseMessage]` buffer is created on entry, resent IN FULL to the model each turn (so
    the model "reads" its own history), and DISCARDED when the loop returns (the "wipe"). Only the
    distilled `result` + the agent's bounded `findings` cross into shared graph state.
  * The model's only tool is `run_python`: it writes code that opens the workbook via `open_data(...)`,
    computes, and prints / `save_result(...)` — so raw rows never enter the buffer, only summaries.

Termination (design doc line 89 — loop until conclusion, no rabbit hole):
  * FINISH sentinel — the model declares a definitive conclusion (`action:"finish"`), distilled to
    the node's sub-state. This is the ONLY early exit.
  * MAX_STEPS — the SOLE backstop. A code step that errors, or an agent still struggling to
    locate/interpret data, keeps looping (regenerating code, trying another column/tab) with the
    error fed back each turn, until it finishes or exhausts MAX_STEPS. No "give up after N errors".
  * per-step timeout — each `run_python` execution is bounded (~150s) so one call can't hang the loop.

A MAX_STEPS exhaustion returns `status="max_steps"` carrying the agent's LAST `save_result(...)`
checkpoint (the PROTOCOL tells agents to checkpoint cumulatively as they go, so partial findings are not
lost when the budget runs out); the calling node reports a non-blocking `agent_status="inconclusive"` and
returns to the orchestrator, which decides whether to re-dispatch (once, with a sharper instruction) or
finalize.

Per-step records are appended to the permanent JSONL trace (if a `trace_path` is given) so a UI can
watch the reasoning live — each one carrying what that turn's LLM call cost (`tokens`, see
token_trace.py); the ephemeral message buffer itself is never persisted.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from otl import llm_adapter as _llm
from otl.code_exec import CodeExecutor
from otl.token_trace import step_scope, step_tokens
from otl.trace import _append_jsonl, _digest, _now_iso

_PY_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)
_STREAM_TRUNC = 12000       # cap stdout/stderr fed back into the buffer
_RESULT_TRUNC = 12000       # cap the save_result payload digest


# The action protocol, appended to the task message so the loop is self-contained regardless of the
# node's system prompt.
PROTOCOL = """\
You work in a reason→act→observe loop. Reply with EXACTLY ONE JSON object per turn, no prose outside it:

  To run code:   {"thought": "<why>", "action": "run_python", "code": "<python>"}
  To conclude:   {"thought": "<why>", "action": "finish", "result": { ... your typed findings ... }}

In `code` you have: INPUTS (dict of the inputs given to you), open_data(path) -> a validated Path you
MUST use to open any data file (it enforces the read window), list_sheets(path) -> the readable sheet
names of a workbook, and save_result(obj) to persist a JSON-serializable result. print(...) whatever
you need to observe. Compute summaries — never dump raw rows. You will receive an OBSERVATION with
stdout/stderr/result after each run_python. When you are certain of the answer, emit action:"finish"
with the result. Keep going and try another approach if a step errors or the data isn't where you
expected.

READING WORKBOOKS: your workbooks were already profiled on disk before this loop started, so do not
spend turns finding out what they hold. In your code, manifest() returns that profile — per workbook the
engine, the usable sheet names, and the tabs grouped by shape (each shape gives a 0-based header row
index and the column labels) — and find_tab(person) returns the tabs that could belong to a person, each
with its shape. Both read a staged file, so they are free and instant; both return {} or [] if nothing
was staged for you. Treat the manifest as a PRIOR, not a guarantee: raw reads are still available, and if
what you read disagrees with the manifest, trust your read and say so. list_sheets(path) is still there
when you want the sheet names alone. Stream a large sheet with openpyxl
load_workbook(read_only=True, data_only=True) and ws.iter_rows(). A read-only worksheet has NO
.dimensions attribute — asking for it raises AttributeError — and its .max_row is None unless the tool
that wrote the file happened to record one. So do not call either. Count rows as you iterate instead.

FINDING PEOPLE: do NOT scan a workbook to find a person. When your node works from the POB report, the
helicopter manifest or the staff listing, those files were read in full before this loop started and one
model pass over every distinct name in them already picked out the disputed people. person_index() in your
code returns the result: for each person the spelling the FILE uses (read off the sheet, so it is exact),
that person's rows already pulled and keyed by COLUMN LABEL, and the tab's column-role map (role -> label
-> column index) so you never have to guess which column is the person key, the reports-to pointer or the
job profile. It also says which roles that tab does NOT carry, which tabs were skipped and why, and for a
person nothing was selected for, how many values share their surname — so "absent from the file" can be
told apart from "not looked for", and absence is a finding you report rather than re-scan for. Same
doctrine as the manifest: it is a PRIOR, not a guarantee. Raw reads stay available, and if what you read
disagrees with the index, trust your read and say so. Say which spelling you adopted for each person and
why. person_index() returns {} when nothing was staged for you — then read the files yourself.

CHECKPOINT AS YOU GO: call save_result(...) with your findings-so-far as soon as you establish anything
solid — do not save only at the end. Each call REPLACES the previous checkpoint, so always pass the
CUMULATIVE result object (everything resolved so far), shaped like the result you would `finish` with;
leave fields you have not determined yet empty. If you exhaust the step budget before emitting `finish`,
your LAST checkpoint is what survives — an unsaved finding is a lost finding."""


@dataclass
class AgentLoopResult:
    status: str                                  # "finished" | "max_steps" | "error"
    result: dict = field(default_factory=dict)   # the model's finish payload (or best partial)
    steps: int = 0
    steps_log: list[dict] = field(default_factory=list)
    last_observation: str = ""


def _trunc(s: str, limit: int) -> str:
    s = s or ""
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _format_observation(r: dict) -> str:
    """Compact, model-facing rendering of a CodeExecutor.run result."""
    head = [f"ok={r.get('ok')}"]
    if r.get("timed_out"):
        head.append("TIMED_OUT")
    if r.get("error"):
        head.append(f"error={r['error']}")
    body: list[str] = []
    if r.get("stdout"):
        body.append("STDOUT:\n" + _trunc(r["stdout"], _STREAM_TRUNC))
    if r.get("stderr"):
        body.append("STDERR:\n" + _trunc(r["stderr"], _STREAM_TRUNC))
    if r.get("result") is not None:
        body.append("RESULT(save_result):\n" + _digest(r["result"], _RESULT_TRUNC))
    return " ".join(head) + ("\n" + "\n".join(body) if body else "")


def _extract_action(reply: str) -> dict:
    """Parse the turn into an action dict. Tolerates a bare ```python fence when JSON is absent."""
    try:
        obj = _llm.first_json(reply)
        if isinstance(obj, dict) and obj.get("action") in ("run_python", "finish"):
            return obj
    except Exception:
        obj = None
    # Fallback: a code fence with no JSON envelope → treat as a run_python action.
    m = _PY_FENCE_RE.search(reply)
    if m:
        return {"thought": "", "action": "run_python", "code": m.group(1).strip()}
    return {"action": "malformed", "raw": reply}


def _max_steps(default: int = 50) -> int:
    try:
        return max(1, int(os.getenv("OTL_MAX_STEPS", str(default))))
    except ValueError:
        return default


def run_agent_loop(*, node: str, system: str, task: str, executor: CodeExecutor,
                   inputs: dict | None = None, max_steps: int | None = None,
                   step_timeout: int = 150, trace_path: str | None = None,
                   run_id: str = "", case_id: str = "") -> AgentLoopResult:
    """Run the reason→act→observe loop for `node` and return its distilled outcome.

    Args:
      node:         node name (selects the model tier via llm_adapter).
      system:       the agent's system prompt (inline rules from prompts.py).
      task:         the concrete task + any forwarded open_questions hints.
      executor:     a CodeExecutor already scoped to this case's read roots + scratch.
      inputs:       dict exposed to the code as INPUTS (paths, seed names, questioned amount, …).
      max_steps:    override the OTL_MAX_STEPS backstop.
      step_timeout: per-run_python execution timeout (seconds).
      trace_path/run_id/case_id: when set, append one JSONL line per step to the permanent trace.
    """
    steps_cap = max_steps or _max_steps()
    messages = [
        _llm.SystemMessage(content=system),
        _llm.HumanMessage(content=f"{task}\n\n{PROTOCOL}"),
    ]
    steps_log: list[dict] = []
    last_obs = ""
    # The most recent non-empty save_result(...) payload — the agent's running checkpoint. Returned
    # verbatim if MAX_STEPS is exhausted, so an agent that resolved most of its work but never reached
    # `finish` (e.g. it burned the budget on a later step) does not lose everything it had established.
    partial: dict = {}

    def _trace_step(idx: int, kind: str, payload: dict, tokens: dict) -> None:
        """One step's permanent record. `tokens` is what THIS turn's LLM call cost (see token_trace):
        exactly one call per step, so summing these blocks over a node reproduces its `by_node` total
        in the report footer."""
        rec = {"ts": _now_iso(), "run_id": run_id, "case_id": case_id, "agent": node,
               "action": f"react_step:{kind}", "step": idx, "key_outputs": payload,
               "tokens": tokens}
        steps_log.append(rec)
        _append_jsonl(trace_path, rec)

    # Stable identity for THIS buffer, so each turn's context check is anchored on the token usage
    # the gateway reported for the previous turn instead of re-estimating the whole history.
    conv_key = f"{run_id}:{case_id}:{node}"

    for step in range(1, steps_cap + 1):
        # One LLM call per step, attributed to this step so its cost lands on this step's record.
        with step_scope(step):
            reply = _llm.invoke_messages(node, messages, conv_key=conv_key)
        tokens = step_tokens(node, step)
        messages.append(_llm.AIMessage(content=reply))
        action = _extract_action(reply)
        kind = action.get("action")

        if kind == "finish":
            result = action.get("result") if isinstance(action.get("result"), dict) else {}
            _trace_step(step, "finish", {"thought": _trunc(action.get("thought", ""), 300),
                                         "result": result}, tokens)
            return AgentLoopResult(status="finished", result=result, steps=step,
                                   steps_log=steps_log, last_observation=last_obs)

        if kind == "run_python":
            code = action.get("code") or ""
            exec_result = executor.run(code, inputs=inputs or {}, timeout=step_timeout)
            last_obs = _format_observation(exec_result)
            saved = exec_result.get("result")
            if isinstance(saved, dict) and saved:
                partial = saved          # checkpoint — survives a MAX_STEPS exhaustion
            _trace_step(step, "run_python",
                        {"thought": _trunc(action.get("thought", ""), 300),
                         "ok": exec_result.get("ok"), "error": exec_result.get("error"),
                         "result": exec_result.get("result")}, tokens)
            messages.append(_llm.HumanMessage(content=f"OBSERVATION:\n{last_obs}"))
            continue

        # malformed turn — nudge back onto the protocol and keep looping (counts as a step)
        _trace_step(step, "malformed", {"raw": _trunc(action.get("raw", ""), 300)}, tokens)
        messages.append(_llm.HumanMessage(
            content="Your reply was not a single valid JSON action object. Reply with exactly one "
                    'JSON object using action "run_python" or "finish" per the protocol.'))

    # Exhausted the budget without a `finish`. Return the last save_result(...) checkpoint rather than
    # {} so partially-resolved findings reach the calling node (which still reports "inconclusive").
    return AgentLoopResult(status="max_steps", result=partial, steps=steps_cap,
                           steps_log=steps_log, last_observation=last_obs)


__all__ = ["run_agent_loop", "AgentLoopResult", "PROTOCOL"]
