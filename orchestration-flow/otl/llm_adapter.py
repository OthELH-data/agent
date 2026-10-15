"""Labor adapter over the shared LLM client, adding per-node model tiering.

The extractor exposes three tiers — `high` / `med` / `light` — resolved to concrete models by the
selected `LLM_PROVIDER` (ICA: opus/sonnet/haiku; Azure: gpt-5.4-pro/gpt-5.4/gpt-5.4-mini), with
`structured_call`/`text_call` that already wire token accounting (`tokens.USAGE`) and JSON repair.
This module maps each OTL node to its tier (design doc line 91) and forwards the call, so agent code
says `structured(node, system, user, Schema)` without knowing which model backs it.

Tier assignment:
  high   — orchestrator (highest-stakes routing), drafter (final reasoning/synthesis),
           rebuttal_history
  med    — jade_analyzer, location, job_title, matching, supervision (code-exec specialists that
           reason+code+loop over data) + verify_evidence (interpretation of findings) + rebuttal_compaction
           (history compression)
  light  — intake, category_check (light parse/classify), wiki (index page
           selection only), agreement_retriever (candidate-ID reranking only) + jade_people (pulling
           person names out of the JADE ledger's name column)
"""
from __future__ import annotations

import os
from typing import Type

# Re-exported message classes so agents / the ReAct loop have one import site.
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: F401
from pydantic import BaseModel

from shared import llm as _xllm
from shared.hooks import HOOKS
from shared.token_accounting import estimate_message_tokens
from shared.tokens import USAGE as _USAGE


HOOK_MODULE = "labor"

# node name -> tier
_NODE_TIER: dict[str, str] = {
    "orchestrator": "high",     # the hub supervisor — highest-stakes routing decision, so top tier
    "drafter": "high",          # final reasoning/synthesis of the rebuttal
    "jade_analyzer": "med",     # code-exec specialist
    "location": "med",          # code-exec specialist
    "job_title": "med",         # code-exec specialist
    "matching": "med",          # code-exec specialist streaming the 59k-row CFR-mapping workbook
    "supervision": "med",       # code-exec specialist walking the staff-listing org tree (person_key/reports_to)
    "verify_evidence": "high",
    "intake": "light",
    "category_check": "med",
    "wiki": "light",              # independent index-first precedent selection sidecar
    "agreement_retriever": "light",  # ranks IDs; source chunks are hydrated in code afterward
    "rebuttal_history": "high",  # durable cross-cycle compaction and continuity synthesis
    "rebuttal_compaction": "med",  # compression only; cycle-delta assessment stays high tier
    # Not a graph node: the name-extraction call `jade_analyzer` makes before its ReAct loop starts
    # (otl/ledger_people.py). It is listed here so the call is attributed to the light tier it actually
    # runs on — `tier_for` would otherwise fall back to "med" and misreport the cost.
    "jade_people": "light",
    # Not a graph node: only used when no extractor manifest is available. It extracts filenames from
    # the auditor's current exception description before the shared discovery layer resolves them.
    "auditor_attachment_citations": "light",
    # Not a graph node: all pre-loop corpus-name matching calls made by the location / job_title /
    # supervision specialists (otl/person_index.py). The call only selects workbook spellings for
    # already-known disputed people; the specialists do the substantive evidence reasoning later.
    "person_index": "light",
}
_DEFAULT_TIER = "med"
_VALID_TIERS = ("high", "med", "light")

# The reason+code+loop specialists — the "code-execution" agents. Their tier is overridable in one
# shot via `set_code_exec_tier()` / the `OTL_CODE_EXEC_TIER` env var (the `--code-exec-tier` CLI flag),
# without touching the orchestrator (routing) or drafter (final reasoning) tiers.
_CODE_EXEC_NODES = ("jade_analyzer", "location", "job_title", "matching", "supervision")


def set_code_exec_tier(tier: str) -> str:
    """Point every code-exec specialist node at `tier` (high|med|light). Returns the applied tier.

    Mutates the process-global `_NODE_TIER`, so call it once at startup BEFORE the graph runs (nodes
    read their tier per-call via `tier_for`, so any time before `graph.invoke` takes effect)."""
    t = (tier or "").strip().lower()
    if t not in _VALID_TIERS:
        raise ValueError(f"code-exec tier must be one of {_VALID_TIERS}, got {tier!r}")
    for node in _CODE_EXEC_NODES:
        _NODE_TIER[node] = t
    return t


# Honor an env-var override at import so it propagates to `run.py` subprocesses (run_batch passes the
# parent env). The CLI flag calls `set_code_exec_tier` directly; this is the belt-and-suspenders path.
_env_tier = os.getenv("OTL_CODE_EXEC_TIER")
if _env_tier and _env_tier.strip().lower() in _VALID_TIERS:
    set_code_exec_tier(_env_tier)


def tier_for(node: str) -> str:
    """The model tier backing a node (falls back to 'med' for unknown nodes)."""
    return _NODE_TIER.get(node, _DEFAULT_TIER)


def node_tier_map() -> dict[str, str]:
    """Snapshot of node → tier, after any `set_code_exec_tier` override. For `run.py --llm-config`.

    A copy, so a caller printing the map can't mutate the routing table it is reporting on.
    """
    return dict(_NODE_TIER)


def context_window_for_node(node: str) -> int:
    """Configured input context window for the model serving ``node``."""
    tier = tier_for(node)
    model = _xllm.model_for_tier(tier)
    return int(_xllm._max_input_tokens(tier, model))


def estimate_text_tokens(text: str) -> int:
    """Conservative provider-independent estimate for one history prompt block."""
    if not text:
        return 0
    return int(estimate_message_tokens([HumanMessage(content=text)]))


def structured(node: str, system: str, user: str, schema: Type[BaseModel],
               retries: int = 1, temperature: float = 0.0,
               *, allow_budget_overrun: bool = False) -> BaseModel:
    """Tier-resolved `structured_call`: returns a validated instance of `schema`.

    The `attribute` scope tags the call (and any parse retry, each of which is billed and recorded
    separately) with this node, so `tokens.USAGE` can break the tier's totals down per node — see
    `otl/token_trace.py`. graph.py already scopes the whole node execution; naming the node here too
    keeps attribution right when an agent function is called directly (tests).
    """
    with _USAGE.attribute(node=node, kind="structured"):
        return _xllm.structured_call(tier_for(node), system, user, schema,
                                     retries=retries, temperature=temperature,
                                     hook_module=HOOK_MODULE, hook_stage=node,
                                     allow_budget_overrun=allow_budget_overrun)


def text(node: str, system: str, user: str, temperature: float = 0.0) -> str:
    """Tier-resolved `text_call`: returns the raw reply text."""
    with _USAGE.attribute(node=node, kind="text"):
        return _xllm.text_call(tier_for(node), system, user, temperature=temperature,
                               hook_module=HOOK_MODULE, hook_stage=node)


def invoke_messages(node: str, messages: list, temperature: float = 0.0,
                    conv_key: str = "") -> str:
    """Tier-resolved raw chat turn over a full message buffer (the ReAct loop's primitive).

    Unlike `text`/`structured` (which build a fresh 2-message prompt), this sends the entire
    working-memory buffer — system + the running human/assistant exchange — and returns the reply,
    recording token usage on the node's tier just like the extractor's own calls do.

    `conv_key` identifies this buffer across turns so the context check can anchor turn n+1 on the
    usage the gateway reported for turn n, rather than re-estimating a 50-turn history from scratch.
    This is the ONE call path with a growing buffer, so it is the only one that needs it.
    """
    tier = tier_for(node)
    with _USAGE.attribute(node=node, kind="react_turn"):
        HOOKS.before_call(
            module=HOOK_MODULE, call_type="text_call", stage=node,
            usage_report=_USAGE.report(), usage_generation=_USAGE.generation(),
        )
        # Route through _invoke_tier (not get_llm + _invoke) so this path gets the Azure→ICA fallback.
        reply, model = _xllm._invoke_tier(tier, messages, temperature, conv_key=conv_key)
        _xllm.record_reply_usage(tier, model, reply, messages)
        HOOKS.after_call(
            module=HOOK_MODULE, call_type="text_call", stage=node,
            usage_report=_USAGE.report(), usage_generation=_USAGE.generation(),
        )
    return reply.text


def first_json(text_reply: str) -> dict:
    """Extract the first balanced top-level JSON object from a model reply (reuses the shared
    parser, including its stray-backslash repair)."""
    return _xllm._extract_json(text_reply)


__all__ = [
    "HOOK_MODULE", "tier_for", "set_code_exec_tier", "structured", "text", "invoke_messages",
    "first_json",
    "context_window_for_node", "smallest_context_window", "estimate_text_tokens",
    "AIMessage", "HumanMessage", "SystemMessage",
]
