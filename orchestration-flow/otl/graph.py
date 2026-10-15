"""The OTL graph — a sequential hub-and-spoke LLM supervisor (orchestrator).

The deterministic prep spine runs first. `verify_evidence` is an optional, YAML-configured HARD
pass/fail gate: when enabled and a configured evidence decision is triggered, the spine short-circuits
directly to the drafter. Unsupported auditor
evidence denies the exception; supplied evidence that cannot be accessed defers to HITL. In both cases,
jade_analyzer, the orchestrator and the specialists are skipped.
When the gate PASSES, `jade_analyzer` (the last spine node — the questioned-amount reconstruction is
needed for every case) runs, then a highest-tier `orchestrator` node becomes the hub: on each pass it
reads every sub-agent's returned state and decides the next hop, dispatching exactly ONE specialist
(with a custom instruction) or finalizing through the wiki and agreement stages to the drafter. Each specialist returns to the orchestrator,
which re-decides with the new findings — so "location AND job_title" is two successive hops, and a
specialist's custom instruction can be informed by what earlier specialists found. Each specialist is
dispatched AT MOST ONCE — the orchestrator never revisits a node it has already run (see
agents/orchestrator.py). Three edges are content-driven: `route_from_verify_evidence` (the configured gate),
`route_from_jade_analyzer` (an operational analysis-block guard) and `route_from_orchestrator` (which just
reads the decision the orchestrator already wrote to `next_agent`).

`jade_analyzer` owns ED-02: a completed sum/half reconstruction outside the established ±2.5% tolerance
routes directly to a deterministic denial. When verification is disabled and JADE cannot operate, it
still records an operationally inconclusive analysis and routes to HITL without applying ED-01.

    intake → rebuttal_history → category_check

    category_check --node gate enabled--> verify_evidence
                   --node gate disabled-> jade_analyzer

    verify_evidence  --route_from_verify_evidence-->  drafter (gate fail) | jade_analyzer (gate pass)
    jade_analyzer    --route_from_jade_analyzer-->    drafter (analysis blocked) | orchestrator (normal)

    orchestrator  --route_from_orchestrator-->  location | job_title | matching | supervision | wiki
    location | job_title | matching | supervision   --(static)-->   orchestrator
    agreement_retriever → wiki → drafter → END

The hard evidence-failure paths target ``drafter`` directly because no substantive labor analysis exists
to enrich. The orchestrator's terminal route targets ``agreement_retriever``; governing authority is
selected first, then ``wiki`` supplies a bounded precedent-status fact and finalizes chargeability hooks
before the drafter runs.

The loop is bounded by the orchestrator's own single-visit rule + hop/run-count guards
(agents/orchestrator.py), with `recursion_limit` (run.py) as the engine-level backstop. Node
implementations are resolved progressively: `build_otl_graph()` imports each `<name>_node` from
`otl.agents.<module>` if present, else uses an inline pass-through stub. So the graph compiles with
all stubs (the orchestrator stub takes the terminal agreement_retriever → wiki → drafter path) and lights up phase by phase without
any rewiring here. Every registered node is additionally wrapped in `token_trace.node_visit`, which is
what attributes each LLM call to the node that made it (report footer `by_node`, trace `tokens`).
"""
from __future__ import annotations

import functools
import importlib

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from shared.rebuttal_history import write_agent_analysis

from shared.pipeline_config import node_gate_enabled
from shared.hooks_config import chargeability_hook_definitions, node_hook_definitions
from otl.state import OTLAuditState
from otl.token_trace import node_visit
from otl.trace import trace_event

# graph node name -> module under otl.agents that defines `<node>_node`
_NODE_MODULES: dict[str, str] = {
    "rebuttal_history": "rebuttal_history",
    "intake": "intake",
    "category_check": "category_check",
    "verify_evidence": "verify_evidence",
    "orchestrator": "orchestrator",
    "jade_analyzer": "jade_analyzer",
    "location": "location",
    "job_title": "job_title",
    "matching": "matching",
    "supervision": "supervision",
    "wiki": "wiki",
    "agreement_retriever": "agreement_retriever",
    "drafter": "drafter",
}

_ANALYSIS_WRITER_NODES = {
    "intake", "category_check", "verify_evidence", "jade_analyzer",
    "orchestrator", "location", "job_title", "matching", "supervision", "wiki",
    "agreement_retriever", "drafter",
}

# Routing defaults each stub emits so the skeleton flows to END on the happy path in P1. The
# orchestrator stub finalizes immediately (through wiki to the drafter) so an all-stubs graph terminates.
_STUB_DEFAULTS: dict[str, dict] = {
    "category_check": {"auditor_question": []},
    "verify_evidence": {},
    "orchestrator": {"next_agent": "drafter", "final_mode": "terminal"},
}


# --------------------------------------------------------------------------------------------------
# router functions
# --------------------------------------------------------------------------------------------------
def route_from_category_check(state: dict) -> str:
    """Run the configurable evidence gate or bypass it to the next spine node.

    The route is controlled by ``configs/pipeline.yaml`` (or ``OTL_PIPELINE_CONFIG``). Disabling
    the gate bypasses the entire ``verify_evidence`` node, including its LLM call, and proceeds directly
    to ``jade_analyzer``. Future node gates can reuse ``node_gate_enabled`` with their own next-node route.
    """
    auditor_question = set(state.get("auditor_question") or ["other"])
    analytic = bool(auditor_question.intersection({"job_title", "location", "hierarchy"}))
    third_party_invoice = bool(
        state.get("third_party_invoice_requested")
        or "third_party_invoice_requested" in auditor_question
    )
    if not analytic and not third_party_invoice:
        return "drafter"
    if node_gate_enabled("verify_evidence"):
        return "verify_evidence"
    return "jade_analyzer" if analytic else "drafter"


def route_from_verify_evidence(state: dict) -> str:
    """The hard evidence gate. When the auditor's OWN cited evidence is insufficient
    (the aggregate decision is triggered), select the direct drafter route. The drafter deterministically denies
    missing support and defers inaccessible supplied evidence without reconstructing the amount or
    dispatching any specialist. A missing decision fails open to the full pipeline."""
    decision = state.get("evidence_gate_decision") or {}
    if decision.get("triggered") is True:
        return "drafter"
    questions = set(state.get("auditor_question") or [])
    if (
        not questions.intersection({"job_title", "location", "hierarchy"})
        and (state.get("third_party_invoice_requested")
             or "third_party_invoice_requested" in questions)
    ):
        return "drafter"
    revisit = state.get("rebuttal_specialist_revisit") or {}
    if state.get("rebuttal_history") and revisit.get("jade_analyzer") is False:
        return "orchestrator"
    return "jade_analyzer"


def route_from_jade_analyzer(state: dict) -> str:
    """Route a configured ED-02 decision or operational block; otherwise hand off to the orchestrator.

    Missing-ledger policy belongs exclusively to the optional ``verify_evidence`` node. When that node
    is bypassed, JADE may still discover that analysis is operationally impossible; ``analysis_blocked``
    routes that inconclusive condition to the drafter without recreating an ED-01 denial here.
    """
    decision = state.get("jade_hook_decision") or {}
    if decision.get("triggered") is True:
        return "drafter"
    return "drafter" if state.get("analysis_blocked") is True else "orchestrator"


def route_from_orchestrator(state: dict) -> str:
    """Hub → exactly one spoke. Reads `next_agent` (written by the orchestrator) and clamps it to a
    valid target; an unknown/missing value fails safe to the drafter so the run always terminates.
    `jade_analyzer` is NOT a target — it runs once in the spine, before the hub."""
    valid = {"location", "job_title", "matching", "supervision", "drafter"}
    nxt = state.get("next_agent")
    return nxt if nxt in valid else "drafter"


# --------------------------------------------------------------------------------------------------
# node resolution (real impl if present, else stub)
# --------------------------------------------------------------------------------------------------
def _make_stub(node: str):
    """A pass-through node: emits a trace event and any routing defaults so the skeleton is runnable."""
    defaults = _STUB_DEFAULTS.get(node, {})

    def _node(state: dict) -> dict:
        update = dict(defaults)
        update.update(trace_event(state, node, "stub pass-through",
                                  key_outputs=defaults, nl_summary=f"{node} (stub)"))
        return update

    _node.__name__ = f"{node}_stub"
    return _node


def _instrument(name: str, fn):
    """Wrap a node so every LLM call it makes is attributed to it (token_trace.node_visit).

    Registered for EVERY node in one place, so the per-node token roll-up in the report footer and the
    per-step blocks in the trace can never miss a node. `functools.wraps` keeps LangGraph's signature
    introspection pointed at the original `(state: dict) -> dict`, and the passthrough keeps working if
    a node ever takes a second (config) argument.
    """
    @functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        with node_visit(name):
            update = fn(*args, **kwargs)
            if name in _ANALYSIS_WRITER_NODES and args and isinstance(update, dict):
                write_agent_analysis({**args[0], **update}, name, update)
            return update

    return _wrapped


def _resolve_node(node: str):
    """Import `otl.agents.<module>.<node>_node`; fall back to a stub if it isn't implemented yet."""
    module = _NODE_MODULES[node]
    try:
        mod = importlib.import_module(f"otl.agents.{module}")
        fn = getattr(mod, f"{node}_node", None)
        if callable(fn):
            return fn
    except ModuleNotFoundError:
        pass
    return _make_stub(node)


# --------------------------------------------------------------------------------------------------
# graph builder
# --------------------------------------------------------------------------------------------------
def build_otl_graph(checkpointer=None, nodes: dict | None = None):
    """Compile the OTL StateGraph.

    Args:
      checkpointer: a LangGraph checkpointer (defaults to an in-memory MemorySaver; run.py uses
                    thread_id=case_id when invoking).
      nodes:        optional explicit {node_name: callable} overrides; unspecified nodes are resolved
                    from otl.agents so adding the wiki sidecar does not break existing callers.
    """
    # Fail fast on malformed production decision-hook configuration before a case enters the graph.
    node_hook_definitions("labor", "verify_evidence")
    node_hook_definitions("labor", "jade_analyzer")
    chargeability_hook_definitions()
    resolved = {name: _resolve_node(name) for name in _NODE_MODULES}
    resolved.update(nodes or {})
    g = StateGraph(OTLAuditState)
    for name, fn in resolved.items():
        g.add_node(name, _instrument(name, fn))

    # Intake first parses only the current submission. Cross-cycle memory then compares that fresh
    # intake with prior analysis before any downstream reasoning or routing node runs.
    g.add_edge(START, "intake")
    g.add_edge("intake", "rebuttal_history")
    g.add_edge("rebuttal_history", "category_check")

    # static prep spine up to the evidence gate.
    g.add_conditional_edges("category_check", route_from_category_check, {
        "verify_evidence": "verify_evidence",
        "jade_analyzer": "jade_analyzer",
        "drafter": "drafter",
    })

    # Configured verify-evidence decision: failures go directly to the drafter; passes continue.
    g.add_conditional_edges("verify_evidence", route_from_verify_evidence, {
        "drafter": "drafter",
        "jade_analyzer": "jade_analyzer",
        "orchestrator": "orchestrator",
    })
    # JADE routes a configured ED-02 mismatch or an operationally blocked analysis directly to drafter.
    g.add_conditional_edges("jade_analyzer", route_from_jade_analyzer, {
        "drafter": "drafter",
        "orchestrator": "orchestrator",
    })

    # hub → exactly one spoke (the only content-driven edge). Invariant: the orchestrator has ONLY
    # this conditional edge — no static out-edge — so exactly one node runs per super-step. jade is
    # absent here on purpose (it already ran in the spine and is never re-dispatched).
    g.add_conditional_edges("orchestrator", route_from_orchestrator, {
        "location": "location",
        "job_title": "job_title",
        "matching": "matching",
        "supervision": "supervision",
        "drafter": "agreement_retriever",
    })

    # spokes → hub (static return edges only; the orchestrator re-decides with the new findings)
    g.add_edge("location", "orchestrator")
    g.add_edge("job_title", "orchestrator")
    g.add_edge("matching", "orchestrator")
    g.add_edge("supervision", "orchestrator")

    # Terminal enrichment: governing authority is selected first; precedent then supplies the final
    # configured precedence fact and finalizes chargeability hooks before drafting.
    g.add_edge("agreement_retriever", "wiki")
    g.add_edge("wiki", "drafter")
    g.add_edge("drafter", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())


__all__ = ["build_otl_graph", "route_from_category_check", "route_from_verify_evidence",
           "route_from_jade_analyzer", "route_from_orchestrator"]
