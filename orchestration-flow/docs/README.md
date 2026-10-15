# OTL Orchestration Flow

Automates the drafting of operator **rebuttals** to Shell JV audit exceptions in the **"Labor (OTL, Affiliates, Payroll)"** category (OTL — *Offsite / affiliate Technical Labor* charges). Given one exception's evidence packet (the auditor's report + the workbooks they cited + the extractor's `synthesis.json`), the pipeline reconstructs the questioned amount, locates the disputed people, resolves their roles, classifies chargeability against the CFR settlement categories (and, for hierarchy disputes, walks the org tree), and emits either a **terminal rebuttal** or a **needs-evidence deferral note** for human review.

It is a [LangGraph](https://langchain-ai.github.io/langgraph/) state machine built as a **sequential hub-and-spoke LLM supervisor**: a deterministic prep spine runs first (ending in a hard evidence gate), then a highest-tier `orchestrator` node dispatches one specialist at a time until it can finalize.

---

## Table of contents

1. [Where it sits](#where-it-sits)
2. [Quick start](#quick-start)
3. [Architecture overview](#architecture-overview)
4. [The graph in detail](#the-graph-in-detail)
   - [Prep spine + hard gate](#1-prep-spine--hard-evidence-gate)
   - [Orchestrator hub](#2-orchestrator-the-hub)
   - [Specialist spokes](#3-specialist-spokes)
   - [Drafter](#4-drafter-the-sink)
5. [HITL & gate logic](#hitl--gate-logic)
6. [State model](#state-model)
7. [Model tiers & LLM provider (Azure vs ICA)](#model-tiers--llm-provider-azure-vs-ica)
8. [The ReAct loop & code sandbox](#the-react-loop--code-sandbox)
9. [The major-construction screen](#the-major-construction-screen)
10. [Synthesis passthrough](#synthesis-passthrough)
11. [Inputs & outputs](#inputs--outputs)
12. [Running it](#running-it)
13. [Worked example: exception 0003](#worked-example-exception-0003)
14. [Module map](#module-map)
15. [Environment & configuration](#environment--configuration)

---

## Where it sits

This layer is the **consumer** of the `excel_extractor` pipeline. The extractor reads each exception's raw workbooks and produces a `rebuttal_<cycle>_synthesis.json` — a structured, interpreted description of the exception (what the auditor claims, which attachments/tabs support it, the likely basis for the amount, open questions). This layer reads that synthesis, does the audit reasoning, and writes a sibling `rebuttal_<cycle>_otl.json`.

```
excel_extractor/  ──►  rebuttal_N_synthesis.json  ──►  orchestration-flow/  ──►  rebuttal_N_otl.json
   (reads workbooks)      (interpreted exception)         (audit reasoning)         (rebuttal / deferral)
                                                                  │
                                                                  └─► rebuttal_N_analysis.md
```

It **reuses** the application toolchain wholesale — the LLM client (`shared.llm`), token accounting (`shared.tokens.USAGE`), output sinks (`shared.output_paths`), and the extractor-owned corpus resolver (`extractor.case_resolver`). The `otl` package is importable only after a one-time bootstrap ([`otl/_bootstrap.py`](otl/_bootstrap.py)) puts the repository packages on `sys.path` and loads the repo-root `.env` first.

---

## Quick start

```bash
# From orchestration-flow/. Uses the repo-root .venv, writes output locally (never SharePoint).
OUTPUT_TARGET=local ./run_otl_cases.sh --year 2023 --exc 0003
```

- `run_otl_cases.sh` is a thin wrapper over the **parallel** batch runner (`run_batch.py`).
- `OUTPUT_TARGET=local` writes to `OUTPUT_BASE/<year>/<asset>/<exc>/rebuttal_0_otl.json` instead of uploading to SharePoint.
- Drop the filters to run every case in the CSV (any topic; add `--type "Labor Issues"` to keep to that topic); add `--jobs N` to raise concurrency (default 4).
- Add `--provider azure` to switch the LLM backend for the run (default is whatever `LLM_PROVIDER` is in `.env`); see [Model tiers & LLM provider](#model-tiers--llm-provider-azure-vs-ica).

See [Running it](#running-it) for the full flag reference and the serial vs. parallel distinction.

---

## Architecture overview

A deterministic prep spine runs first and ends in a **hard evidence gate**. If the gate passes, `jade_analyzer` reconstructs the questioned amount (needed for every case) and hands off to a highest-tier `orchestrator` node, which becomes the **hub**: on each pass it dispatches exactly one specialist (with a custom instruction) until it can finalize.

```
          ┌──────────────────────── deterministic prep spine ────────────────────────┐
 START ─► intake ─► rebuttal_history ─► category_check ─► verify_evidence
                                                                     │
                          route_from_verify_evidence (HARD GATE)     │
                     decision.triggered ─────────► agreement_retriever ─► wiki ─► drafter
                                     else       └─► jade_analyzer ─► orchestrator
                                                          │               │
                            route_from_jade_analyzer ─────┘               │
                            (no readable ledger ──► wiki)                 │
     ┌──────────────────────────────────────────────────────────────────┤ (route_from_orchestrator
     ▼          ▼            ▼             ▼                    ▼          │  reads next_agent)
 location   job_title    matching    supervision             agreement_retriever ─► wiki ─► drafter ─► END
     │          │            │             │
     └──────────┴────────────┴─────────────┘  (static return edges → orchestrator re-decides; each spoke runs at most once)
```

Design rules that make this safe and predictable:

- **Three edges are content-driven; everything else is static.** `route_from_verify_evidence` (the configured evidence decision), `route_from_jade_analyzer` (an operational analysis-block guard when verification was bypassed), and `route_from_orchestrator` (which just reads the `next_agent` string the orchestrator already wrote). So exactly one node runs per super-step and there are no concurrent-write conflicts.
- **`jade_analyzer` is the last spine node, not a spoke.** The questioned-amount reconstruction is needed for every case that clears the gate, so it always runs — statically, right before the hub. The orchestrator therefore only ever chooses among `{location, job_title, matching, supervision, drafter}`.
- **The orchestrator owns every routing decision, and never revisits a node.** It dispatches *one* specialist per hop (with a custom instruction), each specialist **at most once** (the single-visit rule), and it alone chooses when to finalize and whether the outcome is `terminal` or `needs_evidence`. "Location **and** job_title" is two successive hops — and the second instruction can be informed by what the first specialist found.
- **Dispatch is category-aware.** The orchestrator only offers specialists *relevant to the dispute the auditor actually raised* (from `category_check`), so e.g. `location` is never dispatched for a pure job-title question.
- **Agreement retrieval is authoritative terminal context.** Before precedent review, `agreement_retriever` uses the original auditor exception as its primary query and additive specialist findings as a second Shell-theory query. It searches only documents registered for the canonical asset, reranks candidate IDs, reloads exact chunks from SQLite, and supplies the combined CFR/COPAS authority fact used by configured chargeability hooks.
- **The wiki remains a raw-text sidecar but supplies one structured fact.** After agreement retrieval, its index-first lookup records `precedence_status`, finalizes chargeability hooks, and routes to the drafter. Raw precedent pages remain excluded from drafter prose. An empty or unavailable wiki records `unavailable_or_unclear` rather than inventing precedent support.

Specialists never block; they return a non-blocking `agent_status` (`objective_met` | `inconclusive`) and hand control back to the hub. Only the hard gate and the orchestrator's deterministic guards can force a `needs_evidence` outcome.

---

## The graph in detail

Defined in [`otl/graph.py`](otl/graph.py). Nodes are resolved **progressively**: `build_otl_graph()` imports `<name>_node` from `otl.agents.<module>` if it exists, else substitutes an inline pass-through stub — so the graph always compiles and can be brought up phase by phase.

### 1. Prep spine + hard evidence gate

Runs once, in order, before the hub. Never loops.

| Node | Tier | What it does |
|---|---|---|
| **intake** | light | Captures supported assets and the requested amount, while `collected_evidence` is populated deterministically from response-filtered classified attachments. It does not identify people. |
| **category_check** | light | Identifies analytic questions plus distinct request categories for attestation, third-party invoices, and other additional support. A third-party invoice request is not also classified as generic additional support unless a separate information request remains. This drives both the gate and category-aware dispatch. |
| **verify_evidence** | med | **Detects evidence facts + executes configured ED hooks.** The med model reads only row-free cited-tab metadata and detects POB, heli, and hierarchy evidence. Code profiles the JADE/general-ledger manifest, applies the category_check third-party-invoice bucket, detects current/prior Shell invoices, and evaluates the rules configured directly under `labor.hooks.verify_evidence` in `configs/hooks.yaml`. Applicable failures aggregate into one typed decision. |

Then `route_from_verify_evidence`: a triggered configured decision short-circuits to the drafter; a third-party-invoice-only case that passes goes to the drafter's normal information-request path; an analytic case that passes continues to `jade_analyzer`. Generic additional-support-only cases route directly to the drafter without verification. Disabling the node in `configs/pipeline.yaml` bypasses every configured ED rule.

| Node | Tier | What it does |
|---|---|---|
| **jade_analyzer** | med | The last spine node (ReAct + code). It settles the grounded person list, reconstructs the questioned amount, and runs the major-construction screen. After a finished reconstruction, code recomputes the selected sum/half comparison using the inclusive ±2.5% rule; configured ED-02 mismatches route directly to the drafter, while passing or disabled-hook cases continue to the orchestrator. |

Then `route_from_jade_analyzer`: a normal run continues to the orchestrator. If verification was bypassed and JADE cannot operate, `analysis_blocked=True` routes to an operational `needs_evidence` HITL without recreating an evidence-rule denial.

### 2. Orchestrator (the hub)

[`otl/agents/orchestrator.py`](otl/agents/orchestrator.py), **high tier**. The hub between the spine and the drafter — reached *after* `jade_analyzer` has run, so the questioned amount is already reconstructed. On each pass it applies five **deterministic guards in code before the LLM is consulted**, then (if none fire) asks the supervisor LLM for the next hop:

1. **Loop guard.** Counts its own passes (`orchestrator_hops`) and each specialist's dispatches (`agent_run_counts`). Hitting `MAX_HOPS` (50) or `MAX_AGENT_RUNS` (50) forces a graceful finalize to the drafter with `final_mode="needs_evidence"` and a loop-guard `hitl_reason` — the LLM is **not** consulted, so the graph can never infinite-loop. (With the single-visit rule, the real bound is a handful of hops; these caps are just a backstop, and `recursion_limit=150` in `run.py` is the engine-level seatbelt above them.)
2. **No-dispute-category guard.** If `category_check` found no dispute category (empty `auditor_question`), there is no routing path — finalize `needs_evidence` asking a human to confirm which data sources to check (never a blind dispatch-all).
3. **Technical-role override (Scenario 1, Request H).** If `matching` has run but classified no CFR category **and** `job_title` established the role is technical in nature → finalize **`terminal`**: a technical role is chargeable by *function*, not location (POB is not dispositive).
4. **Non-technical-role defer (Scenario 2).** If `matching` has run with no CFR category **and** `job_title` found the role is **not** technical → finalize `needs_evidence`: the AI cannot determine whether the duty is defensible as technical, so a human must confirm. (Mutually exclusive with #3, which is checked first.)
5. **Location-only ⇒ `job_title` always follows `location`.** `invoke_job_title_after_location` is `True` **iff** `category_check` identified **only** `location` — no other category, in any combination. When it is `True` and `location` has run, the orchestrator **dispatches `job_title` itself** (the LLM is not consulted on that hop): a location-only dispute cannot be closed on physical presence alone, because a role that is *technical in nature* is chargeable by **function**, not location. This is the only guard that **dispatches** rather than finalizes, and it fires at most once (the single-visit check makes it a no-op afterwards). The flag is derived from `auditor_question` on every read — never stored as an independent source of truth — and mirrored onto `orchestrator.invoke_job_title_after_location` for the trace.
   `False` for every other combination, which **does not** mean `job_title` is skipped: whenever `job_title` is one of the categories it stays in `available_specialists` and the supervisor dispatches it as usual. The asymmetry is deliberate — there is no `job_title ⇒ location` equivalent, since whether a role is technical does not depend on where the person was.

If no guard fires, it builds a **compact, row-free decision surface** (`_digest_for_llm`, capped at 12 KB): case header, `auditor_question`, trimmed `auditor_narrative`, the extractor's interpretive `synthesis` narrative, the `verify_evidence` flags, `open_questions`, every specialist's `agent_status`, `available_specialists`, and a salient subset of each specialist's typed findings. It **never** includes the full synthesis payload or raw workbook rows. Then it asks the supervisor LLM (`_OrchestratorDecision`) to either:

- **dispatch** one of `available_specialists` with a focused `instruction` (merged into `agent_instructions[target]`), or
- **finalize** to `drafter` with `final_mode ∈ {terminal, needs_evidence}` (+ `hitl_reason`/`hitl_stage`/`hitl_evidence_trail` when needs_evidence).

**Category-aware dispatch.** `available_specialists` = the specialists *relevant to the dispute category(ies)* that have not yet run (`_CATEGORY_TARGETS`):

| Dispute category | Specialists offered |
|---|---|
| `job_title` | `job_title`, `matching` (resolve the role, then defend on the OTL/CFR basis; **not** `location` — POB isn't dispositive for a role question) |
| `location` | `location`, `job_title`, `matching` (place them physically; fall back to "were they OTL at all?" — and when `location` is the **only** category, `job_title` is dispatched automatically after `location`, per guard #5) |
| `hierarchy` | `supervision` (above/below first level from the org tree) |
| empty/unknown | *none* (handled by the no-dispute-category guard) |

**Single-visit enforcement.** If the LLM picks a specialist that has already run (or a garbage/unknown token, or spine-only `jade_analyzer`), the node does **not** route to it — it finalizes instead, choosing the mode via `_infer_final_mode` (`terminal` iff every specialist that ran, jade included, reported `objective_met`, else `needs_evidence`). Combined with advertising only the remaining relevant targets, this guarantees the graph never visits the same node twice.

Every decision appends to `orchestrator.decision_log` (`{hop, chose, rationale, status_seen, [final_mode]}`) and emits a trace event.

### 3. Specialist spokes

Each runs when dispatched (**at most once**), returns to the hub via a **static** edge, and reports a non-blocking `agent_status`. Each reads its `agent_instructions[<node>]` directive and the reviewer `open_questions` as **focus hints, never gates**.

| Node | Tier | Kind | Objective |
|---|---|---|---|
| **location** | med | ReAct + code | Where were the people physically? Cross-references POB + heli logs from the Common Reports dir. Both files are read off disk and their names resolved **before** the loop starts (see [the person index](#finding-those-people-in-the-corpus-workbooks-the-person-index)), so the loop reasons over pulled rows instead of screening surnames. Reports findings **only** — it no longer emits any flag about what should run next (the old `invoke_job_title_agent` is gone; see `invoke_job_title_after_location` below). |
| **job_title** | med | deterministic packet + structured shards; code fallback | Stream the audit-year Common Reports staff listing once, collapse monthly rows into per-person title periods, index the active `Job Catalogue` worksheet, and apply the audit-year/asset IPT registry. A bounded structured call returns technical/non-technical status per person; only unresolved people escalate to code execution. |
| **matching** | med | normalized CFR index + structured shards; code fallback | Collapse the 59k-row CFR workbook into distinct profile/title mapping candidates, then use the job-title positions, profiles, summaries and descriptions to select CFR Primary/Secondary/Additional, subpart labels, rationale and source references per person. Matching reports regulatory evidence only; the asset-specific chargeability of offsite technical labor is deferred to agreement retrieval. Only unresolved people escalate to code execution. |
| **supervision** | med | deterministic org tree; code fallback | Fired for a `hierarchy` dispute. Reuses the cached audit-year staff scan, normalizes monthly employee→manager snapshots, and classifies each person without an LLM: above first level iff they have direct reports and at least one report has reports. Conflicting monthly levels, multiple employee keys, missing records, or unsupported schemas escalate only the affected people to code execution. |

(`jade_analyzer` is the same kind of ReAct + code specialist but runs in the **spine** — see the prep-spine table above.) `objective_met` iff the ReAct loop `finished` (matching: iff a category or technical basis was established); otherwise `inconclusive` with the reason. Because a specialist is never re-dispatched, an `inconclusive` result is weighed as-is when the orchestrator decides its next hop or finalizes.

### 4. Drafter (the sink)

[`otl/agents/drafter.py`](otl/agents/drafter.py), **high tier**. The single terminal node, running in one of three modes:

- **EVIDENCE-GATE** (`evidence_gate_decision.triggered`, checked first) — a configured ED rule fired *before* any analysis ran (jade/orchestrator/specialists all skipped). Missing support yields a concise `deny` naming the absent sources. If supplied material cannot be opened or accessed, the same shortcut produces `needs_evidence` with `hitl_review = True` and requests an accessible copy without exposing internal processing detail.
- **TERMINAL** (`final_mode == "terminal"`) — draft the operator's rebuttal grounded **only** in the distilled specialist findings: the JADE reconstruction, POB/heli location, resolved title, the CFR category with its clause quoted verbatim, and (for hierarchy) the supervision credit/chargeable split. Disposition ∈ `{deny, partial, accept}`, plus titled `response_sections` and a concise `co_owner_response`.
- **HITL** (`hitl_review` / `final_mode == "needs_evidence"`) — human judgment, clarification, operational follow-up, or accessibility resolution remains required. The orchestrator may have run without reaching a defensible conclusion, or the evidence gate may have found supplied material inaccessible. Do **not** fabricate a conclusion. Write a deferral note with three labelled parts: (a) findings so far, (b) why processing stopped, (c) what evidence/judgement is missing. Disposition = `needs_evidence`, `hitl_review = True`.

It never opens a workbook and never invents a figure — it reasons only over the sub-states and `nl_summaries_by_node` handed to it, and quotes the CFR clause verbatim from `matching`.

---

## HITL & gate logic

`needs_evidence` is always a HITL/deferred outcome. Missing auditor support is a terminal denial:

| Who owes something | `disposition` | `hitl_review` | What ships |
|---|---|---|---|
| the **auditor** did not supply required usable support | `deny` | **`False`** | a concise auditor-facing denial naming the missing POB, helicopter, or ledger source. |
| supplied evidence cannot be accessed, or a **human reviewer** owes a judgement | `needs_evidence` | **`True`** | an accessibility or analytical deferral with `hitl_reason`, `hitl_stage`, and the technical evidence trail. |
| nobody — the case is answered on the merits | `deny` \| `partial` \| `accept` | **`False`** (written explicitly on terminal finalize) | the drafted rebuttal. |

So filter the batch on `hitl_review` to find the reviewer's queue. `run.py` reports terminal and HITL buckets separately at the end of a batch.

**1. Configured evidence hooks (`verify_evidence` → `route_from_verify_evidence`).** Rules live directly under `labor.hooks.verify_evidence` in `configs/hooks.yaml`; `configs/pipeline.yaml` remains the sole node-level switch:

| Rule | Condition |
|---|---|
| ED-01 | requires a readable JADE or general-ledger tab |
| ED-03 | a `location` question requires POB **and** helicopter-manifest/flight-log evidence |
| ED-04 | a specific third-party-invoice request requires a current or prior structured Shell/SIEP invoice finding |
| ED-05 | a `hierarchy` question requires hierarchy-level evidence for the questioned personnel |

Applicable failures aggregate in ED order and deduplicate their evidence labels. Missing evidence denies and ends with “No credit is due.” Concretely unreadable evidence takes precedence and produces `needs_evidence` plus HITL. Each ED entry has its own `enabled` switch; disabling the node bypasses all four.

**1b. JADE reconciliation hook.** After a finished reconstruction, ED-02 deterministically recomputes whether the selected sum/half amount is within the inclusive ±2.5% tolerance of the auditor's amount. A mismatch routes directly to a denial stating that JADE does not reconcile and that no credit is due. If verification was disabled and JADE cannot operate because no usable ledger exists, that remains an operational `needs_evidence` HITL rather than an ED-01 denial.

**2. The orchestrator's deterministic guards** (see [Orchestrator hub](#2-orchestrator-the-hub)): loop guard, no-dispute-category, and non-technical-role defer (Scenario 2) all finalize `needs_evidence`; the technical-role override (Scenario 1) finalizes `terminal`.

**3. The orchestrator's LLM finalize** — when the supervisor judges a real gap blocks any defensible conclusion, or the single-visit rule forces a finalize and not every specialist reported `objective_met`.

For true HITL cases, including supplied evidence that cannot be accessed, the drafter writes a deferral note rather than a conclusion, and `hitl_stage` / `hitl_reason` / `hitl_evidence_trail` are preserved in the output for the reviewer. The orchestrator owns analytical deferrals; the evidence gate may also raise HITL for inaccessible supplied material. Terminal finalization writes `hitl_review=False` explicitly, so no stale value can flip a rebuttal into a deferral note.

> **Note:** some exceptions legitimately end `needs_evidence` — that is the correct outcome for their evidence, not a bug to "fix".

### Durable rebuttal-cycle analysis

After intake and before summarization, `rebuttal_history` reads every earlier
`<year>/<asset>/<exception>/rebuttal_<N>_analysis.md` in cycle order. The active cycle is excluded so a
forced rerun never consumes its own stale output. Released Shell response prose is removed before the
history becomes prompt context. The combined analytical history is injected into reasoning prompts as
continuity context, with an explicit instruction to re-verify current evidence.

The history node compares the fresh intake/current evidence with prior analysis and emits a
per-specialist cycle-delta matrix. A previously completed specialist is suppressed only when no new
fact, question, person, period, evidence item, amount, role, or authority theory affects its scope.
Uncertain assessments fail open. The orchestrator receives the routing matrix rather than one
undifferentiated history block; each specialist receives only its cherry-picked prior findings and its
separately tagged current questioning ground. The orchestrator may add any other bounded current-case
dependencies or findings it considers necessary through the specialist directive channel.

The files are read locally and token-estimated before any history is sent to an LLM. If the sanitized
history estimate reaches 85% of the smallest configured recipient context window, the medium-tier
`rebuttal_compaction` call compacts it in ordered chunks sized below the compactor's own window. The
high-tier `rebuttal_history` cycle-delta assessment remains separate. Compaction
retains each agent's main finding, material uncertainty, disposition relationship, amount and changes
between cycles while dropping Shell response prose, raw rows, traces, long inventories and repetition.
The result is a versioned snapshot cached by source digest, context window and threshold. An unchanged
history therefore makes no new compactor call; a new or edited cycle creates one new snapshot rather
than compacting a previous summary again and accumulating summary drift.

The active `rebuttal_<N>_analysis.md` uses marker-delimited upserts, not unbounded appends. Repeated
visits replace that agent's section instead of duplicating it. Main findings are whitespace-normalized
and bounded, long strings and collections are capped with explicit omission counts, and each agent has
an overall detail budget. The file therefore stays direct, parseable and auditable even when a case is
retried or a workbook contains a large person-level map.

The cases CSV lists exceptions only. The runner reads the available cycle numbers for each exception from
`datastores/classified_exceptions.json` and runs every available cycle through the selected ceiling,
including Rebuttal 0. An explicit `--rebuttal-cycle N` sets that inclusive upper bound. The exception expands
into every available cycle through N, deduplicates repeated cases, and runs those cycles chronologically.
Different exceptions remain parallel, but cycles of the same exception run sequentially; a failed cycle stops
its later dependent cycles. Each cycle gets its own
`rebuttal_<N>_analysis.md`, `rebuttal_<N>_otl_trace.jsonl`, and `rebuttal_<N>_otl.json`. The classified
sub-JSON is cumulative, so the active cycle records only its last Auditor-role reply as the new event; prior
auditor replies and pipeline-generated responses arrive from the earlier markdown files. Classified
Operator-role replies are never admitted.

`--limit N` counts unique exceptions, not CSV rows or rebuttal cycles. Every selected cycle for each of
the first N exceptions is retained and expanded chronologically.

---

## State model

Defined in [`otl/state.py`](otl/state.py). Two layers:

**Per-agent sub-states** — one Pydantic model per node (`IntakeState`, `JadeState`, `SupervisionState`, …), each carrying a bounded `nl_summary: str` analysis plus that node's typed outputs. A node returns its sub-state under its own key. Because each key has exactly one writer, LangGraph's default last-write-wins channel is correct — no reducer needed.

**Master `OTLAuditState`** — a flat `TypedDict(total=False)` holding every sub-state plus the shared channels. The channels that matter for orchestration:

| Channel | Reducer | Writer | Purpose |
|---|---|---|---|
| `evidence_gate_decision` | LWW | verify_evidence | aggregate ED result; `triggered=True` routes directly to the drafter |
| `analysis_blocked` | LWW | jade_analyzer | operational fallback after verification bypass; `True` routes to HITL |
| `evidence_gap` | LWW | jade_analyzer | what is missing, in the drafter's own words. Set only on the jade half of the gate; it picks the wording of the needs-evidence response so it names the ledger detail rather than POB records. |
| `auditor_question` | LWW | category_check | dispute category(ies) — drives the gate, category-aware dispatch, and the derived `invoke_job_title_after_location` |
| `next_agent` | LWW | orchestrator | the per-hop routing decision the edge reads |
| `final_mode` | LWW | orchestrator | `terminal` \| `needs_evidence` — drives the drafter |
| `orchestrator_hops` | LWW | orchestrator | loop-guard counter |
| `agent_run_counts` | LWW | orchestrator | per-specialist dispatch counts (loop guard + single-visit) |
| `agent_instructions` | **`merge_dict`** | orchestrator | per-specialist custom directive; accumulates across hops |
| `agent_status` | **`merge_dict`** | each specialist | `{status, reason}` keyed by agent |
| `hitl_review` / `hitl_reason` / `hitl_stage` / `hitl_evidence_trail` | LWW | orchestrator / drafter | the HITL deferral trail. `needs_evidence` always sets `hitl_review=True`; unsupported auditor evidence is recorded as `deny`. |

`agent_status` **requires** `merge_dict`: specialists write in different super-steps, so last-write-wins would clobber earlier statuses. The trace channels also use reducers so appends accumulate: `messages` (`add_messages`), `trace` (`operator.add`), `nl_summaries_by_node` (`merge_dict`).

`SUBSTATE_KEYS` lists the sub-state keys in serialization order (`intake, category_check, verify_evidence, orchestrator, jade_analyzer, location, job_title, matching, supervision, drafter`); the output writer walks it, so adding a node needs no writer change.

---

## Model tiers & LLM provider (Azure vs ICA)

[`otl/llm_adapter.py`](../otl/llm_adapter.py) maps each node to one of three tiers and forwards to the shared LLM client ([`shared/llm.py`](../../shared/llm.py)), explicitly passing `HOOK_MODULE = "labor"`. The shared client wires token accounting, hook interception, and JSON repair. Agent code just calls `llm.structured(node, system, user, Schema)`.

| Tier | Nodes | ICA model | Azure model |
|---|---|---|---|
| **high** | `orchestrator`, `drafter` | claude-opus-4-8 | gpt-5.4-pro |
| **med** | `jade_analyzer`, `location`, `job_title`, `matching`, `supervision`, `verify_evidence`, `person_index` | claude-sonnet-4-6 | gpt-5.4 |
| **light** | `intake`, `category_check`, `wiki`, `agreement_retriever`, `jade_people` | claude-haiku-4-5 | gpt-5.4-mini |

Two entries are not graph nodes. `jade_people` is the person-name extraction `jade_analyzer` runs before its loop; `person_index` is the name-selection pass `location`, `job_title` and `supervision` run before theirs (see [the person index](#finding-those-people-in-the-corpus-workbooks-the-person-index)). Each has its own key so its tokens are attributed to the tier it really runs on rather than to the calling node's tier.

The five specialists capable of **code escalation** (`jade_analyzer`, `location`, `job_title`, `matching`, `supervision`) can all be re-pointed to a single tier in one shot via `--code-exec-tier {high|med|light}` (or `OTL_CODE_EXEC_TIER`). JADE, location and job title use structured/deterministic normal paths; code is their fallback.

### Switching provider

The backend is selected by **`LLM_PROVIDER`** (`ica` default, or `azure`), read at import in `llm.py`. Set it in `excel_extractor/.env`, or per-run with `--provider` — accepted by all three entrypoints:

```bash
./run_otl_cases.sh --provider azure --year 2023 --exc 0003
../.venv/bin/python3 run_batch.py --provider azure --year 2023
../.venv/bin/python3 run.py --provider azure --case-id GreatWhite_2023_0003
```

Because the provider is frozen when `llm.py` is imported (and `otl` imports it), the env var alone is too late for a flag: the Python entrypoints call `llm.set_provider()`, which reassigns the module's provider globals *and* re-exports `LLM_PROVIDER` so every spawned case subprocess inherits it. The shell wrapper exports it before Python starts, reaching the same state.

Tier models resolve as `<PROVIDER>_MODEL_<TIER>` → the provider default in the table above. So `AZURE_MODEL_*` and `ICA_MODEL_*` can both live in `.env`; the flag picks one set. See [`excel_extractor/.env.azure.example`](../excel_extractor/.env.azure.example) for the full Azure block.

Check what a switch actually resolved to **before** spending a batch on it — this prints the provider, and per tier the model, Foundry v1 endpoint, and auth *mode* (never the credential), plus the node→tier map. It makes no network call and runs no case:

```bash
./run_otl_cases.sh --provider azure --llm-config
```

**Azure AI Foundry specifics:**
- **One endpoint, three tiers:** `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY` apply to every tier. Only `AZURE_MODEL_{HIGH,MED,LIGHT}` differs by tier.
- **Endpoint shape:** the shared client appends `/openai/v1/` to the configured Foundry project endpoint and sends standard OpenAI-compatible Requests/Responses API calls. Do not add `/openai/v1` yourself.
- **Auth:** API key (`AZURE_OPENAI_API_KEY`).
- **Model names:** each tier's `AZURE_MODEL_<TIER>` value is sent as the model name shown by Foundry. The deployment/model version shown in Foundry is not a REST `api-version` setting.
- **Responses API:** every Azure tier is sent through the Responses API; Foundry v1 does not use a dated `api-version` query parameter.

### Automatic Azure → ICA fallback

When `LLM_PROVIDER=azure` and an Azure request fails, the process retries that request on ICA and uses
ICA for the rest of the case. This automatic fallback has no separate configuration switch; it engages
only when `ICA_API_KEY` is configured. Context-overflow errors do not trigger a provider switch.

> **gen-5 note:** the current-generation reasoning models reject `temperature=0` — the client omits the parameter rather than passing zero.

---

## The ReAct loop & code sandbox

The data-exploration specialists (`jade_analyzer`, `location`, `job_title`, `matching`, `supervision`) cannot fit raw workbook rows in context, so they **write Python** that opens the workbook, computes a summary, and returns only the result. Two modules make this work:

**[`otl/react_loop.py`](otl/react_loop.py)** — a reason→act→observe loop run *inside* a single graph node:
- A fresh message buffer is created on entry, resent in full to the model each turn, and **discarded** when the loop returns. Only the distilled `result` + bounded `findings` cross into shared state.
- The model's only tool is `run_python`. Each turn is one JSON action: `{"action": "run_python", "code": …}` or `{"action": "finish", "result": {…}}`.
- **Termination:** a `finish` sentinel is the only early exit; `MAX_STEPS` (`OTL_MAX_STEPS`, default 50) is the sole backstop; each `run_python` has a ~150s timeout. A code error is fed back and the loop keeps trying.
- **Checkpointing:** the protocol tells every agent to `save_result(...)` its **cumulative** findings as soon as anything is established (e.g. `job_title`/`supervision` checkpoint after each person is resolved), not just at the end. Exhausting `MAX_STEPS` returns `status="max_steps"` carrying the **last checkpoint**, which the node reports as a non-blocking `inconclusive` — so an agent that resolves most of its work but never reaches `finish` no longer loses all of it.
- Every step is appended to the live JSONL trace so a UI can watch the reasoning; the message buffer itself is never persisted.

**[`otl/code_exec.py`](otl/code_exec.py)** — runs that code in a child process (`sys.executable`, `shell=False`, hard timeout):
- **Read scope** rooted at `DATA_ROOT` (env-overridable via `OTL_DATA_ROOT`) plus the case's explicit read roots. A preamble helper `open_data(path)` asserts the path is inside an allowed root — a *soft* guard that raises a clean, in-loop error the agent can react to.
- **Sheet access.** Operator-response tabs are removed inside the sandbox, not merely discouraged in the prompt: `openpyxl.load_workbook`, `pandas.read_excel` and `pandas.ExcelFile` (`.sheet_names` / `.parse`) are all patched, so a hidden tab is structurally absent from every view. `list_sheets(path)` is the cheap readable-sheet view built on that filter, and `PROTOCOL` names it to the model. The preamble itself is ~3,300 tokens and is **never** sent to the model — it is prepended to `script.py` at run time, so the model only ever gets a few lines of prose describing the helpers.
- **Write side** confined to `orchestration-flow/.otl_scratch/<run_id>/<agent>/` (or `$TMPDIR`): `script.py`, `inputs.json`, `result.json`, plus the two read-only payloads the node staged for it — `manifest.json` (workbook shape) and `person_index.json` (the resolved names, rows and column roles, see [the person index](#finding-those-people-in-the-corpus-workbooks-the-person-index)), reached from code as `manifest()` and `person_index()`. Nothing is ever written under `Data Files`. `result.json` is unlinked before each run so a stale payload can't leak, and each staged payload is unlinked when its node passes none, so an edited workbook cannot leave a stale index behind.

> This is a **trusted-local** sandbox (the design requires local `pip install`), **not** an OS-hardened jail — no network isolation. Run it only on evidence you trust.

### Who gets examined: the person list

`intake` extracts explicitly named people as leads. **`jade_analyzer`** resolves the analysis population: it reads the cited JADE tab's name column straight off disk — the `Employee` value per row, falling back to `Line Item Text Orig` on rows where `Employee` is blank or "Not assigned" — then sends those values, local POS suggestions, and intake leads through a **light-tier** call (`jade_people`, system prompt `JADE_PEOPLE_SYSTEM`). The resolved list is stored once as `jade_analyzer.people_found_in_ledger`; `people_named_count` is derived as its exact length, and every specialist derives its population through `_common.disputed_people`.

If verification was bypassed and JADE finds **no usable labor ledger**, there is nothing to reconstruct and nobody to name. It records `analysis_blocked=True` and defers operationally to HITL; it does not author an ED-01 denial.

#### Optional cap

Every ReAct specialist does **per-person** work over a large workbook — a staff-listing stream, a catalogue lookup per profile, an org-tree walk, a ledger filter. The pipeline therefore examines the entire grounded ledger population by default and lets the budget hook stop the run if its configured budget is exhausted.

[`_common.cap_people`](otl/agents/_common.py) dedupes first. It only keeps the first **`max_disputed_people()`** names when an operator explicitly sets a positive cap.

The dedupe compares **word sets**, not strings, because the two sources genuinely disagree on how to write a name: exception 0003's auditor wrote "Kenneth Oyler" and the ledger's line-item text carried "Oyler, Kenneth Willmann". Two names are one person when one's word set contains the other's — order, commas, middle names, initials and generational suffixes all stop mattering — and the first spelling is the one kept. "Adams, Nancy" and "Adams, Sheila" stay two people. The rule only fires when it is **unambiguous**: a bare "Adams" against both Adamses is kept as its own entry rather than guessed at, since guessing wrong would drop a real person from the rebuttal while a redundant entry only costs steps.

The optional bound is set by **`--max-people N`** on either runner. When the flag is absent, `MAX_DISPUTED_PEOPLE` is `0` and there is *no cap*; `0` or less explicitly has the same meaning. It is deliberately **not** env/`.env`-configurable — any cap a run used should be visible in the command that launched it, not in ambient state. Both runners print the effective value in their header line, so the run log always records what a given batch actually examined:

```bash
../.venv/bin/python3 run.py --case-id GreatWhite_2023_0003 --max-people 3 --force
./run_otl_cases.sh                       # no cap — examine everyone the ledger names
```

[`set_max_disputed_people`](otl/agents/_common.py) holds it in a process global, read at *call* time rather than at import. Because a global does not survive a fork, `run_batch.py` forwards `--max-people` on each worker's own command line (alongside `--code-exec-tier`) instead of relying on env inheritance the way `--provider` does.

It is applied in two places: **`jade_analyzer_node`**, where the final resolved population is settled, and **`_common.disputed_people(state)`**, which every later specialist reads so an explicitly configured bound remains enforced.

The model is *not* told to name fewer people. If an explicit cap bites, the pipeline derives the examined and overflow populations on demand from `people_found_in_ledger`, and:

- the trace action line reads `reconstructed amount (…); person list capped to X of N`, and the inputs digest shows `M names from K ledger value(s) → X examined`;
- the orchestrator's decision digest gains a derived `population_coverage` block, so the supervisor weighs partial coverage before finalizing;
- the drafter's prompt gains a **PARTIAL COVERAGE** block (`_coverage_note`, read off the *merged* list) instructing it to state the limitation, name the people not examined, and confine its conclusions to those examined — otherwise the rebuttal reads as though the whole charge was reviewed.

No duplicate person-count or cap-overflow fields are stored in intake or JADE state.

> A capped case still finalizes normally (`terminal` or a HITL deferral on its own merits) — the cap does **not** by itself force human review. If you'd rather every truncated case go to a reviewer, that's a one-line guard in the orchestrator.

### Finding those people in the corpus workbooks: the person index

Knowing a person's name is not the same as finding them in the data. Three specialists have to locate their people inside the year's corpus workbooks: `location` in the POB report and the helicopter manifest, `job_title` and `supervision` in the consolidated staff listing. Those files are large — 32k–48k rows for POB, 80k–224k for heli, 91k–468k for the staff listing — and the names in them are written differently from the names the auditor wrote.

The specialists used to do that search **themselves, from inside the ReAct loop**. The model wrote a surname screen, streamed the whole workbook, read the hit count, wrote the next screen, and streamed it again — once per person. Every one of those steps re-sent the entire message buffer at the specialist's tier. A stored trace shows the work being done by hand (`.otl_scratch/GreatWhite_2023_0033/location/result.json`):

> "FOULTZ = 0 hits (POB and heli); PRITCHARD = 0 hits (only KYLE PRICHARD …); WILLIAMS = 260 POB rows / 858 heli rows across 48+ distinct first names, none 'DONELL' … HARRISON = 35 POB rows across 8 first names, of which JOHN HARRISON (5 rows) is the only first-name match."

[`otl/person_index.py`](otl/person_index.py) moves that work out of the loop. Each file is read **once, off-LLM**, before the loop starts; the **full** distinct-name list goes to a **med-tier** model in one chunked pass that asks which values are the disputed people; then the matched rows are pulled deterministically. The specialist is handed confirmed spellings, their rows, and the **names of the columns those rows came from**. It stops searching and starts reasoning.

This is the same shape as two things already here: [`workbook_manifest.py`](otl/workbook_manifest.py) reads workbook *shape* off disk before the loop, and [`ledger_people.py`](otl/ledger_people.py) harvests values deterministically and then asks a cheap model to read names out of them. The person index is that second pattern applied to name *selection*, at med tier.

**Three phases, kept apart on purpose.** The read is a reproducible fact about the file; only name identity needs judgement.

| Phase | What it does | Model |
|---|---|---|
| 1. `harvest_names` | Walks the manifest the node already built, classifies each tab's columns, then reads every distinct name off every name-bearing tab in **one traversal per workbook**. No row cap, no cap on distinct values — a value left behind is a person left out of the rebuttal. | none |
| 2. `select_names` | Unions the distinct values across every tab of one source, chunks them to fit a character budget so **every** value is seen exactly once, and asks which values are the disputed people. Runs the chunks on a small thread pool. | **med** (`person_index`) |
| 3. `pull_rows` | Re-reads each tab and buffers the rows whose name cell equals a selected spelling, emitting each row as `{column label: value}`. | none |

**Why the columns come back as roles, not labels.** The corpus does not agree with itself. The 2021 staff listing is three different schemas in one workbook — 18 name-bearing tabs in three families, which disagree on the two columns `supervision` depends on most:

| Name column | Tabs | Person key | Reports-to pointer |
|---|---|---|---|
| `Employee Reporting Name` | 6 | `Employee ID` | `Manager ID` |
| `Worker` | 6 | `Employee ID` | `Supervisory Organization - ID` |
| `Employee Name` | 6 | `Employee Number` (and `Employee ID` on one tab) | `Supervisor Number` |

Header rows are 0, 1 or 2 depending on the tab. Column sets move between years too: of the four single-tab staff listings, only the one in the **2022** folder carries a `Job Profile` column — 2023, 2024 and 2025 do not, and the 0033 trace had to discover that by reading ("the consolidated staff listing has no separate 'Job Profile' column, only 'Position'"). So the index returns each tab's columns as **`role → {label: column index}`** — `person_key`, `reports_to`, `manager_name`, `position`, `job_profile`, `period`, `org`, plus POB/heli attributes — and the prompt names the labels it actually found, along with the roles the tab does **not** carry. Nothing downstream hardcodes a column name.

Classification runs off an **explicit allow-list of exact labels**, because "contains the word name" is unusable here. On the 2021 tabs, these all contain a name-ish word and none of them is a person: `Headcount Name`, `Company Code Name`, `Personnel Area Name`, `Business area name`, `Employee Type`, `Employee Group`, `Employee Subgroup`. Those land in the **unclassified** bucket — neither used nor silently dropped, but **counted in the block**, so an agent that needs more knows to consult `manifest()`. A few name-ish labels do carry a real role and are claimed by it rather than left unclassified: `Sup Org Name` and `Master Cost Center Name` are `org` (organizational units, deliberately not people), and `Manager Name` / `Supervisor Name` are `manager_name` — while `Manager Email` and `Manager Position Text` are not.

Two kinds of tab are skipped outright, and every skip is **listed** with its reason. A tab with **no name column** — POB's `GOM` and `UNDEFINED`, the 2021 `Summary`, and the 2025 staff workbook's junk `Sheet1`, whose detected header row is really a row of data with a person's name in it. And a tab with **no header row** — POB's `Summaries` — because roles cannot be assigned positionally. The agent can still read a skipped tab raw.

**The agent learns the result three ways**, so it never has to guess:

1. **The prompt block** (`person_index_block`) — per source: the tabs, the header row, the name column and its kind, the roles found with their labels, the roles absent, the tabs skipped, and each person's selected spelling with a couple of sample rows.
2. **Rows keyed by column label** — `{"Location": "MARS", "Company": "SHELL", "Arrival Date": …}`, so in the common case no column index is needed at all.
3. **`person_index()` in the sandbox** — the complete selection, all rows, and the full column map, staged to the scratch dir as `person_index.json` beside `manifest.json`. This channel is part of the code-exec preamble, which never passes through the model, so it costs **zero tokens**.

Every bound is stated in the block when it bites, and the block shrinks as the person list grows. When nothing was selected for a person, the block says how many values shared their surname — **absence is a finding**, not a silent omission, which is what keeps the WILLIAMS near-miss universe in the record instead of dropping it. Unreadable workbooks are **listed with their error** for the same reason: two files in the 2023 corpus folder fail on every engine, and an agent handed an index that quietly omits one will conclude it does not exist.

The index is a **prior, not a guarantee**. Raw reads stay available, and the prompt tells the agent that if a read disagrees with the index it should trust the read and say so. The agent must also state which spelling it adopted for each person and why — that keeps the identity judgement auditable, which is what the `name_screens` field in the 0033 trace was doing by hand.

**Cost.** The universe pass is not free, and it is the price of sending the full name list rather than a pre-filtered guess. Measured over the corpus, per case:

| Source | Rows | Distinct names | Chunks (≈10k input tokens each) | One pass |
|---|---|---|---|---|
| POB (all years) | 32k–48k | 8.3k–9.6k | 4 | 1.5–2.5 s |
| Heli (all years) | 80k–224k | 12.8k–16.8k | 6–8 | 6–16 s |
| Staff, single tab (2022–2025 folders) | 91k–468k | 11.8k–83.7k | 5–38 | 4–28 s |
| Staff, 18 tabs (2021 folder) | 398k | **74.2k unioned** (104.4k summed per-tab) | 32 | 36 s |

So `location` costs about 11 calls (~110k input tokens) per case, and the staff listing 5–38 depending on the year. The worst single file is the **2024** folder's listing at 83.7k distinct names — 38 calls, ~377k input tokens. The 2021 union is the number the up-front profiling could not predict: unioning the three tab families gives 74.2k distinct against 104.4k summed per-tab, a **29% saving** over asking each family separately. `job_title` and `supervision` **share** one staff-listing pass per case — the index is cached on `(role, file stamps, people)` — so that cost is paid once, not twice.

What it removes is the exploratory high-tier loop those steps used to be. If a year's selection cost ever dominates, the levers are the chunk budget and the cache, not a deterministic pre-filter — a pre-filter would mean not showing the model every value, which is the one thing this pass exists to do.

**Checking it against the real corpus.** [`tools/inspect_person_index.py`](tools/inspect_person_index.py) builds the index over the actual data files and prints, per source and per tab, the header row, the name column, the full column-role map, the skipped tabs with reasons, rows, distinct names, the union, the chunk count and the wall time. Phase 1 only by default, so the deterministic half is checkable at **zero spend**:

```bash
../.venv/bin/python tools/inspect_person_index.py --year 2021 --source staff
```

```bash
../.venv/bin/python tools/inspect_person_index.py --year 2023 --all-versions
```

`--all-versions` indexes every version of a workbook instead of just the latest, which is how the 2023 folder's two structurally unreadable files get exercised — production picks their readable `_v2` siblings. Add `--llm` to run the selection pass and see the chosen spellings, the near-miss pools and the block size (this one spends med-tier tokens).

**Kill switch.** `OTL_PERSON_INDEX=0` returns an empty index with **no model call**, and the three nodes fall back to reading the workbooks inside the loop. That is also the baseline for a before/after comparison of `by_node` token totals.

---

## The major-construction screen

Labor on a **major construction project** is chargeable to the joint account. JADE carries no "JV Object Type Description" text column — only a **coded** `WBS:JV Object Type` (equivalently `JV Object Type`) value per row, a short code such as `C142` or `O143`. So `jade_analyzer` reads that code while it is already scanning each person's rows, inside the same ReAct loop, and looks it up against a fixed 14-code reference table (`configs/major_construction_jv_object_types.yaml`, loaded through `otl/jv_object_types.py`) — an exact dict-membership check the model's own sandboxed code performs, not a judgment call about the description text. No extra file read and no extra model call.

The screen is per **person**, not per case. For each person it records:

- `jv_object_type_by_person` — the code(s) it actually read, verbatim. This is the receipt behind the flag, so a reviewer can check the claim against what the ledger said.
- `major_construction_by_person` — whether any of that person's rows carry a code registered in the reference table.
- `major_construction_basis` — one line naming the column it read. When the cited ledger has no such column, this says that instead and both maps stay empty.

The two summary fields, `major_construction_people` and `major_construction_charge`, are derived **in code** in `_to_jade_state` from `major_construction_by_person`. The model cannot write them. A model that asserts the charge without a per-person reading to support it gets a `False`, and one that denies a charge its own reading supports gets a `True`.

The header spelling varies per case ("WBS:JV Object Type", "JV Object Type"), so the agent resolves the column semantically the same way it resolves the other ledger columns. Once resolved, the code lookup itself is exact: the model normalizes the code (strip whitespace, upper-case) and checks it against the reference table — it is told never to decide MCOH from the description text, an account description, a project name, or the auditor's narrative, and never to guess at a code the table does not list.

**How it reaches a decision.** The flag is **context, not a route**. Nothing in code branches on it. `_salient_nl_summaries` puts `major_construction_charge` in the orchestrator's digest on every pass, and adds the people and the ledger values it read only when the screen fired — a screen that fired for one person of five is not a whole-case basis. `ORCHESTRATOR_SYSTEM` carries the rule: MCOH labor is directly chargeable under the settlement's **Agreed OTL Definition and Billing Standards**, on a basis that turns on neither the CFR category nor where the person physically was, so don't finalize `needs_evidence` for want of a CFR category or a POB hit on a person the screen covered.

**How it gets quoted.** The drafter's `_authority_application` adds one sentence when the screen fired, pinned to `authority.OTL_MAJOR_CONSTRUCTION_PROVISION` — the Construction, Operating and Utilization Agreement's (COUA) Exhibit "C" §III, Paragraph 2.B.a, which charges direct to the Joint Account the salaries, wages, payroll burden and Personal Expenses of personnel performing technical services in connection with the design, construction and installation of a Major Construction project, regardless of location. This is a **different instrument** from the 2019 CVX GOM Settlement Agreement the rest of the registry cites, so the drafter names its own instrument and section rather than reusing the case's `governing_instrument` — and, when a technical/category sentence follows it, that sentence still names the GOM Settlement Agreement in full rather than saying "the same agreement." Because the MCOH sentence is composed in code and does not depend on `matching`, an MCOH case can be finalized straight off the ledger and still carry its authority. The registry entry (`otl-coua-major-construction-exhibit-c-3-2b` in `configs/otl_settlement_sections.yaml`) shares the registry's existing asset scope (Perdido, Silvertip, Great White, GW, Tobago).

The drafter sees the MCOH facts **only when the screen fired**. A `False` flag, or a basis line saying the ledger has no such column, is a fact *against* the operator's charge: the orchestrator weighs it, and `_nl_summary_digest` keeps it out of the response writer's prompt. The raw cell values never travel to the drafter at all — they are trace material, like `ledger_person_values`.

> **One limit of the context-only design.** The Scenario-2 guard (`_untechnical_role_defer`) is enforced in code and knows nothing about MCOH: once `matching` has run without classifying a category on a role that is not technical, it forces `needs_evidence`. So when MCOH is the basis the case will rely on and no CFR route looks likely, the orchestrator is instructed to finalize `terminal` on it rather than dispatch `matching` first for a category the case does not need. Making MCOH override that guard would be a deterministic check in `orchestrator.py`, alongside `_technical_role_override`.

> **A second limit: MCOH does not reach above-first-level supervision.** A person confirmed to be above first-level supervision is overhead, not chargeable, whether or not their rows carry MCOH — the same Agreed OTL Definition and Billing Standards that makes major-construction labor directly chargeable excludes above-first-level supervision as not technical labor. This is also enforced **contextually, not deterministically**: no new guard was added to `orchestrator.py` for it. `ORCHESTRATOR_SYSTEM` states the rule directly (a fired MCOH flag never overrides a supervision finding), and the supervisor is trusted to weigh the two facts together when both are present in state.

---

## Synthesis passthrough

The extractor's `synthesis.json` has a top-level `synthesis` **string** — the extractor's interpretive account of *how* the auditor built the exception (the amount layers, the affiliate named, the WBS/cost objects the charge was booked to). This is distinct from `auditor_description` (the auditor's own words, surfaced as `auditor_narrative`).

That narrative is the single richest orienting field, so it is threaded into **every** reasoning node via two helpers in [`otl/agents/_common.py`](otl/agents/_common.py):

- `synthesis_narrative(state, limit)` — the trimmed string (`""` when absent).
- `synthesis_block(state, limit)` — a labelled prompt block (`""` when absent), appended to each node's prompt as *orienting context, not a substitute for cited evidence*.

It reaches the spine (`intake`, `rebuttal_history`, `category_check`), `verify_evidence`, the orchestrator digest (as `extractor_synthesis`), and each specialist task. The full synthesis payload (raw rows, attachment structure) is **never** forwarded — only the interpretive string.

---

## Inputs & outputs

**Input** per case, resolved by [`otl/case_resolver.py`](otl/case_resolver.py) + [`otl/sharepoint_reader.py`](otl/sharepoint_reader.py):
- the extractor's `rebuttal_<cycle>_synthesis.json` (seeds `synthesis` + `auditor_narrative`);
- the auditor report PDF, the exception's JADE/POB/heli exhibits, the Common Reports dir (POB + heli logs), and the OTL Information dir (staff listing, `Jobs_Catalogue.xlsx`, and the `staff_listing_with_CFR_mapping` workbook).

**Output** per case, written by [`otl/output_writer.py`](otl/output_writer.py):
- **`rebuttal_<cycle>_otl.json`** — the `OTLExceptionReport`: case header, outcome mirrors (`disposition`, `identified_cfr_category`, `cfr_clause`, `hitl_*`), independent top-level `wiki_output`, top-level `agreement_output` with the exact selected chunks and provenance, the drafter's prose, the full per-stage `stages` map (ordered by `SUBSTATE_KEYS`; the bulky `synthesis_json` is stripped), `nl_summaries_by_node`, the `trace` array, and a token/time footer.
- **`rebuttal_<cycle>_otl_trace.jsonl`** — the append-only trace, streamed to a local path live during the run (tailable by a UI) and pushed through the sink afterward.

Both are written as **siblings** to the extractor's synthesis at `<year>/<asset>/<exc>/` and never clobber it.

---

## Running it

Two entry points, both resolving cases from `datastores/labor_classification_test_set_temporal_split.csv` by default. `run.py` is hard-filtered to `type="Labor (OTL, Affiliates, Payroll)"`; `run_batch.py` runs **every** case in the CSV unless you narrow it with `--type` (see below):

### Serial — [`run.py`](run.py)

One case at a time, **in-process**, graph built once and reused. Best for a single case or step-through debugging.

```bash
OUTPUT_TARGET=local ../.venv/bin/python3 run.py --year 2023 --exc 0003 --force
OUTPUT_TARGET=local ../.venv/bin/python3 run.py --case-id GreatWhite_2023_0003 --force
```

Flags: `--year --exc --case-id --exception-type --limit --csv --dry-run --force --provider {azure|ica} --code-exec-tier {high|med|light} --max-people N --llm-config`. `--topic` remains accepted as a legacy alias for `--exception-type`.

`--case-id` reads **no CSV at all** — `<Asset>_<Year>_<Exception>` is self-describing, so the entity folder is matched off disk and the topic pulled from `datastores/clustered_exception_tables.csv` (`--exception-type` overrides; `--topic` is still accepted as its legacy alias). `--csv/--year/--exc` therefore don't apply alongside it, and the `Labor Issues` filter is bypassed with a warning naming the case's real topic. `--csv` takes a bare filename (resolved under `datastores/`) and falls back to the default with a warning if it doesn't exist; the corpus root is not configurable.

### Parallel — [`run_batch.py`](run_batch.py) (via [`run_otl_cases.sh`](run_otl_cases.sh))

Runs many cases concurrently, **one `run.py` subprocess per case**. Parallelism is at the **process** level, not thread — because `shared.tokens.USAGE` is a process-global counter that `run.py` resets/reports per case, so threads would corrupt per-case token counts. Each subprocess gets its own `USAGE`, checkpoint thread, and sink.

The parent supervises the subprocesses: it extracts sibling zips once (serially, to avoid races), builds the sink once (warming SharePoint auth for reuse), and **owns the skip decision** — already-written cases are skipped without paying subprocess-startup cost (the subprocess is always passed `--force` since the parent already decided).

```bash
./run_otl_cases.sh                        # every case in the CSV, any topic, 4 at a time
./run_otl_cases.sh --type "Labor Issues"  # only that topic (slug match: spacing/case don't matter)
./run_otl_cases.sh --jobs 8               # up to 8 concurrent cases
./run_otl_cases.sh --year 2023            # only 2023
./run_otl_cases.sh --exc 0003             # only exception 0003
./run_otl_cases.sh --case-id GreatWhite_2023_0003
./run_otl_cases.sh --limit 3              # first 3 resolved cases
./run_otl_cases.sh --force                # re-run cases already written
./run_otl_cases.sh --dry-run              # list resolved cases and exit
./run_otl_cases.sh --provider azure       # LLM backend for this run: azure | ica (default: .env)
./run_otl_cases.sh --code-exec-tier high  # put every code-exec specialist on high (default: med)
./run_otl_cases.sh --max-people 3         # optionally cap disputed people examined per case (omitted or 0 = no cap)
./run_otl_cases.sh --provider azure --llm-config
                                          # print the resolved LLM config + node->tier map, then
                                          # exit. No network call, no case run.

OUTPUT_TARGET=local ./run_otl_cases.sh --jobs 4   # force local output
```

`--type` is the topic filter. Leave it off and the batch runs every case the CSV lists, whatever its topic; the header then prints how many of those cases are not `Labor Issues`, since the prompts and skills here are written for that topic and other topics give weaker results. `--case-id` bypasses `--type` with a warning naming the case's real topic.

The wrapper picks the repo-root `.venv` python, parses out `--provider` (exporting `LLM_PROVIDER` for the run + every case subprocess), and forwards all other flags to `run_batch.py` (which adds `--jobs N`, default 4, and accepts `--provider` itself). It no longer injects `--csv`: `run_batch.py` owns the default so there is one definition of it.

**One thing the parent has to divide, not just forward.** The [person index](#finding-those-people-in-the-corpus-workbooks-the-person-index) sends the full name list to the med tier in several concurrent calls, each carrying ~10k tokens of names. That pool lives *inside* each `run.py` child, so its size bounds one case and knows nothing about the other three — four children running four calls each puts sixteen in flight against a single deployment's quota, and no in-process object can bound that because nothing in-process crosses the fork. The environment does cross it (`env=os.environ`), so the parent splits a batch-wide budget of 4 across `--jobs` and writes each child's share into `OTL_NAME_SELECT_JOBS`. `--jobs 4` gives each child 1; `--jobs 1` gives it the full 4; the batch-wide total is the same either way. The header line says what it applied, and an operator who sets the variable themselves keeps their value. This is the same reason `--max-people` travels on the child's command line rather than as ambient state.

> **Local vs. SharePoint output.** `OUTPUT_TARGET=local` (default) writes to `OUTPUT_BASE/<rel path>`. `OUTPUT_TARGET=sharepoint` uploads to `SHAREPOINT_WRITE_FOLDER/<same rel path>` (falling back to local on a 401). In a sandbox where SharePoint egress is blocked, always force `OUTPUT_TARGET=local`.

Rebuttal analysis markdown is always persisted locally. Set `JV_AUDIT_REBUTTAL_ROOT` to give it a
separate root; otherwise it uses `OUTPUT_BASE`. `REBUTTAL_HISTORY_COMPACT_RATIO` overrides the default
`0.85` threshold when operational testing requires a different headroom policy.

---

## Worked example: exception 0003

`GreatWhite_2023_0003` (Kenneth Oyler) — a `location` dispute, the canonical happy path. Expected trace:

```
verify_evidence ─(gate pass)─► jade_analyzer ─► orchestrator ─► location ─► orchestrator ─► job_title
                                (spine)                                                        │
                        ┌──────────────────────────────────────────────────────────────────────┘
                        ▼
                  orchestrator ─► matching ─► orchestrator ─► drafter  (final_mode=terminal)
```

The gate passes (POB+heli present for the location question), `jade_analyzer` settles the person list and reconstructs the amount in the spine, then the orchestrator dispatches `location` (Oyler is on Proteus/Olympus — **not** the charged Great White), then `job_title` — dispatched **deterministically** by guard #5, since `location` is the only category (resolve his role + technical flag) — then `matching` (CFR chargeability), then finalizes `terminal` — each specialist exactly once. Expected end-state: `orchestrator_hops == 4`, `agent_run_counts == {location: 1, job_title: 1, matching: 1}` (jade is not counted — it ran in the spine), an `objective_met` per specialist, `disposition ∈ {deny, partial, accept}`, `hitl_review == False`.

Other outcome shapes:

- **Location dispute, missing POB/heli** — the hard gate fails and the run short-circuits to a concise `deny` response explaining that the exception lacks the required auditor support and naming the missing sources. No specialist runs.
- **No usable labor ledger cited** — `jade_analyzer` fails the gate itself and the run denies the unsupported exception, naming the missing charge detail. If supplied ledger material cannot be opened or accessed, the run instead defers to HITL with `needs_evidence`. Its ReAct loop never starts in either case.
- **Job-title dispute, non-technical role** — `job_title` + `matching` run; if no CFR category fits and the role isn't technical, the Scenario-2 guard finalizes `needs_evidence`. If it *is* technical, the Scenario-1 guard finalizes `terminal` (chargeable by function).
- **Hierarchy dispute** — only `supervision` runs; the drafter derives accept/partial/deny from its `credit_amount` / `chargeable_amount` split.
- **Loop-guard trip** — a non-converging case hits `MAX_HOPS`/`MAX_AGENT_RUNS` and finalizes gracefully to `needs_evidence` — no `GraphRecursionError`.

---

## Module map

```
orchestration-flow/
├── run.py                     serial single-case runner (in-process)
├── run_batch.py               parallel runner — one run.py subprocess per case (--jobs N)
├── run_otl_cases.sh           venv wrapper → run_batch.py (parses --provider / --code-exec-tier)
├── code_classifier.py         deterministic COPAS error-code lookup (YAML-backed)
├── copas_error_codes.yaml     the COPAS code → {category, family, tier} registry
├── evidence_inventory.py      evidence-packet inventory helper
├── requirements.txt
├── docs/                      design docs (ARCHITECTURE.md, DESIGN_DOC_OTL.md) — see note below
├── tools/
│   └── inspect_person_index.py  builds the person index over the real corpus and prints what it read
└── otl/
    ├── _bootstrap.py          env-load order + sys.path (imported first by otl/__init__.py)
    ├── graph.py               the StateGraph: spine + hard gate + hub-and-spoke wiring
    ├── state.py               OTLAuditState + per-agent sub-states + reducers
    ├── llm_adapter.py         per-node model tiering over shared.llm (HOOK_MODULE="labor")
    ├── react_loop.py          the in-node reason→act→observe harness
    ├── code_exec.py           read-scoped, scratch-confined child-process sandbox
    ├── output_writer.py       OTLExceptionReport assembly + sink (local/SharePoint)
    ├── case_resolver.py       per-case path resolution (rebuttal dir, exhibits, reports dirs)
    ├── sharepoint_reader.py   reads the required extractor synthesis from SharePoint
    ├── trace.py               trace_event + TraceWriter (the JSONL trace)
    ├── workbook_manifest.py   the deterministic exhibit/tab manifest every code specialist reads
    ├── ledger_people.py       reads the ledger's name columns; the light-tier person-name extraction
    ├── person_index.py        reads the corpus workbooks' full name lists + column roles off disk;
    │                          the med-tier pass that picks the disputed people out of them
    ├── prompts.py             every node's system prompt
    ├── agreements/
    │   ├── corpus.py          tracked asset aliases + document/legal-track manifest loader
    │   ├── record.py          ChunkRecord: the stored chunk row (SQLite / future Delta columns)
    │   ├── store.py           SQLite schema + writer; AgreementIndex reader (BM25 + sqlite-vec pools)
    │   ├── fusion.py          hybrid rank fusion (RRF) and candidate slot selection
    │   ├── embedding.py       Azure embedding settings/client, vector encoding, query bounding
    │   ├── extract.py         per-page PDF text layer → 300-DPI Tesseract fallback
    │   ├── chunk.py           legal-aware chunking into ChunkRecords
    │   ├── tokens.py          cl100k_base token counting shared by chunking and query bounding
    │   ├── cache.py           per-document JSON cache: reuse / re-chunk / OCR, chunk vectors
    │   └── build_index.py     explicit incremental index-build CLI
    └── agents/
        ├── _common.py         shared prompt/resolver helpers (synthesis_block,
        │                      directive_block, sub_get, resolve_staff_listing, pick_latest_by_keyword)
        ├── intake.py          ├─ prep spine
        ├── summarizer.py      │
        ├── category_check.py  │  (job_title/location/hierarchy; hierarchy exclusive)
        ├── verify_evidence.py ┘  HARD location-only gate
        ├── orchestrator.py    the hub supervisor (+ MAX_HOPS / MAX_AGENT_RUNS + deterministic guards)
        ├── jade_analyzer.py   last spine node (ReAct + code)
        ├── location.py        ├─ specialist spokes
        ├── job_title.py       │  (+ technical-in-nature judgement)
        ├── matching.py        │  (ReAct + code: CFR mapping lookup + category classification)
        ├── supervision.py     ┘  (deterministic monthly org tree; unresolved-person code fallback)
        ├── wiki.py            read-only terminal precedent sidecar (index-first; fail-open)
        ├── agreement_retriever.py  dual-query candidate retrieval + ID-only top-5 reranking
        └── drafter.py         terminal rebuttal / HITL deferral note (3 modes)
```

The shared repo-root `wiki/` package owns `kb_fs.py`, the empty/promoted `kb/` layout, schema, and
the type/clause/precedent/entity templates carried over from `or/production-refactor`.

> **Note on `docs/`.** `docs/README.md` predates the current design. **This root README is the source of truth**; treat the older framing in `docs/` as historical.

---

## Environment & configuration

Loaded from the single repo-root `.env` — there are no package-local `.env` files. See [`.env.azure.example`](../excel_extractor/.env.azure.example) for the full Azure block, and `--llm-config` to confirm what actually resolved.

| Variable | Meaning |
|---|---|
| `OUTPUT_TARGET` | `local` (default) or `sharepoint` |
| `INPUT_BASE` | local corpus root for required extractor synthesis when `OUTPUT_TARGET=local` |
| `OUTPUT_BASE` | local root for written reports (and the SharePoint fallback) |
| `SHAREPOINT_URL` / `SHAREPOINT_READ_FOLDER` / `SHAREPOINT_WRITE_FOLDER` | SharePoint IO; the enabled OTL extractor input reads required synthesis artifacts from `SHAREPOINT_READ_FOLDER` when `OUTPUT_TARGET=sharepoint` |
| `LLM_PROVIDER` | `ica` (default) or `azure` — selects the LLM backend |
| `ICA_API_KEY` | ICA gateway credential — **secret; never commit** |
| `ICA_MODEL_HIGH` / `_MED` / `_LIGHT` | ICA per-tier model overrides (default: opus-4-8 / sonnet-4-6 / haiku-4-5) |
| `AZURE_MODEL_HIGH` / `_MED` / `_LIGHT` | Azure per-tier models (default: gpt-5.4-pro / gpt-5.4 / gpt-5.4-mini) |
| `AZURE_OPENAI_ENDPOINT` | Azure AI Foundry project endpoint shared by all tiers; `/openai/v1/` is appended automatically |
| `AZURE_OPENAI_API_KEY` | Azure API-key authentication — **secret** |
| `MODEL_MAX_TOKENS_HIGH` / `_MED` / `_LIGHT` | per-tier max-completion-tokens overrides |
| `MODEL_MAX_INPUT_TOKENS_HIGH` / `_MED` / `_LIGHT` | per-tier input-context override; current Azure GPT-5.6 and ICA tier models default to 1,050,000 |
| `OTL_DATA_ROOT` | override the code sandbox's read-scope root |
| `OTL_AGREEMENT_MANIFEST` | agreement-corpus manifest override; default `configs/agreement_corpus.yaml` |
| `OTL_AGREEMENT_DATA_ROOT` | root containing the source agreement directories; default sibling `Data Files` |
| `OTL_AGREEMENT_INDEX_DIR` | generated OCR/chunk/FTS index; default `<Data Files>/.agreement_rag` |
| `AZURE_EMBEDDING_ENDPOINT` | Azure AI Foundry / Azure OpenAI resource root for the agreement-index embedding deployment, e.g. `https://<resource>.openai.azure.com` (not a project URL, not `/openai/v1`) |
| `AZURE_EMBEDDING_API_KEY` | API key for that resource — **secret** |
| `AZURE_EMBEDDING_DEPLOYMENT` | Foundry deployment name of `text-embedding-3-small` |
| `AZURE_EMBEDDING_MODEL` | model name recorded in the index metadata (default `text-embedding-3-small`) |
| `AZURE_EMBEDDING_DIMENSIONS` | vector size requested from the deployment (default `1536`); changing it re-embeds every chunk |
| `OTL_AGREEMENT_RETRIEVAL_MODE` | agreement candidate retrieval: `bm25` (default, keyword-only) or `hybrid` (BM25 + vector search; needs the `AZURE_EMBEDDING_*` settings and an embedding-enabled index). Any other value runs `bm25` and records `invalid_retrieval_mode` |
| `TIKTOKEN_CACHE_DIR` | optional `tiktoken` cache for the `cl100k_base` tokenizer used by chunking and hybrid query bounding; set it to a pre-populated directory in network-restricted deployments |
| `OTL_MAX_STEPS` | ReAct loop step cap for every code-exec agent (default 50) |
| `OTL_CODE_EXEC_TIER` | put every code-exec specialist on one tier (like `--code-exec-tier`) |
| `OTL_CFR_MAPPING` / `OTL_JOBS_CATALOGUE` | explicit paths to the CFR-mapping / Jobs_Catalogue workbooks (else resolved by keyword) |
| `OTL_STAFF_LISTING` | explicit path to the consolidated staff listing (else the latest `_vN` in the case's year folder) |
| `OTL_PERSON_INDEX` | `0` disables the [person index](#finding-those-people-in-the-corpus-workbooks-the-person-index) — empty index, no med-tier call, specialists search inside their own loops again |
| `WIKI_KB_DIR` | optional candidate/promoted wiki directory override; default is repo-root `wiki/kb` |
| `OTL_NAME_SELECT_CHUNK_CHARS` | characters of names per name-selection call (default 40,000 ≈ 10k tokens; floor 2,000) |
| `OTL_NAME_SELECT_JOBS` | name-selection calls in flight per case (default 4). `run_batch.py` divides a batch-wide budget of 4 across `--jobs` and writes this into each worker's env; an explicit value is left alone |

**Dependencies:** `../.venv` (the repo-root virtualenv). Install with `pip install -r requirements.txt` if building fresh; the extractor's deps (`openpyxl`, `pandas`, `pyxlsb`, `xlrd`, `langgraph`, `langchain-core`, `langchain-openai`) must be present in the same venv since the sandbox runs code with `sys.executable`.

Build or refresh the confidential agreement index explicitly before running cases:

```bash
cd orchestration-flow
../.venv/bin/python -m otl.agreements.build_index
```

The build is incremental by PDF SHA-256 and extraction version. It OCRs scanned pages at 300 DPI,
keeps generated text under `Data Files` rather than Git, and atomically replaces `index.sqlite3` only
after every registered document cache is available. Use `--document DOCUMENT_ID` to force attention to
a named document (missing caches are still built so the final index remains complete), or `--force` to
refresh unchanged sources.

The build also embeds every chunk's `verbatim_text` with the Azure AI Foundry deployment named by the
`AZURE_EMBEDDING_*` variables (repo-root `.env`), so it needs network access and those settings, and it
fails before any OCR if they are missing. Vectors are stored per chunk in
`.agreement_rag/documents/<id>.json` and loaded into the `chunk_vec` (sqlite-vec) table; a rebuild only
calls Azure for new or changed chunk text, or for every chunk after a model/dimension change.

**Hybrid agreement retrieval (opt-in).** Runtime retrieval is keyword-only (FTS5/BM25) by default. With
`OTL_AGREEMENT_RETRIEVAL_MODE=hybrid`, each run:

1. Checks that the index has vectors and that `AZURE_EMBEDDING_MODEL` / `AZURE_EMBEDDING_DIMENSIONS` match
   the index metadata, before reading any credentials or calling Azure.
2. Embeds the auditor query, the Shell query, and the expansion of every detected legal ground in one
   batched call (15-second timeout, one retry; inputs bounded to 8,000 tokens with `tiktoken`).
3. Builds, per query and legal track, a BM25 pool and a sqlite-vec cosine pool (top 20 each; vector
   search runs inside each applicable document's partition on a read-only connection), fused with
   reciprocal rank fusion.
4. Assembles each perspective's candidates (max 12) as: one fused seed per track, one reserved
   slot per track for the best vector-only match (absent from the BM25 pool), one cross-track candidate per detected ground, then fused fill in manifest
   track order. The reranker, top-5 selection and SQLite rehydration are unchanged, and retrieval scores
   never enter the reranker prompt.

If the index has no vectors, settings are missing or mismatched, the embedding call fails, or vector
search fails, the whole run falls back to the keyword-only path. `agreement_output.index_provenance`
records `requested_mode`, `effective_mode`, `fallback_reason` and `fallback_error_type` (exception class
name only), plus embedding model/dimensions, input count, whether input was shortened, embedding and
vector-search seconds, and candidate counts by method for hybrid runs. Keyword-search, rerank and
hydration failures are still reported as `status="error"`. Hybrid reads the existing index; no rebuild
is needed for an index built with embeddings.

**Security posture:** `.env` holds a live `ICA_API_KEY` and (when on Azure) a live bearer token — keep them out of commits and logs. `code_exec.py` is trusted-local, not a hardened jail. When SharePoint egress is restricted, force `OUTPUT_TARGET=local`.
