"""jade_analyzer_node — reconstruct the questioned amount from the JADE ledger (high tier, ReAct).

The first code-exec specialist. It cannot fit 30+ ledger rows × 70 columns in context, so it runs the
in-node reason→act→observe loop (`run_agent_loop`): the model writes Python that opens the auditor's
cited workbook via `open_data(...)`, reasons the column mapping SEMANTICALLY (names vary and may be
misspelled), computes sum(cost+OH) and half of it, and checks BOTH against the questioned amount within
±2.5% — the "sum-or-half" double-count guard. Only the distilled `finish` result crosses into state.

THIS NODE ALSO SETTLES THE PERSON LIST for the whole run. Before the loop starts it reads the ledger's own
name column off disk, then has the light tier reconcile those values, POS suggestions, and intake leads
(`otl/ledger_people.py`). The resolved names are stored once in `people_found_in_ledger`; every specialist
derives its capped working population from that key through `_common.disputed_people`.

The harvest runs BEFORE the loop, so this agent's own loop gets the expanded list too.

WHILE IT SCANS EACH PERSON'S ROWS the loop also runs the MAJOR-CONSTRUCTION SCREEN. JADE carries no "JV
Object Type Description" text column — only a coded "WBS:JV Object Type" (or "JV Object Type") value
per row, e.g. "C142" or "O143". The model reads that code per person and looks it up against the
`major_construction_jv_object_types` INPUT (`jv_object_types.major_construction_jv_object_types()`) —
an exact dict-membership check the model's own code performs, not a text pattern it reasons about. A
code in that table means MCOH — major construction overhead — and labor on a major
construction project is directly chargeable to the joint account under the settlement's Agreed OTL
Definition and Billing Standards, so the flag is a chargeability fact and not a ledger curiosity: it
rides into `major_construction_*` and the orchestrator's decision surface
(`orchestrator._salient_nl_summaries`) weighs it alongside the CFR route. The screen is grounded in the
looked-up code — the model reads the column, never the account description or the narrative — and its
own reading (`jv_object_type_by_person`) is kept beside the flag so the claim can be checked against
what the ledger actually said. The summary fields are derived in code from that per-person map.

This basis does NOT override the above-first-level-supervision exclusion: a person confirmed above
first level is overhead, and not chargeable, whether or not their rows carry MCOH. That reconciliation
happens where supervision is judged (`supervision_node` / `orchestrator.py`), not here — this node only
reports the code-level fact.

Normally the node returns to the orchestrator and reports `agent_status` ("objective_met" when the loop
finished, else "inconclusive"). After a finished loop, ED-02 recomputes the selected sum/half comparison
in code and routes an out-of-range result directly to the drafter. Missing-evidence policy remains in
verify_evidence; if that node was bypassed and JADE cannot operate, this node records an operationally
blocked analysis and routes to a normal needs-evidence HITL path rather than recreating ED-01.

PASSED TO THE LLM (ReAct — no structured schema; the model emits a free-form `finish` result per
react_loop.PROTOCOL):
  * system : JADE_SYSTEM (prompts.py)
  * task   : the inline reconstruction instruction below (describes the finish keys to return)
             + case_context_block(state) + directive_block(state, "jade_analyzer")
  * inputs : {jade_workbooks, disputed_people, questioned_amount, open_questions} — bound as INPUTS in
             the code sandbox; the model writes Python against them via open_data(...). Raw rows never
             enter the context — only what the model's own code prints as observations.

A SECOND, separate LLM call happens outside the loop: the light-tier `jade_people` name-resolution call
(see `llm_adapter._NODE_TIER`), whose system prompt is JADE_PEOPLE_SYSTEM. It reconciles harvested ledger
column values, their local POS suggestions, and the intake's explicitly named-person leads; it never sees
the workbook.
"""
from __future__ import annotations

from otl import ledger_people
from otl.chargeability_hooks import evaluate_state
from otl.code_exec import CodeExecutor
from otl.jade_ledger import (assign_people, cats_present, cost_center_people,
                             extract_jade_ledger, monthly_mcoh_facts)
from otl.jv_object_types import major_construction_jv_object_types
from otl.decision_hooks import JadeAnalyzerHooks, JadeReconciliationFacts
from otl.prompts import JADE_SYSTEM
from otl.react_loop import run_agent_loop
from otl.state import JadeState
from otl.trace import trace_event
from otl.workbook_manifest import build_manifest, manifest_block
from otl.agents._common import (cap_people, case_context_block, directive_block, parse_amount,
                      person_tab_hints, sub_get)


def _jade_workbooks(state: dict) -> list[str]:
    """Compatibility seam for tests and callers that override JADE workbook discovery."""
    return list(state.get("exception_exhibits") or [])


def _intake_people(state: dict) -> list[str]:
    """Return the intake's non-authoritative named-person leads for JADE resolution."""
    people = sub_get(state, "intake", "people_found", []) or []
    return [str(person).strip() for person in people if str(person).strip()]


def _to_jade_state(result: dict, questioned: float, harvest: dict) -> JadeState:
    """Map the model's free-form finish payload onto the typed JadeState, coercing numerics/lists.

    `harvest` carries the person-list fields, which are computed in CODE before the loop runs. They are
    passed in rather than read off the payload precisely because the model must not be able to overwrite
    them — the list of people examined is an audit fact about the run, not a model opinion.
    """
    r = result or {}

    def _list(v) -> list:
        if v is None:
            return []
        return list(v) if isinstance(v, (list, tuple, set)) else [v]

    def _dict(v) -> dict:
        return dict(v) if isinstance(v, dict) else {}

    # The ledger/POS/intake LLM resolution owns the person universe. Snap model-returned map keys back to
    # those exact names and discard any person the ReAct model invents or carries in from narrative context.
    ground_truth_names = [str(name) for name in harvest.get("people_found_in_ledger", []) or []]
    canonical_names = {" ".join(name.lower().split()): name for name in ground_truth_names}

    def _person_map(value, convert) -> dict:
        out: dict = {}
        for raw_name, raw_value in _dict(value).items():
            canonical = canonical_names.get(" ".join(str(raw_name).lower().split()))
            if canonical is not None:
                out[canonical] = convert(raw_value)
        return out

    # The major-construction screen: the model reports the per-person reading, and the two summary
    # fields are derived here rather than read off the payload, so the flag and the people it names can
    # never disagree with the per-person map the reading is recorded in.
    mcoh = _person_map(r.get("major_construction_by_person"), bool)
    object_types = _person_map(
        r.get("jv_object_type_by_person"),
        lambda value: ([str(item) for item in value]
                       if isinstance(value, (list, tuple, set)) else str(value)),
    )

    return JadeState(
        nl_summary=str(r.get("nl_summary", "")),
        resolved_columns={str(k): str(v) for k, v in _dict(r.get("resolved_columns")).items()},
        jade_wbs_codes=[str(x) for x in _list(r.get("jade_wbs_codes"))],
        jade_line_item_texts=[str(x) for x in _list(r.get("jade_line_item_texts"))],
        overhead_mode=str(r.get("overhead_mode", "")),
        overhead_rate=parse_amount(r.get("overhead_rate")) or 0.025,
        amount_basis=str(r.get("amount_basis", "")),
        amount_computed=parse_amount(r.get("amount_computed")),
        amount_questioned=parse_amount(r.get("amount_questioned")) or float(questioned or 0.0),
        amount_in_range=bool(r.get("amount_in_range", False)),
        jade_amount_discrepancy=bool(r.get("jade_amount_discrepancy", False)),
        jv_object_type_by_person=object_types,
        major_construction_by_person=mcoh,
        major_construction_people=[p for p, flagged in mcoh.items() if flagged],
        major_construction_charge=any(mcoh.values()),
        major_construction_basis=str(r.get("major_construction_basis", "")),
        people_named_count=len(ground_truth_names),
        **harvest,
    )


def _manifest_access_issues(manifest: dict) -> list[str]:
    """Compact technical diagnostics for supplied workbook paths that could not be opened."""
    issues: list[str] = []
    for item in manifest.get("unreadable", []) or []:
        name = str(item.get("name") or item.get("path") or "workbook")
        errors = item.get("errors") or {}
        detail = "; ".join(f"{key}: {value}" for key, value in errors.items())
        issues.append(f"{name}: {detail or 'could not be opened'}")
    return issues


def _deterministic_result(state: dict, manifest: dict, workbooks: list[str], questioned: float) -> dict:
    """Normal JADE path: deterministic rows and arithmetic; ReAct is reserved for escalation."""
    case = state.get("case") or {}
    extracted = extract_jade_ledger(manifest, questioned)
    names = ledger_people.extract_people(
        "jade_people", extracted.raw_people, intake_people=_intake_people(state),
    )
    rows = assign_people(extracted.rows, names)
    monthly = monthly_mcoh_facts(rows)
    by_person_amount: dict[str, float] = {}
    object_types: dict[str, list[str]] = {}
    mcoh: dict[str, bool] = {}
    for row in rows:
        if not row.person:
            continue
        if row.amount is not None:
            by_person_amount[row.person] = by_person_amount.get(row.person, 0.0) + float(row.amount)
        if row.jv_object_type:
            object_types.setdefault(row.person, [])
            if row.jv_object_type not in object_types[row.person]:
                object_types[row.person].append(row.jv_object_type)
        if row.mcoh_status:
            mcoh[row.person] = bool(mcoh.get(row.person)) or row.mcoh_status == "mcoh"

    any_mcoh = any(row.mcoh_status == "mcoh" for row in rows)
    tabs = list(dict.fromkeys(
        f"{row.source_ref.workbook}!{row.source_ref.sheet}" for row in rows
    ))
    basis = (
        f"Read {extracted.resolved_columns.get('jv_object_type', 'the JV Object Type column')} on "
        f"{len(rows)} normalized detail row(s); exact registry lookup found "
        f"{sum(1 for row in rows if row.mcoh_status == 'mcoh')} MCOH row(s)."
    )
    nl_summary = (
        f"The labor ledger contains {len(rows)} normalized detail row(s) and "
        f"{len(names)} identified person(s). The reconstructed amount is "
        f"${extracted.amount_computed:,.2f} on the {extracted.amount_basis} basis; "
        f"MCOH is {'present' if any_mcoh else 'not present on resolved rows'}."
    )
    sub = JadeState(
        nl_summary=nl_summary,
        resolved_columns=extracted.resolved_columns,
        jade_wbs_codes=list(dict.fromkeys(row.wbs for row in rows if row.wbs)),
        jade_line_item_texts=list(dict.fromkeys(row.line_item_text for row in rows if row.line_item_text)),
        overhead_mode=extracted.overhead_mode,
        overhead_rate=extracted.overhead_rate,
        amount_basis=extracted.amount_basis,
        amount_computed=extracted.amount_computed,
        amount_questioned=float(questioned or 0.0),
        amount_in_range=extracted.amount_in_range,
        jade_amount_discrepancy=not extracted.amount_in_range,
        charged_amount_by_person={k: round(v, 2) for k, v in by_person_amount.items()},
        ledger_rows=rows,
        mcoh_monthly_facts=monthly,
        cats_timewriting_present=cats_present(rows),
        asset_cost_center_people=cost_center_people(rows, str(case.get("asset") or "")),
        jade_analysis_mode="structured",
        jade_escalation_reasons=extracted.warnings,
        jv_object_type_by_person=object_types,
        major_construction_by_person=mcoh,
        major_construction_people=[person for person, flagged in mcoh.items() if flagged],
        major_construction_charge=any_mcoh,
        major_construction_basis=basis,
        ledger_tabs_read=tabs,
        ledger_person_values=extracted.raw_people,
        people_found_in_ledger=names,
        people_named_count=len(names),
    )
    evaluations, decision, snapshot, reconciliation = JadeAnalyzerHooks.evaluate(
        JadeReconciliationFacts(
            loop_status="finished", amount_basis=sub.amount_basis,
            amount_computed=sub.amount_computed, amount_questioned=sub.amount_questioned,
            model_amount_in_range=sub.amount_in_range,
            model_jade_amount_discrepancy=sub.jade_amount_discrepancy,
        )
    )
    if reconciliation.valid:
        sub.amount_in_range = reconciliation.in_range
        sub.jade_amount_discrepancy = not reconciliation.in_range
    sub.jade_hook_evaluations = [item.model_dump(mode="json") for item in evaluations]
    sub.jade_hook_decision = decision.model_dump(mode="json")
    update: dict = {
        "jade_analyzer": sub, "analysis_blocked": False,
        "jade_hook_decision": sub.jade_hook_decision,
        "jade_hook_evaluations": sub.jade_hook_evaluations,
        "hook_status": {"jade_analyzer": snapshot},
        "agent_status": {"jade_analyzer": {
            "status": "objective_met" if reconciliation.valid else "inconclusive",
            "reason": "" if reconciliation.valid else reconciliation.reason,
        }},
    }
    update.update(evaluate_state({**state, **update}, final=False))
    update.update(trace_event(
        state, "jade_analyzer", "reconstructed amount (structured row analysis)",
        inputs_digest=(f"{len(workbooks)} workbook(s), questioned={questioned}; "
                       f"{len(rows)} ledger row(s), {len(names)} identified person(s)"),
        key_outputs={
            "mode": "structured", "amount_basis": sub.amount_basis,
            "amount_computed": sub.amount_computed, "amount_in_range": sub.amount_in_range,
            "ledger_rows": len(rows), "mcoh_rows": sum(1 for r in rows if r.mcoh_status == "mcoh"),
            "mcoh_monthly_facts": len(monthly), "major_construction_charge": any_mcoh,
            "people_found_in_ledger": names, "resolved_columns": sub.resolved_columns,
        }, nl_summary=sub.nl_summary,
    ))
    return update


def _no_ledger_inconclusive(state: dict, workbooks: list[str], questioned: float, *,
                            why: str, access_issues: list[str] | None = None) -> dict:
    """Stop safely when JADE cannot operate, without recreating an ED-01 decision.

    Configured evidence policy belongs exclusively to ``verify_evidence``. If that optional node was
    bypassed, or a readable ledger later proves unusable, JADE reports an operationally inconclusive
    analysis and routes to HITL rather than applying missing-evidence policy itself.
    """
    access_issues = list(access_issues or [])
    sub = JadeState(
        nl_summary=(f"JADE analysis could not proceed because {why}. The amount and person population "
                  "remain ungrounded, so downstream specialist analysis cannot proceed."),
        amount_questioned=float(questioned or 0.0),
        evidence_gate_status="",
        evidence_access_issues=access_issues,
    )
    update: dict = {
        "jade_analyzer": sub,
        "analysis_blocked": True,
        "final_mode": "needs_evidence",
        "hitl_review": True,
        "hitl_stage": "jade_analyzer",
        "hitl_reason": f"JADE analysis could not proceed because {why}.",
        "agent_status": {"jade_analyzer": {"status": "inconclusive", "reason": why}},
    }
    update.update(trace_event(
        state, "jade_analyzer",
        "ledger analysis blocked — operational HITL deferral",
        inputs_digest=f"{len(workbooks)} cited workbook(s), 0 ledger tab(s)",
        key_outputs={"analysis_blocked": True, "ledger_tabs_read": [],
                     "evidence_access_issues": access_issues},
        nl_summary=sub.nl_summary,
    ))
    return update


def jade_analyzer_node(state: dict) -> dict:
    run_id = state.get("run_id", "")
    case = state.get("case") or {}
    questioned = parse_amount(sub_get(state, "intake", "dollars", 0.0))
    open_questions = list(state.get("open_questions") or [])
    workbooks = _jade_workbooks(state)
    # Read the workbooks' shape off disk once, before the loop starts. Deterministic and cheap (about
    # 2s for a whole case), and it saves the model the turns it used to spend listing sheets and
    # hunting for the header row. People are matched to tabs using the extractor's own tab notes, which
    # already name the person a per-person tab holds — the reads here are only for shape.
    manifest = state.get("auditor_workbook_manifest") or build_manifest(
        workbooks, tab_hints=person_tab_hints(state)
    )

    # The ledger tabs come straight out of that manifest — their header labels and header row are already
    # recorded, so finding the name column costs no model turn and no second read. An inaccessible
    # supplied workbook wins over a denial when no usable ledger exists: the record may contain the
    # required evidence, but its accessibility needs human resolution first.
    if not workbooks:
        return _no_ledger_inconclusive(
            state, workbooks, questioned,
            why="the auditor cited no workbook for this exception",
        )
    try:
        return _deterministic_result(state, manifest, workbooks, questioned)
    except Exception as exc:
        deterministic_failure = f"structured_extraction_failed:{type(exc).__name__}:{exc}"

    ledger_tabs = ledger_people.find_ledger_tabs(manifest)
    manifest_access_issues = _manifest_access_issues(manifest)
    if not ledger_tabs:
        if manifest_access_issues:
            return _no_ledger_inconclusive(
                state, workbooks, questioned,
                why="the supplied evidence includes workbook material that could not be opened or accessed",
                access_issues=manifest_access_issues,
            )
        return _no_ledger_inconclusive(
            state, workbooks, questioned,
            why=f"no labor-ledger tab was found in {len(workbooks)} cited workbook(s)",
        )

    values, tabs_read, harvest_access_issues = ledger_people.harvest_with_diagnostics(manifest)
    access_issues = manifest_access_issues + harvest_access_issues
    if not values:
        if access_issues:
            return _no_ledger_inconclusive(
                state, workbooks, questioned,
                why="the supplied labor-ledger evidence could not be opened or accessed",
                access_issues=access_issues,
            )
        return _no_ledger_inconclusive(
            state, workbooks, questioned,
            why="the cited labor-ledger tab contained no usable employee or line-item detail",
        )

    ledger_names = ledger_people.extract_people(
        "jade_people", values, intake_people=_intake_people(state),
    )
    specialist_people, dropped = cap_people(ledger_names)
    # The POS-assisted review, reconciled with intake leads, settles every identity for the case. Total the
    # full resolved population before the person cap; the cap limits later specialist
    # work only and must never silently omit a charged person's dollars from the final outcome.
    charged_amounts = ledger_people.charged_amount_by_person(manifest, ledger_names)
    harvest = {
        "ledger_tabs_read": tabs_read,
        "ledger_person_values": values,
        "people_found_in_ledger": ledger_names,
        "charged_amount_by_person": charged_amounts,
    }
    # Rebuild with the resolved list so the person → tab lookup covers all final names. Cheap:
    # `_profile_workbook` caches on (path, size, mtime), so no file is reopened — only the person
    # resolution re-runs.
    if ledger_names:
        manifest = build_manifest(workbooks, people=ledger_names, tab_hints=person_tab_hints(state))

    mcoh_codes = major_construction_jv_object_types()
    inputs = {
        "jade_workbooks": workbooks,
        # The major-construction screen covers the complete resolved population. The configured
        # cap applies only when later specialist nodes read people through disputed_people(state).
        "disputed_people": ledger_names,
        "questioned_amount": questioned,
        "open_questions": open_questions,
        # code -> description, keyed by normalized code (see otl/jv_object_types.py). The screen below
        # is exact dict-membership against this table, not a text pattern the model reasons about.
        "major_construction_jv_object_types": mcoh_codes,
    }
    task = (
        f"The deterministic row-first pass escalated because {deterministic_failure}. Resolve only that "
        "exceptional mapping or evidence gap, using raw workbook code when needed. "
        "Reconstruct the questioned dollar amount from the JADE ledger and check "
        "the auditor's figure.\n\n"
        "In your code, INPUTS holds:\n"
        "- jade_workbooks: absolute path(s) to the auditor's cited workbook(s); the JADE ledger is ONE "
        "of the tabs (the per-line-item charge rows — NOT a single-person POB/heli tab).\n"
        f"- disputed_people: {ledger_names} — the final people_found_in_ledger list from the ledger, "
        "POS, and intake-lead LLM resolution pass. Screen every listed person and use these exact names as "
        "the keys in both per-person maps.\n"
        f"- questioned_amount: {questioned} — the auditor's figure to check against.\n"
        "- open_questions: reviewer-flagged gaps to keep in mind (hints only; never a gate).\n"
        f"- major_construction_jv_object_types: a dict of {len(mcoh_codes)} normalized code -> "
        "description, e.g. {'C142': 'AFE;Capex;MajConst OH;Surf Eqp Pen5', ...} — the reference table "
        "for the major-construction screen below. Look a code up EXACTLY (normalize it the same way: "
        "strip whitespace, upper-case) rather than guessing from the description text.\n\n"
        "Open each path with open_data(path). List sheet names with list_sheets(path) (operator-response "
        "tabs are already "
        "excluded from the sandbox, so a tab you expected and cannot see is intentionally hidden, not a "
        "bad path — never try to work around it). Locate the JADE ledger, then reason the "
        "column mapping semantically (WBS / person-name / cost+OH / base / OH RATE / "
        "WBS:JV Object Type — tolerate spelling, spacing and case variation; the name often lives in a "
        "Line-Item-Text column when Employee is blank). Compute sum(cost+OH) and half of it, compare "
        "BOTH to questioned_amount within ±2.5%, and cross-check base × (1 + OH RATE).\n"
        f"The ledger tab(s) whose name column was already read: {tabs_read}. Those reads produced "
        f"{len(values)} distinct value(s); the names above came out of them.\n"
        "CHECKPOINT: call save_result(...) with the cumulative result as soon as the core amount "
        "reconciliation is done (amount_basis, amount_computed, amount_in_range) — before moving on "
        "to any further validation steps.\n"
        "THEN, on the rows you already have, run the MAJOR-CONSTRUCTION SCREEN for EACH person: read "
        "their 'WBS:JV Object Type' value (equivalently just 'JV Object Type' — whatever this ledger "
        "calls the coded object-type column; it is a short code like 'C142' or 'O143', NOT a "
        "description). Record that code verbatim per person in jv_object_type_by_person. Then look each "
        "code up in the major_construction_jv_object_types INPUT (normalize it first: strip whitespace, "
        "upper-case, exact dict lookup) and set major_construction_by_person[person] = True when any of "
        "their rows carry a code found in that table. Say which column you read in "
        "major_construction_basis, and name it in resolved_columns too. This is a code lookup, not text "
        "matching — do NOT decide MCOH from the description text, an account description, or a project "
        "name; a code either is or is not in the table. Leave the maps empty if this ledger has no such "
        "column, and if a code you find is not in the table, that person is simply not flagged — do not "
        "guess at an unlisted code's membership.\n"
        "Then finish with a result object holding: resolved_columns, "
        "jade_wbs_codes, jade_line_item_texts, overhead_mode, overhead_rate, amount_basis "
        "('sum'|'half'), amount_computed, amount_questioned, amount_in_range, jade_amount_discrepancy, "
        "jv_object_type_by_person, major_construction_by_person, major_construction_basis, nl_summary. "
        "Findings must explain the notable ledger support, amount reconstruction, material discrepancies, "
        "and unresolved points without reproducing every line item."
        + case_context_block(state, specialist="jade_analyzer")
        + directive_block(state, "jade_analyzer")
        + manifest_block(manifest)
    )

    executor = CodeExecutor(run_id, "jade_analyzer", read_roots=state.get("sandbox_read_roots"),
                            manifest=manifest)
    loop = run_agent_loop(
        node="jade_analyzer", system=JADE_SYSTEM, task=task, executor=executor, inputs=inputs,
        trace_path=state.get("trace_path"), run_id=run_id, case_id=case.get("case_id", ""),
    )
    sub = _to_jade_state(loop.result, questioned, harvest)
    sub.jade_analysis_mode = "code_escalation"
    sub.jade_escalation_reasons = [deterministic_failure]
    model_amount_in_range = sub.amount_in_range
    model_jade_amount_discrepancy = sub.jade_amount_discrepancy
    hook_evaluations, hook_decision, hook_snapshot, reconciliation = JadeAnalyzerHooks.evaluate(
        JadeReconciliationFacts(
            loop_status=loop.status,
            amount_basis=sub.amount_basis,
            amount_computed=sub.amount_computed,
            amount_questioned=sub.amount_questioned,
            model_amount_in_range=model_amount_in_range,
            model_jade_amount_discrepancy=model_jade_amount_discrepancy,
        )
    )
    if reconciliation.valid:
        sub.amount_in_range = reconciliation.in_range
        sub.jade_amount_discrepancy = not reconciliation.in_range
    sub.jade_hook_evaluations = [item.model_dump(mode="json") for item in hook_evaluations]
    sub.jade_hook_decision = hook_decision.model_dump(mode="json")

    # Report analysis status; a configured ED-02 mismatch may route directly to the drafter.
    status = "objective_met" if loop.status == "finished" and reconciliation.valid else "inconclusive"
    if status == "objective_met":
        reason = ""
    elif loop.status == "finished":
        reason = reconciliation.reason
    else:
        reason = (f"JADE reconstruction did not converge within the step budget ({loop.status}); "
                  f"last observation: {loop.last_observation[:300]}")
    update: dict = {"jade_analyzer": sub, "analysis_blocked": False,
                    "jade_hook_decision": sub.jade_hook_decision,
                    "jade_hook_evaluations": sub.jade_hook_evaluations,
                    "hook_status": {"jade_analyzer": hook_snapshot},
                    "agent_status": {"jade_analyzer": {"status": status, "reason": reason}}}
    update.update(evaluate_state({**state, **update}, final=False))

    action = f"reconstructed amount ({loop.status}, {loop.steps} steps)"
    if dropped:
        action += (f"; screened {len(ledger_names)} ledger person(s); later specialists capped to "
                   f"{len(specialist_people)}")
    if sub.major_construction_charge:
        action += f"; MCOH on {len(sub.major_construction_people)} person(s)"
    if hook_decision.triggered:
        action += f"; hook denial ({', '.join(hook_decision.gate_ids)})"
    update.update(trace_event(
        state, "jade_analyzer", action,
        inputs_digest=(f"{len(workbooks)} workbook(s), questioned={questioned}; people: "
                       f"{len(ledger_names)} from {len(values)} ledger value(s) → "
                       f"{len(ledger_names)} screened; {len(specialist_people)} downstream"),
        key_outputs={"amount_basis": sub.amount_basis, "amount_computed": sub.amount_computed,
                     "amount_questioned": sub.amount_questioned, "amount_in_range": sub.amount_in_range,
                     "jade_amount_discrepancy": sub.jade_amount_discrepancy,
                     "jade_hook_decision": sub.jade_hook_decision,
                     "reconciliation_details": reconciliation.details,
                     "resolved_columns": sub.resolved_columns,
                     "major_construction_charge": sub.major_construction_charge,
                     "major_construction_people": sub.major_construction_people,
                     "jv_object_type_by_person": sub.jv_object_type_by_person,
                     "major_construction_basis": sub.major_construction_basis,
                     "ledger_tabs_read": sub.ledger_tabs_read,
                     "people_found_in_ledger": sub.people_found_in_ledger,
                     "people_named_count": sub.people_named_count,
                     "charged_amount_by_person": sub.charged_amount_by_person},
        nl_summary=sub.nl_summary or (
            f"amount {sub.amount_computed:,.2f} via {sub.amount_basis or '—'}, "
            f"in_range={sub.amount_in_range}; {len(ledger_names)} ledger person(s) screened"),
    ))
    return update


__all__ = ["jade_analyzer_node"]
