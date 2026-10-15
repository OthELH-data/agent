Exception Received
    ↓
[1] Sufficient Evidence?
    NO → Request additional analysis
    YES ↓
[2] Contract Allows This Work Type?
    NO → Check historical precedent → If none, likely deny
    YES ↓
[3] Was Employee On Location?
    YES → Check staff listing + dates → Build rebuttal
    NO ↓
    [3a] Falls Under Exception Categories?
        (Regulatory/Feasibility/Planning)
        YES → Build rebuttal with CFRs
        NO → Check historical + business confirmation
            ↓
[4] Sufficient Evidence Collected?
    YES → Build rebuttal
    NO → [5] Contact business (last resort)


## Architecture Description (OTL)
**Use LangGraph system, human, & ai messages to log for traceability**
**Save a natural language summary of findings at each step into a state variable**
**Working memory is on per-agent basis, wipe it clean after agent pass completes**
- `intake` --> gets the auditor's argument plus the classified exception record. It captures the requested amount, response-filtered classified attachments, supported assets (`Great White`, `Perdido`, `Silvertip`, `Tobago`, `Mars`, `Ursa`, `Europa`, `Appomattox`, `Whale`), and explicitly named people as leads. `jade_analyzer` resolves those leads with ledger values and POS suggestions into the downstream person population.
- `rebuttal_history` --> produces the current cross-cycle posture plus specialist revisit/reuse decisions; intake supplies deterministic error code/family
- `category_check` agent --> if auditor question is job title, location, hierarchy, or combination of any of the 3. Saves to state
- `regex_search` agent --> uses synthesis to identify POB / heli logs tabs (only look for this) --> if yes, auditor has given enough evidence. pass back excel workbook & sheet names to orchestrator. if no, request additional analysis (end flow - return back response)
- `orchestrator` --> invokes below agents in order
- `jade_analyzer` agent --> find WBS code, line item text, overhead rate (cost + OH column) (auditor's req amt should either match sum of column or half of sum of column) - if missing, apply 2.5% overhead rate & calculate this overhead column yourself. actual amount in local currency column should be a fallback if overhead column doesn't exist (auditor's req amt should either match sum of column or half of sum of column. calculate a range +/- 2.5% & if total falls within this range, then proceed). Capture all individual names found in JADE. This happens **before** the agent's own loop and is deterministic: read the `Employee` value on every charge row, fall back to `Line Item Text Orig` on the rows where `Employee` is blank or "Not assigned", then send those raw values to the **light tier** to pull the person names out ([`otl/ledger_people.py`](../otl/ledger_people.py)). The grounded names are stored once as `people_found_in_ledger`; later specialists derive their working population from that key and examine everyone unless `--max-people` was explicitly supplied. Only the auditor's cited JADE workbook is read; if it holds no readable ledger there is nothing to reconstruct and nobody to name, so the run routes straight to the drafter with an insufficient-evidence response. While the loop scans each person's rows it also runs the **major-construction screen**: JADE carries no "JV Object Type Description" text column, only a coded `WBS:JV Object Type` (or `JV Object Type`) value per row — a short code such as `C142` or `O143`. The agent reads that code per person and looks it up, via exact dict membership, against a 14-code reference table (`configs/major_construction_jv_object_types.yaml`, loaded through `otl/jv_object_types.py`) to flag **MCOH** — major construction overhead. Labor on a major construction project is directly chargeable, so the flag rides into `major_construction_by_person` / `major_construction_charge` and into the orchestrator's decision digest as a chargeability fact, quoted against the settlement's own Agreed OTL Definition and Billing Standards — though it does not override the exclusion for a person confirmed above first-level supervision, which is judged separately and is enforced contextually (prompt only), not as a deterministic guard. The lookup is a code-table match only, never inferred from the description text, a WBS code, or a project name, and the summary fields are derived in code from the per-person map. See [The major-construction screen](README.md#the-major-construction-screen).
- `location` agent --> go into POB report & filter on first & last name. Account for name variations (agent needs to reason). First check last name (search on name stem: ex. Smith Jr --> search Smith). Second, check first name (look for name variations here). Pull unique people set. Repeat this same logic on heli logs. State variable --> `found_in_pob`: dict corresponding to each person (True/False), `found_in_heli_logs`: dict corresponding to each person (True/False). Also capture origin & destination along with dates for EACH person found from heli logs along with start & end dates into state. Pull location from POB log. Pass this as natural language to next agent for token savings instead of entire JSON.
- `job_title` agent --> if `category_check` agent identifies job title as one of the questioned categories, then activate this agent (orchestrator does this). Check employee name column (last name stem 1st, first name variations 2nd). Pull position name. Edge case: 2 people have close name match & LLM cannot distinguish, send to HITL for review (path 2) with full evidence trail.
    - **Step 2**: Find duties of job title based on job catalog. Take out any words like "lead", "associate", etc. & find closet matching --> find
    - **Step 3**: Check staff listing & cross check with other 2 job info sheets.
- `matching` agent --> pass in job info from before & 3 OTL technical & chargeable job categories & classify under ONE of the 3 (closest match). If LLM cannot classify at all, send to HITL for review.
- `drafter` agent --> Or has this. Modify context order. Make tone very similar to Shell's.
