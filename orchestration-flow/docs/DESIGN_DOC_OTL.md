# Orchestration Layer Design Requirements for OTL Exception Types
This document outlines the design requirements for a multi-agentic flow to analyze & automate response generation for OTL (Offshore Technical Labor) exception types in the JV (Joint Venture) audit process.

## Architecture
For this section, I will refer to the `intake`, `rebuttal_history`, and `drafter` agents. For these, check the respective files in `orchestration-flow`. Wherever the Excel extractor is referenced, look at the `extractor` folder to determine the synthesis payload shape.

- `intake`
    - Gets auditor's argument + workbook attachments (same as existing implementation).
    - `intake` captures the requested amount, response-filtered classified attachments, and `assets_in_question`, constrained to the supported OTL asset list. People are not extracted in intake; JADE owns the person population.
- `rebuttal_history`
    - Summarizes everything from intake & extracts $ amount being disputed. Save the relevant state variables in this implementation. We already know the category to be OTL so no need for that.
- `category_check` agent
    - Returns if auditor question is job_title, location, hierarchy, or combination of any of the 3. Saves to state as a list.
- `verify_evidence_sufficiency` agent
    - Receives JSON payload from Excel extractor - this will be passed as an additional input to the `verify_evidence_sufficiency` agent. The Excel extractor data is stored on Sharepoint in the directory of `year/asset/exception_id/rebuttal_N_synthesis.json`. Use `case_resolver` to pull the appropriate data from Sharepoint.
    - Uses JSON payload of Excel extractor to pull cited Excel sheets (`.xlsx`, `.xlsb`, `.xlsm`) & looks through each one to find POB & heli logs tabs/data (only look for these). Sometimes the data may be on one tab. DISREGARD sheets with the name "back to auditor" in them (regex match this). Set `pob_data_found` to True/False & `heli_data_found` to True/False as state variables.
    - Evaluate the configured evidence rules and store one aggregate `evidence_gate_decision`. When `triggered=True`, route directly to the drafter with the missing/unreadable evidence labels; otherwise continue to JADE analysis.
- `orchestrator` agent
    - Receives the updated state & invokes below agents in order (happy path) or based on set of conditions from state.
- `jade_analyzer` agent
    - Find WBS code, line item text, overhead rate (`cost + OH`) as columns (ignore case). These column names may not be consistent across every JADE sheet so have the agent reason its way through.
    - The auditor's requested amount (recorded in the state) should either match sum of the `cost + OH` column or half of sum of that column. The reason for half is because sometimes the sum for the column is already there so the agent might get confused & sum everything, which double counts.
    - If the `cost + OH` column is missing, apply 2.5% overhead rate on the regular `Actual Amount in Local Currency` column & instruct the jade analyzer agent to calculate this overhead column itself. `Actual amount in local currency` column should be a fallback if `cost + OH` column doesn't exist (again for this column, auditor's requested amount should either match sum of column or half of sum of column). Calculate a range +/- 2.5% & if total falls within this range, then proceed. If it doesn't then flag this discrepancy in a state variable.
    - Reconcile individual names from JADE ledger values, POS-tagger suggestions, and explicitly named intake leads through the light-tier LLM pass, then store that final population once as `people_found_in_ledger`.
- `location` agent
    - Using the path for "common reports for all assets" folder pulled from `case_resolver`, access the POB & Heli logs (send to an agent to pick). If multiple versions exist of each one, then pick the latest version.
    - Go into POB report & filter on first & last name. Account for name variations (agent needs to reason).
    - First check last name (search on name stem: ex. Smith Jr --> search Smith). Second, check first name (look for name variations here). Pull unique people set.
    - Repeat this same logic on heli logs.
    - State variables --> `found_in_pob`: dict corresponding to the `people_found_in_ledger` population (True/False), `found_in_heli_logs`: dict corresponding to that same population (True/False). All people are examined unless an operator explicitly passes `--max-people`. Also capture origin & destination along with dates for EACH person found from heli logs along with start & end dates into state (`heli_person_origin` as dict & `heli_person_destination` as dict). Store location from POB for each person (`pob_people_location` as dict) in state.
    - For each person not found in POB or heli logs and/or their extracted POB or heli location(s) are not in the `assets_in_question` state variable from `intake` agent, set `invoke_job_title_agent` state variable to `True`. Otherwise, set to `False`. This should tell the orchestrator to fire the `job_title` agent next.
- `job_title` agent
    - If `category_check` agent identifies job title as one of the questioned categories, then activate this agent (orchestrator does this); otherwise, DO NOT activate & proceed to `drafter` agent.
    - Access `Data Files/OTL Information` & get the 2025 Staff listing & job catalog files.
    - In the 2025 Staff listing file, check employee name (worker) column (last name stem first, first name variations 2nd). Use the legal name columns as a last resort if no matches are found in the worker column. Pull position name. Store these in `identified_names_staff_listing` state variable as a dict with name-position key-value pairs.
        - Potential Edge case: 2 people have close name match & LLM cannot distinguish, send to HITL for review (path 2) with full evidence trail (set `hitl_review` state variable to `True`). Send to `drafter` agent prior to HITL review to draft a response & include where the confusion lies or what's missing.
    - Next, check the job catalog. Search for job profile using the identified position name from staff listing (account for position name variations here). Take out any words like "lead", "associate", etc. & find closest matching job profile. Find the job summary for the identified job profile. If there are multiple results, get all unique job listings & summaries. Store these in `identified_job_data` state variable as a dict with job profile-job summary key-value pairs.
        - Edge case: If there are multiple profiles or multiple job summaries found that sound closely related, the agent defered to HITL with `drafter` agent's review message (set `hitl_review` = `True`)
- `matching` agent --> pass in job info from before & 3 OTL technical & chargeable job categories (see CFR categories below) & classify under ONE of the 3 (should be strongly supported verbatim). Store this in a state variable called `identified_cfr_category` as a single string value. Cite the specific category clause verbatim as a `cfr_clause` state variable. If LLM cannot classify at all, send to HITL for review with `drafter` agent review message (set `hitl_review` = `True`).
- `drafter` agent --> Check existing code for this agent & modify prompts / context to fit OTL.

## CFR OTL Technical & Chargeable Categories
The Parties agree that OTL is chargeable for:
1. **Regulatory Activity**
OTL is chargeable by the Operator to the joint account for activities required to comply with
any governmental agency, authority, commission, political subdivision, or any other
governmental body or entity, including but not limited to the Bureau of Safety and
Environmental Enforcement (BSEE), and any applicable law or regulation, including but not
limited to 30 CFR 250 Subpart S (as the mandatory practice following the voluntary API RP
75 which preceded it). Mandatory OTL regulatory requirements for drilling, completion, and
intervention activities will be chargeable by the Operator to the joint account as designated
within the BSEE Certified Federal Register.
OTL charges related to the above regulatory activities must be adequately and properly
documented by the Operator, identifying the specific activity of the individual(s), and will
reference specific program elements as described in the Federal Register. This will be noted
in the Authority for Expenditure document where appropriate.
2. **Feasibility and Project Studies**
OTL associated with approved field improvement or facility improvement projects and for
feasibility studies undertaken to progress future projects is chargeable by the Operator.
3. **Planners & Schedulers, maintenance and turnaround, rig administration**
OTL is chargeable by the Operator to the joint account for planners, schedulers, and plan
execution supporters working on facility maintenance, well maintenance and turnaround
activities.

## State
At every point, the appropriate state variables must be recorded. At every step before exiting the agent, record bounded narrative analysis as `nl_summary` within the state: notable findings, material support, unresolved relevant points, and other decision-relevant context, without reproducing every detail.

Every agent must have its own state & then a master state called OTLAuditState that contains the other states' information.

- `intake` agent: `assets_in_question` (supported canonical assets only), `dollars`, and response-filtered `collected_evidence` from the classified exception record
- `rebuttal_history` agent: current posture findings plus specialist revisit/reuse state
- `category_check` agent: `auditor_question` (list) as [`job_title`, `location`, `hierarchy`] or any combination of the 3
- `verify_evidence_sufficiency` agent: presence facts plus the aggregate `evidence_gate_decision`
- `orchestrator` agent: 
- `jade_analyzer` agent: `people_found_in_ledger` (the sole grounded person-identification list), `people_named_count` (exactly its length, derived in code)
- `location` agent: `found_in_pob` (dict), `found_in_heli_logs` (dict), `origin` (dict), `destination` (dict), `start_date` (dict), `end_date` (dict)
- `job_title` agent: `identified_names_staff_listing` (dict), `identified_job_data` (dict)
- `matching` agent: `identified_cfr_category` (string), `cfr_clause` (string)
- `drafter` agent: same state as current implementation

## Working Memory & Traceability Log
- There MUST be a working memory (sort of like scratch paper for the agent) as each agent parses through the data. Once the flow goes from one agent to the next, the working memory is wiped clean. You can use a log file or similar for this, but it must record all interactions across each agent. The idea behind this is that each agent has its own whiteboard at the time of processing, but the next agent gets a clean slate so it doesn't get confused by the previous agent's memory.
- Additionally, there MUST be a permanent log that records ALL actions & traces across every agent. This is for traceability purposes that will flow into a UI.
- Use LangGraph's system, human, & assistant messages feature for both.

## Functional Requirements
- There must be a `case_resolver` that takes each exception & resolves it into an exception path. For each OTL rebuttal cycle, take the Excel extractor's matching `rebuttal_n_synthesis.json` (never a later cycle's synthesis) & resolve to the corresponding rebuttal folder path (usually titled as "Rebuttal 0...N" matched lower case with regex). If there are no rebuttal folders, then match to the directory one level inside the asset name (Great White, Tobago, etc). Also, pull the path that has the "Common reports for all assets" folder (match using lowercase namespacing) & return this. These paths are propagated into the other agents. Look into the `case_resolver.py` file in `extractor` folder. The Excel extractor results are on Sharepoint, which I'll create a .env file in the `orchestrator-flow` folder with the URL & folder names for you to use, so implement a Sharepoint sink reader & writer with a local write fallback (check `extractor` folder for a similar implementation).
- The `jade_analyzer`, `location`, & `job_title` agents MUST have code execution environments. Due to the lack of a remote sandbox, the code can be written & executed locally. Make sure the agent is able to install any dependencies it needs here.
- The agent nodes must have the ability to reason, act, & execute in loops until a definitive conclusion is reached based on the stated goal of that agent. However, it cannot keep going in loops infinitely or get stuck going down a rabbit hole.
- The entire OTL pipeline must be able to be run from a shell script, similar to `@extractor/run_test_cases.sh` that calls a `@orchestration-flow/run.py` file without any flags & with a built-in filter for labor issues cases from `@datastores/test_cases_46.csv`.
- For model tiers, the agents that require reasoning, coding, & loops during data exploration & extraction will have to be high model tier. Interpretation of findings requires a medium model tier. Final synthesis & draft response require a high model tier. Everything else requires the low model tier.

## Data to Explore During Design Phase
- Excel extractor JSON payload structure
- POB reports
- Heli logs
- JADE data provided by auditor
- Job catalog & staff listing files (if required)
- Use `@datastores/test_cases_46.csv` & filter down to the Labor Issues cases. Parse through different examples of POB, heli, & JADE reports to get an idea of how the data sources are structured.

## Token Counting
Use `tiktoken` to count tokens across each light, medium, & high reasoning tier. Check `extractor` folder to see this implementation.

## Time Taken
In the final output JSON, include the minutes taken from when the first exception started processing to the final JSON output.
