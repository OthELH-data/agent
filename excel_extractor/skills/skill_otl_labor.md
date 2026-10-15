# TYPE SKILL: OTL / Offsite Technical Labor Chargeability

**Applies to families:** `otl`, `offsite-technical-labor`, `cboh`,
`affiliate-labor`, `payroll-technical`, `contractor-labor`, `siep-labor`,
`cats-labor`
**Error-code anchors (authority map):** 06.x Admin/OH-Technical,
09.x Payroll-Technical, and 18.x Affiliate. Resolve the exact authority at
runtime from `registry/authority_map.yaml`; the auditor's title is not routing
authority.
**Source:** Scott Barrios OTL principle (Jul 6), 2022 GW E007 walkthrough,
Prajwala OTL examples, Jul 17 Affiliates walkthrough, and the Jul 23
**OTL Process Map Review**.

**Companion extractor:** `skills/extract_otl_labor.md`. Its facts and joins feed
this decision flow. Do not repeat its work from a narrative summary when its
structured packet is available.

---

## Runtime contract

1. Inventory all already-collected evidence before claiming anything is missing.
2. Complete STEP 0 → STEP 7 for **each person and challenged charge period**.
3. Use location to decide whether the case remains OTL, then select the governing
   instrument before testing the role/activity.
4. Separate four questions: auditor-claim sufficiency, technical-employee status,
   chargeability route, and Shell's proof coverage under that route.
5. Cite the exact verified settlement subsection; a generic “Settlement
   Agreement” citation is incomplete.
6. Employee/manager outreach is the last evidence tier, not the default.
7. Return a disposition only after all steps have a recorded status.
8. Treat this workflow as a **routing tree**, not a universal all-elements gate.
   Apply only the elements required by the selected, verified contract route.

---

## ⛔ MANDATORY NO-SKIP CHECKLIST — run STEP 0 → STEP 7 in order

Do not shortcut from a title, POB row, precedent, or keyword to a disposition.
Record every step under `evidence_packet.otl.walkthrough` as `complete`,
`incomplete`, or `not_applicable`, with evidence IDs and a short finding.

- [ ] **STEP 0** — Claim subtype, collected-evidence inventory, and sufficiency
- [ ] **STEP 1** — JADE population, exact charged dates, amount, and cost object
- [ ] **STEP 2** — POB/helicopter location gate for the challenged dates
- [ ] **STEP 3** — Governing instrument, technical-employee definition, and alternate route
- [ ] **STEP 4** — OTL A/B/C or major-construction/IPT chargeability path
- [ ] **STEP 5** — Tiered corroboration and precedent consistency
- [ ] **STEP 6** — Above-first-level/double-dip screen and proof-coverage finding
- [ ] **STEP 7** — Disposition, exact authority, and scoped operator response

If a step lacks evidence, identify **which party or system should possess the
missing proof**. Do not shift a Shell documentation obligation to the auditor.

---

## STEP 0 — Claim subtype + evidence inventory + sufficiency

Classify the substance, not the exception label:

- `location` / `wrong_property`;
- `not_technical` / administrative-in-nature;
- `outside_A_B_C`;
- `above_first_level`;
- `major_construction_or_IPT`;
- `affiliate_entity_or_rate` (retain OTL eligibility here; hand entity/rate/
  attestation facts to `type_affiliate_labor`);
- `documentation_only` or mixed.

Start with `collected_evidence`, the pre-extracted workbook packet, and every
attachment/tab. Confirm whether the auditor supplied the challenged population,
dates, amounts, and its claimed methodology. An attachment already analyzed in
state is not “missing” merely because a later summary omitted it.

**Sufficiency branch:**

- If the auditor asserts “not on POB” or another analytical result but supplies
  neither the underlying population nor its analysis, return a concise request
  such as: *“Exception not adequately supported. Please provide the underlying
  POB analysis for the challenged personnel and dates.”* Use `needs_evidence`.
- If the auditor supplied enough to identify and test the claim, continue. This
  does **not** mean the auditor has proven the credit.
- If the applicable contract requires the Operator to document the activity,
  treat missing Shell support as `operator_evidence_required`, not as an auditor
  failure. For Chevron §3.A, specific activity/program-element documentation is
  an express part of the route.

Record `auditor_claim_sufficiency` and `operator_proof_coverage` separately.

## STEP 1 — Tie the exact challenged charge to JADE

For each person:

- tie the auditor population to JADE/SAP by employee, document, WBS/cost object,
  posting object/network/work order, period, and amount;
- preserve base labor, burden, and overhead separately; confirm any overhead is
  a separate account/line before arguing it is already recovered;
- capture project/overhead type and the work-order scope/CFR field;
- establish the **actual charged dates or periods** before using location data.

An account description such as `P&T-LABOR-BILLABLE` proves how the line was
recorded, not whether the agreement permits the charge. A JADE tie that cannot
isolate the challenged person/period remains `needs_evidence`.

## STEP 2 — Location gate first (POB + helicopter)

Compare POB/helicopter evidence with the exact charged dates:

- **On audited asset during charged dates:** this is on-site work, not OTL. Stop
  the A/B/C analysis and route to the separate above-first-level check in STEP 6.
  Location can support denial of an OTL/location claim, subject to that check.
- **On another asset, but outside charged dates:** this does not prove the audited
  JV was mischarged. The Kenneth Oyler anchor is controlling: Olympus/Proteus
  presence from Oct 31–Nov 11 did not disprove Great White charges on Oct 19–20.
- **Not on POB during charged dates:** do not concede. Continue to the off-site
  contractual pathways.
- **POB/heli data missing or dates cannot be aligned:** record the exact gap. A
  staff-listing hit cannot substitute for presence.

POB/helicopter records establish location only; they do not establish which JV
was billed or what activity the person performed.

## STEP 3 — Contract before role evidence

Select authority by co-owner, asset, activity year, and project type:

1. Retrieve the governing settlement agreement/JOA/UOA or edition-compatible
   COPAS MFI. For Perdido/Great White/Silvertip/Tobago, the Chevron or BP
   settlement commonly governs; newer/deep-water assets may use MFI 53.
2. Apply the agreement's **technical-employee definition** before A/B/C. Test the
   nature of work—not the title string—for special engineering, geological,
   geophysical, technical, or operational skills and services benefiting the
   Joint Property.
3. Check alternate direct-contract routes:
   - an approved **Integrated Project Team** and in-scope org chart;
   - **major-construction overhead** project type and its exact clarification.
4. If no alternate route applies, continue to the OTL A/B/C test.

Run `agreement_tool.instrument_inventory` before treating the contract package as
complete. `registry_coverage_complete=true` means all locally registered
instruments exist; it does not prove that no later amendment, side letter, or
co-owner-specific agreement exists.

For the verified Chevron instrument, pinpoint:

- **Settlement Agreement §2 / Attachment 1 §2** — technical employees;
- **Attachment 1 §3.A, §3.B, or §3.C** — the selected OTL route;
- **Attachment 1 §5** when affiliate application is material.

Do not cross-cite Chevron language to a BP-only case. If exact BP text is not
machine-verifiable, keep authority open rather than substituting Chevron.
MFI 45 (marine/air) and MFI 38 (materials) are not OTL authority.

## STEP 4 — Select one supported chargeability path

### Route A — Regulatory activity (Attachment 1 §3.A)

Map the person's **specific activity**, not just the title, to a verified
governmental requirement, applicable law/regulation, BSEE/SEMS requirement, or
Federal Register program element. `30 CFR 250` is a search space, not a complete
pinpoint. Where the agreement requires it, verify an approved AFE and adequate
documentation of the activity/program element. A supported A route needs no
redundant B/C justification.

### Route B — Feasibility or project study (Attachment 1 §3.B)

Confirm a field/facility-improvement or feasibility-study scope and its
separately approved AFE. In P16, `EP` in a WBS is a transcript-observed FeasEx
candidate and may trigger retrieval from Freedom; it is not proof of an AFE.
AFE approval does not make every line charged to it billable—person, period,
activity, and agreement eligibility still must tie.

### Route C — Planner/scheduler, maintenance, turnaround, rig administration (Attachment 1 §3.C)

Confirm the technical role/activity and, where the verified clause requires it,
that the function was previously performed offshore/onsite and relocated
onshore. Exclude dedicated non-technical administrative support. Title-only
keyword matching is insufficient.

### Alternate route — Major construction or IPT

When the exact contract and JADE project type establish major-construction
technical labor, or the person/role is on the approved IPT for the period, use
that direct route. Do not manufacture an additional CFR requirement.

Scott's affiliate/OTL notes provide a Great White example: the UOA Addendum to
Exhibit C, Section III, Paragraph 2.B uses the selected direct-charge option for
engineering, design, and drafting costs. The clarification permits qualifying
technical services for Major Construction to be direct charged regardless of
location and applies an instrument-specific actual-day threshold. When that
exact clause governs and the person/project/day facts match, the clause can be
dispositive without forcing an OTL A/B/C or CFR narrative. Do not generalize the
Great White option or threshold to another asset or agreement.

### Dispositive-contract lane

Use `contract_dispositive` only when:

- the provision is in the case's PDF-verified selected authority set;
- the registered governing-instrument inventory is complete for the asset;
- its scope directly resolves the auditor's challenged category;
- project/person/period applicability is tied to source evidence; and
- no unresolved exception, approval, threshold, or scope limitation remains.

This lane removes redundant proof requirements; it does not permit a bare model
assertion that “the contract allows it.”

If the person fails the technical-employee definition and no alternate route
applies, move to precedent/consistency in STEP 5 before concluding.

## STEP 5 — Evidence ladder + historical consistency

Use the strongest available evidence and stop when the selected contract route
is fully proved:

1. **Tier 1 — location:** POB and helicopter logs, already tested in STEP 2.
2. **Tier 2 — system/role:** JADE/work-order scope, staff listing, HR job catalog,
   and verified regulatory mapping.
3. **Tier 3 — project authority:** white paper/allocation memo, project document,
   AFE, approved IPT org chart, or major-construction clarification.
4. **Tier 4 — business confirmation:** employee/manager email or outreach. Use
   only when the available systems cannot explain the in-scope activity.

Search validated precedents for the same normalized role, agreement, and claim
type. Reuse their contract reasoning and general role description, never their
current-period facts. Then:

- repeated, consistent prior grants for a clearly administrative role (Scott's
  cost-specialist/cost-planner example) can support a credit within the approved
  risk threshold without reflexive business outreach;
- repeated, verified defenses can support the role theory but still require the
  current person/period/WBS tie;
- mixed precedent or a new portfolio-wide issue requires human review;
- dollar value informs risk routing but never decides chargeability.

Use `job_catalog_tool.lookup` for an exact job code/profile when available,
otherwise use its Hugging Face semantic match. The catalogue itself says its
profiles are generic and do not describe an individual's actual position or
competencies. It can support technical-versus-administrative role nature; it
cannot establish current activity, reporting level, Joint Property benefit, or
chargeability by itself.

## STEP 6 — Above-first-level + double-dip + proof coverage

Above-first-level supervision is a **separate accounting question** that can
apply to on-site or off-site personnel. COPAS may place above-first-level
supervision in overhead, but “lead” or “manager” in a title is not proof of the
reporting level.

- If the auditor expressly makes this claim, use the staff reporting chain and
  the Time Writing First Level Analysis evidence when available.
- Automated above-first-level determination is a Phase-2 enhancement; until the
  reporting-chain logic and exact authority are grounded, emit `needs_human`.
- Decompose a mixed exception. The auditor must not recover the same amount once
  as OTL/location and again as above-first-level supervision.

Now perform proof coverage against the selected route:

- mark each required element `supported`, `contradicted`, or `unresolved`;
- cite the evidence ID for every supported/contradicted element;
- identify the owner of every unresolved item (`auditor`, `operator`,
  `agreement_repository`, `business`, or `restricted_system`);
- do not say “need evidence” generically when collected evidence already answers
  the question.

## STEP 7 — Disposition and response logic

- **Deny:** the challenged charges tie to the person/period/JV and a verified
  authority route is fully supported; or the auditor's contrary location/title
  analysis is demonstrably based on non-overlapping dates/name-match error.
- **Accept recommendation:** evidence proves the person/period/activity was not
  eligible or was charged to the wrong property, with the amount tied.
- **Partial recommendation:** only a subset of people, periods, or amounts
  fails; quantify it.
- **Needs evidence / human:** the selected route has a material unresolved
  element, exact authority is unavailable, reporting level is ambiguous, or
  systems cannot establish current activity.

The final response should state the exact challenged ask, the analysis performed,
and the outcome. For a denial, name the person, charged dates, property/WBS,
supported route, and exact verified section. For a credit, quantify only the
proved ineligible portion and label `accept`/`partial` as pending human approval.
No credit is posted automatically. Internal employee emails may support the
internal decision but must not be attached or quoted unless visibility permits.

---

## Tool orchestration (in order)

1. **Evidence inventory / extractor packet** — load all attachments and
   `collected_evidence`; reject false missing-proof claims.
2. **`jade_tool.cross_check` / `query_ledger`** — tie person, period, WBS/work
   order, amount, overhead, and project type.
3. **`pob_presence_tool` + helicopter evidence** — compare the audited asset and
   exact charged dates; route on-site away from A/B/C.
4. **`agreement_tool.find_support` + `verify_citation`** — select the
   co-owner/asset/year instrument and verify exact OTL/major-construction/IPT
   language. Run `instrument_inventory` and preserve the Chevron §2 /
   Attachment 1 §2 + selected §3 subsection.
5. **Staff listing + `job_catalog_tool.lookup`** — establish identity and generic
   role context; do not infer current activity or reporting level.
6. **Route-specific lookup** — regulatory source for A; Freedom/AFE for B;
   relocation/activity evidence for C; org chart/clarification for IPT/major
   construction.
7. **Semantic precedent search** — retrieve role + agreement + claim analogs;
   use only verified, fact-compatible reasoning.
8. **White-paper/project-document lookup**, then **business outreach last**.

## Confirmation gate

A final OTL disposition requires all of the following:

- challenged person/period/amount tied to JADE;
- location gate completed or explicitly unresolved;
- governing instrument and exact route identified;
- technical-employee and route-specific elements covered by cited evidence;
- collected-evidence inventory checked before any missing-proof statement;
- above-first-level/double-dip issue resolved or scoped out.

If any is missing, return a targeted `needs_evidence`/`needs_human` with evidence
owner and next action. Never issue a blanket “auditor provided no proof.”

## Difficulty / automation tier

- **Low (auto):** exact JADE tie, date-aligned on-site defense, corrected
  blank-title/name match, and fully documented Chevron A/B/C route.
- **Medium (HITL):** FeasEx candidate requiring Freedom/AFE retrieval, ambiguous
  activity or relocation evidence, BP-only exact-section unavailability, IPT/
  major-construction membership, or mixed precedent.
- **Hard:** above-first-level reporting-chain disputes, novel contract
  interpretation, inconsistent cross-asset precedent, or current activity that
  only the business can confirm.

## Output contribution

Write `evidence_packet.otl` with:

- `claim_subtype`, `auditor_claim_sufficiency`, and `operator_proof_coverage`;
- per-person JADE/period/amount tie and date-aligned location finding;
- `route` (`onsite`, `major_construction`, `ipt`, `3A`, `3B`, `3C`, or
  `unresolved`) and a requirement-by-requirement proof matrix;
- `outcome_basis` (`claim_refuted`, `chargeability`, `contract_dispositive`,
  `ineligibility`, or `mischarge`) and only the route-specific evidence block;
- technical-employee finding, above-first-level/double-dip screen, and precedent
  consistency;
- the eight-step `walkthrough`, unresolved evidence owner, and next action.

`agreement` carries the verified instrument, edition, exact subsection, quote,
page, and verification status. A generic settlement title is incomplete.
`evidence_requests` must exclude artifacts already in `collected_evidence` and
must distinguish auditor analysis from Operator-required documentation.
