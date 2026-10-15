# TYPE SKILL: Affiliate Labor Exceptions (SIEP / SEPCO / affiliate-entity labor)

**Applies to families:** `affiliate-labor`, `siep-labor`, `affiliate`, `intercompany-labor`,
`payroll-affiliate`, `wrong-property-labor`, `admin-in-nature-labor`
**Error-code anchors (authority_map):** COPAS **18.1 "affiliate"** section is the affiliate
anchor *where the exception cites it* (auditor free-text references "18.1 affiliate").
⚠ The auditor's *free-text title* ("SIEP", "affiliate", "payroll") is NOT reliable — go by the
COPAS error type / the description, not the label. Resolve the exact code at runtime.
**Source:** Jul 17 Affiliates Exception Type walkthrough (Scott Barrios, Swaminathan,
Jagadesh — affiliate billing flow, cost certificates, agreed rates, SEPCO/SIEP boundary,
IPT/major-construction pre-checks); Janelle "JV Audit Exception Workflow" (Great White
**Exception 3** — the $47K wrong-property walkthrough); Jul 1 "Perdido 2023 Exception 5"
(SIEP definition, CATS-vs-SIEP, COPAS 18.1).

## Runtime contract

1. Classify the **billing entity/path** before the exception: SEPCO time writing in
   SOI's SAP/JADE under the service-level authority; SIEP/another affiliate via an
   intercompany invoice; or an independent third-party charge passed through an affiliate.
2. Classify the **claim**, which determines the evidence: `cost_attestation`,
   `agreed_rate`, `otl_role`, `wrong_property`, or `nonlabor_pass_through`. Do not force
   every affiliate exception into the wrong-property pattern.
3. Tie the named person/vendor, project code, period, source amount/rate, billed
   amount/rate, and well/WBS to the JADE line or intercompany invoice.
4. Retrieve the asset/co-owner agreement. Use its affiliate, rate, IPT, major-
   construction, or OTL provision as applicable; never substitute a generic COPAS rule.
5. For an OTL-role claim, hand the eligibility question to `type_otl_labor`; keep
   entity, no-profit, agreed-rate, and invoice facts in this packet.
6. A correct certificate/rate/assignment supports `deny`; a proven rate or property
   mismatch supports quantified `partial`/`accept`; a missing certificate, invoice,
   role basis, or assignment confirmation is `needs_evidence`.

## What the auditor is usually claiming
Affiliate (usually SIEP) labor or pass-through cost was charged to the joint account. In
practice the auditor's claim falls into one of these branches:
- **Cost certificate / no-profit** — asks Shell to prove the affiliate billed at cost
  without operator profit, when required by the JOA/UOA/settlement.
- **Agreed-rate mismatch** — source affiliate invoice rate differs from the amount billed
  to the co-owner; the question is whether the co-owner's negotiated rate was applied.
- **Off-site technical / role eligibility** — challenges job duties, CFR mapping, IPT, or
  major-construction treatment. This is an OTL question carried on an affiliate invoice.
- **(a) Wrong property/asset** — the affiliate employee was actually working for a DIFFERENT
  asset (not the audited JV), so the labor should not have hit this joint account.
  [transcript: *"they saw... years that he was working for Olympus, other asset, and not
  working to the Great White account. But they saw that there was a charge for this employee
  for Great White"*]
- **(b) Administrative-in-nature by title** — the role/title looks administrative, so the
  auditor argues it is overhead-recoverable, not directly billable.
  [transcript: *"this is another type of an affiliate coding error, so different... wherein
  they're trying to go by the title that it's an administrative in nature"*]
- **(c) Lacks time-writing support** — no evidence the person worked on behalf of the joint
  account for the charged period. [transcript: *"no additional evidence was provided that
  supported that Mr. Blah, blah, blah was working on behalf of the joint account as opposed
  to the other joint venture"*]

## SIEP / affiliate definition (ground truth)
- **SIEP is a Shell affiliate company.** A SIEP employee is still a Shell employee, just
  *employed by another affiliate entity* — so the charge comes FROM that affiliate company.
  [transcript: *"SIEP is a shell affiliate company"*] and [transcript: *"Someone from SIP,
  it's a Shell employee, but they're just part of... we're still Shell, but they're employed
  by another affiliate company... whatever the charge that they're seeing here, it's coming
  from that affiliate company, which is SIP"*]
- **CATS vs SIEP is a *source/entity* distinction, not a different reconciliation universe.**
  CATS = person is on the SOI (Shell Offshore Inc.) payroll/books; SIEP/affiliate = person
  on another affiliate entity's books. [transcript: *"if you see cats, it means that they
  are in the soy books... our entity is Shell Offshore Incorporation... Affiliate employee
  to sleep labor"*]
- **SIEP/affiliate cost may be a pass-through OR a re-rated intercompany charge.** Sometimes
  the affiliate bills SOI at the same rate charged onward to the joint account; sometimes SOI
  is billed at one rate and the joint account/co-owners at a *different* agreed rate — this is
  intercompany billing governed by the settlement agreement / JOA affiliate section.
  [transcript: *"the C person's cost will sometimes be a pass-through cost... but there are
  times when seed builds soy at one rate and we in turn bill the joint account or the
  co-owners at a different rate. So this is... agreed"*] and [transcript: *"It's an
  intercompany billing"*].
- **Do NOT trust the auditor's free-text label.** "SIEP" / "affiliate" / "payroll" are named
  at the auditor's discretion and are inconsistent across exceptions; classify from the COPAS
  error type + description. [transcript: *"you may see SIP, you may not see SIP, you may see
  affiliate. It just depends how the auditor would want to name that exception... don't rely
  on that so much"*]

### SEPCO is not the same billing path as SIEP

SEPCO is the people-holding entity that can transact for SOI under a service-level
agreement/manual of authorities. Its employee time writing flows through the shared P16
SAP environment into JADE; it does **not** produce the separate intercompany affiliate
invoice seen for SIEP. SIEP is a genuinely separate affiliate/ERP path that invoices over.
Therefore, never request an intercompany invoice merely because a Shell employee's legal
employer is SEPCO: first inspect JADE/reference indicators and establish which path applies.

Independent third-party contractors are different again. Their invoice may legitimately
include the vendor's profit margin; the affiliate **at-cost/no-profit** limitation does not
erase an independent contractor's commercial margin. Still test the contractor's OTL role
and retrieve the vendor invoice when that is the auditor's question.

## Claim branches and disposition logic

### 1. Affiliate cost certificate / at-cost attestation

Some JOAs/UOAs/settlements require an independent accounting-firm attestation that the
affiliate costs billed are **at cost / without profit**. The certificate is the primary
answer to that claim; focus on the conclusion, not irrelevant procedural boilerplate. Verify
that the certificate covers the affiliate, cost population, and audit period in question.
Missing or out-of-period attestation → `needs_evidence`; a covering certificate supports
`deny` for the no-profit claim without demanding unrelated billing explanations. Inspect the
accounting firm's procedures and conclusion; an express at-cost/no-profit conclusion is the
key audit check. Do not use this no-profit test to disallow an independent third-party vendor
margin.

### 2. Agreed-rate rebilling

The affiliate source invoice and the co-owner bill can legitimately differ. The source
charge commonly lands in a **non-billable GL**, then Shell applies the co-owner-specific
agreement rate and flips the allowed amount to the billable GL. The negotiated amount may
be **higher or lower** than the affiliate invoice rate. Recompute:
`eligible units/hours × verified co-owner rate`, tie that to the billable posting, and
compare it with the amount charged. Correct agreed-rate application supports `deny`; a
proven variance supports `partial`/`accept` only for the quantified difference.

### 3. OTL / admin-in-nature role

If the auditor challenges whether the work is technical, administrative, on-site/off-site,
IPT, major construction, or CFR-backed, route the eligibility decision to
`type_otl_labor`. Keep the invoice/entity/rate facts here. Before requesting 50–60 employee
emails, check the verified agreement for an **integrated project team designation** and the
JADE overhead type for an applicable **major-construction statement of clarification**;
either can supply a more direct contract path. Otherwise use staff listing + job catalog +
cost object, then request project-specific duties only when the title/history cannot close
the mapping. A historical role description is reusable context, not proof of the person's
current project work.

Check the project overhead type before forcing CFR/A/B/C mapping. A verified Major
Construction clarification or approved IPT route can govern directly. For the Great White
UOA example in Scott's notes, the selected Section III Paragraph 2.B option permits qualifying
engineering/design/drafting technical services under its own conditions; do not impose a
redundant OTL CFR test. Apply its one-day/actual-day condition only to the exact instrument,
not as a global eight-hour rule.

### 4. Wrong property/asset

The auditor argues the affiliate employee worked on a different property. Go to the
business/employee and confirm the actual assignment for the **charged period**. If the
auditor's POB window falls outside the JADE/invoice dates, the time writing remains valid;
if confirmation shows time billed to this JV while actually assigned elsewhere, credit or
reclass only that portion.

### 5. Non-labor / third-party pass-through

Affiliate invoices occasionally include material or an independent contractor. Tie the
line to the project and obtain the underlying vendor invoice from the affiliate's ERP when
it is not in Shell's SAP packet. This is a documentation/chargeability branch, not a reason
to apply the employee OTL test blindly.

### Worked example (Great White Exception 3 — anchor case)
Joint account charged **~$47,000** for affiliate labor of employee **Kenneth**, role **"lead
production technologies"** (lead production technologist, drilling & completion). Auditor saw
via **POB / helicopter reservation** that Kenneth flew to **Olympus** (the rig **Proteus**),
a *different asset*, and assumed the Great White charge was for the wrong property.
[transcript: *"the joint account was charged 47,000 for the affiliate labor services of this
employee... a lead production technologies"*] Shell went to the business, the employee
confirmed he supported Great White except for a brief Olympus stint **Oct 31–Nov 11**, and
the actual Great White charges were only **two days in October (Oct 19–20) and zero in
November** — i.e. Shell never billed Great White for the Olympus period. Charge held valid.
[transcript: *"the charges to the great white for Kenneth were... only... two days in October
19 and 20 and zero days in November... This excludes the time period of October 31 to November
11, as noted on the POP report"*]

## Evidence to resolve (in order)
1. **Governing agreement + claim branch** — retrieve the co-owner/asset-specific
   affiliate, agreed-rate, attestation, IPT/major-construction, or OTL provision. Do not
   perform a rate or no-profit test without the term that creates it.
2. **Entity/billing path** — distinguish SEPCO time writing in SAP/JADE from a SIEP
   intercompany invoice and from an independent vendor pass-through.
3. **Intercompany invoice with the WELL NUMBER (SIEP path)** — the invoice carries the well
   number that ties the labor to the audited property. In the anchor case the invoice showed
   **GD09 / GD08**, the wells
   attributed to Perdido / Great White. This is the decisive billing evidence.
   [transcript: *"This is from the intercompany invoice. So since this guy is from SIP, so
   they do an intercompany charge and that invoice has the well number right here, the GD09
   and GD08, which is attributed to Perdido Great White"*]
   - Note: auditors sometimes reject the intercompany invoice because it is a Shell-to-Shell
     ("show-to-show") document, not a third-party invoice — but it is still the source that
     shows what well/JV was charged. [transcript: *"sometimes the auditors does not want to
     take it because it's a show-to-show invoice. It's not like a third party, it's an
     intercompany"*]
4. **CATS vs SIEP source** — confirm which entity's books the labor came from (SOI = CATS;
   another affiliate = SIEP/affiliate). This tells you whether it is a pass-through or a
   re-rated intercompany charge and which agreement governs the rate.
5. **JADE / time-writing + agreed-rate posting** — the line-item text and dates show what
   was charged and when. For a rate claim, capture source units/rate, the verified co-owner
   rate, the non-billable landing, and the billable flip; recompute the allowed amount.
   Compare the auditor's disputed window against what JADE shows was billed.
6. **Cost certificate (only for an attestation claim)** — verify affiliate, period, covered
   population, independent issuer, and at-cost/no-profit conclusion.
7. **Role evidence (only for an OTL claim)** — staff listing and job catalog establish the
   general role; JADE/invoice cost object and any prior validated descriptions provide
   context. Use project-specific employee confirmation only when those sources cannot make
   the contract/CFR mapping. Then hand off to `type_otl_labor`.
   Use `job_catalog_tool.lookup` for the source job profile; its generic description can
   corroborate role nature but cannot prove current duties or direct chargeability.
8. **POB / helicopter reservation / rig sequence** — these prove **presence/location** (that
   the employee was physically on some asset) but do **NOT** prove billing. They are the
   auditor's presence evidence; they do not, on their own, establish that the JV was charged
   for the disputed days. Treat them as corroborating location only.
   [transcript: *"this one has no relation or how did they charge you on the shade... This one
   right here is just a confirmation that this guy flew to this asset... It's just a way for
   the auditors to verify... if an employee is really working on the offshore or not"*]
9. **Business confirmation of in-scope period** — reach out to the business/employee to confirm
   the actual assignment for the audited period, because reservation data can be fat-fingered
   and does not reflect the true scope. [transcript: *"we have to reach out to the business if
   what the auditor understands... is correct. Because sometimes people can just fat finger
   stuff whenever they're doing this reservation"*] (The confirmation email itself may be
   internal-only and not attached to the co-owner response — restate the position in words.)

## HANDOFF boundary — Affiliate vs OTL (do NOT conflate)
This skill owns the **affiliate-entity / rate / attestation / property-assignment** question:
- Which billing path is used (SEPCO/SOI SAP time writing, SIEP invoice, or third party)?
- Is the intercompany rate the agreed rate?
- Does the required cost certificate cover the charge?
- Was the employee assigned to THIS property/asset for the charged period?

It does **NOT** own the **offsite-technical (OTL)** question. If the real dispute is whether
**off-site technical labor** qualifies as directly billable (the OTL A/B/C offsite-technical
test), **hand off to `type_otl_labor`** — even when the person happens to be a SIEP/affiliate
employee. The affiliate label does not make it an affiliate exception; the *substance* of the
auditor's claim decides ownership.
- Affiliate skill → entity/rate/attestation/property-assignment.
- OTL skill → offsite-technical A/B/C billability.
[The anchor exception's own response text references "OTL" in passing, showing the two can
appear together — keep them separate: resolve entity/property here, route offsite-technical
billability to OTL.]

## COPAS 18.1 anchor
Where the exception's error type cites **COPAS 18.1 "affiliate"**, that is the affiliate
anchor for this family. [transcript: *"sometimes it says 18.1 affiliate. So it's all that
referencing section from Copus" — "Copus, correct"*] COPAS gives the baseline classification
guidance, but per-JOA/co-owner terms can modify it — confirm against the specific JOA affiliate
section / settlement agreement. Do NOT assert 18.1 when the exception does not cite it.

## Confirmation gate
For the **wrong-property** subtype, no "charge is valid" conclusion is final without a
**business/employee confirmation** of the in-scope assignment. A no-profit conclusion needs
the covering **cost certificate**; a rate conclusion needs the verified agreement rate and
posting arithmetic. An OTL/admin conclusion needs the applicable contract route and the role
evidence required by `type_otl_labor`. Emit `needs_human` for any missing off-system item.

## Difficulty / automation tier
- **Low (auto):** entity-path lookup, invoice/well extraction, JADE period check, and
  agreed-rate arithmetic once the agreement rate is verified.
- **Medium (HITL):** cost-certificate coverage, wrong-property confirmation, and
  IPT/major-construction applicability.
- **Route out:** offsite-technical billability disputes → `type_otl_labor`.

## Output contribution
`evidence_packet.affiliate` = billing path (SEPCO/SOI, SIEP, or third party), source invoice +
well/project code, cost-certificate coverage when relevant, source/agreed/billed rates and
recomputed variance, JADE time-writing for the charged days, and the claim-branch result.
Presence docs
(POB/flight/rig sequence) are attached as *location-only* corroboration, never as billing
proof. If the dispute is really offsite-technical, hand the packet to `type_otl_labor`.
