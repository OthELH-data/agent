# JV Audit Precedent Wiki — SCHEMA & Conventions

This wiki is a **compounding precedent knowledge base** for Shell JV audit-exception
defense. It follows the LLM-Wiki pattern: the LLM **owns and maintains** these
markdown pages; humans read them. Raw sources (exception registers, operator
replies, attachments) are immutable and live outside the wiki — pages here are the
*synthesis* built up from them.

The point: **the LLM should never re-derive precedent from scratch.** Each resolved
exception is integrated into the wiki once, and the knowledge compounds. Ask "how do
we defend an affiliate-labor overhead dispute?" and the answer is already written,
refined across every prior instance, with the winning clauses and the auditor's
usual weak points already noted.

---

## Directory layout

```
wiki/kb/
  index.md              # catalog of every page — READ THIS FIRST on any query
  log.md                # append-only chronological record of ingests/queries
  types/                # one page per recurring EXCEPTION TYPE (the core precedents)
  clauses/              # one page per governing CLAUSE / COPAS MFI interpretation
  precedents/           # learned FACTS ("X upgrade is always billed OpEx not CapEx")
  entities/             # auditors, affiliates, assets, vendors
```

## Page types & when to create them

### `types/<slug>.md` — Exception-type precedent page (THE core artifact)
One per recurring dispute type: `affiliate-labor`, `overhead-admin`, `otl-cboh`,
`material-reconciliation`, `coding-allocation`, `contractor`, `transportation-marine`,
`duplicate`, `payroll-labor`, `documentation`. This page COMPOUNDS — every new
instance of the type refines it. Structure:
- **frontmatter**: `category`, `copas_edition`, `times_seen`, `assets`,
  `typical_disposition`, `updated`, plus the historical labor/rebuttal aggregates
  defined below
- **## Dispute pattern** — what the auditor typically claims
- **## Shell's defense** — the winning interpretive angle, refined over instances
- **## Governing clauses** — `[[wikilinks]]` to clause pages that resolve it
- **## Auditor weak points** — where their evidence/interpretation is typically thin
- **## Instances** — a table: exception_id · asset · $ · disposition · note (one row per ingested case)
- **## Learned facts** — `[[wikilinks]]` to precedent pages

### Historical labor-role and rebuttal aggregates

Labor exception-type and learned-precedent pages carry two nested, append-only
front-matter structures. They are historical guidance, never proof of an incoming
case:

- `role_outcomes` is keyed by a normalized role. Each entry retains
  `display_role`, `aliases`, per-outcome case counts, and a `reasonings` list.
  Every reasoning records `outcome` (`justified`, `conceded`, `mixed`, or
  `unresolved`), the reason, governing basis when actually present, subjects,
  source case, and source rebuttal round. Repeated role names are deduplicated;
  distinct supported reasons are appended rather than overwritten.
- `rebuttal_cases` is keyed by exact `asset|year|exception` case key. Every
  training case touched by the page is represented, including cases for which no
  true rebuttal is recorded in the frozen artifact. Cases with true rebuttals
  retain the R0 response, every substantive R1+ round, why follow-up occurred,
  what the co-owner pursued, how Shell responded, any attachment change, closure
  status, and a grounded R0 avoidance lesson.
- `rebuttal_case_count`, `no_true_rebuttal_case_count`, and
  `total_true_rebuttal_rounds` are recomputed from `rebuttal_cases`; model output
  cannot inflate them.

"No true rebuttal recorded" means only that no later substantive auditor
description appears in the supplied, versioned rebuttal artifact. It is not an
assertion that no later exchange exists anywhere. A closure/status-only row is
not a true rebuttal.

### `clauses/<slug>.md` — Clause interpretation page
One per governing clause / COPAS MFI actually cited (e.g. `coua-exhibit-c-section-iii`,
`copas-mfi-21-overhead`, `settlement-billing-standards`). Structure:
- **frontmatter**: `source_doc`, `section`, `copas_edition`, `applies_to_assets`, `scope`
- **## What it says** — the verbatim/near-verbatim rule + `source_page`
- **## Correct scope** — what the clause ACTUALLY governs (critical for tip #1 below)
- **## How Shell uses it** — the defense application
- **## ⚠ Out-of-context watch** — documented instances where an AUDITOR cited this
  clause out of context / over-generalized it (e.g. clause is wells-only but they
  applied it to a non-wells charge). THIS IS A FIRST-CLASS FIELD.
- **## Used in** — `[[wikilinks]]` to type pages / instances

### `precedents/<slug>.md` — Learned-fact page
A durable factual precedent the corpus teaches, e.g.
`chemical-upgrades-billed-opex.md`: "Reservoir chemical upgrades of type X have
consistently been billed as Operating Expense, not Development — auditor
reclassification attempts have been denied in <cases>." Structure:
- **frontmatter**: legacy discovery fields `fact_type`, `confidence`,
  `times_confirmed`, `assets`, plus this required structured signature:
  `material_type`, `allegation`, `decisive_issue`, `evidence_available`,
  `evidence_that_closed_case`, `governing_assets`, `governing_agreements`,
  `confirmation_count`, plus `validation_status`.
- **historical aggregate frontmatter**: the same `role_outcomes`,
  `rebuttal_cases`, and recomputed count fields described above. This allows an
  otherwise-equivalent precedent to retain both examples that drew rebuttals and
  examples that did not.
- **## The fact** · **## Basis** (clauses + instances) · **## Contradictions** (if any)

The structured signature controls whether inference may treat a page as a direct
precedent. Use stable snake-case values. `evidence_available` lists the evidence
pattern available before the decisive closing artifact was applied; general
evidence classes include
`iwells`, `cji3_ledger`, `goods_issue`, `goods_transfer`,
`reconciliation_workbook`, `invoice_or_po`, and `return_manifest`.
`evidence_that_closed_case` separately records the decisive artifact(s), not every
attachment. `confirmation_count` is the number of independently resolved source
cases and should mirror `times_confirmed`.

A page is eligible for a **strong/direct** match only when its signature is
complete and the incoming exception has the same material type, allegation,
decisive issue, and evidence pattern under the same governing asset/agreement.
A useful match from another asset or agreement is explicitly **analogical**,
never direct. Legacy pages without a complete signature remain weak background
until curated or re-ingested.

`validation_status` is `validated`, `needs_review`, or `retired`. Only a
`validated` page may become direct or analogical guidance. A historical
disposition alone is not validation: the page must identify the decisive
closing evidence. `needs_review` pages remain searchable as weak context while
their source case is checked.

### `entities/<slug>.md` — Entity page
Auditors (`auditor-jordon-lagrone`), affiliates (`baker-petrolite`), assets
(`asset-perdido` with its host/tieback relations), vendors. Track behavior/patterns
(e.g. "this auditor tends to dispute labor coding").

---

## Wikilinks & cross-referencing
Use `[[clauses/coua-exhibit-c-section-iii]]` style links liberally. A single ingest
should touch **many** pages: the type page, each cited clause page, any precedent
fact, and the auditor/affiliate entity pages. Cross-references are the value.

## index.md (content catalog — read first)
Every page listed under its category with a one-line summary + key metadata
(times_seen, disposition). On QUERY, the LLM reads index.md, picks the relevant
type + clause pages, then drills in. This avoids embedding-RAG at this scale.

## log.md (chronological, append-only)
Every entry prefixed `## [YYYY-MM-DD] ingest | <exc_id> <asset> <type>` or
`## [YYYY-MM-DD] query | <exc_id>`. Greppable: `grep "^## \[" log.md | tail -5`.

---

## INGEST workflow (Agent-Wiki writer, at end of BUILD pipeline)
Given a resolved exception (category, disposition, cited clauses, defense argument,
auditor claims, entities):
1. Upsert `types/<category>.md`: add an Instances row; refine Shell's-defense and
   auditor-weak-points sections if this case adds nuance; bump `times_seen`.
2. For each cited clause: upsert `clauses/<slug>.md` — record how it was used; if the
   verifier flagged the auditor citing it out of context, add an **Out-of-context
   watch** entry.
3. If the case establishes/confirms a factual precedent, upsert `precedents/<slug>.md`.
4. Upsert `entities/` for the auditor, affiliate, asset involved.
5. Update `index.md`; append to `log.md`.
6. For labor cases, merge supported role outcomes and the exact case's frozen
   rebuttal chronology into the type/precedent aggregates. Never manufacture a
   role outcome from a case-level disposition, and never collapse multiple
   rebuttal rounds into one generic note.

## QUERY workflow (inference)
1. Read `index.md`. 2. Select the matching `types/` page + its linked `clauses/` and
`precedents/` pages. 3. Return that precedent context (defense angle, winning clauses,
auditor weak points, learned facts) to the drafter, with a confidence based on how
well the incoming exception matches an existing type + its instance count.

---

## Client-taught rules the maintainer MUST encode
1. **Scrutinize clause scope.** When recording a clause, always fill *Correct scope*
   and watch for auditors over-generalizing a specific clause (wells-only cited on a
   non-wells charge). Weak-evidence spots go in *Auditor weak points*.
2. **Learn precedent facts.** "This upgrade type was always billed OpEx" → a
   `precedents/` page that future cases cite.
3. **escalate_commercial** is a valid disposition — record it on the type page when a
   dispute couldn't be agreed and went to the commercial team (may stay open years).
4. **TECO / closed-WBS**: keep `concepts`/precedent note — a closed (TECO'd) WBS must
   be reopened to pay an audit amount; failing to wait for TECO lets auditors preserve
   audit rights and reconcile next year. Flag exceptions touching closed WBS.
5. **No-evidence call-out**: if the auditor asserts a claim with NO supporting evidence
   attached, the type/clause page notes it and the defense calls it out explicitly
   (the GW-2022 Ex-7 pattern: "the support doesn't seem to be attached to verify this").
