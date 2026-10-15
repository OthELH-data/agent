# EXTRACTOR SKILL: Material / Reconciliation Workbooks

**Job:** read the attachments on a material-reconciliation exception and return a
**faithful, structured evidence packet** — charged vs installed quantities, CJI3
lines, iWells rows, offsets, and explicit gaps. **This skill EXTRACTS and TAGS
facts. It does NOT decide the outcome.**

**Applies to families:** `materials`, `inventory`, `reconciliation`, `octg`,
`each-item`, `surplus`, `transfer`

> **⛔ SCOPE BOUNDARY — read this first.**
Here you only:
> **(1) locate** the data, **(2) extract** it faithfully, **(3) tag** known
> patterns, **(4) flag** what's missing. Never adjudicate. When in doubt, extract
> the raw value and tag it `ambiguous` rather than resolving it.
>
> **Downstream:** the structured packet you emit feeds the reconciliation/decision step, which
> owns the actual disposition. Your job stops at faithful extraction + tagging + gaps.

---

## 0. Input map — what arrives and how to recognize it

Material-recon exceptions arrive as a workbook (or several), often `.xlsx`/`.xlsb`/
`.xlsm`, plus the auditor's narrative. File/tab names vary — identify by **content
signature**, not by name.

| Source | Recognize it by | What it holds |
|---|---|---|
| **Auditor description** | prose in the exception; dollar amount, footage, "credit due" | claimed material, disputed qty, per-ft rate, credit demanded |
| **Recon tab** | columns like charged/installed/variance, per-material rows | the co-owner's (or Shell's) reconciliation working |
| **CJI3 export** (SAP) | many rows; WBS/cost-object, doc #, doc type, MMR/material #, movement text, value; inception-to-date | complete transaction list incl. JV cutback |
| **iWells / IVLs report** | joints/feet per item **description** (no reliable MMR field) | what was physically installed/run in hole |
| **BOM** | bill-of-materials listing per well/WBS | historical planned/consumed list; not a universal current attachment |
| **JADE lines** | billed-to-co-owner subset (less than CJI3); sometimes a separate suspense extract | what was actually billed; suspense detail may later roll into a released line |
| **Vendor tally / inventory / schematic** | vendor-named file (Sumitomo, SLB, Superior/SNLC), email excerpt, or manual layout | stock, component lineage, or installation evidence when iWells omits the original material |

If an expected source is **absent**, do not infer it — record it in `gaps` (§5).

---

## 1. Parse the auditor claim (from the description)

Extract, verbatim where possible, then normalize:
- **Material identity:** size (e.g. `5.5"`, also written `5 1/2"`, `9 5/8"`,
  `14-in`), weight (`23#`, `53.5#`), grade, description, **MMR / material number**.
- **Charged quantity** (ft and/or joints) and **installed/set quantity** the auditor
  cites (e.g. *"2,212 ft charged, 937 ft set per iWells"*).
- **Per-unit rate** (`$75.22/ft`), **charged $**, **credit claimed $**, overhead if stated.
- **WBS / cost object / completion code**, well name, **audit exception #**, **audit #**.

**Normalization rules (do not skip):**
- **Size notation:** `5 1/2"` → `5.5`; `9 5/8"` → `9.625`; `14-in` → `14.0`.
- **Product-name variants are assembly candidates, not automatic matches.** Screens,
  blank pipe, pup joints, couplings, and tieback/tool assemblies may share material
  lineage, but group them only when an MMR/part link, `221` movement, vendor tally,
  component schematic, or written inventory confirmation establishes that link.
- **Units:** keep both feet and joints; note the source's unit of measure. Do **not**
  convert footage→joints here as a *decision* — you may compute it as a *derived
  orientation field using an explicitly recorded assumption, but tag it `derived`.
  The tool's 41-ft default is not evidence; actual invoice/tally joint lengths
  control, and Jul 20 examples varied materially by outside diameter.

---

## 1.5 Investigation order — reconstruct the auditor's argument

Work the evidence in **this order** to rebuild how the auditor reached the questioned amount. Each
step narrows to the next; capture what you find (and any break in the chain) as you go.

**Step 1 — Identify (from the auditor description).** Pull three things:
1. the **material name** — written like `4.5" 26# tubing` or a close variant (size in inches with a
   double-quote, weight as the number before `#`);
2. the **WBS code** — e.g. `C.US.GWH…` (may be only a prefix, or absent from the description — if so
   it is recovered from JADE's Cost Object column, see Step 3);
3. the **length / footage discrepancy** the auditor is questioning.

**Step 2 — Recon tab.** Find the `Recon` tab; the data table typically starts around row 10. Read the
`Material` column values, and capture the **Invoice number** and **Vendor name** for the disputed row.

**Step 3 — JADE.** In the JADE sheet, find the material in the **`Material /Service`** column and
filter on the **WBS code column** (`Cost Object WBS/CC (CO) - Key`, around col K). Rows matching the
material *and* the WBS mean the co-owner was in fact billed for that item. (When the description gave
no WBS code, the code of interest is chosen from JADE's Cost Object codes by matching its description
to the well/AFE/project.)

**Step 4 — iWells (under the `WBS_Recon to Auditor` workbook).** In the iWells tab, filter
`Item Description` for every variant that matches the material name; you can also filter the **Size**
column (inches, in double quotes) and the **Weight** column (the number before `#`). **Sum the
length** over the filtered rows. If that total reconciles to the length the auditor questioned,
proceed to Step 5.

**Step 5 — CJI3.** Filter CJI3 on `Material` for the disputed material, read **`Total quantity`**,
and confirm it against the **PUM** (possible/plausible unit of measure — for tubing, PUM = feet/`FT`)
so the quantity makes sense. If CJI3 resolves the purchased-vs-consumed discrepancy, explain the
reasoning. **If a discrepancy still remains, flag `human review`.**

**Step 6 — Bill of Materials (only if a BOM is provided).** Join the BOM to the other sources:
Shell **MMR** = the material codes in Recon = the JADE material columns
(`MATSERV` / `Material /Service` / `Material Service` / `Material`). Filter the BOM's
**`Nominal Section Outside Dia (in)`** column for the tubing OD, and read the **`status`** column,
which shows what became of the material (installed, transferred to another project, etc.).

---

## 2. Extract the Recon tab

- Read each material row: **charged qty, installed/used qty, variance, rate, value.**
- Preserve the source's own numbers exactly; add a `source_cell` reference where
  feasible so a human can trace it.
- If merged/multi-row headers obscure columns, extract best-effort and tag
  `header_ambiguous` rather than guessing a mapping.

---

## 3. Extract CJI3 lines (SAP) — the primary transaction source

- Pull rows at the **WBS level**, inception-to-date, filtered by the **MMR** (or,
  if MMR absent, by material description). CJI3 includes JV cutback; JADE is only
  the billed subset — capture **both** if present and note which is which.
- For **each line**, capture: doc #, **document type**, date, MMR/material #,
  description, movement type (goods-issue / goods-return / transfer /
  condition-change), quantity, **individual line/invoice value**, PO/reference,
  WBS/cost object.
- **Tag (do not adjudicate)** each line with any that apply:
  - `goods_issue` (GI) — material issued to rig / run in hole.
  - `goods_return` — return entry (part of a chain-of-custody, if present).
  - `transfer` — SAP credit entry moving material to another WBS/project.
  - `offset_candidate` — debit/credit pair netting to ~zero, incl. **condition-change
    codes A/B/C/D**.
  - `accessory_issue` — movement `221`, a cue that material was issued for
    fabrication/accessories; retain both A2GL/A7GL legs for later netting.
  - `installation_movement` — movement `281`, an observed installed-material cue.
  - `zero_net_control` — `Z5`/`Z6` control or approval entries, but only after the
    paired values are extracted so the recon step can confirm zero net effect.
  - `manual_journal` — document type `ZU` or `SA`; capture the posting support and
    linked PO/invoice instead of assuming a physical-goods movement exists.
  - `suspense_candidate` — asterisk, document type `JS`, or separate suspense-JADE
    source. These are cues only; extract the possible released roll-up counterpart.
  - `seven_series` — **material # starts with `7` + zeros**, OR description begins
    with **"OA"/"Ohe"** (service/fee, not physical material).
  - `sub_10k` — **individual** line/invoice value **< $10,000**. Never infer this
    from the material prefix or from the aggregate exception amount.
  - `non_trackable` — rethreading, tees, consignment fees, accessories.
- These tags are **hints for the recon skill's gates**, not classifications of the
  final 5-way annotation (matched/offset/new/transferred/installed) — that
  annotation is a *decision* the recon step makes.

---

## 4. Extract iWells / vendor installed quantities

- **iWells:** filter on **item description** (no reliable MMR field). Sum the
  **installed length / joints** only for rows whose identity/assembly linkage is
  supported under §1. Capture the rows used and total. Do not assume 41 ft/joint:
  retain actual tally/invoice joint lengths where available.
- **Vendor tally / inventory / schematic:** when iWells doesn't capture the
  original material (for example, tubing transformed into screens/accessories),
  extract quantity plus the MMR/part/serial/assembly linkage. Tag `vendor_manual`
  and `visibility_review`; acceptance and shareability are case-specific. The
  visibility tool must classify the exact artifact before it is cited.

---

## 5. Gaps — flag what's missing (this is a first-class output)

Explicitly record anything that blocks a clean reconciliation. These feed the recon
skill's `needs_evidence` / `needs_human` decisions — **the extractor's honesty here
is what prevents a false "deny."**
- **No WBS / cost-object tie** on the charged lines.
- **No iWells row** for the disputed material (installed qty unconfirmable).
- **No goods-return / manifest** chain present (relevant to presumption-installed).
- **WBS status = REL (pre-TECO)** → note that **no final reconciliation exists yet**
  (available 6 months after TECO) — extract the status, don't rule prematurity.
- **Provenance unknown** — a cited document (CJI3/iWells/BOM) that isn't in the
  attachments Shell provided.
- **BOM-era unknown** — a response assumes a BOM should exist without establishing
  the historical/current practice that applied when the response was prepared.
- **Possible suspense duplicate** — suspense detail and a released JADE roll-up
  appear to represent the same economic transactions but have not been reconciled.
- **Assembly lineage missing** — a transformed component is asserted to contain the
  disputed material without a tally, schematic, movement, or part/serial link.
- **Audit-cutoff blind spot** — transactions dated after the co-owner's audit period
  that they couldn't have seen.
- **Header/label ambiguity** that forced a best-effort mapping.

---

## 6. Output contract (the structured packet handed to `excel_evidence`)

Return a single object shaped like this (illustrative keys — keep them stable):

```
{
  "material": { "size_in": 5.5, "weight_ppf": 23, "grade": null,
                "mmr": null, "description": "5.5\" 23# tubing" },
  "claim":    { "charged_ft": 2212.0, "installed_ft": 937.0,
                "disputed_ft": 1275.0, "per_ft_rate": 75.22,
                "charged_usd": 166346.79, "credit_claimed_usd": 95132.93,
                "wbs": "C.US.GWH.PD.18.019.0943A", "well": "Great White",
                "exception_no": "0010", "unit_of_measure": "feet" },
  "recon_tab": [ { "material": "...", "charged": ..., "installed": ...,
                   "variance": ..., "source_cell": "Recon!D14" } ],
  "cji3_lines": [ { "doc": "...", "doc_type": "ZU", "date": "...",
                    "mmr": "...", "desc": "...", "movement": "manual_journal",
                    "qty": ..., "line_value": ..., "po_reference": "...",
                    "tags": ["seven_series", "manual_journal"] } ],
  "iwells_rows": [ { "description": "...", "joints": ..., "feet": ... } ],
  "installed_summary": { "iwells_ft": 937.0, "iwells_joints": 22.85,
                         "actual_joint_len_ft": null,
                         "orientation_joint_len_ft": 41.0,
                         "orientation_only": true },
  "vendor_manual_sources": [ /* visibility_review, if any */ ],
  "gaps": [ "no_iwells_row_for_couplings", "wbs_status_REL_pre_TECO" ],
  "provenance": { "sources_present": ["cji3","iwells","recon_tab"],
                  "sources_missing": ["bom"] },
  "extractor_notes": "free-text: anything a human should double-check"
}
```

**Contract rules:**
- Every numeric field is **either extracted (with a source) or `null`** — never
  invented. Derived values are tagged `derived: true`.
- `gaps` and `provenance` are **required** even when empty (`[]`) — their emptiness
  is itself a signal.
- Tags are descriptive, never dispositive.

---

## What this skill must NOT do (guardrails)
- ❌ Apply the **3-joint threshold** or decide it's a straight-deny.
- ❌ Apply the **1% thread-loss** verdict or rule footage==joints "invalid."
- ❌ Decide **joint-vs-footage** basis, or that OCTG "should be denied."
- ❌ Assign the final **5-way CJI3 annotation** (matched/offset/new/transferred/
  installed) — extract the raw movement + tag; the recon skill annotates.
- ❌ Emit a **disposition** (`deny` / `credit` / `needs_evidence`) — that's the
  recon skill's job. Emit **facts + tags + gaps** only.
- ❌ Assert an **MFI/COPAS sub-section** — only capture the error-type code if the
  workbook states it.
- ❌ Treat a `7...`/OA identifier, `JS`/asterisk, `ZU`/`SA`, `221`, `281`, or
  `Z5`/`Z6` cue as a final disposition without the value, paired entries, and
  supporting-document context.

## Difficulty / automation tier
- **Low (auto):** parse the claim, read the Recon tab, extract iWells candidates,
  tag movement/document/value cues, and normalize size notation.
- **Medium (HITL):** ambiguous/merged headers, assembly-lineage grouping,
  suspense-roll-up matching, and vendor-manual files with mixed visibility.
- **Hard:** workbooks where sources disagree on quantity and only a human can decide
  which column is authoritative — extract all, tag the conflict, escalate.
