# Architecture

The **Excel Exhibit Explainer** reads an auditor's **audit-report PDF** plus its **Excel exhibits**
(and any sibling **vendor-invoice PDFs**) and produces a JSON that explains *what the evidence
demonstrates* for a given audit exception — pinpointing the specific figures, columns, and rows the
auditor most plausibly used to reach both their reasoning and the questioned dollar amount.

Python does the mechanical work (read PDF text, read Excel tabs, stream/filter ledgers, count
tokens, write output); an LLM — reached through the **IBM ICA gateway** (a LiteLLM proxy that speaks
both dialects; we use its **Anthropic-native** `/messages` route, so prompt caching works and token
counts are exact) or a native **Azure OpenAI** deployment, in three model tiers — does all
interpretation. The package is self-contained: it depends only on
PyMuPDF, openpyxl / pyxlsb / xlrd, anthropic, langchain-openai, pydantic, tiktoken, python-dotenv, and
(optionally) MSAL + Office365-REST for SharePoint output and pytesseract/pillow for invoice OCR.

**Grounding is the central design constraint:** every prompt forbids inventing, inferring, or
computing a value that is not literally printed in the evidence. Unknown abbreviations/codes are
carried verbatim rather than guessed. The output is an *understanding aid*, never a verdict — the
synthesis explains the auditor's reasoning and records gaps in `open_questions`; it never emits a
deny/credit/resolved disposition.

---

## 1. Entry points

There are three ways in. All ultimately call `build_report()` in [run.py](run.py).

### `run.py` — the core CLI (single exception)
There is **one run mode**: resolve a case id, then process every rebuttal cycle it appears in. The
remaining flags are diagnostics or modifiers.

| Mode | Trigger | What it does |
|------|---------|--------------|
| **Rebuttal-cycle** | `--case-id GreatWhite_2024_0040` | Resolves the case, discovers every rebuttal cycle the exception appears in, and writes one JSON per cycle (chained). |
| **Dry run** | `--dry-run` | Resolves the case, prints one line, exits before any LLM call or sink write. |
| **LLM config** | `--llm-config` | Prints the resolved provider / per-tier model / endpoint / auth mode; no network call. |
| **Smoke test** | `--smoke-test` | One round-trip per model tier; prints replies and exits. |
| **Token map** | `--token-map` | Pings `GET /models`, prints each gateway model → its fallback estimator encoding (used only when the provider reports no usage), exits. |

A case id (`<Asset>_<year>_<exc>`) is structurally complete, so it is the **only** required input: the
entity folder is matched off disk under `default_corpus_base()` (the corpus root is a constant, not
configurable), the exception id is normalized to the corpus's 4-digit form, and the topic is looked up
in `datastores/clustered_exception_tables.csv`. No cases CSV is read. Output routing is decided by a
**sink** built up front (`make_sink()`), so SharePoint auth fails fast before any LLM tokens are spent;
already-processed exceptions are skipped (their output folder already holds a synthesis JSON).

Modifiers: `--topic <TOPIC>` overrides the catalog lookup (load-bearing — it selects the skill and
lands in the JSON as `exception_category`); `--skill <name>` forces a specific skill; `--no-skill`
disables skill resolution and runs the plain pipeline; `--provider {azure,ica}` picks the LLM backend
for this run (see §3). Invoice PDFs sitting beside the exhibits are always auto-discovered.

### `run_batch.py` — parallel driver (many exceptions)
Reads a cases CSV (default `datastores/test_cases_46.csv`), resolves each row to a `Case`
(via [case_resolver.py](extractor/case_resolver.py)), and runs each as an **isolated `run.py`
subprocess**, N-at-a-time (`--jobs`, default 4) through a `ThreadPoolExecutor`. Process-level
isolation (not threads) is deliberate: each case gets its own `USAGE` token counter and output sink,
so nothing shared is mutated and per-exception `token_usage` stays correct — the worker threads only
block on `subprocess.run`. Pre-flight (serial, in the parent): idempotently extract any
`Exception-*.zip` archives, and — if output is SharePoint — warm the MSAL token cache once so workers
authenticate silently. `--type <slug>` filters by exception topic, `--year` / `--exc` / `--case-id`
narrow further, `--dry-run` lists resolved cases, and `--limit N` caps the count. Each subprocess's
captured stdout+stderr is printed as one contiguous block to avoid interleaving.

**The child contract is the case id alone**: `_run_case` invokes `run.py --case-id <id>` (plus
`--topic` when the CSV supplies one, and `--no-skill` when set). Everything else the child needs is
re-derived from the id, so the parent and child agree by construction rather than by five flags kept
in sync. `--provider` is applied in the parent via `llm.set_provider()`, which re-exports
`LLM_PROVIDER` so every subprocess inherits it.

### Shell wrappers
- **`run_test_cases.sh`** — thin wrapper over `run_batch.py`; rewrites a bare positional slug into
  `--type <slug>` and forwards all flags. It no longer injects `--csv`: the default cases CSV now lives
  in one place, `resolve_cases_csv()`.
- **`run_exc_batch.sh`** — a hand-curated demonstration batch, now a list of 11 case ids run serially:
  Part A (5 Great White exceptions), Part B (5 material-reconciliation exceptions with invoices),
  Part C (one exception spanning three rebuttal cycles). Also extracts the 2023 Great White exhibit
  zip once. Only `--provider` / `--dry-run` / `--no-skill` are forwarded.

---

## 2. End-to-end flow

```
run.py main()
  └─ build_report(report_pdf, exhibits, exception, title, case_id, …)
       1. extract_pdf(report)                          → report text (PyMuPDF, OCR fallback)
       2. extract_exception_description(text, code)     → the exception's description block
          extract_rebuttal_dialogue(text, code)         → prior operator/auditor dialogue (if any)
       3. resolve_skill(category|title)                 → Skill | None   (skill_<slug>.md)
       4. IF skill:  ── WBS pre-filter ──
            collect_wbs_candidates(path) ∀ exhibits     → {code: description} map
            select_wbs_focus(description, wbs_map)       → WbsFocus (codes | unsure)  [light LLM]
       5. FOR each exhibit:
            extract_excel_full(path, wbs_codes)  (skill) │ extract_excel_bounded(path) (plain)
            interpret_attachment(description, sheets)     → per-tab + per-file summaries [med LLM]
            (skill) sheets_to_json(book)                  → rows accumulated for the skill scan
       6. IF skill:  apply_skill(skill, sheets_json, …)  → chunked scan of ALL rows  [med + light LLM]
       7. discover_invoices + extract_invoice ∀ PDFs
            build_invoice_evidence(description, inv)      → transcribe [light] + reconcile [med]
       8. synthesize(description, attachments, invoices, skill_analysis, previous_rebuttals, …)
                                                          → ExhibitEvidenceReport  [high LLM]
       9. attach token_usage (provider-reported on both backends) + elapsed_minutes
  └─ sink.write(<year>/<asset>/<exc>/rebuttal_<n>_synthesis.json, report)
```

**Two pipeline variants** are chosen by whether a skill resolves:
- **Plain pipeline** (no skill): exhibits read *bounded* (sampled), per-tab/per-file interpretation,
  synthesis. This is the path for labor/allocation-type exceptions.
- **Skill pipeline** (a `skill_<slug>.md` matched the category): exhibits read *fully* (all rows),
  a WBS pre-filter narrows huge JADE ledgers, and a dedicated chunked **skill scan** walks every
  relevant row before synthesis. This is the material-reconciliation path.

### Rebuttal-cycle mode (`run_rebuttal_cycles`)
The audit report is a running document — later cycles embed the full prior operator/auditor dialogue.
This mode discovers each `Rebuttal N` cycle folder, and for each cycle runs the full pipeline scoped
to the **auditor side only**. Cycles are **chained**: the oldest cycle syntheses are folded into a
running summary (rolled forward every 3rd appearance via the high-tier `summarize_rebuttals`) and the
most recent stay verbatim — assembled by `build_previous_rebuttals()` and fed into that cycle's
synthesis so it stays continuous with what came before. Token windows are managed by the caller
(`reset_usage=False`) so the summary-fold cost lands in the right cycle's `token_usage`. A cycle with
a report but no exhibits is still processed (the synthesis records "insufficient evidence"); a bad
report format is caught and skipped rather than killing the chain.

---

## 3. The LLM layer

### Model tiers ([shared/llm.py](../shared/llm.py))
Three tiers map to gateway model ids (overridable via `MODEL_*` env):

| Tier | Default model | Used for |
|------|---------------|----------|
| `light` | `claude-haiku-4-5` | per-tab hints, invoice transcription, WBS selection, findings summary |
| `med` | `claude-sonnet-5` | per-tab / per-file interpretation, per-invoice reconciliation, skill scan |
| `high` | `claude-opus-5` | cross-evidence synthesis, rebuttal-history summary |

These three ids back the `tiers` section of `configs/model_pricing.json`. Rates are resolved by model
id first, so an Azure run prices off that file's `models` section instead; a tier that falls back to
tier rates written for a different id is flagged in `token_usage.cost.model_mismatch` rather than
silently mispriced.

Key functions:
- **`get_llm(tier, temperature, max_tokens)`** — `@lru_cache`d `ChatOpenAI`. Applies a per-tier
  `max_tokens` cap (128K high/med, 64K light) so long structured replies aren't truncated mid-object;
  omits `temperature` for models in `_NO_TEMPERATURE` (opus); `timeout=_REQUEST_TIMEOUT` (default
  300s, env `MODEL_TIMEOUT`).
- **`structured_call(tier, system, user, schema)`** — embeds the Pydantic JSON schema in the system
  message, calls the model, extracts the **first balanced JSON object** (`_extract_json`), validates
  against the schema, and **retries on parse/validation failure only** (re-prompting for pure valid
  JSON). Records token usage every attempt.
- **`text_call(tier, system, user)`** — plain-text variant (used for the rolling summaries).
- **`_invoke(llm, messages)`** — wraps `llm.invoke()` with **exponential-backoff retry on transient
  transport errors** (`APITimeoutError`/`APIConnectionError`/`RateLimitError`/`InternalServerError`).
  These are raised straight out of `invoke()` and would otherwise bypass `structured_call`'s
  parse-retry loop and abort (and SKIP) an entire rebuttal cycle. Retries: `MODEL_TRANSIENT_RETRIES`
  (default 3). This is orthogonal to the parse/validation retry.
- **Prompt caching (native ica path only)** — `_to_anthropic` marks `cache_control: {"type":
  "ephemeral"}` breakpoints so a repeated prefix bills at ~0.1x instead of 1x. Two things are marked,
  both gated on `_CACHE_MIN_TOKENS` (2500) so a small prompt is never marked: a long system prefix
  (`_system_blocks`), and the **ReAct message buffer** — the prompt that actually repeats, since
  `react_loop` only ever appends. Two rolling breakpoints ride the last two user turns, so a write
  refreshes the whole prefix and the older mark still hits if the newest has aged out (5-minute TTL).
  Marking is deliberately **limited to buffers that repeat**: the signal is `conv_key`, which only
  `react_loop` passes, carried to `_to_anthropic` through a `ContextVar` so `_invoke` stays
  provider-blind. A one-shot `structured_call` is left unmarked, because a write with no later read
  costs more than sending the prompt plain. Env: `MODEL_CACHE_CONVERSATION`,
  `MODEL_CACHE_SYSTEM_PREFIX`.
- **Cache lifetime** — `MODEL_CACHE_TTL` is `1h` (the default) or `5m`, resolved in
  `token_accounting.cache_ttl` because it is a **billing** choice as much as a timing one. A 1-hour
  write costs about **2x** the input rate against **1.25x** for five minutes, so it needs three reads
  to pay for itself rather than two; in exchange it survives a gap between steps that would expire a
  5-minute entry, and one `run_python` step can run for 150s. `1h` adds an explicit `ttl` to the
  `cache_control` block; `5m` sends the bare block, which is the shape this gateway was verified
  against. Neither needs a beta header. `pricing.py` reads the same flag and prices the matching
  `cache_write_<ttl>` column from `configs/model_pricing.json`, so flipping the variable reprices the
  run instead of silently under-reporting it.
- **`_extract_json` / `_loads` / `_repair_escapes`** — pull the balanced JSON object out of a reply;
  on an "Invalid \escape" error, double stray backslashes (a lone `\` in an inch-mark, weight, or
  Windows path) so the string parses as the literal text the model intended.
- **`discover_models` / `smoke_test`** — connectivity/diagnostic helpers behind `--token-map` /
  `--smoke-test`.

### Orchestration ([explain.py](extractor/explain.py))
All prompt construction and LLM passes live here. The `_GROUNDING` clause and `_WBS_KNOWLEDGE` block
(how to read a WBS code: project family, asset/field token, project-type classifier, settlement &
billability rules) are appended to the relevant system prompts.

| Function | Tier | Role |
|----------|------|------|
| `interpret_tab` | med | Explains one worksheet tab vs. the exception; extracts `auditor_figures_used` (esp. invoice#↔amount rows). Injects the recognized tab schema + a **raw positional cell grid** so stacked/offset sub-tables are read by true column index. |
| `interpret_attachment` | med | Runs `interpret_tab` per tab, then rolls them into one per-file summary (`role` + `overall_summary`). |
| `select_wbs_focus` | light | Given the description + the `{code: description}` candidate map, picks the WBS code(s) of interest (matching the well/AFE/project) or returns `unsure`; chunks + unions for very large candidate sets. |
| `extract_invoice_fields` | light | Pass 1: faithfully transcribes an entire billing document (coversheet, invoice page, milestones, SAP outputs) into `InvoiceExtraction`. Pure transcription, no reasoning. |
| `contextualize_invoice` | med | Pass 2: reconciles the transcribed invoice against the exhibit interpretations — anchors on the **invoice number** first, falls back to PO/line/description matching; refuses to force a match. |
| `build_invoice_evidence` | light+med | Chains the two invoice passes into one `InvoiceEvidence`. |
| `summarize_rebuttals` | high | Folds a window of prior cycle syntheses into the running rebuttal summary. |
| `synthesize` | high | Cross-evidence synthesis → `ExhibitEvidenceReport`. Assembles exhibit blocks + invoice section + skill-analysis block + prior-rebuttal history + operator dialogue. Writes the synthesis in domain terms (JADE ledger, CJI3, materials reconciliation, BOM, iWells) and **never** uses spreadsheet-mechanics words. |

Helpers: `_render_tab` (compact markdown table with row/col caps), `_exhibit_context` (ledger
line-item anchors for invoice reconciliation), `_render_previous_rebuttals`,
`_render_operator_dialogue`, `_wbs_user_prompt`.

---

## 4. Extraction layer

### PDF — [pdf_extract.py](extractor/pdf_extract.py)
- `extract_pdf(path)` — PyMuPDF text per page; if a page is near-empty it OCRs (`_ocr_page`, requires
  the tesseract binary). Returns `{text, pages, extraction_method, …}`.
- `classify_method(pages)` — `text` / `ocr` / `mixed` / `empty`.
- `_normalize_ws` — folds nbsp and other unicode spaces to plain spaces.

### Exception text — [description.py](extractor/description.py)
- `extract_exception_description(text, code, title)` — locates the exception's `EXCEPTION DETAILS`
  block by code (via `_code_variants` to tolerate `0002`/`2`/`EX-02` forms), slices the `Description:`
  body, with fallbacks to a summary-table row (`_extract_from_summary_table`) and finally the title.
- `extract_rebuttal_dialogue(text, code, title)` — pulls the operator/auditor back-and-forth recorded
  in the running audit report.

### Excel — [excel_bounded.py](extractor/excel_bounded.py)
The workhorse. Reads `.xlsx`/`.xlsm`/`.xlsb`/`.xls` via **content-based dispatch** (`_reader_order`
peeks OLE2 vs. ZIP magic and the zip namelist), so a file with the wrong extension still parses:
adapters `_XlsbWorkbook`/`_XlsbSheet` (pyxlsb) and `_XlsWorkbook`/`_XlsSheet` (xlrd) present the same
`iter_rows` interface as openpyxl; the opener tries the detected reader then falls back through the
others, erroring only if all fail.

- `extract_sheet(ws, max_scan_rows, wbs_codes)` — reads one worksheet: header resolution, tab-type
  detection, row records, a `preview_md` table, and a `raw_grid` positional dump. **Inline JADE WBS
  filter:** when the tab is detected `JADE` and `wbs_codes` is set, it **skips the row cap** and keeps
  only rows whose WBS *code* column matches a selected code (zero matches → empty JADE, surfaced as a
  coverage gap — never a silent fall-back to the full ledger).
- `extract_excel_bounded(path, max_scan_rows)` — plain pipeline: sampled rows per tab.
- `extract_excel_full(path, wbs_codes)` — skill pipeline: all rows (JADE filtered inline).
- `collect_wbs_candidates(path)` — streams every sheet's WBS Key/description columns into a small
  `{code: description}` map (retains only the map, so a 983k-row JADE costs one cheap pass).
- `sheets_to_json(book)` — flattens a read workbook into the row-list format the skill scan consumes.

### Tab typing & WBS columns — [tab_schemas.py](extractor/tab_schemas.py)
Deterministic, **no LLM**. `detect_tab_type(top_rows)` matches column signatures in `_GROUPS` to the
`KNOWN_TYPES` (`JADE`, `CJI3`, `iWells`, `BoM`, `Recon`); `schema_description` returns the standard
layout to inject into the tab prompt. WBS helpers: `wbs_key_columns` (the `… - Key` code columns, col
K fallback), `wbs_desc_column` (the paired description column), `normalize_wbs_codes` (uppercase,
strip trailing `*`/`.`), `wbs_row_matches` (prefix/contains match).

### Invoices — [invoice_extract.py](extractor/invoice_extract.py)
- `extract_invoice(path)` — PDF text (+OCR), strips legal T&C boilerplate pages
  (`_strip_boilerplate`), regex-parses header fields (`_parse_fields`), caps body at 12K chars.
- `discover_invoices(exhibit_paths, report_path)` — finds sibling invoice PDFs next to the exhibits,
  filtering out non-invoice PDFs (`_SKIP_RE`).

---

## 5. Skill framework

A **skill** is a markdown playbook telling the LLM how to understand one exception type. Selection is
deterministic and filename-based — no metadata, no code change to add one.

### Registry — [skills.py](extractor/skills.py)
`resolve_skill(exc_type)` slugifies the CSV topic (`"Material Reconciliation"` →
`material_reconciliation`) and looks for `skills/skill_<slug>.md`. Found → a `Skill(name, instructions,
path)`; not found → `None` (caller runs the plain pipeline). Tolerates an explicit `skill_`-prefixed
name.

**Three skills exist today:**
- `skills/skill_material_reconciliation.md` (the richest — includes the §1.5 Recon→JADE→iWells→CJI3→BOM
  investigation-order playbook and the WBS pre-filter reasoning)
- `skills/skill_affiliate_labor.md`
- `skills/skill_otl_labor.md`

### Scan engine — [skill_engine.py](extractor/skill_engine.py)
`apply_skill(skill, sheets_json, description, meta, tier, wbs_codes)` does **no column/value logic** —
it slices all extracted rows into chunks (`_CHUNK_ROWS=300`) and, per chunk, asks the LLM (guided by
the skill markdown) to pull relevant findings and say whether it now has *enough*.
- `_chunk_plan` orders tabs: mandatory relevant tabs first (smallest-first), then big JADE ledgers,
  then others. A **coverage gate** (`required_ids`) forbids declaring sufficiency until every relevant
  tab is fully scanned. A JADE >`_BIG_JADE_ROWS` (1000) is *optional* coverage — **unless it was
  WBS-filtered** (`wbs_filtered`), in which case it's already small and mandatory.
- The verbatim "findings so far" tail is folded into a rolling summary (`_summarize_findings`, light
  tier) once it exceeds `_FINDINGS_RENDER_CHARS`, so later chunks still see earlier findings compactly.
- `render_analysis(analysis)` renders the findings + coverage into the synthesis prompt block.

All scan passes go through the tracked `structured_call`/`text_call`, so their tokens count toward the
run's `token_usage`.

---

## 6. Output layer

### Paths & sinks — [output_paths.py](../shared/output_paths.py)
- `infer_year_asset(path)` — derives `<year>`/`<asset>` from the corpus path.
- `synthesis_rel_path(year, asset, exc, cycle)` → `<year>/<asset>/<exc>/rebuttal_<n>_synthesis.json`.
- `LocalSink` / `SharePointSink` — both expose `write(rel_path, data)` and `folder_exists(rel_folder)`
  (the "already processed" guard). `SharePointSink` **falls back to local on a 401** so a mid-batch
  token expiry never loses output.
- `make_sink()` — chooses the backend from `OUTPUT_TARGET` (`local` | `sharepoint`).

### SharePoint auth — [sharepoint.py](../shared/sharepoint.py)
MSAL **device-code flow** (works in locked-down IBM tenants), token cached to
`~/.jv_audit_msal_cache.bin`. The writer hands office365 a **refreshing token provider**
(`_get_token`) that it re-invokes per request: `acquire_token_silent` is free when the access token is
valid and transparently uses the refresh token when it expires, so a **multi-hour batch never outlives
its ~1h access token** and only device-flows (`_device_flow`) if the refresh token is also dead.
`_sp_retry` retries transient 429/503/500 with backoff; `_ensure_folders` creates the folder tree on
demand; `folder_has_synthesis` backs the skip guard.

### Token accounting — [tokens.py](../shared/tokens.py) + [token_accounting.py](../shared/token_accounting.py) + [pricing.py](../shared/pricing.py)
**Both** backends record the **provider's own** `usage` block per call (`estimator: "gateway"`) —
Anthropic's on `ica`, Azure OpenAI's on `azure` (`llm._langchain_usage` lifts it off the LangChain
reply). Each block is carried in its own field names on purpose, because `token_accounting.py`
normalises it and decides cache-counter semantics at runtime (never hardcoded) by **dialect first**:
Anthropic field names mean the counters are *exclusive* of `input_tokens`, OpenAI names mean
*inclusive*, with the old magnitude rule kept only as a fallback. Renaming one dialect into the other
would flip that rule and either double the prompt or lose the cache.

Two sibling readers answer two different questions off the same parse, and must not be confused.
`reported_prompt_size` returns the **full** prompt, which is what the context-window guard needs
because cached tokens still occupy the window. `input_buckets` splits that same total into the three
**disjoint billing classes** — new input, cache read, cache write — which is what the report and the
cost estimate need. The invariant `sum(input_buckets(u)) == reported_prompt_size(u)` ties them
together. `output_counters` returns billed output plus the extended-thinking **subset** of it, read by
key presence so a reported `0` is never confused with a missing key.

So `token_usage.tokens_in` is new (uncached) input only; `prompt_tokens` is the full size that
`tokens_in` used to mean, and `tokens_total` stays `prompt_tokens + tokens_out`.

The two providers do not report the same classes, and that is reported rather than papered over.
OpenAI has **no cache-write class** — its caching is automatic and a write is not billed on its own —
so an Azure tier's `cache_creation_tokens` is 0 because the class does not exist there, not because
nothing was cached; its `cache_read_tokens` is the one to read. `thinking_tokens` holds Claude's
thinking and GPT's `reasoning_tokens` in one field, since both are output spent before the answer.

Tiktoken estimation (`estimator: "tiktoken"`, with an offline ~4-chars/token fallback) is now only the
last resort, for a reply that carried no usage block at all; it undercounts output for a thinking
model because it only sees the reply text. The thread-safe `USAGE` singleton
(`reset`/`record`/`report`) is reset per exception (or managed by the rebuttal caller) so each JSON
reports only its own run's `token_usage`, broken down by tier and by node. `build_model_map` backs
`--token-map`. `pricing.py` turns the final per-tier aggregates into `token_usage.cost` from
`configs/model_pricing.json` (`$MODEL_PRICING_FILE` overrides), looking rates up **by model id first
and by tier second** so one file can price both backends; it ships no rates of its own and reports
`priced: false` rather than guessing.
`token_accounting.py` also owns the pre-flight context check (compact at 80% of the model's input
limit, warn at 60%) and `is_context_overflow_error`; see [GATEWAY_FINDINGS.md](../docs/GATEWAY_FINDINGS.md).

---

## 7. Schemas — [schemas.py](extractor/schemas.py)

Pydantic models define every LLM output contract and the final artifact. The top-level
**`ExhibitEvidenceReport`** (deliberate field order for readability) carries: `year`, `asset`,
`exception_category`, `case_id`, `exception_code`, `title`, `auditor_description`, `attachments`
(`AttachmentInterpretation` → `TabInterpretation` → `FigureRef`), `invoices` (`InvoiceEvidence` →
`InvoiceExtraction`/`InvoiceLineItem`/`InvoiceParsedFields`), `likely_basis_for_amount`, `synthesis`,
`open_questions`, `previous_rebuttals`, `operator_dialogue_considered`, `rebuttal_cycle`/`rebuttal_id`,
`token_usage` (`TokenUsageReport`/`TierTokenUsage`, plus `RunCost`/`TierCost` for the priced estimate),
and `elapsed_minutes`. Intermediate schemas:
`AttachmentSummary`, `SynthesisResult`, `InvoiceInterpretation`, and `WbsFocus`
(`wbs_codes`/`material_name`/`length_discrepancy`/`unsure`).

---

## 8. Case resolution & rebuttal discovery

### [case_resolver.py](extractor/case_resolver.py)
Shared by `run.py`, `run_batch.py`, and `orchestration-flow`'s two runners. `resolve_cases(base, csv,
type)` reads a cases CSV, and for each row builds a `Case(case_id, exc_id, title, entity_dir)` by
resolving `<base>/<year>/<asset>` (`resolve_entity`). `_COLUMN_ALIASES` + `_field` let it read both
`test_cases_*.csv` and the clustered `clustered_exception_tables.csv` schema. `slugify` matches the
skill/type slug convention.

For **single-case runs** the CSV is bypassed entirely: `parse_case_id` splits `<Asset>_<year>_<exc>`,
`pad_exc` canonicalizes the exception to 4 digits, `resolve_entity` matches the folder off disk
(`_norm` + prefix, so `GW_2022_9` finds `2022/Great White (BP Only)`), and `lookup_topic` reads the
topic from the master catalog — `resolve_case_id` / `select_case_id` compose those. Two constants live
here so they aren't duplicated per entrypoint: `default_corpus_base()` (derived from `__file__`, no env
override) and `resolve_cases_csv(arg)`, which resolves a bare filename under `datastores/`, warns and
falls back to the default when a named file is missing, and exits cleanly if even the default is gone.

### [rebuttals.py](extractor/rebuttals.py)
`discover_rebuttal_cycles(entity_dir)` finds each `Rebuttal N` cycle folder and builds a `Cycle`
(auditor report PDF + side folders). `find_exception_evidence(cycle, code)` scopes exhibits + invoices
to one exception on the **auditor side only**, and `drop_operator_responses` filters the operator's own
workbooks (response / operator reply / rebuttal / to auditor / operator comments — all years, via the
shared predicate in [response_names.py](extractor/response_names.py)) even when the corpus misfiles them
on the auditor side; a name attributing the reply or comments to the **auditor** is auditor-side and is
kept, and the same predicate excludes response *tabs* in
[excel_bounded.py](extractor/excel_bounded.py). `build_previous_rebuttals(prior_reports, running_summary)` produces the
chained prior-cycle context (running summary + verbatim tail) fed into synthesis.

---

## 9. Configuration (`.env`)

| Var | Purpose |
|-----|---------|
| `LLM_PROVIDER` | `ica` (default) or `azure`. Read once at import; `--provider` overrides it per run via `llm.set_provider()`. |
| `ICA_API_KEY`, `ICA_BASE_URL` | ICA gateway credentials/endpoint. |
| `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` | Azure AI Foundry project endpoint and key shared by all three tiers. The client appends `/openai/v1/`; no dated REST API version is configured. |
| `AZURE_MODEL_HIGH/MED/LIGHT` | Azure tier → model/deployment name. |
| `MODEL_MAX_TOKENS_HIGH/MED/LIGHT` | Per-tier output cap. |
| `MODEL_MAX_INPUT_TOKENS_HIGH/MED/LIGHT` | Per-tier input-context override; the Azure GPT-5.6 and current ICA models default to 1,050,000. |
| `MODEL_TIMEOUT`, `MODEL_TRANSIENT_RETRIES` | Per-request timeout (300s) and transient-error retries (3). |
| `OUTPUT_TARGET` | `local` or `sharepoint`. |
| `OUTPUT_BASE` | Local output root. |
| `SHAREPOINT_URL`, `SHAREPOINT_FOLDER` | SharePoint site + folder (device-code auth on first run). |

---

## 10. Design principles (why it's shaped this way)

- **Grounding over cleverness.** Every prompt forbids invented/inferred/computed values and unknown-
  abbreviation guessing. The output explains reasoning and lists gaps; it never adjudicates.
- **Deterministic where possible.** Tab typing, WBS column resolution, skill selection, and file
  format detection are pure Python — the LLM is reserved for genuine interpretation.
- **Portable structured output.** `structured_call` embeds the JSON schema and validates locally, so
  the pipeline isn't tied to any one gateway's function-calling dialect.
- **Two retry layers, distinct concerns.** Transport retries (`_invoke`) vs. parse/validation retries
  (`structured_call`) — a slow gateway no longer aborts a whole cycle.
- **Read everything, then filter at the source.** Skill runs ingest full ledgers; the WBS pre-filter
  narrows JADE *inside* `extract_sheet` so both interpretation and the skill scan see the same filtered
  data, and nothing is silently truncated.
- **Process-level isolation for parallelism.** Each batch case is its own subprocess → its own token
  counter and sink, zero shared mutable state.
- **Resilient long runs.** Refreshing SharePoint tokens, 401→local fallback, transient-error backoff,
  and per-cycle skip guards keep multi-hour batches alive and idempotent.
