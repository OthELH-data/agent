# Excel Exhibit Explainer

Given an **audit report PDF** and its **Excel exhibits**, this program produces a JSON that
explains *what the exhibits demonstrate* for a given exception — and pinpoints the specific
figures, columns, and data the auditor likely used to reach both their reasoning and the
questioned dollar amount.

It is fully self-contained: it does **not** depend on the `agentic_audit` project or any other
folder. Python does the mechanical work (read PDF text, read Excel tabs); an LLM does the
interpretation — through the IBM ICA gateway by default, or a native Azure OpenAI / AI Foundry
deployment (see [LLM provider](#llm-provider-ica-or-azure)).

## What you need to run it

1. **Python 3.11+**
2. **Dependencies:** `pip install -r requirements.txt`
3. **An LLM credential:** copy `.env.example` to `.env` and fill in `ICA_API_KEY` — or, for the
   Azure backend, copy the block from `.env.azure.example` instead.
4. **Input files:** the audit report PDF and the Excel exhibit(s) you want explained.

```bash
cd excel_extractor
python -m venv .venv && source .venv/bin/activate   # or use uv / your env of choice
pip install -r requirements.txt
cp .env.example .env      # then edit .env and add your ICA_API_KEY
```

## Run it

**Check the LLM setup first (optional):**
```bash
python run.py --llm-config        # resolved provider / model / deployment / endpoint — no network call
python run.py --smoke-test        # expects 'pong' from each model tier (one live call per tier)
```

**See the fallback token estimator mapping (optional):**
```bash
python run.py --token-map         # pings the gateway; lists every model → its fallback encoding
```

**One exception — a case id is the whole input:**
```bash
python run.py --case-id GreatWhite_2024_0040
python run.py --case-id GreatWhite_2024_0040 --dry-run   # resolve and exit, no LLM spend
```

`<Asset>_<Year>_<Exception>` carries everything structural, so nothing else has to be passed:

- the **entity folder** is matched off disk under the corpus root (tolerant of spelling and
  punctuation — `GW_2022_9` finds `2022/Great White (BP Only)`, `greatwhite_2024_40` finds
  `2024/GreatWhite`), and every exhibit, invoice and rebuttal cycle under it is auto-discovered;
- the **exception id** is normalized to the corpus's 4-digit form (`40` → `0040`);
- the **topic** is looked up in `datastores/clustered_exception_tables.csv`. It selects the
  extraction skill, so override it with `--topic "<Exception Topic>"` when the catalog has no row
  (the run warns) or when you want a different skill.

Other flags: `--skill <name>` / `--no-skill` (skills layer), `--provider {azure,ica}` (LLM backend
for this run), `--llm-config` / `--smoke-test` / `--token-map` (diagnostics).

**Many exceptions — `./run_test_cases.sh`** (wraps `run_batch.py`, which runs each case as its own
`run.py` subprocess so per-case token counts stay correct):

```bash
./run_test_cases.sh                                  # every case in the CSV, 4 at a time
./run_test_cases.sh --type material_reconciliation   # one exception type (positional also works)
./run_test_cases.sh --year 2024 --jobs 8
./run_test_cases.sh --exc 0040                       # one exception id, every asset that has it
./run_test_cases.sh --case-id GreatWhite_2024_0040   # one case (bypasses --type, with a warning)
./run_test_cases.sh --dry-run                        # list what would run and exit
```

Cases come from `datastores/test_cases_46.csv`. `--csv <name>` picks another (a bare filename resolves
under `datastores/`; a file that doesn't exist warns and falls back to the default). Also `--limit N`,
`--no-skill`, `--provider {azure,ica}`, `--llm-config`. The corpus root is not configurable.

**`./run_exc_batch.sh`** runs the fixed 11-case demo set (5 Great White exceptions, 5 material
reconciliations with vendor invoices, 1 multi-rebuttal-cycle chain) serially by case id.

## Output

A JSON file (`ExhibitEvidenceReport`) with:

| Field | What it is |
|-------|------------|
| `auditor_description` | the exception's narrative, pulled from the report PDF |
| `claimed_amount` | the questioned USD (auto-detected from the description) |
| `attachments[]` | one per Excel file: `role`, `overall_summary`, and `tabs[]` |
| `attachments[].tabs[]` | per tab: `purpose`, `demonstrates`, `key_columns`, `auditor_figures_used[]`, `ties_to_amount` (`direct`/`supporting`/`unclear`) |
| `likely_basis_for_amount[]` | strongest candidate figures across all exhibits for how the amount was derived, each with a `location` and `why_relevant` |
| `rebuttal_dialogue` | verbatim operator/auditor exchange extracted from the current auditor report, including auditor replies; empty when none is present |
| `rebuttal_dialogue_entries[]` | the same exchange split, in order, into explicit `operator` and `auditor` reply items |
| `synthesis` | narrative of what the exhibits collectively demonstrate |
| `open_questions[]` | evidence gaps a reviewer should check |
| `token_usage` | tokens in/out for the whole run, broken down per model tier (`high`/`med`/`light`) plus overall totals and the run's estimated cost — printed at the bottom of the JSON |

> **Where the numbers come from.** `token_usage.estimator` says which: `gateway` (exact, the
> provider's own `usage` block) or `tiktoken` (an estimate). **Both** backends report their own
> usage — Anthropic's block on `--provider ica`, Azure OpenAI's on `--provider azure` — so `gateway`
> is the normal case either way. The estimate is the fallback for a call the provider reported nothing
> for, and `mixed` means some tiers each way.
>
> **Input is reported as three disjoint classes**, because they bill at three different rates.
> `tokens_in` is **new** (uncached) input at the plain rate, `cache_read_tokens` is served from the
> prompt cache at about 0.1×, and `cache_creation_tokens` is written to it at about 1.25×. The three
> add up to `prompt_tokens`, the full prompt size, and `tokens_total` is `prompt_tokens + tokens_out`.
> So a warm run and a cold one are told apart by where the input sits, not by the total.
>
> `tokens_out` is billed output with extended thinking included. `thinking_tokens` names that part as
> a **subset**, not an addition — it explains a call that billed hundreds of output tokens while its
> reply text was empty.
>
> **The two providers do not report the same classes**, and a 0 can mean "no such class here" rather
> than "nothing happened". Claude reports cache reads, cache writes, and a thinking subset. GPT
> reports cache reads and a reasoning subset, but has **no cache-write class at all** — its caching is
> automatic and a write is not billed separately — so `cache_creation_tokens` is always 0 on an Azure
> tier. Read its `cache_read_tokens` to see whether the cache was working. `thinking_tokens` carries
> Claude's thinking and GPT's reasoning in the same field, because both are output spent before the
> answer. Each tier also lists `models_seen`; more than one entry means the Azure→ICA fallback latched
> part-way through and the tier's tokens span two price lists.
>
> The tiktoken fallback is only an approximation: Claude has no public encoding, so `cl100k_base`
> stands in and **undercounts by roughly 15–20%** on prose (more on code) — see
> `shared/tokens.py`. It undercounts output further for a thinking model, because it only sees the
> reply text. If tiktoken can't download its vocab (offline / proxy), it degrades again to a
> ~4-chars-per-token heuristic so extraction never fails just to count tokens.

> **Cost.** `token_usage.cost` prices the run from the final per-tier aggregates — four
> multiplications per tier, nothing priced per call. Rates live in `configs/model_pricing.json`, which
> holds **Anthropic list prices** for the three Claude tier models plus a null-rate skeleton for the
> Azure models. Point `MODEL_PRICING_FILE` at your own copy to price a run against a gateway contract
> instead; `configs/model_pricing.example.json` documents every key.
>
> Rates are looked up **by model id first, by tier second**. That is what lets one file price both
> backends: a tier name says how important a call was, not what it cost, and `high` is claude-opus-5 on
> `ica` but a gpt-5 model on `azure`. Each priced tier records which answered, as `rates_basis`
> (`model` or `tier`) and `rates_key`. An OpenAI model's entry sets `cache_write_billed: false`, which
> makes its cache-write column an explicit 0 — declared free, as opposed to a rate nobody filled in.
>
> Read `total_usd` as an estimate at list price, not as a bill. Calls go through the ICA gateway or an
> Azure deployment, either of which may charge differently, and `cost.rates_source` / `cost.as_of`
> record which numbers were used. Degradation is explicit rather than guessed: with no usable rates the
> block reports `priced: false` and carries no dollar figures at all, a tier missing any one of its four
> rates is named in `unpriced_tiers` and excluded from the total, a tier that fell back to tier rates
> written for a different model is flagged in `model_mismatch` (still priced, since the rate is per
> tier), and a tier that ran two models is flagged in `mixed_model_tiers` — priced entirely at the last
> one's rates, so its total is wrong by the rate difference on the earlier model's share.
>
> `uncached_equivalent_usd` prices the whole prompt at the plain input rate, so `cache_savings_usd`
> shows what prompt caching was worth. On the verified OTL run that was $0.87 against $1.48, or 41%.

## Runtime hooks

Runtime interception hooks are configured in [`configs/hooks.yaml`](../configs/hooks.yaml). The
top-level keys are application modules, and each module owns a `hooks` mapping. The extractor's
budget hook is cycle-scoped by default:

```yaml
extractor:
  hooks:
    budget:
      enabled: true
      intercepts: [structured_call, text_call]
      parameters:
        scope: rebuttal_cycle
        limit_usd: 7.00
        block_when_pricing_unavailable: true
```

Before every intercepted structured/text attempt, the hook prices the provider-reported usage already
recorded for the current rebuttal cycle. A completed response is always retained; after that response
brings the cycle to the limit, the next intercepted call is blocked before network I/O. The cycle then
writes a schema-valid partial report with defaults for unprocessed fields and details under
`processing.hooks.budget`. `USAGE.reset()` starts a fresh hook generation for the next rebuttal cycle.

Set `HOOKS_CONFIG_FILE` to use a different YAML file. The labor pipeline identifies its shared LLM
calls as module `labor`, so extractor hooks do not apply to it unless a labor hook is configured.

## How it works

```
report PDF ──▶ pdf_extract (PyMuPDF)  ──▶ description.py (regex slice)  ─┐
                                                                        ├─▶ explain.py ──▶ JSON
Excel files ─▶ excel_bounded (openpyxl, row-capped) ────────────────────┘  (LLM: ICA or Azure)
```

- **Python** (`pdf_extract`, `description`, `excel_bounded`): extract text and tab data. The
  Excel reader is row-capped so multi-hundred-thousand-row ledger tabs don't blow up memory.
- **LLM** (`llm`, `explain`): ~3 passes — per-tab interpretation (light model), per-file
  summary (mid model), and a final cross-exhibit synthesis (high model). All output is
  schema-validated Pydantic (`schemas.py`), and prompts require every cited figure to appear
  literally in the extracted data.

## LLM provider (ICA or Azure)

`shared/llm.py` is the single LLM client — for this tool *and* for `orchestration-flow/`. Which
backend it builds is set by **`LLM_PROVIDER`** (`ica`, the default, or `azure`), read once at import.
`--provider {azure,ica}` overrides it per run, on every entrypoint:

```bash
python run.py --provider azure --llm-config              # what would this run use? (no network call)
python run.py --provider azure --case-id GreatWhite_2024_0040
./run_test_cases.sh --provider azure --llm-config
./run_test_cases.sh --provider azure --year 2024         # run the batch on Azure
```

Because the provider is frozen when `shared.llm` is imported, the flag can't work through the
environment alone — `run.py` / `run_batch.py` call `llm.set_provider()`, which reassigns the module's
provider globals *and* re-exports `LLM_PROVIDER` so each case subprocess inherits it. Nothing in
`.env` has to change to try a backend. Three tiers (`high`/`med`/`light`) resolve per provider as
`<PROVIDER>_MODEL_<TIER>` → a built-in default, so `ICA_MODEL_*` and `AZURE_MODEL_*` can both sit in
`.env` and the flag picks a set.

For Azure, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and `AZURE_OPENAI_API_VERSION` are shared
by every tier. The configured `AZURE_MODEL_*` value is also used as that tier's deployment name, and the
endpoint must already be the Azure resource root. Every knob is documented in
[`.env.azure.example`](.env.azure.example); the deeper treatment is in
[`orchestration-flow/README.md`](../orchestration-flow/README.md#model-tiers--llm-provider-azure-vs-ica).

Azure requests always use the Responses API. If an Azure request fails and `ICA_API_KEY` is configured,
the process retries that request on ICA and uses ICA for the rest of the case.

Start with `--llm-config`: a 404 `DeploymentNotFound` or a 401 is nearly always one of the five values
it prints, and it costs nothing to look.

## Files

```
excel_extractor/
├── run.py                    single-case entrypoint (--case-id)
├── run_batch.py              parallel driver — one run.py subprocess per case
├── run_test_cases.sh         venv wrapper → run_batch.py (CSV-driven)
├── run_exc_batch.sh          fixed 11-case demo set, serial
├── requirements.txt
├── .env.example              copy to .env, add your ICA key
├── .env.azure.example        the Azure OpenAI / AI Foundry block (placeholders only)
├── README.md
└── extractor/
    ├── pdf_extract.py        PDF → text (PyMuPDF, optional OCR)
    ├── description.py        slice one exception's Description out of the report
    ├── excel_bounded.py      Excel → per-tab structure + sampled rows (openpyxl)
    ├── case_resolver.py      case id / CSV row → runnable case (entity folder, exc id, topic)
    ├── llm_adapter.py        extractor-owned calls into shared.llm (HOOK_MODULE="extractor")
    ├── explain.py            per-tab / per-file / synthesis orchestration
    └── schemas.py            Pydantic output models

shared/
├── llm.py                    provider client + schema-validated structured/text calls
├── hooks.py                  module-scoped runtime hook interception
├── tokens.py                 per-tier usage tracker + attribution ledger
├── token_schemas.py          shared Pydantic token/cost report models
├── token_accounting.py       provider usage parsing and context thresholds
├── pricing.py                model/tier pricing for token aggregates
├── output_paths.py           local/SharePoint sinks and artifact path helpers
└── sharepoint.py             shared SharePoint authentication and transport
```

## Notes

- **Never share your `.env`** — it holds your API key. Share `.env.example` /
  `.env.azure.example` instead; both are placeholders only.
- The audit report and exhibits are confidential JV audit material; only share input files
  with authorized recipients. The code itself contains none of that data.
