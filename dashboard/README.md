# Shell JV Audit token-spend dashboard

The dashboard reads the same project-root `.env` used by the Excel Extractor and Labor Inference
pipelines. It supports both configured output targets:

| Section | `OUTPUT_TARGET=sharepoint` | `OUTPUT_TARGET=local` |
| --- | --- | --- |
| Excel Extractor | `SHAREPOINT_READ_FOLDER` (then `SHAREPOINT_FOLDER`) | `INPUT_BASE` |
| Labor Inference | `SHAREPOINT_WRITE_FOLDER` | `OUTPUT_BASE` |

Run it from the repository root:

```bash
.venv/bin/python -m dashboard.app
```

Then open `http://127.0.0.1:8765`.

The dashboard reads `.cache/token_metrics.sqlite3`; normal page loads and **Refresh cache** never
scan SharePoint or the output directories. Each extractor and inference artifact writes its metric
record to SQLite before the JSON is written to SharePoint/local storage.

Records are isolated by pipeline, output target/folder, run ID, and relative artifact path. The
dashboard provides a per-section run selector, so repeated exceptions in different runs never merge.
Batch runners generate one run ID and pass it to every case subprocess. Supply a stable ID yourself
when needed:

```bash
.venv/bin/python excel_extractor/run_batch.py --run-id extractor-prod-run-3
.venv/bin/python orchestration-flow/run_batch.py --run-id inference-prod-run-3
```

`JV_AUDIT_RUN_ID` is the equivalent environment override and `TOKEN_METRICS_CACHE_PATH` overrides
the SQLite location.

To preload SQLite before the server starts:

```bash
.venv/bin/python -m dashboard.app --warm
```

Outputs created before metric caching was added can be imported once, explicitly. This is never
triggered by changing `.env` or a folder URL:

```bash
.venv/bin/python -m dashboard.app --legacy-backfill-only
```

The command reuses the pipeline's existing SharePoint MSAL token cache and device-code flow. It is
intentionally not exposed as a normal dashboard action, so dashboard usage can never trigger a
SharePoint scan.

Run the dashboard tests with:

```bash
.venv/bin/python -m unittest dashboard.tests.test_dashboard tests.test_token_metrics_cache
```
