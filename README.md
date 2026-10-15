# Shell-JV-Audit
This repository contains code for an AI solution that automates the audit response process for joint ventures by operator Shell

## Token-spend dashboard

Run the Shell-branded Excel Extractor and Labor Inference spend dashboard with:

```bash
.venv/bin/python -m dashboard.app
```

See [`dashboard/README.md`](dashboard/README.md) for output-folder mapping and usage details.

Pipeline token metrics are persisted to a run-scoped SQLite index before each JSON artifact is
written. Dashboard refreshes therefore do not scan SharePoint. Runs are isolated by pipeline,
output folder, run ID, and artifact path.

## Local artifact routing

With `OUTPUT_TARGET=local`, the pipelines use separate roots from `.env`:

- `OUTPUT_BASE_EXTRACTOR`: Excel Extractor synthesis JSON output.
- `INPUT_BASE_PIPELINE`: extractor JSON input read by the labor pipeline.
- `OUTPUT_BASE_PIPELINE`: labor-pipeline report and trace output.
- `LOCAL_REBUTTAL_FOLDER`: durable rebuttal-analysis history.

The dedicated variables take precedence; the old `OUTPUT_BASE` and `INPUT_BASE` names are retained
only as backward-compatible fallbacks.
