#!/bin/bash
# Fixed demo batch for the exhibit-evidence explainer — 11 hand-picked exceptions, run serially.
#
# All outputs route to the extractor tree defined by OUTPUT_BASE_EXTRACTOR in .env:
#   <OUTPUT_BASE_EXTRACTOR>/<year>/<asset>/<exception-id>/rebuttal_<n>_synthesis.json
# (run.py infers year/asset from the resolved corpus path automatically).
#
# Every case is now named by its CASE ID alone — `<Asset>_<Year>_<Exc>` carries the asset, year and
# exception, so run.py matches the entity folder off disk and pulls the topic from
# datastores/clustered_exception_tables.csv. That replaced the old explicit
# --report/--exhibit/--entity-dir/--exception/--title invocations, which hardcoded 30-odd absolute
# paths and hand-written titles. One consequence worth knowing: the topic is now the catalog's
# (e.g. exc 0027 is "Storage Charges", not the old label "FMC Storage Charges"), and the topic is what
# selects the extraction skill. Pass --topic '<Topic>' after a --case-id to override one.
#
# Part A  = the original 5 Great White exceptions.
# Part B  = 5 Material Reconciliation exceptions that carry vendor invoices (auto-discovered).
# Part C  = 1 exception spanning multiple rebuttal cycles (chain demonstration).
#
# Usage:
#   ./run_exc_batch.sh                    # all 11, serially, provider from .env
#   ./run_exc_batch.sh --provider azure   # LLM backend: azure | ica
#   ./run_exc_batch.sh --dry-run          # resolve all 11 and exit (no LLM spend)
#
# For anything other than this fixed list, use ./run_test_cases.sh (CSV-driven, parallel) or
# `run.py --case-id <ID>` directly.
set -uo pipefail
cd "$(dirname "$0")"
PYTHON="../.venv/bin/python3"

# Forwarded to every run.py call below. --provider is passed through as a real flag now (run.py
# switches the backend in-process), so there is no LLM_PROVIDER export to manage here.
FWD=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --provider|--llm-provider)      FWD+=("--provider" "$2"); shift 2 ;;
    --provider=*|--llm-provider=*)  FWD+=("--provider" "${1#*=}"); shift ;;
    --dry-run|--no-skill)           FWD+=("$1"); shift ;;
    *) echo "error: unknown arg '$1' (accepts --provider, --dry-run, --no-skill)" >&2; exit 2 ;;
  esac
done

# --- prep: the 2023 Great White exhibits ship as a zip; extract once (idempotent) ---
BASE="$(cd ../.. && pwd)/Data Files/Exception Reports and Rebuttals by year"
GW2023_AR="$BASE/2023/Great White/Rebuttal 0/Audit Report"
if [ ! -d "$GW2023_AR/0004" ] && [ -f "$GW2023_AR/Exception-Great White-290526_022736.zip" ]; then
  echo "=== Extracting 2023 Great White exhibit archive (one-time) ==="
  unzip -q -o "$GW2023_AR/Exception-Great White-290526_022736.zip" -d "$GW2023_AR"
fi

# Part A — the original 5 Great White exceptions.
# Part B — 5 Material Reconciliation exceptions WITH vendor invoices: run.py discovers each
#          exception's exhibits and sibling invoice PDFs, and processes every rebuttal cycle it
#          appears in.
# Part C — GreatWhiteBPOnly_2022_0009 appears in Rebuttal 0, 1 and 2, so it produces three chained
#          JSONs (rebuttal_0/1/2_synthesis.json) linked by previous_rebuttals.
CASES=(
  # --- Part A ---
  GreatWhite_2022_0003
  GreatWhite_2022_0007
  GreatWhite_2023_0004
  GreatWhite_2024_0041
  GreatWhite_2024_0074
  # --- Part B ---
  GreatWhite_2024_0027
  GreatWhite_2024_0022
  GreatWhite_2024_0062
  GreatWhite_2024_0001
  GreatWhite_2023_0081
  # --- Part C ---
  GreatWhiteBPOnly_2022_0009
)

FAILED=()
for CASE_ID in "${CASES[@]}"; do
  echo ""
  echo "=== $CASE_ID ==="
  if ! "$PYTHON" run.py --case-id "$CASE_ID" ${FWD[@]+"${FWD[@]}"}; then
    FAILED+=("$CASE_ID")
  fi
done

echo ""
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "=== DONE with $((${#CASES[@]} - ${#FAILED[@]}))/${#CASES[@]} ok — FAILED: ${FAILED[*]} ==="
  exit 1
fi
echo "=== ALL DONE (${#CASES[@]}/${#CASES[@]}) ==="
