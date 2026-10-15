#!/bin/bash
# OTL orchestration batch — runs OTL/affiliate/payroll labor exceptions through the OTL graph IN PARALLEL,
# one isolated process per case. Mirrors excel_extractor/run_test_cases.sh: uses the repo-root venv
# and forwards all flags to run_batch.py (the parallel driver; run.py is the single-case worker).
# Defaults to datastores/labor_classification_test_set_temporal_split.csv; of its rows the graph runs
# only the OTL/affiliate/payroll labor category when --type is supplied.
# --csv <name> picks a different one (a bare filename resolves under datastores/, and a file that
# doesn't exist warns and falls back to the default). The corpus root is not configurable.
#
#   ./run_otl_cases.sh                       # every temporal-split case, 4 at a time (default)
#   ./run_otl_cases.sh --jobs 8              # up to 8 concurrent cases
#   ./run_otl_cases.sh --year 2023           # only 2023
#   ./run_otl_cases.sh --exc 0003            # only exception 0003 (padding-insensitive: 3 == 0003)
#   ./run_otl_cases.sh --case-id GreatWhite_2024_0040
#                                            # one case, no CSV row required: the id is self-describing,
#                                            # so the entity folder is matched off disk and the topic
#                                            # pulled from datastores/clustered_exception_tables.csv.
#                                            # Also BYPASSES the OTL labor filter, with a warning
#                                            # naming the case's actual topic. (Straight to the single-case
#                                            # worker, `run.py --case-id X`, reads no CSV at all.)
#   ./run_otl_cases.sh --case-id X --exception-type "Labor (OTL, Affiliates, Payroll)"
#                                            # set the topic yourself when the catalog has no row
#   ./run_otl_cases.sh --limit 3             # first 3 resolved cases
#   ./run_otl_cases.sh --force               # re-run cases already written
#   ./run_otl_cases.sh --dry-run             # list resolved cases and exit
#   ./run_otl_cases.sh --code-exec-tier high # put every code-exec specialist on high (default: med)
#   ./run_otl_cases.sh --max-people 3        # how many disputed people each specialist examines per case
#                                            # (omitted or 0 = no cap)
#   ./run_otl_cases.sh --provider azure      # LLM backend for this run: azure | ica (default: .env)
#   ./run_otl_cases.sh --provider azure --llm-config
#                                            # print the resolved provider / per-tier model, deployment,
#                                            # endpoint, api-version, auth mode + the node→tier map, then
#                                            # exit. No network call, no case run — do this before a batch.
#
# Output target is inherited from the environment (default local):
#   OUTPUT_TARGET=local ./run_otl_cases.sh --jobs 4
set -uo pipefail
cd "$(dirname "$0")"
# Stream progress as each line is produced, including when run over SSH or tmux.
export PYTHONUNBUFFERED=1

PYTHON="../.venv/bin/python3"
if [[ ! -x "$PYTHON" ]]; then
  echo "error: venv python not found at $PYTHON (expected the repo-root .venv)" >&2
  exit 1
fi

# --provider {azure|ica} exports LLM_PROVIDER for this run (and every spawned case subprocess, which
# inherit env=os.environ). Parsed out here; unset falls back to LLM_PROVIDER in .env. Every other flag
# is forwarded to run_batch.py unchanged.
FWD=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --provider|--llm-provider)      export LLM_PROVIDER="$2"; shift 2 ;;
    --provider=*|--llm-provider=*)  export LLM_PROVIDER="${1#*=}"; shift ;;
    *)                              FWD+=("$1"); shift ;;
  esac
done
if [ -n "${LLM_PROVIDER:-}" ] && [[ "$LLM_PROVIDER" != "azure" && "$LLM_PROVIDER" != "ica" ]]; then
  echo "error: --provider must be 'azure' or 'ica' (got '$LLM_PROVIDER')" >&2
  exit 2
fi
[ -n "${LLM_PROVIDER:-}" ] && echo "=== LLM_PROVIDER=$LLM_PROVIDER ==="

# No --csv is injected: run_batch.py owns the default
# (datastores/labor_classification_test_set_temporal_split.csv), so there is one definition of it.
exec "$PYTHON" run_batch.py ${FWD[@]+"${FWD[@]}"}
