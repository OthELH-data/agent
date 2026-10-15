#!/bin/bash
# Parallel batch runner (shell entrypoint): delegates to run_batch.py which runs each exception as
# an isolated subprocess, N-at-a-time (default 4). Each case gets its own process → own token counter,
# own output sink, zero shared mutable state.
#
# Usage:
#   ./run_test_cases.sh                              # all 44 labor test cases, 4 concurrent
#   ./run_test_cases.sh material_reconciliation      # only mat-recon cases
#   ./run_test_cases.sh --type material_reconciliation --jobs 8
#   ./run_test_cases.sh --year 2024                  # only 2024 cases
#   ./run_test_cases.sh --year 2024 --type material_reconciliation --dry-run
#   ./run_test_cases.sh --jobs 2 --dry-run
#   ./run_test_cases.sh --case-id GreatWhite_2024_0040   # one case, no CSV row required
#
# All flags are passed through to run_batch.py. Recognized flags:
#   --type <slug>      filter to one exception type (positional arg also works)
#   --year <YYYY>      filter to one year (comma-separate for several, e.g. 2023,2024)
#   --exc <ID>         filter to one exception id (padding-insensitive: 40 == 0040)
#   --case-id <ID>     run just this case, e.g. GreatWhite_2024_0040. Needs NO CSV row: the id is
#                      self-describing, so the entity folder is matched off disk and the topic pulled
#                      from datastores/clustered_exception_tables.csv. It also bypasses --type (with a
#                      warning naming the actual topic).
#   --topic <TOPIC>    topic/category for a --case-id no CSV lists; overrides the catalog lookup
#   --jobs N           max concurrent cases (default: 4)
#   --no-skill         disable the skills layer for every case
#   --csv <name>       a different cases CSV. A bare filename resolves under datastores/ (default:
#                      test_cases_46.csv); a file that doesn't exist warns and falls back to the
#                      default. The corpus root is not configurable.
#   --limit N          run only the first N resolved cases
#   --force            re-run cases even when synthesis output already exists
#   --dry-run          list resolved cases and exit
#   --provider <p>     LLM backend for this run: azure | ica (default: LLM_PROVIDER in .env)
#   --llm-config       print the resolved provider/model/deployment/endpoint config and exit. No
#                      network call, no case run — check a provider switch here first:
#                        ./run_test_cases.sh --provider azure --llm-config
set -uo pipefail
cd "$(dirname "$0")"
# Stream progress as each line is produced, including when run over SSH or tmux.
export PYTHONUNBUFFERED=1
PYTHON="../.venv/bin/python3"

# --provider {azure|ica} exports LLM_PROVIDER for this run (and every spawned case subprocess, which
# inherit env=os.environ). Parsed out first; unset falls back to LLM_PROVIDER in .env. Everything else
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

# If the first remaining positional arg is a bare slug (no leading --), rewrite it as --type <slug>
# for backwards compatibility with the old `./run_test_cases.sh material_reconciliation` form.
# (The `${ARGS[@]+…}` guards keep empty-array expansion safe under `set -u` on bash 3.2 / macOS.)
ARGS=(${FWD[@]+"${FWD[@]}"})
if [ "${#FWD[@]}" -gt 0 ] && [[ "${FWD[0]}" != --* ]]; then
  ARGS=("--type" ${FWD[@]+"${FWD[@]}"})
fi

# No --csv is injected: run_batch.py owns the default (datastores/test_cases_46.csv, via
# case_resolver.resolve_cases_csv), so there is one definition of it rather than one per wrapper.
exec "$PYTHON" run_batch.py ${ARGS[@]+"${ARGS[@]}"}
