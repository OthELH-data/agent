#!/usr/bin/env bash
# Push code, env, staged synthesis inputs, and the temporal-test audit data to the Azure VM.
#
#   ./deploy/sync_to_vm.sh code     # repo only (fast, safe to repeat)
#   ./deploy/sync_to_vm.sh env      # root .env -> staging dir on the VM
#   ./deploy/sync_to_vm.sh inputs   # excel_extractor/output (staged synthesis JSONs)
#   ./deploy/sync_to_vm.sh data     # years in temporal test CSV + support folders (replaces VM data)
#   ./deploy/sync_to_vm.sh all      # code + env + inputs + data
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/vm.env.sh"

WHAT="${1:-all}"
DATA_CASES_CSV="${DATA_CASES_CSV:-$LOCAL_REPO/datastores/labor_classification_test_set_temporal_split.csv}"

# macOS ships an older OpenRSYNC (protocol 29) that lacks -s/--protect-args,
# --human-readable, and --info=progress2. These short/legacy options work on
# both the macOS client and the Ubuntu VM's newer rsync.
RSYNC_BASE=(rsync -a -h --progress --partial -e "$SSH_CMD")

CODE_EXCLUDES=(
  --exclude '.venv'
  --exclude '.venv/'
  --exclude '.git/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.DS_Store'
  --exclude '.pytest_cache/'
  --exclude '.otl_scratch/'
  --exclude '.env'                    # runtime config; synced only by sync_env
  --exclude 'excel_extractor/output/'  # synced separately, never with --delete
)

sync_code() {
  echo "==> code mirror -> $VM_BASE/Shell-JV-Audit/"
  ssh_vm "mkdir -p '$VM_BASE/Shell-JV-Audit'"
  # --delete mirrors code changes and removals, including updated shell scripts.
  # The exclusions deliberately preserve the VM's runtime config and output artifacts.
  "${RSYNC_BASE[@]}" --delete "${CODE_EXCLUDES[@]}" \
    "$LOCAL_REPO/" "$VM_HOST:$VM_BASE/Shell-JV-Audit/"
}

sync_env() {
  test -f "$LOCAL_REPO/.env" || {
    echo "ERROR: expected root environment file at $LOCAL_REPO/.env" >&2
    exit 1
  }
  echo "==> root .env -> $VM_BASE/env-staging/  (vm_bootstrap.sh rewrites paths)"
  ssh_vm "mkdir -p '$VM_BASE/env-staging' && chmod 700 '$VM_BASE/env-staging'"
  # macOS OpenRSYNC does not accept rsync's F600 or numeric syntax. This
  # portable symbolic form gives the uploaded file mode 600; the staging
  # directory is separately created with mode 700 above.
  "${RSYNC_BASE[@]}" --chmod='u=rw,go=' \
    "$LOCAL_REPO/.env" "$VM_HOST:$VM_BASE/env-staging/root.env"
}

sync_inputs() {
  echo "==> excel_extractor/output -> VM (no --delete: results live here too)"
  ssh_vm "mkdir -p '$VM_BASE/Shell-JV-Audit/excel_extractor/output'"
  "${RSYNC_BASE[@]}" --exclude '.DS_Store' --exclude '.claude/' \
    "$LOCAL_REPO/excel_extractor/output/" \
    "$VM_HOST:$VM_BASE/Shell-JV-Audit/excel_extractor/output/"
}

sync_data() {
  local data_root year_root support_name support_path year need support_need avail remote_existing
  local -a years=()

  test -f "$DATA_CASES_CSV" || {
    echo "ERROR: temporal test CSV not found: $DATA_CASES_CSV" >&2
    exit 1
  }
  # Parse the CSV with Python rather than line-oriented tools: descriptions can
  # legitimately span multiple physical lines. Pull the unique, four-digit
  # years directly from the selected test set so this deploy cannot accidentally
  # copy unrelated year folders.
  while IFS= read -r year; do
    [[ "$year" =~ ^[0-9]{4}$ ]] && years+=("$year")
  done < <(python3 -c '
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8") as fh:
    print("\n".join(sorted({(row.get("year") or "").strip() for row in csv.DictReader(fh)})))
' "$DATA_CASES_CSV")
  ((${#years[@]})) || {
    echo "ERROR: no four-digit years found in $DATA_CASES_CSV" >&2
    exit 1
  }

  data_root="$LOCAL_PARENT/Data Files"
  year_root="$data_root/Exception Reports and Rebuttals by year"
  need=0
  for year in "${years[@]}"; do
    test -d "$year_root/$year" || {
      echo "ERROR: missing expected data directory: $year_root/$year" >&2
      exit 1
    }
    need=$((need + $(du -sk "$year_root/$year" | awk '{print $1}')))
  done
  # Every visible top-level folder other than the year-organized exception tree
  # is support material (agreements, OTL information, reference documents, etc.).
  # .agreement_rag is also required by the agreement retriever. Other hidden
  # working directories, including .claude, stay local.
  support_need=$(find "$data_root" -mindepth 1 -maxdepth 1 -type d \
    ! -name 'Exception Reports and Rebuttals by year' ! -name '.*' \
    -exec du -sk {} + | awk '{total += $1} END {print total + 0}')
  test -d "$data_root/.agreement_rag" || {
    echo "ERROR: missing expected agreement index: $data_root/.agreement_rag" >&2
    exit 1
  }
  support_need=$((support_need + $(du -sk "$data_root/.agreement_rag" | awk '{print $1}')))
  need=$((need + support_need))
  avail=$(ssh_vm "df -k --output=avail '$VM_BASE' | tail -1")
  remote_existing=$(ssh_vm "if [ -d '$VM_BASE/Data Files' ]; then du -sk '$VM_BASE/Data Files' | awk '{print \$1}'; else echo 0; fi")
  echo "==> temporal years (${years[*]}) + support folders: need $((need/1024/1024)) GiB"
  echo "    VM free now: $((avail/1024/1024)) GiB; reclaiming existing data: $((remote_existing/1024/1024)) GiB"
  if (( avail + remote_existing < need + 3*1024*1024 )); then
    echo "REFUSING: less than 3 GiB would remain after the copy. Free space or attach a data disk." >&2
    exit 1
  fi
  # The caller asked for the VM corpus to exactly match this scoped test set.
  # Validate every local source above before this explicit replacement.
  ssh_vm "rm -rf '$VM_BASE/Data Files' && mkdir -p '$VM_BASE/Data Files/Exception Reports and Rebuttals by year'"

  # Re-run after interruption; --partial resumes each selected year.
  # Literal backslashes preserve spaces for macOS OpenRSYNC's remote shell.
  for year in "${years[@]}"; do
    echo "==> $year -> $VM_BASE/Data Files/Exception Reports and Rebuttals by year/$year/"
    caffeinate -i "${RSYNC_BASE[@]}" --exclude '.DS_Store' \
      "$year_root/$year/" \
      "$VM_HOST:$VM_BASE/Data\\ Files/Exception\\ Reports\\ and\\ Rebuttals\\ by\\ year/$year/"
  done

  # Copy each support folder as a directory (no trailing slash), which preserves
  # its original top-level name under "Data Files" without embedding that name
  # in the remote shell path. Names containing spaces or '&' stay safe this way.
  for support_path in "$data_root"/*/; do
    support_name="${support_path%/}"
    support_name="${support_name##*/}"
    [[ "$support_name" == 'Exception Reports and Rebuttals by year' ]] && continue
    echo "==> support folder: $support_name"
    caffeinate -i "${RSYNC_BASE[@]}" --exclude '.DS_Store' \
      "${support_path%/}" "$VM_HOST:$VM_BASE/Data\\ Files/"
  done

  echo "==> support folder: .agreement_rag"
  caffeinate -i "${RSYNC_BASE[@]}" --exclude '.DS_Store' \
    "$data_root/.agreement_rag" "$VM_HOST:$VM_BASE/Data\\ Files/"
}

case "$WHAT" in
  code)   sync_code ;;
  env)    sync_env ;;
  inputs) sync_inputs ;;
  data)   sync_data ;;
  all)    sync_code; sync_env; sync_inputs; sync_data ;;
  *) echo "usage: $0 {code|env|inputs|data|all}" >&2; exit 2 ;;
esac
echo "done: $WHAT"
