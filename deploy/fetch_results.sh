#!/usr/bin/env bash
# Pull run results off the VM.
#
#   ./deploy/fetch_results.sh              # OTL reports + traces only
#   ./deploy/fetch_results.sh --all        # the whole output tree
#   ./deploy/fetch_results.sh --into-place # merge OTL artifacts into local excel_extractor/output
#
# Default lands in deploy/vm_results/<timestamp>/ so it can never overwrite the staged
# synthesis JSONs that local mode reads as input.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/vm.env.sh"

MODE="${1:-otl}"
REMOTE="$VM_BASE/Shell-JV-Audit/excel_extractor/output/"

# Compatible with macOS OpenRSYNC as well as Ubuntu's newer rsync.
RSYNC=(rsync -a -h --progress --partial -e "$SSH_CMD")

case "$MODE" in
  --all)
    DEST="$LOCAL_REPO/deploy/vm_results/$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$DEST"
    "${RSYNC[@]}" "$VM_HOST:$REMOTE" "$DEST/"
    ;;
  --into-place)
    DEST="$LOCAL_REPO/excel_extractor/output"
    echo "merging OTL artifacts into $DEST (synthesis inputs are left untouched)"
    "${RSYNC[@]}" --include '*/' \
      --include '*_otl.json' --include '*_otl_trace.jsonl' --exclude '*' \
      "$VM_HOST:$REMOTE" "$DEST/"
    ;;
  otl)
    DEST="$LOCAL_REPO/deploy/vm_results/$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$DEST"
    "${RSYNC[@]}" --prune-empty-dirs --include '*/' \
      --include '*_otl.json' --include '*_otl_trace.jsonl' --exclude '*' \
      "$VM_HOST:$REMOTE" "$DEST/"
    ;;
  *) echo "usage: $0 [otl|--all|--into-place]" >&2; exit 2 ;;
esac

echo
echo "downloaded to: $DEST"
find "$DEST" -type f | head -20
echo "($(find "$DEST" -type f | wc -l | tr -d ' ') files)"
