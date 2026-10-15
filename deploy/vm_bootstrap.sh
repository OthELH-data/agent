#!/usr/bin/env bash
# Runs ON the VM. Installs system deps, builds the venv, and generates the Linux root .env.
# Idempotent: safe to re-run after a code sync.
set -euo pipefail

VM_BASE="${VM_BASE:-/home/azureuser/shell-jv-audit}"
PY_VERSION="${PY_VERSION:-3.11}"
REPO="$VM_BASE/Shell-JV-Audit"
VENV="$VM_BASE/venv"
STAGING="$VM_BASE/env-staging"
PY="python${PY_VERSION}"

echo "==> system packages"
export DEBIAN_FRONTEND=noninteractive
if ! command -v "$PY" >/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update -qq
fi
sudo apt-get install -y -qq \
  "${PY}" "${PY}-venv" "${PY}-dev" \
  build-essential tesseract-ocr tmux

echo "==> venv at $VENV ($($PY --version))"
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
# The VM also stores the large audit corpus locally. Avoid retaining downloaded
# wheels in ~/.cache/pip after install, which can otherwise exhaust its OS disk.
"$VENV/bin/pip" install --no-cache-dir --upgrade -q pip wheel
# The orchestration requirements include the extractor requirements and the
# project's shared runtime dependencies, so the VM supports both entry points.
"$VENV/bin/pip" install --no-cache-dir -q -r "$REPO/orchestration-flow/requirements.txt"

# So the documented `.venv/bin/python ...` commands work unchanged from the repo root.
[ -e "$REPO/.venv" ] || ln -s "$VENV" "$REPO/.venv"

test -f "$STAGING/root.env" || {
  echo "ERROR: missing $STAGING/root.env. From your Mac, run: ./deploy/sync_to_vm.sh env" >&2
  exit 1
}

echo "==> generating root .env (local IO, Linux paths)"
"$VENV/bin/python" - "$REPO" "$STAGING" <<'PY'
import sys, pathlib, re

repo, staging = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
mac_repo = "/Users/harshakolachina/Documents/Projects/Shell JV Audit/Shell-JV-Audit"
extractor_output_dir = repo / "output" / "extractor" / "Run 1"
labor_output_dir = repo / "output" / "labor" / "Run 1"
rebuttal_history_dir = repo / "output" / "rebuttal_history" / "Run 1"

# Force local IO with distinct extractor, labor, and rebuttal-history roots.
overrides = {
    "OUTPUT_TARGET": "local",
    "OUTPUT_BASE_EXTRACTOR": str(extractor_output_dir),
    "INPUT_BASE_PIPELINE": str(extractor_output_dir),
    "OUTPUT_BASE_PIPELINE": str(labor_output_dir),
    "LOCAL_REBUTTAL_FOLDER": str(rebuttal_history_dir),
}

for src, dest in [(staging / "root.env", repo / ".env")]:
    lines = src.read_text().splitlines()
    seen = set()
    out = []
    for line in lines:
        m = re.match(r"^(\w+)=(.*)$", line)
        if m and m.group(1) in overrides:
            key = m.group(1)
            out.append(f"{key}={overrides[key]}")
            seen.add(key)
        else:
            # Rewrite any other macOS path that points into the repo.
            out.append(line.replace(mac_repo, str(repo)))
    for key, val in overrides.items():
        if key not in seen:
            out.append(f"{key}={val}")
    dest.write_text("\n".join(out) + "\n")
    dest.chmod(0o600)
    print(f"  wrote {dest}")
PY

mkdir -p "$REPO/output/extractor/Run 1" \
         "$REPO/output/labor/Run 1" \
         "$REPO/output/rebuttal_history/Run 1"

echo
echo "==> verification"
cd "$REPO"
"$VENV/bin/python" - <<'PY'
import os, sys, importlib
sys.path.insert(0, "orchestration-flow")
import otl._bootstrap as b
print("  DATA_ROOT      :", b.DATA_ROOT, "exists" if b.DATA_ROOT.exists() else "MISSING")
print("  OUTPUT_TARGET  :", os.getenv("OUTPUT_TARGET"))
print("  extractor out  :", os.getenv("OUTPUT_BASE_EXTRACTOR"))
print("  pipeline input :", os.getenv("INPUT_BASE_PIPELINE"))
print("  pipeline out   :", os.getenv("OUTPUT_BASE_PIPELINE"))
print("  rebuttal hist. :", os.getenv("LOCAL_REBUTTAL_FOLDER"))
for mod in ("pandas", "fitz", "openpyxl", "langgraph", "anthropic", "rapidfuzz", "pytesseract"):
    try:
        importlib.import_module(mod)
        print(f"  import {mod:<12}: ok")
    except Exception as exc:
        print(f"  import {mod:<12}: FAILED {exc}")
PY
echo
echo "bootstrap complete. Activate with: source $VENV/bin/activate"
