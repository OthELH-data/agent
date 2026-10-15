"""One-time environment + import-path bootstrap for the OTL layer.

Two things have to happen before any reused extractor module is imported:

1. **Env loading.** `shared.llm` / `shared.output_paths` call `load_dotenv(repo_root/.env)` at
   import time. We load the same repo-root `.env` here too, so anything imported from `otl` before an
   extractor module (e.g. `code_classifier`) still sees it. There is a single `.env`, at the repo
   root — no package-local `.env` files.

2. **sys.path.** The extractor uses package-relative imports (`from .tokens import …`), so it must be
   importable as the package `extractor` — which means `excel_extractor/` (its parent) must be on
   `sys.path`. The local `code_classifier` and `evidence_inventory` live directly under
   `orchestration-flow/`, so that directory goes on the path too. The repository root is included for
   the shared top-level `wiki` package used by the OTL sidecar.

Import order matters: `otl/__init__.py` imports this module first, so `import otl.<anything>` triggers
the bootstrap before the submodule's own extractor imports run.
"""
from __future__ import annotations

import sys
from pathlib import Path

# orchestration-flow/otl/_bootstrap.py → parents[0]=otl, [1]=orchestration-flow, [2]=repo root
_OTL_DIR = Path(__file__).resolve().parent
ORCH_DIR = _OTL_DIR.parent                       # orchestration-flow/
REPO_ROOT = ORCH_DIR.parent                      # Shell-JV-Audit/
EXTRACTOR_DIR = REPO_ROOT / "excel_extractor"    # holds the `extractor` package
# DATA_ROOT = two levels above orchestration-flow → ".../Shell JV Audit/Data Files"
DATA_ROOT = REPO_ROOT.parent / "Data Files"

# 1. Load the repo-root .env. Harmless to call again later from shared.llm / shared.output_paths —
#    load_dotenv is idempotent and does not override already-set values.
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ModuleNotFoundError:  # pragma: no cover — python-dotenv is a hard dep in requirements.txt
    pass

# 2. Put the extractor, orchestration layer, and shared repository packages on sys.path, ahead of
#    anything else, without duplicating entries.
for _p in (str(EXTRACTOR_DIR), str(ORCH_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__all__ = ["ORCH_DIR", "REPO_ROOT", "EXTRACTOR_DIR", "DATA_ROOT"]
