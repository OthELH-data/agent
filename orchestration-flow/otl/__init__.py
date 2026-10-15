"""OTL exception-handling orchestration layer.

A multi-agent LangGraph pipeline that automates rebuttal drafting for OTL (Offshore Technical
Labor) "Labor Issues" audit exceptions. It uses `datastores/classified_exceptions.json` as its
baseline case record and can optionally add a locally staged `rebuttal_N_synthesis.json` produced by
the `excel_extractor` pipeline. It runs those inputs through a deterministic prep spine
(intake → rebuttal_history → category_check → optional verify_evidence → jade_analyzer) followed by a
highest-tier `orchestrator` hub that dispatches the remaining specialists (location, job_title,
matching) one at a time — each at most once — before a read-only wiki sidecar, asset-scoped agreement
retrieval, and the drafter.

After intake and before summarization, ``rebuttal_history`` reads every prior
``rebuttal_<n>_analysis.md`` for the same year/asset/exception, compacts it only when it reaches 85% of
the smallest active model context window, compares it with the fresh intake, and initializes the active
cycle file that each substantive agent updates idempotently.

Importing this package performs a one-time bootstrap (`otl._bootstrap`):
  * loads `orchestration-flow/.env` FIRST (so it wins over the extractor's own `.env`), and
  * puts `excel_extractor/` and `orchestration-flow/` on `sys.path` so the reused extractor
    modules (`import shared.llm`, …) and the local `code_classifier` / `evidence_inventory`
    resolve.
"""
from __future__ import annotations

from otl import _bootstrap  # noqa: F401  (side-effecting: env + sys.path)

__all__ = ["_bootstrap"]
