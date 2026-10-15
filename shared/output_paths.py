"""Where synthesis JSONs are written, keyed by year/asset/exception.

Two backends, selected by OUTPUT_TARGET in .env:
  * "local"      (default) → <OUTPUT_BASE_EXTRACTOR>/<year>/<asset>/<exception-id>/
                             rebuttal_<n>_synthesis.json
  * "sharepoint"           → uploaded to <SHAREPOINT_FOLDER>/<year>/<asset>/<exception-id>/... on
                             the SHAREPOINT_URL site (see shared/sharepoint.py).

The relative layout `<year>/<asset>/<exception-id>/rebuttal_<n>_synthesis.json` is identical for
both; only the root/transport differs.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Repo-root .env — no package-local .env; excel_extractor and orchestration-flow both read this file.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ModuleNotFoundError:  # pragma: no cover
    pass

_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")


def output_base() -> Path:
    """Legacy root folder for local outputs (read from ``OUTPUT_BASE`` in .env)."""
    base = os.getenv("OUTPUT_BASE")
    if not base:
        sys.exit("OUTPUT_BASE not set in .env — cannot determine output directory.")
    return Path(base)


def _dedicated_output_base(env_name: str, *, legacy_env: str = "OUTPUT_BASE") -> Path:
    """Resolve one pipeline's local output root, retaining the legacy shared fallback."""
    base = os.getenv(env_name) or os.getenv(legacy_env)
    if not base:
        sys.exit(f"{env_name} (or legacy {legacy_env}) not set in .env — cannot determine output directory.")
    return Path(base)


def extractor_output_base() -> Path:
    """Root for Excel Extractor artifacts (``OUTPUT_BASE_EXTRACTOR``)."""
    return _dedicated_output_base("OUTPUT_BASE_EXTRACTOR")


def pipeline_output_base() -> Path:
    """Root for labor-pipeline artifacts (``OUTPUT_BASE_PIPELINE``)."""
    return _dedicated_output_base("OUTPUT_BASE_PIPELINE")


def infer_year_asset(path: str | Path) -> tuple[str, str]:
    """Pull (year, asset) from a corpus path: the 4-digit year segment and the segment after it.

    e.g. '.../2024/GreatWhite/Rebuttal 0/...' -> ('2024', 'GreatWhite').
    """
    parts = Path(path).parts
    for i, seg in enumerate(parts):
        if _YEAR_RE.match(seg) and i + 1 < len(parts):
            return seg, parts[i + 1]
    return "unknown_year", "unknown_asset"


def synthesis_rel_path(year: str, asset: str, exception_id: str, rebuttal_cycle: int = 0) -> str:
    """Backend-agnostic relative path: '<year>/<asset>/<exception>/rebuttal_<n>_synthesis.json'."""
    return f"{year}/{asset}/{exception_id}/rebuttal_{rebuttal_cycle}_synthesis.json"


def synthesis_path(year: str, asset: str, exception_id: str, rebuttal_cycle: int = 0,
                   base: Path | None = None) -> Path:
    """Full LOCAL path for one cycle's synthesis JSON in the year/asset/exception tree."""
    root = base or output_base()
    return root / synthesis_rel_path(year, asset, exception_id, rebuttal_cycle)


# --- output sinks: a uniform .write(rel_path, data) → location string, local or SharePoint ------

class LocalSink:
    """Write JSON to the local OUTPUT_BASE tree."""

    def __init__(self, base: Path | None = None):
        self.base = base or output_base()

    def write(self, rel_path: str, data: dict) -> str:
        out = self.base / rel_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        return str(out)

    def write_text(self, rel_path: str, text: str) -> str:
        """Write raw text (e.g. line-delimited JSONL) verbatim to `<base>/<rel_path>`."""
        out = self.base / rel_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        return str(out)

    def exists(self, rel_path: str) -> bool:
        """True if the single file `<base>/<rel_path>` already exists locally."""
        return (self.base / rel_path).is_file()

    def folder_exists(self, rel_folder: str) -> bool:
        """Check whether `<base>/<rel_folder>/` exists and contains at least one synthesis JSON."""
        folder = self.base / rel_folder
        if not folder.is_dir():
            return False
        return any(folder.glob("rebuttal_*_synthesis.json"))


class SharePointSink:
    """Upload JSON to the SharePoint site (auth happens once, when constructed).

    Falls back to writing locally — so the pipeline doesn't lose work — on either a 401 Unauthorized
    (expired/revoked token) or a persistent connection error (SharePoint unreachable after
    `_sp_retry` exhausts its backoff). The fallback is sticky: once tripped, all subsequent writes in
    this run go local.
    """

    def __init__(self, site_url: str, folder_path: str, fallback_base: Path | None = None):
        from .sharepoint import SharePointWriter

        self.writer = SharePointWriter(site_url, folder_path)
        self._fallback: LocalSink | None = None
        self._fallback_base = fallback_base

    @staticmethod
    def _fallback_reason(exc: Exception) -> str | None:
        """A short reason string if `exc` is recoverable by writing locally, else None.

        401 = expired/revoked token; a connection/timeout error means SharePoint stayed unreachable
        through every `_sp_retry` attempt. Either way, a local write beats losing the case's output.
        """
        from requests.exceptions import (ChunkedEncodingError,
                                         ConnectionError as ReqConnectionError, Timeout)
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 401:
            return "401 (token expired)"
        if isinstance(exc, (ReqConnectionError, Timeout, ChunkedEncodingError)):
            return f"connection error ({type(exc).__name__})"
        return None

    def _use_fallback(self, reason: str) -> "LocalSink":
        """Activate (once) and return the sticky local fallback sink."""
        if self._fallback is None:
            self._fallback = LocalSink(self._fallback_base)
            print(f"  !! SharePoint {reason} — falling back to local: {self._fallback.base}",
                  file=sys.stderr)
        return self._fallback

    def folder_exists(self, rel_folder: str) -> bool:
        """Check whether `<base>/<rel_folder>/` exists on SharePoint and has a synthesis JSON."""
        if self._fallback:
            return self._fallback.folder_exists(rel_folder)
        try:
            return self.writer.folder_has_synthesis(rel_folder)
        except Exception as exc:
            reason = self._fallback_reason(exc)
            if reason:
                return self._use_fallback(reason).folder_exists(rel_folder)
            return False

    def write(self, rel_path: str, data: dict) -> str:
        if self._fallback:
            return self._fallback.write(rel_path, data)
        try:
            return self.writer.write_json(rel_path, data)
        except Exception as exc:
            reason = self._fallback_reason(exc)
            if reason:
                return self._use_fallback(reason).write(rel_path, data)
            raise

    def write_text(self, rel_path: str, text: str) -> str:
        if self._fallback:
            return self._fallback.write_text(rel_path, text)
        try:
            return self.writer.write_text(rel_path, text)
        except Exception as exc:
            reason = self._fallback_reason(exc)
            if reason:
                return self._use_fallback(reason).write_text(rel_path, text)
            raise

    def exists(self, rel_path: str) -> bool:
        """True if the file already exists on SharePoint (checks the local fallback once tripped).

        A non-recoverable error is swallowed as False so an unconfirmable check never wrongly skips a
        case — better to attempt the run (its write falls back to local) than silently drop it.
        """
        if self._fallback:
            return self._fallback.exists(rel_path)
        try:
            return self.writer.file_exists(rel_path)
        except Exception as exc:
            reason = self._fallback_reason(exc)
            if reason:
                return self._use_fallback(reason).exists(rel_path)
            return False


def make_sink():
    """Build the extractor output sink using its dedicated local root."""
    target = (os.getenv("OUTPUT_TARGET") or "local").strip().lower()
    local_base = extractor_output_base()
    if target == "sharepoint":
        url = os.getenv("SHAREPOINT_URL")
        folder = os.getenv("SHAREPOINT_FOLDER")
        if not (url and folder):
            sys.exit("OUTPUT_TARGET=sharepoint requires SHAREPOINT_URL and SHAREPOINT_FOLDER in .env")
        return SharePointSink(url, folder, fallback_base=local_base)
    if target != "local":
        sys.exit(f"Unknown OUTPUT_TARGET={target!r} (expected 'local' or 'sharepoint')")
    return LocalSink(local_base)
