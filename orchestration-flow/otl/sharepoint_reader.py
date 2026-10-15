"""Read extractor synthesis JSON artifacts that seed OTL runs.

The extractor writes `rebuttal_<n>_synthesis.json` per exception. OTL uses the cycle-specific public
methods so a rebuttal run reads only its matching synthesis. The legacy latest-cycle methods remain
available for callers that explicitly need them. Both backends use the same
`<year>/<asset>/<exc>/rebuttal_<n>_synthesis.json` relative layout.

SharePoint auth is reused wholesale from `shared.sharepoint.SharePointWriter` (MSAL device-code +
refreshing token). In sandboxes where SharePoint egress is blocked the reader simply lands on the
local fallback, which is the path exercised by the P1 verification.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

_SYNTH_RE = re.compile(r"rebuttal_(\d+)_synthesis\.json$", re.I)


def _rel_folder(year: str, asset: str, exc_id: str) -> str:
    return f"{year}/{asset}/{exc_id}"


class SharePointReader:
    """Fetch exact-cycle or explicitly requested latest synthesis JSON artifacts."""

    def __init__(self, site_url: str | None = None, read_folder: str | None = None,
                 input_base: str | None = None):
        self.site_url = site_url or os.getenv("SHAREPOINT_URL") or ""
        self.read_folder = read_folder or os.getenv("SHAREPOINT_READ_FOLDER") or ""
        self.input_base = input_base or os.getenv("INPUT_BASE_PIPELINE") or os.getenv("INPUT_BASE") or ""
        self.target = (os.getenv("OUTPUT_TARGET") or "local").strip().lower()
        self._writer = None            # lazily built SharePointWriter (auth on first use)
        self._sp_disabled = False      # sticky: once SharePoint fails, don't keep retrying the batch

    # -- SharePoint side ---------------------------------------------------------------------------
    def _get_writer(self):
        if self._writer is None:
            from shared.sharepoint import SharePointWriter
            self._writer = SharePointWriter(self.site_url, self.read_folder)
        return self._writer

    def _sharepoint_latest(self, rel_folder: str) -> dict | None:
        """List `<read_folder>/<rel_folder>/`, pick the max-N synthesis, download + parse it."""
        from shared.sharepoint import _sp_retry

        writer = self._get_writer()
        ctx = writer.ctx
        folder_url = f"{writer.base}/{rel_folder.strip('/')}"
        folder = ctx.web.get_folder_by_server_relative_url(folder_url)
        files = folder.files
        ctx.load(files)
        _sp_retry(lambda: ctx.execute_query())

        best_n, best_name = -1, None
        for f in files:
            m = _SYNTH_RE.search(f.properties.get("Name", ""))
            if m and int(m.group(1)) > best_n:
                best_n, best_name = int(m.group(1)), f.properties["Name"]
        if best_name is None:
            return None

        file_url = f"{folder_url}/{best_name}"
        binary = ctx.web.get_file_by_server_relative_url(file_url).get_content()
        _sp_retry(lambda: ctx.execute_query())
        raw = binary.value if hasattr(binary, "value") else binary
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    def _sharepoint_cycle(self, rel_folder: str, rebuttal_cycle: int) -> dict | None:
        """Download and parse one exact cycle from the configured SharePoint read folder."""
        from shared.sharepoint import _sp_retry

        writer = self._get_writer()
        folder_url = f"{writer.base}/{rel_folder.strip('/')}"
        file_url = f"{folder_url}/rebuttal_{int(rebuttal_cycle)}_synthesis.json"
        binary = writer.ctx.web.get_file_by_server_relative_url(file_url).get_content()
        _sp_retry(lambda: writer.ctx.execute_query())
        raw = binary.value if hasattr(binary, "value") else binary
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    # -- local fallback ----------------------------------------------------------------------------
    def _local_latest(self, rel_folder: str) -> dict | None:
        """Load the max-N synthesis from `INPUT_BASE/<rel_folder>/`."""
        if not self.input_base:
            return None
        folder = Path(self.input_base) / rel_folder
        if not folder.is_dir():
            return None
        best_n, best_path = -1, None
        for p in folder.glob("rebuttal_*_synthesis.json"):
            m = _SYNTH_RE.search(p.name)
            if m and int(m.group(1)) > best_n:
                best_n, best_path = int(m.group(1)), p
        if best_path is None:
            return None
        return json.loads(best_path.read_text(encoding="utf-8"))

    def _local_cycle(self, rel_folder: str, rebuttal_cycle: int) -> dict | None:
        """Load one exact cycle from `INPUT_BASE/<rel_folder>/`."""
        if not self.input_base:
            return None
        path = (
            Path(self.input_base)
            / rel_folder
            / f"rebuttal_{int(rebuttal_cycle)}_synthesis.json"
        )
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # -- public ------------------------------------------------------------------------------------
    def sharepoint_synthesis(self, year: str, asset: str, exc_id: str,
                             rebuttal_cycle: int) -> dict | None:
        """Read one exact rebuttal-cycle synthesis from SharePoint only."""
        if not self.site_url or not self.read_folder:
            return None
        try:
            return self._sharepoint_cycle(
                _rel_folder(year, asset, exc_id), rebuttal_cycle
            )
        except Exception as exc:
            print(
                "  !! SharePoint extractor read failed "
                f"({type(exc).__name__}: {exc}) — treating rebuttal {rebuttal_cycle} "
                "synthesis as unavailable",
                file=sys.stderr,
            )
            return None

    def local_synthesis(self, year: str, asset: str, exc_id: str,
                        rebuttal_cycle: int) -> dict | None:
        """Read one exact rebuttal-cycle synthesis from local ``INPUT_BASE`` only."""
        return self._local_cycle(
            _rel_folder(year, asset, exc_id), rebuttal_cycle
        )


__all__ = ["SharePointReader"]
