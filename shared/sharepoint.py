"""Upload synthesis JSONs to the client SharePoint / OneDrive-for-Business site.

Auth uses the MSAL device-code flow (works in locked-down IBM tenants): on first use it prints a
URL + one-time code to sign in with. The token is cached to disk (~/.jv_audit_msal_cache.bin) and
reused silently on subsequent runs, so a multi-exception batch prompts for the code only once —
not once per `python run.py` invocation.

The writer hands office365 a *refreshing* token provider (SharePointWriter._get_token) that it
re-invokes per request: access tokens live only ~1h, so a long-running batch would otherwise 401
partway through: the provider transparently refreshes (via the cached refresh token) instead.

Folders (<year>/<asset>/<exception>/) are created on demand; each report is uploaded as UTF-8
JSON. Transient throttling (429/503/500) is retried with exponential backoff.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import urlparse

# Public client id for "Microsoft Graph Command Line Tools" — a Microsoft first-party app
# registration that supports the device-code flow without needing our own app registration.
_CLIENT_ID = "14d82eec-204b-4c2f-b7e8-296a70dab67e"
_CACHE_PATH = Path.home() / ".jv_audit_msal_cache.bin"


def _sp_retry(func, max_retries: int = 8):
    """Retry a SharePoint call on transient errors with exponential backoff.

    Two failure classes are transient and retried: HTTP throttling / server errors (429/503/500,
    surfaced as ClientRequestException) and lower-level network drops — connection resets, "remote
    end closed connection without response", and read timeouts — which surface as
    requests.ConnectionError / Timeout / ChunkedEncodingError, NOT as a ClientRequestException.
    """
    from office365.runtime.client_request_exception import ClientRequestException
    from requests.exceptions import (ChunkedEncodingError, ConnectionError as ReqConnectionError,
                                     Timeout)

    _TRANSIENT_NET = (ReqConnectionError, Timeout, ChunkedEncodingError)
    for attempt in range(max_retries):
        try:
            return func()
        except ClientRequestException as exc:
            resp = getattr(exc, "response", None)
            status = getattr(resp, "status_code", None)
            if status in (429, 503, 500) and attempt < max_retries - 1:
                wait = 2 ** attempt * 5
                print(f"    SharePoint {status}; retrying in {wait}s "
                      f"(attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
        except _TRANSIENT_NET as exc:
            if attempt < max_retries - 1:
                wait = 2 ** attempt * 5
                print(f"    SharePoint connection error ({type(exc).__name__}); retrying in {wait}s "
                      f"(attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                raise


class SharePointWriter:
    """Authenticated writer that uploads JSON into a folder tree on a SharePoint site."""

    def __init__(self, site_url: str, folder_path: str):
        import msal
        from office365.sharepoint.client_context import ClientContext

        self.site_url = site_url.rstrip("/")
        # get_folder_by_server_relative_url needs a *server-relative* url (from the domain root):
        #   site '/personal/rose_kim_ibm_com' + folder 'Documents/.../Excel Extractor Output'
        site_path = urlparse(self.site_url).path.rstrip("/")
        self.base = f"{site_path}/{folder_path.strip('/')}"

        parsed = urlparse(self.site_url)
        self._scopes = [f"{parsed.scheme}://{parsed.hostname}/.default"]
        self._cache = msal.SerializableTokenCache()
        if _CACHE_PATH.exists():
            self._cache.deserialize(_CACHE_PATH.read_text())
        self._app = msal.PublicClientApplication(
            _CLIENT_ID, authority="https://login.microsoftonline.com/common",
            token_cache=self._cache)

        # Acquire once up front so any device-code sign-in happens now (not lazily mid-batch), then
        # hand office365 a *refreshing* provider: it re-invokes _get_token per request, so a batch
        # longer than the ~1h access-token lifetime keeps minting fresh tokens instead of 401ing.
        self._get_token()
        self.ctx = ClientContext(self.site_url).with_access_token(self._get_token)

    def _get_token(self):
        """Return a currently-valid TokenResponse, refreshing (or re-signing-in) as needed.

        Called by office365 on every request. acquire_token_silent is cheap when the cached access
        token is still valid (in-memory, no network) and transparently uses the refresh token when it
        has expired — so a multi-hour batch never outlives its token. Falls back to the interactive
        device-code flow only if the refresh token is also dead.
        """
        from office365.runtime.auth.token_response import TokenResponse

        accounts = self._app.get_accounts()
        result = self._app.acquire_token_silent(self._scopes, account=accounts[0]) if accounts else None
        if not result or "access_token" not in result:
            result = self._device_flow()
        if self._cache.has_state_changed:
            _CACHE_PATH.write_text(self._cache.serialize())
        if "access_token" not in result:
            raise RuntimeError(f"Auth failed: {result.get('error_description', 'unknown error')}")
        token = TokenResponse(access_token=result["access_token"], token_type="Bearer")
        token.expiresIn = result.get("expires_in", 3600)
        return token

    def _device_flow(self) -> dict:
        """Interactive device-code sign-in (used on first run or when the refresh token is dead)."""
        flow = self._app.initiate_device_flow(scopes=self._scopes)
        if "user_code" not in flow:
            raise RuntimeError(f"Device flow failed: {flow.get('error_description', '?')}")
        print(f"\n{'=' * 60}")
        print(f"To sign in, open: {flow['verification_uri']}")
        print(f"Enter code: {flow['user_code']}")
        print(f"{'=' * 60}\n")
        return self._app.acquire_token_by_device_flow(flow)

    def _ensure_folders(self, rel_dir: str) -> str:
        """Create each segment of <base>/<rel_dir> if missing; return its server-relative url."""
        cur = self.base
        for seg in (s for s in rel_dir.split("/") if s):
            cur = f"{cur}/{seg}"
            self.ctx.web.folders.add(cur)  # idempotent: returns the folder if it already exists
        _sp_retry(lambda: self.ctx.execute_query())
        return cur

    def folder_has_synthesis(self, rel_folder: str) -> bool:
        """Check if <base>/<rel_folder>/ exists and contains at least one rebuttal_*_synthesis.json."""
        from office365.runtime.client_request_exception import ClientRequestException

        folder_url = f"{self.base}/{rel_folder.strip('/')}"
        try:
            folder = self.ctx.web.get_folder_by_server_relative_url(folder_url)
            files = folder.files.filter("substringof('_synthesis.json', Name)")
            self.ctx.load(files)
            _sp_retry(lambda: self.ctx.execute_query())
            return len(files) > 0
        except ClientRequestException as exc:
            resp = getattr(exc, "response", None)
            status = getattr(resp, "status_code", None)
            if status == 404:
                return False
            raise

    def file_exists(self, rel_path: str) -> bool:
        """Check if the single file <base>/<rel_path> already exists on SharePoint."""
        from office365.runtime.client_request_exception import ClientRequestException

        rel_dir, _, name = rel_path.rpartition("/")
        folder_url = f"{self.base}/{rel_dir.strip('/')}" if rel_dir else self.base
        try:
            folder = self.ctx.web.get_folder_by_server_relative_url(folder_url)
            files = folder.files.filter(f"Name eq '{name}'")
            self.ctx.load(files)
            _sp_retry(lambda: self.ctx.execute_query())
            return len(files) > 0
        except ClientRequestException as exc:
            resp = getattr(exc, "response", None)
            status = getattr(resp, "status_code", None)
            if status == 404:            # the containing folder doesn't exist yet
                return False
            raise

    def list_files(self, rel_dir: str) -> list[str]:
        """List file names under <base>/<rel_dir>; return [] when the folder is absent."""
        from office365.runtime.client_request_exception import ClientRequestException

        folder_url = f"{self.base}/{rel_dir.strip('/')}" if rel_dir else self.base
        try:
            folder = self.ctx.web.get_folder_by_server_relative_url(folder_url)
            files = folder.files
            self.ctx.load(files)
            _sp_retry(lambda: self.ctx.execute_query())
            return [str(item.properties.get("Name") or "") for item in files]
        except ClientRequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 404:
                return []
            raise

    def read_text(self, rel_path: str) -> str | None:
        """Read one UTF-8 text file under the configured base; return None when absent."""
        from office365.runtime.client_request_exception import ClientRequestException

        rel_dir, _, name = rel_path.rpartition("/")
        folder_url = f"{self.base}/{rel_dir.strip('/')}" if rel_dir else self.base
        file_url = f"{folder_url}/{name}"
        try:
            binary = self.ctx.web.get_file_by_server_relative_url(file_url).get_content()
            _sp_retry(lambda: self.ctx.execute_query())
            raw = binary.value if hasattr(binary, "value") else binary
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            return str(raw)
        except ClientRequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 404:
                return None
            raise

    def _upload(self, rel_path: str, content: bytes) -> str:
        """Upload raw bytes to <base>/<rel_path> (creating folders); returns the server-relative url."""
        rel_dir, _, name = rel_path.rpartition("/")
        target_url = self._ensure_folders(rel_dir)
        target = self.ctx.web.get_folder_by_server_relative_url(target_url)
        _sp_retry(lambda: target.upload_file(name, content).execute_query())
        return f"{target_url}/{name}"

    def write_json(self, rel_path: str, data: dict) -> str:
        """Upload `data` as pretty JSON to <base>/<rel_path>; returns the server-relative url."""
        content = json.dumps(
            data, indent=2, ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
        return self._upload(rel_path, content)

    def write_text(self, rel_path: str, text: str) -> str:
        """Upload `text` verbatim (e.g. line-delimited JSONL) to <base>/<rel_path>; returns its url."""
        return self._upload(rel_path, text.encode("utf-8"))
