"""Bitwarden CLI wrapper with in-memory session management.

Security contract:
- Master password is entered via the tkinter UI dialog and piped to `bw unlock`
  via an env var (cleared right after). It never lives in Python variables beyond
  the unlock call, never hits disk, and never passes through the Claude Agent SDK
  (so Anthropic never sees it).
- Session key lives ONLY in this process's memory. Not written to disk. Idle-
  timeout auto-locks.
- Individual item passwords are fetched only when a local action tool needs them,
  and are NOT returned to the Claude model layer. Tools exposed to Claude return
  outcomes like "copied to clipboard" or "email sent" — never the raw secret.
- Every operation is audit-logged to logs/bitwarden_audit.log with timestamp,
  action, item name (if applicable), and result. No secrets in the log.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

APP_DIR = Path(__file__).parent
LOG_DIR = APP_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
AUDIT_LOG = LOG_DIR / "bitwarden_audit.log"

DEFAULT_IDLE_TIMEOUT_SEC = 1800  # 30 min
_NO_WINDOW = 0x08000000  # subprocess.CREATE_NO_WINDOW


def _find_bw() -> str:
    """Locate bw.exe. Try PATH first, then the winget install location."""
    p = shutil.which("bw")
    if p:
        return p
    candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages" / "Bitwarden.CLI_Microsoft.Winget.Source_8wekyb3d8bbwe" / "bw.exe"
    if candidate.exists():
        return str(candidate)
    return "bw"  # fallback, may fail


BW_BIN = _find_bw()


def _audit(action: str, detail: str = "", success: bool = True):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    result = "OK" if success else "FAIL"
    line = f"[{ts}] {result} {action} {detail}\n"
    try:
        with open(AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def _run_bw(args: list[str], env_extra: dict | None = None, timeout: int = 30) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [BW_BIN, *args],
        capture_output=True, text=True, timeout=timeout, env=env,
        creationflags=_NO_WINDOW if os.name == "nt" else 0,
    )


class BitwardenSession:
    def __init__(self, idle_timeout_sec: int = DEFAULT_IDLE_TIMEOUT_SEC):
        self._session_key: str | None = None
        self._last_used: float = 0.0
        self._idle_timeout = idle_timeout_sec
        self._lock = threading.Lock()

    # --- state ---------------------------------------------------------------
    def cli_available(self) -> bool:
        try:
            r = _run_bw(["--version"], timeout=10)
            return r.returncode == 0
        except Exception:
            return False

    def is_unlocked(self) -> bool:
        with self._lock:
            if self._session_key is None:
                return False
            if time.time() - self._last_used > self._idle_timeout:
                self._session_key = None
                _audit("idle-auto-lock")
                return False
            return True

    def status(self) -> dict:
        """Non-secret status summary."""
        if not self.cli_available():
            return {"state": "cli-missing", "bw_bin": BW_BIN}
        try:
            r = _run_bw(["status"], timeout=10)
            data = json.loads(r.stdout) if r.stdout else {}
        except Exception as e:
            return {"state": "error", "detail": str(e)[:200]}
        return {
            "state": data.get("status", "unknown"),   # "unauthenticated" | "locked" | "unlocked"
            "server_url": data.get("serverUrl"),
            "email": data.get("userEmail"),
            "in_memory_unlocked": self.is_unlocked(),
            "session_timeout_remaining": (
                max(0, int(self._idle_timeout - (time.time() - self._last_used)))
                if self._session_key else None
            ),
        }

    # --- unlock / lock -------------------------------------------------------
    def unlock_with_password(self, master_password: str) -> tuple[bool, str]:
        """Unlock the vault. Password flows via env var (not argv / not stdout)."""
        if not self.cli_available():
            return False, "bw CLI not installed or not found"
        try:
            r = _run_bw(
                ["unlock", "--raw", "--passwordenv", "BW_PASSWORD"],
                env_extra={"BW_PASSWORD": master_password},
                timeout=30,
            )
        except Exception as e:
            _audit("unlock", success=False, detail=f"exception: {type(e).__name__}")
            return False, f"unlock failed: {e}"

        if r.returncode != 0:
            _audit("unlock", success=False, detail=f"exit {r.returncode}")
            return False, "unlock failed (wrong password, not logged in, or CLI error)"

        session_key = r.stdout.strip()
        if not session_key:
            _audit("unlock", success=False, detail="empty session key")
            return False, "unlock returned empty session key"

        with self._lock:
            self._session_key = session_key
            self._last_used = time.time()
        _audit("unlock")
        return True, "vault unlocked"

    def lock(self) -> bool:
        try:
            _run_bw(["lock"], timeout=10)
        except Exception:
            pass
        with self._lock:
            self._session_key = None
        _audit("lock")
        return True

    # --- reads ---------------------------------------------------------------
    def _run_read(self, args: list[str]) -> tuple[bool, str]:
        if not self.is_unlocked():
            return False, "vault is locked"
        with self._lock:
            key = self._session_key
            self._last_used = time.time()
        try:
            r = _run_bw([*args, "--session", key], timeout=30)
        except Exception as e:
            return False, f"bw exec failed: {e}"
        if r.returncode != 0:
            return False, f"bw exit {r.returncode}: {r.stderr.strip()[:200]}"
        return True, r.stdout.strip()

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """Returns non-secret metadata only (no passwords/TOTP/notes)."""
        ok, output = self._run_read(["list", "items", "--search", query])
        if not ok:
            _audit("search", detail=f"query={query!r}", success=False)
            return []
        try:
            items = json.loads(output)
        except Exception:
            return []
        _audit("search", detail=f"query={query!r} hits={len(items)}")
        out = []
        for i in items[:limit]:
            login = i.get("login") or {}
            uris = login.get("uris") or []
            first_uri = uris[0].get("uri") if uris else None
            out.append({
                "id": i.get("id"),
                "name": i.get("name"),
                "username": login.get("username"),
                "uri": first_uri,
                "has_totp": bool(login.get("totp")),
            })
        return out

    # The get_* methods fetch secrets. DO NOT expose their returns to Claude tools.
    def get_password(self, name_or_id: str) -> str | None:
        ok, output = self._run_read(["get", "password", name_or_id])
        _audit("get-password", detail=f"item={name_or_id!r}", success=ok)
        return output if ok else None

    def get_username(self, name_or_id: str) -> str | None:
        ok, output = self._run_read(["get", "username", name_or_id])
        _audit("get-username", detail=f"item={name_or_id!r}", success=ok)
        return output if ok else None

    def get_totp(self, name_or_id: str) -> str | None:
        ok, output = self._run_read(["get", "totp", name_or_id])
        _audit("get-totp", detail=f"item={name_or_id!r}", success=ok)
        return output if ok else None

    def get_uri(self, name_or_id: str) -> str | None:
        ok, output = self._run_read(["get", "uri", name_or_id])
        _audit("get-uri", detail=f"item={name_or_id!r}", success=ok)
        return output if ok else None
