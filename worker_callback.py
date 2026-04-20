"""HTTP callback server: workers -> orchestrator IPC.

Dispatched workers are separate `claude` subprocesses. They can't share
Python state with the orchestrator, so they talk to it over a localhost
HTTP endpoint. The orchestrator stands this server up at boot on a random
free port and passes the URL + a random auth token to every worker via
env vars (RAIKEN_CALLBACK_URL / RAIKEN_CALLBACK_TOKEN). Workers invoke
`worker_tools.py` which POSTs here.

Endpoints:
  POST /status            — push a short phase label ("researching", etc.)
  POST /dispatch_sub      — parent wants to spawn a sub-agent
  POST /escalate          — parent asks for permission to spawn Opus-tier
  POST /ask_raiken        — parent asks Raiken Agent for advice (depth-1 cap)
  POST /compact_memory    — parent logs a memory-compaction event

Every request must carry X-Raiken-Token matching the server token or is
rejected 401. Binding to 127.0.0.1 is a belt-and-suspenders defense on top
of that — the token is the primary gate.
"""
from __future__ import annotations

import json
import secrets
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable


class _Handler(BaseHTTPRequestHandler):
    # Subclass sets these at construction time (see WorkerCallbackServer.start).
    server_token: str = ""
    on_status: Callable[[dict], dict] | None = None
    on_dispatch_sub: Callable[[dict], dict] | None = None
    on_escalate: Callable[[dict], dict] | None = None
    on_ask_raiken: Callable[[dict], dict] | None = None
    on_compact_memory: Callable[[dict], dict] | None = None

    def log_message(self, format, *args):   # noqa: A003
        # Silence the default stderr access log — it fires on every worker
        # heartbeat and spams the console. Real errors go through our
        # _send_json(500) path.
        return

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def _read_json(self) -> dict | None:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if n <= 0 or n > 64 * 1024:
            return None
        raw = self.rfile.read(n)
        try:
            obj = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        return obj if isinstance(obj, dict) else None

    def do_POST(self):   # noqa: N802
        token = self.headers.get("X-Raiken-Token") or ""
        if not self.server_token or token != self.server_token:
            return self._send_json(401, {"error": "bad token"})

        payload = self._read_json()
        if payload is None:
            return self._send_json(400, {"error": "invalid JSON body"})

        try:
            if self.path == "/status" and self.on_status is not None:
                return self._send_json(200, self.on_status(payload) or {"ok": True})
            if self.path == "/dispatch_sub" and self.on_dispatch_sub is not None:
                return self._send_json(200, self.on_dispatch_sub(payload))
            if self.path == "/escalate" and self.on_escalate is not None:
                return self._send_json(200, self.on_escalate(payload))
            if self.path == "/ask_raiken" and self.on_ask_raiken is not None:
                return self._send_json(200, self.on_ask_raiken(payload))
            if self.path == "/compact_memory" and self.on_compact_memory is not None:
                return self._send_json(200, self.on_compact_memory(payload))
        except Exception as e:
            return self._send_json(
                500, {"error": f"{type(e).__name__}: {e}"},
            )
        return self._send_json(404, {"error": "unknown path"})


def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class WorkerCallbackServer:
    """Thin wrapper around ThreadingHTTPServer bound to 127.0.0.1.

    Usage:
        server = WorkerCallbackServer(
            on_status=app.handle_status,
            on_dispatch_sub=app.handle_dispatch_sub,
            on_escalate=app.handle_escalate,
        )
        url, token = server.start()   # non-blocking; returns http://127.0.0.1:<port>
        # ... workers hit `url` with header X-Raiken-Token: token ...
        server.stop()
    """

    def __init__(
        self,
        on_status: Callable[[dict], dict],
        on_dispatch_sub: Callable[[dict], dict],
        on_escalate: Callable[[dict], dict],
        on_ask_raiken: Callable[[dict], dict] | None = None,
        on_compact_memory: Callable[[dict], dict] | None = None,
    ):
        self.on_status = on_status
        self.on_dispatch_sub = on_dispatch_sub
        self.on_escalate = on_escalate
        self.on_ask_raiken = on_ask_raiken
        self.on_compact_memory = on_compact_memory
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.url: str = ""
        self.token: str = ""

    def start(self) -> tuple[str, str]:
        port = _find_free_port()
        token = secrets.token_urlsafe(24)

        # Build a fresh handler subclass per server so the closure-captured
        # callbacks + token don't leak between instances (important for
        # hot-reload during tests).
        handler_cls = type(
            "_BoundHandler",
            (_Handler,),
            {
                "server_token": token,
                "on_status": staticmethod(self.on_status),
                "on_dispatch_sub": staticmethod(self.on_dispatch_sub),
                "on_escalate": staticmethod(self.on_escalate),
                "on_ask_raiken": staticmethod(self.on_ask_raiken) if self.on_ask_raiken else None,
                "on_compact_memory": staticmethod(self.on_compact_memory) if self.on_compact_memory else None,
            },
        )
        self._server = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="raiken-worker-callback",
            daemon=True,
        )
        self._thread.start()
        self.url = f"http://127.0.0.1:{port}"
        self.token = token
        return self.url, self.token

    def stop(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
            self._server = None
