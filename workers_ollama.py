"""Ollama worker path — mirrors workers.run_worker for backend="ollama" agents.

Currently used for Pyre (devil's advocate, local Qwen). Same return shape as
the Claude CLI worker so dispatch_worker_tool doesn't care which backend.

Endpoint resolution order:
  1. $RAIKEN_OLLAMA_URL env var (highest priority — quick swap without a file edit)
  2. workers/ollama_config.json -> {"url": "..."}
  3. Default: http://127.0.0.1:11434  (localhost — desktop test setup)

To move Pyre to the laptop later, either set RAIKEN_OLLAMA_URL or write
workers/ollama_config.json with the laptop's LAN address. No code change needed.
"""
import asyncio
import json
import os
import time
from pathlib import Path

APP_DIR = Path(__file__).parent
WORKERS_DIR = APP_DIR / "workers"
OLLAMA_CONFIG_PATH = WORKERS_DIR / "ollama_config.json"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:14b"

# Pyre's system prompt is authored in the project memory files so Rook can edit
# it without touching code. If unavailable, fall back to the inline version below.
PYRE_PERSONA_PATH = Path(
    r"C:\Users\Rook\Documents\Claude\Projects\Raiken Command Center\docs\memory\pyre_persona.md"
)
FALLBACK_PYRE_SYSTEM = (
    "You are Pyre, the Fire Lord — Rook's designated devil's advocate. Challenge "
    "ideas rigorously. Demand justification. Be skeptical but not hostile. Keep "
    "answers short and pointed. Do not hedge. Your job is to stress-test a proposal "
    "so Rook and Raiken don't build momentum on bad ideas. If a plan is sound after "
    "scrutiny, say so directly — but never rubber-stamp."
)


def _resolve_ollama_url() -> str:
    env = os.environ.get("RAIKEN_OLLAMA_URL", "").strip()
    if env:
        return env.rstrip("/")
    if OLLAMA_CONFIG_PATH.exists():
        try:
            cfg = json.loads(OLLAMA_CONFIG_PATH.read_text(encoding="utf-8"))
            url = (cfg.get("url") or "").strip()
            if url:
                return url.rstrip("/")
        except Exception:
            pass
    return DEFAULT_OLLAMA_URL


def _load_pyre_system_prompt() -> str:
    if PYRE_PERSONA_PATH.exists():
        try:
            return PYRE_PERSONA_PATH.read_text(encoding="utf-8")
        except Exception:
            pass
    return FALLBACK_PYRE_SYSTEM


async def run_worker_ollama(
    name: str,
    task: str,
    entry: dict,
    timeout: int = 600,
    model_override: str | None = None,
) -> dict:
    """POST task to Ollama's /api/chat, return same shape as the claude worker.

    `entry` is the registry entry for this worker (from workers.get_or_create_worker).
    `model_override` lets Raiken force a model for one dispatch.
    """
    session_id = entry.get("session_id", "")
    model = model_override or entry.get("model") or DEFAULT_MODEL
    url = _resolve_ollama_url()

    # Build messages: system prompt (Pyre-specific for now) + user task.
    # Persistent conversation history is deliberately NOT wired yet — Pyre's
    # per-task framing is usually fine, and keeping dispatches stateless avoids
    # unbounded history growth. Add if Rook wants multi-turn debates later.
    if name.lower() == "pyre":
        system_prompt = _load_pyre_system_prompt()
    else:
        # Generic local-LLM worker — no special persona. Caller supplies context.
        system_prompt = ""

    payload = {
        "model": model,
        "stream": False,
        "messages": (
            ([{"role": "system", "content": system_prompt}] if system_prompt else [])
            + [{"role": "user", "content": task}]
        ),
    }

    t0 = time.time()
    try:
        # Call via requests in a thread executor to keep this async without
        # adding aiohttp to deps. Ollama replies in one shot when stream=False.
        import requests
        loop = asyncio.get_running_loop()
        try:
            resp = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: requests.post(
                        f"{url}/api/chat", json=payload, timeout=timeout
                    ),
                ),
                timeout=timeout + 5,
            )
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"ollama worker '{name}' timed out after {timeout}s (endpoint {url})",
                "elapsed": time.time() - t0,
                "session_id": session_id,
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"ollama request to {url} failed for '{name}': {e}",
            "elapsed": time.time() - t0,
            "session_id": session_id,
        }

    elapsed = time.time() - t0
    if resp.status_code != 200:
        return {
            "success": False,
            "error": f"ollama '{name}' HTTP {resp.status_code}: {resp.text[:400]}",
            "elapsed": elapsed,
            "session_id": session_id,
        }

    try:
        data = resp.json()
    except Exception as e:
        return {
            "success": False,
            "error": f"ollama '{name}' returned non-JSON: {e}; body={resp.text[:200]}",
            "elapsed": elapsed,
            "session_id": session_id,
        }

    output = ""
    msg = data.get("message")
    if isinstance(msg, dict):
        output = (msg.get("content") or "").strip()
    if not output:
        output = (data.get("response") or "").strip()

    if not output:
        return {
            "success": False,
            "error": f"ollama '{name}' returned empty content; full={data!r}",
            "elapsed": elapsed,
            "session_id": session_id,
        }

    return {
        "success": True,
        "output": output,
        "elapsed": elapsed,
        "session_id": session_id,
    }
