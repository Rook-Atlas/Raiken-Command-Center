"""Worker dispatch for Raiken orchestrator.

Each worker is a persistent Claude Code session identified by a human-readable name
("compliance", "research", "refactor", etc.). Under the hood, the name maps to a
stable UUID session-id stored in workers/registry.json. Dispatches spawn the bundled
`claude -p --session-id <id> <task>` CLI as an async subprocess and stream back its
output. Workers survive Raiken restarts AND machine reboots because the session
transcript lives on disk under Claude's session storage.

Auth: the subprocess inherits the user's `claude login` credentials (same Max quota
the orchestrator is using). Zero extra billing.
"""
import asyncio
import json
import threading
import time
import uuid
from pathlib import Path

APP_DIR = Path(__file__).parent
WORKERS_DIR = APP_DIR / "workers"
WORKERS_DIR.mkdir(exist_ok=True)
REGISTRY_PATH = WORKERS_DIR / "registry.json"

# In-memory registry cache. The UI polls list_workers() every second to paint
# the agent panel; without a cache that's a disk read + JSON parse at 1 Hz on
# the Tk main thread. The cache is invalidated on every _save_registry, and
# all writers go through _save_registry, so it stays authoritative.
_REGISTRY_CACHE: dict | None = None
_REGISTRY_CACHE_LOCK = threading.Lock()

# Claude Code CLI bundled with the Agent SDK.
CLAUDE_BIN = APP_DIR / ".venv" / "Lib" / "site-packages" / "claude_agent_sdk" / "_bundled" / "claude.exe"

DEFAULT_TIMEOUT_SEC = 600  # 10 minutes cap per worker dispatch

# Prepended to every dispatched worker's task. Without this, worker models
# self-restrict to their cwd and refuse to touch sibling directories even when
# `additionalDirectories` in settings.json technically allows it. Stating the
# grant explicitly in the user message resolves the ambiguity so workers
# proceed instead of bouncing back with "outside my allowed path" reports.
WORKER_PERMISSION_PREAMBLE = (
    "You are a dispatched worker running under RCC (Raiken Command Center). "
    "Rook — your user — has granted you full read and write access to:\n"
    "  * C:\\Users\\Rook\\AI\\ (all subfolders: raiken, xtts, ffmpeg-shared, etc.)\n"
    "  * C:\\Users\\Rook\\Documents\\Claude\\ (all project folders)\n"
    "  * C:\\Users\\Rook\\Downloads\\\n"
    "  * C:\\Users\\Rook\\.claude\\ (config + settings only; do not edit without explicit instruction)\n"
    "\n"
    "Do NOT refuse or request permission on the grounds that a path is "
    "'outside your project' or 'outside your allowed working directory.' "
    "It is not. Your cwd is where you were spawned, but your authority extends "
    "across all paths listed above. Proceed with edits, writes, reads, shell "
    "commands, and any tool calls the task requires. If you hit a real "
    "permission-system block, report it precisely (tool name + path + error) — "
    "but do not pre-refuse based on your own assumptions about scope.\n"
    "\n"
    "--- Orchestrator tools (RCC callback CLI) ---\n"
    "You can talk back to the RCC orchestrator by invoking the following "
    "shell command (Python helper bundled at C:\\Users\\Rook\\AI\\raiken\\worker_tools.py — "
    "invoke via `python C:\\Users\\Rook\\AI\\raiken\\worker_tools.py <subcommand>`). "
    "Env vars RAIKEN_CALLBACK_URL / RAIKEN_CALLBACK_TOKEN / RAIKEN_WORKER_NAME are "
    "already set in your environment; you don't need to pass them.\n"
    "\n"
    "  status \"<short phase label>\"\n"
    "     Push a 3-6 word verb-led present-progressive label to the RCC UI "
    "(examples: \"researching web UI\", \"writing up findings\", \"running tests\"). "
    "Call this on significant phase transitions so Rook can see what you're doing. "
    "Overrides whatever the initial dispatch heuristic set.\n"
    "\n"
    "  dispatch-sub --tier <haiku|sonnet|opus> \"<task>\"\n"
    "     Spawn a sub-agent under YOUR name. RCC picks a random thematic name "
    "from the requested tier's pool and prints it on stdout. This call BLOCKS "
    "until the sub-agent completes, then prints its final text output — treat "
    "it like a Task tool: fire-and-read. Use for parallelizable subtasks (one "
    "sub researches docs while you keep writing code). Haiku = fast/simple "
    "scouts, Sonnet = specialists (default), Opus = heavy hitters (require "
    "escalation approval; use sparingly).\n"
    "\n"
    "  escalate --tier <haiku|sonnet|opus> \"<task>\"\n"
    "     Request approval BEFORE dispatching at a tier that requires "
    "escalation (currently opus). Returns JSON {\"approved\": bool, \"reason\": str}. "
    "If approved, follow up with dispatch-sub.\n"
    "\n"
    "These are ordinary Bash-tool shell invocations. If the commands exit with "
    "a non-zero code, the error is printed to stderr — do not retry blindly, "
    "read the message and adjust.\n"
    "\n"
    "--- Task ---\n"
)


# =============================================================================
# Worker registry (name -> session_id mapping)
# =============================================================================
def _load_registry() -> dict:
    global _REGISTRY_CACHE
    with _REGISTRY_CACHE_LOCK:
        if _REGISTRY_CACHE is not None:
            return _REGISTRY_CACHE
        if not REGISTRY_PATH.exists():
            _REGISTRY_CACHE = {}
            return _REGISTRY_CACHE
        try:
            _REGISTRY_CACHE = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            _REGISTRY_CACHE = {}
        return _REGISTRY_CACHE


def _save_registry(reg: dict):
    global _REGISTRY_CACHE
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    with _REGISTRY_CACHE_LOCK:
        _REGISTRY_CACHE = reg


def get_or_create_worker(name: str) -> dict:
    """Return registry entry for the named worker, creating one if first time."""
    reg = _load_registry()
    if name in reg:
        reg[name]["last_used"] = time.time()
        _save_registry(reg)
        return reg[name]
    entry = {
        "name": name,
        "session_id": str(uuid.uuid4()),
        "created": time.time(),
        "last_used": time.time(),
        "dispatches": 0,
        "session_created": False,  # flips True after first successful dispatch
        "cwd": None,
        "model": None,           # preferred default model, or None = caller decides
        "backend": "claude",    # "claude" = bundled Claude Code CLI; "ollama" = local LLM
        "preferred_tier": None, # "opus" | "sonnet" | "haiku" | None
    }
    reg[name] = entry
    _save_registry(reg)
    return entry


def register_named_worker(
    name: str,
    session_id: str | None = None,
    cwd: str | None = None,
    model: str | None = None,
    backend: str = "claude",
    preferred_tier: str | None = None,
) -> dict:
    """Pre-seed a named worker in the registry.

    If `session_id` is provided, the worker is marked `session_created=True` so
    the very first dispatch uses `--resume` (for continuing an existing Claude
    Code session created elsewhere, e.g. Marl / CMMC Wizard).

    Idempotent: if an entry with this name already exists, missing fields are
    filled in but session_id is NEVER overwritten.
    """
    reg = _load_registry()
    existing = reg.get(name)
    if existing:
        # Config fields (cwd / model / backend / preferred_tier) are authoritative
        # from seed_named_workers — always update them on re-seed so path tweaks
        # actually propagate. session_id is NEVER overwritten (that would orphan
        # the live session transcript).
        for key, val in (
            ("cwd", cwd), ("model", model), ("backend", backend),
            ("preferred_tier", preferred_tier),
        ):
            if val is not None:
                existing[key] = val
        # Canonical flag distinguishes pre-seeded named agents from ad-hoc worker
        # entries (e.g. old "test-worker" / "raiken-ui" dispatches). UI uses this
        # to filter the agent panel.
        existing["is_canonical"] = True
        _save_registry(reg)
        return existing

    sid = session_id or str(uuid.uuid4())
    entry = {
        "name": name,
        "session_id": sid,
        "created": time.time(),
        "last_used": time.time(),
        "dispatches": 0,
        "session_created": bool(session_id),  # True = first dispatch uses --resume
        "cwd": cwd,
        "model": model,
        "backend": backend,
        "preferred_tier": preferred_tier,
        "is_canonical": True,
    }
    reg[name] = entry
    _save_registry(reg)
    return entry


def seed_named_workers():
    """Pre-populate the registry with Rook's canonical named agents.

    Called once at RCC startup. Idempotent — safe to call on every boot.
    Config fields (cwd, tier, etc.) are refreshed on every call so path tweaks
    propagate without requiring registry cleanup.
    """
    # Project-scoped agents — cwd narrowed to the specific project so their
    # attention stays on that codebase. They still get the global permission
    # preamble for cross-directory work when needed.
    register_named_worker(
        "Marl",
        session_id="d048154a-86eb-417f-946e-db7598bc8483",
        cwd=r"C:\Users\Rook\Documents\Claude\Projects\Royal Hearts",
        backend="claude",
        preferred_tier="opus",
    )
    register_named_worker(
        "CMMC Wizard",
        session_id="a448e86b-bf45-42e1-92ab-36d9bb7cd27e",
        cwd=r"C:\Users\Rook\Documents\Claude\Projects\APP CMMC Assessment",
        backend="claude",
        preferred_tier="opus",
    )
    # General-purpose agents — cwd widened to `C:\Users\Rook` so they can
    # traverse AI\, Documents\Claude\, and Downloads\ without self-restricting.
    # Combined with the permission preamble this is what unblocks Shadowling's
    # RCC-side edits to `C:\Users\Rook\AI\raiken\`.
    general_cwd = r"C:\Users\Rook"
    register_named_worker(
        "Shadowling Commander",
        cwd=general_cwd, backend="claude", preferred_tier="opus",
    )
    register_named_worker(
        "Oracle",
        cwd=general_cwd, backend="claude", preferred_tier="opus",
    )
    for sonnet_agent in ("Ledger", "Herald", "Scribe", "Cipher"):
        register_named_worker(
            sonnet_agent, cwd=general_cwd, backend="claude", preferred_tier="sonnet",
        )
    register_named_worker(
        "Keeper",
        cwd=general_cwd, backend="claude", preferred_tier="haiku",
    )
    # Raiken Agent — 11th canonical agent, Raiken's own problem-solver half.
    # Opus max-effort by design. RARELY used: Dispatcher only escalates to her
    # for genuinely hard problems or after consistent failures from other
    # agents. Not the default for routine heavy work — that's Shadowling
    # Commander's job. Raiken Agent runs expensive sub-agent validation on her
    # answers, so every dispatch costs several workers' worth of tokens.
    register_named_worker(
        "Raiken Agent",
        cwd=general_cwd, backend="claude", preferred_tier="opus",
    )
    # Pyre — local LLM via Ollama. Backend routing flips this to workers_ollama.
    register_named_worker(
        "Pyre",
        backend="ollama",
        model="qwen2.5:14b",
    )


def list_workers() -> list[dict]:
    """All named workers with metadata."""
    reg = _load_registry()
    return list(reg.values())


def delete_worker(name: str) -> bool:
    """Drop a worker from the registry. Does NOT delete the Claude Code session
    transcript (can be recovered by re-registering with the same name + session-id)."""
    reg = _load_registry()
    if name not in reg:
        return False
    del reg[name]
    _save_registry(reg)
    return True


# =============================================================================
# Dispatch helpers
# =============================================================================
def _try_emit_todo_event(worker_name: str, line: str, on_event) -> None:
    """Parse one stream-json line and call on_event when a TodoWrite tool use
    contains an in_progress item.  Silently no-ops on any parse error."""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return
    if obj.get("type") != "assistant":
        return
    msg = obj.get("message") or {}
    content = msg.get("content") or []
    if isinstance(content, str):
        return
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_use" or block.get("name") != "TodoWrite":
            continue
        todos = (block.get("input") or {}).get("todos") or []
        in_progress = [
            t for t in todos
            if isinstance(t, dict) and t.get("status") == "in_progress"
        ]
        if in_progress:
            summary = str(in_progress[0].get("content", "")).strip()
        elif todos and all(
            isinstance(t, dict) and t.get("status") == "completed" for t in todos
        ):
            summary = "wrapping up"
        else:
            return
        if summary:
            on_event({"type": "todo_update", "name": worker_name, "summary": summary})
        return  # one TodoWrite per assistant message is enough


def _parse_worker_output(raw: str) -> tuple[str, str | None]:
    """Parse --output-format json output. Returns (response_text, session_id).

    Falls back to treating raw as plain text if JSON parsing fails, so the
    function is safe regardless of CLI version.
    """
    if not raw:
        return raw, None
    # Single JSON document (normal case).
    try:
        data = json.loads(raw)
        return data.get("result", raw), data.get("session_id")
    except json.JSONDecodeError:
        pass
    # Last non-empty line (stream-json or multi-object output).
    for line in reversed(raw.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if "result" in data or "session_id" in data:
                return data.get("result", raw), data.get("session_id")
        except json.JSONDecodeError:
            continue
    return raw, None


def _reset_worker_session(name: str) -> str:
    """Assign a new random session UUID to a worker and clear session_created.

    Called when --resume fails with a stale session ID so the next dispatch
    starts a fresh session rather than retrying a dead UUID forever.
    Returns the new session_id.
    """
    new_sid = str(uuid.uuid4())
    reg = _load_registry()
    if name in reg:
        reg[name]["session_id"] = new_sid
        reg[name]["session_created"] = False
        _save_registry(reg)
    return new_sid


# =============================================================================
# Dispatch
# =============================================================================
async def run_worker(
    name: str,
    task: str,
    timeout: int = DEFAULT_TIMEOUT_SEC,
    model: str | None = None,
    _is_retry: bool = False,
    on_event: "callable | None" = None,
    callback_env: "dict | None" = None,
) -> dict:
    """Dispatch a task to the named worker. Blocks until it completes.

    If the worker's registry entry has backend="ollama", routes to the Ollama
    path (local LLM) instead of the Claude Code CLI.

    `model` arg overrides the registry's preferred model for this one dispatch
    (e.g. for tier-based routing decided by Raiken per-task).

    Returns:
        dict with keys:
          success: bool
          output: str (worker's final response text) if success
          error: str if failure
          elapsed: float (seconds)
          session_id: str (worker's CC session)
    """
    entry = get_or_create_worker(name)
    session_id = entry["session_id"]

    # Route to Ollama backend if configured.
    if entry.get("backend") == "ollama":
        try:
            from workers_ollama import run_worker_ollama
        except ImportError as e:
            return {
                "success": False,
                "error": f"ollama backend requested for '{name}' but workers_ollama not importable: {e}",
                "elapsed": 0.0,
                "session_id": session_id,
            }
        return await run_worker_ollama(name, task, entry, timeout=timeout, model_override=model)

    if not CLAUDE_BIN.exists():
        return {
            "success": False,
            "error": f"claude CLI not found at {CLAUDE_BIN}",
            "elapsed": 0.0,
            "session_id": session_id,
        }

    # First dispatch uses --session-id to CREATE the session with our chosen UUID.
    # All subsequent dispatches use --resume to continue it.
    session_created = entry.get("session_created", False)
    if session_created:
        session_flag = ["--resume", session_id]
    else:
        session_flag = ["--session-id", session_id]

    # Resolve model: per-call override > registry model > None (CLI default).
    effective_model = model or entry.get("model")
    model_flag = ["--model", effective_model] if effective_model else []

    # acceptEdits auto-approves file-edit tools within the worker's allowed
    # dirs — the spawned `claude -p` is headless (no tty, no UI to click the
    # permission dialog), so a normal-mode prompt would hang forever. Bash is
    # still gated by default; upgrade to `bypassPermissions` if a worker needs
    # unattended shell too.
    # --add-dir extends the worker's filesystem scope explicitly so Rook's core
    # directories are reachable regardless of cwd. Combined with the permission
    # preamble in `prefixed_task`, workers shouldn't self-refuse cross-directory
    # edits nor hit CC's system prompt.
    prefixed_task = WORKER_PERMISSION_PREAMBLE + task
    add_dir_flags: list[str] = []
    for extra in (
        r"C:\Users\Rook\AI",
        r"C:\Users\Rook\Documents\Claude",
        r"C:\Users\Rook\Downloads",
    ):
        add_dir_flags.extend(["--add-dir", extra])
    cmd = [
        str(CLAUDE_BIN),
        "-p",
        *session_flag,
        *model_flag,
        *add_dir_flags,
        "--permission-mode", "acceptEdits",
        # --output-format stream-json REQUIRES --verbose; the CLI rejects it
        # otherwise with "stream-json output format requires the verbose flag".
        # Verbose adds per-message metadata to stdout which the stream parser
        # in _drain_stdout happily ignores for non-TodoWrite lines.
        "--verbose",
        "--output-format", "stream-json",
        prefixed_task,
    ]

    # Use the worker's configured cwd (e.g. Marl runs in Royal Hearts project),
    # falling back to the RCC app dir if unset.
    worker_cwd = entry.get("cwd") or str(APP_DIR)

    t0 = time.time()
    collected_lines: list[str] = []

    try:
        # Raise the per-line StreamReader buffer limit. asyncio defaults to 64 KB
        # per line; `--output-format stream-json` routinely produces lines much
        # larger than that (a single tool result serialized inline can be 200 KB+),
        # which crashed dispatch with "separator is found, but chunk is longer
        # than limit". 16 MB is well past anything a realistic worker turn will
        # emit on one line.
        # Inject the orchestrator callback URL + token so the worker's
        # worker_tools.py helper can reach us. Missing env means worker_tools
        # bails out with a clear message rather than posting into the void.
        env = None
        if callback_env:
            import os as _os
            env = _os.environ.copy()
            env.update({k: v for k, v in callback_env.items() if v is not None})
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=worker_cwd,
            limit=16 * 1024 * 1024,
            env=env,
        )

        async def _drain_stdout():
            """Read stdout line-by-line, emitting TodoWrite events as they arrive.
            LimitOverrunError is caught in case a single JSON line ever exceeds
            the 16 MB buffer — we skip that line rather than crashing the whole
            dispatch; the worker's final result usually arrives on a later line."""
            while True:
                try:
                    raw_line = await proc.stdout.readline()
                except asyncio.LimitOverrunError as e:
                    # Drain the over-long line from the buffer so we can keep reading.
                    try:
                        await proc.stdout.readexactly(e.consumed)
                    except Exception:
                        pass
                    print(f"[dispatch] {name}: skipped oversized line ({e.consumed} bytes)", flush=True)
                    continue
                except Exception:
                    break
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line:
                    collected_lines.append(line)
                    if on_event is not None:
                        _try_emit_todo_event(name, line, on_event)

        try:
            results = await asyncio.wait_for(
                asyncio.gather(_drain_stdout(), proc.stderr.read()),
                timeout=timeout,
            )
            stderr = results[1]  # gather: [None from _drain_stdout, bytes from stderr.read]
            await proc.wait()
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {
                "success": False,
                "error": f"worker '{name}' timed out after {timeout}s",
                "elapsed": time.time() - t0,
                "session_id": session_id,
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"failed to spawn worker '{name}': {e}",
            "elapsed": time.time() - t0,
            "session_id": session_id,
        }

    elapsed = time.time() - t0

    if proc.returncode != 0:
        err_text = stderr.decode("utf-8", errors="replace")
        # Stale session: --resume referenced a UUID that no longer exists (e.g.
        # the session was compacted and Claude Code assigned it a new UUID). Reset
        # and retry once with a fresh --session-id so the worker name stays stable.
        if (
            not _is_retry
            and entry.get("session_created")
            and "No conversation found" in err_text
        ):
            _reset_worker_session(name)
            return await run_worker(
                name, task, timeout=timeout, model=model,
                _is_retry=True, on_event=on_event, callback_env=callback_env,
            )
        return {
            "success": False,
            "error": f"worker '{name}' exited with code {proc.returncode}: "
                     f"{err_text[:600]}",
            "elapsed": elapsed,
            "session_id": session_id,
        }

    raw_out = "\n".join(collected_lines).strip()
    output_text, returned_sid = _parse_worker_output(raw_out)

    # Bump dispatch counter and persist the real session ID returned by the CLI.
    # After a compaction Claude Code issues a new UUID — capturing it here ensures
    # the next --resume points at the live session rather than the compacted one.
    reg = _load_registry()
    if name in reg:
        reg[name]["dispatches"] = reg[name].get("dispatches", 0) + 1
        reg[name]["last_used"] = time.time()
        reg[name]["session_created"] = True
        if returned_sid:
            reg[name]["session_id"] = returned_sid
        _save_registry(reg)

    return {
        "success": True,
        "output": output_text,
        "elapsed": elapsed,
        "session_id": returned_sid or session_id,
    }
