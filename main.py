"""
Raiken standalone app. Voice + text input, orchestrator via Claude Agent SDK,
voice out via local XTTS server, tkinter UI on main thread, asyncio core in a
worker thread.

Requires:
  - TTS server will be auto-launched (C:\\Users\\Rook\\AI\\xtts\\start_tts_server.bat)
  - `claude login` already done so Max auth works
  - F2 for push-to-talk (configurable below)

Run:  .venv\\Scripts\\python.exe main.py
Stop: tray icon -> Quit, or Ctrl+C
"""
import asyncio
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

# Under pythonw.exe (no console) sys.stdout is None. Redirect prints to a log file
# so the existing print() calls don't crash when running hidden.
_APP_DIR = Path(__file__).parent
_LOG_DIR = _APP_DIR / "logs"
_LOG_DIR.mkdir(exist_ok=True)
if sys.stdout is None or sys.stderr is None:
    _log_fp = open(_LOG_DIR / "raiken.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = _log_fp
    sys.stderr = _log_fp
elif hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Windows: register every NVIDIA CUDA DLL directory so faster-whisper's ctranslate2
# backend can load cublas / cudnn / cudart / cuda_nvrtc / etc. The nvidia-*-cu12 pip
# packages drop DLLs under .venv/Lib/site-packages/nvidia/<name>/bin/ but those paths
# are NOT on the Windows DLL search path. Must run BEFORE importing faster_whisper.
# BOTH os.add_dll_directory and PATH env var are required — ctranslate2's native loader
# falls back to classic PATH lookup and ignores the Python-3.8+ add_dll_directory API.
if os.name == "nt":
    _venv_root = Path(sys.executable).parent.parent
    _nvidia_root = _venv_root / "Lib" / "site-packages" / "nvidia"
    if _nvidia_root.is_dir():
        _nvidia_paths = []
        for _pkg_dir in _nvidia_root.iterdir():
            _bin = _pkg_dir / "bin"
            if _bin.is_dir():
                os.add_dll_directory(str(_bin))
                _nvidia_paths.append(str(_bin))
        if _nvidia_paths:
            os.environ["PATH"] = os.pathsep.join(_nvidia_paths) + os.pathsep + os.environ.get("PATH", "")

    # Declare PerMonitorV2 DPI awareness before any window is created so Windows
    # does not bitmap-scale the process. Without this pythonw.exe defaults to
    # DPI-unaware and the OS blurs text on high-DPI / 4K displays.
    import ctypes as _ctypes
    try:
        _ctypes.windll.shcore.SetProcessDpiAwareness(2)   # PROCESS_PER_MONITOR_DPI_AWARE_V2
    except (AttributeError, OSError):
        try:
            _ctypes.windll.user32.SetProcessDPIAware()    # fallback: system DPI aware
        except (AttributeError, OSError):
            pass

import httpx
import keyboard
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
    ResultMessage,
    RateLimitEvent,
    tool,
    create_sdk_mcp_server,
)

from workers import run_worker, list_workers, seed_named_workers, get_or_create_worker
from ui import RaikenWindow, RaikenTray, ChatEvent, StatusEvent, WorkerDoneEvent, DispatchBadgeEvent, PresenceEvent, WorkerStatusEvent
from bitwarden import BitwardenSession

import pyperclip

# =============================================================================
# Config
# =============================================================================
HOTKEY = "f2"
SAMPLE_RATE = 16000
WHISPER_MODEL = "large-v3"
TTS_URL = "http://127.0.0.1:7851/speak"
TTS_HEALTH_URL = "http://127.0.0.1:7851/health"
TTS_LAUNCHER_BAT = r"C:\Users\Rook\AI\xtts\start_tts_server.bat"
TTS_STARTUP_TIMEOUT_SEC = 90
MIN_AUDIO_SEC = 0.3

SYSTEM_PROMPT = """You are Raiken, Rook's AI right-hand man. Your responses are read aloud
via text-to-speech AND shown in a chat window, so:
- Plain text only. No markdown, no emoji, no special characters.
- Keep sentences short and clear — each sentence becomes a TTS chunk.
- Be direct. Skip preamble. Get to the point.
- ALWAYS acknowledge first when you expect a long pause. If Rook asks for
  something that requires research, multi-step tool calls, file reads, worker
  dispatch, or any work that will take more than a few seconds before you can
  speak, say a short acknowledgment FIRST so he hears you heard him. Something
  like "on it", "looking now", "checking", or "one sec, digging in". Then do
  the work. Never leave Rook in silence wondering if you caught the command.
- Speak in the first person about yourself. Say "me", "my config", "my memory",
  "I can check", etc. Do NOT say "the Raiken config", "Raiken can check",
  or refer to yourself in the third person. You ARE Raiken.
- DON'T use numbered lists in your speech. No "1. 2. 3." format. If you're
  enumerating, say "first, second, third" inline as a normal sentence.
- When you produce a DELIVERABLE that's meant to be copy-pasted — a filename, a
  Nano Banana prompt, a command, a code snippet, a URL Rook will use — wrap it
  in triple backticks. Code blocks are silent in TTS but visible in the chat
  window, so Rook can read and copy them without hearing them read letter-by-
  letter. Announce what you're delivering in a short narration sentence BEFORE
  the block, e.g. "Here's the Nano Banana prompt:" then the triple-backtick block.

TTS readability rules (the voice pipeline is imperfect; write around its quirks):
- Put ALL technical identifiers in code blocks, not prose. That means file paths,
  class names, method names, variable names, snake_case, camelCase, CLI flags,
  error strings, and URLs. Never speak them. Narrate the behavior in plain
  English and drop the specifics into a silent code block for Rook to read.
- Don't use Title Case for generic concepts in speech. Phrases like "Action Log"
  or "Work Log" get vocalized with a hard stop between words ("action period log
  period"). Either lowercase them in a natural sentence ("the action log") or
  fold them into a surrounding phrase ("a log of what I did").
- Don't end sentences on a short, terse word right before the period — the TTS
  sometimes vocalizes the period as "dot". Instead of "both files compile clean"
  say "both files compiled without errors". Favor fuller sentence endings.
- Spell out abbreviations in prose. Write "five hour" not "5hr", "percent" not
  the % sign, "config" not "cfg", "repo" not "repo.", etc. Numerals are fine
  for normal counting; it's abbreviated units and acronyms to watch.
- Avoid special characters in narration. No underscores, dots, slashes, dashes,
  parentheses, quotes around keywords, or inline code style backticks. If it
  needs any of those, it belongs in a code block.

Barge-in handling:
- If a user message arrives with a leading tag like "[INTERRUPT — your previous
  response was cut off]", that means Rook pressed F2 while you were still speaking
  the prior response. His earlier partial turn(s) may or may not be relevant.
- Use judgment: if the earlier exchange was trivial ("what about now", "still there?"),
  ignore it and focus on the CURRENT message. If the earlier exchange had substance
  Rook's building on, weave it together and respond to the whole thread.
- Don't apologize for being cut off or explain the barge-in. Just answer what
  Rook needs NOW.

Worker dispatch — YOU DO NOT HAVE DISPATCH TOOLS.

A parallel SDK session called the Foreman handles ALL worker dispatch silently.
Every message Rook sends goes to both of you at the same time. You respond
conversationally; she evaluates the same message and dispatches the appropriate
worker in parallel. You will never see her, she never speaks to Rook. You only
see the RESULTS of her dispatches via [worker-updates] preambles on later turns.

How this changes your behavior:
- When Rook asks for work, acknowledge briefly ("on it", "looking now", "sending
  someone on it") — Foreman will have already started dispatching by the time
  you finish that sentence. Your acknowledgment is courtesy, not a trigger.
- Do NOT try to dispatch yourself. You have no dispatch tool. Attempting it
  does nothing.
- Do NOT say "dispatching Oracle" as if YOU are deciding the agent — you don't
  know which agent Foreman picked. Say generic acknowledgments ("sending
  someone", "on it", "worker's heading out"). If Rook asks which agent,
  truthfully say you'll know when the return shows up.
- On your NEXT turn, any workers that completed appear in a [worker-updates]
  preamble. Narrate the completion: "Oracle's back — headline is X." Don't
  repeat the full output (it's in the Log tab); summarize.
- If your turn fires with the text starting "[WORKER-RETURN]", it's an auto-wake
  — Rook did NOT speak. Narrate ONE or TWO short declarative sentences about
  what returned. Don't ask a question. End the turn.
- DON'T NARRATE INTERNAL ORCHESTRATION NOISE. Dispatch failures, retries, stale
  sessions — Foreman handles those silently. You only speak about RESULTS Rook
  asked for, or genuine decisions he needs to make.

What YOU still do yourself:
- Conversational replies (questions, preferences, yes/no)
- Vault operations (unlock, search, copy credentials — you have those tools)
- Narrating worker returns
- Asking Rook genuine clarifying questions when needed

What agents Foreman has at her disposal — so you can narrate returns fluently:
    Marl — Royal Hearts work (Opus)
    CMMC Wizard — CMMC compliance (Opus)
    Shadowling Commander — general heavy work (Opus)
    Oracle — research, web summarization (Opus)
    Ledger — finance, debt, budget (Sonnet)
    Herald — email, Discord, messaging triage (Sonnet)
    Scribe — writing, docs, copy (Sonnet)
    Cipher — security audits, vault admin (Sonnet)
    Keeper — memory file upkeep (Haiku)
    Pyre — devil's-advocate critic, runs on local Qwen

Bitwarden vault:
- You have vault_status, vault_unlock, vault_search, vault_copy_password,
  vault_copy_username, vault_copy_totp, vault_lock.
- You NEVER see passwords. Fetch tools copy to clipboard only; you get an "OK"
  outcome. This is the design — don't ask Rook to paste the password back at you.
- If Rook asks for a credential and the vault is locked, call vault_unlock. That
  pops a native dialog in the Raiken window for him to type his master password.
  His master password does NOT pass through you.
- Use vault_search first when you're unsure of the exact item name; show Rook
  the matches and ask which one he wants.
- For TOTP codes, remember they're time-based — tell Rook they expire in ~30s.

Memory files live in C:\\Users\\Rook\\Documents\\Claude\\Projects\\Royal Hearts\\docs\\memory
— read MANIFEST.md or CLAUDE_CONVENTIONS.md when you need durable project context.
Also C:\\Users\\Rook\\.claude\\projects\\C--Users-Rook-Documents-Claude-Projects-Royal-Hearts\\memory\\MEMORY.md
for user-scoped durable facts.
"""

FOREMAN_SYSTEM_PROMPT = """You are the Foreman — a silent dispatcher running in parallel with Raiken
inside Raiken Command Center. You share every user message with Raiken, but
your role is entirely separate: she converses with Rook, you dispatch workers.

YOUR RULES:

1. NEVER produce text output aimed at Rook. No greetings, explanations, or
   narration. Your stream is discarded. Only your TOOL CALLS matter.

2. Decide per message whether a worker dispatch is warranted:
   - YES: any request to DO something — edit code, fix a bug, investigate,
          build, research, audit, analyze, refactor, summarize emails, pull
          finance data, ship a feature, debug RCC itself, etc.
   - NO:  conversational chat, questions about Raiken's state, yes/no answers,
          preferences, "hi", vault operations (Raiken owns those), or generic
          small talk. When in doubt: do nothing.

3. Prefer canonical named agents. Each has a stable persistent session and a
   defined role:
     Marl                 — Royal Hearts project (Opus)
     CMMC Wizard          — CMMC compliance project (Opus)
     Shadowling Commander — general heavy work, RCC internals, code (Opus)
     Oracle               — research, web summarization (Opus)
     Ledger               — finance, debt, budget, bank CSVs (Sonnet)
     Herald               — email, Discord, messaging triage (Sonnet)
     Scribe               — writing, docs, long-form copy (Sonnet)
     Cipher               — security audits, vault admin, install/PATH (Sonnet)
     Keeper               — memory file upkeep, small maintenance (Haiku)
     Pyre                 — devil's-advocate critic (local Qwen via Ollama)
   NEVER invent a new worker name if a canonical agent fits. Only invent when
   the task genuinely doesn't match any canonical role (rare).

4. Write the task message clearly with full context — the worker only sees what
   you send. Include file paths, goals, constraints, relevant prior attempts.
   Assume the worker has no access to this conversation or Rook's verbal tone.

5. If a [WORKER-RETURN] tag arrives, that's an auto-wake — Rook didn't speak.
   Usually: do nothing. Raiken narrates to Rook. Only dispatch a follow-up if
   a worker explicitly needs to be respawned (e.g. a clarifying answer needed).

6. If [INTERRUPT] tag appears (Rook barged-in), evaluate the current message
   alone; don't try to reconcile the cut-off prior thread.

7. Do not ask Rook clarifying questions — that's Raiken's job. If a message is
   ambiguous, make your best-guess dispatch or don't dispatch.

8. FAILURE MEANS RETRY, NOT REPORT. If a dispatch fails (stale session, worker
   crashed, timeout), silently redispatch — same agent first, fall back to
   another canonical agent only if the same name keeps failing. Don't narrate
   failures to Rook — Raiken handles what he needs to hear.

You have exactly two tools: `dispatch_worker(name, task)` and `list_workers()`.
Use them. Stay silent otherwise.
"""


PROJECT_VOCAB = (
    "Raiken, Claude, Claude Code, Claude Agent SDK, Claude Max, Anthropic, "
    "Nakama, Godot, Royal Hearts, Kratos, Damien Black, Pyre, Rook, Bitwarden, "
    "XTTS, TTS, Whisper, Ollama, Qwen, Sonnet, Opus."
)
STT_REPLACEMENTS = [
    (r"\bCloud\s+code\b", "Claude code"),
    (r"\bCloud\s+desktop\b", "Claude desktop"),
    (r"\bCloud\s+chat\b", "Claude chat"),
    (r"\bCloud\s+Agent\b", "Claude Agent"),
    (r"\bCloud\s+SDK\b", "Claude SDK"),
    (r"\bCloud\s+Max\b", "Claude Max"),
    (r"\bCloud\s+CLI\b", "Claude CLI"),
    (r"\bCloud\s+session\b", "Claude session"),
    (r"\bCloud's\b", "Claude's"),
    (r"\bRyken\b", "Raiken"),
    (r"\bRycken\b", "Raiken"),
    (r"\bRikan\b", "Raiken"),
    (r"\bRikken\b", "Raiken"),
    (r"\bRakin\b", "Raiken"),
    (r"\bRaken\b", "Raiken"),
    (r"\bRikin\b", "Raiken"),
    (r"\bRyeken\b", "Raiken"),
    (r"\bRye-ken\b", "Raiken"),
    (r"\bRye ken\b", "Raiken"),
]

END_OF_TURN = object()


# =============================================================================
# Tier-based model routing (Raiken's autonomous Opus/Sonnet/Haiku pick)
# =============================================================================
# Thresholds from agent_architecture.md. Rook gave Raiken discretion to override
# these when a task justifies it — see raiken_persona.md / agent_architecture.md.
_OPUS_BUDGET_CEIL_PCT = 35.0   # weekly usage below → opus budget-allowed
_SONNET_BUDGET_CEIL_PCT = 70.0 # 35-70% → sonnet; >70% → haiku
_TIER_RANK = {"opus": 3, "sonnet": 2, "haiku": 1}


def _weekly_utilization_pct(rate_limits: dict) -> float | None:
    """Extract the weekly rate-limit utilization as 0-100%, or None if unknown."""
    for rl_type, info in rate_limits.items():
        t = (rl_type or "").lower()
        if "week" in t or "day" in t or "seven" in t:
            util = info.get("utilization")
            if isinstance(util, (int, float)):
                return float(util) * 100 if util <= 1 else float(util)
    return None


def _pick_model_tier(preferred: str | None, weekly_pct: float | None) -> str:
    """Pick model tier. Budget overrides a named agent's preferred tier (per Rook)."""
    if weekly_pct is None:
        # No signal — trust the named-agent preferred tier, default sonnet.
        return preferred or "sonnet"
    if weekly_pct > _SONNET_BUDGET_CEIL_PCT:
        budget_ceiling = "haiku"
    elif weekly_pct > _OPUS_BUDGET_CEIL_PCT:
        budget_ceiling = "sonnet"
    else:
        budget_ceiling = "opus"
    pref = preferred or budget_ceiling
    # If preferred exceeds budget ceiling, budget wins.
    if _TIER_RANK.get(pref, 2) > _TIER_RANK[budget_ceiling]:
        return budget_ceiling
    return pref


# =============================================================================
# SDK tool: dispatch_worker
# =============================================================================
@tool(
    "dispatch_worker",
    "Dispatch a heavy task to a named background worker (a persistent Claude Code "
    "session). Use for code, research, file edits, deep analysis — not for quick "
    "conversational replies. Calling the same worker name continues its conversation.",
    {"name": str, "task": str},
)
async def dispatch_worker_tool(args):
    name = args.get("name", "").strip() or "default"
    task = args.get("task", "").strip()
    if not task:
        return {"content": [{"type": "text", "text": "error: empty task"}]}

    # Tier routing: pick a model before dispatch.
    entry = get_or_create_worker(name)
    is_ollama = entry.get("backend") == "ollama"
    if is_ollama:
        tier = None
        tier_label = entry.get("model") or "local-llm"
        weekly_pct = None
    else:
        rl = _APP_REF.rate_limits_snapshot() if _APP_REF is not None else {}
        weekly_pct = _weekly_utilization_pct(rl)
        tier = _pick_model_tier(entry.get("preferred_tier"), weekly_pct)
        tier_label = tier.capitalize()

    # Announce the dispatch as an orange bordered badge in chat.
    if _APP_REF is not None:
        badge_tier = tier_label
        if weekly_pct is not None:
            badge_tier = f"{tier_label}  {weekly_pct:.0f}% weekly"
        _APP_REF.raiken._emit_dispatch_badge(name=name, tier_label=badge_tier)

    print(f"[dispatch] worker='{name}' tier={tier} task={task[:80]!r}", flush=True)

    # Mark active immediately so the panel lights up the moment Raiken dispatches.
    if _APP_REF is not None:
        _APP_REF.register_active_worker(name, task)

    # Fire-and-forget. Raiken's turn is NOT blocked by the worker — he returns
    # from this tool call immediately and can keep chatting, dispatch more
    # workers, handle barge-ins. The background task emits the worker's output
    # as a labeled chat bubble and queues the completion into
    # _pending_worker_results so Raiken narrates it on his next turn.
    # Per-name lock serializes concurrent dispatches to the SAME worker (two
    # `claude --resume <same-id>` processes would race the session transcript).

    def _on_worker_event(event_dict: dict):
        """Route intermediate worker events (TodoWrite) to the UI panel."""
        if event_dict.get("type") == "todo_update" and _APP_REF is not None:
            _APP_REF.raiken.ui_event_queue.put(
                WorkerStatusEvent(name=name, summary=event_dict["summary"])
            )

    async def _background_run():
        try:
            lock = _APP_REF._get_worker_lock(name) if _APP_REF is not None else None
            if lock is not None:
                async with lock:
                    result = await run_worker(name, task, model=tier, on_event=_on_worker_event)
            else:
                result = await run_worker(name, task, model=tier, on_event=_on_worker_event)
        except Exception as e:
            result = {
                "success": False,
                "error": f"dispatch crashed: {type(e).__name__}: {e}",
                "elapsed": 0.0,
                "session_id": "",
            }
        finally:
            if _APP_REF is not None:
                _APP_REF.unregister_active_worker(name)

        if _APP_REF is None:
            return
        if result.get("success"):
            print(f"[dispatch] {name} done ({result.get('elapsed', 0):.1f}s)", flush=True)
        else:
            print(f"[dispatch] {name} FAILED: {result.get('error')}", flush=True)

        # Store transcript + surface as a clickable badge (chat) and a full
        # sectioned entry in the Log tab. No raw-dump into the main chat.
        origin_at_dispatch = _APP_REF._current_origin if _APP_REF else "local"
        run_id = _APP_REF.store_worker_result(name, task, result, origin=origin_at_dispatch)
        _APP_REF.raiken._emit_worker_done(
            name=name,
            success=bool(result.get("success")),
            elapsed=float(result.get("elapsed", 0.0) or 0.0),
            run_id=run_id,
        )
        _APP_REF.record_worker_result(name, result)

        # Route the return through the notification router — it decides which
        # external channels to hit (phone push, Discord reply, etc.) based on
        # origin, presence, idle duration, and task duration. The TTS/chat
        # side is still driven by the auto-wake below.
        try:
            _APP_REF.route_worker_return_notification(
                worker_name=name,
                success=bool(result.get("success")),
                elapsed=float(result.get("elapsed", 0.0) or 0.0),
                run_id=run_id,
            )
        except Exception as ex:
            print(f"[notify] router raised: {ex}", flush=True)

        # Auto-fire a worker-return narration turn over TTS. We skip it only
        # when Rook is clearly away (15+ min idle) — at that point the phone
        # push is what will reach him; TTS to an empty room is waste. If he's
        # active OR idle (nearby), narrate.
        #
        # PTT gate: if Rook is currently holding push-to-talk, don't fire now.
        # The result is already in _pending_worker_results, so it will surface
        # either in his [worker-updates] preamble when he speaks, or via
        # _flush_deferred_wake() the moment he releases PTT without speaking.
        if not _APP_REF.raiken.turn_in_progress:
            presence = _APP_REF.get_effective_presence()
            if presence != "away":
                if _APP_REF.raiken.recording:
                    print(f"[worker-return] PTT held — deferring auto-wake for '{name}'", flush=True)
                else:
                    loop = _APP_REF.raiken.loop
                    if loop is not None:
                        try:
                            asyncio.run_coroutine_threadsafe(
                                _APP_REF.raiken.submit("[WORKER-RETURN]"),
                                loop,
                            )
                        except Exception as ex:
                            print(f"[worker-return] auto-fire failed: {ex}", flush=True)

    asyncio.create_task(_background_run())

    return {
        "content": [{
            "type": "text",
            "text": (
                f"Dispatched '{name}' ({tier_label}). Running in background; your "
                f"turn is free. The result will appear in the chat UI the moment "
                f"it returns, and a [worker-updates] preamble will surface on your "
                f"next turn so you can narrate the completion to Rook."
            ),
        }]
    }


@tool(
    "list_workers",
    "List all named background workers that have been used this installation, with "
    "their dispatch counts and last-used timestamps.",
    {},
)
async def list_workers_tool(args):
    workers = list_workers()
    if not workers:
        return {"content": [{"type": "text", "text": "No workers registered yet."}]}
    lines = [
        f"- {w['name']}: {w.get('dispatches', 0)} dispatches, session={w.get('session_id', '?')[:8]}"
        for w in workers
    ]
    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


WORKER_MCP_SERVER = create_sdk_mcp_server(
    name="raiken-workers",
    version="0.1.0",
    tools=[dispatch_worker_tool, list_workers_tool],
)


# =============================================================================
# SDK tools: Bitwarden vault (credentials stay local; model sees outcomes only)
# =============================================================================
_BW_SESSION = BitwardenSession()
_APP_REF = None  # type: RaikenApp | None  — set by RaikenApp.__init__


@tool(
    "vault_status",
    "Return the Bitwarden vault status. Non-secret info only: state is one of "
    "'unauthenticated', 'locked', 'unlocked', or 'cli-missing'. Use this before "
    "attempting vault operations.",
    {},
)
async def vault_status_tool(args):
    status = _BW_SESSION.status()
    lines = [f"state: {status.get('state')}"]
    if status.get("email"):
        lines.append(f"account: {status['email']}")
    if status.get("session_timeout_remaining") is not None:
        lines.append(f"idle timeout remaining: {status['session_timeout_remaining']}s")
    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


@tool(
    "vault_unlock",
    "Show the master-password dialog in the Raiken UI so Rook can unlock the vault. "
    "The password is entered natively and never passes through you. Returns 'unlocked', "
    "'failed', or 'cancelled'.",
    {},
)
async def vault_unlock_tool(args):
    if _BW_SESSION.is_unlocked():
        return {"content": [{"type": "text", "text": "vault already unlocked"}]}
    if _APP_REF is None or _APP_REF.window is None:
        return {"content": [{"type": "text", "text": "UI not available for password prompt"}]}

    pw = _APP_REF.window.request_password_from_asyncio(
        "Bitwarden — Unlock Vault",
        "Master password:",
        timeout_sec=120,
    )
    if pw is None or pw == "":
        return {"content": [{"type": "text", "text": "cancelled"}]}

    ok, msg = _BW_SESSION.unlock_with_password(pw)
    pw = None  # wipe reference
    return {"content": [{"type": "text", "text": msg}]}


@tool(
    "vault_search",
    "Search the Bitwarden vault by item name or URL. Returns a list of matching items "
    "with NON-SECRET metadata only: id, name, username (if present), uri, and whether "
    "the item has TOTP. Never returns passwords.",
    {"query": str},
)
async def vault_search_tool(args):
    query = args.get("query", "").strip()
    if not query:
        return {"content": [{"type": "text", "text": "error: empty query"}]}
    if not _BW_SESSION.is_unlocked():
        return {"content": [{"type": "text", "text": "vault is locked — call vault_unlock first"}]}
    items = _BW_SESSION.search(query)
    if not items:
        return {"content": [{"type": "text", "text": f"no items matching {query!r}"}]}
    lines = [f"Found {len(items)} matching {query!r}:"]
    for i in items:
        parts = [f"- {i['name']}"]
        if i.get("username"):
            parts.append(f"user={i['username']}")
        if i.get("uri"):
            parts.append(f"uri={i['uri']}")
        if i.get("has_totp"):
            parts.append("[TOTP]")
        lines.append(" ".join(parts))
    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


def _copy_to_clipboard(value: str) -> tuple[bool, str]:
    try:
        pyperclip.copy(value)
        return True, "copied"
    except Exception as e:
        return False, f"clipboard error: {e}"


@tool(
    "vault_copy_password",
    "Fetch the password for a vault item and copy it to Rook's clipboard. You do NOT "
    "receive the password value. Returns 'copied' on success or a failure reason.",
    {"item": str},
)
async def vault_copy_password_tool(args):
    item = args.get("item", "").strip()
    if not item:
        return {"content": [{"type": "text", "text": "error: empty item name"}]}
    if not _BW_SESSION.is_unlocked():
        return {"content": [{"type": "text", "text": "vault is locked"}]}
    pw = _BW_SESSION.get_password(item)
    if pw is None:
        return {"content": [{"type": "text", "text": f"item not found or no password: {item!r}"}]}
    ok, msg = _copy_to_clipboard(pw)
    pw = None  # wipe
    return {"content": [{"type": "text", "text": f"password for {item!r}: {msg}"}]}


@tool(
    "vault_copy_username",
    "Fetch the username for a vault item and copy it to Rook's clipboard.",
    {"item": str},
)
async def vault_copy_username_tool(args):
    item = args.get("item", "").strip()
    if not item:
        return {"content": [{"type": "text", "text": "error: empty item name"}]}
    if not _BW_SESSION.is_unlocked():
        return {"content": [{"type": "text", "text": "vault is locked"}]}
    u = _BW_SESSION.get_username(item)
    if u is None:
        return {"content": [{"type": "text", "text": f"no username for {item!r}"}]}
    ok, msg = _copy_to_clipboard(u)
    return {"content": [{"type": "text", "text": f"username for {item!r}: {msg}"}]}


@tool(
    "vault_copy_totp",
    "Fetch the current TOTP (2FA) code for a vault item and copy it to Rook's clipboard.",
    {"item": str},
)
async def vault_copy_totp_tool(args):
    item = args.get("item", "").strip()
    if not item:
        return {"content": [{"type": "text", "text": "error: empty item"}]}
    if not _BW_SESSION.is_unlocked():
        return {"content": [{"type": "text", "text": "vault is locked"}]}
    code = _BW_SESSION.get_totp(item)
    if not code:
        return {"content": [{"type": "text", "text": f"no TOTP for {item!r}"}]}
    ok, msg = _copy_to_clipboard(code)
    return {"content": [{"type": "text", "text": f"TOTP for {item!r}: {msg}"}]}


@tool(
    "vault_lock",
    "Lock the Bitwarden vault immediately. Rook will need to re-enter master password "
    "to unlock again.",
    {},
)
async def vault_lock_tool(args):
    _BW_SESSION.lock()
    return {"content": [{"type": "text", "text": "vault locked"}]}


VAULT_MCP_SERVER = create_sdk_mcp_server(
    name="raiken-vault",
    version="0.1.0",
    tools=[
        vault_status_tool,
        vault_unlock_tool,
        vault_search_tool,
        vault_copy_password_tool,
        vault_copy_username_tool,
        vault_copy_totp_tool,
        vault_lock_tool,
    ],
)


# =============================================================================
# Raiken core (runs on asyncio worker thread)
# =============================================================================
class Raiken:
    def __init__(self, ui_event_queue: queue.Queue):
        self.whisper: WhisperModel | None = None
        self.client: ClaudeSDKClient | None = None
        # Foreman — silent dispatcher, runs in parallel with Raiken. See
        # FOREMAN_SYSTEM_PROMPT + Raiken.run() for wiring. None until run().
        self.foreman_client: ClaudeSDKClient | None = None
        # Serialize Foreman turns so we don't ask the SDK to interleave queries
        # on the same client; each broadcast waits for the previous to finish.
        self._foreman_lock: asyncio.Lock | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.ui_event_queue = ui_event_queue

        self.recording = False
        self.audio_buffer: list[np.ndarray] = []
        self.buffer_lock = threading.Lock()
        self.turn_in_progress = False

        # Single-slot queue for a message submitted while a turn is still
        # winding down (typically after a barge-in). Drained by submit()'s
        # finally block so the user never silently loses a message.
        self._pending_submit: str | None = None

        self.sentence_q: queue.Queue = queue.Queue()
        self.wav_q: queue.Queue = queue.Queue()

        # Barge-in machinery
        self._current_ffplay: subprocess.Popen | None = None
        self._current_ffplay_lock = threading.Lock()
        self._barge_flag = False
        self._just_barged = False  # set True when barge happens; consumed by next submit

        # Usage tracking (populated during turns; UI polls).
        self._context_usage: dict | None = None   # {total, max, pct, model}
        self._rate_limits: dict = {}              # type -> {utilization, resets_at, status}
        self._usage_lock = threading.Lock()

    # --- UI emit helpers ------------------------------------------------------
    def _emit_chat(self, role: str, text: str, append: bool = False, done: bool = False):
        self.ui_event_queue.put(ChatEvent(role=role, text=text, append=append, done=done))

    def _emit_dispatch_badge(self, name: str, tier_label: str):
        """Surface a worker dispatch as an orange bordered badge in chat."""
        self.ui_event_queue.put(DispatchBadgeEvent(name=name, tier_label=tier_label))

    def _emit_worker_done(self, name: str, success: bool, elapsed: float, run_id: int):
        """Surface a completed worker as a clickable badge in chat + detail entry in Log."""
        self.ui_event_queue.put(WorkerDoneEvent(
            name=name, success=success, elapsed=elapsed, run_id=run_id,
        ))

    def _emit_presence(self, state: str, detail: str = ""):
        self.ui_event_queue.put(PresenceEvent(state=state, detail=detail))

    def _emit_status(self, component: str, state: str, detail: str = ""):
        self.ui_event_queue.put(StatusEvent(component=component, state=state, detail=detail))

    # --- Audio + hotkey -------------------------------------------------------
    def _audio_callback(self, indata, frames, time_info, status):
        if self.recording:
            with self.buffer_lock:
                self.audio_buffer.append(indata.copy())

    def _on_press(self, _e):
        if self.recording:
            return
        # Barge-in: if Raiken is currently speaking (or has pending speech), kill it
        # and flush the pipeline so the user can talk immediately.
        speaking = (
            self._current_ffplay is not None
            or not self.wav_q.empty()
            or not self.sentence_q.empty()
        )
        if speaking:
            self._barge_in()
        elif self.turn_in_progress:
            # Turn is generating but no audio queued yet — still barge to abort the SDK stream.
            self._barge_in()
        with self.buffer_lock:
            self.audio_buffer.clear()
        self.recording = True
        self._emit_status("ptt", "busy", "recording")
        print("[recording...]", flush=True)

    def _barge_in(self):
        """Kill active TTS playback, drain the pipeline, tell the SDK to abort."""
        self._barge_flag = True
        # Kill the currently-playing ffplay process.
        with self._current_ffplay_lock:
            if self._current_ffplay is not None and self._current_ffplay.poll() is None:
                try:
                    self._current_ffplay.kill()
                except Exception:
                    pass
        # Drain both pipeline queues.
        for q in (self.sentence_q, self.wav_q):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        # NOTE: we intentionally do NOT call client.interrupt() here — it was
        # putting the SDK into a state where the NEXT query returned no content.
        # The abandoned turn keeps generating silently (small quota cost) but the
        # receive_response loop breaks out as soon as its next iteration sees
        # barge_flag, so turn_in_progress clears and the new turn proceeds cleanly.
        self._just_barged = True  # signals next submit() to tag the message for Claude
        self._emit_chat("system", "(barge-in)")
        print("[barge-in]", flush=True)

    def _flush_deferred_wake(self):
        """Fire a deferred [WORKER-RETURN] auto-wake when PTT releases with no speech.

        Called only from _on_release when no usable audio was captured, so the
        normal [worker-updates] preamble path won't run. If there are pending
        worker results and Rook is reachable, inject the auto-wake now.
        """
        if _APP_REF is None or self.loop is None or self.turn_in_progress:
            return
        with _APP_REF._pending_worker_results_lock:
            has_pending = bool(_APP_REF._pending_worker_results)
        if not has_pending:
            return
        presence = _APP_REF.get_effective_presence()
        if presence == "away":
            return
        print("[worker-return] PTT released (no speech) — flushing deferred wake", flush=True)
        try:
            asyncio.run_coroutine_threadsafe(self.submit("[WORKER-RETURN]"), self.loop)
        except Exception as ex:
            print(f"[worker-return] deferred flush failed: {ex}", flush=True)

    def _on_release(self, _e):
        if not self.recording:
            return
        self.recording = False
        self._emit_status("ptt", "up")
        with self.buffer_lock:
            if not self.audio_buffer:
                # No audio at all — maybe a mis-tap. Still flush any deferred wakes.
                self._flush_deferred_wake()
                return
            audio = np.concatenate(self.audio_buffer).flatten().astype(np.float32)
            self.audio_buffer.clear()
        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_AUDIO_SEC:
            print(f"[ignored, {duration:.2f}s too short]", flush=True)
            # Audio too short to be speech — flush deferred wakes now.
            self._flush_deferred_wake()
            return
        # Real speech captured: voice_turn → submit() → _execute_turn() will prepend
        # any pending worker results via [worker-updates] preamble. Don't also fire
        # the auto-wake or the results would surface twice.
        asyncio.run_coroutine_threadsafe(self.voice_turn(audio), self.loop)

    # --- TTS lifecycle --------------------------------------------------------
    def _tts_up(self) -> bool:
        try:
            r = httpx.get(TTS_HEALTH_URL, timeout=2)
            r.raise_for_status()
            return True
        except Exception:
            return False

    def _launch_tts_if_needed(self) -> bool:
        if self._tts_up():
            print("[raiken] TTS server already running", flush=True)
            self._emit_status("tts", "up")
            return False

        # Invoke the xtts venv's python directly with server.py — no bat, no console.
        xtts_dir = Path(r"C:\Users\Rook\AI\xtts")
        xtts_python = xtts_dir / ".venv" / "Scripts" / "python.exe"
        server_py = xtts_dir / "server.py"
        if not xtts_python.exists() or not server_py.exists():
            print(f"[raiken] ERR: xtts venv or server.py missing under {xtts_dir}", flush=True)
            self._emit_status("tts", "down", "xtts files missing")
            return False

        print("[raiken] TTS server down. launching hidden...", flush=True)
        self._emit_status("tts", "busy", "starting")
        log_path = _LOG_DIR / "tts_server.log"
        log_fp = open(log_path, "a", encoding="utf-8", buffering=1)
        env = {**os.environ, "COQUI_TOS_AGREED": "1"}
        subprocess.Popen(
            [str(xtts_python), str(server_py)],
            cwd=str(xtts_dir),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
            close_fds=True,
        )
        return True

    async def _wait_for_tts(self, timeout_sec: int = TTS_STARTUP_TIMEOUT_SEC):
        waited = 0
        while waited < timeout_sec:
            await asyncio.sleep(2)
            waited += 2
            if self._tts_up():
                print(f"[raiken] TTS up after {waited}s", flush=True)
                self._emit_status("tts", "up")
                return True
        print(f"[raiken] WARN: TTS didn't come up in {timeout_sec}s", flush=True)
        self._emit_status("tts", "down", "timeout")
        return False

    # --- Synthesis + playback pipeline ---------------------------------------
    def _synthesis_worker(self):
        with httpx.Client(timeout=90) as http:
            while True:
                item = self.sentence_q.get()
                if item is END_OF_TURN:
                    self.wav_q.put(END_OF_TURN)
                    continue
                # Drop items that were queued before a barge-in.
                if self._barge_flag:
                    continue
                try:
                    r = http.post(TTS_URL, json={"text": item, "play": False})
                    r.raise_for_status()
                    wav_path = r.json().get("path")
                    # Check again after synth — the user may have barged DURING the POST.
                    if wav_path and not self._barge_flag:
                        self.wav_q.put(wav_path)
                except Exception as e:
                    print(f"[synth fail: {e}]", flush=True)

    def _playback_worker(self):
        while True:
            item = self.wav_q.get()
            if item is END_OF_TURN:
                continue
            # Skip any items that arrived after a barge-in but slipped through the drain.
            if self._barge_flag:
                continue
            try:
                proc = subprocess.Popen(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", item],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                with self._current_ffplay_lock:
                    self._current_ffplay = proc
                proc.wait()
            except Exception as e:
                print(f"[playback fail: {e}]", flush=True)
            finally:
                with self._current_ffplay_lock:
                    self._current_ffplay = None

    # --- STT ------------------------------------------------------------------
    def _run_stt_sync(self, audio: np.ndarray) -> str:
        print(f"[stt] duration={len(audio)/SAMPLE_RATE:.2f}s", flush=True)
        segments, info = self.whisper.transcribe(
            audio, beam_size=1, language="en",
            initial_prompt=PROJECT_VOCAB,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        for pattern, replacement in STT_REPLACEMENTS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    # --- Turn handling --------------------------------------------------------
    async def submit(self, text: str):
        """Unified entry: voice and UI text both funnel here."""
        text = (text or "").strip()
        if not text:
            return

        # Mark user interaction for presence detection — unless this is an
        # auto-fire synthetic turn (worker-return wake-up), which isn't Rook.
        is_synthetic = text.startswith("[WORKER-RETURN]")
        if _APP_REF is not None and not is_synthetic:
            _APP_REF.mark_user_interaction()

        # If a turn is still finishing up (barge-in case), give it a brief moment.
        if self.turn_in_progress and self._barge_flag:
            for _ in range(15):   # up to ~1.5 sec
                await asyncio.sleep(0.1)
                if not self.turn_in_progress:
                    break
        if self.turn_in_progress:
            # Synthetic worker-return wake-ups defer silently — the current
            # turn's finally block will pick up pending worker results and fire
            # another synthetic turn if still needed.
            if is_synthetic:
                return
            # Queue the message instead of dropping it. The finally block of
            # the in-flight turn drains self._pending_submit, so the user's
            # input is never silently lost. Only the most recent queued
            # message survives — if more arrive, they overwrite.
            if self._pending_submit is not None:
                self._emit_chat("system", "(replaced queued message)")
            self._pending_submit = text
            self._emit_chat("system", "(queued — will run after current turn)")
            return
        self.turn_in_progress = True
        self._emit_status("orchestrator", "busy")
        try:
            await self._execute_turn(text)
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            print(f"[turn error] {err_msg}", flush=True)
            self._emit_chat("error", f"Turn failed: {err_msg}")
        finally:
            self.turn_in_progress = False
            self._emit_status("orchestrator", "up")
            # Drain any message queued during this turn (e.g. barge-in + resend).
            pending = self._pending_submit
            self._pending_submit = None
            if pending:
                asyncio.create_task(self.submit(pending))
            else:
                # User didn't queue anything. If a worker completed during our
                # turn, auto-fire a wake so Raiken narrates — unless Rook is
                # clearly away (15+ min idle), in which case the phone push
                # is the channel that matters; TTS to an empty room is waste.
                if _APP_REF is not None:
                    with _APP_REF._pending_worker_results_lock:
                        has_worker_updates = bool(_APP_REF._pending_worker_results)
                    if has_worker_updates:
                        presence = _APP_REF.get_effective_presence()
                        # PTT gate: if Rook grabbed PTT right as our turn ended,
                        # let _on_release handle the wake so it coalesces with
                        # any speech he's about to deliver.
                        if presence != "away" and not self.recording:
                            asyncio.create_task(self.submit("[WORKER-RETURN]"))

    def _strip_code_blocks_for_tts(self, chunk: str) -> str:
        """Filter out ``` ``` code-block content from a streamed chunk for TTS.
        Maintains in-code-block state across calls via self._in_code_block.
        The UI chat window still shows the full chunk (incl. code blocks) — only
        TTS routing is affected."""
        out = []
        i = 0
        while i < len(chunk):
            if self._in_code_block:
                idx = chunk.find("```", i)
                if idx == -1:
                    return "".join(out)  # still in block, ignore rest of chunk
                i = idx + 3
                self._in_code_block = False
            else:
                idx = chunk.find("```", i)
                if idx == -1:
                    out.append(chunk[i:])
                    return "".join(out)
                out.append(chunk[i:idx])
                i = idx + 3
                self._in_code_block = True
        return "".join(out)

    async def _execute_turn(self, text: str):
        # Clear barge flag at the start of every turn.
        self._barge_flag = False
        # Per-turn streaming state.
        self._in_code_block = False
        # Short-sentence accumulator so we don't synthesize single-word chunks.
        MIN_TTS_CHARS = 40
        pending_short = ""

        # Prepend any background-worker completions that landed since the last
        # turn. Raiken sees them as a preamble before Rook's message and can
        # narrate the completion ("Oracle is back with the research — ...").
        # A [WORKER-RETURN] synthetic text is an auto-wake from record_worker_result
        # — Rook didn't say anything; Raiken should narrate briefly and stop.
        is_worker_return_wake = text.strip().startswith("[WORKER-RETURN]")

        # Foreman broadcast. Fire in parallel with Raiken's turn so dispatch
        # evaluation doesn't block conversation. Skip synthetic worker-return
        # wakes — those are narration-only; Foreman has nothing to decide.
        if not is_worker_return_wake and self.foreman_client is not None:
            asyncio.create_task(self._execute_foreman_turn(text))

        _display_text = text  # saved before preamble injection so the UI shows only Rook's words
        if _APP_REF is not None:
            pending_results = _APP_REF.drain_pending_worker_results()
            if pending_results:
                lines = ["[worker-updates since your last turn:]"]
                for r in pending_results:
                    elapsed = r.get("elapsed", 0) or 0
                    if r.get("success"):
                        out = (r.get("output") or "").strip().replace("\r", "")
                        # Trim large outputs — the full text is already in the
                        # Log tab; Raiken only needs enough to summarize.
                        if len(out) > 800:
                            out = out[:800] + "... [truncated; full output in the Log tab]"
                        lines.append(f"- {r['name']} finished in {elapsed:.0f}s: {out}")
                    else:
                        lines.append(
                            f"- {r['name']} FAILED in {elapsed:.0f}s: "
                            f"{r.get('error', 'unknown')}"
                        )
                lines.append("[end worker-updates]")
                if is_worker_return_wake:
                    lines.append(
                        "[This is an auto-wake — Rook has NOT spoken. Narrate the "
                        "completion over TTS in one or two short sentences, do not "
                        "ask a question, then end your turn.]"
                    )
                    text = "\n".join(lines)
                else:
                    text = "\n".join(lines) + "\n\n" + text
            elif is_worker_return_wake:
                # No results in the queue (already drained by an earlier turn).
                # Silently bail rather than firing a pointless empty turn.
                print("[worker-return] wake fired but queue already empty; skipping", flush=True)
                return

        # Drain any stale items left over from a prior barge-in — synth/playback
        # threads might still be holding stuff.
        for _q in (self.sentence_q, self.wav_q):
            while True:
                try:
                    _q.get_nowait()
                except queue.Empty:
                    break

        # Show the user's original text in the UI. Skip for auto-wake turns —
        # Rook didn't say anything, and the worker output is already visible via
        # the clickable badge; emitting the raw preamble would double-post it.
        if not is_worker_return_wake:
            self._emit_chat("user", _display_text, done=True)
        print(f"[user] {text!r}", flush=True)

        # If this turn follows a barge-in, prepend a hidden tag so Claude knows
        # the prior response was cut off and uses judgment about whether to
        # integrate or ignore the earlier partial context.
        prompt_for_claude = text
        if self._just_barged:
            prompt_for_claude = (
                "[INTERRUPT — your previous response was cut off mid-stream. "
                "Use judgment on whether earlier context still applies.] " + text
            )
            self._just_barged = False

        await self.client.query(prompt_for_claude)

        print("[raiken] ", end="", flush=True)
        first_text = True
        buffer = ""
        async for msg in self.client.receive_response():
            # Bail early if the user barged in.
            if self._barge_flag:
                print(" [aborted]", flush=True)
                break
            if isinstance(msg, RateLimitEvent):
                # Capture rate-limit info so the UI can show it live.
                self._capture_rate_limit(msg)
                continue
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if self._barge_flag:
                        break
                    if isinstance(block, TextBlock):
                        chunk = block.text
                        print(chunk, end="", flush=True)
                        # UI gets the full chunk (code blocks + all).
                        self._emit_chat("raiken", chunk, append=not first_text)
                        first_text = False
                        # TTS gets only the non-code-block portion.
                        tts_chunk = self._strip_code_blocks_for_tts(chunk)
                        buffer += tts_chunk
                        while True:
                            m = re.search(r"[.!?](\s|$)", buffer)
                            if not m:
                                break
                            end = m.end()
                            sentence = buffer[:end].strip()
                            buffer = buffer[end:].lstrip()
                            if not sentence or self._barge_flag:
                                continue
                            # Merge too-short sentences so XTTS has enough context.
                            if len(sentence) < MIN_TTS_CHARS:
                                pending_short = (pending_short + " " + sentence).strip()
                                if len(pending_short) >= MIN_TTS_CHARS:
                                    self.sentence_q.put(pending_short)
                                    pending_short = ""
                            else:
                                if pending_short:
                                    sentence = (pending_short + " " + sentence).strip()
                                    pending_short = ""
                                self.sentence_q.put(sentence)
            elif isinstance(msg, ResultMessage):
                break

        if not self._barge_flag:
            # Flush remainder + any held-short accumulator.
            tail_bits = [x for x in (pending_short.strip(), buffer.strip()) if x]
            if tail_bits:
                self.sentence_q.put(" ".join(tail_bits))
        self.sentence_q.put(END_OF_TURN)

        self._emit_chat("raiken", "", done=True)
        print()

        # Refresh context usage after the turn completes. ContextUsageResponse is
        # a TypedDict (i.e. a plain dict), so dict access — not attribute access.
        try:
            ctx = await self.client.get_context_usage()
            def _g(k):
                if isinstance(ctx, dict):
                    return ctx.get(k)
                return getattr(ctx, k, None)
            with self._usage_lock:
                self._context_usage = {
                    "total": _g("totalTokens"),
                    "max": _g("maxTokens"),
                    "pct": _g("percentage"),
                    "model": _g("model"),
                }
            print(f"[ctx] {self._context_usage}", flush=True)
        except Exception as e:
            print(f"[ctx-usage fetch failed: {type(e).__name__}: {e}]", flush=True)

    def _capture_rate_limit(self, evt: RateLimitEvent):
        info = getattr(evt, "rate_limit_info", None)
        if info is None:
            return
        rl_type = getattr(info, "rate_limit_type", None) or "unknown"
        raw = getattr(info, "raw", None)
        entry = {
            "utilization": getattr(info, "utilization", None),
            "resets_at": getattr(info, "resets_at", None),
            "status": getattr(info, "status", None),
            "overage_status": getattr(info, "overage_status", None),
        }
        # If utilization is missing from the typed field but present in raw, grab it.
        if entry["utilization"] is None and isinstance(raw, dict):
            for k in ("utilization", "used", "utilization_pct", "consumed"):
                if k in raw and isinstance(raw[k], (int, float)):
                    entry["utilization"] = raw[k]
                    break
        with self._usage_lock:
            self._rate_limits[str(rl_type)] = entry
        print(f"[rl] {rl_type}: {entry} raw={raw}", flush=True)

    async def voice_turn(self, audio: np.ndarray):
        text = self._run_stt_sync(audio)
        print(f"[stt] >>> {text!r}", flush=True)
        if text:
            await self.submit(text)

    # --- Foreman (silent dispatch twin) ---------------------------------------
    async def _execute_foreman_turn(self, text: str):
        """Broadcast the user's text to the Foreman SDK session in parallel with
        Raiken's conversational turn. Foreman evaluates the message and, if
        warranted, calls `dispatch_worker` — the SDK handles that tool invocation
        internally via the MCP server, so by the time her stream ends the work
        is already scheduled. Her text output is drained but never emitted to
        the UI — she's silent by design.

        Not awaited from `_execute_turn`; runs as a background task. Failures
        are logged and swallowed so they don't crash Raiken's side.
        """
        if self.foreman_client is None or self._foreman_lock is None:
            return
        async with self._foreman_lock:
            try:
                await self.foreman_client.query(text)
                async for msg in self.foreman_client.receive_response():
                    # Drain the stream. We don't emit Foreman's assistant text
                    # to the UI — her job is side-effectful tool calls only.
                    if isinstance(msg, ResultMessage):
                        break
            except Exception as e:
                print(f"[foreman turn error] {type(e).__name__}: {e}", flush=True)

    # --- Boot -----------------------------------------------------------------
    async def run(self):
        self.loop = asyncio.get_running_loop()
        # Lock lives on the loop; only create it inside run() so it binds to the
        # active asyncio loop rather than a stale one from a prior attempt.
        self._foreman_lock = asyncio.Lock()
        self._emit_status("orchestrator", "busy", "starting")

        tts_launched = self._launch_tts_if_needed()

        print(f"[raiken] loading Whisper {WHISPER_MODEL}...", flush=True)
        t0 = time.time()
        self.whisper = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
        print(f"[raiken] Whisper ready ({time.time()-t0:.1f}s)", flush=True)

        if tts_launched:
            await self._wait_for_tts()

        threading.Thread(target=self._synthesis_worker, daemon=True).start()
        threading.Thread(target=self._playback_worker, daemon=True).start()

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            callback=self._audio_callback,
        )
        stream.start()

        keyboard.on_press_key(HOTKEY, self._on_press)
        keyboard.on_release_key(HOTKEY, self._on_release)

        # --- Foreman / Raiken split --------------------------------------------
        # Two ClaudeSDKClient sessions live side-by-side:
        #   * Raiken (self.client)       — conversation, TTS, vault ops.
        #   * Foreman (self.foreman_client) — silent dispatcher, worker tools only.
        # Every user submit broadcasts to BOTH. Raiken responds with speech;
        # Foreman evaluates the same message and dispatches in parallel. Tool
        # access is split at the SDK level so double-dispatch is impossible by
        # construction (Raiken doesn't even SEE the dispatch tool).
        raiken_options = ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT,
            permission_mode="bypassPermissions",
            mcp_servers={
                "raiken-vault": VAULT_MCP_SERVER,
            },
            allowed_tools=[
                "mcp__raiken-vault__vault_status",
                "mcp__raiken-vault__vault_unlock",
                "mcp__raiken-vault__vault_search",
                "mcp__raiken-vault__vault_copy_password",
                "mcp__raiken-vault__vault_copy_username",
                "mcp__raiken-vault__vault_copy_totp",
                "mcp__raiken-vault__vault_lock",
            ],
        )
        foreman_options = ClaudeAgentOptions(
            system_prompt=FOREMAN_SYSTEM_PROMPT,
            permission_mode="bypassPermissions",
            mcp_servers={
                "raiken-workers": WORKER_MCP_SERVER,
            },
            allowed_tools=[
                "mcp__raiken-workers__dispatch_worker",
                "mcp__raiken-workers__list_workers",
            ],
        )
        async with ClaudeSDKClient(options=raiken_options) as self.client, \
                   ClaudeSDKClient(options=foreman_options) as self.foreman_client:
            self._emit_status("orchestrator", "up")
            self._emit_status("ptt", "up")
            self._emit_chat("system", f"Raiken ready. Hold {HOTKEY.upper()} to talk, or type below.")
            print(f"[raiken] ready. hold {HOTKEY.upper()} to talk.", flush=True)
            try:
                while True:
                    await asyncio.sleep(1)
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass

        stream.stop()


# =============================================================================
# RaikenApp — owns UI (main thread) + asyncio worker thread
# =============================================================================
class RaikenApp:
    def __init__(self):
        self.ui_event_queue: queue.Queue = queue.Queue()
        self.raiken = Raiken(self.ui_event_queue)
        self.asyncio_loop: asyncio.AbstractEventLoop | None = None
        self.asyncio_thread: threading.Thread | None = None
        self.window: RaikenWindow | None = None
        self.tray: RaikenTray | None = None

        # Live workers tracking for the UI panel.
        self._active_workers: dict[str, dict] = {}
        self._active_workers_lock = threading.Lock()

        # Worker completions queued while Raiken was busy or idle. Drained at
        # the start of each turn so Raiken sees the results as a preamble and
        # can narrate them to Rook.
        self._pending_worker_results: list[dict] = []
        self._pending_worker_results_lock = threading.Lock()

        # Per-name asyncio locks — serialize concurrent dispatches to the SAME
        # worker (two `claude --resume <id>` subprocesses would race the session).
        self._worker_locks: dict[str, asyncio.Lock] = {}

        # Full worker transcripts keyed by run_id — the Log tab + click-to-open
        # flow both read from here. Counter advances per dispatch.
        self._worker_transcripts: dict[int, dict] = {}
        self._worker_transcripts_lock = threading.Lock()
        self._worker_run_counter = 0

        # Presence model. Auto state = 3-tier (active/idle/away) derived from
        # MIN(kb/mouse idle from GetLastInputInfo, seconds since last in-app
        # voice/UI turn). Manual override = 2-state force ('active'/'away').
        self._presence_idle_ms: int = 10**9       # kb/mouse idle; polled every 2s
        self._presence_override: str = "auto"     # "auto" | "active" | "away"
        self._presence_lock = threading.Lock()
        self._last_user_interaction_ts: float = 0.0  # updated on every submit()

        # Origin tracking — set by the input layer (voice/UI = "local";
        # Discord bot when it lands = "discord"; ntfy reply channel = "ntfy").
        # Every dispatch reads this at register_active_worker time so the
        # notification router can send the return to the right place.
        self._current_origin: str = "local"

        # Module-level reference so SDK tools can reach the UI + workers.
        global _APP_REF
        _APP_REF = self

        # Pre-seed the named-worker registry (Marl, CMMC Wizard, Shadowling
        # Commander, Ledger, Herald, Scribe, Oracle, Cipher, Keeper, Pyre).
        # Idempotent — safe on every boot.
        try:
            seed_named_workers()
        except Exception as e:
            print(f"[raiken] seed_named_workers failed: {e}", flush=True)

    def register_active_worker(self, name: str, task: str):
        # Reference-counted so concurrent same-name dispatches don't race: the
        # panel entry survives until the last background task for that name
        # unregisters. Task text updates to the most recent dispatch so Rook
        # sees what's currently running. Origin is captured at registration so
        # the notification router knows where to send the reply.
        origin = self._current_origin
        with self._active_workers_lock:
            entry = self._active_workers.get(name)
            if entry is None:
                self._active_workers[name] = {
                    "task": task, "started_at": time.time(), "count": 1,
                    "origin": origin,
                }
            else:
                entry["count"] = entry.get("count", 0) + 1
                entry["task"] = task
                entry["origin"] = origin  # latest dispatch's origin wins

    def unregister_active_worker(self, name: str):
        with self._active_workers_lock:
            entry = self._active_workers.get(name)
            if entry is None:
                return
            entry["count"] = entry.get("count", 1) - 1
            if entry["count"] <= 0:
                self._active_workers.pop(name, None)

    def active_workers_snapshot(self) -> dict[str, dict]:
        with self._active_workers_lock:
            return {k: dict(v) for k, v in self._active_workers.items()}

    def named_agents_snapshot(self) -> list[dict]:
        """Canonical named agents only (the pre-seeded roster).

        Ad-hoc worker entries from old dispatches (e.g. "test-worker",
        "raiken-ui", "research") are filtered out so the UI shows the real
        agent roster, not stale registry noise.
        """
        try:
            return [a for a in list_workers() if a.get("is_canonical")]
        except Exception:
            return []

    def store_worker_result(self, name: str, task: str, result: dict, origin: str = "local") -> int:
        """Persist a completed worker's full transcript and return a run_id the
        UI can use to look it up (badge click -> jump to log mark). `origin`
        is carried forward from dispatch time so notification routing knows
        where to send the reply (local TTS vs Discord reply vs phone push)."""
        with self._worker_transcripts_lock:
            self._worker_run_counter += 1
            run_id = self._worker_run_counter
            self._worker_transcripts[run_id] = {
                "name": name,
                "task": task,
                "output": result.get("output", "") or "",
                "error": result.get("error", "") or "",
                "elapsed": float(result.get("elapsed", 0.0) or 0.0),
                "success": bool(result.get("success")),
                "origin": origin,
                "ts": time.time(),
            }
            return run_id

    def get_worker_result(self, run_id: int) -> dict | None:
        with self._worker_transcripts_lock:
            data = self._worker_transcripts.get(run_id)
            return dict(data) if data else None

    # --- Notification routing -----------------------------------------------
    def route_worker_return_notification(
        self, worker_name: str, success: bool, elapsed: float, run_id: int
    ):
        """Decide which external channels to hit for a worker return.

        Inputs that drive the decision:
          * origin  — where did the dispatch come from? (local / discord / phone)
          * idle    — seconds since Rook's last activity
          * elapsed — how long did the task run

        Local TTS + chat + Log entry always happen (handled elsewhere). This
        method only picks the EXTRA channels to ping — phone push, desktop
        toast, Discord reply. Right now those backends aren't wired yet, so
        decisions are logged and the actual pushes are no-ops. Wire them up
        as backends come online.
        """
        transcript = self.get_worker_result(run_id) or {}
        origin = transcript.get("origin", "local")
        idle_sec = self.get_idle_seconds()
        idle_min = idle_sec / 60.0

        channels: list[str] = []

        # 1. If dispatched from a remote surface, always reply to that surface.
        if origin != "local":
            channels.append(f"reply-{origin}")

        # 2. Phone push triggers: Rook away 15+ min, OR queued a long-running
        #    task (>5 min) then drifted away (5+ min idle).
        should_phone_push = (
            idle_sec >= self.PRESENCE_IDLE_MAX_SEC
            or (elapsed >= 300 and idle_sec >= 300)
        )
        if should_phone_push:
            channels.append("phone-push")

        # 3. Desktop toast whenever Rook is not clearly at the PC — he'll see
        #    it when he glances back even if TTS already played.
        if idle_sec >= self.PRESENCE_ACTIVE_MAX_SEC:
            channels.append("desktop-toast")

        if not channels:
            print(
                f"[notify] {worker_name} return: local only "
                f"(origin={origin}, idle={idle_min:.1f}m, elapsed={elapsed:.0f}s)",
                flush=True,
            )
            return

        # Backends not yet wired — log the decision. Replace each log line with
        # a real push when the corresponding backend exists.
        print(
            f"[notify] {worker_name} return routing: {channels} "
            f"(origin={origin}, idle={idle_min:.1f}m, elapsed={elapsed:.0f}s, "
            f"success={success})",
            flush=True,
        )
        for ch in channels:
            if ch == "phone-push":
                # TODO: POST to ntfy.sh topic per notification_stack.md
                print(f"[notify] (stub) phone-push: {worker_name} done in {elapsed:.0f}s", flush=True)
            elif ch == "desktop-toast":
                # TODO: win10toast-click with click-to-focus RCC
                print(f"[notify] (stub) desktop-toast: {worker_name} done", flush=True)
            elif ch.startswith("reply-"):
                target = ch.removeprefix("reply-")
                # TODO: route through Discord bot / ntfy reply channel
                print(f"[notify] (stub) reply-to-{target}: {worker_name} done", flush=True)

    # --- Presence -----------------------------------------------------------
    def set_presence_override(self, override: str):
        """Called from the UI toggle: 'auto' | 'active' | 'away'."""
        if override not in ("auto", "active", "away"):
            return
        with self._presence_lock:
            self._presence_override = override
        # Immediately re-emit so the dot reflects the override.
        self._refresh_presence_ui()

    # Presence thresholds (seconds). Below ACTIVE = at the PC right now.
    # ACTIVE..IDLE = nearby but not touching anything (reading, on a call, etc.).
    # Above IDLE = likely away from the machine, phone-push the next thing that
    # wants Rook's attention.
    PRESENCE_ACTIVE_MAX_SEC = 120    # <2 min idle
    PRESENCE_IDLE_MAX_SEC = 900      # 2-15 min idle
    # Voice-turn window — PTT hotkey doesn't register as kb/mouse to Windows,
    # so we bridge using our own interaction timestamp. Same threshold as
    # ACTIVE so in-app voice counts as "at the PC".
    VOICE_PRESENCE_WINDOW_SEC = 120

    def get_effective_presence(self) -> str:
        """Returns 'active' | 'idle' | 'away'. Manual override wins; otherwise
        combine in-app voice/UI interaction with Windows GetLastInputInfo."""
        with self._presence_lock:
            if self._presence_override != "auto":
                # Map the 2-state override ('active'/'away') onto the 3-state model.
                return "active" if self._presence_override == "active" else "away"
            now = time.time()
            voice_idle = now - self._last_user_interaction_ts
            kb_idle = self._presence_idle_ms / 1000.0 if hasattr(self, "_presence_idle_ms") else 10**9
            # Minimum of the two — any recent signal at all counts.
            effective_idle = min(voice_idle, kb_idle)
        if effective_idle < self.PRESENCE_ACTIVE_MAX_SEC:
            return "active"
        if effective_idle < self.PRESENCE_IDLE_MAX_SEC:
            return "idle"
        return "away"

    def get_idle_seconds(self) -> float:
        """Seconds since Rook's last observed activity (kb/mouse or in-app voice).
        Notification routing uses this to decide whether a phone push is warranted."""
        with self._presence_lock:
            voice_idle = time.time() - self._last_user_interaction_ts
            kb_idle = self._presence_idle_ms / 1000.0 if hasattr(self, "_presence_idle_ms") else 10**9
            return min(voice_idle, kb_idle)

    def mark_user_interaction(self):
        """Called whenever Rook talks / submits. The 3-tier auto presence in
        get_effective_presence() uses this timestamp as the in-app signal."""
        with self._presence_lock:
            self._last_user_interaction_ts = time.time()

    def _refresh_presence_ui(self):
        state = self.get_effective_presence()
        self.raiken._emit_presence(state)

    def record_worker_result(self, name: str, result: dict):
        """Called from a background dispatch task when a worker completes.
        Queues the result for inclusion in Raiken's next turn preamble."""
        with self._pending_worker_results_lock:
            self._pending_worker_results.append({
                "name": name,
                "success": bool(result.get("success")),
                "output": result.get("output", "") if result.get("success") else "",
                "error": result.get("error", "") if not result.get("success") else "",
                "elapsed": result.get("elapsed", 0.0) or 0.0,
                "completed_at": time.time(),
            })

    def drain_pending_worker_results(self) -> list[dict]:
        """Pop every queued worker result (called by Raiken at turn start)."""
        with self._pending_worker_results_lock:
            drained = list(self._pending_worker_results)
            self._pending_worker_results.clear()
        return drained

    def _get_worker_lock(self, name: str) -> asyncio.Lock:
        """Per-name asyncio.Lock for serializing same-worker dispatches.
        Caller must be on the asyncio loop (uses `async with`)."""
        lock = self._worker_locks.get(name)
        if lock is None:
            lock = asyncio.Lock()
            self._worker_locks[name] = lock
        return lock

    def context_usage_snapshot(self) -> dict | None:
        with self.raiken._usage_lock:
            return dict(self.raiken._context_usage) if self.raiken._context_usage else None

    def rate_limits_snapshot(self) -> dict:
        with self.raiken._usage_lock:
            return {k: dict(v) for k, v in self.raiken._rate_limits.items()}

    # --- Asyncio thread -------------------------------------------------------
    def _run_asyncio(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.asyncio_loop = loop
        try:
            loop.run_until_complete(self.raiken.run())
        except Exception as e:
            print(f"[asyncio thread error] {e}", flush=True)
        finally:
            loop.close()

    def start_asyncio(self):
        self.asyncio_thread = threading.Thread(target=self._run_asyncio, daemon=True, name="raiken-asyncio")
        self.asyncio_thread.start()
        # Wait briefly for the loop to be assigned.
        deadline = time.time() + 5
        while self.asyncio_loop is None and time.time() < deadline:
            time.sleep(0.05)

    def start_presence_monitor(self):
        """Background thread polling Windows GetLastInputInfo every ~2 sec.
        Derives `active` / `maybe` / `away` from the resulting idle-ms delta.
        Manual UI override (if != 'auto') wins against this auto state."""
        t = threading.Thread(
            target=self._presence_monitor_loop,
            daemon=True,
            name="raiken-presence",
        )
        t.start()

    def _presence_monitor_loop(self):
        # Windows-only; on other OSes the monitor stays quietly in "away" and
        # the manual UI toggle is the only input.
        if os.name != "nt":
            return
        try:
            import ctypes
            from ctypes import wintypes

            class LASTINPUTINFO(ctypes.Structure):
                _fields_ = [
                    ("cbSize", wintypes.UINT),
                    ("dwTime",  wintypes.DWORD),
                ]

            get_last_input = ctypes.windll.user32.GetLastInputInfo
            get_tick_count = ctypes.windll.kernel32.GetTickCount
        except Exception as e:
            print(f"[presence] ctypes bindings unavailable: {e}", flush=True)
            return

        lii = LASTINPUTINFO()
        lii.cbSize = ctypes.sizeof(LASTINPUTINFO)

        last_emitted_state: str | None = None
        last_emitted_bucket: int | None = None
        while True:
            try:
                if get_last_input(ctypes.byref(lii)):
                    idle_ms = get_tick_count() - lii.dwTime
                else:
                    idle_ms = 10**9
                with self._presence_lock:
                    self._presence_idle_ms = idle_ms
                effective = self.get_effective_presence()
                idle_sec = self.get_idle_seconds()
                # Bucket by minute so the indicator updates steadily but not
                # on every 2s tick (that'd flicker).
                bucket = int(idle_sec // 60)
                if effective != last_emitted_state or bucket != last_emitted_bucket:
                    last_emitted_state = effective
                    last_emitted_bucket = bucket
                    with self._presence_lock:
                        override = self._presence_override
                    detail = ""
                    if override != "auto":
                        detail = f"forced {override}"
                    elif effective == "active":
                        # Don't clutter when actually at the PC.
                        detail = ""
                    else:
                        mins = int(idle_sec // 60)
                        if mins < 1:
                            detail = f"idle {int(idle_sec)}s"
                        else:
                            detail = f"idle {mins}m"
                    self.raiken._emit_presence(effective, detail)
            except Exception as e:
                print(f"[presence] monitor tick failed: {e}", flush=True)
            time.sleep(2.0)

    # --- UI callbacks ---------------------------------------------------------
    def submit_from_ui(self, text: str):
        if self.asyncio_loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.raiken.submit(text), self.asyncio_loop)

    # --- Bitwarden vault UI helpers -----------------------------------------
    # `bw status` shells out and can block 100-300ms+. Calling it from the Tk
    # main thread every 4 seconds was the source of visible lag spikes. Instead
    # keep a cached snapshot refreshed by a low-frequency background thread and
    # let the UI read the cache instantly.
    def vault_state_snapshot(self) -> dict:
        """Instant, non-blocking read for the UI button. Combines the live
        `is_unlocked()` flag (cheap, in-memory) with the last cached `bw status`
        result (refreshed by the background poller)."""
        unlocked_now = False
        try:
            unlocked_now = _BW_SESSION.is_unlocked()
        except Exception:
            pass
        cached = getattr(self, "_vault_status_cache", None) or {"state": "unknown"}
        snap = dict(cached)
        snap["in_memory_unlocked"] = unlocked_now
        if unlocked_now:
            snap["state"] = "unlocked"
        return snap

    def _vault_status_poll_loop(self):
        """Background thread — shells out to `bw status` every 30 seconds and
        caches the result. The UI reads the cache instantly via
        `vault_state_snapshot` so no main-thread blocking happens."""
        while True:
            try:
                self._vault_status_cache = _BW_SESSION.status()
            except Exception as e:
                self._vault_status_cache = {"state": "error", "detail": str(e)[:120]}
            time.sleep(30.0)

    def start_vault_status_monitor(self):
        self._vault_status_cache = {"state": "unknown"}
        threading.Thread(
            target=self._vault_status_poll_loop,
            daemon=True,
            name="raiken-vault-poll",
        ).start()

    def unlock_vault_from_ui(self):
        """Click-handler for the UI vault button. Prompts Rook for the master
        password via the native dialog (same path `vault_unlock` tool uses),
        unlocks, posts a system message with the outcome."""
        if self.window is None:
            return
        status = self.vault_state_snapshot()
        state = status.get("state")
        if state == "cli-missing":
            self.ui_event_queue.put(ChatEvent(
                role="system", text="Bitwarden CLI not found — install bw first.", done=True,
            ))
            return
        if state == "unauthenticated":
            self.ui_event_queue.put(ChatEvent(
                role="system", text="Bitwarden not logged in — run `bw login` in a terminal first.", done=True,
            ))
            return
        if _BW_SESSION.is_unlocked():
            self.ui_event_queue.put(ChatEvent(
                role="system", text="Vault already unlocked.", done=True,
            ))
            return
        pw = self.window.request_password_from_asyncio(
            "Bitwarden — Unlock Vault", "Master password:", timeout_sec=120,
        )
        if not pw:
            self.ui_event_queue.put(ChatEvent(
                role="system", text="Vault unlock cancelled.", done=True,
            ))
            return
        ok, msg = _BW_SESSION.unlock_with_password(pw)
        pw = None  # wipe reference
        self.ui_event_queue.put(ChatEvent(
            role="system" if ok else "error", text=f"Vault: {msg}", done=True,
        ))

    def lock_vault_from_ui(self):
        """Click-handler when the vault is already unlocked — relock it."""
        try:
            _BW_SESSION.lock()
        except Exception as e:
            self.ui_event_queue.put(ChatEvent(
                role="error", text=f"Vault lock failed: {e}", done=True,
            ))
            return
        self.ui_event_queue.put(ChatEvent(
            role="system", text="Vault locked.", done=True,
        ))

    def restart_tts_from_ui(self):
        """Kill current TTS; raiken.run_worker checks will re-launch on next use."""
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "$p = (Get-NetTCPConnection -LocalPort 7851 -ErrorAction SilentlyContinue | Select-Object -First 1).OwningProcess; if ($p) { Stop-Process -Id $p -Force }"],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self.ui_event_queue.put(StatusEvent("tts", "busy", "restarting"))
        except Exception as e:
            print(f"[restart_tts fail] {e}", flush=True)
        # Kick a relaunch attempt on the asyncio thread.
        if self.asyncio_loop:
            async def relaunch():
                self.raiken._launch_tts_if_needed()
                await self.raiken._wait_for_tts()
            asyncio.run_coroutine_threadsafe(relaunch(), self.asyncio_loop)

    def quit(self):
        # Stop the asyncio loop.
        if self.asyncio_loop:
            try:
                self.asyncio_loop.call_soon_threadsafe(self.asyncio_loop.stop)
            except Exception:
                pass
        # Stop the tray.
        if self.tray:
            try:
                self.tray.stop()
            except Exception:
                pass
        # Quit tkinter.
        if self.window:
            try:
                self.window.quit()
            except Exception:
                pass
        os._exit(0)  # hard exit; daemon threads might hold otherwise

    def relaunch(self):
        """Spawn a fresh Raiken instance via the hidden VBS launcher, then quit.
        The TTS server is detached so it stays alive across the relaunch."""
        try:
            from pathlib import Path as _P
            vbs = _P(r"C:\Users\Rook\AI\raiken\launch_raiken.vbs")
            if vbs.exists():
                subprocess.Popen(
                    ["wscript.exe", str(vbs)],
                    close_fds=True,
                )
            else:
                subprocess.Popen(
                    ["cmd", "/c", "start", "", r"C:\Users\Rook\AI\raiken\start_raiken.bat"],
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    close_fds=True,
                )
        except Exception as e:
            print(f"[relaunch failed: {e}]", flush=True)
        # Brief delay so the spawned process gets past file-lock contention.
        time.sleep(0.4)
        self.quit()

    def run(self):
        self.start_asyncio()
        self.start_presence_monitor()
        self.start_vault_status_monitor()
        self.window = RaikenWindow(self)
        self.tray = RaikenTray(self, self.window)
        self.tray.run_detached()
        try:
            self.window.run()
        except KeyboardInterrupt:
            pass
        self.quit()


if __name__ == "__main__":
    try:
        RaikenApp().run()
    except KeyboardInterrupt:
        print("\n[raiken] shutting down", flush=True)
