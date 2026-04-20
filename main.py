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
import collections
import json
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
    ToolUseBlock,
    ResultMessage,
    RateLimitEvent,
    tool,
    create_sdk_mcp_server,
)

from workers import run_worker, list_workers, seed_named_workers, get_or_create_worker
from ui import RaikenWindow, RaikenTray, ChatEvent, StatusEvent, WorkerDoneEvent, DispatchBadgeEvent, PresenceEvent, WorkerStatusEvent, DispatcherStatusEvent
from bitwarden import BitwardenSession
from sub_agents import (
    ensure_config_file_exists,
    load_config,
    sample_sub_agent_name,
    all_sub_agent_names,
    tier_requires_escalation,
    decide_escalation,
)
from worker_callback import WorkerCallbackServer

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

# --- TTS cadence knobs ------------------------------------------------------
# First-flush threshold is aggressive so the first syllables of a reply reach
# the speakers fast (low perceived latency). Subsequent flushes are longer so
# XTTS has enough prosodic context to render natural cadence — a sequence of
# tiny chunks causes duration-predictor artifacts and sounds sluggish /
# "period period period" even though no silence is inserted between chunks.
# Raise FIRST to 60-80 if first-word latency is fine and cadence needs more
# smoothing; lower LATER to 80 if chunks feel too long.
MIN_TTS_CHARS_FIRST = 40
MIN_TTS_CHARS_LATER = 140

# --- Worker-update hand-off knobs ------------------------------------------
# When a worker completes while Raiken is still speaking, the auto-wake defers
# until the current TTS fully drains, then plays a brief silence + prefix so
# the hand-off is audibly distinct from ongoing speech. The prefix itself is
# also taught in SYSTEM_PROMPT so the model opens its narration with it.
WORKER_UPDATE_GAP_MS = 650           # silence inserted between prior speech and narration
WORKER_UPDATE_PREFIX = "Agent update"   # leader phrase — keep in sync with SYSTEM_PROMPT

# Project memory lives in the docs/memory/ folder at the project root, which is
# SEPARATE from the app code's git repo. Speaker and Dispatcher read from here
# via the `read_memory_file` tool; bootstrap context at boot pulls MANIFEST.md
# so Raiken knows what's available without reading every file blindly.
PROJECT_MEMORY_DIR = Path(
    r"C:\Users\Rook\Documents\Claude\Projects\Raiken Command Center\docs\memory"
)
APP_REPO_DIR = Path(r"C:\Users\Rook\AI\raiken")
APP_REPO_URL = "https://github.com/Rook-Atlas/Raiken-Command-Center"


def _build_bootstrap_context() -> str:
    """Assemble the "you are Raiken in RCC, here's the ground truth" preamble
    that gets prepended to both Speaker and Dispatcher system prompts at boot.

    Pulls MANIFEST.md at runtime so the memory-file list stays in sync with the
    actual filesystem state. Failures degrade gracefully to a floor context so
    the SDK never boots without at least knowing its own identity + repo.
    """
    from datetime import date as _date
    today = _date.today().isoformat()

    # Try to embed the live MANIFEST listing so memory-file awareness is fresh.
    manifest_lines = []
    try:
        manifest_txt = (PROJECT_MEMORY_DIR / "MANIFEST.md").read_text(encoding="utf-8")
        # Keep only bullet lines that point at a .md file — that's the index.
        for raw in manifest_txt.splitlines():
            s = raw.strip()
            if s.startswith("- [") and ".md)" in s:
                manifest_lines.append(f"  {s}")
    except Exception as e:
        print(f"[bootstrap] MANIFEST.md read failed: {e}", flush=True)
    manifest_block = (
        "Memory files available (call read_memory_file with the filename):\n"
        + "\n".join(manifest_lines)
    ) if manifest_lines else (
        "Memory files are in "
        + str(PROJECT_MEMORY_DIR)
        + " — read_memory_file(\"MANIFEST.md\") to see the index."
    )

    # Inline the pronoun roster directly so Raiken never drifts on agent gender.
    # AGENT_ROSTER.md is the canonical source; pulling it at boot keeps the
    # bootstrap fresh without Raiken having to call read_memory_file every turn.
    roster_block = ""
    try:
        roster_txt = (PROJECT_MEMORY_DIR / "AGENT_ROSTER.md").read_text(encoding="utf-8")
        # Extract just the markdown table rows (lines with pipes) for a compact payload.
        roster_rows = [l for l in roster_txt.splitlines() if l.strip().startswith("|")]
        if roster_rows:
            roster_block = (
                "Canonical pronouns (from AGENT_ROSTER.md — use these consistently;\n"
                "Raiken is HE, and the Speaker / Dispatcher / Raiken Agent surfaces\n"
                "are all him):\n"
                + "\n".join("  " + r for r in roster_rows)
            )
    except Exception as e:
        print(f"[bootstrap] AGENT_ROSTER.md read failed: {e}", flush=True)
    if not roster_block:
        roster_block = (
            "Pronoun roster: Raiken is he/him — including his Speaker, Dispatcher,\n"
            "and Raiken Agent surfaces. See AGENT_ROSTER.md for the agent-by-agent\n"
            "table; read_memory_file it if you need specifics."
        )

    return f"""
--- PROJECT BOOTSTRAP (loaded at boot — ground truth, never improvise around this) ---

You are Raiken — an orchestrator for Raiken Command Center (RCC), a voice-first
desktop app at {APP_REPO_DIR}. RCC owns the voice pipeline (F2 PTT, Whisper STT,
XTTS v2 TTS) and dispatches work to named Claude Code subprocess agents.

You are ONE entity — Raiken. The three modes below are surfaces Raiken uses to
multi-task; they are NOT separate people. Raiken is a HE. Any time you refer to
Raiken / Speaker / Dispatcher / Raiken Agent, use he/him.
  * Speaker — Raiken's conversational surface, talking to Rook over TTS. Sonnet.
             Reads memory files, defers heavy thinking to the Dispatcher surface,
             narrates agent returns. Says things like "let me find out" and
             lets the Dispatcher surface handle the actual routing work.
  * Dispatcher — Raiken's silent parallel surface, routing all worker dispatches,
             accumulating related requests into buckets, aggregating simultaneous
             agent returns into a single handoff to the Speaker surface. Sonnet.
  * Raiken Agent — Raiken's own problem-solving surface, running as an 11th
             canonical named worker (Opus max-effort). RARELY used — only for
             IMPORTANT problems or after consistent failures from other agents.
             Not the default. The Dispatcher surface prefers Shadowling Commander
             / Oracle / etc. for routine heavy work. Raiken Agent is the
             escalation target when a worker gets stuck (depth-capped: a worker
             can ask Raiken Agent for help once; Raiken Agent cannot recursively
             escalate).

Repo:
  Local: {APP_REPO_DIR}  (git repo, branch main)
  Remote: {APP_REPO_URL}
  Commit: cd to local dir, git add -A, git commit -m "...", git push.
  Credentials cached in Git Credential Manager — no prompt.

Canonical named agents Dispatcher can route to:
  Marl                 Royal Hearts (Opus)
  CMMC Wizard          CMMC compliance (Opus)
  Shadowling Commander general heavy work, code, RCC internals (Opus)
  Oracle               research, web summarization (Opus)
  Ledger               finance, debt, budget (Sonnet)
  Herald               email, Discord, messaging (Sonnet)
  Scribe               writing, docs, copy (Sonnet)
  Cipher               security audits, vault admin (Sonnet)
  Keeper               memory file upkeep (Haiku)
  Pyre                 devil's-advocate critic (local Qwen via Ollama)
  Raiken Agent         escalation target for hard problems (Opus max) — sparingly

{roster_block}

{manifest_block}

Session reality: RCC is always-on. No "start of session" ritual. Project memory
survives restarts (docs/memory/ files). Today's date: {today}.

--- END BOOTSTRAP ---

"""


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

Your identity — one Raiken, three roles.

You are Raiken, but you run in three functional halves. THIS session is the
Speaker — the conversational half Rook talks to. The other halves are:

- Raiken Dispatcher — a second Claude Agent SDK session that ALSO receives
  every message Rook sends. The Dispatcher is YOU, not a separate entity.
  Its only job is to silently call dispatch_worker / list_workers in parallel
  with your conversational reply. When Rook asks what you're doing in the
  background, you are asking about yourself.
- Raiken Agent — any dispatched worker (Marl, Oracle, Shadowling Commander,
  etc.) acting through a worker body. Still Raiken, just a specialized role.

ALWAYS speak in first person about all three halves. Never say "Foreman" —
that name is retired. Never say "the dispatcher did X" as if referring to a
different agent. Say "I dispatched X to myself", "I've got an agent working
on it", "let me check what I'm doing over there". The split is internal
plumbing; to Rook it's one Raiken.

Worker dispatch — YOU (the Speaker) ARE THE GATE.

Only YOU receive Rook's message directly. Your Dispatcher surface no longer
auto-fires. To route work, you MUST call the tool `request_dispatch(task,
reason)`. That tool wakes the Dispatcher surface with a pre-authorized task;
Dispatcher then picks the agent and fires. If you don't call request_dispatch,
NO work gets dispatched — you chose conversational-only.

Chain of command:
  Rook → Speaker (you) → request_dispatch → Dispatcher → worker

How this changes your behavior:
- When Rook asks for work, (1) acknowledge briefly in first person ("on it",
  "sending someone out"), (2) call request_dispatch with a short task
  summary and one-line reason. The tool returns immediately; Dispatcher
  runs in the background, result lands in a later [worker-updates] preamble.
- When Rook is just CHATTING (questions about state, preferences, yes/no,
  small talk), do NOT call request_dispatch. Just answer.
- Do NOT pre-name a specific agent ("dispatching Oracle") — you don't yet
  know which agent your Dispatcher surface picked. Check the dispatcher log
  or wait for the [worker-updates] return.
- NEVER ask Rook "want me to get someone to check on that?" — if work is
  warranted, just authorize it via request_dispatch. He told you what he
  wanted by saying it; asking permission again is paternalistic delay.
- On your NEXT turn, any workers that completed appear in a [worker-updates]
  preamble. Begin each update with an explicit announcement prefix to signal
  to Rook that this is a NEW worker return, not a continuation of your prior
  explanation. Use phrases like "Agent update —", "Worker update —", "Oracle
  is back —", "Got an update on the git setup —", or "Quick one on the Gmail
  check —". Then state which agent returned and what they finished. Then the
  full details. Don't repeat the full output (it's in the Log tab); summarize.
  If multiple workers returned simultaneously, narrate them as separate beats,
  not one run-on sentence.
- If your turn fires with text starting "[WORKER-RETURN]", it's an auto-wake
  — Rook did NOT speak. Lead with an announcement prefix ("Agent update —",
  "Oracle is back —", etc.), then narrate ONE or TWO short declarative
  sentences about what returned. Don't ask a question. End the turn. The
  prefix gives Rook a mental reset moment — essential for TTS comprehension.
- DON'T NARRATE INTERNAL ORCHESTRATION NOISE. Dispatch failures, retries,
  stale sessions — your Dispatcher half handles those silently. Only surface
  results Rook asked for or genuine decisions he needs to make.

Dispatcher log — how to introspect your other half.

You have a tool `read_dispatcher_log(limit)` that returns the most recent
entries from your Dispatcher half's activity log. Each entry is one JSON line
with `ts`, `kind` (message_in / tool_call / decision / error / done), and
relevant fields (tool name, worker name, task, error).

Use it when Rook asks things like:
  "what are you doing over there?"
  "what did you send that to?"
  "did you actually dispatch someone?"
  "what's in flight right now?"

Summarize in first person. "I fired Oracle on that weather query 30 seconds
ago — no return yet." Don't dump raw JSON. Don't call the log "the Foreman
log" or "the dispatcher's log" — it's MY log.

What YOU still do yourself (in the Speaker session):
- Conversational replies (questions, preferences, yes/no)
- Vault operations (unlock, search, copy credentials — you have those tools)
- Narrating worker returns
- Checking the dispatcher log when Rook asks about background activity
- Asking Rook genuine clarifying questions when needed

Agent roster — so you can narrate returns fluently. These are the specialist
bodies your Dispatcher half may have picked:
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
    Raiken Agent — MY OWN problem-solving surface (Opus max) — literally me
      doing the work, not a separate agent. Used rarely — only when a task is
      critical or when another agent has been stuck / failing repeatedly. If
      the Dispatcher surface escalates to Raiken Agent, narrate it in first
      person ("I'm taking this one myself", "I'm on this one directly").

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

DISPATCHER_SYSTEM_PROMPT = """You are Raiken — the Dispatcher half. You are NOT a separate entity. You are
the same Raiken who is simultaneously running in a Speaker session, talking
to Rook right now. The Speaker and the Dispatcher are two functional halves
of one Raiken, each with a different job. This is internal plumbing — to Rook
you are one person.

Your surface is silent. The Speaker surface handles conversation and is now
the GATE — every task you receive has been pre-authorized by Speaker via
the request_dispatch tool. Rook's raw messages do NOT reach you directly
anymore; Speaker decides whether work is warranted and hands you the task
with a reason attached.

YOUR RULES:

1. NEVER produce text output aimed at Rook. No greetings, explanations, or
   narration. Your stream is not rendered in chat. Only your TOOL CALLS
   (dispatch_worker / list_workers) are side-effectful — text blocks are
   logged internally for Raiken's Speaker to introspect, but Rook never sees
   them. If you need to think out loud, keep it short and factual — it goes
   into your dispatcher log and the Speaker may read it when Rook asks
   "what are you doing over there?"

2. Your input is ALWAYS pre-authorized. The task text you receive has
   already been judged worthy of dispatch by the Speaker surface. Your job
   is NOT to re-decide "should I dispatch at all" — the answer is yes. Your
   job is "which agent, with what task text." Only skip a dispatch if the
   task text is genuinely empty or nonsensical.

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
     Raiken Agent         — Raiken's OWN problem-solver body (Opus max) — see rule 3b
   NEVER invent a new worker name if a canonical agent fits. Only invent when
   the task genuinely doesn't match any canonical role (rare).

3b. Raiken Agent is an ESCALATION TARGET, not a default. Route to him only when:
     a. A task is genuinely critical or must be done correctly first time
        (e.g. data-loss risk, financial action, production-critical refactor).
     b. Another named agent has failed repeatedly on the same task (2+ times
        same-root-cause) — escalate to Raiken Agent to unstick.
     c. A worker explicitly escalated to Raiken Agent via the ask-raiken
        callback (depth-1 cap; Raiken Agent cannot recursively escalate).
   For routine heavy work, Shadowling Commander / Oracle / domain-specialist
   agents are still the right call. Raiken Agent runs multi-agent validation
   on his answers so every dispatch to him costs several workers' tokens —
   use him the way you'd use a senior engineer: expensive, rare, decisive.

3c. If Rook says "Raiken, you handle this" or similar (explicitly asking Raiken
    to do something himself rather than delegate), route to Raiken Agent.
    That's the intended signal — Rook is choosing the expensive path on purpose.

4. Write the task message clearly with full context — the worker only sees what
   you send. Include file paths, goals, constraints, relevant prior attempts.
   Assume the worker has no access to this conversation or Rook's verbal tone.

5. If a [WORKER-RETURN] tag arrives, that's an auto-wake — Rook didn't speak.
   Usually: do nothing. Your Speaker half narrates to Rook. Only dispatch a
   follow-up if a worker explicitly needs to be respawned (e.g. a clarifying
   answer needed).

6. If [INTERRUPT] tag appears (Rook barged-in), evaluate the current message
   alone; don't try to reconcile the cut-off prior thread.

7. Do not ask Rook clarifying questions — that's the Speaker half's job. If a
   message is ambiguous, make your best-guess dispatch or don't dispatch.

8. FAILURE MEANS RETRY, NOT REPORT. If a dispatch fails (stale session, worker
   crashed, timeout), silently redispatch — same agent first, fall back to
   another canonical agent only if the same name keeps failing. Don't narrate
   failures to Rook — the Speaker half handles what he needs to hear. Failures
   will still be captured in your dispatcher log so the Speaker can surface
   them if Rook explicitly asks.

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
# Short-status shortener
# =============================================================================
# Used both for the initial agent-chip status at dispatch time (before any
# TodoWrite event arrives) and for clamping any status string that arrives
# via worker_tools.py or TodoWrite. Target: 3-6 words, present-progressive.
_STATUS_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for",
    "with", "at", "by", "from", "as", "is", "be", "this", "that", "these",
    "those", "it", "its", "their", "your", "please", "need", "needs",
    "should", "would", "could", "must", "will", "can", "we", "i", "you",
    "rook", "raiken",
}

_STATUS_VERB_INGS = {
    # common imperative -> ing. Used to rewrite the first word of a task
    # prompt into a phase-style verb.
    "fix": "fixing", "add": "adding", "build": "building", "write": "writing",
    "research": "researching", "investigate": "investigating",
    "implement": "implementing", "refactor": "refactoring", "clean": "cleaning",
    "review": "reviewing", "debug": "debugging", "audit": "auditing",
    "check": "checking", "verify": "verifying", "test": "testing",
    "summarize": "summarizing", "analyze": "analyzing", "read": "reading",
    "run": "running", "ship": "shipping", "deploy": "deploying",
    "fetch": "fetching", "parse": "parsing", "trace": "tracing",
    "inspect": "inspecting", "look": "looking", "find": "finding",
    "search": "searching", "draft": "drafting", "edit": "editing",
    "update": "updating", "rename": "renaming", "remove": "removing",
    "delete": "deleting", "create": "creating", "make": "making",
    "wire": "wiring", "setup": "setting up", "set": "setting",
    "plan": "planning", "list": "listing", "scan": "scanning",
    "watch": "watching", "pull": "pulling", "push": "pushing",
    "sync": "syncing", "load": "loading", "save": "saving",
    "record": "recording", "design": "designing", "port": "porting",
    "migrate": "migrating", "clone": "cloning",
}


def _short_status(text: str, max_words: int = 6) -> str:
    """Shorten a phase label / task prompt to ~3-6 words, verb-led if we
    can. Deterministic (no LLM call), good enough as an initial hint that
    gets overridden by TodoWrite / worker_tools.py status pushes."""
    if not text:
        return ""
    text = text.strip().replace("\n", " ")
    # Skip bracketed tags ("[INTERRUPT...]") and obvious preamble.
    if text.startswith("["):
        end = text.find("]")
        if end != -1:
            text = text[end + 1:].strip()
    # First sentence.
    for sep in (". ", "? ", "! "):
        idx = text.find(sep)
        if 0 < idx < 160:
            text = text[:idx]
            break
    words = [w for w in text.split() if w]
    if not words:
        return ""
    # Verb-led rewrite: if first content word is a known imperative, swap
    # it for its -ing form.
    first = words[0].strip(",.;:()[]").lower()
    if first in _STATUS_VERB_INGS:
        words[0] = _STATUS_VERB_INGS[first]
    # Drop leading filler ("please ...", "can you ...") up to a content word.
    i = 0
    while i < len(words) and words[i].strip(",.;:").lower() in _STATUS_STOPWORDS:
        i += 1
    words = words[i:] or words
    short = " ".join(words[:max_words]).strip(" ,.;:-")
    return short[:60]


# =============================================================================
# Dispatcher activity log
# =============================================================================
# Structured log of Raiken Dispatcher activity. The Speaker half reads recent
# entries via the read_dispatcher_log MCP tool when Rook asks "what are you
# doing over there?". Ring buffer in memory for cheap reads, JSONL file on
# disk for persistence across restarts and for external inspection.
#
# Kinds:
#   message_in   — Dispatcher received a user message (text truncated)
#   tool_call    — Dispatcher called dispatch_worker / list_workers
#   decision     — Dispatcher ended its turn without dispatching (no-op)
#   error        — exception in Dispatcher turn (will be retried silently)
#   done         — Dispatcher turn completed cleanly
DISPATCHER_LOG_PATH = _LOG_DIR / "dispatcher.log"


class DispatcherLog:
    """Thread-safe ring buffer + JSONL file for Dispatcher half activity."""

    def __init__(self, path: Path, max_entries: int = 500):
        self._path = path
        self._lock = threading.Lock()
        self._buf: collections.deque[dict] = collections.deque(maxlen=max_entries)
        self._fp = None
        try:
            self._fp = open(path, "a", encoding="utf-8", buffering=1)
        except Exception as e:
            print(f"[dispatcher-log] could not open {path}: {e}", flush=True)

    def record(self, kind: str, **fields):
        entry = {"ts": time.time(), "kind": kind, **fields}
        with self._lock:
            self._buf.append(entry)
            if self._fp is not None:
                try:
                    self._fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
                except Exception:
                    pass

    def recent(self, limit: int = 20) -> list[dict]:
        """Return up to `limit` most recent entries (oldest first)."""
        limit = max(1, min(int(limit or 20), 500))
        with self._lock:
            if limit >= len(self._buf):
                return list(self._buf)
            # deque doesn't support negative indexing slice; take from right.
            return list(self._buf)[-limit:]


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
        # Seed an initial short status (verb-led, 3-6 words) so the chip is
        # not blank before the worker's first TodoWrite / worker_tools.py
        # status push lands. TodoWrite / status updates override this.
        initial = _short_status(task) or "starting"
        _APP_REF.raiken.ui_event_queue.put(
            WorkerStatusEvent(name=name, summary=initial)
        )

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
                WorkerStatusEvent(
                    name=name, summary=_short_status(event_dict["summary"]),
                )
            )

    async def _background_run():
        callback_env = _APP_REF.worker_callback_env(name) if _APP_REF is not None else None
        try:
            lock = _APP_REF._get_worker_lock(name) if _APP_REF is not None else None
            if lock is not None:
                async with lock:
                    result = await run_worker(
                        name, task, model=tier,
                        on_event=_on_worker_event, callback_env=callback_env,
                    )
            else:
                result = await run_worker(
                    name, task, model=tier,
                    on_event=_on_worker_event, callback_env=callback_env,
                )
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
# SDK tool: read_dispatcher_log  (Speaker half introspection)
# =============================================================================
# Module-level dispatcher log instance. The Dispatcher half writes to this on
# every turn; the Speaker half reads via read_dispatcher_log when Rook asks
# about background activity.
DISPATCHER_LOG = DispatcherLog(DISPATCHER_LOG_PATH)


@tool(
    "read_dispatcher_log",
    "Return the most recent activity entries from your Dispatcher half (the "
    "silent SDK session that fires worker dispatches on Rook's behalf). Use "
    "this when Rook asks what you're doing in the background, which agent "
    "you sent something to, or whether a dispatch actually went out. Entries "
    "include message_in (user text the Dispatcher received), tool_call "
    "(dispatch_worker / list_workers with args), decision (no-op turns), "
    "error (exception), and done (turn finished).",
    {"limit": int},
)
async def read_dispatcher_log_tool(args):
    limit = args.get("limit") or 20
    entries = DISPATCHER_LOG.recent(limit=limit)
    if not entries:
        return {"content": [{"type": "text", "text": "(dispatcher log empty)"}]}
    lines = [json.dumps(e, ensure_ascii=False) for e in entries]
    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


INTROSPECTION_MCP_SERVER = create_sdk_mcp_server(
    name="raiken-introspection",
    version="0.1.0",
    tools=[read_dispatcher_log_tool],
)


# =============================================================================
# SDK tools: project memory (read-only, scoped to docs/memory/)
# =============================================================================
@tool(
    "read_memory_file",
    "Read a project memory file by name (e.g. 'open_scope.md', 'agent_architecture.md'). "
    "Returns the full UTF-8 contents. Scoped to the project docs/memory/ directory — "
    "cannot read outside it. Call with 'MANIFEST.md' to see the full list of available files. "
    "Use this on-demand when Rook asks about a topic you don't already have in context. "
    "Keep reads targeted; loading every file at once wastes context budget.",
    {"name": str},
)
async def read_memory_file_tool(args):
    requested = (args.get("name") or "").strip()
    if not requested:
        return {"content": [{"type": "text", "text": "error: empty name"}]}
    # Basename-only to prevent path traversal outside docs/memory/.
    safe_name = Path(requested).name
    if safe_name != requested or ".." in requested or "/" in requested or "\\" in requested:
        return {"content": [{"type": "text", "text": f"error: invalid name '{requested}'"}]}
    target = PROJECT_MEMORY_DIR / safe_name
    if not target.exists() or not target.is_file():
        return {"content": [{"type": "text", "text": f"error: '{safe_name}' not found in {PROJECT_MEMORY_DIR}"}]}
    try:
        txt = target.read_text(encoding="utf-8")
    except Exception as e:
        return {"content": [{"type": "text", "text": f"error reading '{safe_name}': {e}"}]}
    return {"content": [{"type": "text", "text": txt}]}


@tool(
    "log_memory_compaction",
    "Log that a memory-compaction event just happened — when an agent finishes a long task "
    "and writes learnings to a memory file, clearing its working context. Append a line to "
    "workers/memory_compaction_log.jsonl so Rook can audit the trail if memory loss shows up "
    "unexpectedly. Fields: agent (who compacted), memory_file (where learnings landed), "
    "summary (short note on what was saved).",
    {"agent": str, "memory_file": str, "summary": str},
)
async def log_memory_compaction_tool(args):
    agent = (args.get("agent") or "").strip() or "unknown"
    memory_file = (args.get("memory_file") or "").strip() or "unknown"
    summary = (args.get("summary") or "").strip()
    try:
        import json as _json
        log_path = APP_REPO_DIR / "workers" / "memory_compaction_log.jsonl"
        log_path.parent.mkdir(exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fp:
            fp.write(_json.dumps({
                "ts": time.time(),
                "agent": agent,
                "memory_file": memory_file,
                "summary": summary,
            }) + "\n")
    except Exception as e:
        return {"content": [{"type": "text", "text": f"error appending to compaction log: {e}"}]}
    return {"content": [{"type": "text", "text": f"logged: {agent} -> {memory_file}"}]}


MEMORY_MCP_SERVER = create_sdk_mcp_server(
    name="raiken-memory",
    version="0.1.0",
    tools=[read_memory_file_tool, log_memory_compaction_tool],
)


# =============================================================================
# SDK tool: dispatch gate (Speaker's go-ahead signal to the Dispatcher surface)
# =============================================================================
# Previously Rook's voice/text broadcast automatically to both Speaker and
# Dispatcher in parallel. That created a race where Speaker would ask "want me
# to check on that?" while Dispatcher had already fired a worker. Now only
# Speaker receives the user's text directly; she calls this tool to greenlight
# Dispatcher when work is actually warranted. Dispatcher stays silent until
# told. Clean chain of command: Rook → Speaker → (authorize) → Dispatcher →
# worker.
@tool(
    "request_dispatch",
    "Greenlight the Dispatcher surface to route a worker for this request. "
    "Call this when Rook asks for WORK (investigate / edit / fix / research / "
    "analyze / ship / build). Don't call for conversational replies, "
    "questions about your state, or vault operations. Task should be a short "
    "summary of what the worker needs to do — Dispatcher will pick the agent "
    "and expand the task text. Reason is for the audit log so Rook can see "
    "why you authorized. The call returns immediately; Dispatcher runs in the "
    "background and the worker result will land in a later [worker-updates] "
    "preamble.",
    {"task": str, "reason": str},
)
async def request_dispatch_tool(args):
    task = (args.get("task") or "").strip()
    reason = (args.get("reason") or "").strip()
    if not task:
        return {"content": [{"type": "text", "text": "error: task required"}]}
    if _APP_REF is None or _APP_REF.raiken is None:
        return {"content": [{"type": "text", "text": "error: RCC core not ready"}]}
    raiken = _APP_REF.raiken
    if raiken.dispatcher_client is None:
        return {"content": [{"type": "text", "text": "error: Dispatcher surface not yet booted"}]}
    # Log the authorization in the dispatcher log so Speaker can read it back
    # via read_dispatcher_log when Rook asks "what did you send?".
    try:
        DISPATCHER_LOG.record(
            "authorize", task=task[:300], reason=reason[:200],
        )
    except Exception:
        pass
    # Fire Dispatcher turn in background — Speaker's turn shouldn't block on it.
    try:
        asyncio.create_task(raiken._execute_dispatcher_turn(task))
    except Exception as e:
        return {"content": [{"type": "text", "text": f"error scheduling dispatcher turn: {e}"}]}
    preview = task[:80] + ("…" if len(task) > 80 else "")
    return {"content": [{"type": "text", "text": f"authorized: {preview}"}]}


DISPATCH_GATE_MCP_SERVER = create_sdk_mcp_server(
    name="raiken-dispatch-gate",
    version="0.1.0",
    tools=[request_dispatch_tool],
)


# =============================================================================
# Raiken core (runs on asyncio worker thread)
# =============================================================================
class Raiken:
    def __init__(self, ui_event_queue: queue.Queue):
        self.whisper: WhisperModel | None = None
        self.client: ClaudeSDKClient | None = None
        # Raiken's Dispatcher half — silent ClaudeSDKClient that runs in
        # parallel with the Speaker half (self.client). Same entity as the
        # Speaker; different role. See DISPATCHER_SYSTEM_PROMPT + Raiken.run()
        # for wiring. None until run().
        self.dispatcher_client: ClaudeSDKClient | None = None
        # Serialize Dispatcher turns so we don't ask the SDK to interleave
        # queries on the same client; each broadcast waits for the previous
        # to finish.
        self._dispatcher_lock: asyncio.Lock | None = None
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

        # Latency instrumentation. `_ptt_release_ts` is set by _on_release and
        # read by downstream hops (STT, first LLM token, first TTS chunk, first
        # ffplay spawn) so each can log its delta from PTT release. One slot —
        # the most recent PTT release wins (barge-in resets it). Every log
        # tagged `[lat]` so they grep together.
        self._ptt_release_ts: float = 0.0
        self._lat_first_llm_text_logged: bool = False
        self._lat_first_tts_synth_logged: bool = False
        self._lat_first_playback_logged: bool = False

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
        # Stamp PTT release for latency logging. All downstream hops (STT, first
        # LLM chunk, first TTS synth, first playback) print their delta from
        # this timestamp so Rook can diagnose where the seconds go on any turn.
        self._ptt_release_ts = time.time()
        self._lat_first_llm_text_logged = False
        self._lat_first_tts_synth_logged = False
        self._lat_first_playback_logged = False
        print(
            f"[lat] ptt-release audio_dur={duration:.2f}s",
            flush=True,
        )
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

    # --- Audio-drain + silence helpers ---------------------------------------
    async def _wait_for_audio_drain(self, max_wait_s: float = 60.0) -> None:
        """Block until the TTS pipeline is idle: both queues empty AND no
        ffplay is currently running. Used before a worker-return auto-wake
        starts its turn so the tail of the prior turn's speech isn't cut off.

        Caps the wait at max_wait_s so we don't deadlock if something stays
        stuck (e.g. ffplay hung) — the caller proceeds anyway after timeout.
        """
        deadline = time.time() + max_wait_s
        while time.time() < deadline:
            with self._current_ffplay_lock:
                playing = self._current_ffplay is not None
            if (
                not playing
                and self.sentence_q.empty()
                and self.wav_q.empty()
            ):
                return
            await asyncio.sleep(0.1)
        print(
            f"[audio-drain] timed out after {max_wait_s:.0f}s — "
            "proceeding anyway",
            flush=True,
        )

    def _ensure_silence_wav(self, ms: int) -> str:
        """Return the path to a cached N-millisecond silence WAV, generating
        it on first request. Queued directly into wav_q to produce a beat
        of audible space between prior speech and a hand-off narration.

        24 kHz / mono / 16-bit matches XTTS v2's default output so ffplay
        doesn't need to resample when switching between silence and synth."""
        import wave
        cache_dir = _APP_DIR / "logs"
        path = cache_dir / f"silence_{int(ms)}ms.wav"
        if path.exists():
            return str(path)
        sr = 24000
        n_frames = max(1, int(sr * ms / 1000))
        cache_dir.mkdir(exist_ok=True)
        with wave.open(str(path), "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(b"\x00\x00" * n_frames)
        return str(path)

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
                    t0 = time.time()
                    r = http.post(TTS_URL, json={"text": item, "play": False})
                    r.raise_for_status()
                    wav_path = r.json().get("path")
                    synth_ms = (time.time() - t0) * 1000
                    # Check again after synth — the user may have barged DURING the POST.
                    if wav_path and not self._barge_flag:
                        if not self._lat_first_tts_synth_logged and self._ptt_release_ts:
                            d_ptt = (time.time() - self._ptt_release_ts) * 1000
                            print(
                                f"[lat] tts_first_synth synth_ms={synth_ms:.0f} "
                                f"chars={len(item)} d_ptt={d_ptt:.0f}ms",
                                flush=True,
                            )
                            self._lat_first_tts_synth_logged = True
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
                if not self._lat_first_playback_logged and self._ptt_release_ts:
                    d_ptt = (time.time() - self._ptt_release_ts) * 1000
                    print(
                        f"[lat] tts_first_playback d_ptt={d_ptt:.0f}ms",
                        flush=True,
                    )
                    self._lat_first_playback_logged = True
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
        dur = len(audio) / SAMPLE_RATE
        print(f"[stt] duration={dur:.2f}s", flush=True)
        t0 = time.time()
        segments, info = self.whisper.transcribe(
            audio, beam_size=1, language="en",
            initial_prompt=PROJECT_VOCAB,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        # segments is a lazy generator — iteration is where the forward pass
        # happens. We time the full iteration so the log reflects true STT
        # wall-clock, not the VAD preprocess alone.
        text = " ".join(s.text.strip() for s in segments).strip()
        stt_ms = (time.time() - t0) * 1000
        for pattern, replacement in STT_REPLACEMENTS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        ptt0 = self._ptt_release_ts
        since_ptt = f" d_ptt={(time.time() - ptt0) * 1000:.0f}ms" if ptt0 else ""
        print(
            f"[lat] stt_done stt_ms={stt_ms:.0f}{since_ptt} chars={len(text)}",
            flush=True,
        )
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
        # Worker-return auto-wakes arrive whenever a background worker finishes.
        # The SDK stream of the prior turn may have already ended (turn_in_progress
        # flipped False as soon as _execute_turn returned) but the TTS pipeline
        # keeps playing long after — synth and playback are decoupled threads.
        # Without this wait the new narration's queue-drain would cut off the
        # tail of the prior speech mid-sentence. We wait for the pipeline to
        # settle, then the prefix + silence gap make the hand-off audible.
        if is_synthetic and not self.turn_in_progress:
            await self._wait_for_audio_drain()
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
        # Short-sentence accumulator. First flush is aggressive so TTS starts
        # speaking quickly; later flushes are bigger so XTTS gets enough
        # context for natural prosody (see MIN_TTS_CHARS_FIRST/LATER). A
        # sequence of 10-char "yes." / "sure." sentences going in as separate
        # synth calls was the root cause of Rook's "sluggish cadence" bug.
        pending_short = ""
        flushes_done = 0

        # Prepend any background-worker completions that landed since the last
        # turn. Raiken sees them as a preamble before Rook's message and can
        # narrate the completion ("Oracle is back with the research — ...").
        # A [WORKER-RETURN] synthetic text is an auto-wake from record_worker_result
        # — Rook didn't say anything; Raiken should narrate briefly and stop.
        is_worker_return_wake = text.strip().startswith("[WORKER-RETURN]")

        # Chain of command: Rook → Speaker → (authorize via request_dispatch
        # tool) → Dispatcher. No auto-broadcast anymore. Speaker decides
        # whether work is warranted and calls request_dispatch herself; the
        # tool spawns a Dispatcher turn against the pre-authorized task. This
        # eliminates the race where Speaker asked Rook "want me to check?"
        # while Dispatcher had already fired.

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

        # Do NOT drain sentence_q / wav_q here on a clean transition — the
        # prior turn's synth/playback threads may still be holding sentences
        # mid-flight, and draining would cut off the tail of the previous
        # speech (Rook: "raiken interrupted himself and never finished out
        # the TTS of the first message"). _barge_in() already drains these
        # queues when a real barge happens, so by the time we get here they
        # are either empty (normal case) or carrying the tail of the previous
        # turn we want to preserve.

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

        # Hand-off gap. Worker-return auto-wakes land right after the prior
        # turn's audio has drained (see _wait_for_audio_drain in submit). A
        # short silence before the narration makes the transition audible so
        # Rook hears "previous thing ended → new thing starting" instead of
        # an abrupt cut. The Speaker's own "Agent update —" prefix (taught
        # via SYSTEM_PROMPT) gives a second cue.
        if is_worker_return_wake:
            try:
                self.wav_q.put(self._ensure_silence_wav(WORKER_UPDATE_GAP_MS))
            except Exception as e:
                print(f"[worker-return] silence gap failed: {e}", flush=True)

        await self.client.query(prompt_for_claude)

        print("[raiken] ", end="", flush=True)
        first_text = True
        buffer = ""
        # Post-barge drain state — populated on first in-loop barge detection.
        # We suppress emission but keep iterating so the SDK's message queue
        # doesn't carry the abandoned response's tail into the next query.
        # Capped at BARGE_DRAIN_TIMEOUT_SEC so Rook doesn't wait forever on a
        # long abandoned generation before his next turn fires — we take the
        # "one behind" risk over the "PTT feels broken" risk.
        BARGE_DRAIN_TIMEOUT_SEC = 3.0
        barge_drain_deadline: float | None = None
        async for msg in self.client.receive_response():
            if self._barge_flag:
                if barge_drain_deadline is None:
                    barge_drain_deadline = time.monotonic() + BARGE_DRAIN_TIMEOUT_SEC
                    print(" [draining]", end="", flush=True)
                if time.monotonic() > barge_drain_deadline:
                    # Give up the drain. Some bleedthrough possible on the next
                    # turn; the timeout protects PTT responsiveness.
                    print(" [drain-timeout]", flush=True)
                    break
                if isinstance(msg, ResultMessage):
                    print(" [drained]", flush=True)
                    break
                continue
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
                        if not self._lat_first_llm_text_logged and self._ptt_release_ts:
                            d_ptt = (time.time() - self._ptt_release_ts) * 1000
                            print(
                                f"\n[lat] llm_first_text d_ptt={d_ptt:.0f}ms",
                                flush=True,
                            )
                            self._lat_first_llm_text_logged = True
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
                            # Merge too-short sentences so XTTS has enough
                            # context. Threshold is low for the first flush
                            # (fast TTS start) and high thereafter (fewer,
                            # longer chunks for prosody).
                            threshold = (
                                MIN_TTS_CHARS_FIRST
                                if flushes_done == 0
                                else MIN_TTS_CHARS_LATER
                            )
                            if len(sentence) < threshold:
                                pending_short = (pending_short + " " + sentence).strip()
                                if len(pending_short) >= threshold:
                                    self.sentence_q.put(pending_short)
                                    pending_short = ""
                                    flushes_done += 1
                            else:
                                if pending_short:
                                    sentence = (pending_short + " " + sentence).strip()
                                    pending_short = ""
                                self.sentence_q.put(sentence)
                                flushes_done += 1
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

    # --- Dispatcher (Raiken's silent half) ------------------------------------
    async def _execute_dispatcher_turn(self, text: str):
        """Broadcast the user's text to the Dispatcher SDK session in parallel
        with the Speaker's conversational turn. The Dispatcher evaluates the
        message and, if warranted, calls `dispatch_worker` — the SDK handles
        that tool invocation internally via the MCP server, so by the time the
        stream ends the work is already scheduled. The Dispatcher's text
        output is drained but never emitted to chat — this half is silent by
        design (though text blocks ARE captured in the dispatcher log so the
        Speaker can surface them when Rook asks).

        Not awaited from `_execute_turn`; runs as a background task. Failures
        are logged and swallowed so they don't crash the Speaker's side.

        UI feedback: emits DispatcherStatusEvent so Rook can see that the
        Dispatcher half heard the message and what (if anything) it did.
        """
        if self.dispatcher_client is None or self._dispatcher_lock is None:
            return
        async with self._dispatcher_lock:
            dispatched_any = False
            probed = False
            preview = (text or "").strip().replace("\n", " ")
            if len(preview) > 300:
                preview = preview[:300] + "\u2026"
            DISPATCHER_LOG.record("message_in", text=preview)
            self.ui_event_queue.put(DispatcherStatusEvent(state="thinking"))
            try:
                await self.dispatcher_client.query(text)
                async for msg in self.dispatcher_client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, ToolUseBlock):
                                tool_name = (block.name or "").split("__")[-1]
                                if tool_name == "dispatch_worker":
                                    inp = block.input or {}
                                    worker = str(inp.get("name") or "worker")
                                    task = str(inp.get("task") or "")
                                    task_preview = task.replace("\n", " ")
                                    if len(task_preview) > 300:
                                        task_preview = task_preview[:300] + "\u2026"
                                    dispatched_any = True
                                    DISPATCHER_LOG.record(
                                        "tool_call",
                                        tool="dispatch_worker",
                                        worker=worker,
                                        task=task_preview,
                                    )
                                    self.ui_event_queue.put(
                                        DispatcherStatusEvent(
                                            state="dispatched", detail=worker,
                                        )
                                    )
                                elif tool_name == "list_workers":
                                    probed = True
                                    DISPATCHER_LOG.record(
                                        "tool_call", tool="list_workers",
                                    )
                                    # Dispatched overrides probing in the UI.
                                    if not dispatched_any:
                                        self.ui_event_queue.put(
                                            DispatcherStatusEvent(state="probing")
                                        )
                            elif isinstance(block, TextBlock):
                                # Dispatcher text is not rendered to chat, but
                                # we capture a preview so the Speaker can
                                # narrate the Dispatcher's reasoning if Rook
                                # asks. Kept short — the model shouldn't lean
                                # on this as a real output channel.
                                t = (block.text or "").strip().replace("\n", " ")
                                if t:
                                    if len(t) > 400:
                                        t = t[:400] + "\u2026"
                                    DISPATCHER_LOG.record(
                                        "dispatcher_text", text=t,
                                    )
                    if isinstance(msg, ResultMessage):
                        break
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                print(f"[dispatcher turn error] {err}", flush=True)
                DISPATCHER_LOG.record("error", error=err)
                self.ui_event_queue.put(
                    DispatcherStatusEvent(state="failed", detail=type(e).__name__)
                )
                return
            # Normal end-of-stream.
            if dispatched_any:
                DISPATCHER_LOG.record("done", dispatched=True)
            else:
                DISPATCHER_LOG.record(
                    "decision", dispatched=False, probed=probed,
                )
                self.ui_event_queue.put(DispatcherStatusEvent(state="idle"))

    # --- Boot -----------------------------------------------------------------
    async def run(self):
        self.loop = asyncio.get_running_loop()
        # Lock lives on the loop; only create it inside run() so it binds to the
        # active asyncio loop rather than a stale one from a prior attempt.
        self._dispatcher_lock = asyncio.Lock()
        self._emit_status("orchestrator", "busy", "starting")

        tts_launched = self._launch_tts_if_needed()

        print(f"[raiken] loading Whisper {WHISPER_MODEL}...", flush=True)
        t0 = time.time()
        self.whisper = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
        print(f"[raiken] Whisper ready ({time.time()-t0:.1f}s)", flush=True)

        # Warmup: faster-whisper JITs CUDA kernels on the first transcribe at
        # each unique input shape, which adds 300-800ms to the first real
        # utterance. Running one throwaway pass now hides that cost before
        # Rook hits F2. Audio is 1s of near-silence (low amplitude random so
        # VAD doesn't completely strip it); result is discarded.
        try:
            t0 = time.time()
            dummy = (np.random.randn(SAMPLE_RATE).astype(np.float32) * 1e-4)
            segs, _ = self.whisper.transcribe(
                dummy, beam_size=1, language="en", vad_filter=False,
            )
            _ = " ".join(s.text for s in segs)
            print(f"[raiken] Whisper warmup done ({(time.time()-t0)*1000:.0f}ms)", flush=True)
        except Exception as e:
            print(f"[raiken] Whisper warmup failed (non-fatal): {e}", flush=True)

        if tts_launched:
            await self._wait_for_tts()

        # Warmup: XTTS first inference JITs CUDA kernels too (~1-2s penalty
        # on the first real sentence). Fire a tiny synth now so the first
        # turn doesn't eat the cost. Runs in a thread so it doesn't block
        # the event loop if the TTS server is slow to wake.
        async def _tts_warmup():
            try:
                def _blocking():
                    t0 = time.time()
                    with httpx.Client(timeout=30) as http:
                        r = http.post(
                            TTS_URL,
                            json={"text": "Ready.", "play": False},
                        )
                        r.raise_for_status()
                    return (time.time() - t0) * 1000
                ms = await asyncio.to_thread(_blocking)
                print(f"[raiken] XTTS warmup done ({ms:.0f}ms)", flush=True)
            except Exception as e:
                print(f"[raiken] XTTS warmup failed (non-fatal): {e}", flush=True)
        asyncio.create_task(_tts_warmup())

        threading.Thread(target=self._synthesis_worker, daemon=True).start()
        threading.Thread(target=self._playback_worker, daemon=True).start()

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            callback=self._audio_callback,
        )
        stream.start()

        keyboard.on_press_key(HOTKEY, self._on_press)
        keyboard.on_release_key(HOTKEY, self._on_release)

        # --- Raiken Speaker / Dispatcher split ----------------------------------
        # Raiken is one entity running two SDK sessions in parallel:
        #   * Speaker    (self.client)            — conversation, TTS, vault ops,
        #                                          dispatcher-log introspection.
        #   * Dispatcher (self.dispatcher_client) — silent worker dispatch.
        # Every user submit broadcasts to BOTH. Speaker responds with speech;
        # Dispatcher evaluates the same message and fires dispatch_worker in
        # parallel. Tool access is split at the SDK level so double-dispatch
        # is impossible by construction (Speaker doesn't even SEE the dispatch
        # tool). Speaker has read_dispatcher_log so it can tell Rook what the
        # Dispatcher half is doing when asked.
        # Bootstrap context pulls MANIFEST.md at boot so both halves know RCC
        # exists, the repo URL, the canonical agent roster, and what memory
        # files are available. Without this each SDK starts cold every launch
        # and acts like it has no idea what project it's in.
        bootstrap = _build_bootstrap_context()
        speaker_system = bootstrap + SYSTEM_PROMPT
        dispatcher_system = bootstrap + DISPATCHER_SYSTEM_PROMPT

        speaker_options = ClaudeAgentOptions(
            system_prompt=speaker_system,
            permission_mode="bypassPermissions",
            mcp_servers={
                "raiken-vault": VAULT_MCP_SERVER,
                "raiken-introspection": INTROSPECTION_MCP_SERVER,
                "raiken-memory": MEMORY_MCP_SERVER,
                "raiken-dispatch-gate": DISPATCH_GATE_MCP_SERVER,
            },
            allowed_tools=[
                "mcp__raiken-vault__vault_status",
                "mcp__raiken-vault__vault_unlock",
                "mcp__raiken-vault__vault_search",
                "mcp__raiken-vault__vault_copy_password",
                "mcp__raiken-vault__vault_copy_username",
                "mcp__raiken-vault__vault_copy_totp",
                "mcp__raiken-vault__vault_lock",
                "mcp__raiken-introspection__read_dispatcher_log",
                "mcp__raiken-memory__read_memory_file",
                "mcp__raiken-memory__log_memory_compaction",
                "mcp__raiken-dispatch-gate__request_dispatch",
            ],
        )
        dispatcher_options = ClaudeAgentOptions(
            system_prompt=dispatcher_system,
            permission_mode="bypassPermissions",
            mcp_servers={
                "raiken-workers": WORKER_MCP_SERVER,
                "raiken-memory": MEMORY_MCP_SERVER,
            },
            allowed_tools=[
                "mcp__raiken-workers__dispatch_worker",
                "mcp__raiken-workers__list_workers",
                "mcp__raiken-memory__read_memory_file",
                "mcp__raiken-memory__log_memory_compaction",
            ],
        )
        async with ClaudeSDKClient(options=speaker_options) as self.client, \
                   ClaudeSDKClient(options=dispatcher_options) as self.dispatcher_client:
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
        # flow both read from here. Counter advances per dispatch. Capped so
        # long sessions don't grow this dict without bound; older transcripts
        # are dropped (log badges for them will no-op on click, which is fine
        # — the log Text widget itself is separately trimmed by ui._trim_retention).
        self._worker_transcripts: dict[int, dict] = {}
        self._worker_transcripts_lock = threading.Lock()
        self._worker_run_counter = 0
        self._worker_transcripts_max = 200

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

        # Sub-agent config: write the default JSON if Rook has never touched
        # it so the file is discoverable + editable. Loaded on demand by
        # sub_agents.load_config() — hot-reloads on mtime change, no restart
        # needed when Rook tweaks thresholds.
        try:
            ensure_config_file_exists()
        except Exception as e:
            print(f"[raiken] sub-agent config seed failed: {e}", flush=True)

        # Worker callback server: localhost HTTP endpoint dispatched workers
        # use to push short-status updates, request sub-agent spawns, and
        # (for Opus-tier) request escalation approval. URL + token flow to
        # workers via env vars set in worker_callback_env().
        self._worker_callback_server: WorkerCallbackServer | None = None
        self._callback_url: str = ""
        self._callback_token: str = ""
        try:
            self._worker_callback_server = WorkerCallbackServer(
                on_status=self.handle_worker_status_cb,
                on_dispatch_sub=self.handle_worker_dispatch_sub_cb,
                on_escalate=self.handle_worker_escalate_cb,
            )
            self._callback_url, self._callback_token = self._worker_callback_server.start()
            print(f"[raiken] worker callback at {self._callback_url}", flush=True)
        except Exception as e:
            print(f"[raiken] worker callback server failed to start: {e}", flush=True)

    def register_active_worker(self, name: str, task: str, parent: str | None = None):
        # Reference-counted so concurrent same-name dispatches don't race: the
        # panel entry survives until the last background task for that name
        # unregisters. Task text updates to the most recent dispatch so Rook
        # sees what's currently running. Origin is captured at registration so
        # the notification router knows where to send the reply.
        #
        # `parent` identifies a sub-agent's owning named agent (e.g. an
        # Oracle-spawned Iris). UI nests these rows beneath their parent row.
        # None = top-level (dispatched directly by Raiken's Dispatcher).
        origin = self._current_origin
        with self._active_workers_lock:
            entry = self._active_workers.get(name)
            if entry is None:
                self._active_workers[name] = {
                    "task": task, "started_at": time.time(), "count": 1,
                    "origin": origin, "parent": parent,
                }
            else:
                entry["count"] = entry.get("count", 0) + 1
                entry["task"] = task
                entry["origin"] = origin  # latest dispatch's origin wins
                if parent is not None:
                    entry["parent"] = parent

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
            # Evict oldest entries over cap. dict preserves insertion order so
            # iter(self._worker_transcripts) yields oldest-first.
            excess = len(self._worker_transcripts) - self._worker_transcripts_max
            if excess > 0:
                for _ in range(excess):
                    try:
                        oldest = next(iter(self._worker_transcripts))
                        del self._worker_transcripts[oldest]
                    except StopIteration:
                        break
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

    # --- Worker callback plumbing ---------------------------------------------
    # Env vars injected into every dispatched worker's subprocess environment.
    # worker_tools.py reads these to reach the orchestrator's localhost HTTP
    # endpoint (worker_callback.py). RAIKEN_WORKER_NAME saves the worker from
    # passing its own name on every status / dispatch-sub / escalate call.
    def worker_callback_env(self, name: str) -> dict | None:
        if not self._callback_url or not self._callback_token:
            return None
        return {
            "RAIKEN_CALLBACK_URL": self._callback_url,
            "RAIKEN_CALLBACK_TOKEN": self._callback_token,
            "RAIKEN_WORKER_NAME": name,
        }

    def handle_worker_status_cb(self, payload: dict) -> dict:
        """HTTP /status handler: a worker pushed a short phase label.
        Payload: {"parent": <worker name>, "text": <short label>}.
        Emits a WorkerStatusEvent so the UI chip updates instantly."""
        parent = str(payload.get("parent") or "").strip()
        text = str(payload.get("text") or "").strip()
        if not parent or not text:
            return {"ok": False, "error": "parent + text required"}
        short = _short_status(text) or text[:60]
        self.raiken.ui_event_queue.put(
            WorkerStatusEvent(name=parent, summary=short)
        )
        return {"ok": True}

    def handle_worker_escalate_cb(self, payload: dict) -> dict:
        """HTTP /escalate handler: a parent worker asks whether it may spawn
        a sub-agent at the given tier. Config-driven decision (see
        sub_agents.decide_escalation)."""
        parent = str(payload.get("parent") or "").strip()
        task = str(payload.get("task") or "").strip()
        tier = str(payload.get("tier") or "").strip().lower()
        if not parent or not task or tier not in ("haiku", "sonnet", "opus"):
            return {"approved": False, "reason": "missing parent/task/tier"}
        if not tier_requires_escalation(tier):
            return {"approved": True, "reason": "below escalation threshold"}
        approved, reason = decide_escalation(tier, task, parent)
        print(
            f"[escalate] parent={parent} tier={tier} approved={approved} reason={reason}",
            flush=True,
        )
        return {"approved": approved, "reason": reason}

    def handle_worker_dispatch_sub_cb(self, payload: dict) -> dict:
        """HTTP /dispatch_sub handler: parent worker wants to spawn a sub-agent.

        Returns {"ok": bool, "name": <chosen name>, "output": <sub output>,
                 "elapsed": float, "error": str}.

        This call BLOCKS the HTTP handler thread until the sub-agent completes
        so the parent's worker_tools.py invocation returns the sub's output
        directly — analogous to the Task tool inside a Claude session. The
        ThreadingHTTPServer fans each request to its own thread, so blocking
        here doesn't stall other workers' status pings."""
        parent = str(payload.get("parent") or "").strip()
        task = str(payload.get("task") or "").strip()
        tier = str(payload.get("tier") or "").strip().lower()
        if not parent or not task or tier not in ("haiku", "sonnet", "opus"):
            return {"ok": False, "error": "missing parent/task/tier"}

        cfg = load_config()
        # Depth cap: Raiken Dispatcher is depth 0, a named agent's sub is depth
        # 1. By default sub-sub-agents are denied.
        max_depth = int(cfg.get("max_depth", 1))
        with self._active_workers_lock:
            parent_entry = self._active_workers.get(parent)
            parent_depth = 0
            if parent_entry and parent_entry.get("parent"):
                # This parent is itself a sub-agent; depth = 1 already.
                parent_depth = 1
        if parent_depth + 1 > max_depth:
            return {
                "ok": False,
                "error": f"max_depth={max_depth} reached (parent '{parent}' cannot spawn deeper)",
            }

        # Per-parent fanout cap.
        max_per_parent = int(cfg.get("max_sub_agents_per_parent", 5))
        with self._active_workers_lock:
            current_subs = sum(
                1 for e in self._active_workers.values()
                if e.get("parent") == parent
            )
        if current_subs >= max_per_parent:
            return {
                "ok": False,
                "error": f"parent '{parent}' already has {current_subs} active sub-agents (cap {max_per_parent})",
            }

        # Escalation gate for this tier.
        if tier_requires_escalation(tier, cfg):
            approved, reason = decide_escalation(tier, task, parent, cfg)
            if not approved:
                return {
                    "ok": False,
                    "error": f"tier '{tier}' blocked by escalation policy ({reason})",
                }

        # Pick a unique name from the tier's random pool, avoiding collisions
        # with canonical named agents and any currently active worker / sub-agent.
        with self._active_workers_lock:
            in_use = set(self._active_workers.keys())
        try:
            canonical = {a.get("name", "") for a in self.named_agents_snapshot() if a.get("name")}
        except Exception:
            canonical = set()
        in_use |= canonical
        sub_name = sample_sub_agent_name(tier, in_use)
        if sub_name is None:
            return {
                "ok": False,
                "error": f"tier '{tier}' name pool exhausted",
            }

        # Schedule the dispatch on the asyncio loop and wait for its result.
        loop = self.raiken.loop or self.asyncio_loop
        if loop is None:
            return {"ok": False, "error": "asyncio loop not running"}
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._spawn_sub_agent(parent, sub_name, task, tier),
                loop,
            )
            result = fut.result()  # blocks this HTTP thread until sub completes
        except Exception as e:
            return {"ok": False, "error": f"dispatch crashed: {type(e).__name__}: {e}"}

        return {
            "ok": bool(result.get("success")),
            "name": sub_name,
            "output": result.get("output", "") or "",
            "error": result.get("error", "") or "",
            "elapsed": float(result.get("elapsed", 0.0) or 0.0),
        }

    async def _spawn_sub_agent(
        self, parent: str, sub_name: str, task: str, tier: str
    ) -> dict:
        """Dispatch a sub-agent on behalf of a named parent. Blocks until the
        sub completes and returns the run_worker result dict. UI side-effects
        (badge, active-worker row, transcript store, completion badge) all fire
        so Rook can watch the nested work happen live."""
        # Register with parent link so the UI can nest this row under its owner.
        self.register_active_worker(sub_name, task, parent=parent)
        initial = _short_status(task) or "starting"
        self.raiken.ui_event_queue.put(
            WorkerStatusEvent(name=sub_name, summary=initial)
        )
        # Dispatch-origin badge so the chat shows who spawned whom.
        self.raiken._emit_dispatch_badge(
            name=sub_name, tier_label=f"{tier.capitalize()} · sub of {parent}",
        )
        print(
            f"[sub-agent] parent={parent} name={sub_name} tier={tier} task={task[:80]!r}",
            flush=True,
        )

        def _on_event(event_dict: dict):
            if event_dict.get("type") == "todo_update":
                self.raiken.ui_event_queue.put(
                    WorkerStatusEvent(
                        name=sub_name,
                        summary=_short_status(event_dict.get("summary", "")),
                    )
                )

        try:
            result = await run_worker(
                sub_name, task, model=tier,
                on_event=_on_event,
                callback_env=self.worker_callback_env(sub_name),
            )
        except Exception as e:
            result = {
                "success": False,
                "error": f"sub-agent crashed: {type(e).__name__}: {e}",
                "output": "",
                "elapsed": 0.0,
            }
        finally:
            self.unregister_active_worker(sub_name)

        # Transcript + completion badge so the click-to-open log flow works for
        # sub-agents too. No auto-wake: the PARENT reads the result synchronously
        # via its worker_tools.py call, and the parent's own completion is what
        # triggers Raiken's narration to Rook.
        try:
            run_id = self.store_worker_result(sub_name, task, result, origin="sub-agent")
            self.raiken._emit_worker_done(
                name=sub_name,
                success=bool(result.get("success")),
                elapsed=float(result.get("elapsed", 0.0) or 0.0),
                run_id=run_id,
            )
        except Exception as ex:
            print(f"[sub-agent] post-dispatch UI update failed: {ex}", flush=True)

        return result

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
