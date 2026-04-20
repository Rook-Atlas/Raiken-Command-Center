"""CLI helper that dispatched workers invoke to talk back to the RCC orchestrator.

Exposed to workers as three verb-led subcommands matching the tools
described in their permission preamble:

  status "<short text>"
    Push a phase label to the UI. Verb-led present-progressive, ~3-6 words
    ("researching web UI", "writing up findings").

  dispatch-sub --tier <haiku|sonnet|opus> "<task>"
    Spawn a sub-agent under THIS worker's name. The orchestrator picks a
    random name from the requested tier pool and returns it on stdout so
    the caller can refer to it in follow-ups. Opus (by default) triggers
    an escalation check before firing — see `escalate` below.

  escalate --tier <haiku|sonnet|opus> "<task>"
    Request permission to dispatch a sub-agent at the given tier. Returns
    JSON {"approved": bool, "reason": str}.

  ask-raiken "<question>"
    Ask Raiken Agent (his advisor surface) for guidance on something the
    caller is stuck on. BLOCKS until Raiken Agent returns; prints his
    advice on stdout. Depth-1 capped — Raiken Agent himself cannot call
    ask-raiken. Use when you've tried, failed, and want a second opinion
    BEFORE escalating the task to Rook. Raiken Agent returns ADVICE only
    — the caller continues owning the task.

  compact-memory --file <name> --summary "<note>"
    Audit-log that this agent has just compacted its working context and
    written learnings to the given memory file. Fire-and-forget (doesn't
    block). Call this AFTER you've actually edited the memory file; this
    only writes an entry to workers/memory_compaction_log.jsonl so Rook
    can trace memory changes if loss shows up later.

Transport: localhost HTTP to the orchestrator's callback server. URL and
token arrive via env vars RAIKEN_CALLBACK_URL and RAIKEN_CALLBACK_TOKEN,
plus RAIKEN_WORKER_NAME so the worker doesn't have to pass its own name
on every call.

Exit codes: 0 success, 1 argument / env error, 2 HTTP / orchestrator error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def _env(name: str, required: bool = True) -> str:
    v = os.environ.get(name, "").strip()
    if not v and required:
        print(
            f"[worker_tools] missing env var {name}; are you running under RCC?",
            file=sys.stderr,
        )
        sys.exit(1)
    return v


def _post(path: str, body: dict, timeout: float = 120.0) -> dict:
    url = _env("RAIKEN_CALLBACK_URL").rstrip("/") + path
    token = _env("RAIKEN_CALLBACK_TOKEN")
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Raiken-Token": token,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        print(f"[worker_tools] {path} -> {e.code} {body}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"[worker_tools] {path} call failed: {e}", file=sys.stderr)
        sys.exit(2)
    try:
        return json.loads(raw)
    except Exception:
        print(f"[worker_tools] {path} returned non-JSON: {raw[:200]}", file=sys.stderr)
        sys.exit(2)


def _cmd_status(args) -> int:
    text = " ".join(args.text).strip()
    if not text:
        print("[worker_tools] status requires non-empty text", file=sys.stderr)
        return 1
    parent = _env("RAIKEN_WORKER_NAME")
    _post("/status", {"parent": parent, "text": text})
    return 0


def _cmd_dispatch_sub(args) -> int:
    task = " ".join(args.task).strip()
    if not task:
        print("[worker_tools] dispatch-sub requires a task", file=sys.stderr)
        return 1
    parent = _env("RAIKEN_WORKER_NAME")
    # dispatch-sub blocks on the orchestrator side until the sub-agent
    # completes, so give it headroom past the orchestrator's own 600s worker
    # timeout before the HTTP side times out. Treat this as "wait as long as
    # a Task tool call would."
    resp = _post(
        "/dispatch_sub",
        {"parent": parent, "task": task, "tier": args.tier},
        timeout=900.0,
    )
    name = resp.get("name", "") or "?"
    if not resp.get("ok"):
        err = resp.get("error") or "unknown"
        print(
            f"[worker_tools] dispatch-sub rejected: {err}",
            file=sys.stderr,
        )
        return 2
    # Print a structured header the parent can parse out, followed by the
    # sub-agent's full output text. Parents read this on stdout like any
    # other subprocess.
    elapsed = resp.get("elapsed", 0.0)
    print(f"--- sub-agent dispatched: {name} (elapsed {elapsed:.1f}s) ---")
    output = resp.get("output", "")
    if output:
        print(output)
    return 0


def _cmd_escalate(args) -> int:
    task = " ".join(args.task).strip()
    if not task:
        print("[worker_tools] escalate requires a task", file=sys.stderr)
        return 1
    parent = _env("RAIKEN_WORKER_NAME")
    resp = _post(
        "/escalate",
        {"parent": parent, "task": task, "tier": args.tier},
    )
    print(json.dumps(resp))
    return 0


def _cmd_ask_raiken(args) -> int:
    question = " ".join(args.question).strip()
    if not question:
        print("[worker_tools] ask-raiken requires a question", file=sys.stderr)
        return 1
    parent = _env("RAIKEN_WORKER_NAME")
    # Blocks on the orchestrator side while Raiken Agent runs — give plenty
    # of headroom. Match dispatch-sub's 900s ceiling.
    resp = _post(
        "/ask_raiken",
        {"parent": parent, "question": question},
        timeout=900.0,
    )
    if not resp.get("ok"):
        err = resp.get("error") or "unknown"
        print(f"[worker_tools] ask-raiken rejected: {err}", file=sys.stderr)
        return 2
    elapsed = resp.get("elapsed", 0.0)
    print(f"--- raiken advice (elapsed {elapsed:.1f}s) ---")
    advice = resp.get("advice", "") or ""
    if advice:
        print(advice)
    return 0


def _cmd_compact_memory(args) -> int:
    if not args.file:
        print("[worker_tools] compact-memory requires --file", file=sys.stderr)
        return 1
    summary = " ".join(args.summary).strip() if args.summary else ""
    parent = _env("RAIKEN_WORKER_NAME")
    resp = _post(
        "/compact_memory",
        {"parent": parent, "file": args.file, "summary": summary},
    )
    if not resp.get("ok"):
        err = resp.get("error") or "unknown"
        print(f"[worker_tools] compact-memory rejected: {err}", file=sys.stderr)
        return 2
    print(f"logged: {parent} -> {args.file}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="worker_tools")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("status", help="Push a short phase label to the UI")
    s.add_argument("text", nargs="+")
    s.set_defaults(func=_cmd_status)

    d = sub.add_parser("dispatch-sub", help="Spawn a sub-agent under this worker")
    d.add_argument("--tier", required=True, choices=["haiku", "sonnet", "opus"])
    d.add_argument("task", nargs="+")
    d.set_defaults(func=_cmd_dispatch_sub)

    e = sub.add_parser("escalate", help="Request approval for a tiered dispatch")
    e.add_argument("--tier", required=True, choices=["haiku", "sonnet", "opus"])
    e.add_argument("task", nargs="+")
    e.set_defaults(func=_cmd_escalate)

    a = sub.add_parser("ask-raiken", help="Get advice from Raiken Agent (depth-1 capped)")
    a.add_argument("question", nargs="+")
    a.set_defaults(func=_cmd_ask_raiken)

    cm = sub.add_parser("compact-memory", help="Audit-log a memory compaction event")
    cm.add_argument("--file", required=True, help="memory filename that was updated")
    cm.add_argument("summary", nargs="*", help="short note on what was saved")
    cm.set_defaults(func=_cmd_compact_memory)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
