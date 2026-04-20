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

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
