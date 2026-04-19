"""Quick standalone test of worker dispatch — spawn a claude -p subprocess,
wait for result, print output. Verifies the subprocess path + auth + session
registry all work before we wire it into the SDK."""
import asyncio
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from workers import run_worker, list_workers


async def main():
    print("=== dispatch 1: create 'test-worker' and say hello ===", flush=True)
    r = await run_worker(
        "test-worker",
        "Say hello, tell me your name, and confirm this is worker session 1.",
        timeout=60,
    )
    print(f"success={r['success']} elapsed={r['elapsed']:.1f}s", flush=True)
    if r["success"]:
        print(f"output: {r['output']!r}", flush=True)
    else:
        print(f"error: {r.get('error')}", flush=True)

    print()
    print("=== dispatch 2: same worker, continuation test ===", flush=True)
    r2 = await run_worker(
        "test-worker",
        "What did I just ask you in my previous message? Keep it short.",
        timeout=60,
    )
    print(f"success={r2['success']} elapsed={r2['elapsed']:.1f}s", flush=True)
    if r2["success"]:
        print(f"output: {r2['output']!r}", flush=True)
    else:
        print(f"error: {r2.get('error')}", flush=True)

    print()
    print("=== registry ===", flush=True)
    for w in list_workers():
        print(f"  {w}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
