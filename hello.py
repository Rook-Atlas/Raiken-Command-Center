"""Raiken hello-world: verify the Agent SDK works end-to-end with Max auth."""
import asyncio
import sys

# Windows cp1252 can't encode emoji/unicode; force UTF-8 on stdout.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
    SystemMessage,
    ResultMessage,
    UserMessage,
)


async def main():
    options = ClaudeAgentOptions(
        system_prompt="You are Raiken. Respond briefly for this connectivity test.",
        permission_mode="bypassPermissions",
    )

    async with ClaudeSDKClient(options=options) as client:
        print("[raiken] connected. sending test prompt...", flush=True)
        await client.query("Say hello. Confirm you are reachable and identify the model you are running on.")

        async for msg in client.receive_response():
            cls = type(msg).__name__
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        print(block.text, end="", flush=True)
            elif isinstance(msg, ResultMessage):
                print(f"\n[raiken] result received ({cls}); conversation done.", flush=True)
                break
            elif isinstance(msg, SystemMessage):
                subtype = getattr(msg, "subtype", "?")
                print(f"\n[sys] {subtype}", flush=True)
            elif isinstance(msg, UserMessage):
                pass  # ignore echoes
            else:
                print(f"\n[other {cls}]", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
