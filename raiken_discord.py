"""Discord bridge — lets Rook DM Raiken from his phone.

Incoming DM from a whitelisted user -> pipes into Raiken.submit(text) on the
asyncio loop with origin="discord". While the turn runs, Raiken's streaming
text chunks are captured into a buffer. When the turn ends, the buffer is
flushed back to the user as a Discord DM (chunked if >2000 chars, Discord's
limit).

Disabled by default. Opt-in via env vars:
  RAIKEN_DISCORD_BOT_TOKEN    — from https://discord.com/developers/applications
  RAIKEN_DISCORD_USER_IDS     — comma-separated Discord user IDs the bot will
                                respond to (others get a polite "not
                                authorized" reply per access_policy.md)

When active, the desktop chat UI mirrors incoming DMs as ROOK messages with a
"(via discord)" tag on the label so the desktop session sees remote activity
live. Raiken's reply goes to both the desktop UI AND the Discord DM.

Voice (Discord call) is out of scope for this first pass — needs VAD on the
Discord voice gateway + Opus encoding. Text DMs cover 90% of what Rook
actually needs remotely.
"""
from __future__ import annotations

import asyncio
import os
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main import RaikenApp

try:
    import discord
except ImportError:
    discord = None  # type: ignore


DISCORD_MSG_MAX = 2000  # hard Discord limit
REPLY_TURN_TIMEOUT_SEC = 600  # cap on how long we wait for Raiken's reply


def _chunk_for_discord(text: str, limit: int = DISCORD_MSG_MAX) -> list[str]:
    """Split a reply into <=2000 char chunks, trying to break on newlines
    or sentence boundaries rather than mid-word. Discord hard-caps at 2000
    per message so we must chunk; cosmetic quality matters less than not
    losing text."""
    text = text or ""
    if len(text) <= limit:
        return [text] if text else []
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        # Prefer to break at the last newline, then last sentence boundary,
        # then last space, then just hard-cut.
        slice_end = limit
        for sep in ("\n\n", "\n", ". ", " "):
            idx = remaining.rfind(sep, 0, limit)
            if idx > limit // 2:  # don't chunk into tiny fragments
                slice_end = idx + len(sep)
                break
        chunks.append(remaining[:slice_end].rstrip())
        remaining = remaining[slice_end:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


class DiscordBridge:
    """Glue between the Discord gateway and the RCC orchestrator.

    Runs the discord.Client on its own asyncio loop in a background thread so
    it doesn't interfere with Raiken's asyncio loop. Cross-thread calls:
    - inbound (DM -> RCC): bridge captures the DM, schedules
      `submit()` on Raiken's loop via run_coroutine_threadsafe, awaits the
      response via a per-request asyncio.Queue fed by a registered listener.
    - outbound (reply -> Discord): reply text sent from Raiken's loop via
      run_coroutine_threadsafe to the Discord client's loop.
    """

    def __init__(self, app: "RaikenApp", token: str, allowed_user_ids: set[int]):
        if discord is None:
            raise RuntimeError("discord.py not installed")
        self.app = app
        self._token = token
        self._allowed_user_ids = allowed_user_ids
        self._thread: threading.Thread | None = None
        self._client: discord.Client | None = None
        self._discord_loop: asyncio.AbstractEventLoop | None = None
        self._ready_event = threading.Event()
        self._stopped = False

    # --- Public lifecycle ---------------------------------------------------
    def start(self) -> None:
        """Spin up the discord.Client in a daemon thread. Non-blocking."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_client, name="raiken-discord", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stopped = True
        if self._client is not None and self._discord_loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._client.close(), self._discord_loop,
                )
            except Exception:
                pass

    # --- Discord thread -----------------------------------------------------
    def _run_client(self) -> None:
        # Each thread needs its own loop for discord.py.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._discord_loop = loop

        intents = discord.Intents.default()
        intents.message_content = True
        intents.dm_messages = True
        self._client = discord.Client(intents=intents)

        @self._client.event
        async def on_ready():
            user = self._client.user
            print(f"[discord] bridge online as {user} (id={user.id})", flush=True)
            self._ready_event.set()

        @self._client.event
        async def on_message(message: discord.Message):
            # Only DMs, not server channels.
            if not isinstance(message.channel, discord.DMChannel):
                return
            # Don't respond to ourselves.
            if message.author == self._client.user:
                return
            await self._handle_dm(message)

        try:
            loop.run_until_complete(self._client.start(self._token))
        except Exception as e:
            print(f"[discord] client exited: {type(e).__name__}: {e}", flush=True)
        finally:
            try:
                loop.close()
            except Exception:
                pass

    # --- DM handler ---------------------------------------------------------
    async def _handle_dm(self, message: "discord.Message") -> None:
        author_id = int(message.author.id)
        text = (message.content or "").strip()
        if not text:
            return

        # Whitelist gate. Unknown users get a polite deflect per access_policy.md.
        if author_id not in self._allowed_user_ids:
            try:
                await message.channel.send(
                    "Hi — I'm Raiken, Rook's personal orchestrator. I only take "
                    "commands from Rook. If you need to reach him, message him "
                    "directly."
                )
            except Exception:
                pass
            print(
                f"[discord] unauthorized DM from {message.author} (id={author_id})",
                flush=True,
            )
            return

        print(
            f"[discord] DM from {message.author} ({len(text)} chars): {text[:80]!r}",
            flush=True,
        )

        # Mirror the inbound message into the desktop chat UI so Rook (or
        # whoever's at the PC) sees the remote conversation in real time.
        try:
            from ui import ChatEvent
            self.app.ui_event_queue.put(
                ChatEvent(
                    role="user", text=text + "\n", done=True,
                    # log_only keeps this out of chat? No — we want it in chat.
                )
            )
        except Exception:
            pass

        # Call submit on Raiken's loop and collect the streaming reply.
        try:
            reply_text = await self._submit_and_collect(text)
        except asyncio.TimeoutError:
            reply_text = "(Raiken turn timed out; try again)"
        except Exception as e:
            print(f"[discord] submit failed: {type(e).__name__}: {e}", flush=True)
            reply_text = f"(orchestrator error: {type(e).__name__})"

        if not reply_text:
            # Empty reply usually means Raiken delegated to a worker silently
            # and hasn't finished yet. Let the user know something's in flight.
            reply_text = (
                "_(I'm on it — no immediate reply yet. Dispatched work will "
                "come back in a follow-up message.)_"
            )

        for chunk in _chunk_for_discord(reply_text):
            try:
                await message.channel.send(chunk)
            except Exception as e:
                print(f"[discord] reply send failed: {e}", flush=True)
                break

    # --- Submit + collect response ------------------------------------------
    async def _submit_and_collect(self, text: str) -> str:
        """Schedule Raiken.submit(text) on his loop, capture every 'raiken'
        chat chunk emitted during the resulting turn, return the concatenated
        reply once the turn ends.

        Threading: listener fires on Raiken's asyncio loop (wherever _emit_chat
        is called from); we need to flip the done_event which lives on THIS
        loop (Discord's). Capture the current loop in the closure and use
        call_soon_threadsafe to bridge.
        """
        raiken = self.app.raiken
        loop = raiken.loop or self.app.asyncio_loop
        if loop is None:
            return "(RCC core not ready)"

        discord_loop = asyncio.get_event_loop()
        buffer: list[str] = []
        done_event = asyncio.Event()

        def _listener(role: str, chunk: str, done: bool) -> None:
            if role != "raiken":
                return
            if chunk:
                buffer.append(chunk)
            if done:
                try:
                    discord_loop.call_soon_threadsafe(done_event.set)
                except Exception:
                    pass

        raiken.register_raiken_text_listener(_listener)
        prev_origin = self.app._current_origin
        self.app._current_origin = "discord"
        try:
            try:
                asyncio.run_coroutine_threadsafe(raiken.submit(text), loop)
            except Exception as e:
                return f"(submit crashed: {type(e).__name__}: {e})"
            try:
                await asyncio.wait_for(done_event.wait(), timeout=REPLY_TURN_TIMEOUT_SEC)
            except asyncio.TimeoutError:
                pass
        finally:
            raiken.unregister_raiken_text_listener(_listener)
            self.app._current_origin = prev_origin

        return "".join(buffer).strip()


def maybe_start_discord_bridge(app: "RaikenApp") -> "DiscordBridge | None":
    """Bootstrap helper — called from main.py during RCC startup. Returns the
    running bridge or None if disabled / misconfigured. Safe to call even
    when discord.py isn't installed."""
    if discord is None:
        print("[discord] discord.py not installed; bridge disabled", flush=True)
        return None
    token = os.environ.get("RAIKEN_DISCORD_BOT_TOKEN", "").strip()
    if not token:
        print("[discord] RAIKEN_DISCORD_BOT_TOKEN unset; bridge disabled", flush=True)
        return None
    raw_ids = os.environ.get("RAIKEN_DISCORD_USER_IDS", "").strip()
    if not raw_ids:
        print(
            "[discord] RAIKEN_DISCORD_USER_IDS unset; bridge refuses to run "
            "(would accept commands from anyone)",
            flush=True,
        )
        return None
    allowed: set[int] = set()
    for piece in raw_ids.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            allowed.add(int(piece))
        except ValueError:
            print(f"[discord] bad user id in whitelist: {piece!r}", flush=True)
    if not allowed:
        print("[discord] whitelist parsed to empty; bridge disabled", flush=True)
        return None
    bridge = DiscordBridge(app, token=token, allowed_user_ids=allowed)
    bridge.start()
    print(
        f"[discord] bridge starting (whitelist: {len(allowed)} user id(s))",
        flush=True,
    )
    return bridge
