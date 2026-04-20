# Known Issues

Documented bugs, quirks, and quality-of-life issues in RCC. File issues here with context so they don't get lost across sessions or model changes.

---

## TTS pipeline jumbles pre-tool acknowledgments with post-tool narration

**Date observed:** 2026-04-20 (observed twice on this date)

**Priority:** P1 quality-of-life — undermines the whole purpose of the mandatory pre-task acknowledgment rule in the system prompt.

**Symptom:** When one assistant turn contains both a short pre-tool acknowledgment and a longer post-tool narration, the two speech segments get treated as continuous audio with no natural pause or sentence break. The short ack either disappears or runs together with the following sentence.

**Occurrence 1:** A short acknowledgment like "on it, pulling up what I've got stored" before a memory-file read fused into the following longer response instead of speaking cleanly on its own.

**Occurrence 2:** "on it, firing the CMMC wizard for a two sentence debrief" before a dispatch tool call fused into the post-tool sentence "authorized, he's running in the background now."

**Suspected root cause (unknown — needs investigation):** Either (a) how the TTS chunker splits the turn's text around tool-call boundaries, or (b) how the streaming text buffer flushes before/after tool calls.

**Scope note:** Documentation only. No fix attempted — bug fix will be handled separately.

---

## Pre-seeded named worker stuck on "Session ID is already in use"

**Date observed:** 2026-04-20

**Priority:** P1 — blocks dispatching to any pre-seeded worker (Marl, CMMC Wizard) until the registry is corrected.

**Symptom:** Dispatches to a pre-seeded named worker fail with `Error: Session ID <uuid> is already in use.` Survives RCC restart, desktop-app close, and full process cleanup — looks like a stale lock but isn't one.

**Root cause (verified):** Logic bug in `register_named_worker` (workers.py). When `seed_named_workers` re-seeds a pre-existing entry that has `session_created=False` in the registry, the function only updates a handful of config fields and never flips `session_created` to True. On dispatch, workers.py:427-431 sees `session_created=False` and passes `--session-id <uuid>` instead of `--resume <uuid>`. The bundled `claude.exe` checks `sessionIdExists(uuid)` (function `eH8` in the binary) and rejects creation because the `.jsonl` already exists at `~/.claude/projects/<slug>/<uuid>.jsonl` — surfacing as the "already in use" error even though no process holds it.

**How to recognize it:** In `workers/registry.json`, look for an entry where `session_id` is set, `dispatches=0`, and `session_created=false`. Both Marl and CMMC Wizard had this state on 2026-04-20.

**Important — registry is cached in RCC memory:** `_REGISTRY_CACHE` in workers.py is loaded once and never re-read from disk. Hand-editing `registry.json` while RCC is running is futile — the cache will overwrite it on the next dispatch's `_save_registry`. Real fix requires restarting RCC so `seed_named_workers` runs again.

**Fix applied:** Defensive check added to `register_named_worker` — when re-seeding an entry whose existing `session_id` matches the seed's `session_id` and `session_created` is False, flip it to True. Self-corrects on every RCC startup. Disk registry was also edited but will only stick after a restart (see cache note above).

**If it recurs:** (1) confirm RCC has been restarted since the workers.py fix landed; (2) check `workers/registry.json` for the bad state pattern; (3) if RCC is offline, edit registry.json directly — if RCC is online, just restart it.
