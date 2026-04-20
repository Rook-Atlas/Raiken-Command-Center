"""Sub-agent support for RCC named agents.

A named agent (Oracle, Shadowling Commander, etc.) can fan out work by
dispatching ephemeral sub-agents that render nested beneath the parent in
the UI. Sub-agents are picked by tier from a thematic random-name pool
(so two Oracles running at once don't collide and Rook can tell them
apart at a glance).

Escalation: spawning a sub-agent at or above the configured threshold
tier (default "opus") requires Speaker approval rather than firing
unconditionally. The approval policy is config-driven so Raiken can tune
without a code edit.

Config file: `config/sub_agents.json` alongside this module. Missing /
malformed config falls back to DEFAULT_CONFIG below. `load_config()` is
called every time an approval decision is made so edits to the JSON
propagate without a restart.
"""
from __future__ import annotations

import json
import random
import threading
from pathlib import Path

_APP_DIR = Path(__file__).parent
CONFIG_PATH = _APP_DIR / "config" / "sub_agents.json"


# =============================================================================
# Random-name pool (tiered by model intelligence)
# =============================================================================
# Thematically consistent with the canonical roster (Shadowling, Pyre, Cipher,
# Keeper, Scribe) — short, evocative, unique per tier so the tier is legible
# from the name alone. Haiku = scouts/simple roles. Sonnet = specialists /
# journeymen. Opus = adepts / heavy hitters (rare, require escalation by
# default).
#
# Rook: the previously "agreed" list couldn't be found in the repo. This is a
# fresh list — rename freely if you want a different palette.
HAIKU_NAMES: list[str] = [
    "Nettle", "Quill", "Sparrow", "Briar", "Lint", "Ash", "Vellum",
    "Tack", "Whisk", "Sprig", "Ember", "Iris", "Flint", "Cinder",
    "Linnet", "Wren", "Moss", "Twig", "Reed", "Dust",
]
SONNET_NAMES: list[str] = [
    "Harrow", "Vesper", "Rune", "Talon", "Lantern", "Thorne", "Kestrel",
    "Quiver", "Hollow", "Dusk", "Moth", "Valor", "Glass", "Mirth", "Ode",
    "Sable", "Wick", "Frost", "Garnet", "Tide",
]
OPUS_NAMES: list[str] = [
    "Nocturne", "Archive", "Obsidian", "Fathom", "Solace", "Verdant",
    "Crescendo", "Relic", "Onyx", "Reverie", "Phantom", "Vanguard",
    "Aegis", "Requiem", "Bastion", "Echo",
]

NAMES_BY_TIER: dict[str, list[str]] = {
    "haiku": HAIKU_NAMES,
    "sonnet": SONNET_NAMES,
    "opus": OPUS_NAMES,
}

TIER_RANK = {"haiku": 0, "sonnet": 1, "opus": 2}


def all_sub_agent_names() -> set[str]:
    return {n for pool in NAMES_BY_TIER.values() for n in pool}


def sample_sub_agent_name(tier: str, in_use: set[str]) -> str | None:
    """Return an unused random name from the requested tier's pool.

    `in_use` is the set of names currently occupied (other active sub-agents
    + every canonical named agent, so we never collide with Marl / Oracle /
    etc.). Returns None if the pool is exhausted; caller should treat that
    as a dispatch failure rather than silently collide.
    """
    pool = NAMES_BY_TIER.get(tier.lower())
    if not pool:
        return None
    free = [n for n in pool if n not in in_use]
    if not free:
        return None
    return random.choice(free)


def infer_tier_from_name(name: str) -> str | None:
    """Reverse lookup: given a sub-agent name, which tier's pool did it come
    from? Returns None for non-pool names (the canonical roster)."""
    for tier, pool in NAMES_BY_TIER.items():
        if name in pool:
            return tier
    return None


# =============================================================================
# Escalation / spawn-policy config
# =============================================================================
# escalation_threshold_tier — sub-agent dispatches at or above this tier
#   require approval. "opus" = only Opus spawns need approval. "sonnet" =
#   both Sonnet and Opus need approval. "none" = nothing needs approval.
# escalation_policy — what happens when a dispatch hits the threshold.
#   "auto_approve" = always ok (threshold effectively disabled)
#   "auto_deny"    = always denied (parent falls back to lower tier)
#   "ask_speaker"  = forwarded to Speaker for a silent yes/no (future)
# max_sub_agents_per_parent — hard ceiling so a runaway parent can't spawn
#   indefinitely.
# max_depth — 1 = sub-agents cannot spawn sub-sub-agents. Raiken Foreman
#   is depth 0; a named agent's sub-agent is depth 1. Rook's spec: "Raiken
#   Foreman should NOT dispatch sub-sub-agents. Foreman only picks the
#   top-level named agent. That agent then fans out as it sees fit." This
#   means named agents can spawn depth-1 sub-agents; we stop there unless
#   config raises it.
DEFAULT_CONFIG: dict = {
    "escalation_threshold_tier": "opus",
    "escalation_policy": "auto_deny",
    "max_sub_agents_per_parent": 5,
    "max_depth": 1,
}

_CONFIG_LOCK = threading.Lock()
_CACHED_CONFIG: dict | None = None
_CACHED_MTIME: float | None = None


def load_config() -> dict:
    """Return the current sub-agent config. Re-reads the file if its mtime
    has changed since the last load so edits propagate without a restart.
    Any missing keys fall back to DEFAULT_CONFIG values."""
    global _CACHED_CONFIG, _CACHED_MTIME
    with _CONFIG_LOCK:
        try:
            mtime = CONFIG_PATH.stat().st_mtime
        except FileNotFoundError:
            mtime = None
        if _CACHED_CONFIG is not None and mtime == _CACHED_MTIME:
            return dict(_CACHED_CONFIG)
        cfg = dict(DEFAULT_CONFIG)
        if mtime is not None:
            try:
                raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    cfg.update({k: v for k, v in raw.items() if k in DEFAULT_CONFIG})
            except Exception as e:
                print(f"[sub_agents] config load failed, using defaults: {e}", flush=True)
        _CACHED_CONFIG = cfg
        _CACHED_MTIME = mtime
        return dict(cfg)


def ensure_config_file_exists() -> None:
    """Write DEFAULT_CONFIG to disk if no config file exists yet. Lets Rook
    see + edit the file without having to know the defaults."""
    if CONFIG_PATH.exists():
        return
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(
            json.dumps(DEFAULT_CONFIG, indent=2) + "\n", encoding="utf-8"
        )
    except Exception as e:
        print(f"[sub_agents] could not seed config: {e}", flush=True)


def tier_requires_escalation(tier: str, cfg: dict | None = None) -> bool:
    """True if a dispatch at `tier` needs parent-level approval per config."""
    cfg = cfg or load_config()
    threshold = cfg.get("escalation_threshold_tier", "opus")
    if threshold == "none":
        return False
    t_rank = TIER_RANK.get(tier.lower(), 1)
    th_rank = TIER_RANK.get(threshold.lower(), 2)
    return t_rank >= th_rank


def decide_escalation(
    tier: str,
    task: str,
    parent: str,
    cfg: dict | None = None,
) -> tuple[bool, str]:
    """Apply the configured escalation policy. Returns (approved, reason).

    auto_approve — always True.
    auto_deny    — always False.
    ask_speaker  — currently behaves like auto_deny (LLM-mediated approval
                   is a future hook; the plumbing is wired but there's no
                   Speaker arbiter session yet). The reason string flags
                   this so Rook can see the default hit.
    """
    cfg = cfg or load_config()
    policy = cfg.get("escalation_policy", "auto_deny")
    if policy == "auto_approve":
        return True, "auto_approve"
    if policy == "ask_speaker":
        # TODO: wire a one-shot Claude query against a frozen arbiter prompt.
        return False, "ask_speaker (not implemented — falling back to deny)"
    return False, "auto_deny"
