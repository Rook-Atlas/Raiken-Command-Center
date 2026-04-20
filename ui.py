"""Raiken UI — tkinter main window + pystray tray icon.

Runs on the main thread. Communicates with the Raiken asyncio core via:
  - ui_event_queue: Raiken pushes ChatEvent / StatusEvent instances; UI polls
  - app.submit_from_ui(text): UI pushes text turns to Raiken
  - app.active_workers: dict the UI polls every second for worker panel display
"""
from __future__ import annotations

import ctypes
import os
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime, timezone
from tkinter import ttk, scrolledtext, simpledialog

import pystray
from PIL import Image, ImageDraw


def _parse_reset_time(resets_at) -> datetime | None:
    if resets_at is None:
        return None
    if isinstance(resets_at, datetime):
        return resets_at
    if isinstance(resets_at, (int, float)):
        try:
            return datetime.fromtimestamp(resets_at, tz=timezone.utc)
        except Exception:
            return None
    if isinstance(resets_at, str):
        try:
            return datetime.fromisoformat(resets_at.replace("Z", "+00:00"))
        except Exception:
            return None
    return None


def _format_reset_delta(resets_at) -> str:
    dt = _parse_reset_time(resets_at)
    if dt is None:
        return ""
    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    delta = dt - now
    secs = int(delta.total_seconds())
    if secs <= 0:
        return "resets now"
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days > 0:
        return f"resets in {days}d {hours}h"
    if hours > 0:
        return f"resets in {hours}h {minutes}m"
    return f"resets in {minutes}m"


def _progress_bar(pct: float | None, length: int = 20) -> str:
    """ASCII progress bar. Brackets + # filled + - empty — guaranteed to render."""
    if pct is None:
        return "[" + ("-" * length) + "]"
    p = max(0.0, min(100.0, float(pct)))
    filled = int(round(p / 100 * length))
    return "[" + ("#" * filled) + ("-" * (length - filled)) + "]"


def _bar_color(pct: float | None) -> str:
    if pct is None:
        return FG_MUTED
    if pct >= 85:
        return "#d9534f"
    if pct >= 65:
        return "#f0ad4e"
    return "#9cc"


def _classify_rate_limit(rl_type: str) -> str:
    """Map SDK rate-limit-type string to our known categories.

    SDK emits: 'five_hour', 'seven_day', 'seven_day_opus', 'seven_day_sonnet', 'overage'.
    """
    t = (rl_type or "").lower()
    if "week" in t or "day" in t or "seven" in t:
        return "weekly"
    if "hour" in t or "five" in t or t.startswith("5"):
        return "hourly"
    return "other"


# -----------------------------------------------------------------------------
# Theme colors
# -----------------------------------------------------------------------------
BG_MAIN = "#1e1e1e"
BG_PANEL = "#252526"
BG_INPUT = "#2d2d2d"
FG_TEXT = "#ddd"
FG_MUTED = "#888"
ACCENT = "#0e639c"


# -----------------------------------------------------------------------------
# Event types
# -----------------------------------------------------------------------------
@dataclass
class ChatEvent:
    role: str        # 'user', 'raiken', 'system', 'tool', or a worker name
    text: str
    append: bool = False  # True = append to streaming bubble
    done: bool = False
    log_only: bool = False  # True = append to Log tab only, skip main Chat tab


@dataclass
class StatusEvent:
    component: str   # 'tts', 'orchestrator', 'ptt'
    state: str       # 'up', 'down', 'busy'
    detail: str = ""


@dataclass
class WorkerDoneEvent:
    """Emitted when a dispatched worker finishes. UI inserts a clickable badge
    in the chat tab and stores the full transcript so a click opens the Log tab
    scrolled to that entry."""
    name: str
    success: bool
    elapsed: float
    run_id: int


@dataclass
class DispatchBadgeEvent:
    """Emitted when Raiken dispatches a worker. UI inserts an orange bordered badge in chat."""
    name: str
    tier_label: str


@dataclass
class PresenceEvent:
    """Updates the UI's presence indicator (active / away / maybe)."""
    state: str   # 'active' | 'maybe' | 'away'
    detail: str = ""


@dataclass
class WorkerStatusEvent:
    """Emitted when a worker's in-progress TodoWrite item changes.
    The agents panel updates in-place without waiting for the next 1-second poll."""
    name: str
    summary: str  # text of the currently in_progress todo item


@dataclass
class DispatcherStatusEvent:
    """Emitted by Raiken's Dispatcher half (the silent SDK) so the UI can
    confirm that the Dispatcher heard the user's message and show what it's
    doing. Raiken is one entity running in two modes: the Speaker (converses
    with Rook) and the Dispatcher (silently fires worker dispatches). Both
    receive every user message.

    state values:
      'thinking'    — stream opened, no tool call yet
      'dispatched'  — Dispatcher called dispatch_worker (detail = worker name)
      'idle'        — stream ended with no dispatch (heard, chose not to act)
      'failed'      — stream errored out
      'probing'     — Dispatcher called list_workers (informational)
    """
    state: str
    detail: str = ""


# -----------------------------------------------------------------------------
# Task-summary heuristic (no LLM call — just first meaningful line of prompt)
# -----------------------------------------------------------------------------
def _extract_task_summary(task: str, max_len: int = 80) -> str:
    """Return a short one-line summary from a dispatch task prompt.

    Skips blank lines, bracketed tags, and greeting-style preamble so the
    first REAL sentence describing the goal surfaces as the initial status.
    """
    if not task:
        return ""
    for line in task.splitlines():
        line = line.strip().lstrip("-•*#").strip()
        if len(line) < 12:
            continue
        if line.startswith("[") or line.lower().startswith("you are"):
            continue
        return (line[:max_len] + "…") if len(line) > max_len else line
    return (task[:max_len] + "…") if len(task) > max_len else task


# -----------------------------------------------------------------------------
# Dark title bar (Windows 10 1909+ / Windows 11)
# -----------------------------------------------------------------------------
def _apply_dark_title_bar(root: tk.Tk):
    if os.name != "nt":
        return
    try:
        root.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        value = ctypes.c_int(1)
        # Try the modern attribute first (Win 10 20H1+), fall back to legacy 19.
        for attr in (20, 19):
            res = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attr, ctypes.byref(value), ctypes.sizeof(value)
            )
            if res == 0:
                break
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Main window
# -----------------------------------------------------------------------------
class RaikenWindow:
    ROLE_COLORS = {
        "user": "#9ad",
        "raiken": "#fff",
        "system": "#888",
        "tool": "#dcd",
        "error": "#e77",
    }

    # Palette for worker bubbles — one color per distinct role, stable by hash.
    WORKER_PALETTE = [
        "#d4a373", "#a8dadc", "#ffd166", "#e29578",
        "#9ee6b4", "#c8b6ff", "#f4978e", "#83c5be",
    ]

    # Where to persist window geometry ("960x640+200+100") across launches.
    # Kept next to the app so it's easy to inspect / wipe if it ever gets
    # stuck offscreen. Saved on close/quit; loaded on __init__.
    _GEOMETRY_STATE_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "workers", "window_state.json",
    )

    @classmethod
    def _load_saved_geometry(cls) -> str | None:
        try:
            import json as _json
            with open(cls._GEOMETRY_STATE_PATH, "r", encoding="utf-8") as fp:
                data = _json.load(fp)
            geom = data.get("geometry")
            if isinstance(geom, str) and geom:
                return geom
        except Exception:
            pass
        return None

    def _save_geometry(self):
        try:
            import json as _json
            os.makedirs(os.path.dirname(self._GEOMETRY_STATE_PATH), exist_ok=True)
            geom = self.root.geometry()
            with open(self._GEOMETRY_STATE_PATH, "w", encoding="utf-8") as fp:
                _json.dump({"geometry": geom}, fp)
        except Exception:
            pass

    def __init__(self, app):
        self.app = app
        self.root = tk.Tk()
        self.root.title("Raiken Command Center")
        # Calibrate Tkinter's internal scaling factor to the actual screen DPI so
        # fonts render at their declared pt sizes rather than at 96-DPI equivalents.
        # winfo_fpixels('1i') returns pixels-per-inch for the current monitor; the
        # canonical Tk baseline is 72 pt/in.  Must be called after Tk() but before
        # any widget geometry is committed.
        try:
            actual_dpi = self.root.winfo_fpixels("1i")
            self.root.tk.call("tk", "scaling", actual_dpi / 72.0)
        except Exception:
            pass
        # Restore saved geometry ("960x640+200+100") or fall back to default.
        self.root.geometry(self._load_saved_geometry() or "960x640")
        self.root.configure(bg=BG_MAIN)
        # Window / taskbar / alt-tab icon. .ico works natively on Windows;
        # iconbitmap is quicker than iconphoto and gives the proper multi-res icon.
        try:
            ico_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "raiken.ico"
            )
            if os.path.exists(ico_path):
                self.root.iconbitmap(default=ico_path)
        except Exception:
            pass

        self._streaming_tag = None
        # Structured message log for copy/debug-dump. Capped — see _trim_retention.
        self._messages: list[dict] = []   # [{role, text, ts}]
        # Retention caps. Long sessions otherwise grow chat/log Text widgets
        # without bound; tk.Text layout and .see() cost scale with content,
        # and each embedded badge (dispatch/worker_done) is a live Tk widget
        # tree that costs layout work on every window resize. Trimming keeps
        # memory flat and resize smooth over multi-hour sessions.
        self._chat_max_lines: int = 4000
        self._log_max_lines: int = 12000
        self._messages_max: int = 800
        # Ticks between trim passes. Trim is cheap when the widget is under
        # the cap (just a line-count probe), so 1 Hz is fine; don't run it on
        # every event batch because a streaming turn doesn't change line
        # totals fast enough to matter.
        self._trim_interval_ms: int = 2000
        # Dynamically-registered chat tags for worker roles (e.g. "shadowling commander").
        self._dynamic_role_tags: set[str] = set()
        # Streaming state tracked separately for the Chat tab and the Log tab,
        # since code-block filtering may produce different bubble boundaries.
        self._log_streaming_tag: str | None = None
        # Set True while _poll_events is draining a batch. Suppresses per-event
        # configure(state)/see() in _apply_chat, _append_to_log, etc. — the
        # poll loop opens/closes widgets once for the whole batch instead.
        self._batch_updating: bool = False
        # Python-side busy flag kept in sync by _apply_status so _animate_spinner
        # can check it without a cget() Tcl roundtrip on every tick.
        self._orchestrator_busy: bool = False
        # Per-role marks in the log Text widget so badge clicks can scroll back
        # to a specific worker transcript.
        self._log_marks: dict[int, str] = {}   # run_id -> mark name
        # Dispatch badge widgets keyed by worker name — flipped in-place when done.
        self._active_dispatch_badges: dict[str, tk.Frame] = {}
        # Live per-worker status strings from TodoWrite streams. Updated on each
        # WorkerStatusEvent; cleared when the worker finishes.
        self._worker_live_status: dict[str, str] = {}
        # Singleton tooltip window (only one visible at a time).
        self._tooltip_window: tk.Toplevel | None = None
        self._setup_styles()
        self._build()
        _apply_dark_title_bar(self.root)

        self.root.protocol("WM_DELETE_WINDOW", self.hide)
        self.root.after(100, self._poll_events)
        self.root.after(1000, self._poll_workers)
        self.root.after(2000, self._poll_usage)
        self.root.after(100, self._animate_spinner)
        self.root.after(1500, self._poll_vault)
        self.root.after(self._trim_interval_ms, self._trim_retention)

    # --- ttk theming for dark scrollbars --------------------------------------
    def _setup_styles(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")  # 'clam' lets us fully customize colors
        except tk.TclError:
            pass
        style.configure(
            "Dark.Vertical.TScrollbar",
            troughcolor=BG_MAIN,
            background="#3a3a3a",
            darkcolor=BG_MAIN,
            lightcolor="#3a3a3a",
            bordercolor=BG_MAIN,
            arrowcolor=FG_TEXT,
            gripcount=0,
        )
        style.map(
            "Dark.Vertical.TScrollbar",
            background=[("active", "#555"), ("pressed", "#666")],
        )

    # --- Build widgets --------------------------------------------------------
    def _build(self):
        # --- Status strip (top row: services) --------------------------------
        status = tk.Frame(self.root, bg=BG_PANEL, height=36)
        status.pack(fill=tk.X, side=tk.TOP)
        self.status_dots: dict[str, tk.Label] = {}
        for name in ("TTS", "Orchestrator", "PTT"):
            dot = tk.Label(
                status, text=f"● {name}", fg=FG_MUTED, bg=BG_PANEL,
                font=("Segoe UI", 9, "bold"), padx=12, pady=8,
            )
            dot.pack(side=tk.LEFT)
            self.status_dots[name.lower()] = dot

        # Thinking spinner — shows only when orchestrator is busy.
        self.spinner_label = tk.Label(
            status, text="", fg="#f0ad4e", bg=BG_PANEL,
            font=("Segoe UI", 10, "bold"), padx=8, pady=8,
        )
        self.spinner_label.pack(side=tk.LEFT)
        self._spinner_frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self._spinner_idx = 0

        # Dispatcher activity chip. Raiken has two SDK "halves": the Speaker
        # (conversational, this spinner above) and the Dispatcher (silent,
        # fires workers). Sits next to the Speaker spinner so Rook can see
        # both halves at a glance. Distinctly purple palette so it can't be
        # confused with the Speaker's orange spinner. States:
        #   idle/—      gray      (dim — no turn in flight)
        #   thinking    violet    (stream opened, no tool call yet)
        #   probing     steel     (Dispatcher called list_workers)
        #   dispatched  green     (Dispatcher dispatched a worker; detail=name)
        #   failed      red       (Dispatcher stream errored)
        self.dispatcher_label = tk.Label(
            status, text="\u25C6 Dispatcher: —", fg="#666", bg=BG_PANEL,
            font=("Segoe UI", 9, "bold"), padx=10, pady=8,
        )
        self.dispatcher_label.pack(side=tk.LEFT)
        self._dispatcher_state: str = "idle"
        self._dispatcher_detail: str = ""
        # When a dispatched/idle/failed state arrives we show it for a few
        # seconds then auto-fade back to the dim "—" idle look, so the chip
        # doesn't get stuck stale between turns.
        self._dispatcher_clear_after_id: str | None = None
        # Remembered last dispatch target — surfaced as a tooltip on the chip
        # so Rook can see what was last sent even after the chip fades.
        self._dispatcher_last_target: str = ""

        # Presence indicator + manual override toggle. Three display states:
        # active (green), maybe (amber — recent but no very-recent input),
        # away (grey). The toggle cycles: auto → force-active → force-away → auto.
        self.presence_label = tk.Label(
            status, text="\u25CF away", fg="#888", bg=BG_PANEL,
            font=("Segoe UI", 9), padx=12, pady=8,
        )
        self.presence_label.pack(side=tk.LEFT)
        self.presence_toggle = tk.Button(
            status, text="auto", bg="#333", fg=FG_MUTED, relief=tk.FLAT,
            activebackground="#444", activeforeground=FG_TEXT,
            command=self._cycle_presence_override, padx=8, pady=2, bd=0,
            font=("Segoe UI", 8),
        )
        self.presence_toggle.pack(side=tk.LEFT, padx=(0, 8), pady=4)

        # Buttons pack right-to-left: pack Close first so it ends up rightmost.
        tk.Button(
            status, text="Close", bg="#c0392b", fg="white", relief=tk.FLAT,
            activebackground="#e74c3c", activeforeground="white",
            command=self.app.quit, padx=12, pady=2, bd=0,
            font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.RIGHT, padx=(4, 8), pady=4)
        tk.Button(
            status, text="Relaunch", bg="#8e44ad", fg="white", relief=tk.FLAT,
            activebackground="#9b59b6", activeforeground="white",
            command=self.app.relaunch, padx=12, pady=2, bd=0,
            font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.RIGHT, padx=4, pady=4)
        tk.Button(
            status, text="Restart TTS", bg="#333", fg=FG_TEXT, relief=tk.FLAT,
            activebackground="#444", activeforeground=FG_TEXT,
            command=self.app.restart_tts_from_ui, padx=10, pady=2, bd=0,
        ).pack(side=tk.RIGHT, padx=4, pady=4)
        # Vault button: shows current state (locked/unlocked/-) and flips it.
        # Poll refreshes the label from `vault_state_snapshot` every few seconds.
        self.vault_button = tk.Button(
            status, text="Vault: —", bg="#333", fg=FG_MUTED, relief=tk.FLAT,
            activebackground="#444", activeforeground=FG_TEXT,
            command=self._on_vault_click, padx=10, pady=2, bd=0,
            font=("Segoe UI", 9, "bold"),
        )
        self.vault_button.pack(side=tk.RIGHT, padx=4, pady=4)
        tk.Button(
            status, text="Debug", bg="#444", fg=FG_TEXT, relief=tk.FLAT,
            activebackground="#555", activeforeground=FG_TEXT,
            command=self._dump_debug_log, padx=10, pady=2, bd=0,
        ).pack(side=tk.RIGHT, padx=4, pady=4)
        tk.Button(
            status, text="Copy Last", bg="#333", fg=FG_TEXT, relief=tk.FLAT,
            activebackground="#444", activeforeground=FG_TEXT,
            command=self._copy_last_assistant, padx=10, pady=2, bd=0,
        ).pack(side=tk.RIGHT, padx=4, pady=4)

        # --- Usage strip (single row: context bar + readout + 5h reset) -----
        usage = tk.Frame(self.root, bg="#1a1a1a")
        usage.pack(fill=tk.X, side=tk.TOP)

        row_font = ("Consolas", 9)
        row_font_bold = ("Segoe UI", 9, "bold")

        tk.Label(
            usage, text="Context:", fg=FG_TEXT, bg="#1a1a1a",
            font=row_font_bold, padx=12, pady=6,
        ).pack(side=tk.LEFT)

        # Graphical progress bar (replaces the old ASCII bar).
        self._context_bar_width = 180
        self._context_bar_height = 12
        self.context_canvas = tk.Canvas(
            usage, width=self._context_bar_width, height=self._context_bar_height,
            bg="#0e0e0e", highlightthickness=1, highlightbackground="#333", bd=0,
        )
        self.context_canvas.pack(side=tk.LEFT, padx=(0, 8), pady=6)
        self._context_bar_fill = self.context_canvas.create_rectangle(
            0, 0, 0, self._context_bar_height, fill=FG_MUTED, width=0,
        )

        self.context_readout = tk.Label(
            usage, text="—", fg=FG_MUTED, bg="#1a1a1a",
            font=row_font, anchor="w",
        )
        self.context_readout.pack(side=tk.LEFT)

        # Five-hour reset countdown on the right side of the same row.
        self.hourly_reset_label = tk.Label(
            usage, text="", fg=FG_MUTED, bg="#1a1a1a",
            font=row_font, anchor="e", padx=12,
        )
        self.hourly_reset_label.pack(side=tk.RIGHT)

        # --- Body: left (chat) | right (workers panel) -----------------------
        # tk.PanedWindow provides a draggable vertical splitter. Sash position
        # is persisted to ui_state.json so it survives across sessions.
        self.body_paned = tk.PanedWindow(
            self.root, orient=tk.HORIZONTAL,
            bg="#3a3a3a", sashwidth=5, sashpad=0,
            sashcursor="sb_h_double_arrow",
            showhandle=False, relief=tk.FLAT,
        )
        self.body_paned.pack(fill=tk.BOTH, expand=True)

        # Chat area (left) — wrapped in a Notebook so Rook can flip between the
        # conversational Chat tab and the verbose Log tab. Chat shows narrative
        # + worker badges + code-block markers. Log shows everything verbatim
        # including full worker outputs and code blocks.
        chat_area = tk.Frame(self.body_paned, bg=BG_MAIN)
        self.body_paned.add(chat_area, minsize=300, stretch="always")

        style = ttk.Style(self.root)
        # The clam theme's TNotebook element draws a 3D client border via
        # lightcolor / darkcolor / bordercolor. Force them all to BG_MAIN so
        # the outline around the chat/log area disappears. `tabmargins` at 0
        # also kills residual padding around the tab row.
        style.configure(
            "Raiken.TNotebook",
            background=BG_MAIN, borderwidth=0, relief="flat",
            tabmargins=[0, 0, 0, 0], padding=0,
            lightcolor=BG_MAIN, darkcolor=BG_MAIN, bordercolor=BG_MAIN,
        )
        # Inactive tabs sit smaller and dimmer; the selected tab gets bolder
        # text, bigger padding, and a lift to make it visually dominant.
        style.configure(
            "Raiken.TNotebook.Tab",
            background="#1f1f1f", foreground=FG_MUTED,
            padding=(10, 3), borderwidth=0,
            font=("Segoe UI", 9),
            lightcolor="#1f1f1f", darkcolor="#1f1f1f", bordercolor=BG_MAIN,
        )
        style.map(
            "Raiken.TNotebook.Tab",
            background=[("selected", BG_MAIN), ("active", "#2a2a2a")],
            foreground=[("selected", "#fff"), ("active", "#ddd")],
            padding=[("selected", (18, 7))],
            font=[("selected", ("Segoe UI", 10, "bold"))],
            expand=[("selected", (1, 1, 1, 0))],
            lightcolor=[("selected", BG_MAIN)],
            darkcolor=[("selected", BG_MAIN)],
            bordercolor=[("selected", BG_MAIN), ("active", BG_MAIN)],
        )

        self.notebook = ttk.Notebook(chat_area, style="Raiken.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # --- Tab 1: Chat (conversational) --------------------------------
        chat_tab = tk.Frame(self.notebook, bg=BG_MAIN)
        self.notebook.add(chat_tab, text=" Chat ")

        self.chat = tk.Text(
            chat_tab, wrap=tk.WORD, state="disabled",
            font=("Segoe UI", 10), bg=BG_MAIN, fg=FG_TEXT,
            insertbackground=FG_TEXT, borderwidth=0, padx=12, pady=8,
            highlightthickness=0, relief="flat", selectborderwidth=0,
        )
        chat_scroll = ttk.Scrollbar(
            chat_tab, orient="vertical", command=self.chat.yview,
            style="Dark.Vertical.TScrollbar",
        )
        self.chat.configure(yscrollcommand=chat_scroll.set)
        chat_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        for role, color in self.ROLE_COLORS.items():
            self.chat.tag_configure(role, foreground=color, spacing1=6, spacing3=6)
        self.chat.tag_configure("label", foreground="#666", font=("Segoe UI", 8, "bold"))

        # Right-click context menu for the chat area.
        self._chat_menu = tk.Menu(self.chat, tearoff=0, bg=BG_PANEL, fg=FG_TEXT,
                                  activebackground=ACCENT, activeforeground="white")
        self._chat_menu.add_command(label="Copy selection",
                                    command=self._copy_selection)
        self._chat_menu.add_command(label="Copy last assistant response",
                                    command=self._copy_last_assistant)
        self._chat_menu.add_command(label="Copy last user message",
                                    command=self._copy_last_user)
        self._chat_menu.add_separator()
        self._chat_menu.add_command(label="Copy all chat",
                                    command=self._copy_all_chat)
        self._chat_menu.add_command(label="Dump to debug log",
                                    command=self._dump_debug_log)
        self.chat.bind("<Button-3>", self._show_chat_menu)

        # --- Tab 2: Log (verbose mirror) ---------------------------------
        log_tab = tk.Frame(self.notebook, bg=BG_MAIN)
        self.notebook.add(log_tab, text=" Log ")

        self.log_text = tk.Text(
            log_tab, wrap=tk.WORD, state="disabled",
            font=("Consolas", 9), bg="#141414", fg=FG_TEXT,
            insertbackground=FG_TEXT, borderwidth=0, padx=12, pady=8,
            highlightthickness=0, relief="flat", selectborderwidth=0,
        )
        log_scroll = ttk.Scrollbar(
            log_tab, orient="vertical", command=self.log_text.yview,
            style="Dark.Vertical.TScrollbar",
        )
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        for role, color in self.ROLE_COLORS.items():
            self.log_text.tag_configure(role, foreground=color, spacing1=4, spacing3=4)
        self.log_text.tag_configure("label", foreground="#666", font=("Segoe UI", 8, "bold"))
        self.log_text.tag_configure("worker_section", foreground="#c8b6ff",
                                    font=("Consolas", 9, "bold"), spacing1=8, spacing3=4)

        # Worker panel (right) — scrollable so the full agent list is reachable
        # when the window isn't tall enough to show every row.
        worker_frame = tk.Frame(self.body_paned, bg=BG_PANEL)
        self.body_paned.add(worker_frame, minsize=160, stretch="never")
        # PanedWindow manages sizing directly; no pack_propagate needed.

        # Restore saved sash position after the window is fully laid out.
        def _restore_sash():
            try:
                pos = self._load_ui_state().get("sash_x", 700)
                self.body_paned.sash_place(0, pos, 0)
            except Exception:
                pass
        self.root.after(200, _restore_sash)
        self.body_paned.bind("<ButtonRelease-1>", self._on_sash_released)

        tk.Label(
            worker_frame, text="  AGENTS", fg=FG_MUTED, bg=BG_PANEL,
            font=("Segoe UI", 8, "bold"), anchor="w",
        ).pack(fill=tk.X, pady=(8, 4))

        # Canvas + scrollbar harness. The actual rows live in an inner Frame
        # placed as a window inside the canvas; canvas yview handles scrolling.
        worker_scroll_holder = tk.Frame(worker_frame, bg=BG_PANEL)
        worker_scroll_holder.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._worker_canvas = tk.Canvas(
            worker_scroll_holder, bg=BG_PANEL, highlightthickness=0, bd=0,
        )
        worker_scrollbar = ttk.Scrollbar(
            worker_scroll_holder, orient="vertical",
            command=self._worker_canvas.yview,
            style="Dark.Vertical.TScrollbar",
        )
        self._worker_canvas.configure(yscrollcommand=worker_scrollbar.set)
        worker_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._worker_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.worker_list_container = tk.Frame(self._worker_canvas, bg=BG_PANEL)
        self._worker_window_id = self._worker_canvas.create_window(
            (0, 0), window=self.worker_list_container, anchor="nw",
        )

        # Keep scrollregion in sync with inner frame size, and match inner
        # frame width to canvas width so rows fill horizontally.
        #
        # Why trailing-edge debounce (not leading):
        # Windows fires WM_SIZE at ~60 Hz during a drag. A leading-edge
        # debounce (fire first, ignore within window, rearm) still samples
        # at roughly `1000/window` Hz and never actually coalesces — it just
        # caps the rate. With ~20 agent rows × ~7 widgets each, even a
        # 60 Hz reflow is enough Tcl/geometry work to stall the drag.
        # Cancel-and-reschedule means during an active drag NO reflow runs,
        # Tk's native widget stretching still updates the window live, and
        # the single coalesced reflow fires ~120 ms after the drag settles.
        self._worker_reflow_pending: str | None = None
        self._worker_canvas_reflow_pending: str | None = None
        self._worker_last_applied_width: int = -1
        RESIZE_DEBOUNCE_MS = 120

        def _reflow_inner():
            self._worker_reflow_pending = None
            try:
                bbox = self._worker_canvas.bbox("all")
            except tk.TclError:
                return
            if not bbox:
                return
            # Clamp scrollregion height to at least the visible canvas height
            # so when the canvas is taller than content, the scrollbar doesn't
            # think there's room to scroll above. That was the "blank space
            # above the agents list after resize" bug.
            canvas_h = max(1, self._worker_canvas.winfo_height())
            x0, y0, x1, y1 = bbox
            effective_y1 = max(y1, canvas_h)
            self._worker_canvas.configure(scrollregion=(x0, y0, x1, effective_y1))
            # If we're already at the top and the scrollregion got smaller,
            # forcibly park at top so phantom space doesn't appear above.
            if self._worker_canvas.yview()[0] <= 0.01:
                self._worker_canvas.yview_moveto(0)

        def _reflow_canvas_width():
            self._worker_canvas_reflow_pending = None
            try:
                w = self._worker_canvas.winfo_width()
            except tk.TclError:
                return
            # Skip redundant itemconfigure calls — Windows fires Configure
            # on move-without-resize, DPI ticks, and focus changes too, and
            # re-applying the same width still cascades a layout pass across
            # every nested worker-row widget.
            if w == self._worker_last_applied_width:
                return
            self._worker_last_applied_width = w
            self._worker_canvas.itemconfigure(self._worker_window_id, width=w)

        def _schedule_inner_reflow():
            if self._worker_reflow_pending is not None:
                try:
                    self.root.after_cancel(self._worker_reflow_pending)
                except Exception:
                    pass
            self._worker_reflow_pending = self.root.after(
                RESIZE_DEBOUNCE_MS, _reflow_inner
            )

        def _schedule_canvas_reflow():
            if self._worker_canvas_reflow_pending is not None:
                try:
                    self.root.after_cancel(self._worker_canvas_reflow_pending)
                except Exception:
                    pass
            self._worker_canvas_reflow_pending = self.root.after(
                RESIZE_DEBOUNCE_MS, _reflow_canvas_width
            )

        def _on_inner_configure(_e):
            _schedule_inner_reflow()

        def _on_canvas_configure(_e):
            # Only schedule the width reflow here. The width change will
            # itself fire an inner Configure, whose handler schedules the
            # scrollregion reflow — no need to double-schedule.
            _schedule_canvas_reflow()

        self.worker_list_container.bind("<Configure>", _on_inner_configure)
        self._worker_canvas.bind("<Configure>", _on_canvas_configure)

        # Mousewheel routing: bind globally but only scroll the worker canvas
        # when the pointer is actually over it (so chat scrolling is unaffected).
        self.root.bind_all(
            "<MouseWheel>", self._on_worker_mousewheel, add="+",
        )

        self._worker_rows: dict[str, tk.Frame] = {}

        # --- Input bar (bottom) ----------------------------------------------
        input_frame = tk.Frame(self.root, bg=BG_PANEL)
        input_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.input_var = tk.StringVar()
        self.entry = tk.Entry(
            input_frame, textvariable=self.input_var,
            font=("Segoe UI", 11), bg=BG_INPUT, fg=FG_TEXT,
            insertbackground=FG_TEXT, relief=tk.FLAT, highlightthickness=0,
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5), pady=10, ipady=7)
        self.entry.bind("<Return>", self._on_submit)

        tk.Button(
            input_frame, text="Send", bg=ACCENT, fg="white",
            activebackground="#1177bb", activeforeground="white",
            relief=tk.FLAT, padx=16, command=self._on_submit, bd=0,
        ).pack(side=tk.RIGHT, padx=10, pady=10, ipady=5)

        self.entry.focus_set()

    # --- UI state persistence (sash position, etc.) ---------------------------
    def _load_ui_state(self) -> dict:
        import json
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui_state.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_ui_state(self, state: dict):
        import json
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui_state.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f)
        except Exception:
            pass

    def _on_sash_released(self, _event=None):
        try:
            x, _y = self.body_paned.sash_coord(0)
            state = self._load_ui_state()
            state["sash_x"] = x
            self._save_ui_state(state)
        except Exception:
            pass

    # --- Event polling --------------------------------------------------------
    # Per-tick cap — when an assistant turn is streaming, the asyncio side can
    # enqueue hundreds of ChatEvent deltas in a single tick. Draining them all
    # synchronously blocks the Tk mainloop for tens of ms (each insert +
    # chat.see triggers a Text-widget layout pass). We process up to this many
    # per tick and yield back to the event loop; remaining events are picked
    # up on the immediate next tick. 64 is enough to clear a normal batch
    # without being felt as latency.
    _POLL_EVENTS_MAX_PER_TICK = 64

    def _poll_events(self):
        processed = 0
        had_events = False
        had_chat = False   # True if any event wrote to self.chat
        had_log = False    # True if any event wrote to self.log_text
        self._batch_updating = True
        try:
            while processed < self._POLL_EVENTS_MAX_PER_TICK:
                evt = self.app.ui_event_queue.get_nowait()
                if not had_events:
                    # Pre-open both text widgets once for the whole batch.
                    # Sub-handlers (apply_chat, append_to_log, etc.) skip their
                    # own per-event configure()/see() while _batch_updating=True.
                    self.chat.configure(state="normal")
                    self.log_text.configure(state="normal")
                had_events = True
                processed += 1
                if isinstance(evt, ChatEvent):
                    if not evt.log_only:
                        had_chat = True
                    had_log = True
                    self._apply_chat(evt)
                elif isinstance(evt, StatusEvent):
                    self._apply_status(evt)
                elif isinstance(evt, WorkerDoneEvent):
                    had_chat = True   # may add badge; always writes log
                    had_log = True
                    self._apply_worker_done(evt)
                elif isinstance(evt, DispatchBadgeEvent):
                    had_chat = True
                    self._apply_dispatch_badge(evt)
                elif isinstance(evt, WorkerStatusEvent):
                    self._apply_worker_status(evt)
                elif isinstance(evt, PresenceEvent):
                    self._apply_presence(evt)
                elif isinstance(evt, DispatcherStatusEvent):
                    self._apply_dispatcher_status(evt)
        except queue.Empty:
            pass
        finally:
            self._batch_updating = False
            if had_events:
                # One see()/configure() pair per batch instead of per-event.
                # Widgets were pre-opened on first event; close them here.
                if had_chat:
                    self.chat.see(tk.END)
                if had_log:
                    self.log_text.see(tk.END)
                # Always re-disable both (they were opened unconditionally on
                # the first event regardless of event type).
                self.chat.configure(state="disabled")
                self.log_text.configure(state="disabled")
        # Fast re-entry when there might be more work queued (cap was hit or
        # events were flowing); slower idle tick when the queue's been empty.
        # 5 ms floor (was 1 ms) gives the mainloop breathing room for input
        # and resize events without adding perceptible latency.
        # Idle at 150 ms keeps the poll rate at ~6.7 Hz.
        if processed >= self._POLL_EVENTS_MAX_PER_TICK:
            delay = 5
        elif had_events:
            delay = 30
        else:
            delay = 150
        self.root.after(delay, self._poll_events)

    def _ensure_role_tag(self, role: str):
        """Register chat/log tags for a role we haven't seen before (e.g. worker names)."""
        if role in self.ROLE_COLORS or role in self._dynamic_role_tags:
            return
        idx = hash(role) % len(self.WORKER_PALETTE)
        color = self.WORKER_PALETTE[idx]
        self.chat.tag_configure(role, foreground=color, spacing1=6, spacing3=6)
        self.log_text.tag_configure(role, foreground=color, spacing1=4, spacing3=4)
        self._dynamic_role_tags.add(role)

    def _display_role(self, role: str) -> str:
        """Map internal role keys to the label shown in chat bubbles."""
        if role == "user":
            return "ROOK"
        return role.upper()

    def _apply_chat(self, evt: ChatEvent):
        # Log tab no longer mirrors chat — it's reserved for worker reports
        # (see _apply_worker_done's collapsible cards). Rook didn't want to
        # see his own messages and Raiken's spoken replies cluttering the log.
        self._ensure_role_tag(evt.role)

        # log_only events skip the main chat (used for full worker outputs —
        # those surface in chat as a clickable badge instead via WorkerDoneEvent).
        if evt.log_only:
            return

        if not self._batch_updating:
            self.chat.configure(state="normal")
        new_bubble = self._streaming_tag != evt.role
        if new_bubble:
            ts_str = time.strftime("%H:%M:%S")
            self.chat.insert(tk.END, f"\n{self._display_role(evt.role)}   {ts_str}\n", "label")
            # Start a new structured message entry.
            self._messages.append({
                "role": evt.role,
                "text": "",
                "ts": time.time(),
            })
        if evt.text:
            self.chat.insert(tk.END, evt.text, evt.role)
            # Append to the last message's text buffer for copy/debug.
            if self._messages:
                self._messages[-1]["text"] += evt.text
        if evt.done:
            self._streaming_tag = None
            self.chat.insert(tk.END, "\n")
        else:
            self._streaming_tag = evt.role
        if not self._batch_updating:
            self.chat.see(tk.END)
            self.chat.configure(state="disabled")

    def _apply_dispatch_badge(self, evt: DispatchBadgeEvent):
        """Insert an orange dispatch card in chat when a worker is launched.
        The card flips to done/failed in-place when WorkerDoneEvent arrives."""
        if not self._batch_updating:
            self.chat.configure(state="normal")
        self.chat.insert(tk.END, "\n")
        bg = "#1e1a10"
        badge = tk.Frame(
            self.chat, bg=bg, cursor="hand2", padx=8, pady=6,
            relief=tk.FLAT, highlightthickness=1, highlightbackground="#b06820",
        )
        # Row 1: glyph + "worker" label + agent name
        row1 = tk.Frame(badge, bg=bg)
        row1.pack(fill=tk.X)
        tk.Label(row1, text="\u2197", fg="#c87020", bg=bg, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0, 5))
        tk.Label(row1, text="worker", fg="#777", bg=bg, font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(row1, text=evt.name, fg="#e89030", bg=bg, font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        # Row 2: tier label + live running indicator
        row2 = tk.Frame(badge, bg=bg)
        row2.pack(fill=tk.X, pady=(2, 0))
        tk.Label(row2, text=f"  {evt.tier_label}", fg="#a06820", bg=bg, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        status_lbl = tk.Label(row2, text="  \u25cf running", fg="#f0ad4e", bg=bg, font=("Segoe UI", 8))
        status_lbl.pack(side=tk.LEFT, padx=(8, 0))

        # Handles for in-place completion flip (wired by _apply_worker_done).
        badge._status_lbl = status_lbl
        badge._dispatch_run_id = None
        badge._dispatch_done = False

        def _click(_e=None):
            if badge._dispatch_run_id is not None:
                self._jump_to_log_mark(badge._dispatch_run_id)

        def _hi(_e, f=badge):
            f.configure(highlightbackground="#d08030")

        def _lo(_e, f=badge):
            f.configure(highlightbackground="#3a3a3a" if badge._dispatch_done else "#b06820")

        def _bind_tree(w):
            w.bind("<Button-1>", _click)
            w.bind("<Enter>", _hi)
            w.bind("<Leave>", _lo)
            for child in w.winfo_children():
                _bind_tree(child)

        _bind_tree(badge)

        self.chat.window_create(tk.END, window=badge)
        self.chat.insert(tk.END, "\n")
        if not self._batch_updating:
            self.chat.see(tk.END)
            self.chat.configure(state="disabled")

        # Register for in-place completion flip.
        self._active_dispatch_badges[evt.name] = badge

    # --- Worker status classification (heuristic from output text) ----------
    _WORKER_STATUS_NEEDS_INPUT_MARKERS = (
        "need your input", "need input from", "please grant", "please allow",
        "please approve", "please confirm", "please verify", "needs approval",
        "permission required", "needs your okay", "waiting for you to",
        "could you confirm", "let me know if", "please add", "please provide",
        "i need you to", "requires your", "awaiting your",
    )
    _WORKER_STATUS_NOOP_MARKERS = (
        "no changes needed", "no-op", "already done", "nothing to do",
    )
    _WORKER_STATUS_NEEDS_INFO_MARKERS = (
        "need more info", "need more context", "need clarification",
        "ambiguous", "unclear which", "could you clarify",
    )

    def _classify_worker_status(self, success: bool, output: str, error: str) -> tuple[str, str]:
        """Derive a short status label + color from the worker's outcome.
        Used as the headline on the log card so Rook can scan results at a
        glance without expanding each one."""
        if not success:
            return ("failed", "#d9534f")
        out_lower = (output or "").lower()
        for m in self._WORKER_STATUS_NEEDS_INPUT_MARKERS:
            if m in out_lower:
                return ("needs input from Rook", "#f0ad4e")
        for m in self._WORKER_STATUS_NEEDS_INFO_MARKERS:
            if m in out_lower:
                return ("needs more info", "#f0ad4e")
        for m in self._WORKER_STATUS_NOOP_MARKERS:
            if m in out_lower:
                return ("no-op", "#888")
        return ("succeeded", "#5cb85c")

    def _insert_worker_card_in_log(self, evt: "WorkerDoneEvent", result: dict | None):
        """Append a collapsible worker-result card to the Log tab. Header shows
        agent name + status; body (hidden by default) holds task + output."""
        task = (result.get("task") or "").strip() if result else ""
        output = (result.get("output") or "").strip() if result else ""
        error = (result.get("error") or "").strip() if result else ""
        status_label, status_color = self._classify_worker_status(
            evt.success, output, error
        )

        card_bg = "#1f1f1f"
        body_bg = "#141414"
        border_default = "#2d2d2d"
        border_hover = "#555"

        card = tk.Frame(
            self.log_text, bg=card_bg, padx=0, pady=0,
            relief=tk.FLAT, highlightthickness=1, highlightbackground=border_default,
        )

        header = tk.Frame(card, bg=card_bg, cursor="hand2")
        header.pack(fill=tk.X)
        glyph = tk.Label(
            header, text="\u25B8", fg="#888", bg=card_bg,
            font=("Segoe UI", 10, "bold"), padx=8, pady=4,
        )
        glyph.pack(side=tk.LEFT)
        tk.Label(
            header, text=evt.name, fg="#ddd", bg=card_bg,
            font=("Segoe UI", 10, "bold"), pady=4,
        ).pack(side=tk.LEFT)
        tk.Label(
            header, text="  -  Task Reported", fg="#888", bg=card_bg,
            font=("Segoe UI", 9), pady=4,
        ).pack(side=tk.LEFT)
        tk.Label(
            header, text=f"  [{status_label}]", fg=status_color, bg=card_bg,
            font=("Segoe UI", 9, "bold"), pady=4,
        ).pack(side=tk.LEFT, padx=(6, 0))
        tk.Label(
            header, text=f"{evt.elapsed:.1f}s", fg="#666", bg=card_bg,
            font=("Segoe UI", 8), padx=10, pady=4,
        ).pack(side=tk.RIGHT)

        body = tk.Frame(card, bg=body_bg, padx=10, pady=8)
        # Body sections — built once; visibility toggled on click.
        if task:
            tk.Label(
                body, text="TASK", fg="#666", bg=body_bg,
                font=("Segoe UI", 8, "bold"), anchor="w",
            ).pack(fill=tk.X)
            task_w = tk.Text(
                body, wrap=tk.WORD, bg=body_bg, fg="#9ad",
                font=("Consolas", 9), borderwidth=0, highlightthickness=0,
                width=80, height=min(12, max(2, task.count("\n") + 2)),
                padx=0, pady=0,
            )
            task_w.insert("1.0", task)
            task_w.configure(state="disabled")
            task_w.pack(fill=tk.X, pady=(2, 8))

        section_label = "OUTPUT" if evt.success else "ERROR"
        body_text = output if evt.success else (error or "(no details)")
        tk.Label(
            body, text=section_label, fg="#666", bg=body_bg,
            font=("Segoe UI", 8, "bold"), anchor="w",
        ).pack(fill=tk.X)
        body_w = tk.Text(
            body, wrap=tk.WORD, bg=body_bg,
            fg=("#ddd" if evt.success else "#e77"),
            font=("Consolas", 9), borderwidth=0, highlightthickness=0,
            width=80, height=min(30, max(3, body_text.count("\n") + 2)),
            padx=0, pady=0,
        )
        body_w.insert("1.0", body_text)
        body_w.configure(state="disabled")
        body_w.pack(fill=tk.X, pady=(2, 0))

        expanded = [False]

        def _toggle(_e=None):
            if expanded[0]:
                body.pack_forget()
                glyph.configure(text="\u25B8")
            else:
                body.pack(fill=tk.X)
                glyph.configure(text="\u25BE")
            expanded[0] = not expanded[0]

        def _hi(_e=None):
            card.configure(highlightbackground=border_hover)

        def _lo(_e=None):
            card.configure(highlightbackground=border_default)

        # Bind on header + every label inside it so any click on the row toggles.
        header.bind("<Button-1>", _toggle)
        header.bind("<Enter>", _hi)
        header.bind("<Leave>", _lo)
        for child in header.winfo_children():
            child.bind("<Button-1>", _toggle)
            child.bind("<Enter>", _hi)
            child.bind("<Leave>", _lo)

        # Embed the card in the log Text widget.
        if not self._batch_updating:
            self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, "\n")
        # Mark for badge-jump scrollto.
        mark_name = f"worker_{evt.run_id}"
        try:
            self.log_text.mark_set(mark_name, "end-1c")
            self.log_text.mark_gravity(mark_name, "left")
            self._log_marks[evt.run_id] = mark_name
        except Exception:
            pass
        self.log_text.window_create(tk.END, window=card)
        self.log_text.insert(tk.END, "\n")
        if not self._batch_updating:
            self.log_text.see(tk.END)
            self.log_text.configure(state="disabled")

        # Failures auto-expand so Rook sees the error without clicking.
        if not evt.success:
            _toggle()

    def _apply_worker_done(self, evt: WorkerDoneEvent):
        """Insert a compact clickable badge in chat AND a collapsible card in
        the Log tab. Click the chat badge → switch to Log tab + scroll to the
        card. Click the card header → expand/collapse the body."""
        self._ensure_role_tag(evt.name)
        result = self.app.get_worker_result(evt.run_id) if hasattr(self.app, "get_worker_result") else None
        self._insert_worker_card_in_log(evt, result)

        # Clear live status so the panel shows "done" state on next refresh.
        self._worker_live_status.pop(evt.name, None)

        # Flip the existing dispatch card in-place (preferred); fall back to a
        # new standalone done badge only if the dispatch card is gone.
        status_color = "#5cb85c" if evt.success else "#d9534f"
        status_glyph_ui = "\u2713" if evt.success else "\u2717"
        run_id_captured = evt.run_id

        flipped = False
        existing = self._active_dispatch_badges.pop(evt.name, None)
        if existing is not None:
            try:
                existing._status_lbl.configure(
                    text=f"  {status_glyph_ui} done  {evt.elapsed:.1f}s",
                    fg=status_color,
                )
                existing.configure(highlightbackground="#3a3a3a")
                existing._dispatch_run_id = run_id_captured
                existing._dispatch_done = True

                def _open(_e=None, rid=run_id_captured):
                    self._jump_to_log_mark(rid)

                def _hi(_e, f=existing):
                    f.configure(highlightbackground="#888")

                def _lo(_e, f=existing):
                    f.configure(highlightbackground="#3a3a3a")

                def _rebind(w):
                    w.bind("<Button-1>", _open)
                    w.bind("<Enter>", _hi)
                    w.bind("<Leave>", _lo)
                    for child in w.winfo_children():
                        _rebind(child)

                _rebind(existing)
                flipped = True
            except Exception:
                pass

        if not flipped:
            # Dispatch card not in map — insert a standalone done badge.
            if not self._batch_updating:
                self.chat.configure(state="normal")
            self.chat.insert(tk.END, "\n")
            bg = "#252526"
            badge = tk.Frame(
                self.chat, bg=bg, cursor="hand2", padx=8, pady=4,
                relief=tk.FLAT, highlightthickness=1, highlightbackground="#3a3a3a",
            )
            tk.Label(badge, text="\u2699", fg=FG_MUTED, bg=bg, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0, 5))
            tk.Label(badge, text=evt.name, fg="#ddd", bg=bg, font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
            tk.Label(badge, text=f"  {status_glyph_ui}  {evt.elapsed:.1f}s",
                     fg=status_color, bg=bg, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(6, 0))
            tk.Label(badge, text="  view in Log \u2192", fg="#555", bg=bg, font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(8, 0))

            def _open(_e=None, rid=run_id_captured):
                self._jump_to_log_mark(rid)

            def _hi(_e, f=badge):
                f.configure(highlightbackground="#888")

            def _lo(_e, f=badge):
                f.configure(highlightbackground="#3a3a3a")

            badge.bind("<Button-1>", _open)
            badge.bind("<Enter>", _hi)
            badge.bind("<Leave>", _lo)
            for child in badge.winfo_children():
                child.bind("<Button-1>", _open)

            self.chat.window_create(tk.END, window=badge)
            self.chat.insert(tk.END, "\n")
            if not self._batch_updating:
                self.chat.see(tk.END)
                self.chat.configure(state="disabled")

    def _jump_to_log_mark(self, run_id: int):
        """Switch to the Log tab and scroll to the given worker's mark."""
        try:
            self.notebook.select(1)  # Log tab
        except Exception:
            pass
        mark = self._log_marks.get(run_id)
        if not mark:
            return
        try:
            self.log_text.see(mark)
        except Exception:
            pass

    # --- Tooltip ------------------------------------------------------------------
    def _show_tooltip(self, widget: tk.Widget, text: str):
        """Show a floating tooltip below widget. Replaces any previous tooltip."""
        self._hide_tooltip()
        if not text:
            return
        try:
            x = widget.winfo_rootx() + 8
            y = widget.winfo_rooty() + widget.winfo_height() + 2
            self._tooltip_window = tk.Toplevel(self.root)
            self._tooltip_window.wm_overrideredirect(True)
            self._tooltip_window.wm_geometry(f"+{x}+{y}")
            tk.Label(
                self._tooltip_window, text=text,
                bg="#2d2d2d", fg="#ddd", relief="solid", borderwidth=1,
                font=("Segoe UI", 8), wraplength=350, justify="left",
                padx=6, pady=3,
            ).pack()
        except Exception:
            self._tooltip_window = None

    def _hide_tooltip(self, _event=None):
        if self._tooltip_window is not None:
            try:
                self._tooltip_window.destroy()
            except Exception:
                pass
            self._tooltip_window = None

    # --- Live worker status (TodoWrite stream) ---------------------------------
    def _apply_worker_status(self, evt: WorkerStatusEvent):
        """Store the latest in-progress summary and update the panel row immediately."""
        self._worker_live_status[evt.name] = evt.summary
        self._update_worker_row_status(evt.name, evt.summary)

    def _update_worker_row_status(self, name: str, summary: str):
        """Push a live status string into an existing worker row without waiting
        for the next 1-second _poll_workers cycle."""
        row = self._worker_rows.get(name)
        if row is None:
            return
        preview = (summary[:80] + "…") if len(summary) > 80 else summary
        try:
            row.task_lbl.configure(
                text=preview, fg="#aaa", font=("Segoe UI", 8, "italic"),
            )
            row.task_lbl._full_text = summary
            if not row.task_lbl.winfo_ismapped():
                row.task_lbl.pack(fill=tk.X)
        except Exception:
            pass

    def _apply_presence(self, evt: PresenceEvent):
        """Update the presence indicator (status strip). 3-tier states:
        active (at PC), idle (2-15 min absence), away (15+ min)."""
        if not hasattr(self, "presence_label") or self.presence_label is None:
            return
        colors = {"active": "#5cb85c", "idle": "#f0ad4e", "away": "#888"}
        label = {"active": "here", "idle": "nearby", "away": "away"}.get(
            evt.state, evt.state
        )
        txt = f"\u25CF {label}"
        if evt.detail:
            txt = f"{txt} \u2014 {evt.detail}"
        self.presence_label.configure(
            fg=colors.get(evt.state, FG_MUTED), text=txt
        )

    def _apply_dispatcher_status(self, evt: "DispatcherStatusEvent"):
        """Update the Dispatcher activity chip. Confirms to Rook that
        Raiken's Dispatcher half heard the message (it runs silently, so
        without this the Dispatcher side gives no visible feedback).
        Purple-family palette so it's distinct from the Speaker's orange
        spinner."""
        if not hasattr(self, "dispatcher_label") or self.dispatcher_label is None:
            return
        self._dispatcher_state = evt.state
        self._dispatcher_detail = evt.detail or ""

        if evt.state == "thinking":
            txt = "\u25C6 Dispatcher: listening\u2026"
            fg = "#a78bfa"   # violet
            sticky = True
        elif evt.state == "probing":
            txt = "\u25C6 Dispatcher: checking roster\u2026"
            fg = "#8aa7d8"   # steel blue
            sticky = True
        elif evt.state == "dispatched":
            name = (evt.detail or "worker").strip()
            self._dispatcher_last_target = name
            txt = f"\u25C6 Dispatcher \u2192 {name}"
            fg = "#5cb85c"   # green
            sticky = False
        elif evt.state == "failed":
            extra = f" ({evt.detail})" if evt.detail else ""
            txt = f"\u25C6 Dispatcher: failed{extra}"
            fg = "#d9534f"   # red
            sticky = False
        else:  # idle / fallback
            txt = "\u25C6 Dispatcher: heard, no-op"
            fg = "#888"
            sticky = False

        self.dispatcher_label.configure(text=txt, fg=fg)
        # Tooltip text: remembers the last dispatch target across fade-outs.
        self.dispatcher_label._full_text = (
            f"last dispatch: {self._dispatcher_last_target}"
            if self._dispatcher_last_target
            else "Raiken Dispatcher \u2014 silent worker dispatch half"
        )

        # Cancel any pending auto-clear from a previous turn.
        if self._dispatcher_clear_after_id is not None:
            try:
                self.root.after_cancel(self._dispatcher_clear_after_id)
            except Exception:
                pass
            self._dispatcher_clear_after_id = None

        # Non-sticky terminal states fade back to dim after a few seconds
        # so the chip isn't stuck showing stale info between turns.
        if not sticky:
            self._dispatcher_clear_after_id = self.root.after(
                6000, self._clear_dispatcher_label
            )

    def _clear_dispatcher_label(self):
        self._dispatcher_clear_after_id = None
        if not hasattr(self, "dispatcher_label") or self.dispatcher_label is None:
            return
        self._dispatcher_state = "idle"
        self._dispatcher_detail = ""
        if self._dispatcher_last_target:
            txt = f"\u25C6 Dispatcher: \u2014  (last: {self._dispatcher_last_target})"
        else:
            txt = "\u25C6 Dispatcher: \u2014"
        self.dispatcher_label.configure(text=txt, fg="#666")

    def _cycle_presence_override(self):
        """Rotate the manual presence override: auto \u2192 active \u2192 away \u2192 auto.
        Manual overrides win against the idle detector."""
        cur = getattr(self, "_presence_override", "auto")
        nxt = {"auto": "active", "active": "away", "away": "auto"}.get(cur, "auto")
        self._presence_override = nxt
        colors = {"auto": FG_MUTED, "active": "#5cb85c", "away": "#888"}
        self.presence_toggle.configure(text=nxt, fg=colors.get(nxt, FG_MUTED))
        if hasattr(self.app, "set_presence_override"):
            self.app.set_presence_override(nxt)

    # --- Vault button ---------------------------------------------------------
    def _poll_vault(self):
        """Refresh the Vault button label + color from the app's snapshot."""
        try:
            status = self.app.vault_state_snapshot() if hasattr(self.app, "vault_state_snapshot") else {}
        except Exception:
            status = {}
        state = status.get("state", "unknown")
        # Trust the session-key flag over `bw status` for visible unlocked-ness —
        # bw's "unlocked" can lag our in-memory session by a few seconds.
        if status.get("in_memory_unlocked"):
            state = "unlocked"
        label_map = {
            "unlocked": ("Vault: unlocked", "#5cb85c"),
            "locked":   ("Vault: locked",   "#d9534f"),
            "unauthenticated": ("Vault: not logged in", "#888"),
            "cli-missing": ("Vault: no CLI", "#888"),
            "error": ("Vault: error", "#d9534f"),
        }
        text, color = label_map.get(state, (f"Vault: {state}", FG_MUTED))
        if hasattr(self, "vault_button") and self.vault_button is not None:
            # Dirty-check — vault state rarely changes, and a no-op .configure()
            # still costs a Tcl roundtrip per second.
            sig = (text, color)
            if sig != getattr(self, "_vault_button_sig", None):
                self._vault_button_sig = sig
                self.vault_button.configure(text=text, fg=color)
        # 2 s is plenty — the underlying `bw status` poll only runs every 30 s,
        # so sub-second refresh here buys nothing but Tcl calls.
        self.root.after(2000, self._poll_vault)

    def _on_vault_click(self):
        """Flip vault state: unlock if locked, lock if unlocked. Unauth'd state
        surfaces a message in chat rather than trying to auto-login (that's a
        manual CLI step the first time)."""
        try:
            status = self.app.vault_state_snapshot()
        except Exception:
            status = {}
        if status.get("in_memory_unlocked"):
            if hasattr(self.app, "lock_vault_from_ui"):
                self.app.lock_vault_from_ui()
            return
        if hasattr(self.app, "unlock_vault_from_ui"):
            # Run the unlock flow on a worker thread so the prompt dialog
            # doesn't block the Tk main loop.
            threading.Thread(
                target=self.app.unlock_vault_from_ui, daemon=True,
                name="vault-unlock-ui",
            ).start()

    # --- Clipboard / debug helpers -------------------------------------------
    def _copy_to_clipboard(self, text: str):
        if not text:
            return
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.root.update()
        except Exception:
            pass

    def _copy_selection(self):
        try:
            sel = self.chat.get("sel.first", "sel.last")
            self._copy_to_clipboard(sel)
        except tk.TclError:
            pass

    def _last_message(self, role: str) -> str:
        for m in reversed(self._messages):
            if m["role"] == role and m["text"].strip():
                return m["text"]
        return ""

    def _copy_last_assistant(self):
        # Raiken emits his turns with role "raiken" (not "assistant") per the
        # rename in round A of the UI refactor. Searching "assistant" always
        # missed and silently copied an empty string.
        self._copy_to_clipboard(self._last_message("raiken").strip())

    def _copy_last_user(self):
        self._copy_to_clipboard(self._last_message("user").strip())

    def _copy_all_chat(self):
        # Snapshot the list before iterating — the asyncio thread appends to
        # self._messages concurrently, so iterating the live list could skip
        # entries or blow up under rare timing.
        msgs = list(self._messages)
        lines = [f"[{m['role'].upper()}] {m['text'].strip()}" for m in msgs]
        self._copy_to_clipboard("\n\n".join(lines))

    def _show_chat_menu(self, event):
        try:
            self._chat_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._chat_menu.grab_release()

    def _dump_debug_log(self):
        """Overwrite logs/last_convo.txt with the full structured chat."""
        try:
            import pathlib
            p = pathlib.Path(__file__).parent / "logs" / "last_convo.txt"
            p.parent.mkdir(exist_ok=True)
            lines = ["=== Raiken chat debug dump ==="]
            lines.append(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"messages: {len(self._messages)}")
            lines.append("")
            for m in self._messages:
                ts = time.strftime('%H:%M:%S', time.localtime(m.get('ts', 0)))
                lines.append(f"[{ts}] [{m['role'].upper()}]")
                lines.append(m['text'].rstrip())
                lines.append("")
            p.write_text("\n".join(lines), encoding="utf-8")
            self._flash_status_text(f"Chat dumped to {p}")
        except Exception as e:
            self._flash_status_text(f"Debug dump failed: {e}")

    def _flash_status_text(self, msg: str):
        """Briefly show a message by inserting a system-style line in chat."""
        self.chat.configure(state="normal")
        self.chat.insert(tk.END, f"\n[debug] {msg}\n", "system")
        self.chat.see(tk.END)
        self.chat.configure(state="disabled")

    def _apply_status(self, evt: StatusEvent):
        key = evt.component.lower()
        if key not in self.status_dots:
            return
        colors = {"up": "#5cb85c", "busy": "#f0ad4e", "down": "#d9534f"}
        label_txt = {"tts": "TTS", "orchestrator": "Orchestrator", "ptt": "PTT"}.get(key, evt.component)
        txt = f"● {label_txt}"
        if evt.detail:
            txt = f"{txt} — {evt.detail}"
        self.status_dots[key].configure(fg=colors.get(evt.state, FG_MUTED), text=txt)
        if key == "orchestrator":
            self._orchestrator_busy = (evt.state == "busy")

    # --- Worker panel polling -------------------------------------------------
    # Tier presentation: glyph + color. Opus = distinguished, Sonnet = everyday,
    # Haiku = lightweight, Ollama = local LLM (Pyre).
    TIER_PRESENTATION = {
        "opus":   {"glyph": "●", "color": "#f0c674", "label": "Opus"},
        "sonnet": {"glyph": "●", "color": "#b5bd68", "label": "Sonnet"},
        "haiku":  {"glyph": "●", "color": "#8abeb7", "label": "Haiku"},
        "ollama": {"glyph": "◆", "color": "#cc6666", "label": "Local"},
    }

    def _poll_workers(self):
        active = self.app.active_workers_snapshot()
        named = self.app.named_agents_snapshot()
        # Dirty check — if nothing relevant changed since the last tick, skip the
        # full panel refresh. `_refresh_worker_panel` is a ~200-Tcl-call/paint
        # pass; running it at 1 Hz while the panel is static was a big chunk of
        # the visible UI lag.
        key = self._panel_state_key(active, named)
        if key != getattr(self, "_last_panel_state_key", None):
            self._last_panel_state_key = key
            self._refresh_worker_panel(active, named)
        # When no worker is running the panel's paintable state only changes
        # on registry writes (rare), so we can poll at 2 s instead of 1 s.
        # 1 s is only needed while an active worker's elapsed-seconds counter
        # is visible — that's the one field that ticks every second.
        interval = 1000 if active else 2500
        self.root.after(interval, self._poll_workers)

    def _panel_state_key(self, active: dict[str, dict], named: list[dict]) -> tuple:
        """Tuple that captures everything the panel renders. If this is equal to
        the previous tick's key, nothing painted matters and we can skip the
        refresh. Includes active elapsed-second bucket so the counter still
        ticks once per second when a worker is running."""
        now_int = int(time.time())
        active_key = tuple(
            sorted(
                (
                    n,
                    (info.get("task") or "")[:60],
                    now_int - int(info.get("started_at", now_int)),
                )
                for n, info in active.items()
            )
        )
        named_key = tuple(
            (
                a.get("name", ""),
                a.get("preferred_tier"),
                a.get("backend"),
            )
            for a in named
        )
        return (active_key, named_key)

    def _trim_text_widget(self, widget: tk.Text, max_lines: int) -> int:
        # Trim oldest lines from a Text widget, destroying any embedded
        # windows in the trimmed range first (tk.Text.window_create inserts
        # live widget trees that leak unless explicitly destroy()'d before
        # the text range is deleted).
        try:
            end_idx = widget.index("end-1c")
            total = int(end_idx.split(".")[0])
        except Exception:
            return 0
        if total <= max_lines:
            return 0
        cutoff = total - max_lines + 1
        cutoff_idx = f"{cutoff}.0"
        destroyed = 0
        try:
            for kind, value, _idx in widget.dump("1.0", cutoff_idx, window=True):
                if kind != "window" or not value:
                    continue
                try:
                    w = widget.nametowidget(value)
                except Exception:
                    continue
                try:
                    w.destroy()
                    destroyed += 1
                except Exception:
                    pass
        except Exception:
            pass
        was_disabled = str(widget.cget("state")) == "disabled"
        if was_disabled:
            widget.configure(state="normal")
        try:
            widget.delete("1.0", cutoff_idx)
        except Exception:
            pass
        if was_disabled:
            widget.configure(state="disabled")
        return destroyed

    def _trim_retention(self):
        # Keep chat/log widgets and the structured message list bounded so
        # memory stays flat and resize/layout work doesn't degrade over
        # multi-hour sessions. Cheap when already under cap.
        try:
            chat_destroyed = self._trim_text_widget(self.chat, self._chat_max_lines)
            self._trim_text_widget(self.log_text, self._log_max_lines)
            # If we destroyed embedded dispatch badges, clear stale references
            # so _apply_worker_done doesn't try to flip a destroyed widget.
            if chat_destroyed and self._active_dispatch_badges:
                dead = [
                    name for name, w in self._active_dispatch_badges.items()
                    if not bool(w.winfo_exists())
                ]
                for name in dead:
                    self._active_dispatch_badges.pop(name, None)
            if len(self._messages) > self._messages_max:
                excess = len(self._messages) - self._messages_max
                del self._messages[:excess]
            # Prune log marks whose text line is gone (mark resolves to "1.0"
            # when its anchor was deleted). Keeps _log_marks from growing
            # forever as worker badges accumulate.
            if self._log_marks:
                stale: list[int] = []
                for run_id, mark in self._log_marks.items():
                    try:
                        self.log_text.index(mark)
                    except tk.TclError:
                        stale.append(run_id)
                for run_id in stale:
                    self._log_marks.pop(run_id, None)
        except Exception:
            pass
        self.root.after(self._trim_interval_ms, self._trim_retention)

    def _animate_spinner(self):
        # Show spinner while orchestrator is busy. _orchestrator_busy is kept
        # in sync by _apply_status so we avoid a cget() Tcl roundtrip per tick.
        busy = self._orchestrator_busy
        if busy:
            ch = self._spinner_frames[self._spinner_idx % len(self._spinner_frames)]
            self._spinner_idx += 1
            self.spinner_label.configure(text=f"{ch} thinking")
            self._spinner_is_blank = False
            delay = 100
        else:
            # Clear once on the transition from busy → idle, then stop
            # repainting until we go busy again.
            if not getattr(self, "_spinner_is_blank", False):
                self.spinner_label.configure(text="")
                self._spinner_is_blank = True
            delay = 500
        self.root.after(delay, self._animate_spinner)

    def _poll_usage(self):
        # --- Context (graphical bar + readout) ---
        ctx = self.app.context_usage_snapshot()
        if ctx:
            total = ctx.get("total")
            mx = ctx.get("max")
            pct = ctx.get("pct")
            p = max(0.0, min(100.0, float(pct))) if isinstance(pct, (int, float)) else 0.0
            fill_w = int(round(p / 100.0 * self._context_bar_width))
            color = _bar_color(pct)
            self.context_canvas.coords(
                self._context_bar_fill, 0, 0, fill_w, self._context_bar_height,
            )
            self.context_canvas.itemconfigure(self._context_bar_fill, fill=color)
            if total is not None and mx:
                if pct is not None:
                    readout = f"{total:,} / {mx:,}  ({pct:.1f}%)"
                else:
                    readout = f"{total:,} / {mx:,}"
            else:
                readout = "—"
            self.context_readout.configure(text=readout, fg=color)

        # --- Five-hour reset countdown (usage data not reliable; just show reset) ---
        rls = self.app.rate_limits_snapshot()
        hourly_info = None
        for rl_type, info in rls.items():
            if _classify_rate_limit(rl_type) == "hourly":
                hourly_info = info
                break
        if hourly_info:
            reset_str = _format_reset_delta(hourly_info.get("resets_at"))
            self.hourly_reset_label.configure(
                text=f"5-hour {reset_str}" if reset_str else "",
                fg=FG_MUTED,
            )
        else:
            self.hourly_reset_label.configure(text="")

        self.root.after(3000, self._poll_usage)

    def _agent_tier(self, entry: dict) -> str:
        """Tier key for an agent registry entry (for the glyph + ordering)."""
        if (entry.get("backend") or "claude") == "ollama":
            return "ollama"
        t = (entry.get("preferred_tier") or "sonnet").lower()
        return t if t in ("opus", "sonnet", "haiku") else "sonnet"

    def _tier_sort_key(self, tier: str) -> int:
        return {"opus": 0, "sonnet": 1, "haiku": 2, "ollama": 3}.get(tier, 4)

    def _refresh_worker_panel(self, active: dict[str, dict], named: list[dict]):
        """Render hierarchical agent list. Named agents always visible (dimmed
        when inactive); ephemeral active workers (not in registry) append at end.

        This is called on a timer, so it repaints the same state over and over.
        We dirty-check per-row and short-circuit when nothing visible changed —
        avoids ~200 Tcl .configure() calls per second that were cascading
        geometry work through the agents panel.
        """
        now = time.time()

        # Build the ordered list of rows to render.
        #   1. Named agents from registry, sorted by tier (opus > sonnet > haiku > ollama)
        #      then alphabetical within tier.
        #   2. Any currently-active workers not already in the registry view.
        # Sub-agents (active entries with parent=<named agent>) are inserted
        # immediately after their parent and rendered with an indent so the
        # tree is legible at a glance.
        named_by_name = {a.get("name", ""): a for a in named if a.get("name")}
        top_level: list[str] = sorted(
            named_by_name.keys(),
            key=lambda n: (self._tier_sort_key(self._agent_tier(named_by_name[n])), n.lower()),
        )
        # Ephemeral top-level actives (not in registry, no parent) after named.
        for n in active.keys():
            info = active.get(n, {}) or {}
            if n in named_by_name:
                continue
            if info.get("parent"):
                continue
            top_level.append(n)

        # Group sub-agents by parent (parent name -> list of sub names).
        subs_by_parent: dict[str, list[str]] = {}
        for n, info in active.items():
            parent_n = (info or {}).get("parent")
            if parent_n:
                subs_by_parent.setdefault(parent_n, []).append(n)
        for lst in subs_by_parent.values():
            lst.sort(key=lambda s: (active.get(s, {}).get("started_at", 0), s.lower()))

        # Flattened render order with sub-agents following their parent.
        ordered_names: list[str] = []
        for n in top_level:
            ordered_names.append(n)
            for sub in subs_by_parent.get(n, ()):
                ordered_names.append(sub)
        # Orphan sub-agents (parent no longer active) appear at the end so they
        # don't disappear mid-dispatch.
        seen = set(ordered_names)
        for n in active.keys():
            if n not in seen:
                ordered_names.append(n)

        # Remove rows for agents that dropped out (shouldn't happen for named,
        # but ephemeral workers do finish).
        for n in list(self._worker_rows.keys()):
            if n not in ordered_names:
                self._worker_rows[n].destroy()
                del self._worker_rows[n]

        for name in ordered_names:
            entry = named_by_name.get(name, {})
            is_active = name in active
            act_info = active.get(name, {})
            # Sub-agents carry a parent link set at register_active_worker time.
            # We render them indented, in a darker band, with a small chevron
            # prefix so the tree relationship is obvious at a glance.
            sub_parent = (act_info or {}).get("parent") or ""
            is_sub = bool(sub_parent)
            tier = self._agent_tier(entry) if entry else "sonnet"
            tier_pres = self.TIER_PRESENTATION.get(tier, self.TIER_PRESENTATION["sonnet"])

            # Active vs idle styling.
            row_bg = "#2d2d2d" if is_active else "#242424"
            if is_sub:
                # Slightly darker band + indent to signal nesting without adding
                # a second tree widget. Active sub stays readable against parent.
                row_bg = "#272727" if is_active else "#202020"
            name_fg = "#fff" if is_active else FG_MUTED
            tier_fg = tier_pres["color"] if is_active else "#555"
            status_txt = "active" if is_active else "idle"
            status_fg = "#f0ad4e" if is_active else "#555"
            row_padx = 8 if not is_sub else 22          # indent sub-agents
            name_prefix = "" if not is_sub else "\u2937 "  # down-right arrow

            task_preview = ""
            task_is_live = False    # True when preview comes from a TodoWrite stream
            task_full = ""          # untruncated text shown in tooltip
            elapsed_txt = ""
            if is_active:
                live_status = self._worker_live_status.get(name)
                if live_status:
                    task_full = live_status
                    task_preview = (live_status[:80] + "…") if len(live_status) > 80 else live_status
                    task_is_live = True
                else:
                    raw_task = act_info.get("task", "")
                    task_full = raw_task
                    task_preview = _extract_task_summary(raw_task, max_len=80)
                elapsed_txt = f"{int(now - act_info.get('started_at', now))}s"

            if name not in self._worker_rows:
                row = tk.Frame(self.worker_list_container, bg=row_bg, pady=5, padx=row_padx)
                row.pack(fill=tk.X, pady=1)

                top = tk.Frame(row, bg=row_bg)
                top.pack(fill=tk.X)
                tier_lbl = tk.Label(
                    top, text=tier_pres["glyph"], fg=tier_fg, bg=row_bg,
                    font=("Segoe UI", 10, "bold"),
                )
                tier_lbl.pack(side=tk.LEFT, padx=(0, 6))
                name_lbl = tk.Label(
                    top, text=f"{name_prefix}{name}", fg=name_fg, bg=row_bg,
                    font=("Segoe UI", 10, "bold"), anchor="w",
                )
                name_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
                status_lbl = tk.Label(
                    top, text=status_txt, fg=status_fg, bg=row_bg,
                    font=("Segoe UI", 8),
                )
                status_lbl.pack(side=tk.RIGHT)

                tier_label_lbl = tk.Label(
                    row, text=tier_pres["label"], fg="#666", bg=row_bg,
                    font=("Segoe UI", 7), anchor="w",
                )
                tier_label_lbl.pack(fill=tk.X)

                task_lbl_fg = "#aaa" if task_is_live else FG_MUTED
                task_lbl_font = ("Segoe UI", 8, "italic") if task_is_live else ("Segoe UI", 8)
                task_lbl = tk.Label(
                    row, text=task_preview, fg=task_lbl_fg, bg=row_bg,
                    font=task_lbl_font, anchor="w", wraplength=210, justify="left",
                )
                task_lbl._full_text = task_full
                task_lbl.bind(
                    "<Enter>",
                    lambda _e, lbl=task_lbl: self._show_tooltip(lbl, lbl._full_text),
                )
                task_lbl.bind("<Leave>", self._hide_tooltip)
                if task_preview:
                    task_lbl.pack(fill=tk.X)

                bottom = tk.Frame(row, bg=row_bg)
                elapsed_lbl = tk.Label(
                    bottom, text=elapsed_txt, fg="#f0ad4e", bg=row_bg,
                    font=("Segoe UI", 8),
                )
                elapsed_lbl.pack(side=tk.LEFT)
                # Kill button — only meaningful when active. Currently a no-op
                # placeholder; wiring actual process-kill needs run_worker to
                # expose the Popen handle.
                kill_btn = tk.Button(
                    bottom, text="\u00d7", bg="#3a1f1f", fg="#e77",
                    activebackground="#5a2f2f", activeforeground="#fff",
                    relief=tk.FLAT, bd=0, padx=6, pady=0,
                    font=("Segoe UI", 8, "bold"),
                    command=lambda n=name: self._on_kill_click(n),
                )
                if is_active:
                    kill_btn.pack(side=tk.RIGHT)
                if elapsed_txt or is_active:
                    bottom.pack(fill=tk.X, pady=(2, 0))

                row.top = top
                row.tier_lbl = tier_lbl
                row.name_lbl = name_lbl
                row.status_lbl = status_lbl
                row.tier_label_lbl = tier_label_lbl
                row.task_lbl = task_lbl
                row.bottom = bottom
                row.elapsed_lbl = elapsed_lbl
                row.kill_btn = kill_btn
                self._worker_rows[name] = row
            else:
                row = self._worker_rows[name]
                task_lbl_fg = "#aaa" if task_is_live else FG_MUTED
                task_lbl_font = ("Segoe UI", 8, "italic") if task_is_live else ("Segoe UI", 8)
                kill_bg = row_bg if not is_active else "#3a1f1f"

                # Dirty-check signature — if every visible value matches what
                # we painted last tick, skip every .configure() call. Each
                # skipped call is a Tcl roundtrip + a potential geometry pass
                # that can cascade through the agents panel.
                sig = (
                    row_bg, tier_fg, name_fg, status_fg, status_txt,
                    task_preview, task_lbl_fg, task_lbl_font,
                    elapsed_txt, kill_bg, is_active, is_sub,
                )
                last_sig = getattr(row, "_last_sig", None)
                row.task_lbl._full_text = task_full  # tooltip text is cheap — always refresh
                if sig != last_sig:
                    row.configure(bg=row_bg, padx=row_padx)
                    row.top.configure(bg=row_bg)
                    row.bottom.configure(bg=row_bg)
                    row.tier_lbl.configure(fg=tier_fg, bg=row_bg)
                    row.name_lbl.configure(
                        fg=name_fg, bg=row_bg, text=f"{name_prefix}{name}",
                    )
                    row.status_lbl.configure(fg=status_fg, bg=row_bg, text=status_txt)
                    row.tier_label_lbl.configure(bg=row_bg)
                    row.task_lbl.configure(
                        text=task_preview, bg=row_bg,
                        fg=task_lbl_fg, font=task_lbl_font,
                    )
                    if task_preview and not row.task_lbl.winfo_ismapped():
                        row.task_lbl.pack(fill=tk.X)
                    elif not task_preview and row.task_lbl.winfo_ismapped():
                        row.task_lbl.pack_forget()
                    row.elapsed_lbl.configure(text=elapsed_txt, bg=row_bg)
                    row.kill_btn.configure(bg=kill_bg)
                    if is_active and not row.kill_btn.winfo_ismapped():
                        row.kill_btn.pack(side=tk.RIGHT)
                    elif not is_active and row.kill_btn.winfo_ismapped():
                        row.kill_btn.pack_forget()
                    row._last_sig = sig

    def _on_worker_mousewheel(self, event):
        """Scroll the worker canvas only when the pointer is over it.

        Bound via bind_all so rows and labels inside the canvas all forward
        wheel events, but we check pointer position to avoid hijacking wheel
        scrolling over the chat area.
        """
        try:
            x, y = self.root.winfo_pointerxy()
            target = self.root.winfo_containing(x, y)
        except Exception:
            return
        p = target
        while p is not None:
            if p is self._worker_canvas:
                self._worker_canvas.yview_scroll(
                    int(-1 * (event.delta / 120)), "units",
                )
                return "break"
            p = getattr(p, "master", None)
        return None

    def _on_kill_click(self, name: str):
        # Real kill wiring requires exposing Popen handles from run_worker.
        # For now, just surface the gap in chat so Rook sees something happened.
        self.chat.configure(state="normal")
        self.chat.insert(
            tk.END,
            f"\n[ui] kill requested for '{name}' — not yet implemented; "
            f"Popen handles aren't exposed from workers.run_worker.\n",
            "system",
        )
        self.chat.see(tk.END)
        self.chat.configure(state="disabled")

    # --- Input ----------------------------------------------------------------
    def _on_submit(self, _event=None):
        text = self.input_var.get().strip()
        if not text:
            return
        self.input_var.set("")
        self.app.submit_from_ui(text)

    # --- Visibility -----------------------------------------------------------
    def hide(self):
        # Save geometry when the user closes to tray so the next open restores
        # the same size/position. Also saved on full quit (see quit()).
        self._save_geometry()
        self.root.withdraw()

    def show(self):
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self.entry.focus_set()

    def run(self):
        self.root.mainloop()

    def quit(self):
        try:
            self._save_geometry()
        except Exception:
            pass
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

    # --- Password dialog (callable from asyncio thread) -----------------------
    def request_password_from_asyncio(self, title: str, prompt: str, timeout_sec: int = 120) -> str | None:
        result: list = []
        done = threading.Event()

        def show():
            try:
                self.show()
                pw = simpledialog.askstring(title, prompt, show="*", parent=self.root)
                result.append(pw)
            except Exception:
                result.append(None)
            finally:
                done.set()

        self.root.after(0, show)
        done.wait(timeout=timeout_sec)
        return result[0] if result else None


# -----------------------------------------------------------------------------
# Tray icon
# -----------------------------------------------------------------------------
def _make_tray_image() -> Image.Image:
    """Load Rook's custom Raiken icon (Blue Eyes) for the system tray, falling
    back to a drawn placeholder if the file isn't present."""
    png_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "raiken_blue.png"
    )
    try:
        if os.path.exists(png_path):
            return Image.open(png_path).convert("RGBA")
    except Exception:
        pass
    img = Image.new("RGB", (64, 64), "#1e1e1e")
    d = ImageDraw.Draw(img)
    d.ellipse([8, 8, 56, 56], fill="#c0392b", outline="#e74c3c", width=2)
    d.text((22, 18), "R", fill="white")
    return img


class RaikenTray:
    def __init__(self, app, window: RaikenWindow):
        self.app = app
        self.window = window
        self._icon: pystray.Icon | None = None
        self._thread: threading.Thread | None = None

    def _build(self) -> pystray.Icon:
        menu = pystray.Menu(
            pystray.MenuItem("Open Raiken", self._on_show, default=True),
            pystray.MenuItem("Restart TTS server", self._on_restart_tts),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._on_quit),
        )
        return pystray.Icon("raiken", _make_tray_image(), "Raiken Command Center", menu)

    def _on_show(self, _icon=None, _item=None):
        self.window.root.after(0, self.window.show)

    def _on_restart_tts(self, _icon=None, _item=None):
        self.app.restart_tts_from_ui()

    def _on_quit(self, _icon=None, _item=None):
        try:
            self._icon.stop()
        except Exception:
            pass
        self.window.root.after(0, self.app.quit)

    def run_detached(self):
        self._icon = self._build()
        self._thread = threading.Thread(target=self._icon.run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._icon:
            try:
                self._icon.stop()
            except Exception:
                pass
