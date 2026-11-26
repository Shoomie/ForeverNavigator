"""ForeverNavigator — Adapted MapNavigate GUI + Tactical Ruler/Whiteboard + Raid Sync + Replay Scrubber."""

from __future__ import annotations

import argparse
import ctypes
import os
import queue
import socket
import sys
import threading
import time
import tkinter as tk
from collections import OrderedDict
from pathlib import Path
from tkinter import filedialog, simpledialog, ttk
from typing import Any, Dict, List, Optional


def _ensure_stdio() -> None:
    """Redirect None stdout/stderr to a log file (or devnull) when launched via pythonw.exe."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        stream = open(log_dir / "forever_navigator.log", "a", encoding="utf-8", buffering=1)
    except Exception:
        stream = open(os.devnull, "w", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream


_ensure_stdio()

# Modern Dark Tactical Theme Palette
BG_APP = "#0d1117"
BG_SURFACE = "#161b22"
BG_ELEVATED = "#21262d"
BORDER_COLOR = "#30363d"
TEXT_PRIMARY = "#f0f6fc"
TEXT_MUTED = "#8b949e"
ACCENT_BLUE = "#58a6ff"
ACCENT_GOLD = "#f0b429"
ACCENT_GREEN = "#238636"
ACCENT_RED = "#da3633"
ACCENT_PURPLE = "#8957e5"

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import cv2
import mss
import numpy as np
from PIL import Image, ImageTk

from ..combatlog.parser import iter_combatlog_file
from ..combatlog.stats import analyze_combat_events
from ..navigator.dev_logger import DevGameplayLogger
from ..navigator.gpu_viewer3d import GPU3DViewer, find_wow_dir
from ..navigator.minimap_tracker import MinimapROI, MinimapTracker, load_roi, save_roi
from ..navigator.settings import (
    AppSettings,
    detect_storage_type,
    load_settings,
    resolve_storage_mode,
    save_settings,
)
from ..navigator.spawns import get_spawns_for_dungeon
from ..navigator.tracker import LightweightDungeonStore, TrackerEstimate
from ..net.client import ForeverNavigatorClient
from ..net.protocol import (
    WOW_CLASS_COLORS,
    PlayerIdentity,
    PlayerState,
    SyncPreferences,
    now_ms,
)
from ..net.server import HostConfig, ForeverNavigatorServer
from ..replay.session_store import SessionStore
from ..replay.timeline import TimelineController
from ..tactics.tools import PRESET_RANGES_YD, RulerMeasurement, measure_yards, yards_to_map_px
from ..tactics.whiteboard import TacticalWhiteboard, WhiteboardOp


class ROIPickerDialog(tk.Toplevel):
    """True see-through overlay over the selected monitor to drag a box around the WoW minimap."""

    def __init__(self, parent: tk.Tk, mon: dict, on_done):
        super().__init__(parent)
        self.on_done = on_done
        self.mon_w, self.mon_h = int(mon["width"]), int(mon["height"])
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.attributes("-alpha", 0.30)
        self.geometry(f"{self.mon_w}x{self.mon_h}+{int(mon['left'])}+{int(mon['top'])}")
        self.configure(bg="#0a0c10")
        self.canvas = tk.Canvas(self, width=self.mon_w, height=self.mon_h, cursor="crosshair", bg="#0a0c10", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.create_text(self.mon_w // 2, 48, text="Drag a box around your WoW Minimap  (ESC to cancel)", fill="#ffe066", font=("Segoe UI", 18, "bold"))
        self._start = None
        self._rect = None
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Escape>", lambda _e: self.destroy())
        self.focus_force()

    def _on_press(self, event):
        self._start = (event.x, event.y)
        if self._rect:
            self.canvas.delete(self._rect)
        self._rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#ffe066", width=3)

    def _on_drag(self, event):
        if self._start and self._rect:
            self.canvas.coords(self._rect, self._start[0], self._start[1], event.x, event.y)

    def _on_release(self, event):
        if not self._start:
            return
        x0, x1 = sorted([self._start[0], event.x])
        y0, y1 = sorted([self._start[1], event.y])
        if x1 - x0 < 20 or y1 - y0 < 20:
            return
        self.on_done(MinimapROI(x0 / self.mon_w, y0 / self.mon_h, x1 / self.mon_w, y1 / self.mon_h))
        self.destroy()


def _role_badge(role: str) -> str:
    r = (role or "DPS").upper()
    if r == "LEADER":
        return "L"
    if r == "OFFICER":
        return "O"
    if r == "TANK":
        return "T"
    if r == "HEALER":
        return "H"
    return "D"


def _detect_local_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return str(s.getsockname()[0])
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


class RaidSyncDialog(tk.Toplevel):
    """Host or Join a ForeverNavigator session with granular opt-in privacy & sync controls."""

    def __init__(self, app: "ForeverNavigatorApp"):
        super().__init__(app.root)
        self.app = app
        self.title("ForeverNavigator — Host / Join Raid Sync")
        self.configure(bg=BG_APP)
        self.resizable(False, False)
        self.transient(app.root)
        self.geometry("460x585")

        cb_opts = {
            "bg": BG_APP,
            "fg": TEXT_PRIMARY,
            "selectcolor": BG_ELEVATED,
            "activebackground": BG_APP,
            "activeforeground": TEXT_PRIMARY,
            "font": ("Segoe UI", 9),
        }

        def hdr(txt: str):
            tk.Label(self, text=txt, bg=BG_APP, fg=ACCENT_GOLD, font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, padx=14, pady=(10, 2))

        # Status & LAN IP banner
        lan_ip = app.get_local_lan_ip()
        status_box = tk.Frame(self, bg=BG_SURFACE, padx=10, pady=8, highlightbackground=BORDER_COLOR, highlightthickness=1)
        status_box.pack(fill=tk.X, padx=14, pady=(10, 2))
        tk.Label(status_box, text=app.get_network_status_summary(), bg=BG_SURFACE, fg=ACCENT_BLUE, font=("Segoe UI", 9, "bold"), anchor=tk.W).pack(fill=tk.X)
        ip_row = tk.Frame(status_box, bg=BG_SURFACE)
        ip_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(ip_row, text="Your Local LAN IP:", bg=BG_SURFACE, fg=TEXT_MUTED, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        ip_entry = tk.Entry(ip_row, width=18, bg=BG_ELEVATED, fg="#3fb950", relief=tk.FLAT, font=("Consolas", 9, "bold"), readonlybackground=BG_ELEVATED)
        ip_entry.insert(0, lan_ip)
        ip_entry.config(state="readonly")
        ip_entry.pack(side=tk.LEFT, padx=6)

        hdr("Player Identity:")
        r1 = tk.Frame(self, bg=BG_APP)
        r1.pack(fill=tk.X, padx=14)
        self.name_var = tk.StringVar(value=app.identity.name)
        tk.Entry(r1, textvariable=self.name_var, width=16, bg=BG_ELEVATED, fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY, relief=tk.FLAT, highlightbackground=BORDER_COLOR, highlightthickness=1).pack(side=tk.LEFT, padx=(0, 6))
        self.class_var = tk.StringVar(value=app.identity.wow_class)
        ttk.Combobox(r1, textvariable=self.class_var, values=list(WOW_CLASS_COLORS.keys()), width=10, state="readonly").pack(side=tk.LEFT, padx=(0, 6))
        self.role_var = tk.StringVar(value=app.identity.role)
        ttk.Combobox(r1, textvariable=self.role_var, values=["LEADER", "OFFICER", "TANK", "HEALER", "DPS"], width=8, state="readonly").pack(side=tk.LEFT)

        hdr("Granular Opt-In Sync Preferences:")
        self.opt_pos = tk.BooleanVar(value=app.sync_prefs.sync_position)
        tk.Checkbutton(self, text="Share Live Position with Raid", variable=self.opt_pos, **cb_opts).pack(anchor=tk.W, padx=14)

        self.opt_vid = tk.BooleanVar(value=app.sync_prefs.sync_video)
        tk.Checkbutton(self, text="Opt-In Video Stream / Buffer", variable=self.opt_vid, **cb_opts).pack(anchor=tk.W, padx=14)
        rv = tk.Frame(self, bg=BG_APP)
        rv.pack(fill=tk.X, padx=28, pady=2)
        tk.Label(rv, text="Quality:", bg=BG_APP, fg=TEXT_MUTED).pack(side=tk.LEFT)
        self.vid_q = tk.StringVar(value=app.sync_prefs.video_quality)
        ttk.Combobox(rv, textvariable=self.vid_q, values=["low", "medium", "high"], width=8, state="readonly").pack(side=tk.LEFT, padx=4)
        tk.Label(rv, text="Delivery:", bg=BG_APP, fg=TEXT_MUTED).pack(side=tk.LEFT, padx=(6, 0))
        self.vid_m = tk.StringVar(value=app.sync_prefs.video_mode)
        ttk.Combobox(rv, textvariable=self.vid_m, values=["realtime", "out_of_combat", "post_raid"], width=13, state="readonly").pack(side=tk.LEFT, padx=4)

        self.opt_cl = tk.BooleanVar(value=app.sync_prefs.sync_combatlog)
        tk.Checkbutton(self, text="Share Combat Log (after /reload or raid)", variable=self.opt_cl, **cb_opts).pack(anchor=tk.W, padx=14)

        hdr("Connection (Host or Join):")
        rc = tk.Frame(self, bg=BG_APP)
        rc.pack(fill=tk.X, padx=14, pady=2)
        tk.Label(rc, text="Host/IP:", bg=BG_APP, fg=TEXT_MUTED).pack(side=tk.LEFT)
        self.host_var = tk.StringVar(value="127.0.0.1")
        tk.Entry(rc, textvariable=self.host_var, width=20, bg=BG_ELEVATED, fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY, relief=tk.FLAT, highlightbackground=BORDER_COLOR, highlightthickness=1).pack(side=tk.LEFT, padx=4)
        tk.Label(rc, text="Port:", bg=BG_APP, fg=TEXT_MUTED).pack(side=tk.LEFT)
        self.port_var = tk.StringVar(value="8765")
        tk.Entry(rc, textvariable=self.port_var, width=6, bg=BG_ELEVATED, fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY, relief=tk.FLAT, highlightbackground=BORDER_COLOR, highlightthickness=1).pack(side=tk.LEFT, padx=4)

        rp = tk.Frame(self, bg=BG_APP)
        rp.pack(fill=tk.X, padx=14, pady=4)
        tk.Label(rp, text="Passphrase:", bg=BG_APP, fg=TEXT_MUTED).pack(side=tk.LEFT)
        self.pass_var = tk.StringVar(value="")
        tk.Entry(rp, textvariable=self.pass_var, width=14, show="*", bg=BG_ELEVATED, fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY, relief=tk.FLAT, highlightbackground=BORDER_COLOR, highlightthickness=1).pack(side=tk.LEFT, padx=4)
        tk.Label(rp, text="Officer Key:", bg=BG_APP, fg=TEXT_MUTED).pack(side=tk.LEFT, padx=(6, 0))
        self.officer_var = tk.StringVar(value="")
        tk.Entry(rp, textvariable=self.officer_var, width=10, show="*", bg=BG_ELEVATED, fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY, relief=tk.FLAT, highlightbackground=BORDER_COLOR, highlightthickness=1).pack(side=tk.LEFT, padx=4)

        rw = tk.Frame(self, bg=BG_APP)
        rw.pack(fill=tk.X, padx=14, pady=2)
        tk.Label(rw, text="Host Whiteboard Permission:", bg=BG_APP, fg=TEXT_MUTED).pack(side=tk.LEFT)
        self.wb_perm_var = tk.StringVar(value=app.host_whiteboard_perm)
        ttk.Combobox(rw, textvariable=self.wb_perm_var, values=["all", "officers", "leader"], width=10, state="readonly").pack(side=tk.LEFT, padx=6)

        self.err_lbl = tk.Label(self, text="", bg=BG_APP, fg="#f85149", font=("Segoe UI", 8))
        self.err_lbl.pack(anchor=tk.W, padx=14, pady=(4, 0))

        btns = tk.Frame(self, bg=BG_APP)
        btns.pack(fill=tk.X, padx=14, pady=10)
        tk.Button(btns, text="Host Session", command=self._host, bg=ACCENT_GREEN, fg="white", relief=tk.FLAT, padx=12, pady=6, font=("Segoe UI", 9, "bold"), cursor="hand2").pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(btns, text="Join Session", command=self._join, bg="#1f6feb", fg="white", relief=tk.FLAT, padx=12, pady=6, font=("Segoe UI", 9, "bold"), cursor="hand2").pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(btns, text="Disconnect", command=self._disconnect, bg=ACCENT_RED, fg="white", relief=tk.FLAT, padx=12, pady=6, cursor="hand2").pack(side=tk.LEFT)

    def _save_identity(self):
        self.app.identity.name = self.name_var.get().strip() or "Raider"
        self.app.identity.wow_class = self.class_var.get()
        self.app.identity.role = self.role_var.get()
        self.app.host_whiteboard_perm = self.wb_perm_var.get()
        self.app.sync_prefs = SyncPreferences(
            sync_position=bool(self.opt_pos.get()),
            poll_hz=self.app.fps,
            sync_video=bool(self.opt_vid.get()),
            video_quality=self.vid_q.get(),
            video_mode=self.vid_m.get(),
            sync_combatlog=bool(self.opt_cl.get()),
        )

    def _parse_port(self) -> Optional[int]:
        try:
            port = int(self.port_var.get().strip())
            if not (1 <= port <= 65535):
                raise ValueError
            return port
        except ValueError:
            self.err_lbl.config(text="Invalid port (must be 1-65535).")
            return None

    def _host(self):
        port = self._parse_port()
        if port is None:
            return
        self._save_identity()
        ok = self.app.start_host(
            port,
            self.pass_var.get(),
            whiteboard_permission=self.wb_perm_var.get(),
            officer_token=self.officer_var.get(),
        )
        if not ok:
            err = self.app.last_net_error or "Failed to start Host server."
            self.err_lbl.config(text=err)
            return
        self.destroy()

    def _join(self):
        port = self._parse_port()
        if port is None:
            return
        raw_host = self.host_var.get().strip()
        if not raw_host:
            self.err_lbl.config(text="Enter Host IP or ws:// URL.")
            return
        self._save_identity()
        if raw_host.startswith(("ws://", "wss://")):
            url = raw_host
        elif ":" in raw_host and not raw_host.startswith("["):
            url = f"ws://{raw_host}"
        else:
            url = f"ws://{raw_host}:{port}"
        ok = self.app.start_client(url, self.pass_var.get(), officer_token=self.officer_var.get())
        if not ok:
            err = self.app.last_net_error or "Failed to start Client."
            self.err_lbl.config(text=err)
            return
        self.destroy()

    def _disconnect(self):
        self.app.stop_network()
        self.destroy()


class CombatStatsDialog(tk.Toplevel):
    """Loads and displays extensive WoWCombatLog statistics (DPS, HPS, Overheal, Deaths)."""

    def __init__(self, parent: tk.Tk, default_logs_dir: Optional[Path] = None):
        super().__init__(parent)
        self.title("ForeverNavigator — Combat Log Analytics")
        self.configure(bg=BG_APP)
        self.geometry("680x500")
        self.default_logs_dir = default_logs_dir

        top = tk.Frame(self, bg=BG_SURFACE, highlightbackground=BORDER_COLOR, highlightthickness=1)
        top.pack(fill=tk.X, padx=10, pady=(10, 6))
        tk.Button(top, text="Load WoWCombatLog-*.txt...", command=self._load_log, bg="#1f6feb", fg="white", relief=tk.FLAT, padx=12, pady=5, font=("Segoe UI", 9, "bold"), cursor="hand2").pack(side=tk.LEFT, padx=8, pady=6)

        self.txt = tk.Text(self, bg=BG_SURFACE, fg=TEXT_PRIMARY, font=("Consolas", 9), relief=tk.FLAT, padx=12, pady=10, highlightbackground=BORDER_COLOR, highlightthickness=1)
        self.txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.txt.insert(tk.END, "Click 'Load WoWCombatLog-*.txt...' to analyze DPS, HPS, Overhealing %, Interrupts, Dispels & Death Recaps.\n")

    def _load_log(self):
        init_dir = str(self.default_logs_dir) if self.default_logs_dir and self.default_logs_dir.exists() else "."
        path = filedialog.askopenfilename(initialdir=init_dir, title="Select WoWCombatLog file", filetypes=[("Text logs", "*.txt")])
        if not path:
            return
        summary = analyze_combat_events(iter_combatlog_file(path), session_name=Path(path).name)
        lines = [
            f"=== {summary.name} ({summary.duration_sec:.1f}s) ===",
            "",
            "--- Top DPS ---",
        ]
        for name, dps, total in summary.dps_leaderboard()[:15]:
            p = summary.players[name]
            lines.append(f"  {name:18s} | {dps:8.1f} DPS | Total: {total:10,d} | Int: {p.interrupts} | Deaths: {p.deaths}")
        lines.append("\n--- Top HPS ---")
        for name, hps, ovh in summary.hps_leaderboard()[:10]:
            p = summary.players[name]
            lines.append(f"  {name:18s} | {hps:8.1f} HPS | Overheal: {ovh:4.1f}% | Dispels: {p.dispels}")
        lines.append(f"\n--- Death Recaps ({len(summary.death_recaps)}) ---")
        for dr in summary.death_recaps[:10]:
            last_hit = dr.recent_events[-1] if dr.recent_events else None
            hit_txt = f"{last_hit.spell_name} ({last_hit.amount})" if last_hit else "unknown"
            lines.append(f"  {dr.player_name:18s} | Killing/Recent event: {hit_txt}")

        self.txt.delete("1.0", tk.END)
        self.txt.insert(tk.END, "\n".join(lines))


class ForeverNavigatorApp:
    def __init__(self, index_dir: str | Path, fps: Optional[float] = None):
        self.index_dir = Path(index_dir)
        self.data_dir = self.index_dir.parent
        self.roi_path = self.data_dir / "minimap_roi.json"
        self.settings_path = self.data_dir / "settings.json"
        self.settings: AppSettings = load_settings(self.settings_path)
        if fps is not None:
            self.settings.fps = float(fps)
        self._detected_storage: str = detect_storage_type(self.data_dir)
        self._effective_storage_mode: str = resolve_storage_mode(self.settings.storage_mode, self.data_dir)

        self.wow_dir = find_wow_dir(self.settings.wow_dir or None)
        self.wow_logs_dir = (self.wow_dir / "_forever_" / "Logs") if self.wow_dir else None

        self.tracker_3d = LightweightDungeonStore(index_dir, max_preview_px=512)
        self.tracker_mm = MinimapTracker(
            dungeons=self.tracker_3d.dungeons,
            dungeon_order=self.tracker_3d.dungeon_order,
            roi=load_roi(self.roi_path),
            rotate_minimap=self.settings.rotate_minimap,
            high_precision=self.settings.high_precision,
            lazy_precompute=True,
        )
        self.viewer_3d = GPU3DViewer(wow_dir=self.wow_dir, listfile_path=self.data_dir / "listfile.csv", product="wowf")
        try:
            from ..navigator.world_maps import WorldMapRegistry

            self.world_maps: Optional[Any] = WorldMapRegistry(
                index_dir=self.index_dir,
                cache_dir=self.data_dir / "cache" / "maps",
                wow_dir=self.wow_dir,
            )
            self.world_maps.set_storage_mode(self._effective_storage_mode)
        except Exception:
            self.world_maps = None

        # ForeverNavigator subsystems
        self.identity = PlayerIdentity(name="Player", wow_class="WARRIOR", role="DPS")
        self.sync_prefs = SyncPreferences(poll_hz=self.settings.fps)
        self.server: Optional[ForeverNavigatorServer] = None
        self.client: Optional[ForeverNavigatorClient] = None
        self.raid_players: Dict[str, PlayerState] = {}
        self.host_whiteboard_perm: str = "all"
        self.last_net_error: Optional[str] = None
        self._net_connected: bool = False
        self._ui_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._closing: bool = False
        self.whiteboard = TacticalWhiteboard()
        self.active_ruler: Optional[RulerMeasurement] = None
        self._active_stroke_pts: list[tuple[float, float]] = []

        # Synced session recording & timeline scrubber
        self.session_store = SessionStore(self.data_dir / "sessions" / f"session_{int(time.time())}.db")
        self.dev_logger = DevGameplayLogger(self.data_dir / "dev_gameplay")
        self.timeline = TimelineController(self.session_store)
        self.replay_mode: bool = False
        self.replay_players: Dict[str, PlayerState] = {}

        self.fps = self.settings.fps
        self.running = False
        self._stop_capture_event = threading.Event()
        self._poll_after_id: Optional[str] = None
        self._render_after_id: Optional[str] = None
        self._settle_after_id: Optional[str] = None
        self._is_interacting: bool = False
        self._last_rendered_est: Optional[TrackerEstimate] = None
        self._last_roster_sig: Optional[tuple] = None
        self._selected_monitor: int = max(1, int(self.settings.monitor))
        self._latest_est: Optional[TrackerEstimate] = None
        self._latest_mm_crop: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self.zoom = 1.0
        self.pan_x, self.pan_y = 512.0, 512.0
        self._drag_last = None
        self._rdrag_last = None
        self._cached_pil_maps: OrderedDict[str, Image.Image] = OrderedDict()
        self._cached_mips: dict[str, list[tuple[int, Image.Image]]] = {}
        self._cached_np_mips: dict[str, list[tuple[int, np.ndarray]]] = {}
        self._streaming_slugs: set[str] = set()
        self._continent_maps: dict[str, Any] = {}
        self._continent_loading: set[str] = set()
        self._cont_l1_cache: dict[str, tuple[np.ndarray, int]] = {}
        self._cont_l1_loading: set[str] = set()
        self._deep_tile_view: Optional[tuple[tuple, Image.Image]] = None
        self._deep_tile_req_id: int = 0
        self._showing_azeroth_overview: bool = True
        self._azeroth_overview: Optional[Any] = None
        self._azeroth_overview_arr: Optional[np.ndarray] = None
        self._overview_zoom: float = 1.0
        self._overview_pan_x: float = 1920.0
        self._overview_pan_y: float = 1080.0
        self._transition_after_id: Optional[str] = None
        self._transition_state: Optional[dict[str, Any]] = None
        self._zoom_target: Optional[float] = None
        self._zoom_anchor: Optional[tuple[float, float]] = None
        self._zoom_anim_id: Optional[str] = None
        self._last_pills_sig: Optional[tuple] = None
        self._last_dungeon_lbl: str = ""
        self.view_3d_mode = False

        self.root = tk.Tk()
        self.root.title("ForeverNavigator — WoW: Forever Tactical Navigator & Replay")
        self.root.configure(bg=BG_APP)
        self.root.geometry("1200x840")
        self.root.minsize(820, 560)

        self._all_map_entries: List[Any] = []
        self._filtered_map_entries: List[Any] = []
        self._map_loading_slug: Optional[str] = None

        self._build_ui()
        self._refresh_map_catalog()
        self._load_azeroth_overview_async()
        self._preload_cached_maps()
        self._render_current_map(None)
        self.root.after(50, self._drain_ui_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_theme(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=BG_APP)
        style.configure("Surface.TFrame", background=BG_SURFACE)
        style.configure("TLabel", background=BG_APP, foreground=TEXT_PRIMARY, font=("Segoe UI", 9))
        style.configure("Surface.TLabel", background=BG_SURFACE, foreground=TEXT_PRIMARY, font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=BG_SURFACE, font=("Segoe UI", 11, "bold"), foreground=ACCENT_GOLD)
        style.configure("Dim.TLabel", background=BG_SURFACE, foreground=TEXT_MUTED, font=("Segoe UI", 8))

        style.configure(
            "TCombobox",
            fieldbackground=BG_ELEVATED,
            background=BG_ELEVATED,
            foreground=TEXT_PRIMARY,
            bordercolor=BORDER_COLOR,
            arrowcolor=TEXT_MUTED,
            lightcolor=BG_ELEVATED,
            darkcolor=BG_ELEVATED,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", BG_ELEVATED)],
            selectbackground=[("readonly", BG_ELEVATED)],
            selectforeground=[("readonly", TEXT_PRIMARY)],
            foreground=[("readonly", TEXT_PRIMARY)],
        )

        style.configure("TNotebook", background=BG_SURFACE, borderwidth=0, tabmargins=[2, 4, 2, 0])
        style.configure(
            "TNotebook.Tab",
            background=BG_SURFACE,
            foreground=TEXT_MUTED,
            padding=[10, 5],
            font=("Segoe UI", 9, "bold"),
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", BG_ELEVATED), ("active", BG_ELEVATED)],
            foreground=[("selected", ACCENT_BLUE), ("active", TEXT_PRIMARY)],
        )

        style.configure(
            "Vertical.TScrollbar",
            background=BG_ELEVATED,
            troughcolor=BG_APP,
            bordercolor=BORDER_COLOR,
            arrowcolor=TEXT_MUTED,
        )
        style.configure(
            "Treeview",
            background=BG_APP,
            fieldbackground=BG_APP,
            foreground=TEXT_PRIMARY,
            borderwidth=0,
            rowheight=22,
            font=("Segoe UI", 8),
        )
        style.configure(
            "Treeview.Heading",
            background=BG_ELEVATED,
            foreground=TEXT_MUTED,
            borderwidth=0,
            font=("Segoe UI", 8, "bold"),
        )
        style.map("Treeview", background=[("selected", "#1f6feb")], foreground=[("selected", "white")])

    def _build_ui(self) -> None:
        self._configure_theme()

        # --- Top Status Bar with Live Pills ---
        topbar = tk.Frame(self.root, bg=BG_SURFACE, height=46, highlightbackground=BORDER_COLOR, highlightthickness=1)
        topbar.pack(fill=tk.X)
        topbar.pack_propagate(False)

        left_hdr = tk.Frame(topbar, bg=BG_SURFACE)
        left_hdr.pack(side=tk.LEFT, fill=tk.Y, padx=12)
        tk.Label(left_hdr, text="FOREVERNAVIGATOR", bg=BG_SURFACE, fg=ACCENT_GOLD, font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        self.lbl_dungeon = tk.Label(left_hdr, text="ForeverNavigator Ready", bg=BG_SURFACE, fg=TEXT_PRIMARY, font=("Segoe UI", 10, "bold"))
        self.lbl_dungeon.pack(side=tk.LEFT)

        pills = tk.Frame(topbar, bg=BG_SURFACE)
        pills.pack(side=tk.RIGHT, padx=10)
        pill_kw = {"bg": BG_ELEVATED, "fg": TEXT_MUTED, "font": ("Segoe UI", 8, "bold"), "padx": 8, "pady": 3, "highlightbackground": BORDER_COLOR, "highlightthickness": 1}
        self.pill_fps = tk.Label(pills, text=f"{self.fps:.1f} Hz", **pill_kw)
        self.pill_fps.pack(side=tk.RIGHT, padx=(6, 0))
        self.pill_pos = tk.Label(pills, text="Pos: --", **pill_kw)
        self.pill_pos.pack(side=tk.RIGHT, padx=(6, 0))
        self.pill_sync = tk.Label(pills, text="Offline Solo", cursor="hand2", **{**pill_kw, "fg": ACCENT_BLUE})
        self.pill_sync.pack(side=tk.RIGHT, padx=(6, 0))
        self.pill_sync.bind("<Button-1>", lambda _e: RaidSyncDialog(self))

        # --- Main Body: Left Sidebar + Right Viewport ---
        body = tk.Frame(self.root, bg=BG_APP)
        body.pack(fill=tk.BOTH, expand=True)

        sidebar = tk.Frame(body, bg=BG_SURFACE, width=300, highlightbackground=BORDER_COLOR, highlightthickness=1)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        nb = ttk.Notebook(sidebar)
        nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        tab_nav = tk.Frame(nb, bg=BG_SURFACE)
        tab_tactics = tk.Frame(nb, bg=BG_SURFACE)
        tab_raid = tk.Frame(nb, bg=BG_SURFACE)
        nb.add(tab_nav, text="Map & Nav")
        nb.add(tab_tactics, text="Tactics")
        nb.add(tab_raid, text="Raid & Replay")

        def make_card(parent: tk.Widget, title: str) -> tk.Frame:
            outer = tk.Frame(parent, bg=BG_APP, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=8, pady=8)
            outer.pack(fill=tk.X, padx=6, pady=(6, 2))
            tk.Label(outer, text=title.upper(), bg=BG_APP, fg=TEXT_MUTED, font=("Segoe UI", 7, "bold")).pack(anchor=tk.W, pady=(0, 6))
            return outer

        # === TAB 1: MAP & NAV ===
        c_cap = make_card(tab_nav, "Capture & View")
        cap_btns = tk.Frame(c_cap, bg=BG_APP)
        cap_btns.pack(fill=tk.X)
        self.btn_toggle = tk.Button(cap_btns, text="Start Capture", command=self._toggle_capture, bg="#1f6feb", fg="white", relief=tk.FLAT, padx=10, pady=5, font=("Segoe UI", 9, "bold"), cursor="hand2")
        self.btn_toggle.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        tk.Button(cap_btns, text="Pick Minimap...", command=self._pick_minimap_box, bg=BG_ELEVATED, fg=TEXT_PRIMARY, relief=tk.FLAT, padx=8, pady=5, font=("Segoe UI", 8), cursor="hand2").pack(side=tk.LEFT)

        cap_opts = tk.Frame(c_cap, bg=BG_APP)
        cap_opts.pack(fill=tk.X, pady=(6, 0))
        with mss.MSS() as sct:
            n_mons = max(1, len(sct.monitors) - 1)
        tk.Label(cap_opts, text="Mon:", bg=BG_APP, fg=TEXT_MUTED, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self._selected_monitor = min(self.settings.monitor, n_mons)
        self.mon_var = tk.StringVar(value=str(self._selected_monitor))
        mon_cb = ttk.Combobox(cap_opts, textvariable=self.mon_var, values=[str(i) for i in range(1, n_mons + 1)], width=3, state="readonly")
        mon_cb.pack(side=tk.LEFT, padx=(3, 8))
        mon_cb.bind("<<ComboboxSelected>>", lambda _e: setattr(self, "_selected_monitor", max(1, int(self.mon_var.get()))))

        self.follow_var = tk.BooleanVar(value=bool(self.settings.follow_player))
        tk.Checkbutton(cap_opts, text="Center Player", variable=self.follow_var, command=self._on_follow_toggle, bg=BG_APP, fg=ACCENT_GOLD, selectcolor=BG_ELEVATED, activebackground=BG_APP, activeforeground=ACCENT_GOLD, font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(0, 4))
        self.spawns_var = tk.BooleanVar(value=bool(self.settings.show_spawns))
        tk.Checkbutton(cap_opts, text="Spawns", variable=self.spawns_var, command=lambda: self._render_current_map(self._latest_est, self._latest_mm_crop), bg=BG_APP, fg="#ffa53c", selectcolor=BG_ELEVATED, activebackground=BG_APP, activeforeground="#ffa53c", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(0, 4))
        self.high_precision_var = tk.BooleanVar(value=bool(self.settings.high_precision))
        tk.Checkbutton(cap_opts, text="Hi-Prec", variable=self.high_precision_var, command=self._toggle_high_precision, bg=BG_APP, fg=ACCENT_BLUE, selectcolor=BG_ELEVATED, activebackground=BG_APP, activeforeground=ACCENT_BLUE, font=("Segoe UI", 8)).pack(side=tk.LEFT)

        stor_row = tk.Frame(c_cap, bg=BG_APP)
        stor_row.pack(fill=tk.X, pady=(6, 0))
        tk.Label(stor_row, text="Storage:", bg=BG_APP, fg=TEXT_MUTED, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        auto_lbl = f"Auto ({self._detected_storage.upper()})"
        self._storage_label_to_mode = {
            auto_lbl: "auto",
            "SSD (Low RAM)": "ssd",
            "HDD (More Cache)": "hdd",
        }
        mode_to_lbl = {v: k for k, v in self._storage_label_to_mode.items()}
        self.storage_var = tk.StringVar(value=mode_to_lbl.get(self.settings.storage_mode, auto_lbl))
        stor_cb = ttk.Combobox(
            stor_row,
            textvariable=self.storage_var,
            values=list(self._storage_label_to_mode.keys()),
            width=16,
            state="readonly",
        )
        stor_cb.pack(side=tk.LEFT, padx=(4, 0), fill=tk.X, expand=True)
        stor_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_storage_mode_change())

        self.btn_view3d = tk.Button(c_cap, text="View: 2D Map", command=self._toggle_3d_view, bg=BG_ELEVATED, fg=TEXT_PRIMARY, relief=tk.FLAT, padx=8, pady=4, font=("Segoe UI", 8, "bold"), cursor="hand2")
        self.btn_view3d.pack(fill=tk.X, pady=(6, 0))

        # All Maps Card
        c_maps = tk.Frame(tab_nav, bg=BG_APP, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=8, pady=8)
        c_maps.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        hdr_maps = tk.Frame(c_maps, bg=BG_APP)
        hdr_maps.pack(fill=tk.X, pady=(0, 4))
        tk.Label(hdr_maps, text="ALL MAPS (WOW: FOREVER)", bg=BG_APP, fg=TEXT_MUTED, font=("Segoe UI", 7, "bold")).pack(side=tk.LEFT)
        tk.Button(hdr_maps, text="Auto-Detect", command=self._set_auto_detect, bg=BG_ELEVATED, fg=ACCENT_BLUE, relief=tk.FLAT, padx=6, pady=1, font=("Segoe UI", 7, "bold"), cursor="hand2").pack(side=tk.RIGHT)
        tk.Button(hdr_maps, text="Index Zones", command=self._preload_all_zones, bg=BG_ELEVATED, fg=ACCENT_GOLD, relief=tk.FLAT, padx=6, pady=1, font=("Segoe UI", 7, "bold"), cursor="hand2").pack(side=tk.RIGHT, padx=(0, 4))

        self.dungeon_var = tk.StringVar(value="Auto-Detect")
        self.map_search_var = tk.StringVar(value="")
        self.map_search_var.trace_add("write", lambda *_a: self._filter_map_list())
        tk.Entry(c_maps, textvariable=self.map_search_var, bg=BG_ELEVATED, fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY, relief=tk.FLAT, highlightbackground=BORDER_COLOR, highlightthickness=1, font=("Segoe UI", 9)).pack(fill=tk.X, pady=(0, 4))

        self.map_cat_var = tk.StringVar(value="All")
        cat_cb = ttk.Combobox(
            c_maps,
            textvariable=self.map_cat_var,
            values=["All", "Raids", "Dungeons", "Battlegrounds", "Continents", "Zones", "Cities", "Caves & Buildings", "Other"],
            state="readonly",
        )
        cat_cb.pack(fill=tk.X, pady=(0, 4))
        cat_cb.bind("<<ComboboxSelected>>", lambda _e: self._filter_map_list())

        list_frame = tk.Frame(c_maps, bg=BG_APP)
        list_frame.pack(fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.map_listbox = tk.Listbox(
            list_frame,
            bg=BG_SURFACE,
            fg=TEXT_PRIMARY,
            selectbackground="#1f6feb",
            selectforeground="white",
            relief=tk.FLAT,
            highlightthickness=0,
            font=("Segoe UI", 8),
            yscrollcommand=sb.set,
        )
        sb.config(command=self.map_listbox.yview)
        self.map_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.map_listbox.bind("<Double-Button-1>", lambda _e: self._load_selected_map())
        self.map_listbox.bind("<Return>", lambda _e: self._load_selected_map())

        map_act = tk.Frame(c_maps, bg=BG_APP)
        map_act.pack(fill=tk.X, pady=(4, 0))
        tk.Button(map_act, text="Load Selected Map", command=self._load_selected_map, bg=ACCENT_GREEN, fg="white", relief=tk.FLAT, padx=8, pady=4, font=("Segoe UI", 8, "bold"), cursor="hand2").pack(fill=tk.X)
        self.lbl_map_status = tk.Label(c_maps, text="", bg=BG_APP, fg=TEXT_MUTED, font=("Segoe UI", 7), anchor=tk.W)
        self.lbl_map_status.pack(fill=tk.X, pady=(2, 0))

        # === TAB 2: TACTICS ===
        c_tools = make_card(tab_tactics, "Tactical Tools")
        self.tool_var = tk.StringVar(value="Pan")
        tool_grid = tk.Frame(c_tools, bg=BG_APP)
        tool_grid.pack(fill=tk.X)
        tools_list = [
            ("Pan", "Pan"),
            ("Ruler (yd)", "Ruler (yd)"),
            ("Draw", "Draw"),
            ("Text", "Text"),
            ("Range 8yd", "8 yd"),
            ("Range 30yd", "30 yd"),
            ("Range 40yd", "40 yd"),
        ]
        for idx, (val, label) in enumerate(tools_list):
            rb = tk.Radiobutton(
                tool_grid,
                text=label,
                variable=self.tool_var,
                value=val,
                indicatoron=False,
                bg=BG_ELEVATED,
                fg=TEXT_PRIMARY,
                selectcolor="#1f6feb",
                activebackground=BG_ELEVATED,
                activeforeground="white",
                relief=tk.FLAT,
                padx=8,
                pady=4,
                font=("Segoe UI", 8, "bold"),
                cursor="hand2",
            )
            rb.grid(row=idx // 2, column=idx % 2, sticky="ew", padx=2, pady=2)
        tool_grid.columnconfigure(0, weight=1)
        tool_grid.columnconfigure(1, weight=1)

        c_wb = make_card(tab_tactics, "Whiteboard")
        wb_btns = tk.Frame(c_wb, bg=BG_APP)
        wb_btns.pack(fill=tk.X)
        tk.Button(wb_btns, text="Undo", command=self._undo_draw, bg=BG_ELEVATED, fg=TEXT_PRIMARY, relief=tk.FLAT, padx=10, pady=4, font=("Segoe UI", 8), cursor="hand2").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        tk.Button(wb_btns, text="Clear Board", command=self._clear_board, bg=BG_ELEVATED, fg="#f85149", relief=tk.FLAT, padx=10, pady=4, font=("Segoe UI", 8), cursor="hand2").pack(side=tk.LEFT, fill=tk.X, expand=True)

        c_zoom = make_card(tab_tactics, "Viewport Zoom")
        zr = tk.Frame(c_zoom, bg=BG_APP)
        zr.pack(fill=tk.X)
        tk.Button(zr, text="-", command=lambda: self._adjust_zoom(1 / 1.25), bg=BG_ELEVATED, fg=TEXT_PRIMARY, relief=tk.FLAT, padx=10, pady=3, cursor="hand2").pack(side=tk.LEFT)
        self.lbl_zoom = tk.Label(zr, text="100%", bg=BG_APP, fg=TEXT_PRIMARY, width=6, font=("Segoe UI", 9, "bold"))
        self.lbl_zoom.pack(side=tk.LEFT, expand=True)
        tk.Button(zr, text="+", command=lambda: self._adjust_zoom(1.25), bg=BG_ELEVATED, fg=TEXT_PRIMARY, relief=tk.FLAT, padx=10, pady=3, cursor="hand2").pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(zr, text="Fit", command=self._reset_view, bg=BG_ELEVATED, fg=ACCENT_BLUE, relief=tk.FLAT, padx=10, pady=3, cursor="hand2").pack(side=tk.LEFT)

        # === TAB 3: RAID & REPLAY ===
        c_sync = make_card(tab_raid, "Multiplayer Raid Sync")
        tk.Button(c_sync, text="Raid Sync... (Host / Join)", command=lambda: RaidSyncDialog(self), bg=ACCENT_PURPLE, fg="white", relief=tk.FLAT, padx=10, pady=6, font=("Segoe UI", 9, "bold"), cursor="hand2").pack(fill=tk.X)
        self.lbl_roster_hdr = tk.Label(c_sync, text="Guildmates (0)", bg=BG_APP, fg=TEXT_MUTED, font=("Segoe UI", 8, "bold"), anchor=tk.W)
        self.lbl_roster_hdr.pack(fill=tk.X, pady=(8, 2))
        self.roster_tree = ttk.Treeview(c_sync, columns=("role", "zone", "state"), show="tree headings", height=8)
        self.roster_tree.heading("#0", text="Raider")
        self.roster_tree.heading("role", text="Role")
        self.roster_tree.heading("zone", text="Zone")
        self.roster_tree.heading("state", text="HP")
        self.roster_tree.column("#0", width=90)
        self.roster_tree.column("role", width=36, anchor=tk.CENTER)
        self.roster_tree.column("zone", width=90)
        self.roster_tree.column("state", width=44, anchor=tk.CENTER)
        self.roster_tree.pack(fill=tk.BOTH, expand=True)

        c_ana = make_card(tab_raid, "Combat Analytics")
        tk.Button(c_ana, text="Combat Stats...", command=lambda: CombatStatsDialog(self.root, self.wow_logs_dir), bg=BG_ELEVATED, fg=TEXT_PRIMARY, relief=tk.FLAT, padx=10, pady=5, font=("Segoe UI", 9), cursor="hand2").pack(fill=tk.X)

        # --- Right Viewport + Bottom Status & Scrubber ---
        viewport = tk.Frame(body, bg=BG_APP)
        viewport.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.map_canvas = tk.Canvas(viewport, bg=BG_APP, highlightthickness=0, cursor="crosshair")
        self.map_canvas.pack(fill=tk.BOTH, expand=True)
        self.map_canvas.bind("<Configure>", lambda _e: self._schedule_render(interacting=True))
        self.map_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.map_canvas.bind("<ButtonPress-1>", self._on_lpress)
        self.map_canvas.bind("<B1-Motion>", self._on_ldrag)
        self.map_canvas.bind("<ButtonRelease-1>", self._on_lrelease)
        self.map_canvas.bind("<ButtonPress-3>", self._on_rpress)
        self.map_canvas.bind("<B3-Motion>", self._on_rdrag)
        self.map_canvas.bind("<ButtonRelease-3>", lambda _e: setattr(self, "_rdrag_last", None))

        # Bottom Synced Timeline Scrubber + Status Bar
        bottom = tk.Frame(viewport, bg=BG_SURFACE, highlightbackground=BORDER_COLOR, highlightthickness=1)
        bottom.pack(fill=tk.X)

        scrub = tk.Frame(bottom, bg=BG_SURFACE, padx=10, pady=4)
        scrub.pack(fill=tk.X)
        self.replay_var = tk.BooleanVar(value=False)
        tk.Checkbutton(scrub, text="Replay Scrub", variable=self.replay_var, command=self._on_replay_toggle, bg=BG_SURFACE, fg="#3fb950", selectcolor=BG_ELEVATED, activebackground=BG_SURFACE, activeforeground="#3fb950", font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, padx=(0, 8))
        self.scrub_var = tk.DoubleVar(value=100.0)
        tk.Scale(scrub, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.scrub_var, command=self._on_scrub, showvalue=False, bg=BG_SURFACE, troughcolor=BG_ELEVATED, highlightthickness=0, sliderrelief=tk.FLAT, activebackground=ACCENT_BLUE).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_scrub = tk.Label(scrub, text="LIVE", bg=BG_SURFACE, fg=TEXT_PRIMARY, width=10, anchor=tk.E, font=("Consolas", 8, "bold"))
        self.lbl_scrub.pack(side=tk.RIGHT, padx=(6, 0))

        self.lbl_stats = tk.Label(bottom, text="Offline Solo | Use Ruler/Draw tools or click 'Raid Sync...' to Host/Join", bg=BG_SURFACE, fg=TEXT_MUTED, font=("Segoe UI", 8), anchor=tk.W, padx=10, pady=3)
        self.lbl_stats.pack(fill=tk.X)

        self._tk_img = None

    # --- Networking Hooks ---
    def get_local_lan_ip(self) -> str:
        if self.server is not None:
            ip = getattr(self.server, "local_ip", None)
            if ip:
                return str(ip)
        return _detect_local_lan_ip()

    def get_network_status_summary(self) -> str:
        if self.server and self.server.is_running:
            cnt = getattr(self.server, "connected_count", len(self.server._clients) if hasattr(self.server, "_clients") else 0)
            return f"Hosting on {self.get_local_lan_ip()}:{self.server.config.port} ({cnt} connected, {len(self.raid_players)} synced)"
        if self.client:
            if getattr(self.client, "connected", False):
                role = getattr(self.client, "assigned_role", "raider")
                perm = getattr(self.client, "whiteboard_permission", "all")
                return f"Connected to {self.client.url} ({role} | WB: {perm} | offset: {self.client.clock_offset_ms}ms)"
            return f"Connecting to {self.client.url}..."
        return "Offline Solo"

    def _can_edit_whiteboard(self) -> bool:
        if self.client is not None:
            can_draw_fn = getattr(self.client, "can_draw", None)
            if callable(can_draw_fn):
                return bool(can_draw_fn())
            perm = getattr(self.client, "whiteboard_permission", "all")
            role = getattr(self.client, "assigned_role", "raider")
            if perm == "all":
                return True
            if perm == "officers":
                return role in ("officer", "leader")
            return role == "leader"
        return True

    def start_host(
        self,
        port: int,
        passphrase: str,
        whiteboard_permission: str = "all",
        officer_token: str = "",
    ) -> bool:
        self.stop_network()
        self.last_net_error = None
        self.host_whiteboard_perm = whiteboard_permission
        self.server = ForeverNavigatorServer(
            HostConfig(
                port=port,
                room_passphrase=passphrase,
                officer_token=officer_token,
                whiteboard_permission=whiteboard_permission,
            ),
            on_raid_state=self._on_raid_state,
            on_draw_op=self._on_remote_draw,
            on_log=lambda m: self._ui_queue.put(("log", m)),
            on_status=lambda ok, m: self._ui_queue.put(("status", (ok, m))),
        )
        ok = self.server.start()
        if not ok:
            self.last_net_error = getattr(self.server, "last_error", None) or f"Failed to bind port {port}"
            self.lbl_stats.config(text=self.last_net_error)
            self.server = None
            return False
        self._net_connected = True
        self.lbl_stats.config(text=f"Hosting on {self.get_local_lan_ip()}:{port} (WB: {whiteboard_permission})")
        return True

    def start_client(self, url: str, passphrase: str, officer_token: str = "") -> bool:
        self.stop_network()
        self.last_net_error = None
        self.client = ForeverNavigatorClient(
            url=url,
            identity=self.identity,
            prefs=self.sync_prefs,
            room_passphrase=passphrase,
            officer_token=officer_token,
            on_raid_state=self._on_raid_state,
            on_draw_op=self._on_remote_draw,
            on_log=lambda m: self._ui_queue.put(("log", m)),
            on_status=lambda ok, m: self._ui_queue.put(("status", (ok, m))),
        )
        ok = self.client.start()
        if not ok:
            self.last_net_error = getattr(self.client, "last_error", None) or "Failed to start client"
            self.lbl_stats.config(text=self.last_net_error)
            self.client = None
            return False
        self.lbl_stats.config(text=f"Connecting to {url}...")
        return True

    def stop_network(self) -> None:
        if self.server:
            self.server.stop()
            self.server = None
        if self.client:
            self.client.stop()
            self.client = None
        self._net_connected = False
        self.raid_players.clear()
        if hasattr(self, "lbl_stats") and not self._closing:
            self.lbl_stats.config(text="Offline Solo | Use Ruler/Draw tools or click 'Raid Sync...' to Host/Join")
            self._render_current_map(self._latest_est, self._latest_mm_crop)

    def _on_raid_state(self, players: Dict[str, PlayerState]) -> None:
        self._ui_queue.put(("raid_state", players))

    def _on_remote_draw(self, msg: dict) -> None:
        self._ui_queue.put(("draw_op", msg))

    def _drain_ui_queue(self) -> None:
        if self._closing:
            return
        needs_render = False
        while True:
            try:
                kind, payload = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "raid_state":
                self.raid_players = payload
                needs_render = True
            elif kind == "draw_op":
                self.whiteboard.apply_remote_dict(payload)
                needs_render = True
            elif kind == "log":
                self.lbl_stats.config(text=str(payload))
            elif kind == "status":
                ok, msg = payload
                self._net_connected = bool(ok)
                if not ok:
                    self.last_net_error = str(msg)
                self.lbl_stats.config(text=str(msg))
                needs_render = True
            elif kind == "continent_loaded":
                folder, cont_d = payload
                self._continent_loading.discard(folder)
                if cont_d is not None:
                    # Keep only 1 lightweight continent backdrop in RAM at a time
                    for old_f in list(self._continent_maps.keys()):
                        if old_f != folder:
                            self._continent_maps.pop(old_f, None)
                            self._evict_cache_key(f"__cont_{old_f}")
                    self._continent_maps[folder] = cont_d
                    self._cached_pil_maps[f"__cont_{folder}"] = Image.fromarray(cont_d.map_image)
                    self._cached_mips.pop(f"__cont_{folder}", None)
                    needs_render = True
            elif kind == "zone_fullres_loaded":
                slug, pil_img = payload
                self._streaming_slugs.discard(slug)
                if pil_img is not None:
                    self._put_display_pil(slug, pil_img)
                    if slug == self._active_slug():
                        needs_render = True
            elif kind == "azeroth_overview_loaded":
                ov = payload
                if ov is not None:
                    if self._azeroth_overview is not None:
                        self._azeroth_overview.close()
                    self._cached_mips.pop("__azeroth_ov", None)
                    self._cached_np_mips.pop("__azeroth_ov", None)
                    self._azeroth_overview = ov
                    self._azeroth_overview_arr = np.asarray(ov.image, dtype=np.uint8)
                    self._overview_pan_x = ov.width * 0.5
                    self._overview_pan_y = ov.height * 0.5
                    self._overview_zoom = 1.0
                    if self._showing_azeroth_overview:
                        needs_render = True
            elif kind == "cont_l1_loaded":
                cont_slug, res = payload
                self._cont_l1_loading.discard(cont_slug)
                if res is not None:
                    # Keep only 1 Level-1 4096px continent array in RAM at a time
                    for old_s in list(self._cont_l1_cache.keys()):
                        if old_s != cont_slug:
                            self._cont_l1_cache.pop(old_s, None)
                            self._cached_np_mips.pop(f"__l1_{old_s}", None)
                    self._cont_l1_cache[cont_slug] = res
                    self._cached_np_mips.pop(f"__l1_{cont_slug}", None)
                    if self._showing_azeroth_overview:
                        needs_render = True
            elif kind == "deep_tile_ready":
                req_id, key, vp_img = payload
                if req_id == self._deep_tile_req_id and vp_img is not None and not self._is_interacting:
                    self._deep_tile_view = (key, vp_img)
                    needs_render = True
            elif kind == "map_cached_register":
                slug, dungeon = payload
                if dungeon is not None and slug not in self.tracker_3d.dungeons:
                    if hasattr(self.tracker_3d, "register_dungeon"):
                        self.tracker_3d.register_dungeon(dungeon)
                    else:
                        self.tracker_3d.dungeons[slug] = dungeon
                    if hasattr(self.tracker_mm, "register_dungeon"):
                        self.tracker_mm.register_dungeon(dungeon)
                    else:
                        self.tracker_mm.dungeons[slug] = dungeon
            elif kind == "map_loaded":
                slug, dungeon, err = payload
                self._map_loading_slug = None
                if err is not None or dungeon is None:
                    self.lbl_map_status.config(text=f"Load failed: {err}")
                    self.lbl_stats.config(text=f"Map load error ({slug}): {err}")
                else:
                    if hasattr(self.tracker_3d, "register_dungeon"):
                        self.tracker_3d.register_dungeon(dungeon)
                    else:
                        self.tracker_3d.dungeons[slug] = dungeon
                    if hasattr(self.tracker_mm, "register_dungeon"):
                        self.tracker_mm.register_dungeon(dungeon)
                    else:
                        self.tracker_mm.dungeons[slug] = dungeon
                    self.tracker_mm.reset()
                    self.tracker_mm.locked_dungeon_slug = slug
                    self.tracker_mm.active_slug = slug
                    self.dungeon_var.set(dungeon.name)
                    self._latest_est = None
                    self._cancel_fix_transition()
                    self._cancel_smooth_zoom()
                    self._showing_azeroth_overview = False
                    entry = self.world_maps.get_entry(slug) if self.world_maps is not None else None
                    if entry is not None and entry.is_shared:
                        self.lbl_map_status.config(text=f"Loaded {dungeon.name} (Shared Cave/Building)")
                    else:
                        self.lbl_map_status.config(text=f"Loaded {dungeon.name}")
                    self._reset_view()
                    needs_render = True
        if needs_render:
            self._render_current_map(self._latest_est, self._latest_mm_crop)
        if not self._closing:
            self.root.after(50, self._drain_ui_queue)

    def _broadcast_draw(self, op_dict: dict) -> None:
        if self.server:
            self.server.broadcast_draw_op_sync(op_dict)
        elif self.client:
            self.client.send_draw_op_sync(op_dict)

    # --- Coordinate Helpers ---
    def _active_slug(self) -> str:
        return self._latest_est.dungeon_slug if self._latest_est else self.tracker_mm.active_slug

    def _eff_scale(self, cw: int, ch: int, map_h: int) -> float:
        return (min(max(64, cw), max(64, ch)) / float(map_h)) * self.zoom

    def _canvas_to_uv(self, sx: float, sy: float) -> tuple[float, float]:
        cw, ch = max(1, self.map_canvas.winfo_width()), max(1, self.map_canvas.winfo_height())
        map_h = self.tracker_3d.dungeons[self._active_slug()].map_size_px
        sc = self._eff_scale(cw, ch, map_h)
        mx = self.pan_x + (sx - cw * 0.5) / sc
        my = self.pan_y + (sy - ch * 0.5) / sc
        return mx / float(map_h), my / float(map_h)

    # --- Interactive Tool Handlers ---
    def _warn_whiteboard_restricted(self) -> None:
        perm = getattr(self.client, "whiteboard_permission", "leader") if self.client else "leader"
        self.lbl_stats.config(text=f"Whiteboard editing restricted by Host (permission: {perm})")

    def _on_lpress(self, event) -> None:
        self._drag_last = (event.x, event.y)
        if self.view_3d_mode:
            return
        tool = self.tool_var.get()
        slug = self._active_slug()
        uv = self._canvas_to_uv(event.x, event.y)
        if tool == "Ruler (yd)":
            self.active_ruler = RulerMeasurement(uv, uv, 0.0)
        elif tool == "Draw":
            if not self._can_edit_whiteboard():
                self._warn_whiteboard_restricted()
                self._active_stroke_pts = []
                return
            self._active_stroke_pts = [uv]
        elif tool == "Text":
            if not self._can_edit_whiteboard():
                self._warn_whiteboard_restricted()
                return
            txt = simpledialog.askstring("Map Annotation", "Enter tactical note:", parent=self.root)
            if txt:
                op = WhiteboardOp(author_id=self.identity.player_id, dungeon_slug=slug, kind="text", points=[uv], text=txt, color="#ffe066")
                self.whiteboard.add_op(op)
                self._broadcast_draw(op.to_dict())
                self._render_current_map(self._latest_est, self._latest_mm_crop)
        elif tool.startswith("Range "):
            if not self._can_edit_whiteboard():
                self._warn_whiteboard_restricted()
                return
            yd = float(tool.replace("Range ", "").replace("yd", ""))
            col = "#ed8936" if yd <= 10 else ("#4299e1" if yd <= 30 else "#48bb78")
            op = WhiteboardOp(author_id=self.identity.player_id, dungeon_slug=slug, kind="circle_range", points=[uv], radius_yd=yd, color=col)
            self.whiteboard.add_op(op)
            self._broadcast_draw(op.to_dict())
            self._render_current_map(self._latest_est, self._latest_mm_crop)

    def _on_ldrag(self, event) -> None:
        if self._drag_last is None:
            return
        dx, dy = event.x - self._drag_last[0], event.y - self._drag_last[1]
        self._drag_last = (event.x, event.y)
        if self.view_3d_mode:
            self.viewer_3d.orbit(dx, dy)
            self._schedule_render(interacting=True)
            return

        if self._showing_azeroth_overview:
            self._cancel_fix_transition()
            self._cancel_smooth_zoom()
            self._cancel_smooth_zoom()
            self._deep_tile_view = None
            if self._azeroth_overview is not None:
                cw, ch = max(1, self.map_canvas.winfo_width()), max(1, self.map_canvas.winfo_height())
                sc_ov = self._overview_eff_scale(cw, ch)
                self._overview_pan_x = float(np.clip(self._overview_pan_x - dx / sc_ov, 0, self._azeroth_overview.width))
                self._overview_pan_y = float(np.clip(self._overview_pan_y - dy / sc_ov, 0, self._azeroth_overview.height))
                self._schedule_render(interacting=True)
            return

        tool = self.tool_var.get()
        d = self.tracker_3d.dungeons[self._active_slug()]
        map_h = d.map_size_px
        if tool == "Pan":
            if self.follow_var.get():
                self.follow_var.set(False)
            cw, ch = max(1, self.map_canvas.winfo_width()), max(1, self.map_canvas.winfo_height())
            sc = self._eff_scale(cw, ch, map_h)
            self.pan_x = float(np.clip(self.pan_x - dx / sc, 0, map_h))
            self.pan_y = float(np.clip(self.pan_y - dy / sc, 0, map_h))
            self._schedule_render(interacting=True)
        elif tool == "Ruler (yd)" and self.active_ruler is not None:
            uv = self._canvas_to_uv(event.x, event.y)
            yd = measure_yards(*self.active_ruler.start_uv, *uv, map_h, d.map_scale)
            self.active_ruler = RulerMeasurement(self.active_ruler.start_uv, uv, yd)
            self._schedule_render(interacting=True)
        elif tool == "Draw":
            if not self._active_stroke_pts:
                return
            self._active_stroke_pts.append(self._canvas_to_uv(event.x, event.y))
            self._schedule_render(interacting=True)

    def _on_lrelease(self, _event) -> None:
        self._drag_last = None
        if self.tool_var.get() == "Draw" and len(self._active_stroke_pts) >= 2:
            if not self._can_edit_whiteboard():
                self._active_stroke_pts.clear()
                return
            op = WhiteboardOp(author_id=self.identity.player_id, dungeon_slug=self._active_slug(), kind="stroke", points=list(self._active_stroke_pts), color="#ffe066", width=3)
            self.whiteboard.add_op(op)
            self._broadcast_draw(op.to_dict())
            self._active_stroke_pts.clear()
            self._render_current_map(self._latest_est, self._latest_mm_crop)

    def _on_rpress(self, event) -> None:
        self._rdrag_last = (event.x, event.y)

    def _on_rdrag(self, event) -> None:
        if not self.view_3d_mode or self._rdrag_last is None:
            return
        dx, dy = event.x - self._rdrag_last[0], event.y - self._rdrag_last[1]
        self._rdrag_last = (event.x, event.y)
        self.follow_var.set(False)
        self.viewer_3d.pan(dx, dy, self._active_slug())
        self._schedule_render(interacting=True)

    def _undo_draw(self) -> None:
        if not self._can_edit_whiteboard():
            self._warn_whiteboard_restricted()
            return
        slug = self._active_slug()
        self.whiteboard.undo(slug)
        self._broadcast_draw({"kind": "undo", "dungeon_slug": slug})
        self._render_current_map(self._latest_est, self._latest_mm_crop)

    def _clear_board(self) -> None:
        if not self._can_edit_whiteboard():
            self._warn_whiteboard_restricted()
            return
        slug = self._active_slug()
        self.whiteboard.clear(slug)
        self.active_ruler = None
        self._broadcast_draw({"kind": "clear_all", "dungeon_slug": slug})
        self._render_current_map(self._latest_est, self._latest_mm_crop)

    def _toggle_3d_view(self) -> None:
        self.view_3d_mode = not self.view_3d_mode
        self.btn_view3d.config(text="View: 3D Model (GPU)" if self.view_3d_mode else "View: 2D Map", bg=ACCENT_PURPLE if self.view_3d_mode else BG_ELEVATED)
        self._render_current_map(self._latest_est, self._latest_mm_crop)

    # --- All Maps (WoW: Forever) Browser & Async Loader ---
    def _refresh_map_catalog(self) -> None:
        if self.world_maps is not None:
            try:
                self._all_map_entries = list(self.world_maps.list_all_maps())
            except Exception:
                self._all_map_entries = []
        if not self._all_map_entries:
            from types import SimpleNamespace

            self._all_map_entries = [
                SimpleNamespace(
                    slug=slug,
                    name=d.name,
                    category="Dungeons",
                    cached=True,
                )
                for slug, d in self.tracker_3d.dungeons.items()
            ]
        self._filter_map_list()

    def _filter_map_list(self) -> None:
        cat = self.map_cat_var.get()
        q = self.map_search_var.get().strip().lower()
        filtered = []
        for entry in self._all_map_entries:
            if cat != "All" and getattr(entry, "category", "") != cat:
                continue
            if q and q not in entry.name.lower() and q not in entry.slug.lower():
                continue
            filtered.append(entry)
        self._filtered_map_entries = filtered
        self.map_listbox.delete(0, tk.END)
        for entry in filtered:
            tag = getattr(entry, "category", "Map")[:3].upper()
            self.map_listbox.insert(tk.END, f"[{tag}] {entry.name}")
        self.lbl_map_status.config(text=f"{len(filtered)} / {len(self._all_map_entries)} maps")

    def _set_auto_detect(self) -> None:
        self.dungeon_var.set("Auto-Detect")
        self.tracker_mm.locked_dungeon_slug = None
        self._latest_est = None
        self._cancel_fix_transition()
        self._cancel_smooth_zoom()
        self._showing_azeroth_overview = True
        self._overview_zoom = 1.0
        if self._azeroth_overview is not None:
            self._overview_pan_x = self._azeroth_overview.width * 0.5
            self._overview_pan_y = self._azeroth_overview.height * 0.5
        self.lbl_map_status.config(text="Auto-Detect enabled")
        self._reset_view()

    def _load_azeroth_overview_async(self) -> None:
        """Load the precached 4K Kalimdor + Eastern Kingdoms world overview on a background thread."""

        def _worker() -> None:
            try:
                from ..navigator.world_maps import build_azeroth_overview

                ov = build_azeroth_overview(self.world_maps, width=3840, height=2160, use_disk_cache=True)
                if not self._closing:
                    self._ui_queue.put(("azeroth_overview_loaded", ov))
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True, name="BuildAzerothOverview").start()

    def _overview_eff_scale(self, cw: int, ch: int) -> float:
        ov_w = float(self._azeroth_overview.width) if self._azeroth_overview is not None else 3840.0
        ov_h = float(self._azeroth_overview.height) if self._azeroth_overview is not None else 2160.0
        base_sc = min(max(64, cw) / ov_w, max(64, ch) / ov_h)
        return base_sc * self._overview_zoom

    @staticmethod
    def _is_good_position_fix(est: Optional[TrackerEstimate]) -> bool:
        if est is None:
            return False
        return (est.position_confidence >= 0.42) or (
            est.dungeon_confidence >= 0.55 and est.position_confidence >= 0.35
        )

    def _cancel_fix_transition(self) -> None:
        if self._transition_after_id is not None:
            try:
                self.root.after_cancel(self._transition_after_id)
            except Exception:
                pass
            self._transition_after_id = None
        self._transition_state = None

    def _start_fix_transition(self, est: TrackerEstimate) -> None:
        """Smoothly zoom from the Azeroth overview directly into the player's detected zone & position."""
        self._cancel_fix_transition()
        self._cancel_smooth_zoom()
        d = self.tracker_3d.dungeons.get(est.dungeon_slug)
        if d is None:
            self._showing_azeroth_overview = False
            self._render_current_map(est, self._latest_mm_crop)
            return

        # Start streaming the target zone's full-res map from SSD while the zoom-in plays
        self._ensure_fullres_display(est.dungeon_slug, d)

        if self._azeroth_overview is None:
            self._showing_azeroth_overview = False
            self.pan_x, self.pan_y = float(est.map_px[0]), float(est.map_px[1])
            self.zoom = 1.4
            self.lbl_zoom.config(text=f"{int(round(self.zoom * 100))}%")
            self._render_current_map(est, self._latest_mm_crop)
            return

        entry = self.world_maps.get_entry(est.dungeon_slug) if self.world_maps is not None else None
        folder = entry.minimap_folder if entry is not None else None
        tx_ov, ty_ov = self._azeroth_overview.world_to_overview_px(
            folder, float(est.world_pos[0]), float(est.world_pos[1])
        )
        self._transition_state = {
            "t0": time.perf_counter(),
            "duration": 0.70,
            "start_pan": (self._overview_pan_x, self._overview_pan_y),
            "target_ov": (tx_ov, ty_ov),
            "start_zoom": self._overview_zoom,
            "est": est,
        }
        self._is_interacting = True
        self._step_fix_transition()

    def _step_fix_transition(self) -> None:
        self._transition_after_id = None
        st = self._transition_state
        if st is None or self._closing:
            return
        elapsed = time.perf_counter() - st["t0"]
        t = min(1.0, elapsed / st["duration"])
        est: TrackerEstimate = self._latest_est or st["est"]
        split = 0.62

        if t < split:
            u = t / split
            ease = 1.0 - (1.0 - u) ** 3
            sx, sy = st["start_pan"]
            tx, ty = st["target_ov"]
            self._overview_pan_x = sx + (tx - sx) * ease
            self._overview_pan_y = sy + (ty - sy) * ease
            self._overview_zoom = st["start_zoom"] * (6.5 ** ease)
            self._showing_azeroth_overview = True
            self._render_current_map(est, self._latest_mm_crop)
        else:
            u = (t - split) / max(1e-4, 1.0 - split)
            ease = 1.0 - (1.0 - u) ** 3
            self._showing_azeroth_overview = False
            self.pan_x = float(est.map_px[0])
            self.pan_y = float(est.map_px[1])
            self.zoom = 0.65 + (1.45 - 0.65) * ease
            self.lbl_zoom.config(text=f"{int(round(self.zoom * 100))}%")
            self._render_current_map(est, self._latest_mm_crop)

        if t < 1.0:
            self._transition_after_id = self.root.after(16, self._step_fix_transition)
        else:
            self._transition_state = None
            self._is_interacting = False
            self._render_current_map(est, self._latest_mm_crop)

    def _on_follow_toggle(self) -> None:
        enabled = bool(self.follow_var.get())
        self.settings.follow_player = enabled
        save_settings(self.settings_path, self.settings)
        if enabled and self._latest_est is not None and not self.replay_mode:
            self.pan_x = float(self._latest_est.map_px[0])
            self.pan_y = float(self._latest_est.map_px[1])
            self._render_current_map(self._latest_est, self._latest_mm_crop)

    def _on_storage_mode_change(self) -> None:
        lbl = self.storage_var.get()
        mode = self._storage_label_to_mode.get(lbl, "auto")
        self.settings.storage_mode = mode
        save_settings(self.settings_path, self.settings)
        self._effective_storage_mode = resolve_storage_mode(mode, self.data_dir)
        if self.world_maps is not None and hasattr(self.world_maps, "set_storage_mode"):
            self.world_maps.set_storage_mode(self._effective_storage_mode)
        self._trim_display_cache(keep_slug=self._active_slug())
        self.lbl_stats.config(text=f"Storage streaming: {self._effective_storage_mode.upper()} mode")

    def _display_cache_limit(self) -> int:
        return 3 if self._effective_storage_mode == "hdd" else 1

    def _evict_cache_key(self, key: str) -> None:
        im = self._cached_pil_maps.pop(key, None)
        mips = self._cached_mips.pop(key, None)
        if hasattr(self, "_cached_np_mips"):
            self._cached_np_mips.pop(key, None)
        if mips:
            for _div, mip_im in mips:
                if mip_im is not im:
                    try:
                        mip_im.close()
                    except Exception:
                        pass
        if im is not None:
            try:
                im.close()
            except Exception:
                pass

    @staticmethod
    def _warp_view_cv2(
        arr_rgb: np.ndarray,
        pan_x: float,
        pan_y: float,
        scale: float,
        cw: int,
        ch: int,
        fillcolor: tuple[int, int, int],
    ) -> Image.Image:
        """Ultra-fast AVX2 C++ viewport affine blit (~3-4 ms at 1200x800 vs ~22 ms in PIL)."""
        tx = cw * 0.5 - pan_x * scale
        ty = ch * 0.5 - pan_y * scale
        m = np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)
        out = cv2.warpAffine(
            arr_rgb,
            m,
            (cw, ch),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=fillcolor,
        )
        return Image.fromarray(out)

    def _get_np_mip_for_scale(
        self, key: str, full_arr: np.ndarray, target_scale: float
    ) -> tuple[np.ndarray, int]:
        """Return (np_mip_array, divisor) capped to viewport resolution for 60 FPS OpenCV blits."""
        mips = self._cached_np_mips.get(key)
        if mips is None or mips[0][1] is not full_arr:
            mips = [(1, full_arr)]
            cur = full_arr
            div = 1
            while min(cur.shape[0], cur.shape[1]) > 768 and div < 16:
                nw = max(256, cur.shape[1] // 2)
                nh = max(256, cur.shape[0] // 2)
                cur = cv2.resize(cur, (nw, nh), interpolation=cv2.INTER_AREA)
                div *= 2
                mips.append((div, cur))
            self._cached_np_mips[key] = mips

        chosen_div, chosen_arr = mips[0]
        for div, arr in mips:
            if target_scale * div <= 1.05:
                chosen_div, chosen_arr = div, arr
            else:
                break
        return chosen_arr, chosen_div

    def _ensure_continent_l1_async(self, cont_slug: str) -> None:
        if self.world_maps is None or cont_slug in self._cont_l1_cache or cont_slug in self._cont_l1_loading:
            return
        self._cont_l1_loading.add(cont_slug)

        def _worker(slug: str = cont_slug) -> None:
            res = None
            try:
                res = self.world_maps.load_continent_l1_array(slug, max_px=4096)
            except Exception:
                res = None
            if not self._closing:
                self._ui_queue.put(("cont_l1_loaded", (slug, res)))

        threading.Thread(target=_worker, daemon=True, name=f"LoadL1-{cont_slug}").start()

    def _request_deep_tiles_async(
        self, cont_slug: str, cx_cont: float, cy_cont: float, sc_cont: float, cw: int, ch: int
    ) -> None:
        key = (cont_slug, int(round(cx_cont)), int(round(cy_cont)), round(sc_cont, 3), cw, ch)
        if self._deep_tile_view is not None and self._deep_tile_view[0] == key:
            return
        self._deep_tile_req_id += 1
        req_id = self._deep_tile_req_id

        def _worker() -> None:
            vp = None
            try:
                if self.world_maps is not None and self.world_maps.ensure_full_res_tiles(cont_slug):
                    vp = self.world_maps.render_full_res_viewport(cont_slug, cx_cont, cy_cont, sc_cont, cw, ch)
            except Exception:
                vp = None
            if not self._closing:
                self._ui_queue.put(("deep_tile_ready", (req_id, key, vp)))

        threading.Thread(target=_worker, daemon=True, name=f"DeepTile-{req_id}").start()

    def _trim_display_cache(self, keep_slug: Optional[str] = None) -> None:
        limit = self._display_cache_limit()
        zone_keys = [k for k in self._cached_pil_maps.keys() if not k.startswith("__cont_")]
        while len(zone_keys) > limit:
            evict_k = next((k for k in zone_keys if k != keep_slug), zone_keys[0])
            zone_keys.remove(evict_k)
            self._evict_cache_key(evict_k)

    def _put_display_pil(self, slug: str, pil_img: Image.Image) -> None:
        if slug in self._cached_pil_maps:
            self._evict_cache_key(slug)
        self._cached_pil_maps[slug] = pil_img
        self._cached_pil_maps.move_to_end(slug)
        self._trim_display_cache(keep_slug=slug)

    def _ensure_fullres_display(self, slug: str, d: Any) -> Image.Image:
        """Return the best resident PIL image for `slug`, streaming full-res from storage on-demand."""
        cached = self._cached_pil_maps.get(slug)
        if cached is not None:
            self._cached_pil_maps.move_to_end(slug)
            return cached

        # Kick off async streaming from disk when `d.map_image` is a low-RAM preview
        full_path = getattr(d, "map_full_path", None)
        if (
            full_path is not None
            and d.map_image.shape[0] < d.map_size_px
            and slug not in self._streaming_slugs
        ):
            self._streaming_slugs.add(slug)

            def _stream_worker(p: Path = Path(full_path), target_slug: str = slug) -> None:
                pil_im: Optional[Image.Image] = None
                try:
                    if p.suffix.lower() == ".png" and p.exists():
                        with Image.open(p) as raw_im:
                            pil_im = raw_im.convert("RGB")
                    elif p.exists():
                        with np.load(p, allow_pickle=False) as z:
                            pil_im = Image.fromarray(z["map_image"].astype(np.uint8))
                except Exception:
                    pil_im = None
                if not self._closing:
                    self._ui_queue.put(("zone_fullres_loaded", (target_slug, pil_im)))

            threading.Thread(target=_stream_worker, daemon=True, name=f"StreamMap-{slug}").start()

        # Return lightweight preview immediately (zero stall) without polluting the full-res LRU
        return Image.fromarray(d.map_image)

    def _toggle_high_precision(self) -> None:
        enabled = bool(self.high_precision_var.get())
        self.settings.high_precision = enabled
        save_settings(self.settings_path, self.settings)
        threading.Thread(
            target=lambda: self.tracker_mm.set_high_precision(enabled),
            daemon=True,
            name="ToggleHighPrecision",
        ).start()
        mode_lbl = "High Precision (CPU+GPU)" if enabled else "Standard"
        self.lbl_stats.config(text=f"Tracking mode: {mode_lbl}")

    def _load_selected_map(self) -> None:
        sel = self.map_listbox.curselection()
        if not sel:
            return
        idx = int(sel[0])
        if idx < 0 or idx >= len(self._filtered_map_entries):
            return
        entry = self._filtered_map_entries[idx]
        slug = entry.slug
        if slug in self.tracker_3d.dungeons:
            self.tracker_mm.reset()
            self.tracker_mm.locked_dungeon_slug = slug
            self.tracker_mm.active_slug = slug
            self.dungeon_var.set(entry.name)
            self._latest_est = None
            self._cancel_fix_transition()
            self._cancel_smooth_zoom()
            self._showing_azeroth_overview = False
            self.lbl_map_status.config(text=f"Active: {entry.name}")
            self._reset_view()
            return
        if self.world_maps is None:
            self.lbl_map_status.config(text=f"{entry.name} requires world_maps")
            return
        self._map_loading_slug = slug
        self.lbl_map_status.config(text=f"Loading {entry.name}...")
        self.world_maps.load_map_async(
            slug,
            lambda s, d, e: self._ui_queue.put(("map_loaded", (s, d, e))),
            preview_only=True,
        )

    def _preload_cached_maps(self) -> None:
        if self.world_maps is None:
            return

        def _worker() -> None:
            for entry in list(self._all_map_entries):
                if self._closing:
                    return
                if entry.cached and entry.slug not in self.tracker_3d.dungeons and entry.category != "Continents":
                    try:
                        d = self.world_maps.load_map(entry.slug, preview_only=True)
                        if not self._closing:
                            self._ui_queue.put(("map_cached_register", (entry.slug, d)))
                    except Exception:
                        pass

        threading.Thread(target=_worker, daemon=True, name="PreloadCachedMaps").start()

    def _ensure_continent_backdrop(self, folder: str) -> None:
        """Lazily load only a lightweight (1536px) continent backdrop for the currently viewed zone."""
        if self.world_maps is None or folder in self._continent_maps or folder in self._continent_loading:
            return
        cont_slug = "kalimdor" if folder == "Kalimdor" else ("eastern_kingdoms" if folder == "Azeroth" else None)
        if cont_slug is None:
            return
        self._continent_loading.add(folder)

        def _worker() -> None:
            try:
                cont_d = self.world_maps.load_map(cont_slug, preview_only=True, max_preview_px=1536)
                if not self._closing:
                    self._ui_queue.put(("continent_loaded", (folder, cont_d)))
            except Exception:
                self._continent_loading.discard(folder)

        threading.Thread(target=_worker, daemon=True, name=f"LoadCont-{folder}").start()

    def _get_mip_for_scale(self, key: str, full_pil: Image.Image, target_scale: float) -> tuple[Image.Image, int]:
        """Return (mip_image, divisor) capped to viewport resolution so we never oversample pixels."""
        mips = self._cached_mips.get(key)
        if mips is None or mips[0][1] is not full_pil:
            mips = [(1, full_pil)]
            cur = full_pil
            div = 1
            while min(cur.width, cur.height) > 768 and div < 16:
                nw, nh = max(256, cur.width // 2), max(256, cur.height // 2)
                cur = cur.resize((nw, nh), Image.Resampling.BILINEAR)
                div *= 2
                mips.append((div, cur))
            if not hasattr(self, "_cached_pil_maps") or key in self._cached_pil_maps or key == "__azeroth_ov":
                self._cached_mips[key] = mips

        chosen_div, chosen_im = mips[0]
        for div, im in mips:
            if target_scale * div <= 1.05:
                chosen_div, chosen_im = div, im
            else:
                break
        return chosen_im, chosen_div

    def _preload_all_zones(self) -> None:
        if self.world_maps is None:
            return
        self.lbl_map_status.config(text="Indexing WoW: Forever zones in background...")

        def _worker() -> None:
            done = 0
            targets = [e for e in self._all_map_entries if e.category in ("Zones", "Cities", "Battlegrounds", "Dungeons", "Raids") and e.slug not in self.tracker_3d.dungeons]
            for entry in targets:
                if self._closing:
                    return
                try:
                    d = self.world_maps.load_map(entry.slug, preview_only=True)
                    done += 1
                    if not self._closing:
                        self._ui_queue.put(("map_cached_register", (entry.slug, d)))
                        if done % 5 == 0 or done == len(targets):
                            self._ui_queue.put(("log", f"Auto-Detect indexed {done}/{len(targets)} maps"))
                except Exception:
                    pass
            if not self._closing:
                self._ui_queue.put(("log", f"Auto-Detect ready across {len(self.tracker_mm.dungeons)} maps!"))

        threading.Thread(target=_worker, daemon=True, name="IndexAllZones").start()

    def _update_live_pills(self, est: Optional[TrackerEstimate]) -> None:
        if self.server and self.server.is_running:
            cnt = getattr(self.server, "connected_count", 0)
            sync_txt, sync_fg = f"Host :{self.server.config.port} ({cnt})", "#3fb950"
        elif self.client:
            if getattr(self.client, "connected", False):
                sync_txt, sync_fg = f"Synced ({self.client.clock_offset_ms}ms)", "#3fb950"
            else:
                sync_txt, sync_fg = "Connecting...", ACCENT_GOLD
        else:
            sync_txt, sync_fg = "Offline Solo", ACCENT_BLUE

        if est is not None:
            pos_txt, pos_fg = f"Pos: ({est.world_pos[0]:.0f}, {est.world_pos[1]:.0f}) {est.position_confidence * 100:.0f}%", TEXT_PRIMARY
        else:
            pos_txt, pos_fg = "Pos: --", TEXT_MUTED
        fps_txt = f"{self.fps:.1f} Hz"

        sig = (sync_txt, sync_fg, pos_txt, pos_fg, fps_txt)
        if sig == self._last_pills_sig:
            return
        self._last_pills_sig = sig
        self.pill_sync.config(text=sync_txt, fg=sync_fg)
        self.pill_pos.config(text=pos_txt, fg=pos_fg)
        self.pill_fps.config(text=fps_txt)

    def _set_dungeon_label(self, text: str) -> None:
        if text != self._last_dungeon_lbl:
            self._last_dungeon_lbl = text
            self.lbl_dungeon.config(text=text)

    def _refresh_roster(self, shown_players: Dict[str, PlayerState]) -> None:
        if not hasattr(self, "roster_tree"):
            return
        sig = tuple(
            (pid, p.name, p.role, p.dungeon_slug, bool(getattr(p, "alive", True)))
            for pid, p in shown_players.items()
        )
        if sig == self._last_roster_sig:
            return
        self._last_roster_sig = sig
        for item in self.roster_tree.get_children():
            self.roster_tree.delete(item)
        for p in shown_players.values():
            alive_txt = "ALIVE" if getattr(p, "alive", True) else "DEAD"
            self.roster_tree.insert("", tk.END, text=p.name, values=(_role_badge(p.role), p.dungeon_slug or "-", alive_txt))
        self.lbl_roster_hdr.config(text=f"Guildmates ({len(shown_players)})")

    def _schedule_render(self, interacting: bool = False) -> None:
        """Coalesce rapid pan/zoom/resize events to ~60 FPS and defer full-res CASC tiles until settled."""
        if self._closing:
            return
        if interacting:
            self._is_interacting = True
            if self._settle_after_id is not None:
                try:
                    self.root.after_cancel(self._settle_after_id)
                except Exception:
                    pass
            self._settle_after_id = self.root.after(140, self._on_interaction_settled)
        if self._render_after_id is None:
            self._render_after_id = self.root.after(16, self._flush_scheduled_render)

    def _flush_scheduled_render(self) -> None:
        self._render_after_id = None
        if self._closing:
            return
        self._render_current_map(self._latest_est, self._latest_mm_crop)

    def _on_interaction_settled(self) -> None:
        self._settle_after_id = None
        if self._closing:
            return
        self._is_interacting = False
        self._render_current_map(self._latest_est, self._latest_mm_crop)

    def _cancel_smooth_zoom(self) -> None:
        if self._zoom_anim_id is not None:
            try:
                self.root.after_cancel(self._zoom_anim_id)
            except Exception:
                pass
            self._zoom_anim_id = None
        self._zoom_target = None
        self._zoom_anchor = None

    def _on_mousewheel(self, event) -> None:
        """Smooth, cursor-anchored zoom supporting both stepped clicks and free-rolling wheels."""
        if self._closing:
            return
        delta = float(getattr(event, "delta", 0))
        if abs(delta) < 1e-3:
            return
        # Proportional to wheel delta (free-rolling wheels send small deltas; standard clicks send +/-120)
        factor = 1.24 ** (delta / 120.0)
        self._cancel_fix_transition()
        self._deep_tile_view = None
        if self.view_3d_mode:
            base_z = self._zoom_target if self._zoom_target is not None else self.viewer_3d.zoom
            self._zoom_target = float(np.clip(base_z * factor, 0.4, 6.0))
        elif self._showing_azeroth_overview:
            base_z = self._zoom_target if self._zoom_target is not None else self._overview_zoom
            self._zoom_target = float(np.clip(base_z * factor, 0.5, 64.0))
        else:
            base_z = self._zoom_target if self._zoom_target is not None else self.zoom
            self._zoom_target = float(np.clip(base_z * factor, 0.10, 8.0))
        self._zoom_anchor = (float(event.x), float(event.y))
        self._is_interacting = True
        if self._settle_after_id is not None:
            try:
                self.root.after_cancel(self._settle_after_id)
            except Exception:
                pass
            self._settle_after_id = None
        if self._zoom_anim_id is None:
            self._zoom_anim_id = self.root.after(10, self._step_smooth_zoom)

    def _step_smooth_zoom(self) -> None:
        self._zoom_anim_id = None
        if self._closing or self._zoom_target is None:
            return
        cw = max(1, self.map_canvas.winfo_width())
        ch = max(1, self.map_canvas.winfo_height())
        ax, ay = self._zoom_anchor if self._zoom_anchor is not None else (cw * 0.5, ch * 0.5)
        target = self._zoom_target

        if self.view_3d_mode:
            cur = self.viewer_3d.zoom
            nxt = target if abs(target / max(1e-5, cur) - 1.0) < 0.005 else cur * ((target / cur) ** 0.34)
            self.viewer_3d.zoom = nxt
            self.lbl_zoom.config(text=f"{int(round(nxt * 100))}%")
            done = (nxt == target)
        elif self._showing_azeroth_overview:
            cur = self._overview_zoom
            old_sc = self._overview_eff_scale(cw, ch)
            nxt = target if abs(target / max(1e-5, cur) - 1.0) < 0.004 else cur * ((target / cur) ** 0.34)
            self._overview_zoom = nxt
            new_sc = self._overview_eff_scale(cw, ch)
            # Keep the world point under cursor (ax, ay) pinned while zooming
            if self._azeroth_overview is not None and old_sc > 1e-6 and new_sc > 1e-6:
                wx_a = self._overview_pan_x + (ax - cw * 0.5) / old_sc
                wy_a = self._overview_pan_y + (ay - ch * 0.5) / old_sc
                self._overview_pan_x = float(np.clip(wx_a - (ax - cw * 0.5) / new_sc, 0, self._azeroth_overview.width))
                self._overview_pan_y = float(np.clip(wy_a - (ay - ch * 0.5) / new_sc, 0, self._azeroth_overview.height))
            self.lbl_zoom.config(text=f"{int(round(nxt * 100))}%")
            done = (nxt == target)
        else:
            d = self.tracker_3d.dungeons[self._active_slug()]
            map_h = d.map_size_px
            cur = self.zoom
            old_sc = self._eff_scale(cw, ch, map_h)
            nxt = target if abs(target / max(1e-5, cur) - 1.0) < 0.004 else cur * ((target / cur) ** 0.34)
            self.zoom = nxt
            new_sc = self._eff_scale(cw, ch, map_h)
            if not self.follow_var.get() and old_sc > 1e-6 and new_sc > 1e-6:
                mx_a = self.pan_x + (ax - cw * 0.5) / old_sc
                my_a = self.pan_y + (ay - ch * 0.5) / old_sc
                self.pan_x = float(np.clip(mx_a - (ax - cw * 0.5) / new_sc, 0, map_h))
                self.pan_y = float(np.clip(my_a - (ay - ch * 0.5) / new_sc, 0, map_h))
            self.lbl_zoom.config(text=f"{int(round(nxt * 100))}%")
            done = (nxt == target)

        self._render_current_map(self._latest_est, self._latest_mm_crop)
        if not done:
            self._zoom_anim_id = self.root.after(16, self._step_smooth_zoom)
        else:
            self._zoom_target = None
            self._zoom_anchor = None
            self._settle_after_id = self.root.after(90, self._on_interaction_settled)

    def _adjust_zoom(self, factor: float) -> None:
        self._cancel_smooth_zoom()
        self._deep_tile_view = None
        if self.view_3d_mode:
            self.viewer_3d.zoom = float(np.clip(self.viewer_3d.zoom * factor, 0.4, 6.0))
            self.lbl_zoom.config(text=f"{int(round(self.viewer_3d.zoom * 100))}%")
        elif self._showing_azeroth_overview:
            self._cancel_fix_transition()
            self._cancel_smooth_zoom()
            self._overview_zoom = float(np.clip(self._overview_zoom * factor, 0.5, 64.0))
            self.lbl_zoom.config(text=f"{int(round(self._overview_zoom * 100))}%")
        else:
            self.zoom = float(np.clip(self.zoom * factor, 0.10, 8.0))
            self.lbl_zoom.config(text=f"{int(round(self.zoom * 100))}%")
        self._schedule_render(interacting=True)

    def _reset_view(self) -> None:
        if self.view_3d_mode:
            self.viewer_3d.reset_camera(self._active_slug())
        elif self._showing_azeroth_overview:
            self._overview_zoom = 1.0
            if self._azeroth_overview is not None:
                self._overview_pan_x = self._azeroth_overview.width * 0.5
                self._overview_pan_y = self._azeroth_overview.height * 0.5
        else:
            self.zoom = 1.0
            half = self.tracker_3d.dungeons[self._active_slug()].map_size_px * 0.5
            self.pan_x, self.pan_y = half, half
        self.lbl_zoom.config(text="100%")
        self._render_current_map(self._latest_est, self._latest_mm_crop)

    def _on_dungeon_select(self, _evt=None) -> None:
        sel = self.dungeon_var.get()
        if sel == "Auto-Detect":
            self.tracker_mm.locked_dungeon_slug = None
        else:
            for slug, d in self.tracker_3d.dungeons.items():
                if d.name == sel:
                    self.tracker_mm.reset()
                    self.tracker_mm.locked_dungeon_slug = slug
                    self.tracker_mm.active_slug = slug
                    break
        self._latest_est = None
        self._reset_view()

    def _pick_minimap_box(self) -> None:
        with mss.MSS() as sct:
            mon = dict(sct.monitors[min(self._selected_monitor, len(sct.monitors) - 1)])

        def on_roi(roi: MinimapROI):
            self.tracker_mm.roi = roi
            save_roi(self.roi_path, roi)
            self.lbl_stats.config(text=f"Saved Minimap ROI: ({roi.x0:.2f}, {roi.y0:.2f}) -> ({roi.x1:.2f}, {roi.y1:.2f})")

        ROIPickerDialog(self.root, mon, on_roi)

    def _on_replay_toggle(self) -> None:
        self.replay_mode = bool(self.replay_var.get())
        if not self.replay_mode:
            self.lbl_scrub.config(text="LIVE")
            self.replay_players.clear()
        self._render_current_map(self._latest_est, self._latest_mm_crop)

    def _on_scrub(self, val_str: str) -> None:
        if not self.replay_mode:
            return
        t0, t1 = self.timeline.bounds_ms()
        if t1 <= t0:
            self.lbl_scrub.config(text="No data")
            return
        frac = float(val_str) / 100.0
        target_ms = int(t0 + frac * (t1 - t0))
        snap = self.timeline.scrub_to(target_ms)
        self.replay_players = snap.players
        self.lbl_scrub.config(text=f"+{(target_ms - t0) / 1000.0:.1f}s")
        self._render_current_map(self._latest_est, self._latest_mm_crop)

    # --- Live Capture Loop ---
    def _toggle_capture(self) -> None:
        if self.running:
            self.running = False
            self._stop_capture_event.set()
            if self._poll_after_id is not None:
                try:
                    self.root.after_cancel(self._poll_after_id)
                except Exception:
                    pass
                self._poll_after_id = None
            self.btn_toggle.config(text="Start Capture", bg="#1f6feb")
        else:
            self.running = True
            self._stop_capture_event.clear()
            self.btn_toggle.config(text="Stop Capture", bg=ACCENT_RED)
            threading.Thread(target=self._capture_loop, daemon=True, name="CaptureLoop").start()
            if self._poll_after_id is None:
                self._poll_after_id = self.root.after(200, self._poll_ui)

    def _capture_loop(self) -> None:
        with mss.MSS() as sct:
            while self.running:
                t0 = time.perf_counter()
                interval = 1.0 / max(0.25, self.fps)
                try:
                    mon_idx = min(self._selected_monitor, len(sct.monitors) - 1)
                    mon = sct.monitors[mon_idx]
                    if self.dev_logger.enabled:
                        raw = sct.grab(mon)
                        frame_rgb = cv2.cvtColor(np.asarray(raw), cv2.COLOR_BGRA2RGB)
                        capture_ms = (time.perf_counter() - t0) * 1000.0
                        est = self.tracker_mm.update(frame_rgb)
                        self.dev_logger.log_tracker_tick(
                            est,
                            self.tracker_mm.locked_dungeon_slug,
                            capture_ms,
                            self.tracker_mm.last_crop_rgb,
                            frame_rgb,
                            self.tracker_mm.roi.to_dict(),
                        )
                    else:
                        roi = self.tracker_mm.roi
                        mw, mh = int(mon["width"]), int(mon["height"])
                        rx0 = int(mon["left"]) + max(0, int(round(roi.x0 * mw)))
                        ry0 = int(mon["top"]) + max(0, int(round(roi.y0 * mh)))
                        rw = max(16, int(round((roi.x1 - roi.x0) * mw)))
                        rh = max(16, int(round((roi.y1 - roi.y0) * mh)))
                        mm_raw = sct.grab({"left": rx0, "top": ry0, "width": rw, "height": rh})
                        mm_rgb = cv2.cvtColor(np.asarray(mm_raw), cv2.COLOR_BGRA2RGB)
                        est = self.tracker_mm.update(mm_rgb, is_already_cropped=True)
                    st = PlayerState(
                        player_id=self.identity.player_id,
                        name=self.identity.name,
                        wow_class=self.identity.wow_class,
                        role=self.identity.role,
                        dungeon_slug=est.dungeon_slug,
                        x=float(est.world_pos[0]),
                        y=float(est.world_pos[1]),
                        z=float(est.world_pos[2]),
                        map_px=est.map_px,
                        confidence=est.position_confidence,
                        t_host_ms=self.client.host_now_ms() if self.client else now_ms(),
                    )
                    self.session_store.record_position(st)
                    if self.server:
                        self.server.update_host_position_sync(st)
                    elif self.client:
                        self.client.send_position_sync(st)
                    with self._lock:
                        self._latest_est = est
                        self._latest_mm_crop = self.tracker_mm.last_crop_rgb
                except Exception as exc:
                    self.dev_logger.log_event("capture_error", {"error": str(exc)})
                # High-rate minimap ROI polling between full tracker ticks when dev logger is active
                while self.running:
                    elapsed = time.perf_counter() - t0
                    rem = interval - elapsed
                    if rem <= 0:
                        break
                    if self.dev_logger.enabled and rem > self.dev_logger.minimap_interval_sec:
                        if self._stop_capture_event.wait(timeout=self.dev_logger.minimap_interval_sec):
                            break
                        try:
                            mon_idx = min(self._selected_monitor, len(sct.monitors) - 1)
                            mon = sct.monitors[mon_idx]
                            roi = self.tracker_mm.roi
                            mw, mh = int(mon["width"]), int(mon["height"])
                            rx0 = int(mon["left"]) + max(0, int(round(roi.x0 * mw)))
                            ry0 = int(mon["top"]) + max(0, int(round(roi.y0 * mh)))
                            rw = max(16, int(round((roi.x1 - roi.x0) * mw)))
                            rh = max(16, int(round((roi.y1 - roi.y0) * mh)))
                            mm_raw = sct.grab({"left": rx0, "top": ry0, "width": rw, "height": rh})
                            mm_rgb = cv2.cvtColor(np.asarray(mm_raw), cv2.COLOR_BGRA2RGB)
                            self.dev_logger.log_minimap_roi_sample(mm_rgb, roi.to_dict())
                        except Exception as exc:
                            self.dev_logger.log_event("minimap_poll_error", {"error": str(exc)})
                    else:
                        self._stop_capture_event.wait(timeout=rem)
                        break

    def _poll_ui(self) -> None:
        self._poll_after_id = None
        if not self.running or self._closing:
            return
        with self._lock:
            est, mm = self._latest_est, self._latest_mm_crop
        if est is not None and est is not self._last_rendered_est and not self.replay_mode:
            self._last_rendered_est = est
            if self._showing_azeroth_overview and self._transition_state is None:
                if self._is_good_position_fix(est):
                    self._start_fix_transition(est)
                else:
                    self._render_current_map(est, mm)
            elif self._transition_state is None:
                self._render_current_map(est, mm)
        self._poll_after_id = self.root.after(200, self._poll_ui)

    # --- Rendering ---
    def _render_current_map(self, est: Optional[TrackerEstimate], mm_crop: Optional[np.ndarray] = None) -> None:
        cw, ch = self.map_canvas.winfo_width(), self.map_canvas.winfo_height()
        if cw < 32 or ch < 32:
            cw, ch = 780, 660
        slug = self._active_slug()
        d = self.tracker_3d.dungeons[slug]
        map_h = d.map_size_px
        self.map_canvas.delete("all")

        shown_players = self.replay_players if self.replay_mode else self.raid_players
        self._update_live_pills(est)
        self._refresh_roster(shown_players)

        # Startup / pre-fix 4K Azeroth Overview (3-level prestitched pyramid: 4K World -> 4K Continent L1 -> Async Native Tiles)
        if self._showing_azeroth_overview and not self.view_3d_mode and not self.replay_mode:
            if self._azeroth_overview is not None:
                sc_ov = self._overview_eff_scale(cw, ch)
                ov_view = None
                if sc_ov > 1.15 and self.world_maps is not None:
                    mapped = self._azeroth_overview.overview_px_to_continent_px(
                        self._overview_pan_x, self._overview_pan_y
                    )
                    if mapped is not None:
                        _m_folder, m_cont_slug, cx_cont, cy_cont, cont_per_ov = mapped
                        sc_cont = sc_ov / max(1e-6, cont_per_ov)
                        # Level 2: Async native 512px CASC tiles when deeply zoomed & settled (zero UI stall)
                        if not self._is_interacting and sc_cont >= 1.65:
                            tile_key = (m_cont_slug, int(round(cx_cont)), int(round(cy_cont)), round(sc_cont, 3), cw, ch)
                            if self._deep_tile_view is not None and self._deep_tile_view[0] == tile_key:
                                ov_view = self._deep_tile_view[1]
                            else:
                                self._request_deep_tiles_async(m_cont_slug, cx_cont, cy_cont, sc_cont, cw, ch)
                        # Level 1: Prestitched 4096x4096 WoW: Forever continent JPEG (4 ms/frame cv2 blit, zero CASC I/O)
                        if ov_view is None:
                            self._ensure_continent_l1_async(m_cont_slug)
                            l1_entry = self._cont_l1_cache.get(m_cont_slug)
                            if l1_entry is not None:
                                l1_arr, l1_native_px = l1_entry
                                l1_ratio = l1_arr.shape[0] / max(1.0, float(l1_native_px))
                                sc_l1 = sc_cont / max(1e-5, l1_ratio)
                                mip_arr, m_div = self._get_np_mip_for_scale(f"__l1_{m_cont_slug}", l1_arr, sc_l1)
                                ov_view = self._warp_view_cv2(
                                    mip_arr,
                                    (cx_cont * l1_ratio) / m_div,
                                    (cy_cont * l1_ratio) / m_div,
                                    sc_l1 * m_div,
                                    cw,
                                    ch,
                                    (18, 36, 58),
                                )
                # Level 0: Precached 4K Azeroth overview with fast OpenCV mip pyramid (~3-4 ms/frame!)
                if ov_view is None:
                    if self._azeroth_overview_arr is not None:
                        mip_arr, ov_div = self._get_np_mip_for_scale("__azeroth_ov", self._azeroth_overview_arr, sc_ov)
                        ov_view = self._warp_view_cv2(
                            mip_arr,
                            self._overview_pan_x / ov_div,
                            self._overview_pan_y / ov_div,
                            sc_ov * ov_div,
                            cw,
                            ch,
                            (18, 36, 58),
                        )
                    else:
                        inv_s = 1.0 / max(1e-5, sc_ov)
                        tx = self._overview_pan_x - (cw * 0.5) * inv_s
                        ty = self._overview_pan_y - (ch * 0.5) * inv_s
                        ov_view = self._azeroth_overview.image.transform(
                            (cw, ch),
                            Image.Transform.AFFINE,
                            (inv_s, 0.0, tx, 0.0, inv_s, ty),
                            resample=Image.Resampling.BILINEAR,
                            fillcolor=(18, 36, 58),
                        )
                self._tk_img = ImageTk.PhotoImage(ov_view)
                self.map_canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)
            else:
                self.map_canvas.create_text(
                    cw // 2, ch // 2,
                    text="Loading Azeroth Overview (Kalimdor & Eastern Kingdoms)...",
                    fill=TEXT_MUTED,
                    font=("Segoe UI", 11, "bold"),
                )
            if self._transition_state is not None and est is not None:
                self._set_dungeon_label(f"Position Fix: {est.dungeon_name} — Zooming in...")
            elif self.running:
                self._set_dungeon_label("Azeroth Overview — Searching for Position Fix...")
            else:
                self._set_dungeon_label("Azeroth Overview — Click 'Start Capture' or select a map")
            return

        if self.view_3d_mode:
            img_3d = self.viewer_3d.render(
                slug, d, cw, ch,
                player_world=est.world_pos if est else None,
                follow_player=bool(self.follow_var.get()),
                show_spawns=bool(self.spawns_var.get()),
                raid_players=list(shown_players.values()),
                local_player_id=None if self.replay_mode else self.identity.player_id,
            )
            self._tk_img = ImageTk.PhotoImage(Image.fromarray(img_3d))
            self.map_canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)
            if est is not None:
                self.lbl_dungeon.config(text=f"{est.dungeon_name}  [{est.dungeon_confidence * 100:.0f}% | {len(self.raid_players)} synced]")
            else:
                self.lbl_dungeon.config(text=f"{d.name}  [{len(self.raid_players)} synced]")
            return

        if est is not None and self.follow_var.get() and not self.replay_mode:
            self.pan_x, self.pan_y = float(est.map_px[0]), float(est.map_px[1])

        sc = self._eff_scale(cw, ch, map_h)
        base = self._ensure_fullres_display(slug, d)
        base_ratio = base.width / float(max(1, map_h))

        dx0 = int(round((0.0 - self.pan_x) * sc + cw * 0.5))
        dy0 = int(round((0.0 - self.pan_y) * sc + ch * 0.5))
        dx1 = int(round((float(map_h) - self.pan_x) * sc + cw * 0.5))
        dy1 = int(round((float(map_h) - self.pan_y) * sc + ch * 0.5))
        zone_covers_viewport = (dx0 <= 0 and dy0 <= 0 and dx1 >= cw and dy1 >= ch)

        entry = self.world_maps.get_entry(slug) if self.world_maps is not None else None
        folder = entry.minimap_folder if entry is not None else None
        if folder and entry is not None and entry.category != "Continents":
            self._ensure_continent_backdrop(folder)
        cont = self._continent_maps.get(folder) if (folder and entry is not None and entry.category != "Continents") else None
        if cont is not None and d.map_rot90 == 1 and cont.map_rot90 == 1 and not zone_covers_viewport:
            cont_pil = self._cached_pil_maps.get(f"__cont_{folder}")
            if cont_pil is None:
                cont_pil = Image.fromarray(cont.map_image)
                self._cached_pil_maps[f"__cont_{folder}"] = cont_pil
            # Lightweight continent complement (stays low-RAM; never decodes full-res CASC tiles for surrounding zones)
            wx_c, wy_c = d.map_px_to_world(self.pan_x, self.pan_y)
            ch_cont = float(cont.map_size_px)
            cx_cont = (ch_cont - 1.0) - (wy_c - float(cont.map_origin_xy[1])) * cont.map_scale
            cy_cont = (ch_cont - 1.0) - (wx_c - float(cont.map_origin_xy[0])) * cont.map_scale
            sc_cont = sc * (d.map_scale / max(1e-6, cont.map_scale))
            cont_ratio = cont_pil.width / max(1.0, ch_cont)
            cont_mip, c_div = self._get_mip_for_scale(f"__cont_{folder}", cont_pil, sc_cont / max(1e-5, cont_ratio))
            sc_c_mip = (sc_cont / max(1e-5, cont_ratio)) * c_div
            inv_s_cont = 1.0 / max(1e-5, sc_c_mip)
            tx_cont = (cx_cont * cont_ratio / c_div) - (cw * 0.5) * inv_s_cont
            ty_cont = (cy_cont * cont_ratio / c_div) - (ch * 0.5) * inv_s_cont
            view_img = cont_mip.transform(
                (cw, ch),
                Image.Transform.AFFINE,
                (inv_s_cont, 0.0, tx_cont, 0.0, inv_s_cont, ty_cont),
                resample=Image.Resampling.BILINEAR,
                fillcolor=(22, 46, 72),
            )
            # Composite crisp active zone tile on top of the seamless continent backdrop (viewport-capped LOD mip)
            zw, zh = dx1 - dx0, dy1 - dy0
            if zw >= 4 and zh >= 4 and dx1 > 0 and dy1 > 0 and dx0 < cw and dy0 < ch:
                sx0 = max(0, dx0)
                sy0 = max(0, dy0)
                sx1 = min(cw, dx1)
                sy1 = min(ch, dy1)
                zone_mip, _z_div = self._get_mip_for_scale(slug, base, sc / max(1e-5, base_ratio))
                u0 = max(0.0, (sx0 - dx0) / float(zw) * zone_mip.width)
                v0 = max(0.0, (sy0 - dy0) / float(zh) * zone_mip.height)
                u1 = min(float(zone_mip.width), (sx1 - dx0) / float(zw) * zone_mip.width)
                v1 = min(float(zone_mip.height), (sy1 - dy0) / float(zh) * zone_mip.height)
                if sx1 > sx0 and sy1 > sy0 and u1 > u0 and v1 > v0:
                    zone_patch = zone_mip.crop((u0, v0, u1, v1)).resize((sx1 - sx0, sy1 - sy0), Image.Resampling.BILINEAR)
                    view_img.paste(zone_patch, (sx0, sy0))
        else:
            view_img = None
            if (
                not self._is_interacting
                and sc > 1.05
                and self.world_maps is not None
                and self.world_maps.has_full_res_tiles(slug)
            ):
                view_img = self.world_maps.render_full_res_viewport(slug, self.pan_x, self.pan_y, sc, cw, ch)
            if view_img is None:
                mip_im, m_div = self._get_mip_for_scale(slug, base, sc / max(1e-5, base_ratio))
                inv_s_m = 1.0 / max(1e-5, (sc / max(1e-5, base_ratio)) * m_div)
                tx_m = (self.pan_x * base_ratio / m_div) - (cw * 0.5) * inv_s_m
                ty_m = (self.pan_y * base_ratio / m_div) - (ch * 0.5) * inv_s_m
                view_img = mip_im.transform((cw, ch), Image.Transform.AFFINE, (inv_s_m, 0.0, tx_m, 0.0, inv_s_m, ty_m), resample=Image.Resampling.BILINEAR, fillcolor=(13, 17, 23))
        self._tk_img = ImageTk.PhotoImage(view_img)
        self.map_canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)

        def m2c(mx: float, my: float) -> tuple[float, float]:
            return (mx - self.pan_x) * sc + cw * 0.5, (my - self.pan_y) * sc + ch * 0.5

        # Spawns
        if self.spawns_var.get():
            for sp in get_spawns_for_dungeon(slug):
                sx, sy = m2c(sp.u * map_h, sp.v * map_h)
                col = "#5adcfc" if sp.kind == "entrance" else "#ffa53c"
                self.map_canvas.create_oval(sx - 5, sy - 5, sx + 5, sy + 5, fill=col, outline="#14161c", width=1.5)
                self.map_canvas.create_text(sx + 8, sy, text=sp.label, anchor=tk.W, fill=col, font=("Segoe UI", 8, "bold"))

        # Collaborative Whiteboard ops
        for op in self.whiteboard.ops_for(slug):
            if op.kind == "stroke" and len(op.points) >= 2:
                flat = [c for u, v in op.points for c in m2c(u * map_h, v * map_h)]
                self.map_canvas.create_line(*flat, fill=op.color, width=op.width, smooth=True)
            elif op.kind == "text" and op.points:
                sx, sy = m2c(op.points[0][0] * map_h, op.points[0][1] * map_h)
                self.map_canvas.create_text(sx, sy, text=op.text, fill=op.color, font=("Segoe UI", 10, "bold"))
            elif op.kind == "circle_range" and op.points:
                sx, sy = m2c(op.points[0][0] * map_h, op.points[0][1] * map_h)
                r_px = yards_to_map_px(op.radius_yd, d.map_scale) * sc
                self.map_canvas.create_oval(sx - r_px, sy - r_px, sx + r_px, sy + r_px, outline=op.color, width=2, dash=(5, 3))
                self.map_canvas.create_text(sx, sy - r_px - 8, text=f"{op.radius_yd:.0f} yd", fill=op.color, font=("Segoe UI", 8, "bold"))

        if len(self._active_stroke_pts) >= 2:
            flat = [c for u, v in self._active_stroke_pts for c in m2c(u * map_h, v * map_h)]
            self.map_canvas.create_line(*flat, fill="#ffe066", width=3, smooth=True)

        # Active yard ruler
        if self.active_ruler is not None:
            x0, y0 = m2c(self.active_ruler.start_uv[0] * map_h, self.active_ruler.start_uv[1] * map_h)
            x1, y1 = m2c(self.active_ruler.end_uv[0] * map_h, self.active_ruler.end_uv[1] * map_h)
            self.map_canvas.create_line(x0, y0, x1, y1, fill="#63b3ed", width=2, dash=(4, 2))
            self.map_canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2 - 12, text=f"{self.active_ruler.yards:.1f} yd", fill="#90cdf4", font=("Segoe UI", 10, "bold"))

        # Connected Raid Players (or Replay Scrub players)
        for p in shown_players.values():
            if p.dungeon_slug != slug:
                continue
            sx, sy = m2c(p.map_px[0], p.map_px[1])
            alive = bool(getattr(p, "alive", True))
            col = WOW_CLASS_COLORS.get(p.wow_class, "#ff5a46") if alive else "#878c96"
            is_local = (p.player_id == self.identity.player_id) and not self.replay_mode
            outline = "#f5c96b" if is_local else "#ffffff"
            rad = 7 if is_local else 6
            self.map_canvas.create_oval(sx - rad, sy - rad, sx + rad, sy + rad, fill=col, outline=outline, width=2)
            if not alive:
                self.map_canvas.create_line(sx - 5, sy - 5, sx + 5, sy + 5, fill="#e53e3e", width=2)
                self.map_canvas.create_line(sx - 5, sy + 5, sx + 5, sy - 5, fill="#e53e3e", width=2)
            badge = _role_badge(p.role)
            label = f"[{badge}|DEAD] {p.name}" if not alive else f"[{badge}] {p.name}"
            txt_col = "#fc8181" if not alive else col
            self.map_canvas.create_text(sx + 10, sy + 1, text=label, anchor=tk.W, fill="#0a0c10", font=("Segoe UI", 8, "bold"))
            self.map_canvas.create_text(sx + 9, sy, text=label, anchor=tk.W, fill=txt_col, font=("Segoe UI", 8, "bold"))

        # Local Player dot (when not in replay mode and not already drawn in raid_players)
        if est is not None and not self.replay_mode and self.identity.player_id not in shown_players:
            sx, sy = m2c(est.map_px[0], est.map_px[1])
            self.map_canvas.create_oval(sx - 7, sy - 7, sx + 7, sy + 7, fill="#ff5a46", outline="#ffffff", width=2)

        if est is not None:
            self.lbl_dungeon.config(text=f"{est.dungeon_name}  [{est.dungeon_confidence * 100:.0f}% | {len(self.raid_players)} synced]")
        else:
            self.lbl_dungeon.config(text=f"{d.name}  [{len(self.raid_players)} synced]")

    def _on_close(self) -> None:
        self._closing = True
        self.running = False
        self._stop_capture_event.set()
        self._cancel_fix_transition()
        self._cancel_smooth_zoom()
        for attr in ("_poll_after_id", "_render_after_id", "_settle_after_id"):
            aid = getattr(self, attr, None)
            if aid is not None:
                try:
                    self.root.after_cancel(aid)
                except Exception:
                    pass
                setattr(self, attr, None)
        if self._azeroth_overview is not None:
            self._azeroth_overview.close()
            self._azeroth_overview = None
        self.stop_network()
        self.dev_logger.close()
        self.session_store.close()
        self.tracker_mm.close()
        if self.world_maps is not None and hasattr(self.world_maps, "close"):
            self.world_maps.close()
        self.viewer_3d.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="ForeverNavigator standalone GUI")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--tk", action="store_true", help="Force legacy Tkinter UI instead of GPU PySide6 UI")
    args = parser.parse_args()
    if not args.tk:
        try:
            from .qt_app import run_qt_app

            sys.exit(run_qt_app(index_dir=args.index_dir, fps=args.fps))
        except ImportError:
            pass
    ForeverNavigatorApp(index_dir=args.index_dir, fps=args.fps).run()


if __name__ == "__main__":
    main()
