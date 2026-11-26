"""Typed network protocol messages & granular opt-in sync preferences for ForeverNavigator."""

from __future__ import annotations

import socket
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.parse import urlparse

PROTOCOL_VERSION = "0.1"
DEFAULT_PORT = 8765
VALID_WHITEBOARD_PERMISSIONS = ("leader", "officers", "all")

VideoQuality = Literal["low", "medium", "high"]
VideoDeliveryMode = Literal["off", "realtime", "out_of_combat", "post_raid"]
CombatLogMode = Literal["off", "live_summary", "after_reload", "post_raid"]
RaidRole = Literal["LEADER", "OFFICER", "TANK", "HEALER", "DPS"]

WOW_CLASS_COLORS: Dict[str, str] = {
    "WARRIOR": "#c79c6e",
    "PALADIN": "#f58cba",
    "HUNTER": "#abd473",
    "ROGUE": "#fff569",
    "PRIEST": "#ffffff",
    "SHAMAN": "#0070de",
    "MAGE": "#69ccf0",
    "WARLOCK": "#9482c9",
    "DRUID": "#ff7d0a",
    "UNKNOWN": "#ff5a46",
}


def now_ms() -> int:
    return int(time.time() * 1000)


def get_local_lan_ip() -> str:
    """Best-effort detection of the host machine's LAN IPv4 address."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(0.5)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
    except OSError:
        pass
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip:
            return ip
    except OSError:
        pass
    return "127.0.0.1"


def normalize_ws_url(raw_url: str, default_port: int = DEFAULT_PORT) -> str:
    """Normalize a user-entered host/IP/URL into a valid ws://.../ws WebSocket URL."""
    raw = (raw_url or "").strip()
    if not raw:
        return f"ws://127.0.0.1:{default_port}/ws"
    if "://" not in raw:
        raw = f"ws://{raw}"
    parsed = urlparse(raw)
    scheme = parsed.scheme or "ws"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or default_port
    path = parsed.path or "/ws"
    if not path.startswith("/"):
        path = f"/{path}"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"{scheme}://{host}:{port}{path}"


@dataclass
class SyncPreferences:
    """Granular per-raider opt-in controls for every synced data type."""
    sync_position: bool = True
    poll_hz: float = 1.0
    sync_video: bool = False
    video_quality: VideoQuality = "low"          # low=360p@1fps, medium=540p@5fps, high=720p@15fps
    video_mode: VideoDeliveryMode = "out_of_combat"
    sync_combatlog: bool = True
    combatlog_mode: CombatLogMode = "after_reload"
    sync_ghost_routes: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SyncPreferences":
        valid = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**valid)


@dataclass
class PlayerIdentity:
    player_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Raider"
    wow_class: str = "WARRIOR"
    role: RaidRole = "DPS"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlayerIdentity":
        return cls(
            player_id=str(d.get("player_id") or uuid.uuid4()),
            name=str(d.get("name", "Raider")),
            wow_class=str(d.get("wow_class", "WARRIOR")),
            role=d.get("role", "DPS"),
        )


@dataclass
class PlayerState:
    player_id: str
    name: str
    wow_class: str
    role: str
    dungeon_slug: str
    x: float
    y: float
    z: float
    map_px: Tuple[int, int]
    confidence: float = 1.0
    alive: bool = True
    t_host_ms: int = field(default_factory=now_ms)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["map_px"] = list(self.map_px)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlayerState":
        raw_px = d.get("map_px")
        if isinstance(raw_px, (list, tuple)) and len(raw_px) >= 2:
            map_px = (int(raw_px[0]), int(raw_px[1]))
        else:
            map_px = (0, 0)
        return cls(
            player_id=str(d["player_id"]),
            name=str(d.get("name", "Raider")),
            wow_class=str(d.get("wow_class", "UNKNOWN")),
            role=str(d.get("role", "DPS")),
            dungeon_slug=str(d.get("dungeon_slug", "")),
            x=float(d.get("x", 0.0)),
            y=float(d.get("y", 0.0)),
            z=float(d.get("z", 0.0)),
            map_px=map_px,
            confidence=float(d.get("confidence", 1.0)),
            alive=bool(d.get("alive", True)),
            t_host_ms=int(d.get("t_host_ms", now_ms())),
        )
