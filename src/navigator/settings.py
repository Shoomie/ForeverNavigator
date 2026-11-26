"""Persistent user preferences for MapNavigate (saved to data/settings.json)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class AppSettings:
    fps: float = 1.0
    monitor: int = 1
    mode: str = "Minimap (2D Ultra-Light)"
    follow_player: bool = True
    show_spawns: bool = True
    show_trail: bool = True
    show_pip: bool = True
    always_on_top: bool = False
    window_opacity: float = 1.0      # 0.50 .. 1.00
    cutaway_ceilings: bool = True
    brightness_3d: float = 1.0       # 0.60 .. 1.60
    rotate_minimap: bool = False
    adaptive_ui_mask: bool = True
    wow_dir: str = ""
    window_geometry: str = "780x840"

    def clamped(self) -> "AppSettings":
        self.fps = float(max(0.25, min(5.0, self.fps)))
        self.monitor = max(1, int(self.monitor))
        self.window_opacity = float(max(0.50, min(1.0, self.window_opacity)))
        self.brightness_3d = float(max(0.60, min(1.60, self.brightness_3d)))
        return self


def load_settings(path: str | Path) -> AppSettings:
    p = Path(path)
    if not p.exists():
        return AppSettings()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        valid = {k: raw[k] for k in AppSettings.__dataclass_fields__ if k in raw}
        return AppSettings(**valid).clamped()
    except Exception:
        return AppSettings()


def save_settings(path: str | Path, settings: AppSettings) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(settings.clamped()), indent=2), encoding="utf-8")
