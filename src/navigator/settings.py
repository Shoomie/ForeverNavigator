"""Persistent user preferences for MapNavigate (saved to data/settings.json)."""

from __future__ import annotations

import ctypes
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

VALID_STORAGE_MODES = ("auto", "ssd", "hdd")


def detect_storage_type(path: str | Path = ".") -> str:
    """Best-effort, zero-subprocess detection of whether `path` resides on an SSD or HDD."""
    p = Path(path).resolve()
    if os.name == "nt":
        drive = os.path.splitdrive(str(p))[0]
        if drive:
            try:
                kernel32 = ctypes.windll.kernel32
                FILE_SHARE_READ = 0x1
                FILE_SHARE_WRITE = 0x2
                OPEN_EXISTING = 3
                IOCTL_STORAGE_QUERY_PROPERTY = 0x002D1400

                class _Query(ctypes.Structure):
                    _fields_ = [
                        ("PropertyId", ctypes.c_uint32),
                        ("QueryType", ctypes.c_uint32),
                        ("AdditionalParameters", ctypes.c_byte * 1),
                    ]

                class _SeekPenalty(ctypes.Structure):
                    _fields_ = [
                        ("Version", ctypes.c_uint32),
                        ("Size", ctypes.c_uint32),
                        ("IncursSeekPenalty", ctypes.c_ubyte),
                    ]

                h = kernel32.CreateFileW(
                    f"\\\\.\\{drive}",
                    0,
                    FILE_SHARE_READ | FILE_SHARE_WRITE,
                    None,
                    OPEN_EXISTING,
                    0,
                    None,
                )
                if h != ctypes.c_void_p(-1).value:
                    try:
                        q = _Query(7, 0)  # StorageDeviceSeekPenaltyProperty
                        out = _SeekPenalty()
                        br = ctypes.c_uint32(0)
                        ok = kernel32.DeviceIoControl(
                            h,
                            IOCTL_STORAGE_QUERY_PROPERTY,
                            ctypes.byref(q),
                            ctypes.sizeof(q),
                            ctypes.byref(out),
                            ctypes.sizeof(out),
                            ctypes.byref(br),
                            None,
                        )
                        if ok:
                            return "hdd" if bool(out.IncursSeekPenalty) else "ssd"
                    finally:
                        kernel32.CloseHandle(h)
            except Exception:
                pass
    elif os.name == "posix":
        try:
            st_dev = os.stat(p).st_dev
            major, minor = os.major(st_dev), os.minor(st_dev)
            sys_path = Path(f"/sys/dev/block/{major}:{minor}")
            if sys_path.exists():
                real = sys_path.resolve()
                rot = real / "queue" / "rotational"
                if not rot.exists():
                    rot = real.parent / "queue" / "rotational"
                if rot.exists():
                    return "hdd" if rot.read_text().strip() == "1" else "ssd"
        except Exception:
            pass
    return "ssd"


def resolve_storage_mode(storage_mode: str, path: str | Path = ".") -> str:
    """Resolve 'auto' | 'ssd' | 'hdd' into concrete 'ssd' or 'hdd'."""
    mode = (storage_mode or "auto").strip().lower()
    if mode in ("ssd", "hdd"):
        return mode
    return detect_storage_type(path)


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
    high_precision: bool = False
    use_neural_model: bool = True
    use_opencv_refine: bool = True
    opencv_verify_only: bool = False  # True = run OpenCV only at verifier_hz; False = every frame
    use_verifier_neural: bool = False  # True = run secondary Accurate AI verifier at verifier_hz
    pipeline_preset: str = "hybrid_60hz_cv"
    neural_max_hz: float = 0.0        # 0.0 = every frame (match poll), >0 = cap primary AI Hz
    verifier_hz: float = 2.0          # 0.0 = off, else periodic verifier rate (strictly < AI Hz)
    neural_model_name: str = ""       # "" = auto-select best available ONNX stem in data/models/
    periodic_opencv_anchor: bool = True
    storage_mode: str = "auto"       # "auto", "ssd", "hdd"
    wow_dir: str = ""
    window_geometry: str = "1200x840"

    def clamped(self) -> "AppSettings":
        self.fps = float(max(0.25, min(60.0, self.fps)))
        self.monitor = max(1, int(self.monitor))
        self.window_opacity = float(max(0.50, min(1.0, self.window_opacity)))
        self.brightness_3d = float(max(0.60, min(1.60, self.brightness_3d)))
        if str(self.storage_mode).lower() not in VALID_STORAGE_MODES:
            self.storage_mode = "auto"
        else:
            self.storage_mode = str(self.storage_mode).lower()
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
