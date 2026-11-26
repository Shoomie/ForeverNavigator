"""Developer-only personal gameplay dataset & diagnostics logger (git-ignored output)."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


class DevGameplayLogger:
    """
    Non-blocking background logger for developer gameplay troubleshooting & dataset collection.

    Enabled only when `FOREVERNAVIGATOR_DEV_LOG=1` or `<root_dir>/.enabled` exists locally.
    Output directory (`data/dev_gameplay/`) is git-ignored.
    """

    def __init__(
        self,
        root_dir: str | Path = "data/dev_gameplay",
        minimap_interval_sec: float = 2.0,
        fullscreen_interval_sec: float = 45.0,
        anomaly_cooldown_sec: float = 15.0,
        max_session_mb: float = 1500.0,
        force_enable: Optional[bool] = None,
    ):
        self.root_dir = Path(root_dir)
        if force_enable is not None:
            self.enabled = bool(force_enable)
        else:
            env_flag = os.environ.get("FOREVERNAVIGATOR_DEV_LOG", "").strip().lower() in ("1", "true", "yes")
            marker = (self.root_dir / ".enabled").exists()
            self.enabled = env_flag or marker

        self.minimap_interval_sec = float(minimap_interval_sec)
        self.fullscreen_interval_sec = float(fullscreen_interval_sec)
        self.anomaly_cooldown_sec = float(anomaly_cooldown_sec)
        self.max_bytes = int(max_session_mb * 1024 * 1024)

        self.session_dir: Optional[Path] = None
        self._events_fp = None
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread: Optional[threading.Thread] = None
        self._bytes_written = 0

        self._last_mm_t = 0.0
        self._last_fs_t = 0.0
        self._last_anomaly_t = 0.0
        self._prev_px: Optional[Tuple[int, int]] = None
        self._prev_slug: Optional[str] = None

        if self.enabled:
            self._start_session()

    def _start_session(self) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.root_dir / f"run_{stamp}"
        (self.session_dir / "minimap").mkdir(parents=True, exist_ok=True)
        (self.session_dir / "fullscreen").mkdir(parents=True, exist_ok=True)
        self._events_fp = open(self.session_dir / "events.jsonl", "a", encoding="utf-8", buffering=1)
        self._thread = threading.Thread(target=self._writer_loop, daemon=True, name="DevGameplayLogger")
        self._thread.start()
        self.log_event("session_start", {"pid": os.getpid()})

    def log_event(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled or self.session_dir is None:
            return
        rec = {
            "ts_wall": datetime.now().isoformat(timespec="milliseconds"),
            "ts_mono": round(time.monotonic(), 4),
            "event": event,
        }
        if data:
            rec.update(data)
        self._queue.put(("event", rec))

    def log_minimap_roi_sample(self, crop_rgb: np.ndarray, roi_dict: Dict[str, float]) -> None:
        """High-rate (e.g. 4 Hz) minimap-area sample."""
        if not self.enabled or self.session_dir is None or crop_rgb is None or crop_rgb.size == 0:
            return
        now_m = time.monotonic()
        if now_m - self._last_mm_t < self.minimap_interval_sec:
            return
        self._last_mm_t = now_m
        ts_ms = int(time.time() * 1000)
        rel = f"minimap/mm_{ts_ms}.png"
        self._queue.put(("image_png", (rel, crop_rgb.copy())))
        self.log_event("minimap_roi", {"file": rel, "shape": list(crop_rgb.shape), "roi": roi_dict})

    def log_tracker_tick(
        self,
        est: Any,
        locked_slug: Optional[str],
        capture_ms: float,
        mm_crop_rgb: Optional[np.ndarray],
        full_rgb: Optional[np.ndarray],
        roi_dict: Dict[str, float],
    ) -> None:
        if not self.enabled or self.session_dir is None or est is None:
            return
        now_m = time.monotonic()
        ts_ms = int(time.time() * 1000)

        top_scores = sorted(
            ((k, round(float(v), 4)) for k, v in getattr(est, "all_dungeon_scores", {}).items()),
            key=lambda kv: kv[1],
            reverse=True,
        )[:4]

        mm_rel = None
        if mm_crop_rgb is not None and mm_crop_rgb.size > 0 and (now_m - self._last_mm_t >= self.minimap_interval_sec):
            self._last_mm_t = now_m
            mm_rel = f"minimap/mm_{ts_ms}.png"
            self._queue.put(("image_png", (mm_rel, mm_crop_rgb.copy())))

        jump_px = 0.0
        cur_px = (int(est.map_px[0]), int(est.map_px[1]))
        if self._prev_px is not None and self._prev_slug == est.dungeon_slug:
            jump_px = float(np.hypot(cur_px[0] - self._prev_px[0], cur_px[1] - self._prev_px[1]))
        self._prev_px = cur_px
        self._prev_slug = est.dungeon_slug

        is_anomaly = (float(est.position_confidence) < 0.45) or (jump_px > 80.0)
        fs_rel = None
        if full_rgb is not None and full_rgb.size > 0:
            due_periodic = (now_m - self._last_fs_t) >= self.fullscreen_interval_sec
            due_anomaly = is_anomaly and ((now_m - self._last_anomaly_t) >= self.anomaly_cooldown_sec)
            if due_periodic or due_anomaly:
                self._last_fs_t = now_m
                if due_anomaly:
                    self._last_anomaly_t = now_m
                fs_rel = f"fullscreen/fs_{ts_ms}.jpg"
                self._queue.put(("image_jpg", (fs_rel, full_rgb.copy())))

        self.log_event(
            "tracker_tick",
            {
                "dungeon_slug": est.dungeon_slug,
                "locked_slug": locked_slug,
                "dungeon_conf": round(float(est.dungeon_confidence), 4),
                "pos_conf": round(float(est.position_confidence), 4),
                "map_px": list(cur_px),
                "world_pos": [round(float(x), 2) for x in est.world_pos],
                "inference_ms": round(float(est.inference_ms), 2),
                "capture_ms": round(float(capture_ms), 2),
                "jump_px": round(jump_px, 1),
                "anomaly": is_anomaly,
                "top_scores": top_scores,
                "roi": roi_dict,
                "minimap_file": mm_rel,
                "fullscreen_file": fs_rel,
            },
        )

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            kind, payload = item
            if self._bytes_written >= self.max_bytes:
                continue
            try:
                if kind == "event" and self._events_fp is not None:
                    line = json.dumps(payload, ensure_ascii=False) + "\n"
                    self._events_fp.write(line)
                    self._bytes_written += len(line.encode("utf-8"))
                elif kind == "image_png" and self.session_dir is not None:
                    rel, rgb = payload
                    out_p = self.session_dir / rel
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    if cv2.imwrite(str(out_p), bgr):
                        self._bytes_written += out_p.stat().st_size
                elif kind == "image_jpg" and self.session_dir is not None:
                    rel, rgb = payload
                    out_p = self.session_dir / rel
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    if cv2.imwrite(str(out_p), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82]):
                        self._bytes_written += out_p.stat().st_size
            except Exception:
                pass

    def close(self) -> None:
        if not self.enabled:
            return
        self.log_event("session_end", {"bytes_written": self._bytes_written})
        self._queue.put(None)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._events_fp is not None:
            try:
                self._events_fp.close()
            except Exception:
                pass
            self._events_fp = None
