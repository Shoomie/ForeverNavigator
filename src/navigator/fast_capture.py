"""Ultra-fast screen ROI capturer using Windows DXGI Desktop Duplication (dxcam) with MSS fallback.

On Windows 10/11, DXGI Desktop Duplication captures a minimap ROI in ~1-3 ms (vs ~15-22 ms for
GDI BitBlt / mss), enabling effortless 30-60 Hz tracking without impacting WoW frame rates.
On Linux or if DXGI is unavailable, falls back seamlessly to `mss`.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import mss
import numpy as np


class FastScreenCapturer:
    """Thread-local screen ROI & fullscreen capturer (DXGI on Windows, MSS fallback)."""

    def __init__(self, monitor_index: int = 1):
        self._mon_idx = max(1, int(monitor_index))
        self._sct = mss.MSS()
        self._dxcam = None
        self._dxcam_mon_idx: int = -1
        self._last_roi_rgb: Optional[np.ndarray] = None
        self._last_roi_key: Optional[Tuple[int, int, int, int, int]] = None
        self.backend: str = "mss"
        self._init_dxgi(self._mon_idx)

    def _init_dxgi(self, mon_idx: int) -> None:
        self._release_dxgi()
        self._mon_idx = max(1, int(mon_idx))
        self.backend = "mss"
        if os.name != "nt":
            return
        try:
            try:
                import comtypes

                comtypes.CoInitialize()
            except Exception:
                pass
            import dxcam

            out_idx = max(0, self._mon_idx - 1)
            cam = dxcam.create(output_idx=out_idx, output_color="RGB")
            if cam is not None:
                self._dxcam = cam
                self._dxcam_mon_idx = self._mon_idx
                self.backend = "dxgi"
        except Exception:
            self._dxcam = None
            self.backend = "mss"

    def _release_dxgi(self) -> None:
        if self._dxcam is not None:
            try:
                self._dxcam.release()
            except Exception:
                pass
            self._dxcam = None
            self._dxcam_mon_idx = -1

    def _get_mss_mon(self, mon_idx: int) -> dict:
        idx = min(max(1, int(mon_idx)), len(self._sct.monitors) - 1)
        return self._sct.monitors[idx]

    def grab_roi(self, mon_idx: int, x0_f: float, y0_f: float, x1_f: float, y1_f: float) -> np.ndarray:
        """Capture normalized [0..1] ROI on `mon_idx` and return an RGB uint8 numpy array."""
        mon = self._get_mss_mon(mon_idx)
        mw, mh = int(mon["width"]), int(mon["height"])
        lx0 = max(0, min(mw - 16, int(round(x0_f * mw))))
        ly0 = max(0, min(mh - 16, int(round(y0_f * mh))))
        lx1 = max(lx0 + 16, min(mw, int(round(x1_f * mw))))
        ly1 = max(ly0 + 16, min(mh, int(round(y1_f * mh))))

        roi_key = (mon_idx, lx0, ly0, lx1, ly1)
        if roi_key != self._last_roi_key:
            self._last_roi_rgb = None
            self._last_roi_key = roi_key

        if os.name == "nt" and mon_idx != self._dxcam_mon_idx:
            self._init_dxgi(mon_idx)

        if self._dxcam is not None:
            try:
                frame = self._dxcam.grab(region=(lx0, ly0, lx1, ly1))
                if frame is not None and frame.size > 0:
                    self._last_roi_rgb = frame
                    return frame
                # DXGI returns None when the GPU hasn't presented a new desktop frame yet -> reuse cached frame (<0.1 ms!)
                if self._last_roi_rgb is not None:
                    return self._last_roi_rgb
            except Exception:
                # Fall back to MSS cleanly if DXGI device lost (e.g. UAC / display mode switch)
                self._release_dxgi()
                self.backend = "mss"

        raw = self._sct.grab({
            "left": int(mon["left"]) + lx0,
            "top": int(mon["top"]) + ly0,
            "width": lx1 - lx0,
            "height": ly1 - ly0,
        })
        rgb = cv2.cvtColor(np.asarray(raw), cv2.COLOR_BGRA2RGB)
        self._last_roi_rgb = rgb
        return rgb

    def grab_fullscreen(self, mon_idx: int) -> np.ndarray:
        """Capture full monitor frame (RGB uint8) for periodic dev logging."""
        mon = self._get_mss_mon(mon_idx)
        if self._dxcam is not None and mon_idx == self._dxcam_mon_idx:
            try:
                frame = self._dxcam.grab()
                if frame is not None and frame.size > 0:
                    return frame
            except Exception:
                pass
        raw = self._sct.grab(mon)
        return cv2.cvtColor(np.asarray(raw), cv2.COLOR_BGRA2RGB)

    def close(self) -> None:
        self._release_dxgi()
        try:
            self._sct.close()
        except Exception:
            pass
