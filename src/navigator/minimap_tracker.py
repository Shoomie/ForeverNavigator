"""Ultra-lightweight 2D in-game minimap matcher (zero neural network, pure OpenCV CPU).

Matches a circular crop of the in-game WoW minimap against pre-rendered 2D dungeon floorplans
using multi-scale structural (floor-vs-void silhouette + wall-edge) normalized cross-correlation
with temporal smoothing and multi-dungeon auto-detection.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .gpu_matcher import GPUComputeMatcher
from .tracker import DungeonData, TrackerEstimate


@dataclass
class MinimapROI:
    """Normalized [0..1] screen rectangle for the in-game minimap (default Classic 16:9)."""
    x0: float = 0.885
    y0: float = 0.040
    x1: float = 0.992
    y1: float = 0.228

    def crop(self, frame_rgb: np.ndarray) -> np.ndarray:
        h, w = frame_rgb.shape[:2]
        ix0 = max(0, min(w - 2, int(round(self.x0 * w))))
        iy0 = max(0, min(h - 2, int(round(self.y0 * h))))
        ix1 = max(ix0 + 2, min(w, int(round(self.x1 * w))))
        iy1 = max(iy0 + 2, min(h, int(round(self.y1 * h))))
        return frame_rgb[iy0:iy1, ix0:ix1]

    def to_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}

    @classmethod
    def from_dict(cls, d: dict) -> "MinimapROI":
        return cls(
            x0=float(d.get("x0", 0.885)),
            y0=float(d.get("y0", 0.040)),
            x1=float(d.get("x1", 0.992)),
            y1=float(d.get("y1", 0.228)),
        )


def clean_minimap_overlays(sq_rgb: np.ndarray) -> np.ndarray:
    """Detect and softly inpaint connected bright blue/white quest-area polygon overlays (Questie/RestedXP/WoW)."""
    r = sq_rgb[:, :, 0].astype(np.int16)
    g = sq_rgb[:, :, 1].astype(np.int16)
    b = sq_rgb[:, :, 2].astype(np.int16)
    ov = ((b - r > 22) & (b > 145) & (g > 115)) | ((b > 200) & (g > 190) & (r > 170))
    # Morphological opening rejects isolated single-pixel noise while keeping connected quest-border strokes
    opened = cv2.morphologyEx(ov.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    frac = float(opened.mean())
    if 0.005 < frac < 0.32:
        mask = cv2.dilate(opened, np.ones((5, 5), np.uint8)) * 255
        if float((mask > 0).mean()) < 0.45:
            return cv2.inpaint(sq_rgb, mask, 4, cv2.INPAINT_TELEA)
    return sq_rgb


def extract_structural_channels(img_rgb: np.ndarray, is_floorplan: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 2D map or live WoW minimap crop into two fast, illumination-robust representations:
      1) Normalized 3-channel RGB (TM_CCOEFF_NORMED is natively invariant to brightness/contrast
         while preserving distinctive biome colors: Tirisfal green, lava orange, Naxx green, stone gray)
      2) Edge gradient magnitude (Sobel) highlighting walls, roads, and corridor outlines
    """
    rgb_f = cv2.GaussianBlur(img_rgb.astype(np.float32) / 255.0, (3, 3), 0.8)
    gray = cv2.cvtColor(rgb_f, cv2.COLOR_RGB2GRAY)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.sqrt(gx * gx + gy * gy)
    e_max = float(edges.max())
    if e_max > 1e-4:
        edges = edges / e_max
    return rgb_f.astype(np.float32), edges.astype(np.float32)


def crop_inscribed_minimap(crop_sq: np.ndarray) -> np.ndarray:
    """
    Extract the largest square completely inside the circular minimap (avoiding border/buttons)
    and softly fill the tiny center player arrow so it doesn't bias template matching.
    """
    side = crop_sq.shape[0]
    margin = int(round(side * 0.19))  # leaves central 62% x 62% square inside the circle
    inner = crop_sq[margin : side - margin, margin : side - margin].copy()
    if inner.size == 0:
        return crop_sq
    h, w = inner.shape[:2]
    # Replace the center player arrow (~12% radius) with local neighborhood median
    r = max(2, int(round(min(h, w) * 0.08)))
    cy, cx = h // 2, w // 2
    ring = inner[max(0, cy - 2 * r) : min(h, cy + 2 * r), max(0, cx - 2 * r) : min(w, cx + 2 * r)]
    if ring.size > 0:
        fill = np.median(ring.reshape(-1, 3), axis=0).astype(np.uint8)
        cv2.circle(inner, (cx, cy), r, tuple(int(c) for c in fill), -1)
    return inner


class MinimapTracker:
    """Ultra-lightweight 2D minimap localizer (no neural network; ~2-5 ms on CPU)."""

    def __init__(
        self,
        dungeons: Dict[str, DungeonData],
        dungeon_order: List[str],
        roi: Optional[MinimapROI] = None,
        rotate_minimap: bool = False,
        map_work_size: int = 160,
        high_precision: bool = False,
        use_gpu: bool = True,
    ):
        self.dungeons = dungeons
        self.dungeon_order = dungeon_order
        self.roi = roi or MinimapROI()
        self.rotate_minimap = rotate_minimap
        self.map_work_size = map_work_size
        self.high_precision = bool(high_precision)
        self.use_gpu = bool(use_gpu)
        self._lock = threading.RLock()
        self._executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=min(12, max(2, os.cpu_count() or 4)), thread_name_prefix="MMTrack")
            if self.high_precision
            else None
        )
        self._gpu_matcher: Optional[GPUComputeMatcher] = (
            GPUComputeMatcher() if (self.high_precision and self.use_gpu) else None
        )

        # Precompute structural channels + valid walkable center mask at both fine (128px) & coarse (72px) scales
        self.coarse_size = 72
        self._map_sil: Dict[str, np.ndarray] = {}
        self._map_edge: Dict[str, np.ndarray] = {}
        self._valid_center: Dict[str, np.ndarray] = {}
        self._coarse_sil: Dict[str, np.ndarray] = {}
        self._coarse_valid: Dict[str, np.ndarray] = {}
        self._scale_ratio: Dict[str, float] = {}
        self._coarse_ratio: Dict[str, float] = {}

        for slug, d in self.dungeons.items():
            self._precompute_dungeon(slug, d)

        n_d = len(self.dungeon_order)
        self.dungeon_probs = np.full(n_d, 1.0 / max(1, n_d), dtype=np.float64)
        self.active_slug: str = self.dungeon_order[0]
        self.locked_dungeon_slug: Optional[str] = None

        # Smoothed position in original map pixels + motion-plausibility / teleport confirmation state
        self._smooth_px: Optional[np.ndarray] = None
        self._last_update_t: Optional[float] = None
        self._jump_cand_px: Optional[np.ndarray] = None
        self._jump_confirm_count: int = 0
        self.trail_px: List[Tuple[int, int]] = []
        self.last_crop_rgb: Optional[np.ndarray] = None
        self._frame_count: int = 0
        self._shared_map_slugs: set[str] = {
            s for s in self.dungeon_order if s.startswith("wmo_")
        }
        self._last_outdoor_slug: Optional[str] = None
        self._last_outdoor_world_pos: Optional[np.ndarray] = None

    def mark_shared_map(self, slug: str, is_shared: bool = True) -> None:
        with self._lock:
            if is_shared:
                self._shared_map_slugs.add(slug)
            else:
                self._shared_map_slugs.discard(slug)

    def is_shared_map(self, slug: str) -> bool:
        return slug in self._shared_map_slugs or slug.startswith("wmo_")

    def set_high_precision(self, enabled: bool) -> None:
        """Toggle high-precision mode (higher working res, multi-threaded CPU, GPU ZNCC, full-res local refinement)."""
        enabled = bool(enabled)
        with self._lock:
            if enabled == self.high_precision:
                return
            self.high_precision = enabled
            if enabled:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(
                        max_workers=min(12, max(2, os.cpu_count() or 4)),
                        thread_name_prefix="MMTrack",
                    )
                if self.use_gpu and self._gpu_matcher is None:
                    self._gpu_matcher = GPUComputeMatcher()
            else:
                if self._executor is not None:
                    self._executor.shutdown(wait=False)
                    self._executor = None
                if self._gpu_matcher is not None:
                    self._gpu_matcher.close()
                    self._gpu_matcher = None
            for slug, d in list(self.dungeons.items()):
                self._precompute_dungeon(slug, d)

    def close(self) -> None:
        """Release background worker pool and optional GPU compute matcher."""
        with self._lock:
            if self._executor is not None:
                self._executor.shutdown(wait=False)
                self._executor = None
            if self._gpu_matcher is not None:
                self._gpu_matcher.close()
                self._gpu_matcher = None

    def _precompute_dungeon(self, slug: str, d: DungeonData) -> None:
        """Precompute fine and coarse structural channels for a dungeon/world map."""
        orig_h = d.map_size_px
        img_h = int(d.map_image.shape[0])
        span_yards = float(orig_h) / max(1e-4, float(d.map_scale))
        # Choose working resolution based on world span in yards (independent of native map pixel resolution)
        if self.high_precision:
            if span_yards > 2500.0:
                w_sz, c_sz = min(img_h, 2048), min(img_h, 768)
            elif span_yards > 900.0:
                w_sz, c_sz = min(img_h, 1024), min(img_h, 384)
            else:
                w_sz, c_sz = min(img_h, 512), min(img_h, 160)
        else:
            if span_yards > 2500.0:
                w_sz, c_sz = min(img_h, 512), min(img_h, 256)
            elif span_yards > 900.0:
                w_sz, c_sz = min(img_h, 384), min(img_h, 192)
            else:
                w_sz, c_sz = min(img_h, max(self.map_work_size, 256)), min(img_h, self.coarse_size)

        small = cv2.resize(d.map_image, (w_sz, w_sz), interpolation=cv2.INTER_AREA)
        rgb_f, edge = extract_structural_channels(small, is_floorplan=True)
        lum = rgb_f.mean(axis=-1)
        valid = cv2.dilate((lum > 0.11).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        ratio_f = orig_h / float(w_sz)

        c_small = cv2.resize(d.map_image, (c_sz, c_sz), interpolation=cv2.INTER_AREA)
        c_rgb = cv2.GaussianBlur(c_small.astype(np.float32) / 255.0, (3, 3), 0.8).astype(np.float32)
        c_lum = c_rgb.mean(axis=-1)
        c_valid = cv2.dilate((c_lum > 0.11).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        ratio_c = orig_h / float(c_sz)

        # Restrict valid match centers to the zone's actual world bounds_xy (+80yd border)
        # so padded square zone maps don't steal matches from neighboring zones!
        if d.bounds_xy is not None:
            xmin, ymin, xmax, ymax = d.bounds_xy
            pad = 80.0
            p0 = d.world_to_map_px(np.array([xmin - pad, ymin - pad]))
            p1 = d.world_to_map_px(np.array([xmax + pad, ymax + pad]))
            bx0, bx1 = max(0, min(p0[0], p1[0])), min(orig_h, max(p0[0], p1[0]) + 1)
            by0, by1 = max(0, min(p0[1], p1[1])), min(orig_h, max(p0[1], p1[1]) + 1)

            f_mask = np.zeros_like(valid, dtype=bool)
            f_mask[int(by0 / ratio_f) : int(np.ceil(by1 / ratio_f)), int(bx0 / ratio_f) : int(np.ceil(bx1 / ratio_f))] = True
            valid = valid & f_mask

            c_mask = np.zeros_like(c_valid, dtype=bool)
            c_mask[int(by0 / ratio_c) : int(np.ceil(by1 / ratio_c)), int(bx0 / ratio_c) : int(np.ceil(bx1 / ratio_c))] = True
            c_valid = c_valid & c_mask

        with self._lock:
            self._map_sil[slug] = rgb_f
            self._map_edge[slug] = edge
            if self.high_precision and self._gpu_matcher is not None and self._gpu_matcher.available and w_sz >= 1536:
                self._gpu_matcher.upload_map(slug, rgb_f, edge)
            self._scale_ratio[slug] = ratio_f
            self._coarse_sil[slug] = c_rgb
            self._coarse_ratio[slug] = ratio_c
            self._valid_center[slug] = valid
            self._coarse_valid[slug] = c_valid

    def register_dungeon(self, dungeon: DungeonData, is_shared: bool = False) -> None:
        """Dynamically register an on-demand loaded map/zone/dungeon into the minimap tracker."""
        slug = dungeon.slug
        self._precompute_dungeon(slug, dungeon)
        with self._lock:
            if is_shared or slug.startswith("wmo_"):
                self._shared_map_slugs.add(slug)
            self.dungeons[slug] = dungeon
            if slug not in self.dungeon_order:
                self.dungeon_order.append(slug)
                new_probs = np.zeros(len(self.dungeon_order), dtype=np.float64)
                old_n = len(self.dungeon_probs)
                if old_n > 0:
                    new_probs[:old_n] = self.dungeon_probs * (old_n / float(len(self.dungeon_order)))
                new_probs[-1] = 1.0 / float(len(self.dungeon_order))
                s = float(new_probs.sum())
                self.dungeon_probs = new_probs / s if s > 0 else np.full(len(self.dungeon_order), 1.0 / len(self.dungeon_order))

    def reset(self) -> None:
        with self._lock:
            n_d = len(self.dungeon_order)
            self.dungeon_probs = np.full(n_d, 1.0 / max(1, n_d), dtype=np.float64)
            self._smooth_px = None
            self._last_update_t = None
            self._jump_cand_px = None
            self._jump_confirm_count = 0
            self.trail_px.clear()

    def _match_single_dungeon(
        self,
        slug: str,
        crop_sil: np.ndarray,
        crop_edge: np.ndarray,
        scales_px: Tuple[int, ...],
        angles_deg: Tuple[float, ...],
        prev_work_xy: Optional[Tuple[float, float]] = None,
        plausible_radius_work: Optional[float] = None,
        tpl_cache: Optional[Dict[Tuple[int, float], Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Return (best_score, (work_cx, work_cy)) for one dungeon.
        When prev_work_xy & plausible_radius_work are provided, prefers a strong local match within
        physically plausible travel distance over a marginally higher distant match.
        """
        m_sil = self._map_sil[slug]
        m_edge = self._map_edge[slug]
        mh, mw = m_sil.shape[:2]

        best_global_score = -1.0
        best_global_xy = (mw / 2.0, mh / 2.0)
        best_local_score = -1.0
        best_local_xy = best_global_xy

        for sz in scales_px:
            if sz >= mh - 4 or sz >= mw - 4 or sz < 11:
                continue
            t_sil_base = None
            t_edge_base = None

            for ang in angles_deg:
                cache_key = (sz, ang)
                cached = tpl_cache.get(cache_key) if tpl_cache is not None else None
                if cached is not None:
                    t_sil, t_edge = cached
                else:
                    if t_sil_base is None:
                        t_sil_base = cv2.resize(crop_sil, (sz, sz), interpolation=cv2.INTER_AREA)
                        t_edge_base = cv2.resize(crop_edge, (sz, sz), interpolation=cv2.INTER_AREA)
                    if abs(ang) > 1e-3:
                        rot_m = cv2.getRotationMatrix2D((sz / 2.0, sz / 2.0), ang, 1.0)
                        t_sil = cv2.warpAffine(t_sil_base, rot_m, (sz, sz), borderMode=cv2.BORDER_REPLICATE)
                        t_edge = cv2.warpAffine(t_edge_base, rot_m, (sz, sz), borderMode=cv2.BORDER_REPLICATE)
                    else:
                        t_sil = t_sil_base
                        t_edge = t_edge_base
                    if tpl_cache is not None:
                        tpl_cache[cache_key] = (t_sil, t_edge)

                gpu_combined = None
                if (
                    self.high_precision
                    and self._gpu_matcher is not None
                    and self._gpu_matcher.available
                    and mw >= 1536
                ):
                    gpu_combined = self._gpu_matcher.match(slug, t_sil, t_edge)

                if gpu_combined is not None:
                    combined = gpu_combined
                else:
                    r_sil = cv2.matchTemplate(m_sil, t_sil, cv2.TM_CCOEFF_NORMED)
                    r_edge = cv2.matchTemplate(m_edge, t_edge, cv2.TM_CCOEFF_NORMED)
                    np.nan_to_num(r_sil, copy=False, nan=-1.0, posinf=-1.0, neginf=-1.0)
                    np.nan_to_num(r_edge, copy=False, nan=-1.0, posinf=-1.0, neginf=-1.0)
                    # Reject constant/ocean patches where TM_CCOEFF_NORMED edge spuriously returns 1.0
                    r_edge = np.where(r_sil > 0.05, r_edge, -1.0)
                    combined = 0.70 * r_sil + 0.30 * r_edge

                half = sz // 2
                rh, rw = combined.shape
                vc = self._valid_center[slug][half : half + rh, half : half + rw]
                combined = np.where(vc, combined, -1.0)

                _min_v, max_v, _min_l, max_l = cv2.minMaxLoc(combined)
                if max_v > best_global_score:
                    best_global_score = float(max_v)
                    best_global_xy = (max_l[0] + sz / 2.0, max_l[1] + sz / 2.0)

                # Also check best match within physically plausible movement radius
                if prev_work_xy is not None and plausible_radius_work is not None:
                    px_c = int(round(prev_work_xy[0] - sz / 2.0))
                    py_c = int(round(prev_work_xy[1] - sz / 2.0))
                    rad_i = max(4, int(round(plausible_radius_work)))
                    x0, x1 = max(0, px_c - rad_i), min(rw, px_c + rad_i + 1)
                    y0, y1 = max(0, py_c - rad_i), min(rh, py_c + rad_i + 1)
                    if x1 > x0 and y1 > y0:
                        sub = combined[y0:y1, x0:x1]
                        _lv, lmax, _ll, lloc = cv2.minMaxLoc(sub)
                        if lmax > best_local_score:
                            best_local_score = float(lmax)
                            best_local_xy = (x0 + lloc[0] + sz / 2.0, y0 + lloc[1] + sz / 2.0)

        # Prefer physically plausible local match unless global match is decisively stronger
        if (
            prev_work_xy is not None
            and best_local_score >= 0.45
            and best_local_score >= best_global_score - 0.16
        ):
            return best_local_score, best_local_xy

        return best_global_score, best_global_xy

    def _coarse_score_dungeon(
        self,
        slug: str,
        crop_sil: np.ndarray,
        zoom_radii_yd: Tuple[float, ...],
        angles_deg: Tuple[float, ...],
    ) -> float:
        """Ultra-fast coarse screening score at 72x72 (~0.25 ms per dungeon)."""
        d = self.dungeons[slug]
        m_sil = self._coarse_sil[slug]
        mh, mw = m_sil.shape[:2]
        ratio = self._coarse_ratio[slug]
        raw_scales = [
            min(mh - 6, int(round(2.0 * r_yd * 0.62 * d.map_scale / ratio)))
            for r_yd in zoom_radii_yd
        ]
        valid = sorted(set(s for s in raw_scales if s >= 10))
        scales = valid if valid else [max(10, max(raw_scales))]
        best = -1.0
        for sz in scales:
            t_sil_base = cv2.resize(crop_sil, (sz, sz), interpolation=cv2.INTER_AREA)
            for ang in angles_deg:
                if abs(ang) > 1e-3:
                    rot_m = cv2.getRotationMatrix2D((sz / 2.0, sz / 2.0), ang, 1.0)
                    t_sil = cv2.warpAffine(t_sil_base, rot_m, (sz, sz), borderMode=cv2.BORDER_REPLICATE)
                else:
                    t_sil = t_sil_base
                r_sil = cv2.matchTemplate(m_sil, t_sil, cv2.TM_CCOEFF_NORMED)
                np.nan_to_num(r_sil, copy=False, nan=-1.0, posinf=-1.0, neginf=-1.0)
                half = sz // 2
                rh, rw = r_sil.shape
                vc = self._coarse_valid[slug][half : half + rh, half : half + rw]
                r_sil = np.where(vc, r_sil, -1.0)
                v = float(r_sil.max())
                if v > best:
                    best = v
        return best

    def update(self, frame_rgb: np.ndarray, is_already_cropped: bool = False) -> TrackerEstimate:
        """Process a full screenshot (or pre-cropped minimap) and return dungeon + position."""
        t0 = time.perf_counter()
        crop = frame_rgb if is_already_cropped else self.roi.crop(frame_rgb)
        ch, cw = crop.shape[:2]
        side = min(ch, cw)
        y0 = (ch - side) // 2
        x0 = (cw - side) // 2
        crop_sq = crop[y0 : y0 + side, x0 : x0 + side]
        self.last_crop_rgb = crop_sq.copy()

        inner_sq = crop_inscribed_minimap(crop_sq)
        # Guard against loading screens (pure black/blank frames); allow sparse indoor WMO floorplans (>2% lit)
        lit_frac = float((inner_sq.mean(axis=-1) > 18.0).mean())
        min_lit = 0.02 if (self.locked_dungeon_slug and self.is_shared_map(self.locked_dungeon_slug)) else 0.05
        if lit_frac < min_lit:
            slug = self.locked_dungeon_slug or self.active_slug
            d = self.dungeons[slug]
            px = (
                (int(round(self._smooth_px[0])), int(round(self._smooth_px[1])))
                if self._smooth_px is not None
                else (d.map_size_px // 2, d.map_size_px // 2)
            )
            wx, wy = d.map_px_to_world(px[0], px[1])
            return TrackerEstimate(
                dungeon_slug=slug,
                dungeon_name=d.name,
                dungeon_confidence=1.0 if self.locked_dungeon_slug else 0.0,
                world_pos=np.array([wx, wy, 0.0], dtype=np.float32),
                map_px=px,
                position_confidence=0.0,
                particles_px=[px],
                inference_ms=(time.perf_counter() - t0) * 1000.0,
                all_dungeon_scores={s: 0.0 for s in self.dungeon_order},
                is_shared_minimap=self.is_shared_map(slug),
                outdoor_context_slug=self._last_outdoor_slug,
            )

        base_sq = cv2.resize(inner_sq, (96, 96), interpolation=cv2.INTER_AREA)
        cleaned_sq = clean_minimap_overlays(base_sq)
        dark_void_frac = float((cleaned_sq.max(axis=-1) < 14).mean())
        crop_sil, crop_edge = extract_structural_channels(cleaned_sq, is_floorplan=False)

        if self.high_precision:
            angles_deg = (0.0,) if not self.rotate_minimap else tuple(float(a) for a in range(0, 360, 15))
            zoom_radii_yd = (32.0, 45.0, 62.0, 85.0, 115.0, 155.0, 210.0, 265.0)
            coarse_radii_yd = (60.0, 105.0, 175.0, 260.0)
            top_k_cands = 10
        else:
            angles_deg = (0.0,) if not self.rotate_minimap else tuple(float(a) for a in range(0, 360, 30))
            zoom_radii_yd = (40.0, 62.0, 95.0, 140.0, 200.0, 255.0)
            coarse_radii_yd = (75.0, 140.0, 255.0)
            top_k_cands = 8

        now_t = time.perf_counter()
        dt_sec = 1.0 if self._last_update_t is None else float(np.clip(now_t - self._last_update_t, 0.2, 6.0))
        self._last_update_t = now_t
        # Max physically plausible travel in dt_sec (covers 14 yd/s mount and ~35 yd/s flight paths)
        plausible_yd = max(28.0, 42.0 * dt_sec + 15.0)
        tpl_cache: Optional[Dict[Tuple[int, float], Tuple[np.ndarray, np.ndarray]]] = (
            None if (self._executor is not None and self.high_precision) else {}
        )

        def eval_fine(slug: str) -> Tuple[float, Tuple[float, float]]:
            d = self.dungeons[slug]
            ratio = self._scale_ratio[slug]
            w_sz = self._map_sil[slug].shape[0]
            raw_scales = [
                min(w_sz - 8, int(round(2.0 * r_yd * 0.62 * d.map_scale / ratio)))
                for r_yd in zoom_radii_yd
            ]
            valid_scales = sorted(set(s for s in raw_scales if s >= 10))
            if not valid_scales:
                valid_scales = [max(10, max(raw_scales))]
            scales_px = tuple(valid_scales)
            prev_wxy = None
            plaus_w = None
            if slug == self.active_slug and self._smooth_px is not None:
                prev_wxy = (float(self._smooth_px[0]) / ratio, float(self._smooth_px[1]) / ratio)
                plaus_w = (plausible_yd * d.map_scale) / ratio
            return self._match_single_dungeon(
                slug, crop_sil, crop_edge, scales_px, angles_deg,
                prev_work_xy=prev_wxy, plausible_radius_work=plaus_w,
                tpl_cache=tpl_cache,
            )

        with self._lock:
            order = list(self.dungeon_order)
            probs = self.dungeon_probs.copy()
            locked_slug = self.locked_dungeon_slug
            active_slug = self.active_slug

        raw_scores: Dict[str, float] = {s: 0.0 for s in order}
        best_xy_by_slug: Dict[str, Tuple[float, float]] = {}

        if locked_slug and locked_slug in self.dungeons:
            sc, xy = eval_fine(locked_slug)
            raw_scores[locked_slug] = sc
            best_xy_by_slug[locked_slug] = xy
        else:
            # Fast-path: if already confident in active_slug, check it first—but NEVER get stuck!
            # Always force a full map check every 5 frames (at 1 Hz = every 5s) or if score < 0.68.
            self._frame_count += 1
            active_i = order.index(active_slug) if active_slug in order else 0
            active_prob = float(probs[active_i]) if active_i < len(probs) else 0.0
            force_global = (self._frame_count % 5 == 0)
            if not force_global and active_prob >= 0.78 and len(order) > 6:
                sc_a, xy_a = eval_fine(active_slug)
                raw_scores[active_slug] = sc_a
                best_xy_by_slug[active_slug] = xy_a
                if sc_a >= 0.68:
                    # Unmistakable lock in active dungeon (~11 ms total!)
                    for s in order:
                        if s != active_slug:
                            raw_scores[s] = max(0.0, sc_a - 0.18)
                else:
                    # Match isn't rock-solid (or zoned/summoned): run 2-stage coarse-to-fine across ALL maps!
                    if self._executor is not None and self.high_precision:
                        c_vals = list(self._executor.map(
                            lambda s: self._coarse_score_dungeon(s, crop_sil, coarse_radii_yd, angles_deg),
                            order,
                        ))
                        coarse = dict(zip(order, c_vals))
                    else:
                        coarse = {
                            s: self._coarse_score_dungeon(s, crop_sil, coarse_radii_yd, angles_deg)
                            for s in order
                        }
                    top_cands = sorted(coarse, key=coarse.get, reverse=True)[: min(top_k_cands, 6)]
                    if active_slug not in top_cands:
                        top_cands.append(active_slug)
                    for s in order:
                        raw_scores[s] = coarse[s] * 0.85
                    pending = [s for s in top_cands if s not in best_xy_by_slug]
                    if self._executor is not None and self.high_precision and len(pending) > 1:
                        for s, (sc, xy) in zip(pending, self._executor.map(eval_fine, pending)):
                            raw_scores[s] = sc
                            best_xy_by_slug[s] = xy
                    else:
                        for s in pending:
                            sc, xy = eval_fine(s)
                            raw_scores[s] = sc
                            best_xy_by_slug[s] = xy
            elif len(order) > 6:
                # 2-stage coarse-to-fine global auto-detection across all indexed/cached maps
                if self._executor is not None and self.high_precision:
                    c_vals = list(self._executor.map(
                        lambda s: self._coarse_score_dungeon(s, crop_sil, coarse_radii_yd, angles_deg),
                        order,
                    ))
                    coarse = dict(zip(order, c_vals))
                else:
                    coarse = {
                        s: self._coarse_score_dungeon(s, crop_sil, coarse_radii_yd, angles_deg)
                        for s in order
                    }
                top_cands = sorted(coarse, key=coarse.get, reverse=True)[:top_k_cands]
                if active_slug in order and active_slug not in top_cands:
                    top_cands.append(active_slug)
                for s in order:
                    raw_scores[s] = coarse[s] * 0.85
                if self._executor is not None and self.high_precision and len(top_cands) > 1:
                    for s, (sc, xy) in zip(top_cands, self._executor.map(eval_fine, top_cands)):
                        raw_scores[s] = sc
                        best_xy_by_slug[s] = xy
                else:
                    for s in top_cands:
                        sc, xy = eval_fine(s)
                        raw_scores[s] = sc
                        best_xy_by_slug[s] = xy
            else:
                for s in order:
                    sc, xy = eval_fine(s)
                    raw_scores[s] = sc
                    best_xy_by_slug[s] = xy

        # Update Bayesian dungeon belief under lock (safe even if register_dungeon ran concurrently)
        with self._lock:
            if locked_slug and locked_slug in self.dungeons:
                best_slug = locked_slug
                self.dungeon_probs = np.array(
                    [1.0 if s == best_slug else 0.0 for s in self.dungeon_order], dtype=np.float64
                )
            else:
                scores_arr = np.array([raw_scores[s] for s in order], dtype=np.float64)
                logits = (scores_arr - scores_arr.max()) / 0.06
                like = np.exp(logits)
                like /= like.sum() + 1e-12
                if len(probs) != len(order):
                    probs = np.full(len(order), 1.0 / max(1, len(order)), dtype=np.float64)
                prior = 0.88 * probs + 0.12 / max(1, len(order))
                post = prior * like
                post /= post.sum() + 1e-12
                best_slug = order[int(np.argmax(post))]
                best_raw = float(raw_scores.get(best_slug, 0.0))
                # Guard against un-indexed indoor caves / noisy low-confidence frames causing false zone jumps
                hold_indoor_cave = (
                    dark_void_frac > 0.50
                    and best_raw < 0.48
                    and self._smooth_px is not None
                    and active_slug in self.dungeons
                )
                hold_weak_switch = (
                    best_slug != active_slug
                    and best_raw < 0.40
                    and self._smooth_px is not None
                    and active_slug in self.dungeons
                )
                if hold_indoor_cave or hold_weak_switch:
                    best_slug = active_slug
                elif len(self.dungeon_order) == len(order):
                    self.dungeon_probs = post
                else:
                    padded = np.full(len(self.dungeon_order), 1.0 / len(self.dungeon_order), dtype=np.float64)
                    padded[: len(order)] = post * (len(order) / float(len(self.dungeon_order)))
                    padded /= padded.sum() + 1e-12
                    self.dungeon_probs = padded
            probs_snapshot = self.dungeon_probs.copy()
            order_snapshot = list(self.dungeon_order)

        if best_slug != self.active_slug:
            self.active_slug = best_slug
            self._smooth_px = None
            self._jump_cand_px = None
            self._jump_confirm_count = 0
            self.trail_px.clear()

        d = self.dungeons[best_slug]
        if best_slug not in best_xy_by_slug:
            _, best_xy_by_slug[best_slug] = eval_fine(best_slug)
        wx, wy = best_xy_by_slug[best_slug]
        ratio = self._scale_ratio[best_slug]
        raw_px = np.array([wx * ratio, wy * ratio], dtype=np.float32)

        # If inside an un-indexed dark cave with weak match, hold last known outdoor position
        if not (locked_slug and locked_slug in self.dungeons) and dark_void_frac > 0.50 and raw_scores.get(best_slug, 0.0) < 0.48 and self._smooth_px is not None:
            raw_px = self._smooth_px.copy()

        # High-precision full-resolution local refinement on native map pixels (only when full-res image is resident)
        if (
            self.high_precision
            and ratio > 1.15
            and d.map_image.shape[0] == d.map_size_px
            and raw_scores[best_slug] >= 0.30
        ):
            raw_px = self._refine_full_res(d, inner_sq, raw_px, zoom_radii_yd, angles_deg)

        # Motion-plausibility filter with 2-frame confirmation for real teleports / summons / revives
        max_step_px = plausible_yd * d.map_scale
        if self._smooth_px is None:
            self._smooth_px = raw_px
            self._jump_cand_px = None
            self._jump_confirm_count = 0
        else:
            step_px = float(np.linalg.norm(raw_px - self._smooth_px))
            if step_px <= max_step_px:
                # Normal plausible movement (walking / running / sprinting / mounted)
                self._smooth_px = 0.65 * raw_px + 0.35 * self._smooth_px
                self._jump_cand_px = None
                self._jump_confirm_count = 0
            else:
                # Implausibly large jump: require 2 consecutive consistent frames before snapping
                # (rejects 1-frame minimap glitches while still supporting summons/teleports/revives!)
                confirm_tol_px = max(25.0, 28.0 * d.map_scale)
                if self._jump_cand_px is not None and float(np.linalg.norm(raw_px - self._jump_cand_px)) <= confirm_tol_px:
                    self._jump_confirm_count += 1
                    self._jump_cand_px = 0.5 * (self._jump_cand_px + raw_px)
                else:
                    self._jump_cand_px = raw_px
                    self._jump_confirm_count = 1

                if self._jump_confirm_count >= 2:
                    # Confirmed teleport / summon / floor jump -> snap cleanly and reset trail
                    self._smooth_px = raw_px
                    self._jump_cand_px = None
                    self._jump_confirm_count = 0
                    self.trail_px.clear()
                # else: hold previous smooth_px for 1 frame until confirmed

        px_i, py_i = int(round(float(self._smooth_px[0]))), int(round(float(self._smooth_px[1])))
        self.trail_px.append((px_i, py_i))
        if len(self.trail_px) > 60:
            self.trail_px.pop(0)

        # Convert map pixel back to world (x, y, z_mean) respecting North-Up rotation
        world_x, world_y = d.map_px_to_world(float(self._smooth_px[0]), float(self._smooth_px[1]))
        world_z = float(np.mean(d.waypoints[:, 2])) if len(d.waypoints) > 0 else 0.0
        is_shared = self.is_shared_map(best_slug)
        if not is_shared:
            self._last_outdoor_slug = best_slug
            self._last_outdoor_world_pos = np.array([world_x, world_y, world_z], dtype=np.float32)

        conf = float(np.clip((raw_scores[best_slug] - 0.15) / 0.65, 0.0, 1.0))
        inference_ms = (time.perf_counter() - t0) * 1000.0
        best_idx = order_snapshot.index(best_slug) if best_slug in order_snapshot else 0

        return TrackerEstimate(
            dungeon_slug=best_slug,
            dungeon_name=d.name,
            dungeon_confidence=float(probs_snapshot[best_idx]) if best_idx < len(probs_snapshot) else 0.0,
            world_pos=np.array([world_x, world_y, world_z], dtype=np.float32),
            map_px=(px_i, py_i),
            position_confidence=conf,
            particles_px=[],
            inference_ms=inference_ms,
            all_dungeon_scores={
                s: float(probs_snapshot[i]) for i, s in enumerate(order_snapshot) if i < len(probs_snapshot)
            },
            is_shared_minimap=is_shared,
            outdoor_context_slug=self._last_outdoor_slug if is_shared else None,
        )

    def _refine_full_res(
        self,
        d: DungeonData,
        inner_sq: np.ndarray,
        coarse_px: np.ndarray,
        zoom_radii_yd: Tuple[float, ...],
        angles_deg: Tuple[float, ...],
    ) -> np.ndarray:
        """Refine position against a local native-resolution patch of d.map_image."""
        h, w = d.map_image.shape[:2]
        half_win = 224
        cx, cy = int(round(float(coarse_px[0]))), int(round(float(coarse_px[1])))
        x0 = max(0, cx - half_win)
        y0 = max(0, cy - half_win)
        x1 = min(w, cx + half_win)
        y1 = min(h, cy + half_win)
        if x1 - x0 < 64 or y1 - y0 < 64:
            return coarse_px

        patch = d.map_image[y0:y1, x0:x1]
        p_sil, p_edge = extract_structural_channels(patch, is_floorplan=True)
        crop_sil, crop_edge = extract_structural_channels(cv2.resize(inner_sq, (128, 128), interpolation=cv2.INTER_AREA), is_floorplan=False)

        best_v = -1.0
        best_xy = coarse_px
        ph, pw = patch.shape[:2]
        for r_yd in zoom_radii_yd:
            sz = int(round(2.0 * r_yd * 0.62 * d.map_scale))
            if sz < 16 or sz >= min(ph, pw) - 4:
                continue
            ts_base = cv2.resize(crop_sil, (sz, sz), interpolation=cv2.INTER_AREA)
            te_base = cv2.resize(crop_edge, (sz, sz), interpolation=cv2.INTER_AREA)
            for ang in angles_deg:
                if abs(ang) > 1e-3:
                    rot_m = cv2.getRotationMatrix2D((sz / 2.0, sz / 2.0), ang, 1.0)
                    ts = cv2.warpAffine(ts_base, rot_m, (sz, sz), borderMode=cv2.BORDER_REPLICATE)
                    te = cv2.warpAffine(te_base, rot_m, (sz, sz), borderMode=cv2.BORDER_REPLICATE)
                else:
                    ts, te = ts_base, te_base
                rs = cv2.matchTemplate(p_sil, ts, cv2.TM_CCOEFF_NORMED)
                re_ = cv2.matchTemplate(p_edge, te, cv2.TM_CCOEFF_NORMED)
                comb = 0.70 * np.nan_to_num(rs, nan=-1.0) + 0.30 * np.nan_to_num(re_, nan=-1.0)
                _, mv, _, ml = cv2.minMaxLoc(comb)
                if mv > best_v:
                    best_v = float(mv)
                    best_xy = np.array([x0 + ml[0] + sz * 0.5, y0 + ml[1] + sz * 0.5], dtype=np.float32)
        return best_xy if best_v >= 0.35 else coarse_px


def save_roi(path: str | Path, roi: MinimapROI) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(roi.to_dict(), indent=2), encoding="utf-8")


def load_roi(path: str | Path) -> MinimapROI:
    p = Path(path)
    if p.exists():
        try:
            return MinimapROI.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return MinimapROI()
