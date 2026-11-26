"""Multi-dungeon auto-detector, Graph Particle Filter, Adaptive UI Masker, and CombatLog watcher."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .encoder import ONNXVisualEncoder, build_default_ui_mask


@dataclass
class DungeonData:
    slug: str
    name: str
    combatlog_keywords: List[str]
    waypoints: np.ndarray        # (N, 3) float32
    adjacency: List[List[int]]   # len N
    map_image: np.ndarray        # (H, W, 3) uint8
    map_origin_xy: np.ndarray    # (2,) float32
    map_scale: float
    embeddings_f32: np.ndarray   # (V, D) float32 normalized
    view_wp_idx: np.ndarray      # (V,) int32
    wp_to_views: List[np.ndarray]
    map_rot90: int = 0           # 1 = 90 CCW, 3 = 90 CW (exact WoW North-Up WMO minimap)
    bounds_xy: Optional[Tuple[float, float, float, float]] = None  # (xmin, ymin, xmax, ymax)
    native_map_px: int = 0       # Full native pixel dimension when map_image holds a low-RAM preview
    map_full_path: Optional[Path] = None  # On-disk full-res PNG/NPZ for dynamic SSD/HDD streaming

    @property
    def map_size_px(self) -> int:
        return self.native_map_px if self.native_map_px > 0 else int(self.map_image.shape[0])

    def world_to_map_px(self, xy: np.ndarray) -> Tuple[int, int]:
        h = self.map_size_px
        px_raw = (float(xy[0]) - float(self.map_origin_xy[0])) * self.map_scale
        py_raw = (h - 1) - (float(xy[1]) - float(self.map_origin_xy[1])) * self.map_scale
        if self.map_rot90 == 1:
            px, py = py_raw, (h - 1) - px_raw
        elif self.map_rot90 == 3:
            px, py = (h - 1) - py_raw, px_raw
        else:
            px, py = px_raw, py_raw
        return int(round(px)), int(round(py))

    def map_px_to_world(self, px: float, py: float) -> Tuple[float, float]:
        h = self.map_size_px
        if self.map_rot90 == 1:
            px_raw, py_raw = (h - 1) - float(py), float(px)
        elif self.map_rot90 == 3:
            px_raw, py_raw = float(py), (h - 1) - float(px)
        else:
            px_raw, py_raw = float(px), float(py)
        wx = float(self.map_origin_xy[0]) + px_raw / self.map_scale
        wy = float(self.map_origin_xy[1]) + ((h - 1) - py_raw) / self.map_scale
        return wx, wy


@dataclass
class TrackerEstimate:
    dungeon_slug: str
    dungeon_name: str
    dungeon_confidence: float           # 0..1 probability across all indexed dungeons
    world_pos: np.ndarray               # (3,) float32
    map_px: Tuple[int, int]
    position_confidence: float          # 0..1 concentration of particle cloud
    particles_px: List[Tuple[int, int]]
    inference_ms: float
    all_dungeon_scores: Dict[str, float]
    is_shared_minimap: bool = False
    outdoor_context_slug: Optional[str] = None


class AdaptiveUIMasker:
    """Detects static HUD pixels (custom UIs, raid frames, WeakAuras) over time and masks them."""

    def __init__(self, size: int = 224):
        self.size = size
        self.base_mask = build_default_ui_mask(size, size)
        self._prev_gray: Optional[np.ndarray] = None
        self._static_accum = np.zeros((size, size), dtype=np.float32)
        self._moving_frames = 0

    def update_and_get_mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame_rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)

        if self._prev_gray is not None:
            diff = np.abs(gray - self._prev_gray)
            # Only learn static UI when the camera/world is actually moving
            if float(np.mean(diff)) > 4.0:
                self._moving_frames += 1
                is_static = (diff < 1.5).astype(np.float32)
                self._static_accum = 0.90 * self._static_accum + 0.10 * is_static
        self._prev_gray = gray

        if self._moving_frames < 8:
            return self.base_mask

        # Pixels that stay static >75% of moving frames are almost certainly custom UI panels
        dynamic_ui = (self._static_accum > 0.75).astype(np.float32)
        # Never mask out more than 55% of the screen (safety guard)
        combined = self.base_mask * (1.0 - dynamic_ui)
        if float(np.mean(combined)) < 0.45:
            return self.base_mask
        return combined


class CombatLogWatcher:
    """Optional zero-CPU tailer for WoWCombatLog-*.txt to detect zone/encounter hints."""

    def __init__(self, logs_dir: Optional[str | Path] = None):
        self.logs_dir = Path(logs_dir) if logs_dir else None
        self._active_file: Optional[Path] = None
        self._offset: int = 0
        self.last_zone_hint: Optional[str] = None

    def poll_hint(self, dungeons: List[DungeonData]) -> Optional[str]:
        if self.logs_dir is None or not self.logs_dir.exists():
            return None
        logs = sorted(self.logs_dir.glob("WoWCombatLog-*.txt"), key=lambda p: p.stat().st_mtime)
        if not logs:
            return None
        latest = logs[-1]
        if latest != self._active_file:
            self._active_file = latest
            # Start near the tail of the latest log
            size = latest.stat().st_size
            self._offset = max(0, size - 32768)

        try:
            with open(latest, "r", encoding="utf-8", errors="ignore") as f:
                f.seek(self._offset)
                chunk = f.read(65536)
                self._offset = f.tell()
        except Exception:
            return self.last_zone_hint

        if chunk:
            low = chunk.lower()
            for d in dungeons:
                for kw in d.combatlog_keywords:
                    if kw.lower() in low:
                        self.last_zone_hint = d.slug
        return self.last_zone_hint


def load_dungeon_npz(
    npz_path: Path,
    max_preview_px: Optional[int] = None,
    load_embeddings: bool = True,
) -> DungeonData:
    """Load a DungeonData object from a pre-indexed or cached .npz file.

    When `max_preview_px` is set and the stored image is larger, downscales `map_image` in RAM
    to a lightweight preview while preserving `native_map_px` and `map_full_path` for on-demand streaming.
    When `load_embeddings=False`, skips 3D visual embeddings & graph edges to save ~100+ MB RAM.
    """
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as z:
        waypoints = z["waypoints"].astype(np.float32) if "waypoints" in z else np.zeros((1, 3), dtype=np.float32)
        n_wp = max(1, len(waypoints))

        adj: List[List[int]] = [[] for _ in range(n_wp)]
        if load_embeddings and "edges_src" in z and "edges_dst" in z:
            edges_src = z["edges_src"].astype(np.int32)
            edges_dst = z["edges_dst"].astype(np.int32)
            for u, v in zip(edges_src.tolist(), edges_dst.tolist()):
                adj[u].append(v)

        if load_embeddings and "embeddings_i8" in z:
            embs = z["embeddings_i8"].astype(np.float32) / 127.0
            if len(embs) > 0:
                embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        else:
            embs = np.zeros((0, 1), dtype=np.float32)

        if load_embeddings and "view_wp_idx" in z:
            view_wp_idx = z["view_wp_idx"].astype(np.int32)
            wp_to_views: List[List[int]] = [[] for _ in range(n_wp)]
            for vi, wi in enumerate(view_wp_idx.tolist()):
                if 0 <= wi < n_wp:
                    wp_to_views[wi].append(vi)
            wp_to_views_arr = [np.array(v, dtype=np.int32) for v in wp_to_views]
        else:
            view_wp_idx = np.zeros((0,), dtype=np.int32)
            wp_to_views_arr = []

        kw_raw = z["combatlog_keywords"].tolist() if "combatlog_keywords" in z else []
        b_xy: Optional[Tuple[float, float, float, float]] = None
        if "bounds_xy" in z:
            arr_b = z["bounds_xy"]
            if len(arr_b) == 4 and np.all(np.isfinite(arr_b)):
                b_xy = (float(arr_b[0]), float(arr_b[1]), float(arr_b[2]), float(arr_b[3]))

        map_img = z["map_image"].astype(np.uint8)
        native_px = int(z["native_map_px"]) if "native_map_px" in z else int(map_img.shape[0])
        png_sibling = npz_path.with_suffix(".png")
        full_path = png_sibling if png_sibling.exists() else npz_path
        if max_preview_px is not None and map_img.shape[0] > max_preview_px:
            map_img = cv2.resize(map_img, (max_preview_px, max_preview_px), interpolation=cv2.INTER_AREA)

        return DungeonData(
            slug=str(z["slug"]),
            name=str(z["name"]),
            combatlog_keywords=[str(k) for k in kw_raw],
            waypoints=waypoints,
            adjacency=adj,
            map_image=map_img,
            map_origin_xy=z["map_origin_xy"].astype(np.float32),
            map_scale=float(z["map_scale"]),
            embeddings_f32=embs,
            view_wp_idx=view_wp_idx,
            wp_to_views=wp_to_views_arr,
            map_rot90=int(z["map_rot90"]) if "map_rot90" in z else 0,
            bounds_xy=b_xy,
            native_map_px=native_px,
            map_full_path=full_path,
        )


class LightweightDungeonStore:
    """Zero-ONNX, low-RAM dungeon index container for 2D minimap mode (saves ~160 MB vs MultiDungeonTracker)."""

    def __init__(self, index_dir: str | Path, max_preview_px: int = 512):
        self.index_dir = Path(index_dir)
        self.dungeons: Dict[str, DungeonData] = {}
        self.dungeon_order: List[str] = []
        manifest_path = self.index_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for entry in manifest.get("dungeons", []):
                d = load_dungeon_npz(
                    self.index_dir / entry["file"],
                    max_preview_px=max_preview_px,
                    load_embeddings=False,
                )
                self.dungeons[d.slug] = d
                self.dungeon_order.append(d.slug)

    def register_dungeon(self, dungeon: DungeonData) -> None:
        self.dungeons[dungeon.slug] = dungeon
        if dungeon.slug not in self.dungeon_order:
            self.dungeon_order.append(dungeon.slug)


class MultiDungeonTracker:
    """
    Auto-detects which dungeon the player is in (even after mid-dungeon summons/revives)
    and tracks position using a Graph Particle Filter with global relocalization.
    """

    def __init__(
        self,
        index_dir: str | Path,
        num_particles: int = 240,
        wow_logs_dir: Optional[str | Path] = None,
        enable_adaptive_ui_mask: bool = True,
    ):
        self.index_dir = Path(index_dir)
        manifest = json.loads((self.index_dir / "manifest.json").read_text(encoding="utf-8"))
        self.encoder = ONNXVisualEncoder(self.index_dir / manifest["encoder_onnx"])

        self.dungeons: Dict[str, DungeonData] = {}
        self.dungeon_order: List[str] = []
        for entry in manifest["dungeons"]:
            d = self._load_dungeon(self.index_dir / entry["file"])
            self.dungeons[d.slug] = d
            self.dungeon_order.append(d.slug)

        self.num_particles = num_particles
        self.rng = np.random.default_rng(1234)
        self.ui_masker = AdaptiveUIMasker(self.encoder.input_size) if enable_adaptive_ui_mask else None
        self.combatlog = CombatLogWatcher(wow_logs_dir)

        # Temporal dungeon auto-detection belief (uniform prior across all indexed dungeons)
        n_d = len(self.dungeon_order)
        self.dungeon_probs = np.full(n_d, 1.0 / max(1, n_d), dtype=np.float64)
        self.active_slug: str = self.dungeon_order[0]
        self.locked_dungeon_slug: Optional[str] = None  # User can optionally pin a dungeon

        # Per-dungeon particle filter state: array of waypoint indices
        self.particles: Dict[str, np.ndarray] = {}
        for slug, d in self.dungeons.items():
            self.particles[slug] = self.rng.integers(0, len(d.waypoints), size=num_particles)

        self.trail_world: List[np.ndarray] = []

    def _load_dungeon(self, npz_path: Path) -> DungeonData:
        return load_dungeon_npz(npz_path)

    def register_dungeon(self, dungeon: DungeonData) -> None:
        """Dynamically register an on-demand loaded map/dungeon."""
        slug = dungeon.slug
        self.dungeons[slug] = dungeon
        if slug not in self.dungeon_order:
            self.dungeon_order.append(slug)
            n_wp = max(1, len(dungeon.waypoints))
            self.particles[slug] = self.rng.integers(0, n_wp, size=self.num_particles)
            # Extend dungeon_probs preserving normalized prior
            new_probs = np.zeros(len(self.dungeon_order), dtype=np.float64)
            old_n = len(self.dungeon_probs)
            if old_n > 0:
                new_probs[:old_n] = self.dungeon_probs * (old_n / float(len(self.dungeon_order)))
            new_probs[-1] = 1.0 / float(len(self.dungeon_order))
            s = float(new_probs.sum())
            self.dungeon_probs = new_probs / s if s > 0 else np.full(len(self.dungeon_order), 1.0 / len(self.dungeon_order))

    def reset(self) -> None:
        n_d = len(self.dungeon_order)
        self.dungeon_probs = np.full(n_d, 1.0 / max(1, n_d), dtype=np.float64)
        for slug, d in self.dungeons.items():
            self.particles[slug] = self.rng.integers(0, len(d.waypoints), size=self.num_particles)
        self.trail_world.clear()

    def update(self, frame_rgb: np.ndarray, use_adaptive_mask: bool = True) -> TrackerEstimate:
        """Process one captured frame and return the updated dungeon + position estimate."""
        t0 = time.perf_counter()
        mask = None
        if use_adaptive_mask and self.ui_masker is not None:
            mask = self.ui_masker.update_and_get_mask(frame_rgb)
        emb = self.encoder.encode_single(frame_rgb, ui_mask=mask, color_normalize=True)

        # 1. Global similarity across all dungeons for auto-detection (works anywhere, even mid-dungeon)
        raw_dungeon_scores: Dict[str, float] = {}
        wp_scores_by_dungeon: Dict[str, np.ndarray] = {}

        for slug in self.dungeon_order:
            d = self.dungeons[slug]
            n_wp = len(d.waypoints)
            if len(d.embeddings_f32) == 0:
                wp_scores_by_dungeon[slug] = np.full(n_wp, -1.0, dtype=np.float32)
                raw_dungeon_scores[slug] = -1.0
                continue
            # Fast matrix-vector dot product: (V, D) @ (D,) -> (V,)
            sims = d.embeddings_f32 @ emb
            # Max similarity per waypoint (across all orientations/pitches)
            wp_sim = np.full(n_wp, -1.0, dtype=np.float32)
            np.maximum.at(wp_sim, d.view_wp_idx, sims)
            wp_scores_by_dungeon[slug] = wp_sim
            # Top-5 mean similarity is robust to single outlier views
            k = min(5, n_wp)
            top_k = np.partition(wp_sim, -k)[-k:]
            raw_dungeon_scores[slug] = float(np.mean(top_k))

        # 2. Update temporal dungeon belief (Bayesian filter with soft switch floor)
        if self.locked_dungeon_slug and self.locked_dungeon_slug in self.dungeons:
            best_slug = self.locked_dungeon_slug
            self.dungeon_probs = np.array(
                [1.0 if s == best_slug else 0.0 for s in self.dungeon_order], dtype=np.float64
            )
        else:
            scores_arr = np.array([raw_dungeon_scores[s] for s in self.dungeon_order], dtype=np.float64)
            # Softmax likelihood with temperature
            temp = 0.045
            logits = (scores_arr - scores_arr.max()) / temp
            like = np.exp(logits)
            like /= like.sum() + 1e-12

            # Optional CombatLog hint boost (gentle prior, visual evidence still dominates)
            log_hint = self.combatlog.poll_hint(list(self.dungeons.values()))
            if log_hint is not None and log_hint in self.dungeons:
                hint_i = self.dungeon_order.index(log_hint)
                like[hint_i] *= 1.35
                like /= like.sum() + 1e-12

            # Temporal transition: 92% stay in current dungeon, 8% allow switching (teleports/new zone)
            prior = 0.92 * self.dungeon_probs + 0.08 / len(self.dungeon_order)
            post = prior * like
            post /= post.sum() + 1e-12
            self.dungeon_probs = post
            best_idx = int(np.argmax(self.dungeon_probs))
            best_slug = self.dungeon_order[best_idx]

        if best_slug != self.active_slug:
            self.active_slug = best_slug
            self.trail_world.clear()

        # 3. Update Graph Particle Filter for the active dungeon
        d = self.dungeons[best_slug]
        wp_sim = wp_scores_by_dungeon[best_slug]
        est_pos, pos_conf, p_indices = self._step_particle_filter(d, wp_sim)

        self.trail_world.append(est_pos.copy())
        if len(self.trail_world) > 60:
            self.trail_world.pop(0)

        inference_ms = (time.perf_counter() - t0) * 1000.0
        best_d_idx = self.dungeon_order.index(best_slug)
        return TrackerEstimate(
            dungeon_slug=best_slug,
            dungeon_name=d.name,
            dungeon_confidence=float(self.dungeon_probs[best_d_idx]),
            world_pos=est_pos,
            map_px=d.world_to_map_px(est_pos[:2]),
            position_confidence=pos_conf,
            particles_px=[d.world_to_map_px(d.waypoints[pi, :2]) for pi in p_indices[::3]],
            inference_ms=inference_ms,
            all_dungeon_scores={
                s: float(self.dungeon_probs[i]) for i, s in enumerate(self.dungeon_order)
            },
        )

    def _step_particle_filter(
        self,
        d: DungeonData,
        wp_sim: np.ndarray,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Augmented Graph Particle Filter: local diffusion + global relocalization."""
        parts = self.particles[d.slug]
        n_p = len(parts)
        n_wp = len(d.waypoints)

        # Motion step: 85% diffuse 0-2 hops along walkable dungeon graph, 15% global relocalization
        # (handles warlock summons, battle rez, release spirit, and initial lock-on anywhere!)
        n_global = max(10, int(n_p * 0.15))
        n_local = n_p - n_global

        new_parts = np.empty(n_p, dtype=np.int32)
        for i in range(n_local):
            cur = int(parts[i])
            r = self.rng.random()
            if r < 0.35 or not d.adjacency[cur]:
                new_parts[i] = cur
            else:
                nxt = d.adjacency[cur][int(self.rng.integers(0, len(d.adjacency[cur])))]
                if r > 0.78 and d.adjacency[nxt]:
                    nxt = d.adjacency[nxt][int(self.rng.integers(0, len(d.adjacency[nxt])))]
                new_parts[i] = nxt

        # Global relocalization proposals sampled from top-15 visual matches
        k_top = min(15, n_wp)
        top_wps = np.argpartition(wp_sim, -k_top)[-k_top:]
        top_w = np.exp((wp_sim[top_wps] - wp_sim[top_wps].max()) / 0.05)
        top_w /= top_w.sum() + 1e-12
        new_parts[n_local:] = self.rng.choice(top_wps, size=n_global, p=top_w)

        # Measurement weighting
        p_sims = wp_sim[new_parts]
        weights = np.exp((p_sims - p_sims.max()) / 0.055)
        weights /= weights.sum() + 1e-12

        # Systematic resampling
        positions = (self.rng.random() + np.arange(n_p)) / n_p
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.0
        resampled = new_parts[np.searchsorted(cumsum, positions)]
        self.particles[d.slug] = resampled

        # Estimate position using the dominant spatial mode (robust to multi-modal corridor ambiguity)
        # Pick the particle with highest local neighborhood density
        pts = d.waypoints[resampled]  # (P, 3)
        mode_wp = int(np.bincount(resampled, minlength=n_wp).argmax())
        mode_pt = d.waypoints[mode_wp]
        dists = np.linalg.norm(pts - mode_pt[None, :], axis=1)
        inliers = dists <= 14.0
        est_pos = pts[inliers].mean(axis=0).astype(np.float32) if np.any(inliers) else mode_pt
        pos_conf = float(np.mean(inliers))
        return est_pos, pos_conf, resampled
