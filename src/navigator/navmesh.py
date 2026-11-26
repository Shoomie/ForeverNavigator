"""Walkable floor/waypoint graph extraction and 2D floorplan map rendering from WMO geometry."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from .wmo_parser import WMOGroup


@dataclass
class DungeonNavMesh:
    waypoints: np.ndarray          # (N, 3) float32 world (x, y, z)
    edges_src: np.ndarray          # (E,) int32
    edges_dst: np.ndarray          # (E,) int32
    bbox_min: np.ndarray           # (3,) float32
    bbox_max: np.ndarray           # (3,) float32
    map_image: np.ndarray          # (H, W, 3) uint8 RGB top-down floorplan
    map_origin_xy: np.ndarray      # (2,) float32 world (x_min, y_min) of map
    map_scale: float               # pixels per world yard


def build_navmesh_and_map(
    groups: List[WMOGroup],
    cell_size: float = 5.0,
    max_waypoints: int = 500,
    map_size: int = 512,
) -> DungeonNavMesh:
    """Extract walkable floor waypoints, connectivity graph, and a 2D top-down map."""
    all_verts = []
    floor_centers = []
    floor_areas = []
    wall_points = []

    for g in groups:
        if len(g.vertices) == 0 or len(g.indices) < 3:
            continue
        all_verts.append(g.vertices)
        tris = g.indices.reshape(-1, 3)
        v0 = g.vertices[tris[:, 0]]
        v1 = g.vertices[tris[:, 1]]
        v2 = g.vertices[tris[:, 2]]

        e1 = v1 - v0
        e2 = v2 - v0
        cross = np.cross(e1, e2)
        norms = np.linalg.norm(cross, axis=1)
        valid = norms > 1e-4
        if not np.any(valid):
            continue

        cross = cross[valid]
        norms = norms[valid]
        areas = 0.5 * norms
        nz = cross[:, 2] / norms
        centers = (v0[valid] + v1[valid] + v2[valid]) / 3.0

        # Upward-facing floor triangles
        is_floor = (nz >= 0.65) & (areas >= 0.25)
        if np.any(is_floor):
            floor_centers.append(centers[is_floor])
            floor_areas.append(areas[is_floor])

        # Vertical wall triangles for crisp 2D floorplan outlines
        is_wall = (np.abs(nz) < 0.35) & (areas >= 0.5)
        if np.any(is_wall):
            wall_points.append(centers[is_wall])

    if not all_verts:
        raise ValueError("No geometry found in WMO groups")

    verts_cat = np.concatenate(all_verts, axis=0)
    bbox_min = verts_cat.min(axis=0).astype(np.float32)
    bbox_max = verts_cat.max(axis=0).astype(np.float32)

    if floor_centers:
        fc = np.concatenate(floor_centers, axis=0)
        fa = np.concatenate(floor_areas, axis=0)
    else:
        fc = verts_cat
        fa = np.ones(len(verts_cat), dtype=np.float32)

    # Quantize floor samples into (ix, iy, iz_layer) cells so multi-floor dungeons stay separated
    z_layer_h = 5.0
    ix = np.floor((fc[:, 0] - bbox_min[0]) / cell_size).astype(np.int32)
    iy = np.floor((fc[:, 1] - bbox_min[1]) / cell_size).astype(np.int32)
    iz = np.floor((fc[:, 2] - bbox_min[2]) / z_layer_h).astype(np.int32)

    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    for idx_i, key in enumerate(zip(ix.tolist(), iy.tolist(), iz.tolist())):
        buckets.setdefault(key, []).append(idx_i)

    raw_pts = []
    for _key, idxs in buckets.items():
        pts = fc[idxs]
        w = fa[idxs]
        # Require meaningful floor area in the cell to reject tiny decorative ledges
        if w.sum() < 1.5:
            continue
        mean_pt = (pts * w[:, None]).sum(axis=0) / w.sum()
        raw_pts.append(mean_pt)

    if not raw_pts:
        raw_pts = [fc.mean(axis=0)]

    waypoints = np.array(raw_pts, dtype=np.float32)

    # Subsample evenly if dungeon is huge so prototype builds quickly
    if len(waypoints) > max_waypoints:
        # Farthest-point / stride subsampling sorted by spatial morton-like key
        order = np.lexsort((waypoints[:, 2], waypoints[:, 1], waypoints[:, 0]))
        sel = order[np.linspace(0, len(order) - 1, max_waypoints, dtype=int)]
        waypoints = waypoints[sel]

    # Build walkable connectivity graph
    n = len(waypoints)
    max_xy_dist = cell_size * 1.9
    max_dz = 4.0
    adj: List[List[int]] = [[] for _ in range(n)]
    diff_xy = waypoints[:, None, :2] - waypoints[None, :, :2]
    dist_xy = np.linalg.norm(diff_xy, axis=-1)
    diff_z = np.abs(waypoints[:, None, 2] - waypoints[None, :, 2])
    connected = (dist_xy > 1e-3) & (dist_xy <= max_xy_dist) & (diff_z <= max_dz)

    # Filter out tiny disconnected components (< 4 nodes) if larger components exist
    ys, xs = np.where(connected)
    for u, v in zip(ys.tolist(), xs.tolist()):
        adj[u].append(v)

    visited = [False] * n
    components: List[List[int]] = []
    for i in range(n):
        if visited[i]:
            continue
        comp = []
        q = deque([i])
        visited[i] = True
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nb in adj[cur]:
                if not visited[nb]:
                    visited[nb] = True
                    q.append(nb)
        components.append(comp)

    components.sort(key=len, reverse=True)
    if components and len(components[0]) >= 6:
        keep = [idx for comp in components if len(comp) >= max(4, len(components[0]) // 20) for idx in comp]
        keep = sorted(set(keep))
        waypoints = waypoints[keep]
        n = len(waypoints)
        diff_xy = waypoints[:, None, :2] - waypoints[None, :, :2]
        dist_xy = np.linalg.norm(diff_xy, axis=-1)
        diff_z = np.abs(waypoints[:, None, 2] - waypoints[None, :, 2])
        connected = (dist_xy > 1e-3) & (dist_xy <= max_xy_dist) & (diff_z <= max_dz)
        ys, xs = np.where(connected)

    edges_src = ys.astype(np.int32)
    edges_dst = xs.astype(np.int32)

    # Render clean 2D top-down floorplan map
    map_img, origin_xy, scale = _render_floorplan_2d(
        fc,
        np.concatenate(wall_points, axis=0) if wall_points else np.zeros((0, 3), dtype=np.float32),
        waypoints,
        edges_src,
        edges_dst,
        bbox_min,
        bbox_max,
        map_size=map_size,
    )

    return DungeonNavMesh(
        waypoints=waypoints,
        edges_src=edges_src,
        edges_dst=edges_dst,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        map_image=map_img,
        map_origin_xy=origin_xy,
        map_scale=scale,
    )


def _render_floorplan_2d(
    floor_pts: np.ndarray,
    wall_pts: np.ndarray,
    waypoints: np.ndarray,
    edges_src: np.ndarray,
    edges_dst: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    map_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray, float]:
    pad = 12.0
    span_x = max(10.0, float(bbox_max[0] - bbox_min[0]) + 2 * pad)
    span_y = max(10.0, float(bbox_max[1] - bbox_min[1]) + 2 * pad)
    span = max(span_x, span_y)
    scale = (map_size - 24) / span

    cx = 0.5 * (bbox_min[0] + bbox_max[0])
    cy = 0.5 * (bbox_min[1] + bbox_max[1])
    origin_x = cx - (map_size / 2.0) / scale
    origin_y = cy - (map_size / 2.0) / scale
    origin_xy = np.array([origin_x, origin_y], dtype=np.float32)

    def to_px(xy: np.ndarray) -> np.ndarray:
        px = np.round((xy[:, 0] - origin_x) * scale).astype(np.int32)
        py = np.round(map_size - 1 - (xy[:, 1] - origin_y) * scale).astype(np.int32)
        return np.stack([px, py], axis=-1)

    # Dark parchment / slate background
    canvas = np.full((map_size, map_size, 3), (20, 22, 28), dtype=np.uint8)

    # Draw floor shaded by elevation
    if len(floor_pts) > 0:
        f_px = to_px(floor_pts[:, :2])
        valid = (f_px[:, 0] >= 0) & (f_px[:, 0] < map_size) & (f_px[:, 1] >= 0) & (f_px[:, 1] < map_size)
        f_px = f_px[valid]
        z = floor_pts[valid, 2]
        z_min, z_max = float(z.min()), float(z.max())
        z_norm = (z - z_min) / max(1e-3, z_max - z_min)
        floor_layer = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        for (x, y), zn in zip(f_px, z_norm):
            # Subtle warm stone elevation gradient
            r = int(58 + 52 * zn)
            g = int(54 + 46 * zn)
            b = int(48 + 34 * zn)
            cv2.circle(floor_layer, (int(x), int(y)), max(2, int(scale * 1.4)), (r, g, b), -1)
        mask = floor_layer.sum(axis=-1) > 0
        canvas[mask] = floor_layer[mask]

    # Draw subtle wall outlines
    if len(wall_pts) > 0:
        w_px = to_px(wall_pts[:, :2])
        valid = (w_px[:, 0] >= 0) & (w_px[:, 0] < map_size) & (w_px[:, 1] >= 0) & (w_px[:, 1] < map_size)
        for x, y in w_px[valid][::2]:
            canvas[int(y), int(x)] = (145, 138, 125)

    # Draw subtle walkable graph edges
    wp_px = to_px(waypoints[:, :2])
    for u, v in zip(edges_src.tolist(), edges_dst.tolist()):
        if u < v:
            p1 = (int(wp_px[u, 0]), int(wp_px[u, 1]))
            p2 = (int(wp_px[v, 0]), int(wp_px[v, 1]))
            cv2.line(canvas, p1, p2, (72, 82, 96), 1, cv2.LINE_AA)

    return canvas, origin_xy, float(scale)
