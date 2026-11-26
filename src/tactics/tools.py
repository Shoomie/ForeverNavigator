"""Positioning & yard distance measurement tools for tactical raid coordination."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

# Common WoW Classic/Forever tactical distances (in yards)
PRESET_RANGES_YD = (
    ("Melee (5 yd)", 5.0, "#f56565"),
    ("Spread (8 yd)", 8.0, "#ed8936"),
    ("Chain (10 yd)", 10.0, "#ecc94b"),
    ("Caster (30 yd)", 30.0, "#4299e1"),
    ("Healer (40 yd)", 40.0, "#48bb78"),
)


@dataclass
class RulerMeasurement:
    start_uv: Tuple[float, float]  # normalized [0..1] on map
    end_uv: Tuple[float, float]
    yards: float


def measure_yards(
    u0: float,
    v0: float,
    u1: float,
    v1: float,
    map_size_px: int,
    map_scale_px_per_yd: float,
) -> float:
    """Compute exact Euclidean distance in WoW world yards between two normalized map points."""
    dx_px = (u1 - u0) * float(map_size_px)
    dy_px = (v1 - v0) * float(map_size_px)
    dist_px = math.hypot(dx_px, dy_px)
    return dist_px / max(1e-4, float(map_scale_px_per_yd))


def yards_to_map_px(yards: float, map_scale_px_per_yd: float) -> float:
    return float(yards) * float(map_scale_px_per_yd)
