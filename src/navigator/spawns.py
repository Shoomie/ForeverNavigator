"""Approximate boss & entrance spawn markers for Classic / WoW: Forever dungeons.

Coordinates are stored in normalized 512x512 map coordinates (u, v in [0, 1]) so they align
cleanly on the 2D floorplan AND snap automatically to the nearest 3D floor height (z) in 3D view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class SpawnMarker:
    label: str
    kind: str  # "entrance", "boss", "poi"
    u: float   # 0..1 left-to-right on 512x512 map
    v: float   # 0..1 top-to-bottom on 512x512 map


# Approximate dungeon entrance & boss positions aligned to North-Up maps (u, v in [0, 1])
DUNGEON_SPAWNS: Dict[str, List[SpawnMarker]] = {
    "ragefire_chasm": [
        SpawnMarker("Entrance", "entrance", 0.44, 0.16),
        SpawnMarker("Oggleflint", "boss", 0.73, 0.48),
        SpawnMarker("Taragaman the Hungerer", "boss", 0.35, 0.65),
        SpawnMarker("Jergosh the Invoker", "boss", 0.28, 0.87),
        SpawnMarker("Bazzalan", "boss", 0.75, 0.72),
    ],
    "stockade": [
        SpawnMarker("Entrance", "entrance", 0.11, 0.39),
        SpawnMarker("Targorr the Dread", "boss", 0.35, 0.21),
        SpawnMarker("Kam Deepfury", "boss", 0.55, 0.19),
        SpawnMarker("Hamhock", "boss", 0.84, 0.38),
        SpawnMarker("Bazil Thredd", "boss", 0.24, 0.78),
        SpawnMarker("Dextren Ward", "boss", 0.56, 0.75),
    ],
    "deadmines": [
        SpawnMarker("Entrance", "entrance", 0.88, 0.27),
        SpawnMarker("Rhahk'Zor", "boss", 0.68, 0.46),
        SpawnMarker("Sneed", "boss", 0.45, 0.36),
        SpawnMarker("Gilnid", "boss", 0.26, 0.29),
        SpawnMarker("Mr. Smite / VanCleef", "boss", 0.56, 0.72),
    ],
    "shadowfang_keep": [
        SpawnMarker("Entrance", "entrance", 0.22, 0.32),
        SpawnMarker("Rethilgore", "boss", 0.36, 0.44),
        SpawnMarker("Razorclaw / Silverlaine", "boss", 0.52, 0.55),
        SpawnMarker("Commander Springvale", "boss", 0.62, 0.62),
        SpawnMarker("Fenrus / Nandos", "boss", 0.74, 0.46),
        SpawnMarker("Archmage Arugal", "boss", 0.82, 0.34),
    ],
    "wailing_caverns": [
        SpawnMarker("Entrance", "entrance", 0.48, 0.22),
        SpawnMarker("Lady Anacondra", "boss", 0.66, 0.44),
        SpawnMarker("Lord Cobrahn", "boss", 0.78, 0.52),
        SpawnMarker("Lord Pythas", "boss", 0.60, 0.72),
        SpawnMarker("Lord Serpentis", "boss", 0.32, 0.66),
        SpawnMarker("Mutanus the Devourer", "boss", 0.26, 0.48),
    ],
    "ruins_of_lordaeron_courtyard": [
        SpawnMarker("Courtyard Gate", "entrance", 0.39, 0.42),
        SpawnMarker("Throne Room", "poi", 0.44, 0.68),
        SpawnMarker("Undercity Elevators", "poi", 0.44, 0.80),
    ],
    "ruins_of_lordaeron": [
        SpawnMarker("Sewer Entrance", "entrance", 0.86, 0.56),
        SpawnMarker("Trade Quarter (Center)", "poi", 0.40, 0.54),
        SpawnMarker("Royal Quarter", "boss", 0.38, 0.76),
        SpawnMarker("Apothecarium", "poi", 0.24, 0.32),
    ],
    "molten_core": [
        SpawnMarker("Entrance", "entrance", 0.70, 0.20),
        SpawnMarker("Lucifron", "boss", 0.56, 0.38),
        SpawnMarker("Magmadar", "boss", 0.72, 0.52),
        SpawnMarker("Garr", "boss", 0.44, 0.58),
        SpawnMarker("Baron Geddon / Shazzrah", "boss", 0.50, 0.74),
        SpawnMarker("Ragnaros", "boss", 0.22, 0.82),
    ],
    "onyxias_lair": [
        SpawnMarker("Entrance", "entrance", 0.78, 0.22),
        SpawnMarker("Onyxia", "boss", 0.26, 0.72),
    ],
}


def get_spawns_for_dungeon(slug: str) -> List[SpawnMarker]:
    return DUNGEON_SPAWNS.get(slug, [])
