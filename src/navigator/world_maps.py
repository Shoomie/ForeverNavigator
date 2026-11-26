"""All-World Map Discovery, On-Demand CASC BLP Stitching, and Caching for WoW: Forever.

Supports:
- Pre-indexed dungeons & raids in ``data/index/``
- All WoW: Forever Continents, Zones, Cities, Battlegrounds, Dungeons, Raids, and
  custom WoW: Forever zones (Mount Hyjal, Zephras Isle, Darkspear Islands,
  Riverglades, Shen'dralas, plus SoD/Forever instances)
- Dynamic discovery from ``LocalCASC`` (``UiMap.db2``, ``UiMapAssignment.db2``,
  ``World/Minimaps/<Folder>/mapXX_YY.blp``) + bundled offline catalog fallback
- On-demand BLP tile stitching into ``DungeonData`` with PNG + NPZ caching in
  ``data/cache/maps/``
"""

from __future__ import annotations

import json
import re
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .blp_decoder import decode_blp
from .casc_reader import LocalCASC
from .tracker import DungeonData, load_dungeon_npz

ADT_YARD_SIZE = 533.3333333

MAP_CATEGORIES: Tuple[str, ...] = (
    "Raids",
    "Dungeons",
    "Battlegrounds",
    "Cities",
    "Zones",
    "Continents",
    "Other",
)

# Default local paths (gracefully ignored if absent)
_DEFAULT_WOW_DIRS = (
    Path(r"E:\Spel\World of Warcraft"),
    Path(r"C:\Program Files (x86)\World of Warcraft"),
    Path(r"D:\World of Warcraft"),
)
_DEFAULT_LISTFILES = (
    Path(__file__).resolve().parents[2] / "data" / "listfile.csv",
    Path(r"E:\Programmering\WoW\MapNavigate\data\listfile.csv"),
)

_UIMAP_FDID = 1957206
_UIMAP_ASSIGN_FDID = 1957219


@dataclass(frozen=True)
class WorldMapEntry:
    slug: str
    name: str
    category: str
    map_id: Optional[int] = None
    ui_map_id: Optional[int] = None
    source: str = "catalog"  # "indexed" | "casc_zone" | "casc_minimap" | "catalog"
    cached: bool = False
    playable: bool = True
    minimap_folder: Optional[str] = None
    bounds_xy: Optional[Tuple[float, float, float, float]] = None  # (xmin, ymin, xmax, ymax)
    keywords: Tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Categorization helpers for existing indexed dungeons
# ---------------------------------------------------------------------------

_INDEXED_CATEGORIES: Dict[str, str] = {
    "ahnqiraj_ruins": "Raids",
    "ahnqiraj_temple": "Raids",
    "blackwing_lair": "Raids",
    "molten_core": "Raids",
    "naxxramas": "Raids",
    "onyxias_lair": "Raids",
    "zul_gurub": "Raids",
    "ruins_of_lordaeron": "Cities",
    "ruins_of_lordaeron_courtyard": "Battlegrounds",
}


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s or "unknown_map"


# ---------------------------------------------------------------------------
# Bundled WoW: Forever Offline Catalog (always available even without WoW)
# Tuple: (slug, name, category, ui_map_id, minimap_folder, bounds_xy)
# bounds_xy = (xmin, ymin, xmax, ymax) in WoW world coordinates
# ---------------------------------------------------------------------------

_BUNDLED_WORLD_MAPS: List[
    Tuple[str, str, str, Optional[int], Optional[str], Optional[Tuple[float, float, float, float]]]
] = [
    # Continents
    ("azeroth_world", "Azeroth (World)", "Continents", 947, "Azeroth", (-16000.0, -19200.0, 7500.0, 16000.0)),
    ("eastern_kingdoms", "Eastern Kingdoms", "Continents", 1415, "Azeroth", (-16000.0, -19199.9, 7466.6, 16000.0)),
    ("kalimdor", "Kalimdor", "Continents", 1414, "Kalimdor", (-11733.3, -19733.2, 12799.9, 17066.6)),
    # Capital Cities
    ("stormwind_city", "Stormwind City", "Cities", 1453, "Azeroth", (-9154.2, -14.6, -7995.8, 1722.9)),
    ("orgrimmar", "Orgrimmar", "Cities", 1454, "Kalimdor", (1338.5, -5083.2, 2273.9, -3680.6)),
    ("ironforge", "Ironforge", "Cities", 1455, "Azeroth", (-5096.8, -1504.2, -4569.2, -713.6)),
    ("thunder_bluff", "Thunder Bluff", "Cities", 1456, "Kalimdor", (-1545.8, -527.1, -850.0, 516.7)),
    ("darnassus", "Darnassus", "Cities", 1457, "Kalimdor", (9532.6, 1880.0, 10238.3, 2938.4)),
    ("undercity", "Undercity", "Cities", 1458, "Azeroth", (1237.8, -86.2, 1877.9, 873.2)),
    # Battlegrounds
    ("alterac_valley", "Alterac Valley", "Battlegrounds", 1459, "PVPZone01", (-1739.6, -2456.2, 1085.4, 1781.2)),
    ("warsong_gulch", "Warsong Gulch", "Battlegrounds", 1460, "PVPZone03", (862.5, 895.8, 1627.1, 2041.7)),
    ("arathi_basin", "Arathi Basin", "Battlegrounds", 1461, "PVPZone04", (337.5, 102.1, 1508.3, 1858.3)),
    ("azshara_crater", "Azshara Crater", "Battlegrounds", None, "PVPZone02", None),
    # Eastern Kingdoms Zones
    ("alterac_mountains", "Alterac Mountains", "Zones", 1416, "Azeroth", (-366.7, -2016.7, 1500.0, 783.3)),
    ("arathi_highlands", "Arathi Highlands", "Zones", 1417, "Azeroth", (-2533.3, -4466.7, -133.3, -866.7)),
    ("badlands", "Badlands", "Zones", 1418, "Azeroth", (-7547.9, -4566.7, -5889.6, -2079.2)),
    ("blasted_lands", "Blasted Lands", "Zones", 1419, "Azeroth", (-12800.0, -4591.7, -10566.7, -1241.7)),
    ("tirisfal_glades", "Tirisfal Glades", "Zones", 1420, "Azeroth", (825.0, -1485.4, 3837.5, 3033.3)),
    ("silverpine_forest", "Silverpine Forest", "Zones", 1421, "Azeroth", (-1133.3, -750.0, 1666.7, 3450.0)),
    ("western_plaguelands", "Western Plaguelands", "Zones", 1422, "Azeroth", (500.0, -3883.3, 3366.7, 416.7)),
    ("eastern_plaguelands", "Eastern Plaguelands", "Zones", 1423, "Azeroth", (825.0, -6558.3, 3691.7, -2256.2)),
    ("hillsbrad_foothills", "Hillsbrad Foothills", "Zones", 1424, "Azeroth", (-1733.3, -2133.3, 400.0, 1066.7)),
    ("the_hinterlands", "The Hinterlands", "Zones", 1425, "Azeroth", (-1100.0, -5425.0, 1466.7, -1575.0)),
    ("dun_morogh", "Dun Morogh", "Zones", 1426, "Azeroth", (-7160.4, -3122.9, -3877.1, 1802.1)),
    ("searing_gorge", "Searing Gorge", "Zones", 1427, "Azeroth", (-7587.5, -2554.2, -6100.0, -322.9)),
    ("burning_steppes", "Burning Steppes", "Zones", 1428, "Azeroth", (-8983.3, -3195.8, -7031.2, -266.7)),
    ("elwynn_forest", "Elwynn Forest", "Zones", 1429, "Azeroth", (-10254.2, -1935.4, -7939.6, 1535.4)),
    ("deadwind_pass", "Deadwind Pass", "Zones", 1430, "Azeroth", (-11533.3, -3333.3, -9866.7, -833.3)),
    ("duskwood", "Duskwood", "Zones", 1431, "Azeroth", (-11516.7, -1866.7, -9716.7, 833.3)),
    ("loch_modan", "Loch Modan", "Zones", 1432, "Azeroth", (-6327.1, -4752.1, -4487.5, -1993.7)),
    ("redridge_mountains", "Redridge Mountains", "Zones", 1433, "Azeroth", (-10022.9, -3852.1, -8575.0, -1681.2)),
    ("stranglethorn_vale", "Stranglethorn Vale", "Zones", 1434, "Azeroth", (-15422.9, -4160.4, -11168.8, 2220.8)),
    ("swamp_of_sorrows", "Swamp of Sorrows", "Zones", 1435, "Azeroth", (-11150.0, -4516.7, -9620.8, -2222.9)),
    ("westfall", "Westfall", "Zones", 1436, "Azeroth", (-11733.3, -483.3, -9400.0, 3016.7)),
    ("wetlands", "Wetlands", "Zones", 1437, "Azeroth", (-4904.2, -4525.0, -2147.9, -389.6)),
    # Kalimdor Zones
    ("durotar", "Durotar", "Zones", 1411, "Kalimdor", (-1716.7, -7250.0, 1808.3, -1962.5)),
    ("mulgore", "Mulgore", "Zones", 1412, "Kalimdor", (-3835.4, -3675.0, 266.7, 2479.2)),
    ("the_barrens", "The Barrens", "Zones", 1413, "Kalimdor", (-5143.8, -7510.4, 1612.5, 2622.9)),
    ("teldrassil", "Teldrassil", "Zones", 1438, "Kalimdor", (8437.5, -1277.1, 11831.2, 3814.6)),
    ("darkshore", "Darkshore", "Zones", 1439, "Kalimdor", (3966.7, -3608.3, 8333.3, 2941.7)),
    ("ashenvale", "Ashenvale", "Zones", 1440, "Kalimdor", (829.2, -4066.7, 4672.9, 1700.0)),
    ("thousand_needles", "Thousand Needles", "Zones", 1441, "Kalimdor", (-6900.0, -4833.3, -3966.7, -433.3)),
    ("stonetalon_mountains", "Stonetalon Mountains", "Zones", 1442, "Kalimdor", (-339.6, -1637.5, 2916.7, 3245.8)),
    ("desolace", "Desolace", "Zones", 1443, "Kalimdor", (-2545.8, -262.5, 452.1, 4233.3)),
    ("feralas", "Feralas", "Zones", 1444, "Kalimdor", (-7000.0, -1508.3, -2366.7, 5441.7)),
    ("dustwallow_marsh", "Dustwallow Marsh", "Zones", 1445, "Kalimdor", (-5533.3, -6225.0, -2033.3, -975.0)),
    ("tanaris", "Tanaris", "Zones", 1446, "Kalimdor", (-10475.0, -7118.7, -5875.0, -218.7)),
    ("azshara", "Azshara", "Zones", 1447, "Kalimdor", (1960.4, -8347.9, 5341.7, -3277.1)),
    ("felwood", "Felwood", "Zones", 1448, "Kalimdor", (3300.0, -4108.3, 7133.3, 1641.7)),
    ("un_goro_crater", "Un'Goro Crater", "Zones", 1449, "Kalimdor", (-8433.3, -3166.7, -5966.7, 533.3)),
    ("moonglade", "Moonglade", "Zones", 1450, "Kalimdor", (6952.1, -3689.6, 8491.7, -1381.2)),
    ("silithus", "Silithus", "Zones", 1451, "Kalimdor", (-8281.2, -945.8, -5958.3, 2537.5)),
    ("winterspring", "Winterspring", "Zones", 1452, "Kalimdor", (3800.0, -7416.7, 8533.3, -316.7)),
    # Custom WoW: Forever Zones
    ("mount_hyjal", "Mount Hyjal", "Zones", 2482, "Kalimdor", (3989.6, -4395.8, 6304.2, -922.9)),
    ("zephras_isle", "Zephras Isle", "Zones", 2521, "Azeroth", (1247.9, -1331.2, 4956.2, 4231.2)),
    ("darkspear_islands", "Darkspear Islands", "Zones", 2524, "Kalimdor", (-835.4, 993.8, 447.9, 2918.8)),
    ("riverglades", "Riverglades", "Zones", 2548, "Azeroth", (-9700.0, -6741.7, -6466.7, -1891.7)),
    ("shen_dralas", "Shen'dralas", "Zones", 2652, "Kalimdor", (-3266.7, -25.0, -1900.0, 2025.0)),
    # Additional Classic / SoD / Forever Instances & Raids
    ("caverns_of_time", "Caverns of Time", "Dungeons", None, "CavernsOfTime", None),
    ("emerald_dream", "Emerald Dream", "Raids", None, "EmeraldDream", None),
    ("deeprun_tram", "Deeprun Tram", "Other", None, "DeeprunTram", None),
    ("the_searing_basin", "The Searing Basin", "Dungeons", None, "2720", None),
    ("demon_fall_canyon", "Demon Fall Canyon", "Dungeons", None, "2784", None),
    ("the_tainted_scar", "The Tainted Scar", "Raids", None, "2789", None),
    ("storm_cliffs", "Storm Cliffs", "Raids", None, "2791", None),
    ("the_crystal_vale", "The Crystal Vale", "Raids", None, "2804", None),
    ("shadow_hold", "Shadow Hold", "Dungeons", None, "2806", None),
    ("burning_of_andorhal", "Burning of Andorhal", "Zones", None, "2807", None),
    ("starfall_barrow_den", "Starfall Barrow Den", "Dungeons", None, "2817", None),
    ("nightmare_grove", "Nightmare Grove", "Raids", None, "2832", None),
    ("scarlet_enclave", "Scarlet Enclave", "Raids", None, "2856", None),
    ("epl_scarlet_raid", "Eastern Plaguelands (Scarlet Phase)", "Raids", None, "2868", None),
    ("karazhan_crypts", "Karazhan Crypts", "Dungeons", None, "2875", None),
    ("the_scarab_dais", "The Scarab Dais", "Raids", None, "2902", None),
]


# ---------------------------------------------------------------------------
# WDC5 lightweight parser for UiMap.db2 & UiMapAssignment.db2 in wowf
# ---------------------------------------------------------------------------

def _read_bits(buf: bytes, bit_off: int, bit_len: int, signed: bool = False) -> int:
    if bit_len == 0:
        return 0
    byte_off = bit_off // 8
    shift = bit_off % 8
    n_bytes = (shift + bit_len + 7) // 8
    val = int.from_bytes(buf[byte_off : byte_off + n_bytes], "little")
    val = (val >> shift) & ((1 << bit_len) - 1)
    if signed and (val & (1 << (bit_len - 1))):
        val -= 1 << bit_len
    return val


def _parse_uimap_db2(casc: LocalCASC) -> List[Tuple[int, str, int, Optional[Tuple[float, float, float, float]]]]:
    """Read UiMap.db2 + UiMapAssignment.db2 from CASC -> list of (uid, name, parent_uid, bounds_xy)."""
    d_ui = casc.read_file(_UIMAP_FDID)
    if d_ui[:4] != b"WDC5":
        return []
    (_, _, rec_size, _, _, _, _, _, _, _, _, total_fields, _, _, fsi_sz, common_sz, pallet_sz, sec_cnt) = (
        struct.unpack_from("<IIIIIIIIIHHIIIIIII", d_ui, 136)
    )
    off = 204 + sec_cnt * 40 + total_fields * 4
    fsi = [struct.unpack_from("<HHIIIII", d_ui, off + i * 24) for i in range(fsi_sz // 24)]
    _, file_off, num_recs, *_ = struct.unpack_from("<QIIIIIIII", d_ui, 204)

    uimaps: Dict[int, Tuple[str, int]] = {}
    for r in range(num_recs):
        rec_pos = file_off + r * rec_size
        rec = d_ui[rec_pos : rec_pos + rec_size]
        rel_off = struct.unpack_from("<i", rec, 0)[0]
        abs_str = rec_pos + rel_off
        end_str = d_ui.find(b"\x00", abs_str)
        name = d_ui[abs_str:end_str].decode("utf-8", "ignore") if end_str != -1 else ""
        uid = _read_bits(rec, fsi[1][0], fsi[1][1], signed=True)
        parent = _read_bits(rec, fsi[2][0], fsi[2][1], signed=True)
        if name:
            uimaps[uid] = (name, parent)

    bounds_by_ui: Dict[int, Tuple[float, float, float, float]] = {}
    try:
        d_as = casc.read_file(_UIMAP_ASSIGN_FDID)
        if d_as[:4] == b"WDC5":
            (_, _, a_rec_sz, _, _, _, _, _, _, _, _, a_tf, _, _, a_fsi_sz, _, _, a_sec) = struct.unpack_from(
                "<IIIIIIIIIHHIIIIIII", d_as, 136
            )
            a_off = 204 + a_sec * 40 + a_tf * 4
            a_fsi = [struct.unpack_from("<HHIIIII", d_as, a_off + i * 24) for i in range(a_fsi_sz // 24)]
            _, a_file_off, a_num_recs, *_ = struct.unpack_from("<QIIIIIIII", d_as, 204)
            for r in range(a_num_recs):
                rec = d_as[a_file_off + r * a_rec_sz : a_file_off + (r + 1) * a_rec_sz]
                reg = struct.unpack_from("<6f", rec, 16)
                uimap_id = _read_bits(rec, a_fsi[4][0], a_fsi[4][1], signed=True)
                if uimap_id not in bounds_by_ui:
                    bounds_by_ui[uimap_id] = (float(reg[0]), float(reg[1]), float(reg[3]), float(reg[4]))
    except Exception:
        pass

    out = []
    for uid, (name, parent) in uimaps.items():
        out.append((uid, name, parent, bounds_by_ui.get(uid)))
    return out


# ---------------------------------------------------------------------------
# Registry & On-Demand Stitcher
# ---------------------------------------------------------------------------

class WorldMapRegistry:
    """Discovers all WoW: Forever maps and loads/stitches them on-demand into DungeonData."""

    def __init__(
        self,
        index_dir: str | Path = "data/index",
        cache_dir: str | Path = "data/cache/maps",
        wow_dir: Optional[str | Path] = None,
        listfile_path: Optional[str | Path] = None,
    ):
        self.index_dir = Path(index_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.wow_dir = self._resolve_wow_dir(wow_dir)
        self.listfile_path = self._resolve_listfile(listfile_path)

        self._lock = threading.RLock()
        self._entries: Dict[str, WorldMapEntry] = {}
        self._loaded_dungeons: Dict[str, DungeonData] = {}
        self._minimap_tiles_by_folder: Optional[Dict[str, List[Tuple[int, int, int]]]] = None

        self._build_catalog()

    @staticmethod
    def _resolve_wow_dir(wow_dir: Optional[str | Path]) -> Optional[Path]:
        if wow_dir:
            p = Path(wow_dir)
            if (p / ".build.info").exists():
                return p
        for cand in _DEFAULT_WOW_DIRS:
            if (cand / ".build.info").exists():
                return cand
        return None

    @staticmethod
    def _resolve_listfile(listfile_path: Optional[str | Path]) -> Optional[Path]:
        if listfile_path:
            p = Path(listfile_path)
            if p.exists():
                return p
        for cand in _DEFAULT_LISTFILES:
            if cand.exists():
                return cand
        return None

    def _is_cached(self, slug: str) -> bool:
        return (self.index_dir / f"{slug}.npz").exists() or (self.cache_dir / f"{slug}.npz").exists()

    def _build_catalog(self) -> None:
        # 1. Pre-indexed dungeons/raids from data/index/manifest.json
        manifest_path = self.index_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for item in manifest.get("dungeons", []):
                    slug = item["slug"]
                    cat = _INDEXED_CATEGORIES.get(slug, "Dungeons")
                    self._entries[slug] = WorldMapEntry(
                        slug=slug,
                        name=item["name"],
                        category=cat,
                        source="indexed",
                        cached=True,
                        playable=True,
                        keywords=tuple(item.get("combatlog_keywords", [])),
                    )
            except Exception:
                pass

        # 2. Bundled WoW: Forever catalog (Zones, Cities, Battlegrounds, Continents, Custom Zones, Extra Instances)
        for slug, name, cat, uid, mm_folder, bounds in _BUNDLED_WORLD_MAPS:
            if slug in self._entries:
                continue
            self._entries[slug] = WorldMapEntry(
                slug=slug,
                name=name,
                category=cat,
                ui_map_id=uid,
                source="casc_zone" if bounds else "casc_minimap",
                cached=self._is_cached(slug),
                playable=True,
                minimap_folder=mm_folder,
                bounds_xy=bounds,
                keywords=(name.lower(),),
            )

        # 3. Dynamic discovery from wowf UiMap.db2 if CASC is available
        if self.wow_dir is not None:
            try:
                with LocalCASC(self.wow_dir, product="wowf") as casc:
                    for uid, name, parent, bounds in _parse_uimap_db2(casc):
                        slug = _slugify(name)
                        if slug in self._entries or (slug == "azeroth" and "azeroth_world" in self._entries):
                            continue
                        mm_folder = "Azeroth" if parent == 1415 else ("Kalimdor" if parent == 1414 else "Azeroth")
                        cat = "Continents" if parent in (0, 947) else "Zones"
                        self._entries[slug] = WorldMapEntry(
                            slug=slug,
                            name=name,
                            category=cat,
                            ui_map_id=uid,
                            source="casc_zone",
                            cached=self._is_cached(slug),
                            playable=True,
                            minimap_folder=mm_folder,
                            bounds_xy=bounds,
                            keywords=(name.lower(),),
                        )
            except Exception:
                pass

    def list_all_maps(self, category: Optional[str] = None, query: str = "") -> List[WorldMapEntry]:
        """Return all discovered WoW: Forever maps, optionally filtered by category and search query."""
        q = query.strip().lower()
        cat_filter = None if (not category or category == "All") else category
        res: List[WorldMapEntry] = []
        for e in self._entries.values():
            if cat_filter and e.category != cat_filter:
                continue
            if q and q not in e.name.lower() and q not in e.slug.lower():
                continue
            # Refresh cached flag dynamically
            if not e.cached and self._is_cached(e.slug):
                e = WorldMapEntry(**{**e.__dict__, "cached": True})
                self._entries[e.slug] = e
            res.append(e)

        cat_rank = {c: i for i, c in enumerate(MAP_CATEGORIES)}
        res.sort(key=lambda m: (cat_rank.get(m.category, 99), m.name.lower()))
        return res

    def get_entry(self, slug: str) -> Optional[WorldMapEntry]:
        return self._entries.get(slug)

    def _ensure_minimap_tile_index(self, casc: LocalCASC) -> Dict[str, List[Tuple[int, int, int]]]:
        """Load or build cached folder -> [(tx, ty, fdid)] index from listfile.csv + CASC."""
        if self._minimap_tiles_by_folder is not None:
            return self._minimap_tiles_by_folder

        idx_cache = self.cache_dir / "minimap_tiles_index.json"
        if idx_cache.exists():
            try:
                raw = json.loads(idx_cache.read_text(encoding="utf-8"))
                self._minimap_tiles_by_folder = {
                    k: [(int(a), int(b), int(c)) for a, b, c in v] for k, v in raw.items()
                }
                return self._minimap_tiles_by_folder
            except Exception:
                pass

        tiles: Dict[str, List[Tuple[int, int, int]]] = {}
        if self.listfile_path and self.listfile_path.exists():
            casc._ensure_root()
            pat = re.compile(r"^world/minimaps/([^/]+)/map(\d+)_(\d+)\.blp$", re.I)
            with open(self.listfile_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if ";" not in line:
                        continue
                    fid_s, path = line.strip().split(";", 1)
                    m = pat.match(path)
                    if not m:
                        continue
                    try:
                        fid = int(fid_s)
                    except ValueError:
                        continue
                    if casc.has_file(fid):
                        tiles.setdefault(m.group(1), []).append((int(m.group(2)), int(m.group(3)), fid))
            try:
                idx_cache.write_text(json.dumps(tiles), encoding="utf-8")
            except Exception:
                pass

        self._minimap_tiles_by_folder = tiles
        return tiles

    def load_map(self, slug: str) -> DungeonData:
        """Load a map into DungeonData (from memory, data/index, data/cache/maps, or CASC on-demand)."""
        with self._lock:
            if slug in self._loaded_dungeons:
                return self._loaded_dungeons[slug]

        # 1. Check pre-indexed data/index/<slug>.npz
        idx_npz = self.index_dir / f"{slug}.npz"
        if idx_npz.exists():
            d = load_dungeon_npz(idx_npz)
            with self._lock:
                self._loaded_dungeons[slug] = d
            return d

        # 2. Check stitched cache data/cache/maps/<slug>.npz
        cache_npz = self.cache_dir / f"{slug}.npz"
        entry = self._entries.get(slug)
        if cache_npz.exists():
            d = load_dungeon_npz(cache_npz)
            if entry is None:
                min_size = 1024
            elif entry.category == "Continents":
                min_size = 4096
            elif entry.category in ("Zones", "Cities", "Battlegrounds"):
                min_size = 2048
            else:
                min_size = 1024
            # Upgrade older low-res cached maps when local CASC is available
            if d.map_image.shape[0] >= min_size or self.wow_dir is None or entry is None:
                with self._lock:
                    self._loaded_dungeons[slug] = d
                return d

        if entry is None:
            raise KeyError(f"Unknown map slug: {slug!r}")

        # 3. Stitch on-demand from LocalCASC (if WoW is installed)
        d: Optional[DungeonData] = None
        if self.wow_dir is not None:
            try:
                d = self._stitch_from_casc(entry)
            except Exception:
                d = None

        # 4. Clean offline tactical fallback if WoW is not installed or map has no CASC tiles
        if d is None:
            d = self._build_placeholder_map(entry)

        # Save to data/cache/maps/ (PNG + NPZ) for instant future loads
        self._save_to_cache(d)
        with self._lock:
            self._loaded_dungeons[slug] = d
            self._entries[slug] = WorldMapEntry(**{**entry.__dict__, "cached": True})
        return d

    def load_map_async(
        self,
        slug: str,
        callback: Callable[[str, Optional[DungeonData], Optional[Exception]], None],
    ) -> None:
        """Load/stitch a map on a background thread and invoke callback(slug, dungeon_data, error)."""

        def _worker() -> None:
            try:
                d = self.load_map(slug)
                callback(slug, d, None)
            except Exception as exc:
                callback(slug, None, exc)

        threading.Thread(target=_worker, daemon=True, name=f"LoadMap-{slug}").start()

    def _stitch_from_casc(self, entry: WorldMapEntry) -> Optional[DungeonData]:
        if self.wow_dir is None or not entry.minimap_folder:
            return None

        if entry.category == "Continents":
            map_size = 4096
        elif entry.category in ("Zones", "Cities", "Battlegrounds"):
            map_size = 2048
        else:
            map_size = 1024
        with LocalCASC(self.wow_dir, product="wowf") as casc:
            tiles_map = self._ensure_minimap_tile_index(casc)
            folder_tiles = tiles_map.get(entry.minimap_folder)
            if not folder_tiles:
                # Case-insensitive fallback
                low_target = entry.minimap_folder.lower()
                for k, v in tiles_map.items():
                    if k.lower() == low_target:
                        folder_tiles = v
                        break
            if not folder_tiles:
                return None

            if entry.bounds_xy is not None:
                return self._stitch_bounded_region(casc, entry, folder_tiles, map_size)
            return self._stitch_full_folder(casc, entry, folder_tiles, map_size)

    def _stitch_bounded_region(
        self,
        casc: LocalCASC,
        entry: WorldMapEntry,
        folder_tiles: List[Tuple[int, int, int]],
        map_size: int,
    ) -> Optional[DungeonData]:
        assert entry.bounds_xy is not None
        xmin, ymin, xmax, ymax = entry.bounds_xy
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        span_yards = max(200.0, max(xmax - xmin, ymax - ymin))
        half = 0.5 * span_yards
        wx0, wx1 = cx - half, cx + half
        wy0, wy1 = cy - half, cy + half

        # ADT tile indices overlapping [wx0..wx1] x [wy0..wy1]
        # tx = floor(32 - wy / ADT_YARD_SIZE), ty = floor(32 - wx / ADT_YARD_SIZE)
        tx_min = max(0, int(np.floor(32.0 - wy1 / ADT_YARD_SIZE)))
        tx_max = min(63, int(np.floor(32.0 - wy0 / ADT_YARD_SIZE)))
        ty_min = max(0, int(np.floor(32.0 - wx1 / ADT_YARD_SIZE)))
        ty_max = min(63, int(np.floor(32.0 - wx0 / ADT_YARD_SIZE)))

        tile_lookup = {(tx, ty): fid for tx, ty, fid in folder_tiles}
        n_tiles = max(1, max(tx_max - tx_min + 1, ty_max - ty_min + 1))
        tile_px = 64 if n_tiles > 32 else (128 if n_tiles > 12 else 256)

        bg_color = (22, 46, 72) if entry.category in ("Continents", "Zones", "Cities") else (18, 22, 29)
        grid_w = (tx_max - tx_min + 1) * tile_px
        grid_h = (ty_max - ty_min + 1) * tile_px
        grid = np.full((grid_h, grid_w, 3), bg_color, dtype=np.uint8)
        decoded_any = False

        for ty in range(ty_min, ty_max + 1):
            for tx in range(tx_min, tx_max + 1):
                fid = tile_lookup.get((tx, ty))
                if fid is None:
                    continue
                try:
                    im = decode_blp(casc.read_file(fid)).convert("RGB")
                    if im.width != tile_px or im.height != tile_px:
                        im = im.resize((tile_px, tile_px), Image.Resampling.BILINEAR)
                    arr = np.array(im, dtype=np.uint8)
                    if float(arr.max()) < 2.0:
                        continue
                    ox = (tx - tx_min) * tile_px
                    oy = (ty - ty_min) * tile_px
                    grid[oy : oy + tile_px, ox : ox + tile_px] = arr
                    decoded_any = True
                except Exception:
                    continue

        if not decoded_any:
            return None

        # Crop exact square world window [wx0..wx1] x [wy0..wy1] out of grid
        # Top of grid (py=0) is wx_top = (32 - ty_min) * ADT_YARD_SIZE
        # Left of grid (px=0) is wy_left = (32 - tx_min) * ADT_YARD_SIZE
        px_per_yard = tile_px / ADT_YARD_SIZE
        wy_left = (32 - tx_min) * ADT_YARD_SIZE
        wx_top = (32 - ty_min) * ADT_YARD_SIZE

        gx0 = int(round((wy_left - wy1) * px_per_yard))
        gy0 = int(round((wx_top - wx1) * px_per_yard))
        g_span = max(16, int(round(span_yards * px_per_yard)))

        canvas = np.full((g_span, g_span, 3), bg_color, dtype=np.uint8)
        sx0 = max(0, gx0)
        sy0 = max(0, gy0)
        sx1 = min(grid_w, gx0 + g_span)
        sy1 = min(grid_h, gy0 + g_span)
        if sx1 > sx0 and sy1 > sy0:
            dx0 = sx0 - gx0
            dy0 = sy0 - gy0
            canvas[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = grid[sy0:sy1, sx0:sx1]

        map_img = cv2.resize(canvas, (map_size, map_size), interpolation=cv2.INTER_AREA)
        origin_xy = np.array([wx0, wy0], dtype=np.float32)
        map_scale = float(map_size) / span_yards
        return self._make_dungeon_data(entry, map_img, origin_xy, map_scale, map_rot90=1)

    def _stitch_full_folder(
        self,
        casc: LocalCASC,
        entry: WorldMapEntry,
        folder_tiles: List[Tuple[int, int, int]],
        map_size: int,
    ) -> Optional[DungeonData]:
        tile_px = 128 if len(folder_tiles) <= 100 else 64
        decoded: List[Tuple[int, int, np.ndarray]] = []
        for tx, ty, fid in folder_tiles:
            try:
                im = decode_blp(casc.read_file(fid)).convert("RGB")
                if im.width != tile_px or im.height != tile_px:
                    im = im.resize((tile_px, tile_px), Image.Resampling.BILINEAR)
                arr = np.array(im, dtype=np.uint8)
                if float(arr.max()) < 2.0:
                    continue
                decoded.append((tx, ty, arr))
            except Exception:
                continue

        if not decoded:
            return None

        min_tx = min(t[0] for t in decoded)
        max_tx = max(t[0] for t in decoded)
        min_ty = min(t[1] for t in decoded)
        max_ty = max(t[1] for t in decoded)

        bg_color = (22, 46, 72) if entry.category in ("Continents", "Zones", "Cities") else (18, 22, 29)
        n_span = max(1, max(max_tx - min_tx + 1, max_ty - min_ty + 1))
        span_px = n_span * tile_px
        sq = np.full((span_px, span_px, 3), bg_color, dtype=np.uint8)
        for tx, ty, arr in decoded:
            ox = (tx - min_tx) * tile_px
            oy = (ty - min_ty) * tile_px
            sq[oy : oy + tile_px, ox : ox + tile_px] = arr

        map_img = cv2.resize(sq, (map_size, map_size), interpolation=cv2.INTER_AREA)
        span_yards = n_span * ADT_YARD_SIZE
        # South edge = (32 - (min_ty + n_span)) * ADT_YARD_SIZE; East edge = (32 - (min_tx + n_span)) * ADT_YARD_SIZE
        wx0 = (32 - (min_ty + n_span)) * ADT_YARD_SIZE
        wy0 = (32 - (min_tx + n_span)) * ADT_YARD_SIZE
        origin_xy = np.array([wx0, wy0], dtype=np.float32)
        map_scale = float(map_size) / span_yards
        return self._make_dungeon_data(entry, map_img, origin_xy, map_scale, map_rot90=1)

    def _build_placeholder_map(self, entry: WorldMapEntry, map_size: int = 512) -> DungeonData:
        """Build a clean tactical grid fallback map when WoW CASC is unavailable."""
        img = np.full((map_size, map_size, 3), (18, 22, 29), dtype=np.uint8)
        step = map_size // 8
        for g in range(step, map_size, step):
            cv2.line(img, (g, 0), (g, map_size - 1), (32, 38, 48), 1)
            cv2.line(img, (0, g), (map_size - 1, g), (32, 38, 48), 1)
        cv2.putText(img, entry.name, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (88, 166, 255), 2, cv2.LINE_AA)
        cv2.putText(
            img,
            f"{entry.category} (Offline Tactical Grid)",
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (140, 152, 170),
            1,
            cv2.LINE_AA,
        )

        if entry.bounds_xy is not None:
            xmin, ymin, xmax, ymax = entry.bounds_xy
            cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
            span = max(200.0, max(xmax - xmin, ymax - ymin))
            origin_xy = np.array([cx - 0.5 * span, cy - 0.5 * span], dtype=np.float32)
            map_scale = float(map_size) / span
        else:
            origin_xy = np.array([-500.0, -500.0], dtype=np.float32)
            map_scale = float(map_size) / 1000.0

        return self._make_dungeon_data(entry, img, origin_xy, map_scale, map_rot90=1)

    @staticmethod
    def _make_dungeon_data(
        entry: WorldMapEntry,
        map_img: np.ndarray,
        origin_xy: np.ndarray,
        map_scale: float,
        map_rot90: int = 1,
    ) -> DungeonData:
        h = map_img.shape[0]
        cx_px = h * 0.5
        wx = float(origin_xy[0]) + (cx_px / map_scale)
        wy = float(origin_xy[1]) + (cx_px / map_scale)
        waypoints = np.array([[wx, wy, 0.0]], dtype=np.float32)
        return DungeonData(
            slug=entry.slug,
            name=entry.name,
            combatlog_keywords=list(entry.keywords) or [entry.name.lower()],
            waypoints=waypoints,
            adjacency=[[]],
            map_image=map_img,
            map_origin_xy=origin_xy.astype(np.float32),
            map_scale=float(map_scale),
            embeddings_f32=np.zeros((0, 1), dtype=np.float32),
            view_wp_idx=np.zeros((0,), dtype=np.int32),
            wp_to_views=[np.zeros((0,), dtype=np.int32)],
            map_rot90=map_rot90,
        )

    def _save_to_cache(self, d: DungeonData) -> None:
        try:
            png_path = self.cache_dir / f"{d.slug}.png"
            npz_path = self.cache_dir / f"{d.slug}.npz"
            Image.fromarray(d.map_image, mode="RGB").save(png_path)
            np.savez_compressed(
                npz_path,
                slug=d.slug,
                name=d.name,
                combatlog_keywords=np.array(d.combatlog_keywords, dtype=str),
                waypoints=d.waypoints,
                map_image=d.map_image,
                map_origin_xy=d.map_origin_xy,
                map_scale=np.float32(d.map_scale),
                map_rot90=np.int32(d.map_rot90),
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module-level default registry convenience helpers
# ---------------------------------------------------------------------------

_DEFAULT_REGISTRY: Optional[WorldMapRegistry] = None
_DEFAULT_LOCK = threading.Lock()


def get_world_map_registry(
    index_dir: str | Path = "data/index",
    cache_dir: str | Path = "data/cache/maps",
    wow_dir: Optional[str | Path] = None,
) -> WorldMapRegistry:
    global _DEFAULT_REGISTRY
    with _DEFAULT_LOCK:
        if _DEFAULT_REGISTRY is None:
            _DEFAULT_REGISTRY = WorldMapRegistry(index_dir=index_dir, cache_dir=cache_dir, wow_dir=wow_dir)
        return _DEFAULT_REGISTRY


def list_all_maps(
    category: Optional[str] = None,
    query: str = "",
    wow_dir: Optional[str | Path] = None,
) -> List[WorldMapEntry]:
    return get_world_map_registry(wow_dir=wow_dir).list_all_maps(category=category, query=query)


def load_map(slug: str, wow_dir: Optional[str | Path] = None) -> DungeonData:
    return get_world_map_registry(wow_dir=wow_dir).load_map(slug)


def load_map_async(
    slug: str,
    callback: Callable[[str, Optional[DungeonData], Optional[Exception]], None],
    wow_dir: Optional[str | Path] = None,
) -> None:
    get_world_map_registry(wow_dir=wow_dir).load_map_async(slug, callback)
