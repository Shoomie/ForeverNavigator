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
import sqlite3
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .blp_decoder import decode_blp
from .casc_reader import LocalCASC
from .tracker import DungeonData, load_dungeon_npz
from .wmo_parser import iter_chunks

ADT_YARD_SIZE = 533.3333333
MAX_IN_MEMORY_MAP_PX = 8192
OVERVIEW_MAP_PX = 4096
MINIMAP_INDEX_VERSION = 2

MAP_CATEGORIES: Tuple[str, ...] = (
    "Raids",
    "Dungeons",
    "Battlegrounds",
    "Cities",
    "Zones",
    "Continents",
    "Caves & Buildings",
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
    source: str = "catalog"  # "indexed" | "casc_zone" | "casc_minimap" | "casc_wmo" | "casc_noliquid" | "catalog"
    cached: bool = False
    playable: bool = True
    minimap_folder: Optional[str] = None
    bounds_xy: Optional[Tuple[float, float, float, float]] = None  # (xmin, ymin, xmax, ymax)
    keywords: Tuple[str, ...] = ()
    wmo_model: Optional[str] = None
    is_shared: bool = False
    shared_note: str = ""


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
# Full-Resolution On-Demand Tile Provider (for huge maps > MAX_IN_MEMORY_MAP_PX)
# ---------------------------------------------------------------------------

class FullResTileSource:
    """Lazy, LRU-cached full-resolution BLP tile renderer for large continent/world maps."""

    def __init__(
        self,
        wow_dir: Path,
        tile_lookup: Dict[Tuple[int, int], int],
        native_tile_px: int,
        overview_px: int,
        wx0: float,
        wy0: float,
        span_yards: float,
        bg_color: Tuple[int, int, int] = (22, 46, 72),
        max_cached_tiles: int = 128,
    ):
        self.wow_dir = Path(wow_dir)
        self.tile_lookup = tile_lookup
        self.native_tile_px = int(native_tile_px)
        self.overview_px = int(overview_px)
        self.wx0 = float(wx0)
        self.wy0 = float(wy0)
        self.span_yards = float(span_yards)
        self.bg_color = bg_color
        self.max_cached_tiles = max_cached_tiles
        self.native_scale = float(self.native_tile_px) / ADT_YARD_SIZE
        self.native_span_px = self.span_yards * self.native_scale
        self._lock = threading.Lock()
        self._tile_cache: OrderedDict[Tuple[int, int], Optional[Image.Image]] = OrderedDict()
        self._casc: Optional[LocalCASC] = None

    def set_cache_limit(self, max_cached_tiles: int) -> None:
        """Adjust in-memory tile LRU budget (smaller on SSD, larger on HDD)."""
        with self._lock:
            self.max_cached_tiles = max(8, int(max_cached_tiles))
            while len(self._tile_cache) > self.max_cached_tiles:
                self._tile_cache.popitem(last=False)

    def close(self) -> None:
        with self._lock:
            if self._casc is not None:
                try:
                    self._casc.close()
                except Exception:
                    pass

                self._casc = None
            self._tile_cache.clear()

    def _get_tile(self, tx: int, ty: int) -> Optional[Image.Image]:
        key = (tx, ty)
        with self._lock:
            if key in self._tile_cache:
                self._tile_cache.move_to_end(key)
                return self._tile_cache[key]

        fid = self.tile_lookup.get(key)
        if fid is None:
            return None

        tile_im: Optional[Image.Image] = None
        with self._lock:
            if self._casc is None:
                self._casc = LocalCASC(self.wow_dir, product="wowf")
            try:
                im = decode_blp(self._casc.read_file(fid)).convert("RGB")
                if im.width != self.native_tile_px or im.height != self.native_tile_px:
                    im = im.resize((self.native_tile_px, self.native_tile_px), Image.Resampling.BILINEAR)
                arr = np.asarray(im, dtype=np.uint8)
                if float(arr.max()) >= 2.0:
                    tile_im = im
            except Exception:
                tile_im = None

            self._tile_cache[key] = tile_im
            self._tile_cache.move_to_end(key)
            while len(self._tile_cache) > self.max_cached_tiles:
                self._tile_cache.popitem(last=False)
        return tile_im

    def render_viewport(
        self,
        pan_x_ov: float,
        pan_y_ov: float,
        sc_ov: float,
        cw: int,
        ch: int,
        max_tiles_per_frame: int = 20,
    ) -> Optional[Image.Image]:
        """Render a (cw, ch) viewport centered at overview pixel (pan_x_ov, pan_y_ov) at scale sc_ov.

        Returns None if the visible region spans more than `max_tiles_per_frame` tiles so zoomed-out
        views never thrash SSD/CASC I/O.
        """
        ov_to_nat = self.native_span_px / max(1.0, float(self.overview_px))
        sc_nat = sc_ov / max(1e-6, ov_to_nat)
        pan_x_nat = pan_x_ov * ov_to_nat
        pan_y_nat = pan_y_ov * ov_to_nat

        # Visible native pixel rect [nx0..nx1, ny0..ny1]
        half_w_nat = (cw * 0.5) / max(1e-6, sc_nat)
        half_h_nat = (ch * 0.5) / max(1e-6, sc_nat)
        nx0 = pan_x_nat - half_w_nat
        nx1 = pan_x_nat + half_w_nat
        ny0 = pan_y_nat - half_h_nat
        ny1 = pan_y_nat + half_h_nat

        h_nat = self.native_span_px
        wy_hi = self.wy0 + ((h_nat - 1.0) - nx0) / self.native_scale
        wy_lo = self.wy0 + ((h_nat - 1.0) - nx1) / self.native_scale
        wx_hi = self.wx0 + ((h_nat - 1.0) - ny0) / self.native_scale
        wx_lo = self.wx0 + ((h_nat - 1.0) - ny1) / self.native_scale

        tx_min = max(0, int(np.floor(32.0 - wy_hi / ADT_YARD_SIZE)))
        tx_max = min(63, int(np.floor(32.0 - wy_lo / ADT_YARD_SIZE)))
        ty_min = max(0, int(np.floor(32.0 - wx_hi / ADT_YARD_SIZE)))
        ty_max = min(63, int(np.floor(32.0 - wx_lo / ADT_YARD_SIZE)))

        # Hard I/O cap: count how many actual tiles overlap the viewport before decoding anything
        vis_tiles = [
            (tx, ty)
            for ty in range(ty_min, ty_max + 1)
            for tx in range(tx_min, tx_max + 1)
            if (tx, ty) in self.tile_lookup
        ]
        if len(vis_tiles) > max_tiles_per_frame:
            return None

        out = Image.new("RGB", (cw, ch), self.bg_color)
        for tx, ty in vis_tiles:
                wy_left = (32 - tx) * ADT_YARD_SIZE
                wx_top = (32 - ty) * ADT_YARD_SIZE
                tile_nx0 = (h_nat - 1.0) - (wy_left - self.wy0) * self.native_scale
                tile_ny0 = (h_nat - 1.0) - (wx_top - self.wx0) * self.native_scale

                dx0 = int(round((tile_nx0 - pan_x_nat) * sc_nat + cw * 0.5))
                dy0 = int(round((tile_ny0 - pan_y_nat) * sc_nat + ch * 0.5))
                dx1 = int(round((tile_nx0 + self.native_tile_px - pan_x_nat) * sc_nat + cw * 0.5))
                dy1 = int(round((tile_ny0 + self.native_tile_px - pan_y_nat) * sc_nat + ch * 0.5))
                tw, th = dx1 - dx0, dy1 - dy0
                # Cull off-screen tiles BEFORE decoding BLP from CASC
                if tw < 1 or th < 1 or dx1 <= 0 or dy1 <= 0 or dx0 >= cw or dy0 >= ch:
                    continue
                tile_im = self._get_tile(tx, ty)
                if tile_im is None:
                    continue
                # Viewport-capped crop+resize: never allocate or render pixels outside (0..cw, 0..ch)
                vx0 = max(0, dx0)
                vy0 = max(0, dy0)
                vx1 = min(cw, dx1)
                vy1 = min(ch, dy1)
                vw, vh = vx1 - vx0, vy1 - vy0
                if vw < 1 or vh < 1:
                    continue
                resample = Image.Resampling.BILINEAR if tw >= 64 else Image.Resampling.NEAREST
                if vx0 == dx0 and vy0 == dy0 and vx1 == dx1 and vy1 == dy1:
                    resized = tile_im.resize((vw, vh), resample)
                else:
                    sx0 = (vx0 - dx0) * (tile_im.width / float(tw))
                    sy0 = (vy0 - dy0) * (tile_im.height / float(th))
                    sx1 = (vx1 - dx0) * (tile_im.width / float(tw))
                    sy1 = (vy1 - dy0) * (tile_im.height / float(th))
                    resized = tile_im.crop((sx0, sy0, sx1, sy1)).resize((vw, vh), resample)
                out.paste(resized, (vx0, vy0))
        return out


# ---------------------------------------------------------------------------
# Categorization & Shared Cave/Building Helpers
# ---------------------------------------------------------------------------

_SHARED_WMO_KEYWORDS = (
    "cave", "mine", "barracks", "inn", "house", "tower", "bunker", "tent",
    "barn", "chapel", "crypt", "keep", "castle", "fort", "buildings", "ruins",
    "farm", "mill", "smith", "lodge", "hall", "ship", "zeppelin", "tram",
    "passivedoodads",
)


def _infer_folder_category(folder: str) -> str:
    low = folder.lower()
    if any(k in low for k in ("raid", "ahnqiraj", "blackwing", "moltencore", "naxx", "onyx", "zulgurub", "blacktemple", "sunwell", "ulduar", "icecrown", "firelands", "karaz")):
        return "Raids"
    if any(k in low for k in ("pvp", "arena", "bg", "battleground", "alterac", "warsong", "arathi")):
        return "Battlegrounds"
    if any(k in low for k in ("instance", "dungeon", "monastery", "deadmines", "scholomance", "stratholme", "shadowfang", "stockade", "wailing", "uldaman", "razorfen", "zul", "gnomeregan", "maraudon", "diremaul", "blackrock", "rampart", "karazahn", "barrow")):
        return "Dungeons"
    if low in ("azeroth", "kalimdor", "northrend", "expansion01", "outland", "pandaria", "draenor"):
        return "Continents"
    return "Other"


def _is_shared_wmo_model(model_key: str) -> bool:
    low = model_key.lower()
    if low.startswith("dungeon/"):
        return False
    return any(kw in low for kw in _SHARED_WMO_KEYWORDS)


def _format_wmo_name(model_key: str) -> str:
    parts = [p for p in model_key.replace("\\", "/").split("/") if p]
    if len(parts) >= 2:
        label = f"{parts[-2]} / {parts[-1]}"
    else:
        label = parts[0] if parts else model_key
    return label.replace("_", " ").title()


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
        self._storage_mode: str = "ssd"
        self._entries: Dict[str, WorldMapEntry] = {}
        self._loaded_dungeons: Dict[str, DungeonData] = {}
        self._fullres_sources: Dict[str, FullResTileSource] = {}
        self._minimap_tiles_by_folder: Optional[Dict[str, List[Tuple[int, int, int]]]] = None
        self._noliquid_tiles_by_folder: Optional[Dict[str, List[Tuple[int, int, int]]]] = None
        self._wmo_tiles_by_model: Optional[Dict[str, List[Tuple[int, int, int, int]]]] = None
        self._wmo_root_fdids: Optional[Dict[str, int]] = None
        self._idx_db_path: Optional[Path] = None
        self._idx_meta_counts: Optional[Dict[str, int]] = None
        self._idx_folder_names: List[str] = []
        self._idx_noliquid_names: List[str] = []
        self._idx_wmo_names: List[str] = []

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
        stems = getattr(self, "_startup_cached_stems", None)
        if stems is not None:
            return slug in stems
        return (self.index_dir / f"{slug}.npz").exists() or (self.cache_dir / f"{slug}.npz").exists()

    def _build_catalog(self) -> None:
        cached_stems: set[str] = set()
        for d_dir in (self.index_dir, self.cache_dir):
            if d_dir.exists():
                for p in d_dir.glob("*.npz"):
                    cached_stems.add(p.stem)
        self._startup_cached_stems = cached_stems
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

        # 3. Prefer cached v2 minimap index (zero CASC SSD reads on startup); only scan CASC on first run
        has_cached_idx = self._load_cached_minimap_index()
        if not has_cached_idx and self.wow_dir is not None:
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
                    self._ensure_minimap_tile_index(casc)
            except Exception:
                pass

        # 4. Register every remaining terrain/instance minimap folder (all 715+ folders)
        known_folders_lower = {
            e.minimap_folder.lower()
            for e in self._entries.values()
            if e.minimap_folder and e.bounds_xy is None
        }
        folder_names = sorted(self._minimap_tiles_by_folder.keys()) if self._minimap_tiles_by_folder else self._idx_folder_names
        if folder_names:
            for folder in folder_names:
                if folder.lower() in known_folders_lower:
                    continue
                slug = _slugify(folder)
                if slug in self._entries:
                    slug = f"minimap_{slug}"
                if slug in self._entries:
                    continue
                cat = _infer_folder_category(folder)
                self._entries[slug] = WorldMapEntry(
                    slug=slug,
                    name=folder.replace("_", " "),
                    category=cat,
                    source="casc_minimap",
                    cached=self._is_cached(slug),
                    playable=True,
                    minimap_folder=folder,
                    bounds_xy=None,
                    keywords=(folder.lower(),),
                )

        # 5. Register every noLiquid minimap folder variant
        noliq_names = sorted(self._noliquid_tiles_by_folder.keys()) if self._noliquid_tiles_by_folder else self._idx_noliquid_names
        if noliq_names:
            for folder in noliq_names:
                slug = f"{_slugify(folder)}_noliquid"
                if slug in self._entries:
                    continue
                self._entries[slug] = WorldMapEntry(
                    slug=slug,
                    name=f"{folder.replace('_', ' ')} (No Liquid)",
                    category=_infer_folder_category(folder),
                    source="casc_noliquid",
                    cached=self._is_cached(slug),
                    playable=True,
                    minimap_folder=folder,
                    bounds_xy=None,
                    keywords=(folder.lower(), "noliquid"),
                )

        # 6. Register every WMO / Cave / Building / Doodad minimap model (deduplicated canonical entries)
        wmo_names = sorted(self._wmo_tiles_by_model.keys()) if self._wmo_tiles_by_model else self._idx_wmo_names
        if wmo_names:
            for model_key in wmo_names:
                slug = f"wmo_{_slugify(model_key)}"
                if slug in self._entries:
                    continue
                shared = _is_shared_wmo_model(model_key)
                self._entries[slug] = WorldMapEntry(
                    slug=slug,
                    name=_format_wmo_name(model_key),
                    category="Caves & Buildings",
                    source="casc_wmo",
                    cached=self._is_cached(slug),
                    playable=True,
                    minimap_folder=None,
                    bounds_xy=None,
                    keywords=(model_key.lower(),),
                    wmo_model=model_key,
                    is_shared=shared,
                    shared_note="Shared cave/building minimap (used at multiple world locations)" if shared else "",
                )
        self._startup_cached_stems = None

    def list_all_maps(self, category: Optional[str] = None, query: str = "") -> List[WorldMapEntry]:
        """Return all discovered WoW: Forever maps, optionally filtered by category and search query."""
        q = query.strip().lower()
        cat_filter = None if (not category or category == "All") else category
        cached_stems: set[str] = set()
        for d_dir in (self.index_dir, self.cache_dir):
            if d_dir.exists():
                for p in d_dir.glob("*.npz"):
                    cached_stems.add(p.stem)

        res: List[WorldMapEntry] = []
        with self._lock:
            for slug, e in list(self._entries.items()):
                if cat_filter and e.category != cat_filter:
                    continue
                if q and q not in e.name.lower() and q not in e.slug.lower():
                    continue
                if not e.cached and slug in cached_stems:
                    e = WorldMapEntry(**{**e.__dict__, "cached": True})
                    self._entries[slug] = e
                res.append(e)

        cat_rank = {c: i for i, c in enumerate(MAP_CATEGORIES)}
        res.sort(key=lambda m: (cat_rank.get(m.category, 99), m.name.lower()))
        return res

    def get_entry(self, slug: str) -> Optional[WorldMapEntry]:
        return self._entries.get(slug)

    def _write_sqlite_minimap_index(
        self,
        db_path: Path,
        tiles: Dict[str, List[Tuple[int, int, int]]],
        noliquid: Dict[str, List[Tuple[int, int, int]]],
        wmos: Dict[str, List[Tuple[int, int, int, int]]],
        wmo_roots: Dict[str, int],
    ) -> None:
        tmp = db_path.with_suffix(".db.tmp")
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
        conn = sqlite3.connect(str(tmp))
        try:
            conn.executescript(
                """
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                CREATE TABLE meta (k TEXT PRIMARY KEY, v INTEGER NOT NULL);
                CREATE TABLE catalog_keys (kind TEXT NOT NULL, name TEXT NOT NULL);
                CREATE TABLE folder_tiles (kind TEXT NOT NULL, folder_lower TEXT NOT NULL, tx INTEGER NOT NULL, ty INTEGER NOT NULL, fid INTEGER NOT NULL);
                CREATE TABLE wmo_tiles (model TEXT NOT NULL, grp INTEGER NOT NULL, tx INTEGER NOT NULL, ty INTEGER NOT NULL, fid INTEGER NOT NULL);
                CREATE TABLE wmo_roots (model TEXT PRIMARY KEY, fid INTEGER NOT NULL);
                """
            )
            n_map = sum(len(v) for v in tiles.values())
            n_noliq = sum(len(v) for v in noliquid.values())
            n_wmo = sum(len(v) for v in wmos.values())
            conn.executemany(
                "INSERT INTO meta VALUES (?, ?)",
                [
                    ("version", MINIMAP_INDEX_VERSION),
                    ("map_folders", len(tiles)),
                    ("map_tiles", n_map),
                    ("noliquid_folders", len(noliquid)),
                    ("noliquid_tiles", n_noliq),
                    ("wmo_models", len(wmos)),
                    ("wmo_tiles", n_wmo),
                ],
            )
            cat_rows = (
                [("folder", k) for k in sorted(tiles.keys())]
                + [("noliquid", k) for k in sorted(noliquid.keys())]
                + [("wmo", k) for k in sorted(wmos.keys())]
            )
            conn.executemany("INSERT INTO catalog_keys VALUES (?, ?)", cat_rows)
            ft_rows = [
                ("folder", k.lower(), int(a), int(b), int(c))
                for k, v in tiles.items()
                for a, b, c in v
            ] + [
                ("noliquid", k.lower(), int(a), int(b), int(c))
                for k, v in noliquid.items()
                for a, b, c in v
            ]
            conn.executemany("INSERT INTO folder_tiles VALUES (?, ?, ?, ?, ?)", ft_rows)
            wt_rows = [
                (k, int(g), int(x), int(y), int(fid))
                for k, v in wmos.items()
                for g, x, y, fid in v
            ]
            conn.executemany("INSERT INTO wmo_tiles VALUES (?, ?, ?, ?, ?)", wt_rows)
            conn.executemany(
                "INSERT INTO wmo_roots VALUES (?, ?)",
                [(k, int(fid)) for k, fid in wmo_roots.items()],
            )
            conn.executescript(
                """
                CREATE INDEX idx_ft_kf ON folder_tiles(kind, folder_lower);
                CREATE INDEX idx_wt_m ON wmo_tiles(model);
                """
            )
            conn.commit()
        finally:
            conn.close()
        tmp.replace(db_path)

    def _load_sqlite_minimap_index(self, db_path: Path) -> bool:
        if not db_path.exists():
            return False
        try:
            conn = sqlite3.connect(str(db_path))
            try:
                meta = dict(conn.execute("SELECT k, v FROM meta").fetchall())
                if meta.get("version") != MINIMAP_INDEX_VERSION:
                    return False
                f_names: List[str] = []
                n_names: List[str] = []
                w_names: List[str] = []
                for kind, name in conn.execute("SELECT kind, name FROM catalog_keys"):
                    if kind == "folder":
                        f_names.append(name)
                    elif kind == "noliquid":
                        n_names.append(name)
                    elif kind == "wmo":
                        w_names.append(name)
            finally:
                conn.close()
            self._idx_db_path = db_path
            self._idx_meta_counts = meta
            self._idx_folder_names = f_names
            self._idx_noliquid_names = n_names
            self._idx_wmo_names = w_names
            return True
        except Exception:
            return False

    def _load_cached_minimap_index(self) -> bool:
        """Load fast SQLite minimap index (~5 ms, <1 MB RAM), migrating from JSON once if needed."""
        db_cache = self.cache_dir / "minimap_tiles_index.db"
        idx_cache = self.cache_dir / "minimap_tiles_index.json"
        if db_cache.exists() and (not idx_cache.exists() or db_cache.stat().st_mtime >= idx_cache.stat().st_mtime):
            if self._load_sqlite_minimap_index(db_cache):
                return True
        if not idx_cache.exists():
            return False
        try:
            raw = json.loads(idx_cache.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and raw.get("version") == MINIMAP_INDEX_VERSION:
                tiles = {
                    k: [(int(a), int(b), int(c)) for a, b, c in v]
                    for k, v in raw.get("folders", {}).items()
                }
                noliquid = {
                    k: [(int(a), int(b), int(c)) for a, b, c in v]
                    for k, v in raw.get("noliquid", {}).items()
                }
                wmos = {
                    k: [(int(g), int(x), int(y), int(fid)) for g, x, y, fid in v]
                    for k, v in raw.get("wmos", {}).items()
                }
                wmo_roots = {
                    k: int(fid) for k, fid in raw.get("wmo_roots", {}).items()
                }
                try:
                    self._write_sqlite_minimap_index(db_cache, tiles, noliquid, wmos, wmo_roots)
                    if self._load_sqlite_minimap_index(db_cache):
                        return True
                except Exception:
                    pass
                self._minimap_tiles_by_folder = tiles
                self._noliquid_tiles_by_folder = noliquid
                self._wmo_tiles_by_model = wmos
                self._wmo_root_fdids = wmo_roots
                return True
        except Exception:
            pass

        return False

    def _ensure_minimap_tile_index(self, casc: LocalCASC) -> Dict[str, List[Tuple[int, int, int]]]:
        """Load or build full v2 index of ALL minimap BLP files (mapXX_YY, noLiquid, numeric folders, WMOs) from listfile.csv + CASC."""
        if self._idx_db_path is not None:
            return self._minimap_tiles_by_folder or {}
        if self._minimap_tiles_by_folder is not None and self._wmo_tiles_by_model is not None:
            return self._minimap_tiles_by_folder

        if self._load_cached_minimap_index():
            return self._minimap_tiles_by_folder or {}

        tiles: Dict[str, List[Tuple[int, int, int]]] = {}
        noliquid: Dict[str, List[Tuple[int, int, int]]] = {}
        wmos: Dict[str, List[Tuple[int, int, int, int]]] = {}
        wmo_roots: Dict[str, int] = {}

        if self.listfile_path and self.listfile_path.exists():
            casc._ensure_root()
            pat_map = re.compile(r"^world/minimaps/([^/]+)/map(\d+)_(\d+)\.blp$", re.I)
            pat_folder_tile = re.compile(r"^world/minimaps/([^/]+)/([^/]+?)_(\d+)_(\d+)\.blp$", re.I)
            pat_noliquid = re.compile(r"^world/minimaps/([^/]+)/noliquid_map(\d+)_(\d+)\.blp$", re.I)
            pat_wmo = re.compile(r"^world/minimaps/(?:wmo/)?(.+)_(\d{3})_(\d{2})_(\d{2})(?:_\d+)?\.blp$", re.I)
            pat_wmo_root = re.compile(r"^world/wmo/(.+)\.wmo$", re.I)
            pat_wmo_grp_suffix = re.compile(r"_\d{3}(?:_lod\d+)?\.wmo$", re.I)

            with open(self.listfile_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if ";" not in line:
                        continue
                    fid_s, path = line.strip().split(";", 1)
                    low = path.lower()
                    if low.startswith("world/minimaps/") and low.endswith(".blp"):
                        try:
                            fid = int(fid_s)
                        except ValueError:
                            continue
                        if not casc.has_file(fid):
                            continue
                        m = pat_map.match(path)
                        if m:
                            tiles.setdefault(m.group(1), []).append((int(m.group(2)), int(m.group(3)), fid))
                            continue
                        m = pat_noliquid.match(path)
                        if m:
                            noliquid.setdefault(m.group(1), []).append((int(m.group(2)), int(m.group(3)), fid))
                            continue
                        m = pat_folder_tile.match(path)
                        if m and m.group(2).lower() == m.group(1).lower():
                            tiles.setdefault(m.group(1), []).append((int(m.group(3)), int(m.group(4)), fid))
                            continue
                        m = pat_wmo.match(path)
                        if m:
                            wmos.setdefault(m.group(1).lower(), []).append(
                                (int(m.group(2)), int(m.group(3)), int(m.group(4)), fid)
                            )
                    elif low.startswith("world/wmo/") and low.endswith(".wmo") and not pat_wmo_grp_suffix.search(low):
                        try:
                            fid = int(fid_s)
                        except ValueError:
                            continue
                        if casc.has_file(fid):
                            m = pat_wmo_root.match(path)
                            if m:
                                wmo_roots[m.group(1).lower()] = fid

            try:
                idx_cache = self.cache_dir / "minimap_tiles_index.json"
                idx_cache.write_text(
                    json.dumps({
                        "version": MINIMAP_INDEX_VERSION,
                        "folders": tiles,
                        "noliquid": noliquid,
                        "wmos": wmos,
                        "wmo_roots": wmo_roots,
                    }),
                    encoding="utf-8",
                )
            except Exception:
                pass


        self._minimap_tiles_by_folder = tiles
        self._noliquid_tiles_by_folder = noliquid
        self._wmo_tiles_by_model = wmos
        self._wmo_root_fdids = wmo_roots
        return tiles

    def get_minimap_coverage_stats(self) -> Dict[str, int]:
        """Return counts of indexed minimap files across all categories."""
        if self._idx_meta_counts is not None:
            m = self._idx_meta_counts
            n_map = int(m.get("map_tiles", 0))
            n_noliq = int(m.get("noliquid_tiles", 0))
            n_wmo = int(m.get("wmo_tiles", 0))
            return {
                "map_folders": int(m.get("map_folders", 0)),
                "map_tiles": n_map,
                "noliquid_folders": int(m.get("noliquid_folders", 0)),
                "noliquid_tiles": n_noliq,
                "wmo_models": int(m.get("wmo_models", 0)),
                "wmo_tiles": n_wmo,
                "total_minimap_files": n_map + n_noliq + n_wmo,
                "catalog_entries": len(self._entries),
            }
        folders = self._minimap_tiles_by_folder or {}
        noliq = self._noliquid_tiles_by_folder or {}
        wmos = self._wmo_tiles_by_model or {}
        n_map = sum(len(v) for v in folders.values())
        n_noliq = sum(len(v) for v in noliq.values())
        n_wmo = sum(len(v) for v in wmos.values())
        return {
            "map_folders": len(folders),
            "map_tiles": n_map,
            "noliquid_folders": len(noliq),
            "noliquid_tiles": n_noliq,
            "wmo_models": len(wmos),
            "wmo_tiles": n_wmo,
            "total_minimap_files": n_map + n_noliq + n_wmo,
            "catalog_entries": len(self._entries),
        }

    def has_full_res_tiles(self, slug: str) -> bool:
        return slug in self._fullres_sources

    def ensure_full_res_tiles(self, slug: str) -> bool:
        """Lazily attach a FullResTileSource for a large continent/map only when deep-zoomed."""
        if slug in self._fullres_sources:
            return True
        if self.wow_dir is None:
            return False
        entry = self._entries.get(slug)
        if entry is None:
            return False
        d = self._loaded_dungeons.get(slug)
        if d is None:
            try:
                d = self.load_map(slug, preview_only=True, max_preview_px=512)
            except Exception:
                return False
        self._attach_fullres_source_for_entry(entry, d)
        return slug in self._fullres_sources

    def load_continent_l1_array(self, cont_slug: str, max_px: int = 4096) -> Optional[Tuple[np.ndarray, int]]:
        """Load or build a fast prestitched Level-1 continent JPEG (~1.3 MB, WoW: Forever masked).

        Returns (rgb_uint8_array, native_map_px) so mid-range zoom (3x..15x) is crisp with zero CASC I/O.
        """
        folder = "Kalimdor" if cont_slug == "kalimdor" else ("Azeroth" if cont_slug == "eastern_kingdoms" else None)
        if folder is None:
            return None
        l1_jpg = self.cache_dir / f"{cont_slug}_l1_{max_px}.jpg"
        cache_npz = self.cache_dir / f"{cont_slug}.npz"
        if l1_jpg.exists() and (not cache_npz.exists() or l1_jpg.stat().st_mtime >= cache_npz.stat().st_mtime):
            try:
                with Image.open(l1_jpg) as raw_im:
                    arr = np.asarray(raw_im.convert("RGB"), dtype=np.uint8).copy()
                return arr, int(max(arr.shape[0], OVERVIEW_MAP_PX))
            except Exception:
                pass
        try:
            cont_d = self.load_map(cont_slug, preview_only=False)
        except Exception:
            return None
        arr = cont_d.map_image.copy()
        if arr.shape[0] > max_px:
            arr = cv2.resize(arr, (max_px, max_px), interpolation=cv2.INTER_AREA)
        ph, pw = arr.shape[:2]
        playable = _build_forever_playable_mask(folder, cont_d, ph, pw, pad_yards=360.0)
        ocean = np.array([18, 36, 58], dtype=np.uint8)
        diff_ocean = np.abs(arr.astype(np.int16) - np.array([22, 46, 72], dtype=np.int16)).max(axis=-1)
        diff_dark = np.abs(arr.astype(np.int16) - np.array([18, 22, 29], dtype=np.int16)).max(axis=-1)
        keep = playable & (diff_ocean > 12) & (diff_dark > 12) & (arr.max(axis=-1) > 14)
        arr[~keep] = ocean
        try:
            Image.fromarray(arr, mode="RGB").save(l1_jpg, format="JPEG", quality=90, optimize=True)
        except Exception:
            pass
        return arr, int(cont_d.map_size_px)

    def set_storage_mode(self, mode: str) -> None:
        """Configure tile cache budgets for 'ssd' (low RAM, dynamic streaming) vs 'hdd' (larger cache)."""
        m = "hdd" if str(mode).lower() == "hdd" else "ssd"
        tile_limit = 96 if m == "hdd" else 32
        with self._lock:
            self._storage_mode = m
            for src in self._fullres_sources.values():
                src.set_cache_limit(tile_limit)

    def _tile_cache_limit(self) -> int:
        return 96 if self._storage_mode == "hdd" else 32

    def close(self) -> None:
        """Close all open FullResTileSource CASC archive handles."""
        with self._lock:
            for src in self._fullres_sources.values():
                try:
                    src.close()
                except Exception:
                    pass

            self._fullres_sources.clear()

    def render_full_res_viewport(
        self,
        slug: str,
        pan_x: float,
        pan_y: float,
        sc: float,
        cw: int,
        ch: int,
    ) -> Optional[Image.Image]:
        """Render a viewport using native full-resolution CASC tiles when a map exceeds MAX_IN_MEMORY_MAP_PX."""
        src = self._fullres_sources.get(slug)
        if src is None:
            return None
        return src.render_viewport(pan_x, pan_y, sc, cw, ch)

    def load_map(self, slug: str, preview_only: bool = False, max_preview_px: int = 512) -> DungeonData:
        """Load a map into DungeonData (from memory, data/index, data/cache/maps, or CASC on-demand).

        When `preview_only=True`, large cached maps are kept in RAM as lightweight `max_preview_px`
        previews (preserving `native_map_px` and `map_full_path` for on-demand full-res streaming).
        """
        with self._lock:
            cached_d = self._loaded_dungeons.get(slug)
            if cached_d is not None:
                if preview_only or cached_d.map_image.shape[0] == cached_d.map_size_px:
                    return cached_d

        # 1. Check pre-indexed data/index/<slug>.npz
        idx_npz = self.index_dir / f"{slug}.npz"
        if idx_npz.exists():
            d = load_dungeon_npz(idx_npz, max_preview_px=max_preview_px if preview_only else None)
            with self._lock:
                self._loaded_dungeons[slug] = d
            return d

        # 2. Check stitched cache data/cache/maps/<slug>.npz (with fast tiny preview sidecar for preview_only)
        cache_npz = self.cache_dir / f"{slug}.npz"
        entry = self._entries.get(slug)
        if cache_npz.exists():
            if preview_only:
                sidecar = self.cache_dir / f"{slug}_prev{max_preview_px}.npz"
                if sidecar.exists() and sidecar.stat().st_mtime >= cache_npz.stat().st_mtime:
                    try:
                        d_sc = load_dungeon_npz(sidecar)
                        png_sib = cache_npz.with_suffix(".png")
                        d_sc.map_full_path = png_sib if png_sib.exists() else cache_npz
                        if d_sc.bounds_xy is None and entry is not None and entry.bounds_xy is not None:
                            d_sc.bounds_xy = entry.bounds_xy
                        with self._lock:
                            self._loaded_dungeons[slug] = d_sc
                        return d_sc
                    except Exception:
                        pass

            is_full_res_cache = False
            native_px_cache = 0
            try:
                with np.load(cache_npz, allow_pickle=False) as z_meta:
                    is_full_res_cache = bool(z_meta["full_res"]) if "full_res" in z_meta else False
                    native_px_cache = int(z_meta["native_px"]) if "native_px" in z_meta else 0
            except Exception:
                pass

            d = load_dungeon_npz(cache_npz, max_preview_px=max_preview_px if preview_only else None)
            if d.bounds_xy is None and entry is not None and entry.bounds_xy is not None:
                d.bounds_xy = entry.bounds_xy
            # Upgrade older low-res cached maps when local CASC is available
            if is_full_res_cache or self.wow_dir is None or entry is None:
                if preview_only and d.map_image.shape[0] < d.map_size_px:
                    self._save_preview_sidecar(d, self.cache_dir / f"{slug}_prev{max_preview_px}.npz")
                if not preview_only and entry is not None and native_px_cache > MAX_IN_MEMORY_MAP_PX and self.wow_dir is not None:
                    self._attach_fullres_source_for_entry(entry, d)
                with self._lock:
                    self._loaded_dungeons[slug] = d
                return d

        if entry is None:
            raise KeyError(f"Unknown map slug: {slug!r}")

        # 3. Stitch on-demand from LocalCASC (if WoW is installed)
        d: Optional[DungeonData] = None
        is_full_res = True
        native_px = 0
        if self.wow_dir is not None:
            try:
                d, native_px = self._stitch_from_casc(entry)
            except Exception:
                d = None

        # 4. Clean offline tactical fallback if WoW is not installed or map has no CASC tiles
        if d is None:
            d = self._build_placeholder_map(entry)
            is_full_res = False
            native_px = d.map_image.shape[0]

        # Save to data/cache/maps/ (PNG + NPZ) for instant future loads
        self._save_to_cache(d, full_res=is_full_res, native_px=native_px)
        png_path = self.cache_dir / f"{d.slug}.png"
        d.native_map_px = int(d.map_image.shape[0])
        d.map_full_path = png_path if png_path.exists() else (self.cache_dir / f"{d.slug}.npz")
        if preview_only and d.map_image.shape[0] > max_preview_px:
            d.map_image = cv2.resize(d.map_image, (max_preview_px, max_preview_px), interpolation=cv2.INTER_AREA)
        with self._lock:
            self._loaded_dungeons[slug] = d
            self._entries[slug] = WorldMapEntry(**{**entry.__dict__, "cached": True})
        return d

    def load_map_async(
        self,
        slug: str,
        callback: Callable[[str, Optional[DungeonData], Optional[Exception]], None],
        preview_only: bool = False,
    ) -> None:
        """Load/stitch a map on a background thread and invoke callback(slug, dungeon_data, error)."""

        def _worker() -> None:
            try:
                d = self.load_map(slug, preview_only=preview_only)
                callback(slug, d, None)
            except Exception as exc:
                callback(slug, None, exc)

        threading.Thread(target=_worker, daemon=True, name=f"LoadMap-{slug}").start()

    def _lookup_folder_tiles(
        self,
        tiles_map: Optional[Dict[str, List[Tuple[int, int, int]]]],
        folder: str,
        kind: str = "folder",
    ) -> Optional[List[Tuple[int, int, int]]]:
        if tiles_map:
            folder_tiles = tiles_map.get(folder)
            if not folder_tiles:
                low_target = folder.lower()
                for k, v in tiles_map.items():
                    if k.lower() == low_target:
                        folder_tiles = v
                        break
            if folder_tiles:
                return folder_tiles
        if self._idx_db_path is not None and self._idx_db_path.exists():
            try:
                conn = sqlite3.connect(str(self._idx_db_path))
                try:
                    rows = conn.execute(
                        "SELECT tx, ty, fid FROM folder_tiles WHERE kind = ? AND folder_lower = ?",
                        (kind, folder.lower()),
                    ).fetchall()
                finally:
                    conn.close()
                if rows:
                    return [(int(a), int(b), int(c)) for a, b, c in rows]
            except Exception:
                pass
        return None

    def _lookup_wmo_tiles(self, model_key: str) -> Optional[List[Tuple[int, int, int, int]]]:
        if self._wmo_tiles_by_model:
            res = self._wmo_tiles_by_model.get(model_key)
            if res:
                return res
        if self._idx_db_path is not None and self._idx_db_path.exists():
            try:
                conn = sqlite3.connect(str(self._idx_db_path))
                try:
                    rows = conn.execute(
                        "SELECT grp, tx, ty, fid FROM wmo_tiles WHERE model = ?",
                        (model_key,),
                    ).fetchall()
                finally:
                    conn.close()
                if rows:
                    return [(int(g), int(x), int(y), int(fid)) for g, x, y, fid in rows]
            except Exception:
                pass
        return None

    def _lookup_wmo_root(self, model_key: str) -> Optional[int]:
        if self._wmo_root_fdids:
            fid = self._wmo_root_fdids.get(model_key)
            if fid is not None:
                return fid
        if self._idx_db_path is not None and self._idx_db_path.exists():
            try:
                conn = sqlite3.connect(str(self._idx_db_path))
                try:
                    row = conn.execute("SELECT fid FROM wmo_roots WHERE model = ?", (model_key,)).fetchone()
                finally:
                    conn.close()
                if row:
                    return int(row[0])
            except Exception:
                pass
        return None

    def _attach_fullres_source_for_entry(self, entry: WorldMapEntry, d: DungeonData) -> None:
        if self.wow_dir is None or not entry.minimap_folder:
            return
        try:
            with LocalCASC(self.wow_dir, product="wowf") as casc:
                tiles_map = self._ensure_minimap_tile_index(casc)
                kind = "noliquid" if entry.source == "casc_noliquid" else "folder"
                src_map = self._noliquid_tiles_by_folder if kind == "noliquid" else tiles_map
                folder_tiles = self._lookup_folder_tiles(src_map, entry.minimap_folder, kind=kind)
                if not folder_tiles:
                    return
                tile_lookup = {(tx, ty): fid for tx, ty, fid in folder_tiles}
                # Probe one non-empty tile for native_tile_px
                native_tile_px = 512
                for _, _, fid in folder_tiles[:8]:
                    try:
                        im = decode_blp(casc.read_file(fid))
                        native_tile_px = max(im.width, im.height)
                        break
                    except Exception:
                        continue
                ov_px = d.map_size_px
                span_yards = float(ov_px) / max(1e-6, d.map_scale)
                bg_color = (22, 46, 72) if entry.category in ("Continents", "Zones", "Cities") else (18, 22, 29)
                self._fullres_sources[entry.slug] = FullResTileSource(
                    wow_dir=self.wow_dir,
                    tile_lookup=tile_lookup,
                    native_tile_px=native_tile_px,
                    overview_px=ov_px,
                    wx0=float(d.map_origin_xy[0]),
                    wy0=float(d.map_origin_xy[1]),
                    span_yards=span_yards,
                    bg_color=bg_color,
                    max_cached_tiles=self._tile_cache_limit(),
                )
        except Exception:
            pass


    def _stitch_from_casc(self, entry: WorldMapEntry) -> Tuple[Optional[DungeonData], int]:
        if self.wow_dir is None:
            return None, 0

        with LocalCASC(self.wow_dir, product="wowf") as casc:
            tiles_map = self._ensure_minimap_tile_index(casc)

            if entry.source == "casc_wmo" and entry.wmo_model:
                wmo_tiles = self._lookup_wmo_tiles(entry.wmo_model)
                if not wmo_tiles:
                    return None, 0
                return self._stitch_wmo_model(casc, entry, wmo_tiles)

            if not entry.minimap_folder:
                return None, 0

            if entry.source == "casc_noliquid":
                folder_tiles = self._lookup_folder_tiles(self._noliquid_tiles_by_folder, entry.minimap_folder, kind="noliquid")
                fallback_tiles: Optional[List[Tuple[int, int, int]]] = None
            else:
                folder_tiles = self._lookup_folder_tiles(tiles_map, entry.minimap_folder, kind="folder")
                fallback_tiles = self._lookup_folder_tiles(self._noliquid_tiles_by_folder, entry.minimap_folder, kind="noliquid")

            if not folder_tiles:
                return None, 0

            if entry.bounds_xy is not None:
                return self._stitch_bounded_region(casc, entry, folder_tiles, fallback_tiles=fallback_tiles)
            return self._stitch_full_folder(casc, entry, folder_tiles, fallback_tiles=fallback_tiles)

    def _stitch_bounded_region(
        self,
        casc: LocalCASC,
        entry: WorldMapEntry,
        folder_tiles: List[Tuple[int, int, int]],
        fallback_tiles: Optional[List[Tuple[int, int, int]]] = None,
    ) -> Tuple[Optional[DungeonData], int]:
        assert entry.bounds_xy is not None
        xmin, ymin, xmax, ymax = entry.bounds_xy
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        span_yards = max(200.0, max(xmax - xmin, ymax - ymin))
        half = 0.5 * span_yards
        wx0, wx1 = cx - half, cx + half
        wy0, wy1 = cy - half, cy + half

        # ADT tile indices overlapping [wx0..wx1] x [wy0..wy1]
        tx_min = max(0, int(np.floor(32.0 - wy1 / ADT_YARD_SIZE)))
        tx_max = min(63, int(np.floor(32.0 - wy0 / ADT_YARD_SIZE)))
        ty_min = max(0, int(np.floor(32.0 - wx1 / ADT_YARD_SIZE)))
        ty_max = min(63, int(np.floor(32.0 - wx0 / ADT_YARD_SIZE)))

        tile_lookup = {(tx, ty): fid for tx, ty, fid in folder_tiles}
        if fallback_tiles:
            for tx, ty, fid in fallback_tiles:
                tile_lookup.setdefault((tx, ty), fid)

        # Decode overlapping tiles at FULL NATIVE resolution (zero downsampling)
        decoded_tiles: Dict[Tuple[int, int], np.ndarray] = {}
        native_tile_px = 0
        for ty in range(ty_min, ty_max + 1):
            for tx in range(tx_min, tx_max + 1):
                fid = tile_lookup.get((tx, ty))
                if fid is None:
                    continue
                try:
                    im = decode_blp(casc.read_file(fid)).convert("RGB")
                    arr = np.array(im, dtype=np.uint8)
                    if float(arr.max()) < 2.0:
                        continue
                    native_tile_px = max(native_tile_px, arr.shape[0], arr.shape[1])
                    decoded_tiles[(tx, ty)] = arr
                except Exception:
                    continue

        if not decoded_tiles or native_tile_px <= 0:
            return None, 0

        bg_color = (22, 46, 72) if entry.category in ("Continents", "Zones", "Cities") else (18, 22, 29)
        native_px_per_yard = float(native_tile_px) / ADT_YARD_SIZE
        native_span_px = max(16, int(round(span_yards * native_px_per_yard)))
        origin_xy = np.array([wx0, wy0], dtype=np.float32)

        if native_span_px <= MAX_IN_MEMORY_MAP_PX:
            # 100% native full resolution in memory (no downsampling!)
            grid_w = (tx_max - tx_min + 1) * native_tile_px
            grid_h = (ty_max - ty_min + 1) * native_tile_px
            grid = np.full((grid_h, grid_w, 3), bg_color, dtype=np.uint8)
            for (tx, ty), arr in decoded_tiles.items():
                if arr.shape[0] != native_tile_px or arr.shape[1] != native_tile_px:
                    arr = cv2.resize(arr, (native_tile_px, native_tile_px), interpolation=cv2.INTER_LINEAR)
                ox = (tx - tx_min) * native_tile_px
                oy = (ty - ty_min) * native_tile_px
                grid[oy : oy + native_tile_px, ox : ox + native_tile_px] = arr

            wy_left = (32 - tx_min) * ADT_YARD_SIZE
            wx_top = (32 - ty_min) * ADT_YARD_SIZE
            gx0 = int(round((wy_left - wy1) * native_px_per_yard))
            gy0 = int(round((wx_top - wx1) * native_px_per_yard))

            canvas = np.full((native_span_px, native_span_px, 3), bg_color, dtype=np.uint8)
            sx0 = max(0, gx0)
            sy0 = max(0, gy0)
            sx1 = min(grid_w, gx0 + native_span_px)
            sy1 = min(grid_h, gy0 + native_span_px)
            if sx1 > sx0 and sy1 > sy0:
                dx0 = sx0 - gx0
                dy0 = sy0 - gy0
                canvas[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = grid[sy0:sy1, sx0:sx1]

            map_scale = float(native_span_px) / span_yards
            return self._make_dungeon_data(entry, canvas, origin_xy, map_scale, map_rot90=1), native_span_px

        # Huge continent/world map (> MAX_IN_MEMORY_MAP_PX):
        # Build a crisp OVERVIEW_MAP_PX overview for MinimapTracker + attach FullResTileSource for native 512px zooming
        ov_px = OVERVIEW_MAP_PX
        ov_tile_px = max(16, int(round(ov_px / max(1.0, span_yards / ADT_YARD_SIZE))))
        grid_w = (tx_max - tx_min + 1) * ov_tile_px
        grid_h = (ty_max - ty_min + 1) * ov_tile_px
        grid = np.full((grid_h, grid_w, 3), bg_color, dtype=np.uint8)
        for (tx, ty), arr in decoded_tiles.items():
            ov_arr = cv2.resize(arr, (ov_tile_px, ov_tile_px), interpolation=cv2.INTER_AREA)
            ox = (tx - tx_min) * ov_tile_px
            oy = (ty - ty_min) * ov_tile_px
            grid[oy : oy + ov_tile_px, ox : ox + ov_tile_px] = ov_arr

        ov_px_per_yard = float(ov_tile_px) / ADT_YARD_SIZE
        wy_left = (32 - tx_min) * ADT_YARD_SIZE
        wx_top = (32 - ty_min) * ADT_YARD_SIZE
        gx0 = int(round((wy_left - wy1) * ov_px_per_yard))
        gy0 = int(round((wx_top - wx1) * ov_px_per_yard))
        g_span = max(16, int(round(span_yards * ov_px_per_yard)))
        canvas = np.full((g_span, g_span, 3), bg_color, dtype=np.uint8)
        sx0, sy0 = max(0, gx0), max(0, gy0)
        sx1, sy1 = min(grid_w, gx0 + g_span), min(grid_h, gy0 + g_span)
        if sx1 > sx0 and sy1 > sy0:
            canvas[sy0 - gy0 : sy0 - gy0 + (sy1 - sy0), sx0 - gx0 : sx0 - gx0 + (sx1 - sx0)] = grid[sy0:sy1, sx0:sx1]

        map_img = cv2.resize(canvas, (ov_px, ov_px), interpolation=cv2.INTER_AREA)
        map_scale = float(ov_px) / span_yards
        self._fullres_sources[entry.slug] = FullResTileSource(
            wow_dir=self.wow_dir,
            tile_lookup=tile_lookup,
            native_tile_px=native_tile_px,
            overview_px=ov_px,
            wx0=wx0,
            wy0=wy0,
            span_yards=span_yards,
            bg_color=bg_color,
            max_cached_tiles=self._tile_cache_limit(),
        )
        return self._make_dungeon_data(entry, map_img, origin_xy, map_scale, map_rot90=1), native_span_px

    def _stitch_full_folder(
        self,
        casc: LocalCASC,
        entry: WorldMapEntry,
        folder_tiles: List[Tuple[int, int, int]],
        fallback_tiles: Optional[List[Tuple[int, int, int]]] = None,
    ) -> Tuple[Optional[DungeonData], int]:
        tile_lookup = {(tx, ty): fid for tx, ty, fid in folder_tiles}
        if fallback_tiles:
            for tx, ty, fid in fallback_tiles:
                tile_lookup.setdefault((tx, ty), fid)

        decoded: List[Tuple[int, int, np.ndarray]] = []
        native_tile_px = 0
        for (tx, ty), fid in tile_lookup.items():
            try:
                im = decode_blp(casc.read_file(fid)).convert("RGB")
                arr = np.array(im, dtype=np.uint8)
                if float(arr.max()) < 2.0:
                    continue
                native_tile_px = max(native_tile_px, arr.shape[0], arr.shape[1])
                decoded.append((tx, ty, arr))
            except Exception:
                continue

        if not decoded or native_tile_px <= 0:
            return None, 0

        min_tx = min(t[0] for t in decoded)
        max_tx = max(t[0] for t in decoded)
        min_ty = min(t[1] for t in decoded)
        max_ty = max(t[1] for t in decoded)

        bg_color = (22, 46, 72) if entry.category in ("Continents", "Zones", "Cities") else (18, 22, 29)
        n_span = max(1, max(max_tx - min_tx + 1, max_ty - min_ty + 1))
        native_span_px = n_span * native_tile_px
        span_yards = n_span * ADT_YARD_SIZE
        wx0 = (32 - (min_ty + n_span)) * ADT_YARD_SIZE
        wy0 = (32 - (min_tx + n_span)) * ADT_YARD_SIZE
        origin_xy = np.array([wx0, wy0], dtype=np.float32)

        if native_span_px <= MAX_IN_MEMORY_MAP_PX:
            # 100% native full resolution (zero downsampling!)
            sq = np.full((native_span_px, native_span_px, 3), bg_color, dtype=np.uint8)
            for tx, ty, arr in decoded:
                if arr.shape[0] != native_tile_px or arr.shape[1] != native_tile_px:
                    arr = cv2.resize(arr, (native_tile_px, native_tile_px), interpolation=cv2.INTER_LINEAR)
                ox = (tx - min_tx) * native_tile_px
                oy = (ty - min_ty) * native_tile_px
                sq[oy : oy + native_tile_px, ox : ox + native_tile_px] = arr
            map_scale = float(native_span_px) / span_yards
            return self._make_dungeon_data(entry, sq, origin_xy, map_scale, map_rot90=1), native_span_px

        # Huge folder (> MAX_IN_MEMORY_MAP_PX): overview + FullResTileSource
        ov_px = OVERVIEW_MAP_PX
        ov_tile_px = max(16, ov_px // n_span)
        sq = np.full((n_span * ov_tile_px, n_span * ov_tile_px, 3), bg_color, dtype=np.uint8)
        for tx, ty, arr in decoded:
            ov_arr = cv2.resize(arr, (ov_tile_px, ov_tile_px), interpolation=cv2.INTER_AREA)
            ox = (tx - min_tx) * ov_tile_px
            oy = (ty - min_ty) * ov_tile_px
            sq[oy : oy + ov_tile_px, ox : ox + ov_tile_px] = ov_arr
        map_img = cv2.resize(sq, (ov_px, ov_px), interpolation=cv2.INTER_AREA)
        map_scale = float(ov_px) / span_yards
        self._fullres_sources[entry.slug] = FullResTileSource(
            wow_dir=self.wow_dir,
            tile_lookup=tile_lookup,
            native_tile_px=native_tile_px,
            overview_px=ov_px,
            wx0=wx0,
            wy0=wy0,
            span_yards=span_yards,
            bg_color=bg_color,
            max_cached_tiles=self._tile_cache_limit(),
        )
        return self._make_dungeon_data(entry, map_img, origin_xy, map_scale, map_rot90=1), native_span_px

    def _stitch_wmo_model(
        self,
        casc: LocalCASC,
        entry: WorldMapEntry,
        wmo_tiles: List[Tuple[int, int, int, int]],
    ) -> Tuple[Optional[DungeonData], int]:
        """Stitch all groups & tiles of a WMO/cave/building minimap at 100% native full resolution."""
        groups: Dict[int, List[Tuple[int, int, np.ndarray]]] = {}
        for grp_idx, tx, ty, fid in wmo_tiles:
            try:
                im = decode_blp(casc.read_file(fid)).convert("RGB")
                arr = np.array(im, dtype=np.uint8)
                if float(arr.max()) < 2.0:
                    continue
                groups.setdefault(grp_idx, []).append((tx, ty, arr))
            except Exception:
                continue

        if not groups:
            return None, 0

        # Stitch each group's (tx, ty) tiles at native pixel resolution
        group_imgs: Dict[int, np.ndarray] = {}
        for grp_idx, tiles in groups.items():
            max_tx = max(t[0] for t in tiles)
            max_ty = max(t[1] for t in tiles)
            # In WMO minimaps, tile (0, 0) can be non-256 (e.g. 32x64, 128x64) if single-tile,
            # or 256x256 per tile if multi-tile. Compute exact column widths & row heights.
            col_w: Dict[int, int] = {}
            row_h: Dict[int, int] = {}
            for tx, ty, arr in tiles:
                col_w[tx] = max(col_w.get(tx, 0), arr.shape[1])
                row_h[ty] = max(row_h.get(ty, 0), arr.shape[0])
            x_off = {}
            cur_x = 0
            for tx in range(max_tx + 1):
                x_off[tx] = cur_x
                cur_x += col_w.get(tx, 256)
            y_off = {}
            cur_y = 0
            for ty in range(max_ty + 1):
                y_off[ty] = cur_y
                cur_y += row_h.get(ty, 256)
            g_arr = np.full((max(1, cur_y), max(1, cur_x), 3), (18, 22, 29), dtype=np.uint8)
            for tx, ty, arr in tiles:
                ox, oy = x_off[tx], y_off[ty]
                ah, aw = arr.shape[:2]
                g_arr[oy : oy + ah, ox : ox + aw] = arr
            group_imgs[grp_idx] = g_arr

        # Try reading MOGI group bounding boxes from the matching .wmo root in CASC
        mogi_boxes: Dict[int, Tuple[float, float, float, float]] = {}
        wmo_root_fid = self._lookup_wmo_root(entry.wmo_model or "")
        if wmo_root_fid is not None:
            try:
                root_bytes = casc.read_file(wmo_root_fid)
                for cid, payload in iter_chunks(root_bytes):
                    if cid == "MOGI":
                        cnt = len(payload) // 32
                        for gi in range(cnt):
                            _fl, x0, y0, _z0, x1, y1, _z1, _no = struct.unpack_from("<I6fi", payload, gi * 32)
                            mogi_boxes[gi] = (x0, y0, x1, y1)
                        break
            except Exception:
                mogi_boxes.clear()

        px_per_yard = 2.0  # Standard WoW WMO minimap resolution: 2 px/yard
        if mogi_boxes and any(g in mogi_boxes for g in group_imgs):
            valid_g = [g for g in group_imgs if g in mogi_boxes]
            min_x = min(mogi_boxes[g][0] for g in valid_g)
            min_y = min(mogi_boxes[g][1] for g in valid_g)
            max_x = max(mogi_boxes[g][2] for g in valid_g)
            max_y = max(mogi_boxes[g][3] for g in valid_g)
            span_x = max(16.0, max_x - min_x)
            span_y = max(16.0, max_y - min_y)
            span_yd = max(span_x, span_y)
            raw_px = max(64, int(np.ceil(span_yd * px_per_yard)))
            # Ensure canvas fits every group image at native resolution
            for g in valid_g:
                gx0, gy0, gx1, gy1 = mogi_boxes[g]
                ox = int(round((gx0 - min_x) * px_per_yard))
                oy = int(round((max_y - gy1) * px_per_yard))
                gh, gw = group_imgs[g].shape[:2]
                raw_px = max(raw_px, ox + gw, oy + gh)
            canvas = np.full((raw_px, raw_px, 3), (18, 22, 29), dtype=np.uint8)
            for g in valid_g:
                gx0, gy0, gx1, gy1 = mogi_boxes[g]
                ox = max(0, int(round((gx0 - min_x) * px_per_yard)))
                oy = max(0, int(round((max_y - gy1) * px_per_yard)))
                g_arr = group_imgs[g]
                gh, gw = g_arr.shape[:2]
                # Blend non-black pixels so overlapping group padding doesn't erase adjacent rooms
                dst = canvas[oy : oy + gh, ox : ox + gw]
                mask = g_arr.max(axis=-1) > 4
                dst[mask] = g_arr[mask]
            origin_xy = np.array([min_x, min_y], dtype=np.float32)
            native_px = raw_px
        else:
            # Pack groups cleanly in a square grid at 100% native pixel resolution
            ordered = [group_imgs[g] for g in sorted(group_imgs.keys())]
            cols = max(1, int(np.ceil(np.sqrt(len(ordered)))))
            max_gw = max(im.shape[1] for im in ordered)
            max_gh = max(im.shape[0] for im in ordered)
            pad = 4
            cell_w = max_gw + pad
            cell_h = max_gh + pad
            rows = int(np.ceil(len(ordered) / cols))
            raw_px = max(64, cols * cell_w, rows * cell_h)
            canvas = np.full((raw_px, raw_px, 3), (18, 22, 29), dtype=np.uint8)
            for idx, g_arr in enumerate(ordered):
                r, c = divmod(idx, cols)
                oy = r * cell_h + pad // 2
                ox = c * cell_w + pad // 2
                gh, gw = g_arr.shape[:2]
                canvas[oy : oy + gh, ox : ox + gw] = g_arr
            origin_xy = np.array([0.0, 0.0], dtype=np.float32)
            native_px = raw_px

        if native_px > MAX_IN_MEMORY_MAP_PX:
            scale_f = float(MAX_IN_MEMORY_MAP_PX) / float(native_px)
            canvas = cv2.resize(canvas, (MAX_IN_MEMORY_MAP_PX, MAX_IN_MEMORY_MAP_PX), interpolation=cv2.INTER_AREA)
            px_per_yard *= scale_f
        return self._make_dungeon_data(entry, canvas, origin_xy, px_per_yard, map_rot90=0), native_px

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
            bounds_xy=entry.bounds_xy,
        )

    @staticmethod
    def _save_preview_sidecar(d: DungeonData, sidecar_path: Path) -> None:
        """Save a tiny ~120 KB preview .npz so subsequent startups skip reading 40 MB zone files."""
        try:
            extra = {}
            if d.bounds_xy is not None:
                extra["bounds_xy"] = np.array(d.bounds_xy, dtype=np.float32)
            np.savez_compressed(
                sidecar_path,
                slug=d.slug,
                name=d.name,
                combatlog_keywords=np.array(d.combatlog_keywords, dtype=str),
                waypoints=d.waypoints,
                map_image=d.map_image,
                map_origin_xy=d.map_origin_xy,
                map_scale=np.float32(d.map_scale),
                map_rot90=np.int32(d.map_rot90),
                native_map_px=np.int32(d.map_size_px),
                **extra,
            )
        except Exception:
            pass

    def _save_to_cache(self, d: DungeonData, full_res: bool = True, native_px: int = 0) -> None:
        try:
            png_path = self.cache_dir / f"{d.slug}.png"
            npz_path = self.cache_dir / f"{d.slug}.npz"
            Image.fromarray(d.map_image, mode="RGB").save(png_path)
            extra = {}
            if d.bounds_xy is not None:
                extra["bounds_xy"] = np.array(d.bounds_xy, dtype=np.float32)
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
                full_res=np.bool_(full_res),
                native_px=np.int32(native_px if native_px > 0 else d.map_image.shape[0]),
                **extra,
            )
        except Exception:
            pass


@dataclass
class _ContinentPlacement:
    map_origin_xy: Tuple[float, float]
    map_scale: float
    map_size_px: int
    map_rot90: int
    preview_px: int
    src_box: Tuple[int, int, int, int]   # (sx0, sy0, sx1, sy1) in preview_px coords
    dst_box: Tuple[int, int, int, int]   # (dx0, dy0, dx1, dy1) in overview coords


@dataclass
class AzerothOverview:
    """Lightweight composite world map of Kalimdor (west) + Eastern Kingdoms (east)."""
    image: Image.Image
    placements: Dict[str, _ContinentPlacement]  # "Kalimdor" / "Azeroth"

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height

    def world_to_overview_px(self, folder: Optional[str], wx: float, wy: float) -> Tuple[float, float]:
        """Project a world (wx, wy) coordinate on Kalimdor or Azeroth (EK) onto overview pixels."""
        p = self.placements.get(folder or "")
        if p is None:
            return self.width * 0.5, self.height * 0.5
        h = float(p.map_size_px)
        px_raw = (float(wx) - p.map_origin_xy[0]) * p.map_scale
        py_raw = (h - 1.0) - (float(wy) - p.map_origin_xy[1]) * p.map_scale
        if p.map_rot90 == 1:
            mx, my = py_raw, (h - 1.0) - px_raw
        elif p.map_rot90 == 3:
            mx, my = (h - 1.0) - py_raw, px_raw
        else:
            mx, my = px_raw, py_raw
        # Convert native map pixel -> preview pixel -> overview pixel
        scale_prev = float(p.preview_px) / max(1.0, h)
        px_p = mx * scale_prev
        py_p = my * scale_prev
        sx0, sy0, sx1, sy1 = p.src_box
        dx0, dy0, dx1, dy1 = p.dst_box
        u = (px_p - sx0) / max(1.0, float(sx1 - sx0))
        v = (py_p - sy0) / max(1.0, float(sy1 - sy0))
        ox = float(np.clip(dx0 + u * (dx1 - dx0), 0.0, self.width - 1.0))
        oy = float(np.clip(dy0 + v * (dy1 - dy0), 0.0, self.height - 1.0))
        return ox, oy

    def overview_px_to_continent_px(
        self, ox: float, oy: float
    ) -> Optional[Tuple[str, str, float, float, float]]:
        """Map an overview pixel (ox, oy) to (folder, cont_slug, cont_px_x, cont_px_y, cont_px_per_ov_px)."""
        # Pick the continent whose placement box contains (ox, oy), or nearest half if zoomed in near coast
        chosen_folder: Optional[str] = None
        for folder, p in self.placements.items():
            dx0, dy0, dx1, dy1 = p.dst_box
            if dx0 <= ox <= dx1 and dy0 <= oy <= dy1:
                chosen_folder = folder
                break
        if chosen_folder is None:
            chosen_folder = "Kalimdor" if ox < self.width * 0.5 else "Azeroth"
            if chosen_folder not in self.placements:
                return None
        p = self.placements[chosen_folder]
        cont_slug = "kalimdor" if chosen_folder == "Kalimdor" else "eastern_kingdoms"
        sx0, sy0, sx1, sy1 = p.src_box
        dx0, dy0, dx1, dy1 = p.dst_box
        u = (ox - dx0) / max(1.0, float(dx1 - dx0))
        v = (oy - dy0) / max(1.0, float(dy1 - dy0))
        px_p = sx0 + u * (sx1 - sx0)
        py_p = sy0 + v * (sy1 - sy0)
        nat_per_prev = float(p.map_size_px) / max(1.0, float(p.preview_px))
        cont_x = px_p * nat_per_prev
        cont_y = py_p * nat_per_prev
        cont_px_per_ov_px = ((sx1 - sx0) / max(1.0, float(dx1 - dx0))) * nat_per_prev
        return chosen_folder, cont_slug, cont_x, cont_y, cont_px_per_ov_px

    def close(self) -> None:
        try:
            self.image.close()
        except Exception:
            pass


def _build_forever_playable_mask(
    folder: str,
    cont_d: DungeonData,
    ph: int,
    pw: int,
    pad_yards: float = 280.0,
) -> np.ndarray:
    """Build a boolean mask of playable WoW: Forever zones on a continent preview.

    Excludes unimplemented Retail/TBC/Midnight tiles (e.g. Silvermoon, Eversong, Quel'Danas,
    Zul'Aman, GM Island) that live in the shared modern WoW CASC archive.
    """
    mask = np.zeros((ph, pw), dtype=bool)
    scale_prev = float(ph) / max(1.0, float(cont_d.map_size_px))
    matched_any = False
    for _slug, _name, cat, _uid, mm_folder, bounds in _BUNDLED_WORLD_MAPS:
        if mm_folder != folder or bounds is None or cat == "Continents":
            continue
        xmin, ymin, xmax, ymax = bounds
        p0 = cont_d.world_to_map_px(np.array([xmin - pad_yards, ymin - pad_yards]))
        p1 = cont_d.world_to_map_px(np.array([xmax + pad_yards, ymax + pad_yards]))
        x0 = max(0, int(np.floor(min(p0[0], p1[0]) * scale_prev)))
        x1 = min(pw, int(np.ceil(max(p0[0], p1[0]) * scale_prev)) + 1)
        y0 = max(0, int(np.floor(min(p0[1], p1[1]) * scale_prev)))
        y1 = min(ph, int(np.ceil(max(p0[1], p1[1]) * scale_prev)) + 1)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
            matched_any = True
    if not matched_any:
        return np.ones((ph, pw), dtype=bool)
    return mask


_AZEROTH_OVERVIEW_CACHE_VERSION = 1


def build_azeroth_overview(
    registry: Optional[WorldMapRegistry],
    width: int = 3840,
    height: int = 2160,
    use_disk_cache: bool = True,
) -> AzerothOverview:
    """Load or build a precached, prestitched 4K Kalimdor + Eastern Kingdoms world overview.

    When cached on disk (`azeroth_overview_4k.jpg` + `.json`), loads in ~40 ms with zero CASC I/O.
    """
    cache_img_path: Optional[Path] = None
    cache_meta_path: Optional[Path] = None
    if use_disk_cache and registry is not None:
        cache_img_path = registry.cache_dir / f"azeroth_overview_{width}x{height}.jpg"
        cache_meta_path = registry.cache_dir / f"azeroth_overview_{width}x{height}.json"
        if cache_img_path.exists() and cache_meta_path.exists():
            try:
                meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
                if (
                    meta.get("version") == _AZEROTH_OVERVIEW_CACHE_VERSION
                    and int(meta.get("width", 0)) == width
                    and int(meta.get("height", 0)) == height
                ):
                    with Image.open(cache_img_path) as raw_im:
                        im = raw_im.convert("RGB")
                    placements: Dict[str, _ContinentPlacement] = {}
                    for folder, pd in meta.get("placements", {}).items():
                        placements[folder] = _ContinentPlacement(
                            map_origin_xy=(float(pd["map_origin_xy"][0]), float(pd["map_origin_xy"][1])),
                            map_scale=float(pd["map_scale"]),
                            map_size_px=int(pd["map_size_px"]),
                            map_rot90=int(pd["map_rot90"]),
                            preview_px=int(pd["preview_px"]),
                            src_box=tuple(int(v) for v in pd["src_box"]),
                            dst_box=tuple(int(v) for v in pd["dst_box"]),
                        )
                    return AzerothOverview(image=im, placements=placements)
            except Exception:
                pass

    ocean = (18, 36, 58)
    canvas = np.full((height, width, 3), ocean, dtype=np.uint8)

    # Subtle nautical latitude/longitude grid
    grid_x = max(160, width // 10)
    grid_y = max(150, height // 6)
    for gx in range(grid_x, width, grid_x):
        cv2.line(canvas, (gx, 0), (gx, height - 1), (24, 45, 70), 1)
    for gy in range(grid_y, height, grid_y):
        cv2.line(canvas, (0, gy), (width - 1, gy), (24, 45, 70), 1)

    placements = {}
    specs = [
        ("Kalimdor", "kalimdor", (int(width * 0.03), int(height * 0.07), int(width * 0.47), int(height * 0.96))),
        ("Azeroth", "eastern_kingdoms", (int(width * 0.53), int(height * 0.07), int(width * 0.97), int(height * 0.96))),
    ]

    src_preview_px = min(2048, max(1024, height))
    for folder, slug, (slot_x0, slot_y0, slot_x1, slot_y1) in specs:
        if registry is None:
            continue
        try:
            cont_d = registry.load_map(slug, preview_only=True, max_preview_px=src_preview_px)
        except Exception:
            continue
        arr = cont_d.map_image
        if arr is None or arr.size == 0:
            continue
        ph, pw = arr.shape[:2]
        # Restrict to playable WoW: Forever zones (clips Silvermoon/Midnight/GM Island from shared CASC)
        playable_mask = _build_forever_playable_mask(folder, cont_d, ph, pw, pad_yards=280.0)
        # Detect non-ocean landmass bounds (exclude bg blue, dark slate, and pure-black void ocean tiles)
        diff_ocean = np.abs(arr.astype(np.int16) - np.array([22, 46, 72], dtype=np.int16)).max(axis=-1)
        diff_dark = np.abs(arr.astype(np.int16) - np.array([18, 22, 29], dtype=np.int16)).max(axis=-1)
        land = playable_mask & (diff_ocean > 12) & (diff_dark > 12) & (arr.max(axis=-1) > 14)
        ys, xs = np.where(land)
        if len(xs) >= 100:
            pad = max(8, ph // 45)
            sx0 = max(0, int(xs.min()) - pad)
            sy0 = max(0, int(ys.min()) - pad)
            sx1 = min(pw, int(xs.max()) + pad + 1)
            sy1 = min(ph, int(ys.max()) + pad + 1)
        else:
            sx0, sy0, sx1, sy1 = 0, 0, pw, ph

        crop = arr[sy0:sy1, sx0:sx1]
        crop_land = land[sy0:sy1, sx0:sx1]
        slot_w = max(32, slot_x1 - slot_x0)
        slot_h = max(32, slot_y1 - slot_y0)
        scale_fit = min(slot_w / float(max(1, crop.shape[1])), slot_h / float(max(1, crop.shape[0])))
        dw = max(16, int(round(crop.shape[1] * scale_fit)))
        dh = max(16, int(round(crop.shape[0] * scale_fit)))
        dx0 = slot_x0 + (slot_w - dw) // 2
        dy0 = slot_y0 + (slot_h - dh) // 2
        dx1 = dx0 + dw
        dy1 = dy0 + dh

        resized = cv2.resize(crop, (dw, dh), interpolation=cv2.INTER_AREA)
        mask_f = cv2.resize(crop_land.astype(np.float32), (dw, dh), interpolation=cv2.INTER_LINEAR)
        ksize = 15 if width >= 2560 else 9
        mask_f = cv2.GaussianBlur(mask_f, (ksize, ksize), 3.0)
        alpha = np.clip(mask_f * 1.35, 0.0, 1.0)[:, :, None]
        roi_dst = canvas[dy0:dy1, dx0:dx1].astype(np.float32)
        blended = resized.astype(np.float32) * alpha + roi_dst * (1.0 - alpha)
        canvas[dy0:dy1, dx0:dx1] = np.clip(blended, 0, 255).astype(np.uint8)

        placements[folder] = _ContinentPlacement(
            map_origin_xy=(float(cont_d.map_origin_xy[0]), float(cont_d.map_origin_xy[1])),
            map_scale=float(cont_d.map_scale),
            map_size_px=int(cont_d.map_size_px),
            map_rot90=int(cont_d.map_rot90),
            preview_px=int(ph),
            src_box=(sx0, sy0, sx1, sy1),
            dst_box=(dx0, dy0, dx1, dy1),
        )

    # Clean continent titles scaled proportionally to canvas width
    font = cv2.FONT_HERSHEY_SIMPLEX
    f_scale = 0.65 * (width / 1600.0)
    f_thick = max(2, int(round(2 * (width / 1600.0))))
    for text, cx_f, cy_f, col in (
        ("KALIMDOR", 0.25, 0.045, (235, 200, 115)),
        ("EASTERN KINGDOMS", 0.75, 0.045, (235, 200, 115)),
        ("THE GREAT SEA", 0.50, 0.52, (95, 138, 178)),
    ):
        (tw, _th), _ = cv2.getTextSize(text, font, f_scale, f_thick)
        tx = int(width * cx_f - tw * 0.5)
        ty = int(height * cy_f)
        cv2.putText(canvas, text, (tx + 2, ty + 2), font, f_scale, (10, 18, 28), f_thick, cv2.LINE_AA)
        cv2.putText(canvas, text, (tx, ty), font, f_scale, col, f_thick, cv2.LINE_AA)

    pil_im = Image.fromarray(canvas, mode="RGB")

    # Save precached prestitched overview to disk for instant future launches
    if cache_img_path is not None and cache_meta_path is not None and placements:
        try:
            pil_im.save(cache_img_path, format="JPEG", quality=90, optimize=True)
            meta_out = {
                "version": _AZEROTH_OVERVIEW_CACHE_VERSION,
                "width": width,
                "height": height,
                "placements": {
                    f: {
                        "map_origin_xy": list(p.map_origin_xy),
                        "map_scale": p.map_scale,
                        "map_size_px": p.map_size_px,
                        "map_rot90": p.map_rot90,
                        "preview_px": p.preview_px,
                        "src_box": list(p.src_box),
                        "dst_box": list(p.dst_box),
                    }
                    for f, p in placements.items()
                },
            }
            cache_meta_path.write_text(json.dumps(meta_out), encoding="utf-8")
        except Exception:
            pass

    return AzerothOverview(image=pil_im, placements=placements)


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
