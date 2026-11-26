"""Tests for full-resolution minimap stitching, complete CASC minimap coverage, and shared cave/building accounting."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.navigator.minimap_tracker import MinimapTracker
from src.navigator.tracker import DungeonData
from src.navigator.world_maps import (
    ADT_YARD_SIZE,
    MAX_IN_MEMORY_MAP_PX,
    MINIMAP_INDEX_VERSION,
    OVERVIEW_MAP_PX,
    WorldMapEntry,
    WorldMapRegistry,
)

WOW_DIR = Path(r"E:\Spel\World of Warcraft")
LISTFILE_PATH = Path(r"E:\Programmering\WoW\MapNavigate\data\listfile.csv")
HAS_REAL_CASC = (WOW_DIR / ".build.info").exists() and LISTFILE_PATH.exists()


class TestFullResMinimapsSynthetic(unittest.TestCase):
    def test_v2_index_loads_all_categories_and_shared_wmos(self):
        with tempfile.TemporaryDirectory() as td:
            idx_dir = Path(td) / "index"
            cache_dir = Path(td) / "cache"
            idx_dir.mkdir()
            cache_dir.mkdir()

            v2_index = {
                "version": MINIMAP_INDEX_VERSION,
                "folders": {
                    "TestZone": [[30, 30, 1001], [31, 30, 1002]],
                    "2299": [[17, 24, 1003]],
                },
                "noliquid": {
                    "TestZone": [[30, 30, 1004]],
                },
                "wmos": {
                    "azeroth/buildings/duskwood_barracks/duskwood_barracks": [[0, 0, 0, 2001], [1, 0, 0, 2002]],
                    "dungeon/kl_diremaul/kl_diremaul": [[0, 0, 0, 2003]],
                },
                "wmo_roots": {},
            }
            (cache_dir / "minimap_tiles_index.json").write_text(json.dumps(v2_index), encoding="utf-8")

            reg = WorldMapRegistry(index_dir=idx_dir, cache_dir=cache_dir, wow_dir=None)
            stats = reg.get_minimap_coverage_stats()
            self.assertEqual(stats["map_tiles"], 3)
            self.assertEqual(stats["noliquid_tiles"], 1)
            self.assertEqual(stats["wmo_tiles"], 3)
            self.assertEqual(stats["total_minimap_files"], 7)

            # Shared cave/building WMO is marked is_shared=True, while unique dungeon WMO is False
            barracks = reg.get_entry("wmo_azeroth_buildings_duskwood_barracks_duskwood_barracks")
            self.assertIsNotNone(barracks)
            assert barracks is not None
            self.assertEqual(barracks.category, "Caves & Buildings")
            self.assertTrue(barracks.is_shared)

            dm_wmo = reg.get_entry("wmo_dungeon_kl_diremaul_kl_diremaul")
            self.assertIsNotNone(dm_wmo)
            assert dm_wmo is not None
            self.assertFalse(dm_wmo.is_shared)

            # noLiquid & numeric folder entries exist
            self.assertIsNotNone(reg.get_entry("testzone_noliquid"))
            self.assertIsNotNone(reg.get_entry("2299"))

    def test_lowres_cache_is_upgraded_when_casc_available(self):
        with tempfile.TemporaryDirectory() as td:
            idx_dir = Path(td) / "index"
            cache_dir = Path(td) / "cache"
            idx_dir.mkdir()
            cache_dir.mkdir()

            # Write a legacy low-res .npz without full_res flag
            legacy_img = np.zeros((512, 512, 3), dtype=np.uint8)
            np.savez_compressed(
                cache_dir / "durotar.npz",
                slug="durotar",
                name="Durotar",
                combatlog_keywords=np.array(["durotar"], dtype=str),
                waypoints=np.zeros((1, 3), dtype=np.float32),
                map_image=legacy_img,
                map_origin_xy=np.zeros(2, dtype=np.float32),
                map_scale=np.float32(0.1),
                map_rot90=np.int32(1),
            )

            reg = WorldMapRegistry(index_dir=idx_dir, cache_dir=cache_dir, wow_dir=None)
            reg.wow_dir = Path(td)  # pretend CASC is available
            upgraded_img = np.full((3392, 3392, 3), 120, dtype=np.uint8)
            upgraded_d = DungeonData(
                slug="durotar",
                name="Durotar",
                combatlog_keywords=["durotar"],
                waypoints=np.zeros((1, 3), dtype=np.float32),
                adjacency=[[]],
                map_image=upgraded_img,
                map_origin_xy=np.zeros(2, dtype=np.float32),
                map_scale=float(512.0 / ADT_YARD_SIZE),
                embeddings_f32=np.zeros((0, 1), dtype=np.float32),
                view_wp_idx=np.zeros((0,), dtype=np.int32),
                wp_to_views=[np.zeros((0,), dtype=np.int32)],
                map_rot90=1,
            )
            with patch.object(reg, "_stitch_from_casc", return_value=(upgraded_d, 3392)) as mock_stitch:
                d = reg.load_map("durotar")
                mock_stitch.assert_called_once()
                self.assertEqual(d.map_image.shape[0], 3392)

    def test_minimap_tracker_shared_cave_building_context(self):
        outdoor = DungeonData(
            slug="elwynn_forest",
            name="Elwynn Forest",
            combatlog_keywords=["elwynn"],
            waypoints=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            adjacency=[[]],
            map_image=np.full((512, 512, 3), 80, dtype=np.uint8),
            map_origin_xy=np.array([-500.0, -500.0], dtype=np.float32),
            map_scale=0.5,
            embeddings_f32=np.zeros((0, 1), dtype=np.float32),
            view_wp_idx=np.zeros((0,), dtype=np.int32),
            wp_to_views=[np.zeros((0,), dtype=np.int32)],
            map_rot90=1,
        )
        shared_inn = DungeonData(
            slug="wmo_azeroth_buildings_human_inn",
            name="Buildings / Human Inn",
            combatlog_keywords=["human_inn"],
            waypoints=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            adjacency=[[]],
            map_image=np.full((128, 128, 3), 40, dtype=np.uint8),
            map_origin_xy=np.array([0.0, 0.0], dtype=np.float32),
            map_scale=2.0,
            embeddings_f32=np.zeros((0, 1), dtype=np.float32),
            view_wp_idx=np.zeros((0,), dtype=np.int32),
            wp_to_views=[np.zeros((0,), dtype=np.int32)],
            map_rot90=0,
        )
        tracker = MinimapTracker(
            dungeons={"elwynn_forest": outdoor, shared_inn.slug: shared_inn},
            dungeon_order=["elwynn_forest", shared_inn.slug],
        )
        self.assertTrue(tracker.is_shared_map(shared_inn.slug))
        self.assertFalse(tracker.is_shared_map("elwynn_forest"))

        # Simulate outdoor frame first, then locked shared cave/building frame
        frame = np.full((128, 128, 3), 80, dtype=np.uint8)
        est_out = tracker.update(frame, is_already_cropped=True)
        self.assertFalse(est_out.is_shared_minimap)

        tracker.locked_dungeon_slug = shared_inn.slug
        sparse_indoor = np.zeros((128, 128, 3), dtype=np.uint8)
        sparse_indoor[56:72, 56:72] = 140
        est_in = tracker.update(sparse_indoor, is_already_cropped=True)
        self.assertTrue(est_in.is_shared_minimap)
        self.assertEqual(est_in.outdoor_context_slug, "elwynn_forest")

    def test_high_precision_mode_and_gpu_matcher(self):
        # Verify GPUComputeMatcher matches CPU ZNCC accurately when OpenGL 4.3 is available
        from src.navigator.gpu_matcher import GPUComputeMatcher

        gm = GPUComputeMatcher()
        if gm.available:
            rng = np.random.default_rng(42)
            m_sil = rng.random((256, 256, 3), dtype=np.float32)
            m_edge = rng.random((256, 256), dtype=np.float32)
            t_sil = m_sil[80:112, 90:122].copy()
            t_edge = m_edge[80:112, 90:122].copy()
            gm.upload_map("m", m_sil, m_edge)
            res = gm.match("m", t_sil, t_edge)
            self.assertIsNotNone(res)
            assert res is not None
            py, px = np.unravel_index(np.argmax(res), res.shape)
            self.assertEqual((int(py), int(px)), (80, 90))
            gm.close()

        # Verify MinimapTracker high_precision mode toggle & full-res local refinement
        rng = np.random.default_rng(7)
        big_map = rng.integers(30, 220, size=(2048, 2048, 3), dtype=np.uint8)
        d = DungeonData(
            slug="hi_prec_zone",
            name="Hi Prec Zone",
            combatlog_keywords=["hi_prec"],
            waypoints=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            adjacency=[[]],
            map_image=big_map,
            map_origin_xy=np.array([-1000.0, -1000.0], dtype=np.float32),
            map_scale=1.0,
            embeddings_f32=np.zeros((0, 1), dtype=np.float32),
            view_wp_idx=np.zeros((0,), dtype=np.int32),
            wp_to_views=[np.zeros((0,), dtype=np.int32)],
            map_rot90=1,
        )
        tracker = MinimapTracker(
            dungeons={"hi_prec_zone": d},
            dungeon_order=["hi_prec_zone"],
            high_precision=True,
            use_gpu=False,
        )
        self.assertTrue(tracker.high_precision)
        # Crop a patch centered at (1200, 900)
        patch = big_map[900 - 80 : 900 + 80, 1200 - 80 : 1200 + 80]
        est = tracker.update(patch, is_already_cropped=True)
        self.assertEqual(est.dungeon_slug, "hi_prec_zone")
        self.assertAlmostEqual(est.map_px[0], 1200, delta=6)
        self.assertAlmostEqual(est.map_px[1], 900, delta=6)
        tracker.set_high_precision(False)
        self.assertFalse(tracker.high_precision)

    def test_fullres_tile_source_culls_offscreen_tiles_and_mip_lod(self):
        from src.navigator.world_maps import FullResTileSource

        src = FullResTileSource(
            wow_dir=Path("."),
            tile_lookup={(30, 30): 1, (31, 31): 2, (10, 10): 3},
            native_tile_px=512,
            overview_px=4096,
            wx0=-10000.0,
            wy0=-10000.0,
            span_yards=10000.0,
        )
        decoded_calls = []

        def fake_get_tile(tx, ty):
            decoded_calls.append((tx, ty))
            return Image.new("RGB", (512, 512), (40, 80, 120))

        with patch.object(src, "_get_tile", side_effect=fake_get_tile):
            # Center viewport far away from tile (10, 10)
            vp = src.render_viewport(2048.0, 2048.0, 4.0, 200, 200)
            self.assertEqual(vp.size, (200, 200))
            self.assertNotIn((10, 10), decoded_calls)

        # Verify mipmap LOD helper returns coarser mip when zoomed out, full-res when zoomed in
        from src.ui.app import ForeverNavigatorApp

        dummy = ForeverNavigatorApp.__new__(ForeverNavigatorApp)
        dummy._cached_mips = {}
        full = Image.new("RGB", (4096, 4096), (10, 20, 30))
        im_out, div_out = dummy._get_mip_for_scale("k", full, 0.12)
        self.assertGreater(div_out, 1)
        self.assertLess(im_out.width, 4096)
        im_in, div_in = dummy._get_mip_for_scale("k", full, 2.0)
        self.assertEqual(div_in, 1)
        self.assertEqual(im_in.width, 4096)

    def test_storage_mode_preview_streaming_and_lru_eviction(self):
        from collections import OrderedDict
        from src.navigator.settings import AppSettings, detect_storage_type, resolve_storage_mode
        from src.navigator.tracker import load_dungeon_npz
        from src.ui.app import ForeverNavigatorApp

        self.assertIn(detect_storage_type("."), ("ssd", "hdd"))
        self.assertEqual(resolve_storage_mode("ssd", "."), "ssd")
        self.assertEqual(resolve_storage_mode("hdd", "."), "hdd")
        self.assertEqual(AppSettings(storage_mode="invalid").clamped().storage_mode, "auto")

        # Verify preview_only downscales RAM image while preserving exact native coordinates
        with tempfile.TemporaryDirectory() as td:
            npz_p = Path(td) / "big_zone.npz"
            big = np.zeros((2048, 2048, 3), dtype=np.uint8)
            np.savez_compressed(
                npz_p,
                slug="big_zone",
                name="Big Zone",
                combatlog_keywords=np.array(["big_zone"]),
                waypoints=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                map_image=big,
                map_origin_xy=np.array([-1000.0, -1000.0], dtype=np.float32),
                map_scale=np.float32(1.0),
                map_rot90=np.int32(1),
            )
            d_prev = load_dungeon_npz(npz_p, max_preview_px=512)
            self.assertEqual(d_prev.map_image.shape[:2], (512, 512))
            self.assertEqual(d_prev.map_size_px, 2048)
            px = d_prev.world_to_map_px(np.array([-250.0, 125.0]))
            wx, wy = d_prev.map_px_to_world(px[0], px[1])
            self.assertAlmostEqual(wx, -250.0, delta=1.5)
            self.assertAlmostEqual(wy, 125.0, delta=1.5)

        # Verify LRU display cache evicts to 1 zone on SSD and keeps up to 3 on HDD
        dummy = ForeverNavigatorApp.__new__(ForeverNavigatorApp)
        dummy._cached_pil_maps = OrderedDict()
        dummy._cached_mips = {}
        dummy._effective_storage_mode = "ssd"
        for slug in ("z1", "z2", "z3"):
            dummy._put_display_pil(slug, Image.new("RGB", (64, 64), (10, 20, 30)))
        self.assertEqual(list(dummy._cached_pil_maps.keys()), ["z3"])

        dummy._effective_storage_mode = "hdd"
        for slug in ("z4", "z5", "z6"):
            dummy._put_display_pil(slug, Image.new("RGB", (64, 64), (10, 20, 30)))
        self.assertEqual(len(dummy._cached_pil_maps), 3)
        self.assertEqual(list(dummy._cached_pil_maps.keys()), ["z4", "z5", "z6"])

    def test_azeroth_overview_composite_and_fix_gate(self):
        from src.navigator.tracker import TrackerEstimate
        from src.navigator.world_maps import build_azeroth_overview
        from src.ui.app import ForeverNavigatorApp

        # Works cleanly even with registry=None (offline fallback)
        ov_off = build_azeroth_overview(None, width=800, height=450)
        self.assertEqual((ov_off.width, ov_off.height), (800, 450))
        self.assertEqual(ov_off.world_to_overview_px("Kalimdor", 0.0, 0.0), (400.0, 225.0))
        ov_off.close()

        # Works with real cached continents if available in workspace
        reg = WorldMapRegistry(index_dir="data/index", cache_dir="data/cache/maps")
        ov = build_azeroth_overview(reg, width=1600, height=900)
        self.assertEqual((ov.width, ov.height), (1600, 900))
        if "Kalimdor" in ov.placements and "Azeroth" in ov.placements:
            kx, ky = ov.world_to_overview_px("Kalimdor", -3840.0, 1779.0)  # Feralas
            ex, ey = ov.world_to_overview_px("Azeroth", -8900.0, 600.0)    # Elwynn/Stormwind
            self.assertLess(kx, ov.width * 0.5)
            self.assertGreater(ex, ov.width * 0.5)
            mapped_k = ov.overview_px_to_continent_px(kx, ky)
            mapped_e = ov.overview_px_to_continent_px(ex, ey)
            self.assertIsNotNone(mapped_k)
            self.assertIsNotNone(mapped_e)
            assert mapped_k is not None and mapped_e is not None
            self.assertEqual(mapped_k[1], "kalimdor")
            self.assertEqual(mapped_e[1], "eastern_kingdoms")
            self.assertGreater(mapped_k[4], 1.0)
            self.assertGreater(mapped_e[4], 1.0)
        ov.close()
        reg.close()

        # Verify good-fix threshold rejects weak noise and accepts confident fixes
        weak = TrackerEstimate("feralas", "Feralas", 0.30, np.zeros(3, np.float32), (100, 100), 0.25, [], 10.0, {})
        good = TrackerEstimate("feralas", "Feralas", 0.72, np.zeros(3, np.float32), (100, 100), 0.62, [], 10.0, {})
        self.assertFalse(ForeverNavigatorApp._is_good_position_fix(weak))
        self.assertTrue(ForeverNavigatorApp._is_good_position_fix(good))


@unittest.skipUnless(HAS_REAL_CASC, "Local WoW: Forever CASC + listfile.csv not available")
class TestFullResMinimapsRealCASC(unittest.TestCase):
    def test_all_casc_minimaps_indexed_and_full_resolution_stitching(self):
        with tempfile.TemporaryDirectory() as td:
            reg = WorldMapRegistry(
                index_dir=Path(td) / "index",
                cache_dir=Path(td) / "cache",
                wow_dir=WOW_DIR,
                listfile_path=LISTFILE_PATH,
            )
            stats = reg.get_minimap_coverage_stats()
            self.assertGreaterEqual(stats["map_folders"], 715)
            self.assertGreaterEqual(stats["map_tiles"], 48896)
            self.assertGreaterEqual(stats["noliquid_tiles"], 1027)
            self.assertGreaterEqual(stats["wmo_models"], 3099)
            self.assertGreaterEqual(stats["wmo_tiles"], 51640)
            self.assertGreaterEqual(stats["total_minimap_files"], 101500)

            # 1. Zone stitched at 100% native 512px-per-ADT resolution (zero downsampling)
            durotar = reg.load_map("durotar")
            expected_scale = 512.0 / ADT_YARD_SIZE
            self.assertAlmostEqual(durotar.map_scale, expected_scale, places=3)
            self.assertGreater(durotar.map_image.shape[0], 4096)
            self.assertLessEqual(durotar.map_image.shape[0], MAX_IN_MEMORY_MAP_PX)

            # 2. Huge continent gets overview + FullResTileSource for native 512px zooming
            kalimdor = reg.load_map("kalimdor")
            self.assertEqual(kalimdor.map_image.shape[0], OVERVIEW_MAP_PX)
            self.assertTrue(reg.has_full_res_tiles("kalimdor"))
            vp = reg.render_full_res_viewport("kalimdor", 2048.0, 2048.0, 4.0, 320, 240)
            self.assertIsNotNone(vp)
            assert vp is not None
            self.assertEqual(vp.size, (320, 240))

            # 3. Numeric folder (<Folder>_XX_YY.blp), noLiquid, and shared WMO cave/building
            m2299 = reg.load_map("2299")
            self.assertEqual(m2299.map_image.shape[0], 5 * 512)
            self.assertAlmostEqual(m2299.map_scale, expected_scale, places=3)

            noliq = reg.load_map("firelands1_noliquid")
            self.assertGreater(noliq.map_image.shape[0], 0)

            barracks = reg.load_map("wmo_azeroth_buildings_duskwood_barracks_duskwood_barracks")
            self.assertGreaterEqual(barracks.map_image.shape[0], 64)
            self.assertAlmostEqual(barracks.map_scale, 2.0, places=2)


if __name__ == "__main__":
    unittest.main()
