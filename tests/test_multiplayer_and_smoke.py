"""Multi-client WebSocket integration tests & subsystem smoke tests for ForeverNavigator."""

from __future__ import annotations

import socket
import sys
import tempfile
import threading
import time
import tkinter as tk
import unittest
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.combatlog.parser import parse_combatlog_line
from src.combatlog.stats import analyze_combat_events
from src.navigator.minimap_tracker import MinimapROI, MinimapTracker
from src.navigator.settings import AppSettings, load_settings, save_settings
from src.navigator.tracker import MultiDungeonTracker
from src.navigator.world_maps import MAP_CATEGORIES, WorldMapRegistry
from src.net.client import ForeverNavigatorClient
from src.net.protocol import (
    PlayerIdentity,
    PlayerState,
    SyncPreferences,
    normalize_ws_url,
    now_ms,
)
from src.net.server import HostConfig, ForeverNavigatorServer
from src.replay.session_store import SessionStore
from src.replay.timeline import TimelineController
from src.tactics.tools import yards_to_map_px
from src.tactics.whiteboard import TacticalWhiteboard, WhiteboardOp

ROOT_DIR = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT_DIR / "data" / "index"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_until(predicate, timeout: float = 4.0, step: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return bool(predicate())


class TestMultiplayerWebSocketSync(unittest.TestCase):
    def test_normalize_ws_url(self):
        self.assertEqual(normalize_ws_url("192.168.1.50"), "ws://192.168.1.50:8765/ws")
        self.assertEqual(normalize_ws_url("192.168.1.50:9000"), "ws://192.168.1.50:9000/ws")
        self.assertEqual(normalize_ws_url("ws://example.com:8765/ws"), "ws://example.com:8765/ws")
        self.assertEqual(normalize_ws_url(""), "ws://127.0.0.1:8765/ws")

    def test_host_plus_two_clients_sync_permissions_ntp_and_disconnect(self):
        port = _free_port()
        host_states: List[Dict[str, PlayerState]] = []
        host_draws: List[Dict[str, Any]] = []

        server = ForeverNavigatorServer(
            HostConfig(
                bind_host="127.0.0.1",
                port=port,
                room_passphrase="mc-night",
                officer_token="off-secret",
                whiteboard_permission="all",
            ),
            on_raid_state=lambda st: host_states.append(st),
            on_draw_op=lambda op: host_draws.append(op),
        )
        self.assertTrue(server.start(), f"Server failed to start: {server.last_error}")

        c1_states: List[Dict[str, PlayerState]] = []
        c1_draws: List[Dict[str, Any]] = []
        c2_states: List[Dict[str, PlayerState]] = []
        c2_draws: List[Dict[str, Any]] = []

        id_host = PlayerIdentity(player_id="host-1", name="RaidLead", wow_class="WARRIOR", role="LEADER")
        id_c1 = PlayerIdentity(player_id="raider-1", name="HealBot", wow_class="PRIEST", role="HEALER")
        id_c2 = PlayerIdentity(player_id="officer-1", name="TankOff", wow_class="PALADIN", role="OFFICER")

        c1 = ForeverNavigatorClient(
            f"127.0.0.1:{port}",
            id_c1,
            SyncPreferences(),
            room_passphrase="mc-night",
            on_raid_state=lambda st: c1_states.append(st),
            on_draw_op=lambda op: c1_draws.append(op),
            auto_reconnect=False,
        )
        c2 = ForeverNavigatorClient(
            f"ws://127.0.0.1:{port}/ws",
            id_c2,
            SyncPreferences(),
            room_passphrase="mc-night",
            officer_token="off-secret",
            on_raid_state=lambda st: c2_states.append(st),
            on_draw_op=lambda op: c2_draws.append(op),
            auto_reconnect=False,
        )

        try:
            self.assertTrue(c1.start())
            self.assertTrue(c2.start())
            self.assertTrue(
                _wait_until(lambda: c1.connected and c2.connected and server.connected_count == 2),
                "Both clients should connect to host",
            )

            # Verify welcome role assignment & NTP clock sync
            self.assertTrue(_wait_until(lambda: c1.session_id != "" and c2.session_id != ""))
            self.assertEqual(c1.assigned_role, "raider")
            self.assertEqual(c2.assigned_role, "officer")
            self.assertTrue(_wait_until(lambda: c1._best_rtt_ms is not None and c2._best_rtt_ms is not None))
            self.assertLess(abs(c1.host_now_ms() - now_ms()), 5000)

            # Position updates from Host + 2 Clients -> raid_state has all 3
            server.update_host_position_sync(
                PlayerState("host-1", "RaidLead", "WARRIOR", "LEADER", "molten_core", 10, 20, 0, (100, 120))
            )
            c1.send_position_sync(
                PlayerState("raider-1", "HealBot", "PRIEST", "HEALER", "molten_core", 15, 25, 0, (110, 130))
            )
            c2.send_position_sync(
                PlayerState("officer-1", "TankOff", "PALADIN", "OFFICER", "molten_core", 20, 30, 0, (120, 140), alive=False)
            )

            self.assertTrue(
                _wait_until(
                    lambda: (
                        len(server.players) == 3
                        and bool(c1_states)
                        and len(c1_states[-1]) == 3
                        and bool(c2_states)
                        and len(c2_states[-1]) == 3
                    )
                ),
                "All 3 players should appear in raid_state across host and clients",
            )
            self.assertFalse(c1_states[-1]["officer-1"].alive)
            self.assertEqual(c2_states[-1]["raider-1"].wow_class, "PRIEST")

            # Draw permission: "all" -> raider can draw
            op_all = WhiteboardOp(
                op_id="op-all-1",
                author_id="raider-1",
                dungeon_slug="molten_core",
                kind="text",
                points=[(0.5, 0.5)],
                text="Stack",
            ).to_dict()
            c1.send_draw_op_sync(op_all)
            self.assertTrue(
                _wait_until(
                    lambda: any(o.get("op_id") == "op-all-1" for o in host_draws)
                    and any(o.get("op_id") == "op-all-1" for o in c2_draws)
                )
            )

            # Restrict to "officers" -> raider denied, officer allowed
            server.set_whiteboard_permission("officers")
            self.assertTrue(
                _wait_until(lambda: c1.whiteboard_permission == "officers" and c2.whiteboard_permission == "officers")
            )
            self.assertFalse(c1.can_draw())
            self.assertTrue(c2.can_draw())

            op_denied = WhiteboardOp(
                op_id="op-denied-1",
                author_id="raider-1",
                dungeon_slug="molten_core",
                kind="text",
                points=[(0.2, 0.2)],
                text="Denied",
            ).to_dict()
            c1.send_draw_op_sync(op_denied)
            self.assertTrue(_wait_until(lambda: c1.last_error is not None and "restricted" in c1.last_error.lower()))
            self.assertFalse(any(o.get("op_id") == "op-denied-1" for o in host_draws))

            op_officer = WhiteboardOp(
                op_id="op-off-1",
                author_id="officer-1",
                dungeon_slug="molten_core",
                kind="circle_range",
                points=[(0.4, 0.4)],
                radius_yd=8.0,
            ).to_dict()
            c2.send_draw_op_sync(op_officer)
            self.assertTrue(
                _wait_until(
                    lambda: any(o.get("op_id") == "op-off-1" for o in host_draws)
                    and any(o.get("op_id") == "op-off-1" for o in c1_draws)
                )
            )

            # Restrict to "leader" -> officer denied, host broadcast works
            server.set_whiteboard_permission("leader")
            self.assertTrue(_wait_until(lambda: not c2.can_draw()))
            c2.last_error = None
            c2.send_draw_op_sync(
                WhiteboardOp(op_id="op-off-denied", author_id="officer-1", dungeon_slug="molten_core").to_dict()
            )
            self.assertTrue(_wait_until(lambda: c2.last_error is not None and "restricted" in c2.last_error.lower()))

            server.broadcast_draw_op_sync(
                WhiteboardOp(op_id="op-host-1", author_id="host-1", dungeon_slug="molten_core", text="Pull").to_dict()
            )
            self.assertTrue(_wait_until(lambda: any(o.get("op_id") == "op-host-1" for o in c1_draws)))

            # Disconnect cleanup: stop c2 -> server removes officer-1 and updates c1
            c2.stop()
            self.assertTrue(
                _wait_until(
                    lambda: server.connected_count == 1
                    and "officer-1" not in server.players
                    and "officer-1" not in c1_states[-1]
                )
            )
        finally:
            c1.stop()
            c2.stop()
            server.stop()

    def test_passphrase_rejection_and_port_conflict(self):
        port = _free_port()
        s1 = ForeverNavigatorServer(HostConfig(bind_host="127.0.0.1", port=port, room_passphrase="secret"))
        self.assertTrue(s1.start())
        s2 = ForeverNavigatorServer(HostConfig(bind_host="127.0.0.1", port=port))
        bad_client = ForeverNavigatorClient(
            f"127.0.0.1:{port}",
            PlayerIdentity(player_id="bad-1", name="Intruder"),
            SyncPreferences(),
            room_passphrase="wrong",
            auto_reconnect=True,
        )
        try:
            # Port conflict should return False cleanly with last_error
            self.assertFalse(s2.start())
            self.assertIsNotNone(s2.last_error)

            # Wrong passphrase should be rejected with 1008 and stop reconnecting
            bad_client.start()
            self.assertTrue(
                _wait_until(lambda: bad_client.last_error is not None and "passphrase" in bad_client.last_error.lower()),
                f"Expected passphrase error, got {bad_client.last_error}",
            )
            self.assertFalse(bad_client.connected)
        finally:
            bad_client.stop()
            s2.stop()
            s1.stop()


class TestSubsystemSmoke(unittest.TestCase):
    def test_whiteboard_malformed_and_clear_all(self):
        wb = TacticalWhiteboard()
        wb.apply_remote_dict(
            {
                "op_id": "bad-pts",
                "dungeon_slug": "molten_core",
                "kind": "stroke",
                "points": [[0.1, 0.2], "bad", [0.5], [0.3, 0.4]],
            }
        )
        ops = wb.ops_for("molten_core")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].points, [(0.1, 0.2), (0.3, 0.4)])

        wb.apply_remote_dict({"kind": "clear_all", "dungeon_slug": "molten_core"})
        self.assertEqual(len(wb.ops_for("molten_core")), 0)
        self.assertAlmostEqual(yards_to_map_px(10.0, 2.5), 25.0)

    def test_combatlog_advanced_fields_and_empty(self):
        # Empty log should not divide by zero
        empty_sum = analyze_combat_events([])
        self.assertEqual(empty_sum.dps_leaderboard(), [])
        self.assertEqual(empty_sum.hps_leaderboard(), [])

        # Malformed lines return None
        self.assertIsNone(parse_combatlog_line("not a valid combatlog line"))

        # Interrupt & Dispel & Swing
        lines = [
            '11/25/2026 18:00:00.000  SWING_DAMAGE,Player-1,"Rogue",0x511,0x0,Creature-1,"Garr",0x10a48,0x0,450,0,1,0,0,0,nil,nil,nil',
            '11/25/2026 18:00:01.000  SPELL_INTERRUPT,Player-1,"Rogue",0x511,0x0,Creature-1,"Garr",0x10a48,0x0,1766,"Kick",0x1,116,"Frostbolt"',
            '11/25/2026 18:00:02.000  SPELL_DISPEL,Player-2,"Priest",0x514,0x0,Player-1,"Rogue",0x511,0x0,527,"Dispel Magic",0x2,19702,"Impending Doom"',
        ]
        evs = [e for ln in lines if (e := parse_combatlog_line(ln)) is not None]
        s = analyze_combat_events(evs)
        self.assertEqual(s.players["Rogue"].damage_done, 450)
        self.assertEqual(s.players["Rogue"].interrupts, 1)
        self.assertEqual(s.players["Priest"].dispels, 1)

    def test_session_store_thread_safety(self):
        with tempfile.TemporaryDirectory() as td:
            store = SessionStore(Path(td) / "concurrent.db")
            errors: List[Exception] = []

            def writer(pid: str):
                try:
                    for i in range(25):
                        store.record_position(
                            PlayerState(pid, pid, "MAGE", "DPS", "molten_core", float(i), 0.0, 0.0, (i, i), t_host_ms=1000 + i * 100)
                        )
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=writer, args=(f"p{k}",)) for k in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [])
            tl = TimelineController(store)
            snap = tl.scrub_to(3400)
            self.assertEqual(len(snap.players), 4)
            store.close()

    def test_settings_load_corrupt_and_save(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "settings.json"
            p.write_text("{corrupt json", encoding="utf-8")
            s = load_settings(p)
            self.assertEqual(s.fps, 1.0)
            s.fps = 99.0
            save_settings(p, s)
            s2 = load_settings(p)
            self.assertEqual(s2.fps, 5.0)

    @unittest.skipUnless((INDEX_DIR / "manifest.json").exists(), "Index not present")
    def test_minimap_tracker_smoke_and_coord_roundtrip(self):
        t3d = MultiDungeonTracker(INDEX_DIR)
        d = t3d.dungeons["ragefire_chasm"]
        px, py = d.world_to_map_px(d.waypoints[0, :2])
        wx, wy = d.map_px_to_world(px, py)
        self.assertAlmostEqual(wx, float(d.waypoints[0, 0]), delta=2.0)
        self.assertAlmostEqual(wy, float(d.waypoints[0, 1]), delta=2.0)

        mm = MinimapTracker(t3d.dungeons, t3d.dungeon_order, roi=MinimapROI())
        fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        est = mm.update(fake_frame)
        self.assertIn(est.dungeon_slug, t3d.dungeons)
        self.assertEqual(len(est.map_px), 2)

        # Concurrent register_dungeon() during update() must never raise broadcast shape errors
        from src.navigator.tracker import DungeonData
        errors = []

        def _updater():
            non_black = np.full((256, 256, 3), 120, dtype=np.uint8)
            for _ in range(15):
                try:
                    mm.update(non_black, is_already_cropped=True)
                except Exception as exc:
                    errors.append(exc)

        t = threading.Thread(target=_updater)
        t.start()
        for k in range(5):
            extra = DungeonData(
                slug=f"extra_zone_{k}",
                name=f"Extra Zone {k}",
                combatlog_keywords=[],
                waypoints=np.zeros((1, 3), dtype=np.float32),
                adjacency=[[]],
                map_image=np.full((512, 512, 3), 90, dtype=np.uint8),
                map_origin_xy=np.zeros(2, dtype=np.float32),
                map_scale=0.5,
                embeddings_f32=np.zeros((1, 4), dtype=np.float32),
                view_wp_idx=np.zeros(1, dtype=np.int32),
                wp_to_views=[np.zeros(0, dtype=np.int32)],
            )
            mm.register_dungeon(extra)
        t.join(timeout=5.0)
        self.assertEqual(errors, [])


class TestWorldMapsAndModernUI(unittest.TestCase):
    def test_world_maps_catalog_filtering_and_offline_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            reg = WorldMapRegistry(
                index_dir=INDEX_DIR if (INDEX_DIR / "manifest.json").exists() else Path(td) / "no_index",
                cache_dir=Path(td) / "cache",
                wow_dir=Path(td) / "no_wow",
            )
            all_maps = reg.list_all_maps()
            self.assertGreaterEqual(len(all_maps), 70)
            cats = {m.category for m in all_maps}
            for expected in ("Raids", "Dungeons", "Battlegrounds", "Cities", "Zones", "Continents"):
                self.assertIn(expected, cats)
            self.assertIn("Zones", MAP_CATEGORIES)

            hyjal = reg.list_all_maps(category="Zones", query="hyjal")
            self.assertTrue(any(m.slug == "mount_hyjal" for m in hyjal))
            raids = reg.list_all_maps(category="Raids")
            self.assertTrue(all(m.category == "Raids" for m in raids))

    @unittest.skipUnless((INDEX_DIR / "manifest.json").exists(), "Index not present")
    def test_world_maps_load_async_and_register_dungeon(self):
        with tempfile.TemporaryDirectory() as td:
            reg = WorldMapRegistry(index_dir=INDEX_DIR, cache_dir=Path(td) / "cache")
            done_evt = threading.Event()
            result: Dict[str, Any] = {}

            def on_done(slug, dungeon, exc):
                result["slug"] = slug
                result["dungeon"] = dungeon
                result["exc"] = exc
                done_evt.set()

            reg.load_map_async("molten_core", on_done)
            self.assertTrue(done_evt.wait(timeout=4.0))
            self.assertIsNone(result["exc"])
            mc = result["dungeon"]
            self.assertIsNotNone(mc)

            t3d = MultiDungeonTracker(INDEX_DIR)
            mm = MinimapTracker(t3d.dungeons, t3d.dungeon_order)
            t3d.register_dungeon(mc)
            mm.register_dungeon(mc)
            self.assertIn("molten_core", t3d.dungeons)
            self.assertIn("molten_core", mm.dungeons)

    @unittest.skipUnless(Path(r"E:\Spel\World of Warcraft\.build.info").exists(), "Local WoW CASC not present")
    def test_world_maps_casc_stitch_cache_and_coord_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            reg = WorldMapRegistry(
                index_dir=INDEX_DIR,
                cache_dir=Path(td) / "cache",
                wow_dir=Path(r"E:\Spel\World of Warcraft"),
            )
            d = reg.load_map("elwynn_forest")
            self.assertEqual(d.slug, "elwynn_forest")
            self.assertGreater(d.map_image.shape[0], 0)
            # Verify cached reload works and coordinates roundtrip cleanly
            d_cached = reg.load_map("elwynn_forest")
            px, py = d_cached.world_to_map_px(np.array([-8900.0, -100.0], dtype=np.float32))
            wx, wy = d_cached.map_px_to_world(px, py)
            self.assertAlmostEqual(wx, -8900.0, delta=15.0)
            self.assertAlmostEqual(wy, -100.0, delta=15.0)

    def test_pythonw_none_stdio_safety(self):
        from src.ui.app import _ensure_stdio

        orig_out, orig_err = sys.stdout, sys.stderr
        try:
            sys.stdout = None
            sys.stderr = None
            _ensure_stdio()
            self.assertIsNotNone(sys.stdout)
            self.assertIsNotNone(sys.stderr)
            print("pythonw smoke check")
        finally:
            if sys.stdout not in (None, orig_out):
                try:
                    sys.stdout.close()
                except Exception:
                    pass
            if sys.stderr not in (None, orig_err, sys.stdout):
                try:
                    sys.stderr.close()
                except Exception:
                    pass
            sys.stdout = orig_out
            sys.stderr = orig_err

    @unittest.skipUnless((INDEX_DIR / "manifest.json").exists(), "Index not present")
    def test_modernized_ui_instantiation_and_close(self):
        from src.ui.app import ForeverNavigatorApp

        try:
            app = ForeverNavigatorApp(INDEX_DIR)
        except tk.TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
            return
        try:
            app.root.update()
            self.assertIsNotNone(app.world_maps)
            self.assertGreater(len(app._all_map_entries), 0)
            self.assertTrue(app._can_edit_whiteboard())
        finally:
            app._on_close()

    def test_dev_gameplay_logger_disabled_and_enabled(self):
        import json
        from types import SimpleNamespace
        from src.navigator.dev_logger import DevGameplayLogger

        with tempfile.TemporaryDirectory() as td:
            # Disabled by default when no .enabled marker or env var
            dl_off = DevGameplayLogger(Path(td) / "off", force_enable=False)
            self.assertFalse(dl_off.enabled)
            self.assertIsNone(dl_off.session_dir)
            dl_off.close()

            # Enabled mode writes events.jsonl, minimap PNG, and fullscreen JPG
            dl = DevGameplayLogger(
                Path(td) / "on",
                minimap_interval_sec=0.0,
                fullscreen_interval_sec=0.0,
                force_enable=True,
            )
            self.assertTrue(dl.enabled)
            self.assertIsNotNone(dl.session_dir)
            fake_est = SimpleNamespace(
                dungeon_slug="the_hinterlands",
                dungeon_confidence=0.92,
                position_confidence=0.81,
                map_px=(520, 480),
                world_pos=np.array([100.0, -200.0, 0.0], dtype=np.float32),
                inference_ms=4.5,
                all_dungeon_scores={"the_hinterlands": 0.85, "arathi_highlands": 0.30},
            )
            mm_img = np.zeros((64, 64, 3), dtype=np.uint8)
            fs_img = np.zeros((120, 160, 3), dtype=np.uint8)
            dl.log_tracker_tick(fake_est, "the_hinterlands", 3.2, mm_img, fs_img, {"x0": 0.88, "y0": 0.04, "x1": 0.99, "y1": 0.22})
            dl.log_minimap_roi_sample(mm_img, {"x0": 0.88, "y0": 0.04, "x1": 0.99, "y1": 0.22})
            dl.close()

            events_file = dl.session_dir / "events.jsonl"
            self.assertTrue(events_file.exists())
            lines = [json.loads(ln) for ln in events_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
            kinds = [r["event"] for r in lines]
            self.assertIn("session_start", kinds)
            self.assertIn("tracker_tick", kinds)
            self.assertIn("minimap_roi", kinds)
            self.assertIn("session_end", kinds)
            self.assertTrue(list((dl.session_dir / "minimap").glob("*.png")))
            self.assertTrue(list((dl.session_dir / "fullscreen").glob("*.jpg")))


if __name__ == "__main__":
    unittest.main()
