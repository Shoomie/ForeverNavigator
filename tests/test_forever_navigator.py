"""Automated verification test suite for ForeverNavigator skeleton."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.combatlog.parser import parse_combatlog_line
from src.combatlog.stats import analyze_combat_events
from src.navigator.tracker import MultiDungeonTracker
from src.net.protocol import PlayerIdentity, PlayerState, SyncPreferences
from src.replay.session_store import SessionStore
from src.replay.timeline import TimelineController
from src.tactics.tools import measure_yards
from src.tactics.whiteboard import TacticalWhiteboard, WhiteboardOp

ROOT_DIR = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT_DIR / "data" / "index"


class TestForeverNavigatorCore(unittest.TestCase):
    def test_ruler_distance_measurement(self):
        # On a 1024px map with scale 2.0 px/yd, 0.1 normalized = 102.4 px = 51.2 yards
        yd = measure_yards(0.2, 0.5, 0.3, 0.5, map_size_px=1024, map_scale_px_per_yd=2.0)
        self.assertAlmostEqual(yd, 51.2, places=2)

    def test_whiteboard_ops_undo_and_remote_sync(self):
        wb = TacticalWhiteboard()
        op = WhiteboardOp(dungeon_slug="ragefire_chasm", kind="text", points=[(0.5, 0.5)], text="Stack Here")
        wb.add_op(op)
        self.assertEqual(len(wb.ops_for("ragefire_chasm")), 1)

        # Duplicate op_id is ignored
        wb.apply_remote_dict(op.to_dict())
        self.assertEqual(len(wb.ops_for("ragefire_chasm")), 1)

        wb.apply_remote_dict({"kind": "undo", "dungeon_slug": "ragefire_chasm"})
        self.assertEqual(len(wb.ops_for("ragefire_chasm")), 0)

    def test_combatlog_parser_and_extensive_stats(self):
        sample_lines = [
            '11/25/2026 18:00:00.000  ENCOUNTER_START,663,"Lucifron",9,40',
            '11/25/2026 18:00:02.000  SPELL_DAMAGE,Player-1,"Thrall",0x511,0x0,Creature-1,"Lucifron",0x10a48,0x0,403,"Lightning Bolt",0x8,850,0,8,0,0,0,nil,nil,nil',
            '11/25/2026 18:00:04.000  SPELL_HEAL,Player-2,"Anduin",0x514,0x0,Player-1,"Thrall",0x511,0x0,2060,"Greater Heal",0x2,1200,300,0,nil',
            '11/25/2026 18:00:05.000  SPELL_DAMAGE,Creature-1,"Lucifron",0x10a48,0x0,Player-1,"Thrall",0x511,0x0,19702,"Impending Doom",0x20,2500,0,32,0,0,0,nil,nil,nil',
            '11/25/2026 18:00:06.000  UNIT_DIED,0000000000000000,nil,0x80000000,0x80000000,Player-1,"Thrall",0x511,0x0,0',
            '11/25/2026 18:00:10.000  ENCOUNTER_END,663,"Lucifron",9,40,1',
        ]
        events = [e for line in sample_lines if (e := parse_combatlog_line(line)) is not None]
        summary = analyze_combat_events(events)
        self.assertEqual(summary.name, "Lucifron")
        self.assertTrue(summary.success)
        self.assertEqual(summary.players["Thrall"].damage_done, 850)
        self.assertEqual(summary.players["Thrall"].deaths, 1)
        self.assertEqual(summary.players["Anduin"].effective_healing, 900)
        self.assertAlmostEqual(summary.players["Anduin"].overheal_pct, 25.0, places=1)
        self.assertEqual(len(summary.death_recaps), 1)
        self.assertEqual(summary.death_recaps[0].player_name, "Thrall")

    def test_synced_session_store_and_ghost_timeline(self):
        with tempfile.TemporaryDirectory() as td:
            live_store = SessionStore(Path(td) / "live.db")
            ghost_store = SessionStore(Path(td) / "ghost.db")

            live_store.record_position(PlayerState("p1", "Thrall", "SHAMAN", "HEALER", "ragefire_chasm", 10, 20, 0, (500, 500), t_host_ms=10000))
            live_store.record_position(PlayerState("p1", "Thrall", "SHAMAN", "HEALER", "ragefire_chasm", 25, 30, 0, (520, 510), t_host_ms=12000))
            live_store.record_video_frame(11500, "p1", "low", "frames/p1_11500.jpg")

            ghost_store.record_position(PlayerState("g1", "BestRun", "WARRIOR", "TANK", "ragefire_chasm", 10, 20, 0, (500, 500), t_host_ms=50000))
            ghost_store.record_position(PlayerState("g1", "BestRun", "WARRIOR", "TANK", "ragefire_chasm", 40, 45, 0, (550, 540), t_host_ms=52000))

            tl = TimelineController(live_store, ghost_store=ghost_store)
            snap = tl.scrub_to(12000)
            self.assertIn("p1", snap.players)
            self.assertEqual(snap.players["p1"].map_px, (520, 510))
            self.assertEqual(snap.video_frames_by_player["p1"], "frames/p1_11500.jpg")
            self.assertIsNotNone(snap.ghost_player)
            self.assertEqual(snap.ghost_player.map_px, (550, 540))

            live_store.close()
            ghost_store.close()

    def test_sync_preferences_roundtrip(self):
        p = SyncPreferences(sync_position=True, poll_hz=2.0, sync_video=True, video_quality="medium", video_mode="post_raid")
        p2 = SyncPreferences.from_dict(p.to_dict())
        self.assertEqual(p, p2)

    @unittest.skipUnless((INDEX_DIR / "manifest.json").exists(), "Index not present")
    def test_navigator_index_carried_over(self):
        t = MultiDungeonTracker(INDEX_DIR)
        self.assertGreaterEqual(len(t.dungeons), 28)
        self.assertIn("ragefire_chasm", t.dungeons)


if __name__ == "__main__":
    unittest.main()
