"""SQLite-backed multi-track session store (positions, combat events, video frames, whiteboard, ghost runs)."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..combatlog.parser import CombatEvent
from ..net.protocol import PlayerState


class SessionStore:
    """Persists time-synchronized raid tracks indexed by Host UTC epoch milliseconds."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        # WAL + NORMAL synchronous eliminates 7-20 ms per-commit NTFS fsync stalls on Windows
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS positions (
                t_ms INTEGER NOT NULL,
                player_id TEXT NOT NULL,
                name TEXT NOT NULL,
                wow_class TEXT NOT NULL,
                role TEXT NOT NULL,
                dungeon_slug TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                map_px_x INTEGER NOT NULL,
                map_px_y INTEGER NOT NULL,
                confidence REAL NOT NULL,
                alive INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_pos_t ON positions(t_ms);

            CREATE TABLE IF NOT EXISTS combat_events (
                t_ms INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                source_name TEXT NOT NULL,
                dest_name TEXT NOT NULL,
                spell_name TEXT NOT NULL,
                amount INTEGER NOT NULL,
                overheal INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ce_t ON combat_events(t_ms);

            CREATE TABLE IF NOT EXISTS video_frames (
                t_ms INTEGER NOT NULL,
                player_id TEXT NOT NULL,
                quality TEXT NOT NULL,
                file_path TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_vf_t ON video_frames(t_ms, player_id);

            CREATE TABLE IF NOT EXISTS whiteboard_ops (
                t_ms INTEGER NOT NULL,
                dungeon_slug TEXT NOT NULL,
                op_json TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def record_position(self, st: PlayerState) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO positions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    st.t_host_ms, st.player_id, st.name, st.wow_class, st.role,
                    st.dungeon_slug, st.x, st.y, st.z, st.map_px[0], st.map_px[1],
                    st.confidence, int(st.alive),
                ),
            )
            self._conn.commit()

    def record_combat_event(self, ev: CombatEvent) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO combat_events VALUES (?,?,?,?,?,?,?)",
                (ev.timestamp_ms, ev.event_type, ev.source_name, ev.dest_name, ev.spell_name, ev.amount, ev.overheal),
            )
            self._conn.commit()

    def record_video_frame(self, t_ms: int, player_id: str, quality: str, file_path: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO video_frames VALUES (?,?,?,?)",
                (t_ms, player_id, quality, file_path),
            )
            self._conn.commit()

    def time_bounds_ms(self) -> Tuple[int, int]:
        with self._lock:
            row = self._conn.execute("SELECT MIN(t_ms), MAX(t_ms) FROM positions").fetchone()
        if row and row[0] is not None and row[1] is not None:
            return int(row[0]), int(row[1])
        return 0, 0

    def query_positions_at(self, t_ms: int, window_ms: int = 2500) -> Dict[str, PlayerState]:
        """Return each player's latest recorded position at or just before t_ms."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT p.t_ms, p.player_id, p.name, p.wow_class, p.role, p.dungeon_slug,
                       p.x, p.y, p.z, p.map_px_x, p.map_px_y, p.confidence, p.alive
                FROM positions p
                WHERE p.t_ms BETWEEN ? AND ?
                ORDER BY p.t_ms ASC
                """,
                (t_ms - window_ms, t_ms),
            ).fetchall()
        out: Dict[str, PlayerState] = {}
        for r in rows:
            out[r[1]] = PlayerState(
                t_host_ms=r[0], player_id=r[1], name=r[2], wow_class=r[3], role=r[4],
                dungeon_slug=r[5], x=r[6], y=r[7], z=r[8], map_px=(r[9], r[10]),
                confidence=r[11], alive=bool(r[12]),
            )
        return out

    def query_combat_window(self, t_ms: int, window_ms: int = 4000) -> List[Tuple[int, str, str, str, str, int]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT t_ms, event_type, source_name, dest_name, spell_name, amount
                FROM combat_events
                WHERE t_ms BETWEEN ? AND ?
                ORDER BY t_ms ASC
                """,
                (t_ms - window_ms, t_ms),
            ).fetchall()
        return [(int(r[0]), r[1], r[2], r[3], r[4], int(r[5])) for r in rows]

    def query_video_frame_at(self, t_ms: int, player_id: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT file_path FROM video_frames
                WHERE player_id = ? AND t_ms <= ?
                ORDER BY t_ms DESC LIMIT 1
                """,
                (player_id, t_ms),
            ).fetchone()
        return str(row[0]) if row else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()
