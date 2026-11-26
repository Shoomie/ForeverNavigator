"""Synced multi-track timeline scrubber & 3D Ghost-Run pace comparator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..net.protocol import PlayerState
from .session_store import SessionStore


@dataclass
class ScrubSnapshot:
    t_ms: int
    players: Dict[str, PlayerState]
    recent_combat_events: List[Tuple[int, str, str, str, str, int]]
    video_frames_by_player: Dict[str, str]
    ghost_player: Optional[PlayerState] = None


class TimelineController:
    """Provides synchronized scrubbing over positions, combat logs, video frames, and optional ghost runs."""

    def __init__(self, store: SessionStore, ghost_store: Optional[SessionStore] = None):
        self.store = store
        self.ghost_store = ghost_store

    def bounds_ms(self) -> Tuple[int, int]:
        return self.store.time_bounds_ms()

    def scrub_to(self, t_ms: int) -> ScrubSnapshot:
        players = self.store.query_positions_at(t_ms)
        events = self.store.query_combat_window(t_ms)
        vids: Dict[str, str] = {}
        for pid in players:
            vp = self.store.query_video_frame_at(t_ms, pid)
            if vp:
                vids[pid] = vp

        ghost_player = None
        if self.ghost_store is not None:
            s0, _ = self.store.time_bounds_ms()
            g0, _ = self.ghost_store.time_bounds_ms()
            if s0 > 0 and g0 > 0:
                elapsed = max(0, t_ms - s0)
                g_players = self.ghost_store.query_positions_at(g0 + elapsed)
                if g_players:
                    ghost_player = next(iter(g_players.values()))

        return ScrubSnapshot(
            t_ms=t_ms,
            players=players,
            recent_combat_events=events,
            video_frames_by_player=vids,
            ghost_player=ghost_player,
        )
