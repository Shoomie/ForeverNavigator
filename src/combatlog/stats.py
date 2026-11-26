"""Extensive raid & dungeon combat log analytics (DPS, HPS, Overheal %, Damage Taken, Death Recaps)."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from .parser import CombatEvent


@dataclass
class DeathRecap:
    timestamp_ms: int
    player_name: str
    recent_events: List[CombatEvent] = field(default_factory=list)  # last 8 seconds before death


@dataclass
class PlayerCombatStats:
    name: str
    damage_done: int = 0
    healing_done: int = 0
    effective_healing: int = 0
    overhealing: int = 0
    damage_taken: int = 0
    interrupts: int = 0
    dispels: int = 0
    deaths: int = 0
    damage_taken_by_spell: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    damage_done_by_spell: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def overheal_pct(self) -> float:
        return (100.0 * self.overhealing / self.healing_done) if self.healing_done > 0 else 0.0


@dataclass
class EncounterSummary:
    name: str
    start_ms: int
    end_ms: int
    success: bool
    players: Dict[str, PlayerCombatStats] = field(default_factory=dict)
    death_recaps: List[DeathRecap] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        return max(1.0, (self.end_ms - self.start_ms) / 1000.0)

    def dps_leaderboard(self) -> List[tuple[str, float, int]]:
        dur = self.duration_sec
        rows = [(p.name, p.damage_done / dur, p.damage_done) for p in self.players.values() if p.damage_done > 0]
        return sorted(rows, key=lambda r: -r[1])

    def hps_leaderboard(self) -> List[tuple[str, float, float]]:
        dur = self.duration_sec
        rows = [
            (p.name, p.effective_healing / dur, p.overheal_pct)
            for p in self.players.values()
            if p.healing_done > 0
        ]
        return sorted(rows, key=lambda r: -r[1])


def analyze_combat_events(events: Iterable[CombatEvent], session_name: str = "Raid Session") -> EncounterSummary:
    """Aggregate an iterable of CombatEvents into an extensive EncounterSummary."""
    players: Dict[str, PlayerCombatStats] = {}
    recent_incoming: Dict[str, deque[CombatEvent]] = defaultdict(lambda: deque(maxlen=20))
    death_recaps: List[DeathRecap] = []
    start_ms = 0
    end_ms = 0
    success = False

    def get_p(name: str) -> PlayerCombatStats:
        if name not in players:
            players[name] = PlayerCombatStats(name=name)
        return players[name]

    for ev in events:
        if ev.timestamp_ms > 0:
            if start_ms == 0 or ev.timestamp_ms < start_ms:
                start_ms = ev.timestamp_ms
            if ev.timestamp_ms > end_ms:
                end_ms = ev.timestamp_ms

        if ev.event_type == "ENCOUNTER_START" and ev.source_name:
            session_name = ev.source_name
        elif ev.event_type == "ENCOUNTER_END":
            success = ev.amount == 1
        elif ev.event_type in ("SWING_DAMAGE", "SPELL_DAMAGE", "SPELL_PERIODIC_DAMAGE", "RANGE_DAMAGE"):
            if ev.source_name and ev.source_guid.startswith("Player-"):
                p = get_p(ev.source_name)
                p.damage_done += ev.amount
                p.damage_done_by_spell[ev.spell_name or "Melee"] += ev.amount
            if ev.dest_name and ev.dest_guid.startswith("Player-"):
                dp = get_p(ev.dest_name)
                dp.damage_taken += ev.amount
                dp.damage_taken_by_spell[ev.spell_name or "Melee"] += ev.amount
                recent_incoming[ev.dest_name].append(ev)
        elif ev.event_type in ("SPELL_HEAL", "SPELL_PERIODIC_HEAL"):
            if ev.source_name and ev.source_guid.startswith("Player-"):
                p = get_p(ev.source_name)
                eff = max(0, ev.amount - ev.overheal)
                p.healing_done += ev.amount
                p.effective_healing += eff
                p.overhealing += ev.overheal
            if ev.dest_name and ev.dest_guid.startswith("Player-"):
                recent_incoming[ev.dest_name].append(ev)
        elif ev.event_type == "SPELL_INTERRUPT" and ev.source_guid.startswith("Player-"):
            get_p(ev.source_name).interrupts += 1
        elif ev.event_type == "SPELL_DISPEL" and ev.source_guid.startswith("Player-"):
            get_p(ev.source_name).dispels += 1
        elif ev.event_type == "UNIT_DIED" and ev.dest_guid.startswith("Player-"):
            get_p(ev.dest_name).deaths += 1
            cutoff = ev.timestamp_ms - 8000
            recap = [e for e in recent_incoming[ev.dest_name] if e.timestamp_ms >= cutoff]
            death_recaps.append(DeathRecap(ev.timestamp_ms, ev.dest_name, recap))

    return EncounterSummary(
        name=session_name,
        start_ms=start_ms,
        end_ms=end_ms,
        success=success,
        players=players,
        death_recaps=death_recaps,
    )
