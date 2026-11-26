"""Streaming parser for World of Warcraft WoWCombatLog-*.txt files."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional


@dataclass
class CombatEvent:
    timestamp_ms: int
    event_type: str
    source_guid: str
    source_name: str
    dest_guid: str
    dest_name: str
    spell_id: int = 0
    spell_name: str = ""
    amount: int = 0
    overheal: int = 0
    overkill: int = 0
    extra_spell_name: str = ""


def _parse_wow_timestamp_ms(ts_str: str, year: int) -> int:
    """Parse WoW combat log timestamp like '11/25 18:12:04.512' or '11/25/2026 18:12:04.512'."""
    ts_str = ts_str.strip()
    for fmt in ("%m/%d/%Y %H:%M:%S.%f", "%m/%d %H:%M:%S.%f"):
        try:
            dt = datetime.strptime(ts_str, fmt)
            if dt.year == 1900:
                dt = dt.replace(year=year)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    return 0


def parse_combatlog_line(line: str, year: int = 2026) -> Optional[CombatEvent]:
    """Parse one WoWCombatLog line into a typed CombatEvent, or None if not tracked."""
    if "  " not in line:
        return None
    ts_part, rest = line.split("  ", 1)
    ts_ms = _parse_wow_timestamp_ms(ts_part, year)
    try:
        cols = next(csv.reader(io.StringIO(rest)))
    except Exception:
        return None
    if len(cols) < 1:
        return None

    ev = cols[0].strip()

    if ev in ("ENCOUNTER_START", "ENCOUNTER_END") and len(cols) >= 3:
        return CombatEvent(
            timestamp_ms=ts_ms,
            event_type=ev,
            source_guid="",
            source_name=cols[2].strip('"'),
            dest_guid="",
            dest_name="",
            spell_id=int(cols[1]) if cols[1].isdigit() else 0,
            amount=int(cols[5]) if ev == "ENCOUNTER_END" and len(cols) > 5 and cols[5].isdigit() else 0,
        )

    if len(cols) < 9:
        return None

    src_guid = cols[1]
    src_name = cols[2].strip('"')
    dst_guid = cols[5]
    dst_name = cols[6].strip('"')

    if ev == "UNIT_DIED":
        return CombatEvent(ts_ms, ev, src_guid, src_name, dst_guid, dst_name)

    if ev == "SWING_DAMAGE" and len(cols) >= 11:
        # Classic/Forever SWING_DAMAGE has amount at index 9 (or after advanced combat log block)
        amt, ovk = _extract_amount_pair(cols, default_idx=9)
        return CombatEvent(ts_ms, ev, src_guid, src_name, dst_guid, dst_name, 0, "Melee", amt, 0, ovk)

    if ev in ("SPELL_DAMAGE", "SPELL_PERIODIC_DAMAGE", "RANGE_DAMAGE") and len(cols) >= 13:
        spell_id = int(cols[9]) if cols[9].isdigit() else 0
        spell_name = cols[10].strip('"')
        amt, ovk = _extract_amount_pair(cols, default_idx=12)
        return CombatEvent(ts_ms, ev, src_guid, src_name, dst_guid, dst_name, spell_id, spell_name, amt, 0, ovk)

    if ev in ("SPELL_HEAL", "SPELL_PERIODIC_HEAL") and len(cols) >= 14:
        spell_id = int(cols[9]) if cols[9].isdigit() else 0
        spell_name = cols[10].strip('"')
        amt, ovh = _extract_amount_pair(cols, default_idx=12)
        return CombatEvent(ts_ms, ev, src_guid, src_name, dst_guid, dst_name, spell_id, spell_name, amt, ovh, 0)

    if ev in ("SPELL_INTERRUPT", "SPELL_DISPEL") and len(cols) >= 14:
        spell_id = int(cols[9]) if cols[9].isdigit() else 0
        spell_name = cols[10].strip('"')
        extra = cols[13].strip('"')
        return CombatEvent(
            ts_ms, ev, src_guid, src_name, dst_guid, dst_name, spell_id, spell_name, extra_spell_name=extra
        )

    return None


def _extract_amount_pair(cols: List[str], default_idx: int) -> tuple[int, int]:
    """Handle both standard and Advanced Combat Logging (which inserts 16 extra unit-state columns)."""
    # In Advanced Combat Logging, extra columns start with a GUID at default_idx
    idx = default_idx
    if idx < len(cols) and ("-" in cols[idx] or cols[idx].startswith("0x")):
        # Find the first integer block after the advanced unit state fields
        for probe in (default_idx + 16, default_idx + 17, default_idx + 18):
            if probe < len(cols) and cols[probe].lstrip("-").isdigit():
                idx = probe
                break
    amt = int(cols[idx]) if idx < len(cols) and cols[idx].lstrip("-").isdigit() else 0
    sec = int(cols[idx + 1]) if idx + 1 < len(cols) and cols[idx + 1].lstrip("-").isdigit() else 0
    return max(0, amt), max(0, sec)


def iter_combatlog_file(path: str | Path, year: int = 2026) -> Iterator[CombatEvent]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            ev = parse_combatlog_line(line, year=year)
            if ev is not None:
                yield ev
