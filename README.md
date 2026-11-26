# ForeverNavigator

A lightweight, ToS-safe, collaborative **2D/3D Raid Navigator, Tactical Whiteboard, Combat Log Analytics, and Synced Multi-Track Replay** framework for *World of Warcraft: Forever* (and Classic/Era).

Built on top of the `MapNavigate` 28-dungeon native `1024×1024` minimap & GPU 3D WMO engine.

## Core Architecture & Documentation
* **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** — Full technical evaluation (40-man bandwidth & CPU budgets, embedded host topology, NTP clock sync, privacy opt-in model, Colab A100 roadmap, ToS invariants).
* **[`docs/PROTOCOL.md`](docs/PROTOCOL.md)** — WebSocket JSON/binary protocol specification (`hello`, `pos_update`, `raid_state`, `draw_op`, `video_frame`, `combatlog_chunk`).
* **[`docs/ROADMAP.md`](docs/ROADMAP.md)** — Phased implementation plan (Phase 1–4).
* **[`docs/LLM_CONTEXT.md`](docs/LLM_CONTEXT.md)** — High-signal onboarding guide for future LLM-assisted development sessions.

## Quick Start

### Windows
Double-click **`Start_ForeverNavigator.bat`** (auto-installs lightweight runtime dependencies from `requirements-runtime.txt` into `.venv` on first launch).

### Linux / macOS
```bash
chmod +x start_forever_navigator.sh
./start_forever_navigator.sh
```

## Included Skeleton Capabilities
- **Solid 2D/3D Navigator (`src/navigator/`)**: 28 `1024×1024` North-Up WoW: Forever dungeon & raid maps, GPU 3D `.wmo` model view, motion-plausibility guard, 2-frame teleport confirmation.
- **Tactical Map Tools (`src/tactics/`)**: Interactive yard **Ruler**, preset **Range Circles** (`8 / 30 / 40 yd`), and collaborative **Whiteboard** (freehand drawing, text notes, undo/clear).
- **Embedded Host Server & Client (`src/net/`)**: Any Raid Leader or Officer can click **`Raid Sync...`** → **`Host Session`**; raiders join with granular opt-in toggles for Position, Video (quality + realtime/out-of-combat/post-raid), and Combat Logs.
- **Extensive Combat Log Analytics (`src/combatlog/`)**: Parses `WoWCombatLog-*.txt` into DPS, HPS, Overhealing %, Interrupts, Dispels, Damage Taken by Spell, and 8-second Death Recaps.
- **Synced Multi-Track Session Store & Timeline Scrubber (`src/replay/`)**: SQLite-backed recording of positions, combat events, video frame references, and Ghost-Run pace comparison.

## Running Tests
```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```
