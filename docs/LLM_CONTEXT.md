# ForeverNavigator — LLM-Assisted Development Context Guide

This file is designed as a fast, high-signal onboarding reference for future LLM coding sessions working on **ForeverNavigator**.

## 1. Project Identity & Constraints
- **Target Game**: World of Warcraft: Forever (`wowf`, Classic-era CASC format), also compatible with Classic Era.
- **Strict ToS Boundary**: Passive external screen capture (`mss`) + reading standard `WoWCombatLog-*.txt` + reading static local CASC map files only. **Never** read game memory, inject DLLs, or automate inputs.
- **Performance Target**: Must run smoothly alongside WoW on mid/low-spec gaming PCs (~1–3% of 1 CPU core at 1 Hz in 2D mode; 0% GPU unless 3D WMO mode is toggled on).

## 2. Package Architecture (`src/`)
| Package | Status | Responsibility |
| :--- | :---: | :--- |
| `src/navigator/` | **Production-ready** | Carried over from `MapNavigate`: CASC reader, BLP2 decoder, WMO parser, 28-map `1024×1024` North-Up `MinimapTracker` (with motion plausibility & 2-frame teleport confirmation), `GPU3DViewer` (ModernGL), `spawns.py`, `settings.py`. |
| `src/net/` | **Skeleton + Protocol** | `protocol.py` (typed dataclasses for `SyncPreferences`, `PlayerState`, `WhiteboardOp`, messages), `server.py` (embedded `asyncio` WebSocket Host Server), `client.py` (Raider Client with granular opt-in). |
| `src/tactics/` | **Working Core** | `tools.py` (yard distance ruler & range circle geometry), `whiteboard.py` (collaborative strokes/arrows/circles/text with undo & JSON sync). |
| `src/combatlog/` | **Working Core** | `parser.py` (parses WoW `WoWCombatLog-*.txt` encounters & events), `stats.py` (computes DPS, HPS, overheal %, damage taken by spell, interrupts, dispels, death recaps). |
| `src/replay/` | **Working Core** | `session_store.py` (SQLite/JSONL synced multi-track session recorder & loader), `timeline.py` (time-indexed scrubber query for positions + combat events + video frame lookup + Ghost-Run comparison). |
| `src/ui/` | **Runnable Skeleton** | `app.py` — Adapted `MapNavigate` GUI extended with Ruler tool, Whiteboard drawing/text tool, Host/Join Session dialog, Combat Stats summary dialog, and Timeline Scrubber bar. |

## 3. Key Coordinate & Data Invariants (Do Not Break!)
1. **WMO North-Up Rotation**: All 16 WMO interior dungeons use `map_rot90 = 3` (`cv2.ROTATE_90_CLOCKWISE` from raw WMO coordinates). Always convert between world yards and map pixels using `DungeonData.world_to_map_px()` and `DungeonData.map_px_to_world()`—never compute `(x - ox) * scale` manually!
2. **Normalized Whiteboard Coordinates**: All `WhiteboardOp` points are stored in `[0.0 .. 1.0]` normalized map space so drawings align across different window sizes, zoom levels, and 512/1024 map resolutions.
3. **Canonical Timestamps**: All replay tracks (`positions`, `combat_events`, `video_frames`) use integer **Host UTC epoch milliseconds** (`t_host_ms`).
4. **Local WoW Assets**: Never commit raw `.wmo` or `.blp` files or `data/listfile.csv` to Git. `GPU3DViewer` reads them on demand from the user's local WoW folder via `find_wow_dir()`.

## 4. Development & Verification Commands
```powershell
# Run all automated tests
.venv\Scripts\python.exe -m unittest discover -s tests -v

# Launch ForeverNavigator UI
Start_ForeverNavigator.bat
# or:
.venv\Scripts\python.exe -m src.ui.app
```
