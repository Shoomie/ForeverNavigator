# ForeverNavigator — LLM-Assisted Development Context Guide

This file is a fast, high-signal onboarding reference for future LLM coding sessions working on **ForeverNavigator**.

## 1. Project Identity & Constraints
- **Target Game**: World of Warcraft: Forever (`wowf`, Classic-era CASC format), also compatible with Classic Era.
- **Strict ToS Boundary**: Passive external screen capture (`dxcam` / `mss`) + reading standard `WoWCombatLog-*.txt` + reading static local CASC map files only. **Never** read game memory, inject DLLs, or automate inputs.
- **Performance Target**: Runs smoothly alongside WoW on gaming PCs — ~4–8 ms CPU ONNX inference for 60 Hz dungeon/raid tracking; low background CPU/RAM; zero synthetic datasets or training notebooks committed to Git.

## 2. Package Architecture (`src/`)
| Package | Status | Responsibility |
| :--- | :---: | :--- |
| `src/navigator/` | **Production Core** | CASC reader, BLP2/WMO parsers, `WorldMapRegistry` (28 dungeons/raids + 54 outdoor zones/cities/BGs + caves/buildings), `NeuralMinimapLocalizer` (multi-model ONNX Runtime wrapper), `MinimapTracker` (hybrid ONNX + OpenCV local refinement, periodic anchor, cave/continuity guards), `FastScreenCapturer` (`dxcam` DXGI + `mss` fallback), `GPU3DViewer` (shared-context ModernGL), `calibration_pipeline.py`, `model_tuning_pipeline.py`, `spawns.py`, `settings.py`. |
| `src/ui/` | **Active GPU App** | `qt_app.py` (primary PySide6 + ModernGL GPU UI with `_MapOverlayWidget`, DPI-aware ROI picker, live model/polling/anchor/neural controls, 360° player arrow, 3D WMO view, Raid Sync dialog, Combat Stats dialog, Timeline Scrubber); `app.py` (legacy Tkinter fallback). |
| `src/net/` | **Working Live** | `protocol.py` (typed dataclasses for `SyncPreferences`, `PlayerState`, `WhiteboardOp`), `server.py` (embedded `asyncio` WebSocket Host Server + NTP clock sync), `client.py` (Raider Client with granular opt-in). |
| `src/tactics/` | **Working Core** | `tools.py` (yard distance ruler & range circle geometry), `whiteboard.py` (collaborative strokes/arrows/circles/text with undo & JSON sync). |
| `src/combatlog/` | **Working Core** | `parser.py` (parses WoW `WoWCombatLog-*.txt` encounters & events), `stats.py` (DPS, HPS, overheal %, damage taken by spell, interrupts, dispels, death recaps). |
| `src/replay/` | **Working Core** | `session_store.py` (SQLite synced multi-track session recorder & loader), `timeline.py` (time-indexed scrubber query + Ghost-Run comparison). |

## 3. Key Invariants (Do Not Break!)
1. **WMO North-Up Rotation**: All 16 WMO interior dungeons use `map_rot90 = 3` (`cv2.ROTATE_90_CLOCKWISE` from raw WMO coordinates). Always convert between world yards and map pixels using `DungeonData.world_to_map_px()` and `DungeonData.map_px_to_world()` — never compute `(x - ox) * scale` manually.
2. **Qt + ModernGL Isolation**: Never mix `QPainter` calls inside `GPUMapViewport.paintGL()`. All 2D overlays (player arrow, spawns, ruler, whiteboard, HUD) MUST render in the transparent child `_MapOverlayWidget`.
3. **Shared GL Context for 3D WMO**: `GPU3DViewer` must never call `create_standalone_context()` on the UI thread; prepare geometry off-thread via `prepare_scene_cpu()` and upload/render in `GPUMapViewport`'s shared context via `render_direct_gl()`.
4. **Multi-Model ONNX Convention**: All models in `data/models/` follow `<stem>.onnx` + `<stem>_meta.json` and are auto-discovered by `list_available_onnx_models()`. Never delete older models when adding new or tuned ones.
5. **Normalized Whiteboard & Canonical Timestamps**: `WhiteboardOp` points live in `[0.0 .. 1.0]` normalized map space; replay tracks use integer **Host UTC epoch milliseconds** (`t_host_ms`).
6. **Git Hygiene & Local WoW Assets**: Never commit raw `.wmo`/`.blp` files, `data/listfile.csv`, `data/dev_gameplay/`, `data/calibration/`, or synthetic training tar shards.

## 4. Development, Benchmark & Tuning Commands
```powershell
# Run all automated tests
.venv\Scripts\python.exe -m unittest discover -s tests -v

# Launch GPU ForeverNavigator UI
Start_ForeverNavigator.bat
# or:
.venv\Scripts\python.exe -m src.ui.qt_app

# Compare all ONNX models (synthetic ground truth + real gameplay holdout + weight anatomy)
.venv\Scripts\python.exe -m src.navigator.model_tuning_pipeline --compare --stride 4

# Apply a surgical ONNX head patch (temperature / class biases / reg offsets) and benchmark
.venv\Scripts\python.exe -m src.navigator.model_tuning_pipeline --patch-json patch.json --base forever_minimap_v2_fast --out forever_minimap_v2_fast_tuned
```
