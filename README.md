# ForeverNavigator

> **Note:** This project is in active early development. Expect occasional quirks and rough edges — feedback and bug reports are very welcome!

A **ToS-safe**, live **2D/3D Map Navigator, Tactical Whiteboard, Combat Log Analytics, and Raid Sync** companion app for *World of Warcraft: Forever* (and Classic/Era).

---

## For Players (Quick Overview)

**What it does:**
- **Live Map & Position Tracking:** Watches your WoW minimap on screen and shows your exact position and facing direction on high-resolution dungeon, raid, city, and outdoor world maps — plus an optional **3D dungeon view**.
- **Works in Dungeons, Raids & Outdoors:** Includes all 28 WoW: Forever dungeons & raids (`1024×1024` native maps) and outdoor zones, cities, battlegrounds, and caves loaded directly from your local WoW installation.
- **Tactical Tools & Raid Sync:** Measure yard distances with the **Ruler**, drop **Range Circles** (`8 / 30 / 40 yd`), draw on a shared **Tactical Whiteboard**, and optionally sync positions & drawings live with your party or raid (no external server needed — your Raid Leader clicks **Host Session**).
- **Combat Log Stats:** Load any `WoWCombatLog-*.txt` file to view DPS, HPS, Overhealing %, Interrupts, Dispels, and 8-second Death Recaps.

**100% ToS-Safe:**
ForeverNavigator only looks at your screen like OBS/Discord screen-share and reads standard WoW combat log / static map files. It **never** reads game memory, injects into `Wow.exe`, or sends any clicks or keystrokes.

### Quick Start

**Windows:**
1. Double-click **`Start_ForeverNavigator.bat`** (automatically sets up `.venv` and lightweight dependencies on first launch).
2. Click **`Pick Minimap...`** and drag a box around your WoW minimap (works cleanly at 100%–200% Windows DPI scaling).
3. Click **`Start Capture`**. Use the **Polling** dropdown (`1–60 Hz`), **Model** selector, and **View: 2D / 3D** button as needed.

**Linux / macOS:**
```bash
chmod +x start_forever_navigator.sh
./start_forever_navigator.sh
```

---

## For Technical Users (Architecture Summary)

ForeverNavigator combines a lightweight **Hybrid Neural ONNX + OpenCV localizer** with a **GPU-accelerated PySide6 + ModernGL** 2D/3D desktop frontend and an **embedded `asyncio` WebSocket raid server**:

- **Hybrid Neural + Geometric Localization (`src/navigator/neural_localizer.py`, `src/navigator/minimap_tracker.py`):**
  - **ONNX Multi-Task Network (`data/models/`):** Trained on a globally-shuffled, full-resolution synthetic dataset (`1024×1024` dungeons/raids + up to `4096×4096` CASC outdoor mosaics on Colab A100-80GB). Predicts map ID across 82 classes, normalized `(u, v)` position, minimap auto-centering `(cx, cy, r)`, and 360° player arrow heading in **~4–8 ms on CPU**.
  - **Live Model Switching:** Keeps both `forever_minimap_v2_fast.onnx` (v2, default) and `forever_minimap_fast.onnx` (v1) selectable live from the UI, and auto-discovers any additional `*.onnx` + `*_meta.json` models (such as `ConvNeXt-Tiny` or tuned variants).
  - **Geometric Refinement & Guardrails:** Tight local-patch OpenCV refinement (~1 yd precision), optional periodic full-map OpenCV anchor (`Anchor` toggle), cave/tunnel uncertainty hold, geographic continuity guard, motion-plausibility filter, and 2-frame teleport confirmation.
- **Fast Screen Capture (`src/navigator/fast_capture.py`):** Uses Windows DXGI Desktop Duplication (`dxcam`, ~1–2 ms grab) for high-frequency `30–60 Hz` polling, with automatic `mss` fallback on Linux/sandboxed environments.
- **GPU 2D/3D Renderer (`src/ui/qt_app.py`, `src/navigator/gpu_viewer3d.py`):** PySide6 + ModernGL viewport with a dedicated transparent overlay widget (`_MapOverlayWidget`) and shared-context 3D `.wmo` / `.blp` rendering (ceiling cutaway, async CPU scene prep, local CASC reading — zero Blizzard assets redistributed).
- **Embedded Raid Sync & Replay (`src/net/`, `src/tactics/`, `src/combatlog/`, `src/replay/`):** WebSocket host/client with NTP clock-offset alignment, granular privacy opt-ins (`SyncPreferences`), normalized `[0..1]` whiteboard sync, SQLite session store, and Ghost-Run comparison.
- **Offline Calibration & LLM Model-Tuning Pipeline (`src/navigator/calibration_pipeline.py`, `src/navigator/model_tuning_pipeline.py`):** Replays recorded gameplay holdout logs (`data/dev_gameplay/`) + synthetic ground-truth suites to compare models head-to-head, inspect ONNX head weights/calibration (ECE & optimal temperature `T*`), and apply non-destructive surgical ONNX head patches.

---

## Detailed Documentation
- **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** — System architecture, 40-man bandwidth/CPU budgets, hybrid neural + OpenCV design, NTP clock sync, and ToS invariants.
- **[`docs/PROTOCOL.md`](docs/PROTOCOL.md)** — WebSocket JSON/binary protocol specification (`hello`, `pos_update`, `raid_state`, `draw_op`, `video_frame`, `combatlog_chunk`).
- **[`docs/ROADMAP.md`](docs/ROADMAP.md)** — Completed milestones and upcoming phases.
- **[`docs/LLM_CONTEXT.md`](docs/LLM_CONTEXT.md)** — Concise onboarding reference, coordinate/GL invariants, and CLI commands for LLM-assisted development.

---

## Running Tests & Benchmarks
```powershell
# Full unit & UI smoke test suite
.venv\Scripts\python.exe -m unittest discover -s tests -v

# Head-to-head ONNX model analysis & calibration dossier
.venv\Scripts\python.exe -m src.navigator.model_tuning_pipeline --compare --stride 4

# Single-model gameplay replay & synthetic calibration report
.venv\Scripts\python.exe -m src.navigator.calibration_pipeline --stride 2
```
