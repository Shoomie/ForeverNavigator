# ForeverNavigator — Implementation Roadmap

## Phase 1: Solid Core, GPU UI & Hybrid Neural Localization *(Completed)*
- [x] Native `1024×1024` 28-dungeon/raid index + full WoW: Forever CASC outdoor/city/BG/cave catalog (`src/navigator/world_maps.py`).
- [x] GPU-accelerated PySide6 + ModernGL 2D/3D desktop app (`src/ui/qt_app.py`) with DPI-aware ROI picker, Azeroth overview, and shared-context 3D `.wmo` viewer (`src/navigator/gpu_viewer3d.py`).
- [x] **Hybrid Neural ONNX + OpenCV Localizer** (`src/navigator/neural_localizer.py`, `src/navigator/minimap_tracker.py`) with v1 & v2 `MobileNetV3-Large` ONNX models, live UI model switcher, 360° directional player arrow, auto-centering, togglable periodic OpenCV anchor, cave/tunnel hold, and geographic continuity guard.
- [x] **Windows DXGI Fast Capture** (`src/navigator/fast_capture.py` via `dxcam` + `mss` fallback) supporting up to `60 Hz` polling.
- [x] **Offline Calibration & LLM Surgical Tuning Pipelines** (`src/navigator/calibration_pipeline.py`, `src/navigator/model_tuning_pipeline.py`).
- [x] Interactive **Distance Ruler**, **Range Circles** (`8 / 10 / 30 / 40 yd`), **Collaborative Whiteboard**, **Combat Log Analytics**, and **SQLite Session Store**.

## Phase 2: Live Multi-Raider Sync & High-Capacity Models *(Mostly Complete)*
- [x] Embedded `asyncio` WebSocket Host Server & Raider Client (`src/net/server.py`, `src/net/client.py`) wired into `src/ui/qt_app.py`.
- [x] Live rendering of connected guildmates on the 2D/3D map with class colors, names, and role info.
- [x] Real-time `draw_op` whiteboard sync with Host permission gating (`Leader`, `Officers`, `All`).
- [x] NTP clock-offset synchronization (`clock_ping` / `clock_pong`) canonicalizing timestamps to Host time.
- [ ] Integrate `forever_minimap_v2_accurate.onnx` (`ConvNeXt-Tiny`) and run LLM-guided surgical calibration/tuning passes on both v2 models.

## Phase 3: Opt-In Video Buffering, Combat Log Sync & Synced Timeline Scrubbing
- [ ] Implement local rolling video ring buffer (`src/replay/video_buffer.py`) with `low` / `medium` / `high` presets and automatic **Out-of-Combat** / **Post-Raid** chunked upload to Host.
- [ ] Auto-detect WoW `/reload` & combat-log flush to upload gzipped `WoWCombatLog-*.txt` chunks to Host and deduplicate across raiders.
- [ ] Complete the **Synced Replay Scrubber View**: scrubbing updates (1) all raiders' 2D/3D positions, (2) synchronized combat log events & death markers, and (3) side-by-side raider POV video frames simultaneously.

## Phase 4: Extensive Analytics, 3D Ghost-Runs & Multi-Floor Polish
- [ ] Interactive **Combat Statistics Dashboard** (per-encounter & full-raid tables: DPS, HPS, Overhealing, Avoidable Damage Taken, Interrupts, Dispels, Death Recap timeline).
- [ ] Full **3D Ghost-Run** recording & live pace-delta overlay.
- [ ] **Per-Floor WMO Group Layering** for stacked multi-story dungeons (*Sunken Temple, Gnomeregan, Blackrock Spire, Shadowfang Keep*).
