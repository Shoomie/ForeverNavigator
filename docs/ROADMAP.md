# ForeverNavigator — Implementation Roadmap

## Phase 1: Solid Core & Local Tactical Tools *(Scaffolded & Runnable Now)*
- [x] Carry over hardened `MapNavigate` engine into `src/navigator/` (28 `1024×1024` North-Up maps, GPU 3D WMO viewer, motion-plausibility guard, 2-frame teleport confirmation).
- [x] Interactive **Distance Ruler** (click-and-drag yard measurement) & **Range Circles** (`8 / 10 / 30 / 40 yd`) in `src/tactics/tools.py`.
- [x] **Collaborative Whiteboard Model** (freehand strokes, arrows, range circles, text annotations, undo/clear, JSON serialization) in `src/tactics/whiteboard.py`.
- [x] **Combat Log Parser & Extensive Analytics Engine** (`DPS`, `HPS`, `Overheal %`, `Damage Taken by Ability`, `Interrupts`, `Dispels`, `Death Recaps`) in `src/combatlog/`.
- [x] **Synced Session Store & Timeline Scrubber Skeleton** (`src/replay/`) with Ghost-Run loader/comparator.
- [x] **Protocol & Embedded Host/Client Skeleton** (`src/net/`) with granular `SyncPreferences`.

## Phase 2: Live Multi-Raider Sync & Collaborative Whiteboard
- [ ] Wire `websockets` live event loop in `src/net/server.py` and `src/net/client.py` into a background `asyncio` thread inside `src/ui/app.py`.
- [ ] Render all connected guildmates live on both the **2D Map** and **GPU 3D WMO Model** (class-colored dots, names, role badges, dead/alive state).
- [ ] Broadcast `draw_op` whiteboard strokes/text/range circles in real time with Host permission gating (`Leader`, `Officers`, `All`).
- [ ] Add NTP clock-offset sync on connect so all timestamps are canonicalized to Host time.

## Phase 3: Opt-In Video Buffering, Combat Log Sync & Synced Timeline Scrubbing
- [ ] Implement local rolling video ring buffer (`src/replay/video_buffer.py`) with `low` / `medium` / `high` presets and automatic **Out-of-Combat** / **Post-Raid** chunked upload to Host.
- [ ] Auto-detect WoW `/reload` & combat-log flush to upload gzipped `WoWCombatLog-*.txt` chunks to Host and deduplicate across raiders.
- [ ] Build the **Synced Replay Scrubber View**: scrubbing the timeline updates (1) all raiders' 2D/3D positions, (2) synchronized combat log events & death markers, and (3) side-by-side raider POV video frames simultaneously.

## Phase 4: Extensive Analytics, 3D Ghost-Runs & Multi-Floor Polish
- [ ] Build full interactive **Combat Statistics Dashboard** (per-encounter & full-raid tables: DPS, HPS, Overhealing, Avoidable Damage Taken, Interrupts, Dispels, Death Recap timeline).
- [ ] Enable recording & loading **3D Ghost-Runs** (live moving ghost marker + ahead/behind pace timer).
- [ ] Add **Per-Floor WMO Group Layering** for multi-story dungeons (*Sunken Temple, Gnomeregan, Blackrock Spire, Shadowfang Keep*).
- [ ] Optional Google Colab A100 Sim-to-Real encoder fine-tuning for 3D visual relocalization.
