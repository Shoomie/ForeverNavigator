# ForeverNavigator — System Architecture & Technical Evaluation

## 1. Vision & Core Capabilities

**ForeverNavigator** evolves the standalone `MapNavigate` dungeon/raid localizer into a **collaborative, ToS-safe raid coordination, tactical whiteboard, combat analytics, and synced multi-track 3D replay suite** for *World of Warcraft: Forever* (and Classic/Era).

It integrates three complementary pillars:
1. **Live Guild 3D/2D Command Map**: Every participating raider runs the lightweight local tracker and streams their `(dungeon, x, y, z)` position to an **embedded host server** run directly inside the Raid Leader's (or assigned Officer's) ForeverNavigator app. All participants see each other moving live on the 2D `1024×1024` native minimap or the GPU-rendered 3D `.wmo` model.
2. **Tactical Planning & Measurement Tools**: Click-and-drag **yard ruler**, preset **spell/heal/spread range circles** (`8 yd`, `10 yd`, `30 yd`, `40 yd`), and a **shared collaborative whiteboard** (freehand drawing, arrows, raid target icons, and text callouts) synced live across the raid.
3. **Synced Multi-Track Replay, Extensive Combat Analytics & 3D Ghost-Runs**: Time-synchronized recording and scrubbing of **(a) all raiders' 3D positions**, **(b) full parsed `WoWCombatLog` statistics & death recaps**, **(c) opt-in per-raider video streams/buffers**, and **(d) previous guild "ghost run" trajectories** for route & pace comparison.

---

## 2. Topology & Networking Evaluation

### Embedded Host Server (Zero Cloud Hosting Required)
To avoid mandatory cloud infrastructure costs, ForeverNavigator embeds an `asyncio` WebSocket + HTTP server (`src/net/server.py`) directly inside the desktop application:
- **Host (Raid Leader / Officer)**: Clicks **Host Session**, picks a port (default `8765`), sets a room passphrase, and optionally designates Officer tokens.
- **Participants (Raiders)**: Enter the host address + passphrase (works over LAN, port-forwarding, or zero-config mesh VPNs like **Tailscale / ZeroTier / Hamachi**).
- **Optional Future Relay**: Because the protocol is pure WebSocket JSON/binary frames, the exact same server module can later be deployed headlessly on a \$5/month VPS if a guild doesn't want to port-forward or use Tailscale.

### Bandwidth & CPU Budget Analysis (40-Man Raid)

| Data Stream | Per-Raider Upload | Host Download (40 Raiders) | Host Broadcast to Raiders | Recommended Delivery Mode |
| :--- | :---: | :---: | :---: | :--- |
| **Position `(x, y, z, dungeon)`** @ 1 Hz | ~70 B/s | ~2.8 KB/s | ~110 KB/s total (~2.8 KB/s per raider) | **Realtime** (togglable `0.5–3 Hz`) |
| **Whiteboard / Ruler / Pings** | < 0.5 KB/s burst | < 2 KB/s burst | < 2 KB/s per raider | **Realtime** |
| **Combat Log (Live Summary)** | ~0.5 KB/s | ~20 KB/s | Optional | **Realtime or Out-of-Combat** |
| **Combat Log (Full Raw/Gzipped)** | 1–5 MB / raid | 40–200 MB / raid | Not rebroadcast (host stores) | **After `/reload` or Post-Raid** |
| **Video — Low (`360p @ 1 fps` JPEG)** | ~18–25 KB/s | ~0.8–1.0 MB/s | Only to Leader/Officers scrubbing | **Realtime OR Out-of-Combat** |
| **Video — Medium (`540p @ 5 fps`)** | ~90–140 KB/s | ~3.5–5.5 MB/s | On-demand officer view | **Buffered -> Out-of-Combat / Post-Raid** |
| **Video — High (`720p @ 15 fps`)** | ~300–500 KB/s | ~12–20 MB/s | On-demand officer view | **Buffered -> Post-Raid Only** |

> **Key Engineering Takeaway:** Running **positions + whiteboard in realtime** uses negligible CPU (~1.5% of 1 core at 1 Hz) and negligible bandwidth (~3 KB/s). For **video**, the winning design is a **local rolling ring buffer** (e.g., last 90–180s in RAM/temp disk) that **only uploads out-of-combat or post-raid** by default, with an explicit opt-in for realtime low-FPS (`360p @ 1 fps`) streaming. This guarantees zero FPS drops or latency spikes during boss pulls on lower-spec PCs.

---

## 3. Granular Privacy & Opt-In Design

Every participant controls an explicit `SyncPreferences` profile before and during a session:
- `sync_position`: On / Off + polling rate (`0.5 / 1.0 / 2.0 / 3.0 Hz`)
- `sync_video`: Off / Realtime (Low) / Buffered Out-of-Combat / Buffered Post-Raid + quality preset (`low`, `medium`, `high`)
- `sync_combatlog`: Off / Live Encounter Summary / Full Log After `/reload` or Raid End
- `allow_whiteboard_draw`: Controlled by Host (`Leader Only`, `Leader + Officers`, `All Raiders`)

No data type is ever captured or transmitted unless explicitly opted in by the user.

---

## 4. Time Synchronization Across 40 PCs

Raiders' Windows system clocks can drift by several seconds, which would ruin synced replay scrubbing. ForeverNavigator solves this with an **NTP-style handshake** (`clock_ping` / `clock_pong`):
1. On connect (and every 60s), each client sends `t0 = local_ms`.
2. Host replies with `(t0, host_rx_ms, host_tx_ms)`.
3. Client computes `rtt = (t1 - t0) - (host_tx - host_rx)` and `clock_offset = host_rx - t0 - rtt/2`.
4. All position samples, video frame timestamps, and combat log events are tagged in **Host Canonical Epoch Milliseconds** (`t_host = t_local + clock_offset`), achieving **< 50–100 ms alignment** across all 40 raiders.

---

## 5. Multi-Floor & Navigation Robustness

- **Baseline Localization (`src/navigator/`)**: Native `1024×1024` Blizzard minimap tiles (`World/Minimaps/` & `World/Minimaps/WMO/`) with exact North-Up orientation, 3-channel RGB + edge matching, **motion-plausibility radius** ($R = \max(24, 18\Delta t + 12)$ yd), and **2-frame teleport/summon confirmation**.
- **GPU 3D Model View**: On-demand ModernGL `.wmo` + `.blp` rendering directly from each user's local WoW folder (zero Blizzard asset redistribution) with ceiling cutaway and Z-floor continuity.
- **Future Google Colab A100 Enhancements**:
  1. **Sim-to-Real 3D Visual Relocalizer**: Distill `DINOv2` -> `MobileNetV3-Small` INT8 ONNX to resolve floor-level ambiguity in stacked multi-floor raids (*Naxxramas, Blackrock Spire, Sunken Temple, Gnomeregan*).
  2. **Automatic Wipe / Mechanic Bookmarker**: Lightweight vision tagger trained on synthetic WoW frames to auto-tag timeline moments (*raid-wide AoE, wipe pull, boss enrage*) for instant officer scrubbing.

---

## 6. Blizzard ToS Safety Invariants

All ForeverNavigator modules must strictly uphold four non-negotiable invariants:
1. **Passive External Observation Only**: Uses standard OS screen capture (`mss`) and standard text log files (`WoWCombatLog-*.txt`).
2. **Zero Memory Access / Injection**: Never reads/writes `Wow.exe` memory or hooks DirectX/OpenGL.
3. **Zero Input Automation**: Never sends synthetic keystrokes, clicks, or movement commands.
4. **Zero Hidden Information**: Never reveals stealth, unrendered mobs, or server-only state.
