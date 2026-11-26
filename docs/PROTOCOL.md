# ForeverNavigator — Network & Sync Protocol Specification (`v0.1`)

Transport: **WebSocket** (`ws://<host>:<port>/ws`) carrying UTF-8 JSON control/state messages and optional binary frames for chunked video/combat-log uploads.

## 1. Handshake & Clock Synchronization

### Client -> Host: `hello`
```json
{
  "type": "hello",
  "protocol_version": "0.1",
  "room_passphrase": "optional-shared-secret",
  "officer_token": "optional-officer-secret",
  "player": {
    "player_id": "uuid-v4",
    "name": "Thrall",
    "wow_class": "SHAMAN",
    "role": "HEALER"
  },
  "sync_prefs": {
    "sync_position": true,
    "poll_hz": 1.0,
    "sync_video": false,
    "video_quality": "low",
    "video_mode": "out_of_combat",
    "sync_combatlog": true,
    "combatlog_mode": "after_reload"
  }
}
```

### Host -> Client: `welcome`
```json
{
  "type": "welcome",
  "session_id": "2026-11-25_molten_core_a1b2",
  "assigned_role": "raider",
  "host_time_ms": 1795641234567,
  "whiteboard_permission": "officers",
  "participants": [],
  "whiteboard_ops": []
}
```

### Clock Sync: `clock_ping` / `clock_pong`
- Client sends `{"type": "clock_ping", "t0_ms": 1795641234500}`
- Host replies `{"type": "clock_pong", "t0_ms": 1795641234500, "host_rx_ms": ..., "host_tx_ms": ...}`

---

## 2. Live Position Telemetry

### Client -> Host: `pos_update`
Sent at each client's chosen `poll_hz` (0.5–3.0 Hz) only when `sync_position == true`:
```json
{
  "type": "pos_update",
  "player_id": "uuid-v4",
  "t_host_ms": 1795641235500,
  "dungeon_slug": "ragefire_chasm",
  "x": 142.5,
  "y": -64.2,
  "z": -18.0,
  "map_px": [685, 198],
  "confidence": 0.94,
  "alive": true
}
```

### Host -> All Clients: `raid_state`
Aggregated snapshot broadcast by Host at 1–2 Hz so all participants see each other smoothly:
```json
{
  "type": "raid_state",
  "t_host_ms": 1795641235600,
  "players": {
    "uuid-v4": {
      "name": "Thrall",
      "wow_class": "SHAMAN",
      "role": "HEALER",
      "dungeon_slug": "ragefire_chasm",
      "x": 142.5,
      "y": -64.2,
      "z": -18.0,
      "map_px": [685, 198],
      "confidence": 0.94,
      "alive": true
    }
  }
}
```

---

## 3. Collaborative Tactical Whiteboard

### Client <-> Host: `draw_op`
All coordinates are normalized `[0.0 .. 1.0]` relative to the dungeon's `1024×1024` North-Up map so annotations render identically at any window size or zoom level.
```json
{
  "type": "draw_op",
  "op_id": "op-uuid",
  "author_id": "uuid-v4",
  "dungeon_slug": "ragefire_chasm",
  "action": "add",
  "kind": "stroke",
  "color": "#ffe066",
  "width": 3,
  "points": [[0.42, 0.21], [0.45, 0.24], [0.48, 0.28]],
  "text": "",
  "radius_yd": 0.0
}
```
Supported `kind` values: `"stroke"`, `"arrow"`, `"circle_range"`, `"text"`, `"raid_icon"`, `"clear_all"`, `"undo"`.

---

## 4. Buffered / Realtime Video & Combat Log Sync

### Video Segment Header (`video_frame` / `video_chunk`)
- **Realtime Low (`360p @ 1 fps`)**: Base64 or binary JPEG frame tagged with `(player_id, t_host_ms, quality)`.
- **Buffered Out-of-Combat / Post-Raid**: Chunked sequence of JPEG keyframes or MP4/WebM segment uploaded when `in_combat == false` or on raid end, stored under `sessions/<session_id>/video/<player_id>/`.

### Combat Log Upload (`combatlog_chunk`)
- Triggered automatically when WoW flushes `WoWCombatLog-*.txt` (e.g., after `/reload`, encounter end, or raid end) if `sync_combatlog == true`.
- Sent as gzipped chunks (`chunk_index`, `total_chunks`, `sha256`, `payload_b64`) and merged/deduplicated on the Host into the canonical encounter statistics store.
