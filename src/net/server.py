"""Embedded WebSocket Host Server baked into ForeverNavigator.

Any Raid Leader or assigned Officer can start this server directly from the app so participating
guildmates can connect, share live positions, collaborate on the whiteboard, and upload opt-in
buffered video segments and combat logs.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .protocol import (
    PROTOCOL_VERSION,
    VALID_WHITEBOARD_PERMISSIONS,
    PlayerState,
    get_local_lan_ip,
    now_ms,
)


@dataclass
class HostConfig:
    bind_host: str = "0.0.0.0"
    port: int = 8765
    room_passphrase: str = ""
    officer_token: str = ""
    whiteboard_permission: str = "all"  # "leader", "officers", "all"
    ping_interval: float = 20.0
    ping_timeout: float = 20.0


class ForeverNavigatorServer:
    """Lightweight asyncio WebSocket server running in a background thread."""

    def __init__(
        self,
        config: HostConfig,
        on_raid_state: Optional[Callable[[Dict[str, PlayerState]], None]] = None,
        on_draw_op: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[bool, str], None]] = None,
    ):
        self.config = config
        if self.config.whiteboard_permission not in VALID_WHITEBOARD_PERMISSIONS:
            self.config.whiteboard_permission = "all"

        self.session_id = f"raid_{uuid.uuid4().hex[:8]}"
        self.assigned_role: str = "leader"
        self.local_ip: str = get_local_lan_ip()
        self.last_error: Optional[str] = None

        self.on_raid_state = on_raid_state
        self.on_draw_op = on_draw_op
        self.on_log = on_log
        self.on_status = on_status

        self.players: Dict[str, PlayerState] = {}
        self.whiteboard_ops: List[Dict[str, Any]] = []
        self.event_queue: "queue.SimpleQueue[Tuple[str, Any]]" = queue.SimpleQueue()

        self._clients: Dict[Any, Dict[str, Any]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._ready_event: Optional[threading.Event] = None
        self.is_running: bool = False

    @property
    def connected_count(self) -> int:
        """Number of currently connected remote clients."""
        return len(self._clients)

    def poll_events(self) -> List[Tuple[str, Any]]:
        """Drain queued network events safely from the UI thread."""
        events: List[Tuple[str, Any]] = []
        while True:
            try:
                events.append(self.event_queue.get_nowait())
            except queue.Empty:
                break
        return events

    def _emit_log(self, msg: str) -> None:
        self.event_queue.put(("log", msg))
        if self.on_log:
            self.on_log(msg)

    def _emit_status(self, running: bool, msg: str) -> None:
        self.event_queue.put(("status", (running, msg)))
        if self.on_status:
            self.on_status(running, msg)

    def _emit_raid_state(self, players: Dict[str, PlayerState]) -> None:
        self.event_queue.put(("raid_state", players))
        if self.on_raid_state:
            self.on_raid_state(players)

    def _emit_draw_op(self, op_dict: Dict[str, Any]) -> None:
        self.event_queue.put(("draw_op", op_dict))
        if self.on_draw_op:
            self.on_draw_op(op_dict)

    def start(self) -> bool:
        """Start the embedded host server in a background daemon thread."""
        try:
            import websockets  # noqa: F401
        except ImportError:
            self.last_error = "Missing 'websockets' package. Run Start_ForeverNavigator.bat --update."
            self._emit_log(self.last_error)
            self._emit_status(False, self.last_error)
            return False

        if self.is_running:
            return True

        self.last_error = None
        self.local_ip = get_local_lan_ip()
        self._ready_event = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="ForeverNavigatorServer")
        self._thread.start()
        self._ready_event.wait(timeout=2.5)
        return self.is_running and self.last_error is None

    def stop(self) -> None:
        """Stop the server cleanly and wait briefly for the background thread to exit."""
        self.is_running = False
        if self._loop and not self._loop.is_closed() and self._stop_event:
            try:
                self._loop.call_soon_threadsafe(self._stop_event.set)
            except RuntimeError:
                pass
        if self._thread and self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._stop_event = asyncio.Event()
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as exc:
            self.last_error = f"Host server error: {exc}"
            self._emit_log(self.last_error)
            self._emit_status(False, self.last_error)
        finally:
            self.is_running = False
            if self._ready_event:
                self._ready_event.set()
            try:
                self._loop.close()
            except Exception:
                pass

    async def _serve(self) -> None:
        import websockets

        try:
            async with websockets.serve(
                self._handle_client,
                self.config.bind_host,
                self.config.port,
                ping_interval=self.config.ping_interval,
                ping_timeout=self.config.ping_timeout,
                close_timeout=5,
            ):
                self.is_running = True
                if self._ready_event:
                    self._ready_event.set()
                msg = (
                    f"Hosting '{self.session_id}' on {self.config.bind_host}:{self.config.port} "
                    f"(LAN IP: {self.local_ip})"
                )
                self._emit_log(msg)
                self._emit_status(True, msg)
                assert self._stop_event is not None
                await self._stop_event.wait()
        except OSError as exc:
            err = f"Cannot bind {self.config.bind_host}:{self.config.port} ({exc.strerror or exc})"
            if getattr(exc, "errno", None) in (98, 10048):
                err = f"Port {self.config.port} is already in use. Choose another port or stop the existing host."
            self.last_error = err
            self.is_running = False
            if self._ready_event:
                self._ready_event.set()
            self._emit_log(err)
            self._emit_status(False, err)
            return

        self.is_running = False
        self._emit_log("Host server stopped.")
        self._emit_status(False, "Host server stopped.")

    async def _broadcast(self, payload: Dict[str, Any], exclude: Any = None) -> None:
        if not self._clients:
            return
        raw = json.dumps(payload)
        dead = []
        for ws in list(self._clients.keys()):
            if ws is exclude:
                continue
            try:
                await ws.send(raw)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.pop(ws, None)

    def set_whiteboard_permission(self, permission: str) -> None:
        """Update whiteboard permission mid-session and notify connected clients."""
        if permission not in VALID_WHITEBOARD_PERMISSIONS:
            return
        self.config.whiteboard_permission = permission
        if self._loop and not self._loop.is_closed() and self.is_running:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._broadcast({"type": "session_update", "whiteboard_permission": permission}),
                    self._loop,
                )
            except RuntimeError:
                pass

    def _record_whiteboard_op(self, op_dict: Dict[str, Any]) -> None:
        kind = op_dict.get("kind")
        slug = str(op_dict.get("dungeon_slug", ""))
        if kind == "clear_all":
            self.whiteboard_ops = [
                op for op in self.whiteboard_ops if str(op.get("dungeon_slug", "")) != slug
            ]
            return
        if kind == "undo":
            for i in range(len(self.whiteboard_ops) - 1, -1, -1):
                if str(self.whiteboard_ops[i].get("dungeon_slug", "")) == slug:
                    self.whiteboard_ops.pop(i)
                    break
            return
        op_id = op_dict.get("op_id")
        if op_id and any(op.get("op_id") == op_id for op in self.whiteboard_ops):
            return
        self.whiteboard_ops.append(op_dict)
        if len(self.whiteboard_ops) > 500:
            self.whiteboard_ops = self.whiteboard_ops[-500:]

    def broadcast_draw_op_sync(self, op_dict: Dict[str, Any]) -> None:
        """Thread-safe hook for the Host UI to broadcast its own whiteboard drawing."""
        payload = {"type": "draw_op", **op_dict}
        self._record_whiteboard_op(payload)
        if self._loop and not self._loop.is_closed() and self.is_running:
            try:
                asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)
            except RuntimeError:
                pass

    def update_host_position_sync(self, state: PlayerState) -> None:
        """Thread-safe hook for the Host's own local navigator position."""
        self.players[state.player_id] = state
        if self._loop and not self._loop.is_closed() and self.is_running:
            try:
                asyncio.run_coroutine_threadsafe(self._broadcast_raid_state(), self._loop)
            except RuntimeError:
                pass

    async def _broadcast_raid_state(self) -> None:
        payload = {
            "type": "raid_state",
            "t_host_ms": now_ms(),
            "players": {pid: st.to_dict() for pid, st in self.players.items()},
        }
        await self._broadcast(payload)
        self._emit_raid_state(dict(self.players))

    async def _handle_client(self, ws: Any) -> None:
        player_id: Optional[str] = None
        player_name = "Raider"
        try:
            hello_raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
            hello = json.loads(hello_raw)
            if hello.get("type") != "hello":
                await ws.close(1008, "Expected hello")
                return
            if self.config.room_passphrase and hello.get("room_passphrase") != self.config.room_passphrase:
                await ws.close(1008, "Invalid room passphrase")
                return

            pinfo = hello.get("player", {})
            player_id = str(pinfo.get("player_id") or uuid.uuid4())
            player_name = str(pinfo.get("name", "Raider"))
            is_officer = bool(
                self.config.officer_token and hello.get("officer_token") == self.config.officer_token
            )
            role = "officer" if is_officer else "raider"
            self._clients[ws] = {"player_id": player_id, "role": role, "player": pinfo}
            self._emit_log(f"Connected: {player_name} ({role}) | {self.connected_count} client(s)")

            await ws.send(
                json.dumps(
                    {
                        "type": "welcome",
                        "protocol_version": PROTOCOL_VERSION,
                        "session_id": self.session_id,
                        "assigned_role": role,
                        "host_time_ms": now_ms(),
                        "whiteboard_permission": self.config.whiteboard_permission,
                        "participants": [st.to_dict() for st in self.players.values()],
                        "whiteboard_ops": self.whiteboard_ops[-200:],
                    }
                )
            )

            async for msg_raw in ws:
                if isinstance(msg_raw, (bytes, bytearray)):
                    continue
                msg = json.loads(msg_raw)
                mtype = msg.get("type")

                if mtype == "clock_ping":
                    rx = now_ms()
                    await ws.send(
                        json.dumps(
                            {
                                "type": "clock_pong",
                                "t0_ms": msg.get("t0_ms", rx),
                                "host_rx_ms": rx,
                                "host_tx_ms": now_ms(),
                            }
                        )
                    )
                elif mtype == "pos_update":
                    raw_px = msg.get("map_px")
                    map_px = (
                        (int(raw_px[0]), int(raw_px[1]))
                        if isinstance(raw_px, (list, tuple)) and len(raw_px) >= 2
                        else (0, 0)
                    )
                    st = PlayerState(
                        player_id=player_id,
                        name=player_name,
                        wow_class=str(pinfo.get("wow_class", "UNKNOWN")),
                        role=str(pinfo.get("role", "DPS")),
                        dungeon_slug=str(msg.get("dungeon_slug", "")),
                        x=float(msg.get("x", 0.0)),
                        y=float(msg.get("y", 0.0)),
                        z=float(msg.get("z", 0.0)),
                        map_px=map_px,
                        confidence=float(msg.get("confidence", 1.0)),
                        alive=bool(msg.get("alive", True)),
                        t_host_ms=int(msg.get("t_host_ms", now_ms())),
                    )
                    self.players[player_id] = st
                    await self._broadcast_raid_state()
                elif mtype == "draw_op":
                    perm = self.config.whiteboard_permission
                    if perm == "all" or (perm == "officers" and role in ("officer", "leader")):
                        self._record_whiteboard_op(msg)
                        self._emit_draw_op(msg)
                        await self._broadcast(msg, exclude=ws)
                    else:
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "error",
                                    "code": "draw_denied",
                                    "message": f"Whiteboard drawing is restricted to {perm}.",
                                }
                            )
                        )
                elif mtype in ("video_frame", "combatlog_chunk"):
                    # Phase 3 hook: persist to SessionStore under sessions/<session_id>/
                    pass
        except Exception:
            pass
        finally:
            self._clients.pop(ws, None)
            if player_id:
                still_connected = any(
                    info.get("player_id") == player_id for info in self._clients.values()
                )
                if not still_connected and player_id in self.players:
                    self.players.pop(player_id, None)
                    self._emit_log(f"Disconnected: {player_name} | {self.connected_count} client(s)")
                    if self.is_running:
                        await self._broadcast_raid_state()


RaidServer = ForeverNavigatorServer
