"""Raider WebSocket Client with granular opt-in telemetry & NTP-style clock sync."""

from __future__ import annotations

import asyncio
import inspect
import json
import queue
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from .protocol import (
    PROTOCOL_VERSION,
    PlayerIdentity,
    PlayerState,
    SyncPreferences,
    normalize_ws_url,
    now_ms,
)


class ForeverNavigatorClient:
    """Connects a raider to the Raid Leader's embedded ForeverNavigatorServer."""

    def __init__(
        self,
        url: str,
        identity: PlayerIdentity,
        prefs: SyncPreferences,
        room_passphrase: str = "",
        officer_token: str = "",
        on_raid_state: Optional[Callable[[Dict[str, PlayerState]], None]] = None,
        on_draw_op: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        on_welcome: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_status: Optional[Callable[[bool, str], None]] = None,
        auto_reconnect: bool = True,
        ping_interval: float = 20.0,
        ping_timeout: float = 20.0,
    ):
        self.url = normalize_ws_url(url)
        self.identity = identity
        self.prefs = prefs
        self.room_passphrase = room_passphrase
        self.officer_token = officer_token
        self.on_raid_state = on_raid_state
        self.on_draw_op = on_draw_op
        self.on_log = on_log
        self.on_welcome = on_welcome
        self.on_status = on_status
        self.auto_reconnect = auto_reconnect
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout

        self.clock_offset_ms: int = 0
        self._best_rtt_ms: Optional[int] = None
        self.connected: bool = False
        self.assigned_role: str = "raider"
        self.whiteboard_permission: str = "all"
        self.session_id: str = ""
        self.last_error: Optional[str] = None
        self.event_queue: "queue.SimpleQueue[Tuple[str, Any]]" = queue.SimpleQueue()

        self._ws: Any = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[asyncio.Event] = None

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

    def _emit_status(self, connected: bool, msg: str) -> None:
        self.event_queue.put(("status", (connected, msg)))
        if self.on_status:
            self.on_status(connected, msg)

    def _emit_raid_state(self, players: Dict[str, PlayerState]) -> None:
        self.event_queue.put(("raid_state", players))
        if self.on_raid_state:
            self.on_raid_state(players)

    def _emit_draw_op(self, op_dict: Dict[str, Any]) -> None:
        self.event_queue.put(("draw_op", op_dict))
        if self.on_draw_op:
            self.on_draw_op(op_dict)

    def can_draw(self) -> bool:
        """Whether this client currently has whiteboard drawing permission."""
        if self.whiteboard_permission == "all":
            return True
        if self.whiteboard_permission == "officers":
            return self.assigned_role in ("officer", "leader")
        if self.whiteboard_permission == "leader":
            return self.assigned_role == "leader"
        return False

    def host_now_ms(self) -> int:
        return now_ms() + self.clock_offset_ms

    def start(self) -> bool:
        try:
            import websockets  # noqa: F401
        except ImportError:
            self.last_error = "Missing 'websockets' package. Run Start_ForeverNavigator.bat --update."
            self._emit_log(self.last_error)
            self._emit_status(False, self.last_error)
            return False

        if self._thread and self._thread.is_alive():
            return True
        self.last_error = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="ForeverNavigatorClient")
        self._thread.start()
        return True

    def stop(self) -> None:
        """Disconnect cleanly and stop the background reconnect loop."""
        self.connected = False
        if self._loop and not self._loop.is_closed():
            if self._stop_event:
                try:
                    self._loop.call_soon_threadsafe(self._stop_event.set)
                except RuntimeError:
                    pass
            if self._ws is not None:
                try:
                    asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop)
                except RuntimeError:
                    pass
        if self._thread and self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)

    def send_position_sync(self, state: PlayerState) -> None:
        if (
            not self.connected
            or not self.prefs.sync_position
            or not self._loop
            or self._loop.is_closed()
            or not self._ws
        ):
            return
        payload = {
            "type": "pos_update",
            "player_id": self.identity.player_id,
            "t_host_ms": self.host_now_ms(),
            "dungeon_slug": state.dungeon_slug,
            "x": state.x,
            "y": state.y,
            "z": state.z,
            "map_px": list(state.map_px),
            "confidence": state.confidence,
            "alive": state.alive,
        }
        try:
            asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps(payload)), self._loop)
        except RuntimeError:
            pass

    def send_draw_op_sync(self, op_dict: Dict[str, Any]) -> None:
        if not self.connected or not self._loop or self._loop.is_closed() or not self._ws:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._ws.send(json.dumps({"type": "draw_op", **op_dict})), self._loop
            )
        except RuntimeError:
            pass

    def send_clock_ping_sync(self) -> None:
        if not self.connected or not self._loop or self._loop.is_closed() or not self._ws:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._ws.send(json.dumps({"type": "clock_ping", "t0_ms": now_ms()})),
                self._loop,
            )
        except RuntimeError:
            pass

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._stop_event = asyncio.Event()
        try:
            self._loop.run_until_complete(self._reconnect_loop())
        except Exception as exc:
            self.last_error = str(exc)
            self._emit_log(f"Client error: {exc}")
            self._emit_status(False, self.last_error)
        finally:
            self.connected = False
            self._ws = None
            try:
                self._loop.close()
            except Exception:
                pass

    async def _reconnect_loop(self) -> None:
        import websockets

        backoff = 1.0
        assert self._stop_event is not None

        while not self._stop_event.is_set():
            try:
                await self._connect_and_listen()
                backoff = 1.0
                if self._stop_event.is_set() or not self.auto_reconnect:
                    break
            except websockets.exceptions.ConnectionClosedError as exc:
                self.connected = False
                self._ws = None
                if exc.code == 1008:
                    reason = exc.reason or "Rejected by host"
                    self.last_error = reason
                    self._emit_log(f"Connection rejected: {reason}")
                    self._emit_status(False, f"Rejected: {reason}")
                    break
                if self._stop_event.is_set() or not self.auto_reconnect:
                    break
                msg = f"Disconnected ({exc.reason or exc.code}). Reconnecting in {backoff:.0f}s..."
                self._emit_log(msg)
                self._emit_status(False, msg)
            except Exception as exc:
                self.connected = False
                self._ws = None
                if self._stop_event.is_set() or not self.auto_reconnect:
                    self.last_error = str(exc)
                    self._emit_log(f"Connection failed: {exc}")
                    self._emit_status(False, f"Disconnected: {exc}")
                    break
                msg = f"Connection failed ({exc}). Retrying in {backoff:.0f}s..."
                self.last_error = str(exc)
                self._emit_log(msg)
                self._emit_status(False, msg)

            if self._stop_event.is_set() or not self.auto_reconnect:
                break
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=backoff)
                break
            except asyncio.TimeoutError:
                backoff = min(10.0, backoff * 2.0)

    async def _clock_ping_loop(self, ws: Any) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=30.0)
                break
            except asyncio.TimeoutError:
                try:
                    await ws.send(json.dumps({"type": "clock_ping", "t0_ms": now_ms()}))
                except Exception:
                    break

    async def _connect_and_listen(self) -> None:
        import websockets

        assert self._stop_event is not None
        connect_kwargs: Dict[str, Any] = {
            "ping_interval": self.ping_interval,
            "ping_timeout": self.ping_timeout,
            "open_timeout": 8.0,
            "close_timeout": 5.0,
        }
        if "proxy" in inspect.signature(websockets.connect).parameters:
            connect_kwargs["proxy"] = None

        async with websockets.connect(self.url, **connect_kwargs) as ws:
            self._ws = ws
            self.connected = True
            self._best_rtt_ms = None
            await ws.send(
                json.dumps(
                    {
                        "type": "hello",
                        "protocol_version": PROTOCOL_VERSION,
                        "room_passphrase": self.room_passphrase,
                        "officer_token": self.officer_token,
                        "player": self.identity.to_dict(),
                        "sync_prefs": self.prefs.to_dict(),
                    }
                )
            )
            # Kick off NTP-style clock sync immediately
            await ws.send(json.dumps({"type": "clock_ping", "t0_ms": now_ms()}))
            ping_task = asyncio.create_task(self._clock_ping_loop(ws))

            try:
                async for raw in ws:
                    if self._stop_event.is_set():
                        break
                    if isinstance(raw, (bytes, bytearray)):
                        continue
                    msg = json.loads(raw)
                    mtype = msg.get("type")
                    if mtype == "welcome":
                        self.session_id = str(msg.get("session_id", ""))
                        self.assigned_role = str(msg.get("assigned_role", "raider"))
                        self.whiteboard_permission = str(msg.get("whiteboard_permission", "all"))
                        if "host_time_ms" in msg:
                            self.clock_offset_ms = int(msg["host_time_ms"]) - now_ms()
                        status_msg = f"Joined '{self.session_id}' as {self.assigned_role}"
                        self._emit_log(status_msg)
                        self._emit_status(True, status_msg)
                        self.event_queue.put(("welcome", msg))
                        if self.on_welcome:
                            self.on_welcome(msg)
                        participants = msg.get("participants", [])
                        if participants:
                            players = {
                                str(p["player_id"]): PlayerState.from_dict(p)
                                for p in participants
                                if "player_id" in p
                            }
                            if players:
                                self._emit_raid_state(players)
                        for op in msg.get("whiteboard_ops", []):
                            self._emit_draw_op(op)
                    elif mtype == "session_update":
                        if "whiteboard_permission" in msg:
                            self.whiteboard_permission = str(msg["whiteboard_permission"])
                    elif mtype == "clock_pong":
                        t1 = now_ms()
                        t0 = int(msg["t0_ms"])
                        hrx = int(msg["host_rx_ms"])
                        htx = int(msg["host_tx_ms"])
                        rtt = max(0, (t1 - t0) - (htx - hrx))
                        offset = int(hrx - t0 - rtt // 2)
                        if self._best_rtt_ms is None or rtt <= self._best_rtt_ms:
                            self._best_rtt_ms = rtt
                            self.clock_offset_ms = offset
                        elif rtt <= self._best_rtt_ms + 30:
                            self.clock_offset_ms = int(0.7 * self.clock_offset_ms + 0.3 * offset)
                    elif mtype == "raid_state":
                        players = {
                            pid: PlayerState.from_dict(pd) for pid, pd in msg.get("players", {}).items()
                        }
                        self._emit_raid_state(players)
                    elif mtype == "draw_op":
                        self._emit_draw_op(msg)
                    elif mtype == "error":
                        err_msg = str(msg.get("message", "Server error"))
                        self.last_error = err_msg
                        self._emit_log(err_msg)
            finally:
                ping_task.cancel()
                try:
                    await ping_task
                except asyncio.CancelledError:
                    pass
                self.connected = False
                self._ws = None


RaidClient = ForeverNavigatorClient
