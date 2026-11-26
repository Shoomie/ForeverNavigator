"""Startup hygiene for ForeverNavigator: stale-instance cleanup & leftover file pruning.

Runs once at live application startup (`run_qt_app`):
  1. Checks `data/.forever_navigator.pid` for an orphaned previous ForeverNavigator
     process (verifying it is actually python/pythonw before terminating) so DXGI
     Desktop Duplication and WebSocket port 8765 are never held by a hung instance.
  2. Prunes empty (0-row) `data/sessions/session_*.db` files and orphaned `-wal`/`-shm`
     sidecars left behind by crashed runs.
  3. Prunes empty `data/dev_gameplay/run_*` directories that contain 0 captured frames.
"""

from __future__ import annotations

import atexit
import ctypes
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Dict, Optional


def _is_forever_python_pid(pid: int) -> bool:
    """Verify `pid` is a running python/pythonw process before terminating (prevents PID-reuse accidents)."""
    if pid <= 0 or pid == os.getpid():
        return False
    if os.name == "nt":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return False
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = ctypes.c_uint32(len(buf))
            if not kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                return False
            exe = Path(buf.value).name.lower()
            return exe in ("python.exe", "pythonw.exe")
        finally:
            kernel32.CloseHandle(handle)
    else:
        try:
            os.kill(pid, 0)
            cmdline = Path(f"/proc/{pid}/cmdline").read_text(encoding="utf-8", errors="ignore").lower()
            return "python" in cmdline and "forever" in cmdline or "src.ui" in cmdline
        except Exception:
            return False


def _terminate_pid(pid: int, wait_sec: float = 1.5) -> bool:
    """Terminate a verified stale process and wait briefly for OS handles (DXGI/ports) to release."""
    if os.name == "nt":
        PROCESS_TERMINATE = 0x0001
        SYNCHRONIZE = 0x00100000
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE, False, int(pid))
        if not handle:
            return False
        try:
            ok = bool(kernel32.TerminateProcess(handle, 1))
            if ok:
                kernel32.WaitForSingleObject(handle, int(wait_sec * 1000))
            return ok
        finally:
            kernel32.CloseHandle(handle)
    else:
        import signal

        try:
            os.kill(pid, signal.SIGTERM)
            deadline = time.perf_counter() + wait_sec
            while time.perf_counter() < deadline:
                try:
                    os.kill(pid, 0)
                    time.sleep(0.05)
                except ProcessLookupError:
                    return True
            return True
        except Exception:
            return False


def acquire_instance_lock(pid_file: Path = Path("data/.forever_navigator.pid")) -> Optional[int]:
    """Terminate any orphaned previous ForeverNavigator instance and claim `pid_file`."""
    killed_pid: Optional[int] = None
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    if pid_file.exists():
        try:
            raw = pid_file.read_text(encoding="utf-8").strip()
            old_pid = int(raw)
            if _is_forever_python_pid(old_pid):
                if _terminate_pid(old_pid):
                    killed_pid = old_pid
        except Exception:
            pass

    my_pid = os.getpid()
    try:
        pid_file.write_text(str(my_pid), encoding="utf-8")
    except Exception:
        pass

    def _release() -> None:
        try:
            if pid_file.exists() and pid_file.read_text(encoding="utf-8").strip() == str(my_pid):
                pid_file.unlink(missing_ok=True)
        except Exception:
            pass

    atexit.register(_release)
    return killed_pid


def prune_leftover_sessions(
    sessions_dir: Path = Path("data/sessions"),
    active_db: Optional[Path] = None,
    min_age_sec: float = 5.0,
) -> int:
    """Remove empty (0-row) session_*.db files and orphaned -wal/-shm sidecars."""
    if not sessions_dir.is_dir():
        return 0
    now = time.time()
    active_res = active_db.resolve() if active_db is not None else None
    removed = 0

    for db_path in sorted(sessions_dir.glob("session_*.db")):
        try:
            if active_res is not None and db_path.resolve() == active_res:
                continue
            if (now - db_path.stat().st_mtime) < min_age_sec:
                continue
            is_empty = False
            conn = sqlite3.connect(str(db_path), timeout=0.5)
            try:
                cur = conn.cursor()
                pos_n = cur.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
                ce_n = cur.execute("SELECT COUNT(*) FROM combat_events").fetchone()[0]
                vf_n = cur.execute("SELECT COUNT(*) FROM video_frames").fetchone()[0]
                is_empty = (pos_n == 0 and ce_n == 0 and vf_n == 0)
            except Exception:
                is_empty = False
            finally:
                conn.close()
            if is_empty:
                for suf in ("", "-wal", "-shm"):
                    Path(str(db_path) + suf).unlink(missing_ok=True)
                removed += 1
        except Exception:
            pass

    # Clean orphaned -wal / -shm files whose parent .db no longer exists
    for sidecar in list(sessions_dir.glob("*.db-wal")) + list(sessions_dir.glob("*.db-shm")):
        base_db = Path(str(sidecar)[:-4])
        if not base_db.exists():
            try:
                sidecar.unlink(missing_ok=True)
                removed += 1
            except Exception:
                pass
    return removed


def prune_empty_dev_runs(
    dev_root: Path = Path("data/dev_gameplay"),
    active_run_dir: Optional[Path] = None,
    min_age_sec: float = 10.0,
) -> int:
    """Remove empty `run_*` folders that have 0 minimap and 0 fullscreen captures."""
    if not dev_root.is_dir():
        return 0
    now = time.time()
    active_res = active_run_dir.resolve() if active_run_dir is not None else None
    removed = 0
    for run_dir in sorted(dev_root.glob("run_*")):
        if not run_dir.is_dir():
            continue
        try:
            if active_res is not None and run_dir.resolve() == active_res:
                continue
            if (now - run_dir.stat().st_mtime) < min_age_sec:
                continue
            mm_dir = run_dir / "minimap"
            fs_dir = run_dir / "fullscreen"
            has_mm = mm_dir.is_dir() and any(mm_dir.glob("*.png"))
            has_fs = fs_dir.is_dir() and any(fs_dir.glob("*.png"))
            if not has_mm and not has_fs:
                shutil.rmtree(run_dir, ignore_errors=True)
                removed += 1
        except Exception:
            pass
    return removed


def run_startup_cleanup(data_dir: Path = Path("data")) -> Dict[str, Optional[int]]:
    """Run all startup hygiene steps cleanly before initializing DXGI/UI."""
    killed_pid = acquire_instance_lock(data_dir / ".forever_navigator.pid")
    pruned_db = prune_leftover_sessions(data_dir / "sessions")
    pruned_runs = prune_empty_dev_runs(data_dir / "dev_gameplay")
    return {
        "killed_stale_pid": killed_pid,
        "pruned_empty_sessions": pruned_db,
        "pruned_empty_dev_runs": pruned_runs,
    }
