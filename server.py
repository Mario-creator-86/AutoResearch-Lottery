"""AutoResearch Dashboard backend with multi-lottery support."""

from __future__ import annotations

import asyncio
import atexit
import ctypes
import functools
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
import uuid
import threading
from contextlib import asynccontextmanager
from collections import defaultdict
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path

# Global variable to store agent processes for cleanup
_agent_processes = {}

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.responses import FileResponse

import auto_loop
from atomic_io import read_json_with_recovery, write_json_atomic, write_json_resilient
from lock_utils import current_lock_owner, lock_matches_process, process_runs_script, read_lock_owner, serialize_lock_owner
from load_data import load_lottery_draws, load_specific_lottery_draws
from lottery_config import LotteryConfig, ensure_lottery_storage, get_lottery_config, list_lottery_configs
from walkforward import load_walkforward_result, run_and_save_walkforward

PROJECT_DIR = Path(__file__).parent
SERVER_LOCK_FILE = PROJECT_DIR / ".server.lock"
DEFAULT_SERVER_PORT = 8020  # override: --port N or AUTORESEARCH_PORT

_run_processes: dict[str, subprocess.Popen] = {}
_loop_processes: dict[str, subprocess.Popen] = {}
_process_lock: asyncio.Lock | None = None
STALE_PROGRESS_SECONDS = 20 * 60
MAX_HISTORY_TICKETS_PER_DAY = 20
MAX_HISTORY_TICKETS_PER_DRAW = 20  # Per draw (date+time), not per day

# Per-lottery history cache invalidation tracking
_history_cache_invalidations: set[str] = set()


class _TTLCache:
    """Simple time-to-live cache for server-side response caching."""
    def __init__(self):
        self._store: dict[str, tuple[float, object]] = {}

    def get(self, key: str) -> object | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires, value = entry
        if time.time() > expires:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: object, ttl: float) -> None:
        self._store[key] = (time.time() + ttl, value)

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)


_response_cache = _TTLCache()


def _get_lock() -> asyncio.Lock:
    global _process_lock
    if _process_lock is None:
        _process_lock = asyncio.Lock()
    return _process_lock


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        process = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if process:
            ctypes.windll.kernel32.CloseHandle(process)
            return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _set_current_process_priority(priority_class: int) -> None:
    if sys.platform != "win32" or not priority_class:
        return
    try:
        kernel32 = ctypes.windll.kernel32
        current_process = kernel32.GetCurrentProcess()
        kernel32.SetPriorityClass(current_process, int(priority_class))
    except Exception:
        return


def _acquire_singleton_lock(lock_file: Path, label: str) -> bool:
    current_pid = os.getpid()
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_owner = current_lock_owner(
        script_path=Path(__file__),
        project_dir=PROJECT_DIR,
        kind=label,
    )

    for _ in range(2):
        try:
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing_owner = read_lock_owner(lock_file)
            existing_pid = existing_owner.pid
            if existing_pid == current_pid:
                return True
            if existing_pid is not None and _pid_alive(existing_pid):
                if lock_matches_process(existing_owner, script_path=Path(__file__), project_dir=PROJECT_DIR):
                    return False
            try:
                lock_file.unlink(missing_ok=True)
            except OSError:
                return False
            continue
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(serialize_lock_owner(lock_owner))
                handle.flush()
        except OSError:
            try:
                lock_file.unlink(missing_ok=True)
            except OSError:
                pass
            return False
        return True

    return False


def _release_singleton_lock(lock_file: Path) -> None:
    owner_pid = read_lock_owner(lock_file).pid
    if owner_pid != os.getpid():
        return
    try:
        lock_file.unlink(missing_ok=True)
    except OSError:
        pass


def _register_server_lock_cleanup() -> None:
    atexit.register(_release_singleton_lock, SERVER_LOCK_FILE)


def _reconcile_process_locks() -> None:
    """Remove stale per-lottery lock files and mark stale progress as idle."""
    for config in list_lottery_configs():
        ensure_lottery_storage(config)
        lock_pid = _lock_file_has_alive_pid(config)
        if lock_pid:
            print(f"[startup] {config.id}: found running process PID {lock_pid} (keeping lock)")
        elif config.lock_file.exists():
            try:
                config.lock_file.unlink(missing_ok=True)
                print(f"[startup] {config.id}: removed stale lock file")
            except OSError:
                pass

        # Mark stale progress files as idle if the referenced PID is dead
        # or alive but not associated with this lottery's lock file
        progress_data, _ = read_json_with_recovery(config.progress_file)
        if isinstance(progress_data, dict) and progress_data.get("status") in {"running", "starting"}:
            pid = progress_data.get("pid")
            if pid:
                if not _pid_alive(int(pid)):
                    progress_data["status"] = "idle"
                    progress_data.pop("error", None)
                    progress_data.pop("stale_seconds", None)
                    write_json_atomic(config.progress_file, progress_data, ensure_ascii=True)
                    print(f"[startup] {config.id}: marked stale progress (PID {pid}) as idle")
                elif not lock_pid or lock_pid != int(pid):
                    # PID is alive but doesn't belong to this lottery — likely PID reuse
                    progress_data["status"] = "idle"
                    progress_data.pop("error", None)
                    progress_data.pop("stale_seconds", None)
                    write_json_atomic(config.progress_file, progress_data, ensure_ascii=True)
                    print(f"[startup] {config.id}: marked progress (PID {pid}) as idle (lock mismatch)")
            else:
                progress_data["status"] = "idle"
                progress_data.pop("error", None)
                write_json_atomic(config.progress_file, progress_data, ensure_ascii=True)
                print(f"[startup] {config.id}: marked progress (no PID) as idle")


def _signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT" if signum == signal.SIGINT else f"signal {signum}"
    print(f"Received {signal_name} ({signum}), shutting down agents...")

    # Set a flag to indicate shutdown is in progress
    global _shutdown_in_progress
    _shutdown_in_progress = True

    try:
        _stop_agents(_agent_processes)
    except Exception as e:
        print(f"Error during agent shutdown: {e}")

    # Remove server PID file on shutdown
    try:
        AGENTS_DIR = PROJECT_DIR / "agents"
        server_pid_file = AGENTS_DIR / ".server_pid"
        if server_pid_file.exists():
            server_pid_file.unlink()
            print("  Server PID file removed")
    except Exception as e:
        print(f"Warning: Could not remove server PID file: {e}")

    print("Server shutdown completed.")
    sys.exit(0)


# Global flag to track shutdown state
_shutdown_in_progress = False

async def _lifespan(_app: FastAPI):
    global _agent_processes, _auto_update_task

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    _reconcile_process_locks()
    _agent_processes = _start_required_agents()

    # Start agent monitoring
    _monitor_agents(_agent_processes)

    # Start automatic draw-history updater (downloads history by itself)
    _auto_update_task = asyncio.create_task(_auto_update_history_loop())

    # Register atexit handler as backup
    atexit.register(_stop_agents, _agent_processes)

    try:
        yield
    finally:
        if _auto_update_task:
            _auto_update_task.cancel()
        _stop_agents(_agent_processes)


app = FastAPI(title="AutoResearch Dashboard", lifespan=_lifespan)

# Serve static assets with no-cache headers for JS/CSS so ?v= cache-busting always works
_STATIC_DIR = PROJECT_DIR / "static"


@app.api_route("/static/{file_path:path}", methods=["GET", "HEAD"])
async def serve_static(file_path: str):
    full = _STATIC_DIR / file_path
    if not full.is_file():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not found")
    resp = FileResponse(str(full))
    if file_path.endswith((".js", ".css")):
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
    return resp


def _config(lottery_id: str) -> LotteryConfig:
    config = get_lottery_config(lottery_id)
    ensure_lottery_storage(config)
    return config


def _cleanup_processes() -> None:
    for store in (_run_processes, _loop_processes):
        finished = [lottery for lottery, proc in store.items() if proc.poll() is not None]
        for lottery in finished:
            del store[lottery]


def _lock_file_has_alive_pid(config: LotteryConfig) -> int | None:
    """Check if a lottery's lock file contains a PID that is still alive.
    Returns the alive PID or None."""
    lock_file = getattr(config, "lock_file", None)
    if lock_file is None:
        return None
    owner = read_lock_owner(lock_file)
    pid = owner.pid
    if pid is None:
        return None
    if _pid_alive(pid) and lock_matches_process(owner, script_path=PROJECT_DIR / "auto_loop.py", project_dir=PROJECT_DIR):
        return pid
    return None


def _loop_info_from_state(state: auto_loop.SearchState) -> dict:
    return {
        "iteration": state.iteration,
        "search_phase": state.search_phase,
        "radius": state.radius,
        "step": state.step,
        "no_improve_streak": state.no_improve_count,
        "strategies": list(state.extra_strategies),
        "weights": {
            "tf": state.tf,
            "hc": state.hc,
            "bal": state.bal,
            **{name: state.extra_weight(name) for name in state.extra_strategies},
        },
        "iw": state.iw,
        "rw": state.rw,
        "prior_strength": state.prior_strength,
        "ewma_lambda": state.ewma_lambda,
        "use_anticorrelation": state.use_anticorrelation,
        "anticorrelation_ratio": state.anticorrelation_ratio,
        "best_metric": state.best_metric,
        "best_strategy": state.best_strategy,
    }


def _synthesize_loop_progress(config: LotteryConfig, pid: int, base_data: dict | None = None) -> dict | None:
    loop_state_file = getattr(config, "loop_state_file", None)
    if loop_state_file is None:
        return None
    payload, _source_path = read_json_with_recovery(loop_state_file)
    if not isinstance(payload, dict):
        return None

    state = auto_loop.SearchState.from_dict(payload.get("search_state", {}))
    active_iteration = payload.get("active_iteration")
    data = dict(base_data or {})
    data["status"] = "running"
    data["pid"] = pid
    data["experiment_id"] = f"auto_iter{state.iteration}"
    data["loop_info"] = _loop_info_from_state(state)
    data["updated_at"] = data.get("updated_at") or time.strftime("%Y-%m-%dT%H:%M:%S")

    if isinstance(active_iteration, dict):
        tested = auto_loop._active_iteration_tested_count(active_iteration)
        total = int(active_iteration.get("total", tested))
        data["tested"] = tested
        data["total"] = total
        data["percent"] = round(tested / total * 100, 1) if total > 0 else 0
        data["current_config"] = active_iteration.get("current_config")
        data["best_so_far"] = active_iteration.get("best_progress", {})
        data["recent"] = list(active_iteration.get("recent", []))[-10:]
        data.pop("waiting_for_next_iteration", None)
    else:
        data["waiting_for_next_iteration"] = True
        data.setdefault("tested", 0)
        data.setdefault("total", 0)
        data.setdefault("percent", 0)
        data.setdefault("current_config", None)
        data.setdefault("recent", [])
        data.setdefault(
            "best_so_far",
            {"config": state.best_strategy, "average_hits": state.best_metric},
        )
    return data


def _progress_auto_loop_pid(config: LotteryConfig, data: dict | None) -> int | None:
    if not isinstance(data, dict):
        return None
    if data.get("status") not in {"running", "stalled"}:
        return None
    pid = data.get("pid")
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if not _pid_alive(pid):
        return None
    # PID reciclat de Windows după moartea buclei → nu e auto_loop-ul nostru.
    if not process_runs_script(pid, PROJECT_DIR / "auto_loop.py"):
        return None
    experiment_id = str(data.get("experiment_id") or "")
    if not experiment_id.startswith("auto_iter"):
        return None
    loop_state_file = getattr(config, "loop_state_file", None)
    if loop_state_file is None:
        return None
    payload, _source_path = read_json_with_recovery(loop_state_file)
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("search_state"), dict):
        return None
    return pid


def _progress_starting_pid(data: dict | None) -> int | None:
    if not isinstance(data, dict):
        return None
    if data.get("status") != "starting":
        return None
    pid = data.get("pid")
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if not _pid_alive(pid):
        return None
    return pid


def _active_loop_pids(
    lottery_id: str,
    config: LotteryConfig | None = None,
    progress: dict | None = None,
) -> list[int]:
    config = config or _config(lottery_id)
    pids: list[int] = []
    lock_pid = _lock_file_has_alive_pid(config)
    if lock_pid and lock_pid not in pids:
        pids.append(int(lock_pid))
    progress = progress if progress is not None else _read_progress(config)
    progress_loop_pid = _progress_auto_loop_pid(config, progress)
    if progress_loop_pid and progress_loop_pid not in pids:
        pids.append(int(progress_loop_pid))
    starting_pid = _progress_starting_pid(progress)
    if starting_pid and starting_pid not in pids:
        pids.append(int(starting_pid))
    proc = _loop_processes.get(lottery_id)
    if proc and proc.poll() is None and int(proc.pid) not in pids:
        pids.append(int(proc.pid))
    return pids


def _canonical_loop_pid(
    lottery_id: str,
    config: LotteryConfig | None = None,
    progress: dict | None = None,
) -> int | None:
    pids = _active_loop_pids(lottery_id, config=config, progress=progress)
    return pids[0] if pids else None


def _path_signature(path: Path) -> tuple[str, int | None, int | None]:
    try:
        stat = path.stat()
        return str(path), stat.st_mtime_ns, stat.st_size
    except FileNotFoundError:
        return str(path), None, None


@lru_cache(maxsize=64)
def _parse_results_cached(path_str: str, mtime_ns: int | None, size: int | None) -> tuple[dict, ...]:
    if mtime_ns is None or size is None:
        return ()
    path = Path(path_str)
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
    except FileNotFoundError:
        return ()
    if len(lines) < 2:
        return ()
    headers = lines[0].split("\t")
    experiments: list[dict] = []
    for index, line in enumerate(lines[1:], 1):
        cols = line.split("\t")
        if len(cols) < len(headers):
            continue
        row = dict(zip(headers, cols))
        for key in ("average_hits", "hit_rate_2_plus", "hit_rate_3_plus", "portfolio_avg", "elapsed_s"):
            try:
                row[key] = float(row[key])
            except (KeyError, ValueError):
                pass
        row["experiment_number"] = index
        experiments.append(row)
    return tuple(experiments)


def _parse_results(config: LotteryConfig) -> list[dict]:
    signature = _path_signature(config.results_file)
    return [dict(row) for row in _parse_results_cached(*signature)]


def _recovered_json_mtime(path: Path, source_path: Path | None = None) -> float | None:
    candidate = source_path
    if candidate is None:
        _payload, candidate = read_json_with_recovery(path)
    if candidate is None:
        return None
    try:
        return candidate.stat().st_mtime
    except FileNotFoundError:
        return None


def _latest_activity_mtime(config: LotteryConfig, data: dict, progress_source_path: Path | None) -> float | None:
    activity_mtime = _recovered_json_mtime(config.progress_file, progress_source_path)
    loop_pid = _lock_file_has_alive_pid(config) or _progress_auto_loop_pid(config, data)
    if loop_pid:
        loop_state_file = getattr(config, "loop_state_file", None)
        if loop_state_file is not None:
            loop_state_mtime = _recovered_json_mtime(loop_state_file)
            if loop_state_mtime is not None:
                if activity_mtime is None:
                    activity_mtime = loop_state_mtime
                else:
                    activity_mtime = max(activity_mtime, loop_state_mtime)
    return activity_mtime


def _read_progress(config: LotteryConfig) -> dict:
    data, source_path = read_json_with_recovery(config.progress_file)
    if not isinstance(data, dict):
        data = {"status": "idle"}
    if data.get("status") == "running":
        pid = data.get("pid")
        if pid and not _pid_alive(int(pid)):
            data["status"] = "idle"
            data.pop("error", None)
        else:
            activity_mtime = _latest_activity_mtime(config, data, source_path)
            if activity_mtime is None:
                activity_mtime = time.time()
            stale_seconds = max(0.0, time.time() - activity_mtime)
        if data.get("status") == "running" and stale_seconds >= STALE_PROGRESS_SECONDS:
            data["status"] = "stalled"
            data["stale_seconds"] = round(stale_seconds, 1)
            data["error"] = (
                f"Process {pid} is alive, but progress has not updated for "
                f"{round(stale_seconds / 60, 1)} minutes"
            )
    elif data.get("status") == "stopped":
        # "stopped" is a transient status — once the process is gone, show idle
        pid = data.get("pid") or data.get("stopped_pid")
        if not pid or not _pid_alive(int(pid)):
            data["status"] = "idle"
            data.pop("error", None)
    loop_pid = _lock_file_has_alive_pid(config) or _progress_auto_loop_pid(config, data)
    if loop_pid:
        synthesized = _synthesize_loop_progress(config, loop_pid, data)
        if synthesized is not None:
            if data.get("status") == "stalled":
                synthesized["status"] = "stalled"
                if "stale_seconds" in data:
                    synthesized["stale_seconds"] = data["stale_seconds"]
                if "error" in data:
                    synthesized["error"] = data["error"]
            return synthesized
    return data


def _get_best_strategy(lottery_id: str) -> dict:
    # Always use the best from results, not the active state
    state = auto_loop.load_best_from_results_for_lottery(lottery_id)
    if state.best_metric <= 0:
        # Fallback to active state if no results
        state, _active = auto_loop.load_search_state_for_lottery(lottery_id)
    return {
        "name": state.best_strategy or "unknown",
        "metric": state.best_metric,
        "tf": state.tf,
        "hc": state.hc,
        "b": state.bal,
        "sb": state.sb if "sb" in state.extra_strategies else 1.0,
        "delay": state.delay if "delay" in state.extra_strategies else 1.0,
        "momentum": state.momentum if "momentum" in state.extra_strategies else 1.0,
        "iw": state.iw,
        "rw": state.rw,
        "extra_strategies": list(state.extra_strategies),
    }


def _any_process_running(lottery_id: str) -> str | None:
    _cleanup_processes()
    config = _config(lottery_id)
    progress = _read_progress(config)
    active_loop_pid = _canonical_loop_pid(lottery_id, config=config, progress=progress)
    if active_loop_pid:
        status = progress.get("status")
        if status == "starting":
            return f"Auto-loop already starting for {lottery_id} (PID: {active_loop_pid})"
        return f"Auto-loop already running for {lottery_id} (PID: {active_loop_pid})"
    if lottery_id in _run_processes:
        return f"Experiment already running for {lottery_id} (PID: {_run_processes[lottery_id].pid})"
    proc = _loop_processes.get(lottery_id)
    if proc and proc.poll() is None:
        return f"Auto-loop already running for {lottery_id} (PID: {proc.pid})"
    if progress.get("status") in {"running", "stalled", "starting"}:
        pid = progress.get("pid")
        if pid and _pid_alive(int(pid)):
            return f"External experiment already running for {lottery_id} (PID: {pid})"
    return None


def _spawn_background_process(config: LotteryConfig, args: list[str], log_prefix: str) -> subprocess.Popen:
    creation_flags = 0
    if sys.platform == "win32":
        creation_flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        creation_flags |= getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    stdout_path = config.state_dir / f"{log_prefix}.out.log"
    stderr_path = config.state_dir / f"{log_prefix}.err.log"
    with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr_handle:
        return subprocess.Popen(
            args,
            cwd=str(PROJECT_DIR),
            stdout=stdout_handle,
            stderr=stderr_handle,
            creationflags=creation_flags,
        )


async def _wait_for_exit(proc: subprocess.Popen, timeout_s: int) -> bool:
    for _ in range(timeout_s):
        if proc.poll() is not None:
            return True
        await asyncio.sleep(1)
    return proc.poll() is not None


async def _wait_for_pid_exit(pid: int, timeout_s: int) -> bool:
    for _ in range(timeout_s):
        if not _pid_alive(pid):
            return True
        await asyncio.sleep(1)
    return not _pid_alive(pid)


def _terminate_pid(pid: int) -> bool:
    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except OSError:
        pass
    if sys.platform == "win32":
        try:
            completed = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            return completed.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False
    return False


def _write_stopped_progress(config: LotteryConfig, pid: int | None) -> None:
    write_json_resilient(
        config.progress_file,
        {"status": "stopped", "stopped_pid": pid, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
        ensure_ascii=True,
    )


def _write_starting_progress(config: LotteryConfig, pid: int) -> None:
    write_json_resilient(
        config.progress_file,
        {"status": "starting", "pid": pid, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
        ensure_ascii=True,
    )


def _lottery_summary(config: LotteryConfig) -> dict:
    experiments = _parse_results(config)
    draws_count = 0
    try:
        draws_count = len(load_lottery_draws(config.id))
    except Exception:
        draws_count = 0
    progress = _read_progress(config)
    status = progress.get("status", "idle")
    best_metric = 0.0
    for row in experiments:
        metric = row.get("average_hits", 0)
        if isinstance(metric, (int, float)):
            best_metric = max(best_metric, float(metric))
    return {
        "id": config.id,
        "name": config.name,
        "numbers_drawn": config.numbers_drawn,
        "number_range": config.number_range,
        "ticket_size": config.ticket_size,
        "min_pick_count": config.min_ticket_numbers,
        "max_pick_count": config.max_ticket_numbers,
        "bonus_count": config.bonus_count,
        "bonus_range": config.bonus_range,
        "min_bonus_pick_count": config.min_bonus_numbers,
        "max_bonus_pick_count": config.max_bonus_numbers,
        "draw_count": draws_count,
        "experiment_count": len(experiments),
        "best_metric": best_metric,
        "baseline": round(config.baseline, 6),
        "status": status,
        "draw_days": list(config.draw_days) if config.draw_days else [],
        "draw_times_local": list(config.draw_times_local) if config.draw_times_local else [],
        "next_draw_date": config.next_draw_date().isoformat(),
    }


def _weighted_unique_numbers(candidates: list[int], scores: dict[int, float], count: int) -> list[int]:
    remaining = [int(candidate) for candidate in candidates]
    picked: list[int] = []
    while remaining and len(picked) < count:
        weights = [max(float(scores.get(number, 0.0)), 0.001) for number in remaining]
        choice = int(random.choices(remaining, weights=weights, k=1)[0])
        picked.append(choice)
        remaining = [number for number in remaining if number != choice]
    return picked


def _now_local() -> datetime:
    return datetime.now().astimezone()


def _today_iso() -> str:
    return _now_local().date().isoformat()


def _load_ticket_history(config: LotteryConfig) -> list[dict]:
    ensure_lottery_storage(config)
    payload, _source_path = read_json_with_recovery(config.history_file)
    if not isinstance(payload, dict):
        return []
    entries = payload.get("entries", [])
    return [entry for entry in entries if isinstance(entry, dict)]


def _write_ticket_history(config: LotteryConfig, entries: list[dict]) -> None:
    ensure_lottery_storage(config)
    write_json_atomic(config.history_file, {"version": 1, "entries": entries}, ensure_ascii=True)


def _annotate_progress_source(lottery_id: str, data: dict, config: LotteryConfig | None = None) -> dict:
    annotated = dict(data)
    _cleanup_processes()
    config = config or _config(lottery_id)
    loop_pid = _lock_file_has_alive_pid(config) or _progress_auto_loop_pid(config, annotated)
    if lottery_id in _loop_processes or loop_pid:
        annotated["source"] = "auto_loop"
        if loop_pid and not annotated.get("pid"):
            annotated["pid"] = loop_pid
    elif lottery_id in _run_processes:
        annotated["source"] = "ui"
    elif annotated.get("status") in {"running", "stalled", "starting"}:
        annotated["source"] = "external"

    # Add last draw date from CSV
    try:
        from load_data import load_lottery_draws, load_specific_lottery_draws
        draws = load_lottery_draws(lottery_id)
        if draws:
            last_draw = draws[-1]
            last_date = getattr(last_draw, 'draw_date', None)
            if last_date:
                annotated["last_draw_date"] = last_date.strftime("%d/%m/%Y") if hasattr(last_date, 'strftime') else str(last_date)
                annotated["last_draw_iso_date"] = last_date.isoformat() if hasattr(last_date, 'isoformat') else str(last_date)
    except Exception:
        pass

    return annotated


def _progress_snapshot(lottery_id: str, config: LotteryConfig) -> tuple[dict, tuple[str | None, float | None, str | None, str | None]]:
    data = _annotate_progress_source(lottery_id, _read_progress(config), config)
    _raw_data, source_path = read_json_with_recovery(config.progress_file)
    try:
        source_mtime = source_path.stat().st_mtime if source_path else None
    except FileNotFoundError:
        source_mtime = None
    signature = (
        str(source_path) if source_path else None,
        source_mtime,
        data.get("status"),
        data.get("source"),
    )
    return data, signature


def _ticket_signature(main: list[int], bonus: list[int]) -> str:
    return f"{','.join(str(number) for number in sorted(main))}|{','.join(str(number) for number in sorted(bonus))}"


def _draw_lookup_key(draw_date: str | None, draw_id: str | None, draw_time: str | None) -> tuple[str, str, str]:
    return (draw_date or "", draw_id or "", draw_time or "")


def _draw_date_iso(draw_date) -> str:
    if draw_date is None:
        return ""
    isoformat = getattr(draw_date, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    year = int(getattr(draw_date, "year"))
    month = int(getattr(draw_date, "month"))
    day = int(getattr(draw_date, "day"))
    return f"{year:04d}-{month:02d}-{day:02d}"


def _entry_draw_key(entry: dict) -> tuple[str, str, str]:
    return _draw_lookup_key(
        entry.get("target_draw_date") or entry.get("saved_date"),
        entry.get("target_draw_id"),
        entry.get("target_draw_time"),
    )


def _ticket_save_key(main: list[int], bonus: list[int], draw_date: str | None, draw_id: str | None, draw_time: str | None) -> str:
    draw_part = "|".join(_draw_lookup_key(draw_date, draw_id, draw_time))
    return f"{draw_part}|{_ticket_signature(main, bonus)}"


def _evaluate_ticket_outcome(lottery_id: str, main_hits: int, bonus_hits: int, main_count: int) -> dict:
    if lottery_id == "multi_multi_pl":
        # la 10 numere tabelul oficial plătește de la 6 trafienia ȘI pentru 0 trafienia
        thresholds = {1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 4, 9: 4, 10: 6}
        min_hits = thresholds.get(main_count, 99)
        won = main_hits >= min_hits or (main_count == 10 and main_hits == 0)
        return {"won": won, "reason": f"{main_hits} hit{'s' if main_hits != 1 else ''} / min {min_hits}"
                + (" (or 0)" if main_count == 10 else "")}
    if lottery_id == "kaskada_pl":
        # Evaluate the displayed set; a complementary set must be a separate entry.
        return {"won": main_hits >= 8,
                "reason": f"{main_hits} hit{'s' if main_hits != 1 else ''} / min 8"}
    if lottery_id in {"lotto_pl", "mini_lotto_pl"}:
        return {"won": main_hits >= 3, "reason": f"{main_hits} hit{'s' if main_hits != 1 else ''} / min 3"}
    if lottery_id == "loto_6_49_ro":
        return {"won": main_hits >= 3, "reason": f"{main_hits} hit{'s' if main_hits != 1 else ''} / min 3"}
    if lottery_id == "loto_5_40_ro":
        return {"won": main_hits >= 4, "reason": f"{main_hits} hit{'s' if main_hits != 1 else ''} / min 4"}
    if lottery_id == "joker_ro":
        won = main_hits >= 3 or (main_hits >= 1 and bonus_hits >= 1)
        return {"won": won, "reason": f"{main_hits}+{bonus_hits} / min 3+0 or 1+1"}
    if lottery_id == "superenalotto_it":
        # SuperEnalotto plătește de la 2 punti
        return {"won": main_hits >= 2, "reason": f"{main_hits} hit{'s' if main_hits != 1 else ''} / min 2"}
    if lottery_id == "eurojackpot":
        won = main_hits >= 3 or (main_hits >= 2 and bonus_hits >= 1) or (main_hits >= 1 and bonus_hits >= 2)
        return {"won": won, "reason": f"{main_hits}+{bonus_hits} / min 3+0, 2+1, 1+2"}
    if lottery_id == "powerball_us":
        return {"won": bonus_hits >= 1, "reason": "Powerball hit" if bonus_hits >= 1 else "No Powerball"}
    if lottery_id == "mega_millions_us":
        return {"won": bonus_hits >= 1, "reason": "Mega Ball hit" if bonus_hits >= 1 else "No Mega Ball"}
    if lottery_id == "uk_lotto":
        won = main_hits >= 2 or bonus_hits >= 1
        return {"won": won, "reason": f"{main_hits}+{bonus_hits} / min 2+0 or 0+1"}
    if lottery_id == "euromillions":
        won = main_hits >= 3 or (main_hits >= 2 and bonus_hits >= 1) or (main_hits >= 1 and bonus_hits >= 2)
        return {"won": won, "reason": f"{main_hits}+{bonus_hits} / min 3+0, 2+1, 1+2"}
    return {
        "won": False,
        "reason": f"{main_hits} hit{'s' if main_hits != 1 else ''}" + (f", {bonus_hits} bonus" if bonus_hits else ""),
    }


# === History Caching Functions ===

def _hash_entries(entries: list) -> str:
    """Create a hash of the entries to detect changes for caching."""
    import json
    try:
        # Create a stable serialization by sorting keys
        serialized = json.dumps(entries, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode()).hexdigest()
    except Exception:
        # If hashing fails, return a default hash that will cause cache miss
        return "hash_failed_" + str(len(entries))


@functools.lru_cache(maxsize=128)
def _history_payload_cached(config_id: str, entries_hash: str, entries_json: str, draws_signature: str = "") -> dict:
    """Cached version of _history_payload that only recomputes when entries or draws change.

    draws_signature (mtime+size of the draws CSV) is part of the cache key: without
    it a draw downloaded by auto-update or by ai_ticket_agent (another process)
    left the history on "Waiting for draw" until the server restarted.
    """
    import json
    try:
        # Parse entries from JSON string
        entries = json.loads(entries_json)

        # Reconstruct config (this is simplified - in practice we'd pass more config data)
        # For now, we'll recreate the minimal config needed
        from lottery_config import get_lottery_config
        config = get_lottery_config(config_id)

        # Reimplement the core logic from _history_payload but with cached entries
        from collections import defaultdict
        from load_data import load_specific_lottery_draws
        from atomic_io import read_json_with_recovery

        draws_by_date: dict[str, list[object]] = defaultdict(list)
        try:
            # Get unique dates from ticket entries to load only relevant draws
            ticket_dates = {entry.get("target_draw_date") or entry.get("saved_date") for entry in entries}
            ticket_dates.discard(None)

            # Load only the draws we need instead of all draws
            relevant_draws = load_specific_lottery_draws(config_id, ticket_dates, recent_days=30)

            for draw in relevant_draws:
                from datetime import date
                draw_date = _draw_date_iso(draw.draw_date) if hasattr(draw, 'draw_date') else "unknown"
                draws_by_date[draw_date].append(draw)
        except Exception:
            draws_by_date = defaultdict(list)

        grouped_entries: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for entry in sorted(entries, key=lambda item: (item.get("saved_at", ""), item.get("id", "")), reverse=True):
            grouped_entries[_entry_draw_key(entry)].append(entry)

        groups = []
        today = _today_iso()
        today_save_keys: list[str] = []
        for group_key in sorted(grouped_entries.keys(), reverse=True):
            target_draw_date, target_draw_id, target_draw_time = group_key
            group_entries = grouped_entries[group_key]
            slot_index = group_entries[0].get("target_draw_index")
            from draw_matching import match_draw

            draw = match_draw(
                draws_by_date.get(target_draw_date, []), draw_id=target_draw_id,
                draw_time=target_draw_time, draw_index=slot_index,
            )
            draw_numbers = sorted(getattr(draw, "numbers", ())) if draw else []
            draw_bonus = sorted(getattr(draw, "bonuses", ())) if draw else []
            draw_number_set = set(draw_numbers)
            draw_bonus_set = set(draw_bonus)
            tickets = []
            winning_ticket_count = 0
            matched_ticket_count = 0
            for entry in grouped_entries[group_key]:
                main = sorted(int(number) for number in entry.get("main", []))
                bonus = sorted(int(number) for number in entry.get("bonus", []))
                if entry.get("saved_date") == today and entry.get("source") != "ai":
                    today_save_keys.append(
                        _ticket_save_key(
                            main,
                            bonus,
                            entry.get("target_draw_date") or entry.get("saved_date"),
                            entry.get("target_draw_id"),
                            entry.get("target_draw_time"),
                        )
                    )
                if draw:
                    main_hits = sum(1 for number in main if number in draw_number_set)
                    bonus_hits = sum(1 for number in bonus if number in draw_bonus_set)
                    outcome = _evaluate_ticket_outcome(config_id, main_hits, bonus_hits, len(main))
                    if outcome["won"]:
                        winning_ticket_count += 1
                    if main_hits > 0 or bonus_hits > 0:
                        matched_ticket_count += 1
                else:
                    main_hits = 0
                    bonus_hits = 0
                    outcome = {"won": None, "reason": "Waiting for draw"}
                tickets.append(
                    {
                        "id": entry.get("id"),
                        "saved_at": entry.get("saved_at"),
                        "saved_date": entry.get("saved_date"),
                        "main": main,
                        "bonus": bonus,
                        "strategy": entry.get("strategy"),
                        "metric": entry.get("metric"),
                        "target_draw_date": target_draw_date or entry.get("saved_date"),
                        "target_draw_id": target_draw_id or None,
                        "target_draw_time": target_draw_time or None,
                        "main_hits": main_hits,
                        "bonus_hits": bonus_hits,
                        "won": outcome["won"],
                        "reason": outcome["reason"],
                    }
                )
            # Sort tickets: winning first, then by hits descending
            tickets.sort(key=lambda t: (
                0 if t["won"] is True else (1 if t["won"] is False else 2),
                -(t["main_hits"] + t.get("bonus_hits", 0)),
            ))
            draw_count = len(tickets)
            draw_remaining = max(0, MAX_HISTORY_TICKETS_PER_DRAW - draw_count)
            groups.append(
                {
                    "saved_date": group_entries[0].get("saved_date"),
                    "target_draw_date": target_draw_date,
                    "target_draw_id": target_draw_id or None,
                    "target_draw_time": target_draw_time or None,
                    "source": group_entries[0].get("source") or "user",
                    "draw_available": draw is not None,
                    "draw_numbers": draw_numbers,
                    "draw_bonus": draw_bonus,
                    "winning_ticket_count": winning_ticket_count,
                    "matched_ticket_count": matched_ticket_count,
                    "total_tickets": draw_count,
                    "draw_remaining": draw_remaining,
                    "draw_capacity": MAX_HISTORY_TICKETS_PER_DRAW,
                    "tickets": tickets,
                }
            )

        today_count = sum(
            1 for entry in entries
            if entry.get("saved_date") == today and entry.get("source") != "ai"
        )
        today_remaining = max(0, MAX_HISTORY_TICKETS_PER_DAY - today_count)

        return {
            "groups": groups,
            "today_count": today_count,
            "today_remaining": today_remaining,
            "today_save_keys": today_save_keys,
            "capacity": MAX_HISTORY_TICKETS_PER_DRAW,
        }
    except Exception as e:
        # Log error and return empty payload on failure
        print(f"Cache function error: {e}")
        return {
            "groups": [],
            "today_count": 0,
            "today_remaining": 0,
            "today_save_keys": [],
            "capacity": 0,
        }


def _invalidate_history_cache(lottery_id: str) -> None:
    """Mark a lottery's history cache as stale."""
    _history_cache_invalidations.add(lottery_id)


def _is_history_cache_valid(lottery_id: str) -> bool:
    """Check if the cached history for a lottery is still valid."""
    return lottery_id not in _history_cache_invalidations


def _clear_history_cache_flag(lottery_id: str) -> None:
    """Clear the invalidation flag after fresh data is computed."""
    _history_cache_invalidations.discard(lottery_id)


def _ai_history_entries(lottery_id: str) -> list[dict]:
    """Biletele AI din ai_tickets.json, în formatul de istoric salvat.

    Read-only: istoricul salvat al utilizatorului rămâne separat; biletele AI
    apar în Saved Ticket History ca grupuri suplimentare (read-time merge).
    """
    try:
        from ai_tickets_store import load_ticket_store

        store = load_ticket_store(lottery_id)
    except Exception:
        return []
    out: list[dict] = []
    for e in store.get("entries", []):
        if e.get("status") == "cancelled":
            continue
        slot = e.get("draw_index")
        times = [str(t) for t in (e.get("next_draw_times") or ()) if t]
        slot_time = e.get("draw_time") or (times[slot - 1] if slot and 1 <= slot <= len(times) else "")
        draw_date = e.get("draw_date")
        for i, t in enumerate(e.get("tickets", []), 1):
            out.append({
                "id": f"ai|{draw_date}|{slot or 1}|{i}",
                "saved_at": e.get("created_at", ""),
                "saved_date": draw_date,
                "main": list(t.get("main") or []),
                "bonus": list(t.get("bonus") or []),
                "strategy": f"AI #{i}",
                "metric": None,
                "source": "ai",
                "target_draw_date": draw_date,
                "target_draw_time": slot_time or None,
                "target_draw_index": slot,
            })
    return out


def _draws_signature(config) -> str:
    try:
        stat = Path(config.draws_file).stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"
    except (AttributeError, OSError, TypeError):
        return ""


def _history_payload(config: LotteryConfig) -> dict:
    """Generate history payload with intelligent caching."""
    entries = _load_ticket_history(config)
    entries = entries + _ai_history_entries(config.id)
    entries_hash = _hash_entries(entries)

    import json
    try:
        entries_json = json.dumps(entries, default=str)
    except Exception:
        entries_json = "[]"

    # If this lottery was invalidated, clear the full cache and recompute
    if not _is_history_cache_valid(config.id):
        _history_payload_cached.cache_clear()
        _clear_history_cache_flag(config.id)

    return _history_payload_cached(config.id, entries_hash, entries_json, _draws_signature(config))


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(
        await asyncio.to_thread((PROJECT_DIR / "static" / "index.html").read_text, encoding="utf-8"),
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/lotteries")
async def list_lotteries():
    cached = _response_cache.get("lotteries")
    if cached is not None:
        return cached
    result = await asyncio.to_thread(
        lambda: [_lottery_summary(_config(config.id)) for config in list_lottery_configs()]
    )
    _response_cache.set("lotteries", result, ttl=30)
    return result


@app.get("/api/{lottery_id}/experiments")
async def get_experiments(lottery_id: str):
    cache_key = f"experiments:{lottery_id}"
    cached = _response_cache.get(cache_key)
    if cached is not None:
        return cached
    result = await asyncio.to_thread(_parse_results, _config(lottery_id))
    _response_cache.set(cache_key, result, ttl=10)
    return result


@app.get("/api/{lottery_id}/history")
async def get_history(lottery_id: str, bypass_cache: bool = False):
    config = _config(lottery_id)
    if bypass_cache:
        # Clear cache for this specific lottery and force fresh computation
        _invalidate_history_cache(lottery_id)
        # Load fresh data without using cache
        entries = await asyncio.to_thread(_load_ticket_history, config)
        entries = entries + await asyncio.to_thread(_ai_history_entries, lottery_id)
        entries_hash = _hash_entries(entries)
        import json
        try:
            entries_json = json.dumps(entries, default=str)
        except Exception:
            entries_json = "[]"
        # Call cached function but with cleared cache to ensure fresh computation
        result = await asyncio.to_thread(
            _history_payload_cached.__wrapped__, config.id, entries_hash, entries_json, _draws_signature(config)
        )
    else:
        result = await asyncio.to_thread(_history_payload, config)

    # Add cache headers
    response = JSONResponse(result)
    if not bypass_cache:
        response.headers["Cache-Control"] = "public, max-age=30"  # Cache for 30 seconds
    else:
        response.headers["Cache-Control"] = "no-cache"

    return response


@app.post("/api/{lottery_id}/history")
async def add_history_ticket(lottery_id: str, request: Request):
    config = _config(lottery_id)
    payload = await request.json()
    main = sorted({int(number) for number in payload.get("main", [])})
    bonus = sorted({int(number) for number in payload.get("bonus", [])})
    target_draw_date = payload.get("target_draw_date") or config.next_draw_date().isoformat()
    target_draw_id = payload.get("target_draw_id")
    target_draw_time = payload.get("target_draw_time")
    if not main:
        return JSONResponse({"error": "Ticket must contain at least one main number"}, status_code=400)
    if len(main) < config.min_ticket_numbers or len(main) > config.max_ticket_numbers:
        return JSONResponse(
            {"error": f"Ticket size must be between {config.min_ticket_numbers} and {config.max_ticket_numbers}"},
            status_code=400,
        )
    if any(number < 1 or number > config.number_range for number in main):
        return JSONResponse({"error": f"Main numbers must be between 1 and {config.number_range}"}, status_code=400)
    if config.bonus_count:
        if len(bonus) < config.min_bonus_numbers or len(bonus) > config.max_bonus_numbers:
            return JSONResponse(
                {"error": f"Bonus count must be between {config.min_bonus_numbers} and {config.max_bonus_numbers}"},
                status_code=400,
            )
        if any(number < 1 or number > int(config.bonus_range or 0) for number in bonus):
            return JSONResponse({"error": f"Bonus numbers must be between 1 and {config.bonus_range}"}, status_code=400)
    elif bonus:
        return JSONResponse({"error": "This lottery does not use bonus numbers"}, status_code=400)

    entries = _load_ticket_history(config)
    saved_date = _today_iso()

    # Check per-draw limit (20 tickets per draw, identified by date+time)
    # Match on date and time only (draw_id varies and isn't known when saving)
    same_draw_entries = [
        entry for entry in entries
        if (entry.get("target_draw_date") or entry.get("saved_date")) == target_draw_date
        and (entry.get("target_draw_time") or "") == (target_draw_time or "")
    ]
    if len(same_draw_entries) >= MAX_HISTORY_TICKETS_PER_DRAW:
        draw_label = target_draw_date
        if target_draw_time:
            draw_label += f" {target_draw_time}"
        return JSONResponse(
            {"error": f"History already contains {MAX_HISTORY_TICKETS_PER_DRAW} tickets for draw {draw_label}"},
            status_code=409,
        )

    save_key = _ticket_save_key(main, bonus, target_draw_date, target_draw_id, target_draw_time)
    if any(
        _ticket_save_key(
            entry.get("main", []),
            entry.get("bonus", []),
            entry.get("target_draw_date") or entry.get("saved_date"),
            entry.get("target_draw_id"),
            entry.get("target_draw_time"),
        ) == save_key
        for entry in same_draw_entries
    ):
        return JSONResponse({"error": "This ticket is already saved for this draw"}, status_code=409)

    entry = {
        "id": uuid.uuid4().hex,
        "saved_at": _now_local().isoformat(timespec="seconds"),
        "saved_date": saved_date,
        "target_draw_date": target_draw_date,
        "target_draw_id": target_draw_id,
        "target_draw_time": target_draw_time,
        "main": main,
        "bonus": bonus,
        "strategy": payload.get("strategy"),
        "metric": payload.get("metric"),
    }
    entries.append(entry)
    _write_ticket_history(config, entries)

    # Clear the cache for this specific lottery
    _invalidate_history_cache(lottery_id)

    history = _history_payload(config)
    return {
        "status": "saved",
        "ticket_id": entry["id"],
        "today_count": history["today_count"],
        "today_remaining": history["today_remaining"],
    }


@app.delete("/api/{lottery_id}/history/{ticket_id}")
async def delete_history_ticket(lottery_id: str, ticket_id: str):
    config = _config(lottery_id)
    entries = _load_ticket_history(config)
    filtered = [entry for entry in entries if entry.get("id") != ticket_id]
    if len(filtered) == len(entries):
        return JSONResponse({"error": "History ticket not found"}, status_code=404)
    _write_ticket_history(config, filtered)

    # Clear the cache for this specific lottery
    _invalidate_history_cache(lottery_id)

    history = _history_payload(config)
    return {"status": "deleted", "today_count": history["today_count"], "today_remaining": history["today_remaining"]}


@app.get("/api/{lottery_id}/progress")
async def get_progress(lottery_id: str):
    cache_key = f"progress:{lottery_id}"
    cached = _response_cache.get(cache_key)
    if cached is not None:
        return cached
    config = _config(lottery_id)
    data, _signature = await asyncio.to_thread(_progress_snapshot, lottery_id, config)
    _response_cache.set(cache_key, data, ttl=3)
    return data


@app.get("/api/{lottery_id}/progress/stream")
async def progress_stream(lottery_id: str, request: Request):
    config = _config(lottery_id)

    async def event_gen():
        last_signature: tuple[str | None, float | None, str | None, str | None] | None = None
        while True:
            if await request.is_disconnected():
                break
            data, signature = _progress_snapshot(lottery_id, config)
            if signature != last_signature:
                last_signature = signature
                yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(1.5)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/{lottery_id}/run")
async def run_once(lottery_id: str):
    async with _get_lock():
        conflict = _any_process_running(lottery_id)
        if conflict:
            return JSONResponse({"error": conflict}, status_code=409)
        config = _config(lottery_id)
        python_exe = sys.executable
        proc = _spawn_background_process(
            config,
            [python_exe, str(PROJECT_DIR / "auto_loop.py"), "--lottery", config.id, "--once"],
            "run",
        )
        _run_processes[lottery_id] = proc
        return {"status": "started", "pid": proc.pid}


@app.post("/api/{lottery_id}/stop")
async def stop_experiment(lottery_id: str):
    config = _config(lottery_id)
    _cleanup_processes()
    proc = _run_processes.pop(lottery_id, None)
    pid = None
    if proc and proc.poll() is None:
        pid = proc.pid
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        _write_stopped_progress(config, pid)
        return {"status": "stopped", "pid": pid}
    progress = _read_progress(config)
    pid = progress.get("pid")
    if progress.get("status") in {"running", "stalled"} and pid and _pid_alive(int(pid)):
        config.stop_request_file.write_text("stop\n", encoding="utf-8")
        stopped = await _wait_for_pid_exit(int(pid), 12)
        if not stopped:
            _terminate_pid(int(pid))
            await _wait_for_pid_exit(int(pid), 5)
        _write_stopped_progress(config, int(pid))
        return {"status": "stopped", "pid": int(pid)}
    return JSONResponse({"error": "No running experiment found"}, status_code=404)


@app.post("/api/{lottery_id}/loop/start")
async def start_loop(lottery_id: str):
    async with _get_lock():
        conflict = _any_process_running(lottery_id)
        if conflict:
            return JSONResponse({"error": conflict}, status_code=409)
        config = _config(lottery_id)
        # Resectează pauza — utilizatorul vrea explicit bucla pornită
        try:
            config.state_dir.joinpath(".autoloop.paused").unlink(missing_ok=True)
        except OSError:
            pass
        python_exe = sys.executable
        proc = _spawn_background_process(
            config,
            [python_exe, str(PROJECT_DIR / "auto_loop.py"), "--lottery", config.id],
            "loop",
        )
        _write_starting_progress(config, proc.pid)
        _loop_processes[lottery_id] = proc
        return {"status": "loop_started", "pid": proc.pid}


@app.post("/api/{lottery_id}/loop/stop")
async def stop_loop(lottery_id: str):
    config = _config(lottery_id)
    # Marchează pauza ca lottery_manager să nu repornească bucla automat
    try:
        state_dir = Path(getattr(config, "state_dir", None) or (PROJECT_DIR / "data" / lottery_id))
        state_dir.mkdir(parents=True, exist_ok=True)
        state_dir.joinpath(".autoloop.paused").write_text(
            datetime.now().isoformat(timespec="seconds"), encoding="utf-8"
        )
    except OSError:
        pass
    _cleanup_processes()
    progress = _read_progress(config)
    pids = _active_loop_pids(lottery_id, config=config, progress=progress)
    proc = _loop_processes.pop(lottery_id, None)
    if pids:
        config.stop_request_file.write_text("stop\n", encoding="utf-8")
        primary_pid = int(pids[0])
        if proc and proc.poll() is None and proc.pid == primary_pid:
            graceful = await _wait_for_exit(proc, 12)
            if not graceful:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        for pid in pids:
            graceful = await _wait_for_pid_exit(int(pid), 12)
            if not graceful:
                _terminate_pid(int(pid))
                await _wait_for_pid_exit(int(pid), 5)
        _write_stopped_progress(config, primary_pid)
        return {"status": "loop_stopped", "pid": primary_pid}
    return JSONResponse({"error": "No loop running"}, status_code=404)


@app.get("/api/{lottery_id}/loop/status")
async def loop_status(lottery_id: str):
    _cleanup_processes()
    config = _config(lottery_id)
    progress = _read_progress(config)
    loop_pid = _canonical_loop_pid(lottery_id, config=config, progress=progress)
    if loop_pid:
        return {"running": True, "pid": loop_pid}
    return {"running": False}


@app.get("/api/{lottery_id}/generate")
async def generate_numbers(
    lottery_id: str,
    ticket_count: int = 10,
    nums_per_ticket: int | None = None,
    bonus_per_ticket: int | None = None,
):
    try:
        config = _config(lottery_id)
    except KeyError:
        return JSONResponse({"error": f"Unknown lottery: {lottery_id}"}, status_code=404)
    try:
        ticket_count = max(1, min(20, ticket_count))
        nums_per_ticket = nums_per_ticket or config.ticket_size
        nums_per_ticket = max(config.min_ticket_numbers, min(config.max_ticket_numbers, nums_per_ticket))
        if config.bonus_range:
            default_bonus = config.bonus_count or config.min_bonus_numbers
            bonus_per_ticket = bonus_per_ticket or default_bonus
            bonus_per_ticket = max(config.min_bonus_numbers, min(config.max_bonus_numbers, bonus_per_ticket))
        else:
            bonus_per_ticket = 0

        draws = await asyncio.to_thread(load_lottery_draws, config.id)
        if not draws:
            return JSONResponse({"error": "No draws available"}, status_code=400)

        state, _active = await asyncio.to_thread(auto_loop.load_search_state_for_lottery, config.id)
        if state.best_metric <= 0:
            state = await asyncio.to_thread(auto_loop.load_best_from_results_for_lottery, config.id)

        best = _get_best_strategy(config.id)
        ranked = await asyncio.to_thread(auto_loop.rank_numbers_from_draws, draws, state, lottery_id=config.id)
        ranked = [number for number in ranked if 1 <= int(number) <= config.number_range]
        if len(ranked) < config.number_range:
            seen_ranked = set(int(number) for number in ranked)
            ranked.extend(number for number in range(1, config.number_range + 1) if number not in seen_ranked)
        pool_size = min(config.number_range, max(nums_per_ticket * 2, nums_per_ticket + ticket_count * 3))
        pool = ranked[:pool_size]

        tickets = [sorted(pool[:nums_per_ticket])]
        seen = {tuple(tickets[0])}
        for index in range(1, ticket_count):
            shift = min(index * max(1, nums_per_ticket // 3), max(0, pool_size - nums_per_ticket))
            window = pool[shift:shift + nums_per_ticket + ticket_count]
            n_anchor = max(1, nums_per_ticket // 3)
            n_explore = nums_per_ticket - n_anchor
            anchor = random.sample(pool[:nums_per_ticket], n_anchor)
            remaining = [n for n in window if n not in anchor]
            explore = random.sample(remaining, min(n_explore, len(remaining))) if remaining else []
            ticket = anchor + explore
            if len(ticket) < nums_per_ticket:
                fill = [n for n in pool if n not in ticket]
                ticket += fill[:nums_per_ticket - len(ticket)]
            ticket = sorted(ticket[:nums_per_ticket])
            if tuple(ticket) in seen:
                ticket = sorted(random.sample(pool, nums_per_ticket))
            seen.add(tuple(ticket))
            tickets.append(ticket)

        bonus_ranked = []
        bonus_scores: dict[int, float] = {}
        if config.bonus_count and config.bonus_range:
            bonus_counts = {n: 0 for n in range(1, config.bonus_range + 1)}
            recent_bonus_counts = {n: 0 for n in range(1, config.bonus_range + 1)}
            recent_bonus_draws = draws[-min(100, len(draws)):]
            for draw in draws:
                for bonus in draw.bonuses:
                    if 1 <= bonus <= config.bonus_range:
                        bonus_counts[bonus] += 1
            for draw in recent_bonus_draws:
                for bonus in draw.bonuses:
                    if 1 <= bonus <= config.bonus_range:
                        recent_bonus_counts[bonus] += 1
            bonus_scores = {
                number: (0.7 * bonus_counts[number]) + (0.3 * recent_bonus_counts[number])
                for number in range(1, config.bonus_range + 1)
            }
            bonus_ranked = sorted(
                range(1, config.bonus_range + 1),
                key=lambda n: (-bonus_scores[n], n),
            )

        ticket_payload = []
        used_bonus = set()
        for index, main_ticket in enumerate(tickets):
            bonuses = []
            if bonus_ranked:
                bonus_pool_size = min(
                    len(bonus_ranked),
                    max(8, bonus_per_ticket * 6, ticket_count * 2),
                )
                bonus_pool = bonus_ranked[:bonus_pool_size]
                bonuses = _weighted_unique_numbers(bonus_pool, bonus_scores, bonus_per_ticket)
                if len(bonuses) < bonus_per_ticket:
                    fill = [n for n in bonus_ranked if n not in bonuses]
                    bonuses.extend(fill[:bonus_per_ticket - len(bonuses)])
                if tuple(sorted(bonuses)) in used_bonus:
                    remaining = [n for n in bonus_pool if n not in bonuses]
                    extra = _weighted_unique_numbers(remaining, bonus_scores, bonus_per_ticket)
                    if extra:
                        bonuses = extra
                if tuple(sorted(bonuses)) in used_bonus:
                    for candidate in bonus_ranked:
                        if candidate not in bonuses:
                            bonuses = [candidate] if bonus_per_ticket == 1 else sorted((bonuses + [candidate])[:bonus_per_ticket])
                            if tuple(sorted(bonuses)) not in used_bonus:
                                break
                used_bonus.add(tuple(bonuses))
            ticket_payload.append({"main": main_ticket, "bonus": sorted(bonuses)})

        last_draw = draws[-1]
        draw_date = f"{last_draw.draw_date.day:02d}/{last_draw.draw_date.month:02d}/{last_draw.draw_date.year}"
        # Calculate next draw date — the date tickets should be saved for
        next_draw = config.next_draw_date()
        next_draw_display = f"{next_draw.day:02d}/{next_draw.month:02d}/{next_draw.year}"
        return {
            "lottery": config.id,
            "strategy": best["name"],
            "metric": best["metric"],
            "weights": {
                "tf": best["tf"],
                "hc": best["hc"],
                "b": best["b"],
                "sb": best["sb"],
                "delay": best["delay"],
                "momentum": best["momentum"],
            },
            "windows": {"iw": best["iw"], "rw": best["rw"]},
            "strategies": best["extra_strategies"],
            "ticket_count": ticket_count,
            "nums_per_ticket": nums_per_ticket,
            "bonus_per_ticket": bonus_per_ticket,
            "tickets": ticket_payload,
            "last_draw_date": draw_date,
            "last_draw_iso_date": _draw_date_iso(last_draw.draw_date),
            "last_draw_id": getattr(last_draw, "draw_identifier", None),
            "last_draw_time": getattr(last_draw, "draw_time", None),
            "last_draw_numbers": sorted(last_draw.numbers),
            "last_draw_bonus": sorted(getattr(last_draw, "bonuses", ())),
            "next_draw_date": next_draw_display,
            "next_draw_iso_date": next_draw.isoformat(),
            "next_draw_times": list(config.draw_times_local) if config.draw_times_local else [],
            "draw_days": list(config.draw_days) if config.draw_days else [],
            "mode": "system" if nums_per_ticket > config.ticket_size or bonus_per_ticket > config.bonus_count else "simple",
            "rules": {
                "min_pick_count": config.min_ticket_numbers,
                "max_pick_count": config.max_ticket_numbers,
                "bonus_count": config.bonus_count,
                "bonus_range": config.bonus_range,
                "min_bonus_pick_count": config.min_bonus_numbers,
                "max_bonus_pick_count": config.max_bonus_numbers,
            },
        }
    except Exception as exc:
        import traceback

        return JSONResponse({"error": str(exc), "detail": traceback.format_exc()}, status_code=500)


DEFAULT_LOTTERY = "multi_multi_pl"


@app.get("/api/experiments")
async def default_experiments():
    return await get_experiments(DEFAULT_LOTTERY)


@app.get("/api/progress")
async def default_progress():
    return await get_progress(DEFAULT_LOTTERY)


@app.get("/api/progress/stream")
async def default_progress_stream(request: Request):
    return await progress_stream(DEFAULT_LOTTERY, request)


@app.post("/api/run")
async def default_run():
    return await run_once(DEFAULT_LOTTERY)


@app.post("/api/stop")
async def default_stop():
    return await stop_experiment(DEFAULT_LOTTERY)


@app.post("/api/loop/start")
async def default_loop_start():
    return await start_loop(DEFAULT_LOTTERY)


@app.post("/api/loop/stop")
async def default_loop_stop():
    return await stop_loop(DEFAULT_LOTTERY)


@app.get("/api/loop/status")
async def default_loop_status():
    return await loop_status(DEFAULT_LOTTERY)


@app.get("/api/generate")
async def default_generate(ticket_count: int = 10, nums_per_ticket: int = 10, bonus_per_ticket: int | None = None):
    return await generate_numbers(DEFAULT_LOTTERY, ticket_count=ticket_count, nums_per_ticket=nums_per_ticket, bonus_per_ticket=bonus_per_ticket)


@app.get("/api/{lottery_id}/best-strategy")
async def get_lottery_best_strategy(lottery_id: str):
    """Return the current best strategy for a lottery (for UI badge)."""
    try:
        config = _config(lottery_id)
    except KeyError:
        return JSONResponse({"error": f"Unknown lottery: {lottery_id}"}, status_code=404)
    best = _get_best_strategy(lottery_id)
    payload: dict = {
        "lottery": lottery_id,
        "name": best["name"],
        "metric": best["metric"],
        "weights": {
            "tf": best["tf"],
            "hc": best["hc"],
            "b": best["b"],
            "sb": best.get("sb", 1.0),
            "delay": best.get("delay", 1.0),
            "momentum": best.get("momentum", 1.0),
        },
        "windows": {"iw": best["iw"], "rw": best["rw"]},
        "extra_strategies": best["extra_strategies"],
    }
    # ultima extragere + următoarea, pentru linia „Ultima extragere / Numere"
    try:
        from load_data import load_lottery_draws

        draws = load_lottery_draws(lottery_id)
        if draws:
            last_draw = draws[-1]
            last_date = getattr(last_draw, "draw_date", None)
            if last_date:
                payload["last_draw_date"] = (
                    last_date.strftime("%d/%m/%Y")
                    if hasattr(last_date, "strftime") else str(last_date)
                )
                payload["last_draw_iso_date"] = (
                    last_date.isoformat() if hasattr(last_date, "isoformat") else str(last_date)
                )
            payload["last_draw_numbers"] = sorted(getattr(last_draw, "numbers", ()) or ())
            payload["last_draw_bonus"] = sorted(getattr(last_draw, "bonuses", ()) or ())
        next_date = config.next_draw_date()
        if next_date:
            payload["next_draw_date"] = (
                next_date.strftime("%d/%m/%Y")
                if hasattr(next_date, "strftime") else str(next_date)
            )
    except Exception:
        pass
    return payload


# === Ollama AI integration ===

_ai_analysis_lock = threading.Lock()


@app.get("/api/ai/status")
async def ai_status():
    """Ollama availability, models and resolved default model."""
    try:
        import ollama_ai

        return {
            "host": ollama_ai.OLLAMA_HOST,
            "available": ollama_ai.ollama_available(),
            "models": ollama_ai.list_models(),
            "resolved_model": ollama_ai.resolve_model(),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


@app.get("/api/{lottery_id}/ai-predictions")
async def ai_predictions(lottery_id: str):
    """Latest AI (Ollama) analysis + proposed numbers for the next draw."""
    try:
        _config(lottery_id)
    except KeyError:
        return JSONResponse({"error": f"Unknown lottery: {lottery_id}"}, status_code=404)
    try:
        import ollama_ai

        record = ollama_ai.load_predictions(lottery_id)
        return {"lottery": lottery_id, "prediction": record}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/{lottery_id}/ai-analyze")
async def ai_analyze(lottery_id: str):
    """Trigger an immediate Ollama analysis for a lottery (runs in a worker thread)."""
    try:
        config = _config(lottery_id)
    except KeyError:
        return JSONResponse({"error": f"Unknown lottery: {lottery_id}"}, status_code=404)

    if not _ai_analysis_lock.acquire(blocking=False):
        return JSONResponse({"error": "An AI analysis is already running"}, status_code=409)
    try:
        import ollama_ai

        if not ollama_ai.ollama_available():
            return JSONResponse({"error": f"Ollama not reachable at {ollama_ai.OLLAMA_HOST}"}, status_code=503)
        draws = await asyncio.to_thread(load_lottery_draws, config.id)
        if not draws:
            return JSONResponse({"error": "No draws available — download history first"}, status_code=400)

        def _run() -> dict:
            model = ollama_ai.resolve_model()
            return ollama_ai.analyze_lottery(lottery_id, model=model)  # type: ignore[arg-type]

        try:
            record = await asyncio.wait_for(asyncio.to_thread(_run), timeout=900)
        except asyncio.TimeoutError:
            return JSONResponse({"error": "AI analysis timed out (15 min)"}, status_code=504)
        return {"lottery": lottery_id, "prediction": record}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        _ai_analysis_lock.release()


@app.post("/api/{lottery_id}/scrape-draws")
async def scrape_draws(lottery_id: str):
    """Scrape the latest draw results for a lottery and update the CSV data file.
    Returns the number of new draws found and the date range.
    """
    try:
        config = _config(lottery_id)
    except KeyError:
        return JSONResponse({"error": f"Unknown lottery: {lottery_id}"}, status_code=404)

    try:
        from scrape_lotteries import download_lottery, SCRAPER_MAP, LOTTERIES as SCRAPE_LOTTERIES

        if lottery_id not in SCRAPER_MAP:
            return JSONResponse({"error": f"No scraper available for {lottery_id}"}, status_code=400)

        # Record current file state before scraping
        csv_path = config.draws_file
        old_mtime = csv_path.stat().st_mtime if csv_path.exists() else 0
        old_lines = 0
        if csv_path.exists():
            with open(csv_path, "r", encoding="utf-8") as f:
                old_lines = sum(1 for _ in f) - 1  # subtract header

        # Run the scraper (synchronous — blocks the event loop briefly)
        result = await asyncio.to_thread(download_lottery, lottery_id)

        # Check what changed
        new_lines = 0
        if csv_path.exists():
            with open(csv_path, "r", encoding="utf-8") as f:
                new_lines = sum(1 for _ in f) - 1  # subtract header

        new_draws = max(0, new_lines - old_lines)

        # Invalidate caches
        _invalidate_history_cache(lottery_id)
        _response_cache.set(f"lotteries", None, ttl=0)  # Force refresh lotteries list
        # Also invalidate the draws cache in load_data
        from load_data import _load_lottery_draws_cached
        _load_lottery_draws_cached.cache_clear()

        # Reload draws to get the last draw date
        draws = await asyncio.to_thread(load_lottery_draws, lottery_id)
        last_draw_date = None
        next_draw = config.next_draw_date()
        if draws:
            last_draw_date = _draw_date_iso(draws[-1].draw_date)

        return {
            "lottery": lottery_id,
            "new_draws": new_draws,
            "total_draws": new_lines,
            "last_draw_date": last_draw_date,
            "next_draw_date": next_draw.isoformat(),
            "status": "updated" if new_draws > 0 else "no_new_draws",
        }
    except Exception as exc:
        import traceback
        return JSONResponse(
            {"error": str(exc), "detail": traceback.format_exc()},
            status_code=500,
        )


@app.get("/api/reality-check")
async def reality_check_all():
    """Edge vs zgomot pentru toate loteriile (backtest, walkforward, bilete AI reale)."""
    cached = _response_cache.get("reality-check")
    if cached is not None:
        return cached
    import significance

    def build():
        out = []
        for config in list_lottery_configs():
            try:
                out.append(significance.lottery_reality_check(config.id))
            except Exception as exc:
                out.append({"lottery": config.id, "name": config.name, "error": str(exc)[:200]})
        return out

    result = {"lotteries": await asyncio.to_thread(build),
              "timestamp": datetime.now().isoformat(timespec="seconds")}
    _response_cache.set("reality-check", result, ttl=300)
    return result


@app.get("/api/{lottery_id}/reality-check")
async def reality_check_one(lottery_id: str):
    try:
        _config(lottery_id)
    except KeyError:
        return JSONResponse({"error": f"Unknown lottery: {lottery_id}"}, status_code=404)
    import significance

    return await asyncio.to_thread(significance.lottery_reality_check, lottery_id)


@app.get("/api/walkforward-results")
async def get_walkforward_results():
    """Return walkforward results for all lotteries."""
    results = {}
    for config in list_lottery_configs():
        wf = load_walkforward_result(config)
        if wf:
            results[config.id] = wf
    return {"all_results": results, "timestamp": datetime.now().isoformat(timespec="seconds")}


# === Automatic draw-history updates ===

AUTO_UPDATE_STATE_FILE = PROJECT_DIR / "data" / ".auto_update.json"
AUTO_UPDATE_INTERVAL_SECONDS = 6 * 3600  # refresh history every 6 hours
AUTO_UPDATE_BOOT_DELAY_SECONDS = 20      # let agents/locks settle before first scrape
_auto_update_task: asyncio.Task | None = None


def _read_auto_update_state() -> dict:
    data, _ = read_json_with_recovery(AUTO_UPDATE_STATE_FILE)
    return data if isinstance(data, dict) else {}


def _write_auto_update_state(state: dict) -> None:
    try:
        write_json_atomic(AUTO_UPDATE_STATE_FILE, state, ensure_ascii=True)
    except Exception as exc:
        print(f"[auto-update] could not persist state: {exc}")


def _invalidate_draw_caches(lottery_id: str) -> None:
    _invalidate_history_cache(lottery_id)
    from load_data import _load_lottery_draws_cached
    _load_lottery_draws_cached.cache_clear()


async def _auto_update_history_loop() -> None:
    """Background loop: download/refresh draw history for every lottery.

    Runs shortly after boot (if the last update is stale) and then every
    AUTO_UPDATE_INTERVAL_SECONDS. Sequential per lottery to stay polite.
    """
    await asyncio.sleep(AUTO_UPDATE_BOOT_DELAY_SECONDS)
    while True:
        state = _read_auto_update_state()
        last_run = float(state.get("last_completed", 0))
        if time.time() - last_run < AUTO_UPDATE_INTERVAL_SECONDS:
            await asyncio.sleep(max(60.0, AUTO_UPDATE_INTERVAL_SECONDS - (time.time() - last_run)))
            continue

        print("[auto-update] refreshing draw history for all lotteries...")
        state["last_started"] = time.time()
        state["status"] = "running"
        _write_auto_update_state(state)

        try:
            from scrape_lotteries import SCRAPER_MAP, download_lottery

            results = dict(state.get("results", {}))
            for config in list_lottery_configs():
                lottery_id = config.id
                if lottery_id not in SCRAPER_MAP:
                    continue
                started = time.time()
                try:
                    csv_path = config.draws_file
                    old_lines = 0
                    if csv_path.exists():
                        with open(csv_path, "r", encoding="utf-8") as f:
                            old_lines = max(0, sum(1 for _ in f) - 1)
                    await asyncio.to_thread(download_lottery, lottery_id)
                    new_lines = 0
                    if csv_path.exists():
                        with open(csv_path, "r", encoding="utf-8") as f:
                            new_lines = max(0, sum(1 for _ in f) - 1)
                    _invalidate_draw_caches(lottery_id)
                    results[lottery_id] = {
                        "draws": new_lines,
                        "added": max(0, new_lines - old_lines),
                        "ok": True,
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                        "elapsed_s": round(time.time() - started, 1),
                    }
                    print(f"[auto-update] {lottery_id}: {new_lines} draws (+{max(0, new_lines - old_lines)})")
                except Exception as exc:
                    results[lottery_id] = {
                        "ok": False,
                        "error": str(exc)[:200],
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    print(f"[auto-update] {lottery_id}: FAILED - {exc}")
                state["results"] = results
                _write_auto_update_state(state)

            state["last_completed"] = time.time()
            state["status"] = "completed"
            state.pop("error", None)
            _write_auto_update_state(state)
            print("[auto-update] history refresh completed")
            try:
                # istoric nou → rescanează anomaliile și valoarea așteptată la reportul curent
                import discovery_cycle

                discovery = await asyncio.to_thread(discovery_cycle.run_cycle)
                _response_cache.invalidate("reality-check")
                print(f"[discovery] {discovery.get('summary', '')}")
            except Exception as exc:
                print(f"[discovery] cycle failed: {exc}")
        except Exception as exc:
            state["status"] = "failed"
            state["error"] = str(exc)[:200]
            state["last_completed"] = time.time()  # avoid hot-looping on repeated failures
            _write_auto_update_state(state)
            print(f"[auto-update] cycle failed: {exc}")

        await asyncio.sleep(60)


@app.get("/api/discovery")
async def discovery_status():
    """Ultimul ciclu de descoperire: anomalii reale + valoarea așteptată a biletelor."""
    import discovery_cycle

    state = discovery_cycle.load_state()
    return state or {"status": "never_run", "summary": "Ciclul de descoperire nu a rulat încă."}


@app.post("/api/discovery/run")
async def discovery_run():
    import discovery_cycle

    try:
        state = await asyncio.to_thread(discovery_cycle.run_cycle)
    except Exception as exc:
        return JSONResponse({"error": str(exc)[:300]}, status_code=500)
    _response_cache.invalidate("reality-check")
    return state


@app.get("/api/auto-update/status")
async def auto_update_status():
    """Status of the automatic draw-history updater."""
    state = _read_auto_update_state()
    now = time.time()
    last = float(state.get("last_completed", 0))
    return {
        "status": state.get("status", "idle"),
        "last_completed_ago_s": round(now - last) if last else None,
        "interval_s": AUTO_UPDATE_INTERVAL_SECONDS,
        "next_run_in_s": max(0, round(AUTO_UPDATE_INTERVAL_SECONDS - (now - last))) if last else None,
        "results": state.get("results", {}),
    }


@app.post("/api/auto-update/run")
async def auto_update_run():
    """Force an immediate history refresh for all lotteries (non-blocking)."""
    state = _read_auto_update_state()
    state["last_completed"] = 0  # due immediately
    _write_auto_update_state(state)
    return {"status": "scheduled", "note": "The background updater will start within ~60s"}


# === Decision Ledger, AI tickets & agent activity ===


@app.get("/api/ledger")
async def ledger_global(limit: int = 60, entry_type: str | None = None):
    """Ultimele decizii AI (flux global, cele mai noi primele)."""
    from decision_ledger import read_entries

    return {"entries": read_entries(None, limit=limit, entry_type=entry_type)}


@app.get("/api/ai-actions")
async def ai_actions_view():
    """Acțiunile AI: în așteptarea aprobării + istoricul executat/respinse."""
    import ai_actions

    return {
        "pending": ai_actions.list_pending(),
        "history": ai_actions.list_history(limit=30),
        "telegram_configured": True,
    }


@app.post("/api/ai-actions/{action_id}/approve")
async def ai_action_approve(action_id: str):
    import ai_actions

    result = ai_actions.approve_action(action_id, approved_by="dashboard")
    return result


@app.post("/api/ai-actions/{action_id}/reject")
async def ai_action_reject(action_id: str):
    import ai_actions

    return ai_actions.reject_action(action_id, by="dashboard")


@app.get("/api/{lottery_id}/ledger")
async def ledger_for_lottery(lottery_id: str, limit: int = 60, entry_type: str | None = None):
    try:
        _config(lottery_id)
    except KeyError:
        return JSONResponse({"error": f"Unknown lottery: {lottery_id}"}, status_code=404)
    from decision_ledger import read_entries

    return {"lottery": lottery_id, "entries": read_entries(lottery_id, limit=limit, entry_type=entry_type)}


@app.get("/api/{lottery_id}/ai-tickets")
async def ai_tickets(lottery_id: str):
    """Biletele propuse de AI (+ rezultatele de după extragere)."""
    try:
        _config(lottery_id)
    except KeyError:
        return JSONResponse({"error": f"Unknown lottery: {lottery_id}"}, status_code=404)
    from ai_tickets_store import load_ticket_store

    store = load_ticket_store(lottery_id)
    entries = sorted(store.get("entries", []), key=lambda e: e.get("draw_date", ""), reverse=True)
    return {"lottery": lottery_id, "entries": entries}


@app.get("/api/{lottery_id}/ai-ticket-history")
async def ai_ticket_history(lottery_id: str):
    """Istoricul biletelor AI + progresul cumulat (hits, cost estimat, câștig estimat)."""
    try:
        _config(lottery_id)
    except KeyError:
        return JSONResponse({"error": f"Unknown lottery: {lottery_id}"}, status_code=404)
    from ai_tickets_store import load_ticket_store, summarize_tickets
    from prize_estimates import estimate_ticket_prize

    store = load_ticket_store(lottery_id)
    entries = []
    for e in sorted(store.get("entries", []), key=lambda x: (
            x.get("draw_date", ""), x.get("draw_index") or 0), reverse=True):
        results = e.get("results") or []
        if results:
            total_hits = sum(int(r.get("hit_count", 0) or 0) for r in results)
            best_hits = max(int(r.get("hit_count", 0) or 0) for r in results)
            best_ticket = next(r.get("main") for r in results
                               if int(r.get("hit_count", 0) or 0) == best_hits)
            # Recalculate with current rules, as summarize_tickets does.
            amount, currency = 0.0, ""
            for r in results:
                p = estimate_ticket_prize(
                    lottery_id, int(r.get("hit_count", 0) or 0),
                    len(r.get("bonus_hits") or []))
                if p and p.get("amount"):
                    amount += p["amount"]
                    currency = currency or p["currency"]
            prize = {"amount": round(amount, 2), "currency": currency}
        else:
            total_hits = None  # extragerea nu a avut loc încă
            best_hits = None
            best_ticket = None
            prize = {"amount": 0.0, "currency": ""} if e.get("status") == "pending" else None
        entries.append({
            "draw_date": e.get("draw_date"),
            "draw_index": e.get("draw_index"),
            "draw_time": e.get("draw_time"),
            "n_draw_times": len(e.get("next_draw_times") or ()),
            "status": e.get("status"),
            "draw_numbers": e.get("draw_numbers"),
            "draw_bonus": e.get("draw_bonus"),
            "n_tickets": len(e.get("tickets", [])),
            "total_hits": total_hits,
            "best_ticket": best_ticket,
            "best_hits": best_hits,
            "estimated_prize": prize,
        })
    return {"lottery": lottery_id, "entries": entries,
            "summary": summarize_tickets(lottery_id, store)}


_activity_state: dict[str, dict] = {}


def _activity_since(key: str, state: str, task: str) -> float:
    """Ține evidența de când rulează (source, state, task) curent."""
    import time as _time

    now = _time.time()
    prev = _activity_state.get(key)
    if not prev or prev.get("state") != state or prev.get("task") != task:
        _activity_state[key] = {"state": state, "task": task, "since": now}
        return now
    return float(prev.get("since", now))


@app.get("/api/agent-activity")
async def agent_activity():
    """Starea globală a agentului AI: ce face acum și de când.

    State precedence: coding > researching > thinking > working > idle.
    """
    from decision_ledger import read_entries

    details: list[dict] = []

    def _agent_state(agent_id: str, keywords: dict[str, str]) -> dict | None:
        status = _get_agent_status(agent_id)
        if not status.get("running"):
            return None
        task = str(status.get("current_task") or "").lower()
        state = "working"
        for needle, mapped in keywords.items():
            if needle in task:
                state = mapped
                break
        task_text = str(status.get("current_task") or agent_id)
        since = _activity_since(agent_id, state, task_text)
        return {"source": agent_id, "state": state, "task": task_text,
                "since": round(since), "ts": status.get("last_activity")}

    coder = _agent_state("ai_strategy_coder", {"scrie strategia": "coding", "backtest": "working"})
    if coder:
        coder["state"] = "coding"
    strategist = _agent_state("ollama_strategist", {"analyzing": "thinking", "ai analyzing": "thinking"})
    if strategist:
        if strategist["state"] == "thinking":
            pass
        elif "cercet" in strategist["task"].lower() or "research" in strategist["task"].lower():
            strategist["state"] = "researching"
        else:
            strategist["state"] = "thinking"
    ticket_agent = _agent_state("ai_ticket_agent", {"construiește": "thinking", "verific": "working"})

    for item in (coder, strategist, ticket_agent):
        if item:
            details.append(item)

    # buclele auto_loop active
    loops_running = 0
    for config in list_lottery_configs():
        lock_pid = _lock_file_has_alive_pid(config)
        if not lock_pid:
            continue
        loops_running += 1
        progress = _read_progress(config)
        if isinstance(progress, dict) and progress.get("status") == "running":
            task = f"optimizează {config.id}: {str(progress.get('current_config') or '')[:60]}"
            elapsed = float(progress.get("elapsed_seconds") or 0)
        else:
            task = f"optimizează {config.id}"
            elapsed = 0
        since = _activity_since(f"loop:{config.id}", "working", task)
        details.append({"source": f"loop:{config.id}", "state": "working",
                        "task": task, "since": since,
                        "elapsed_s": round(elapsed, 1)})

    # stare agregată (prioritate: coding > thinking/researching > working > idle)
    priority = {"coding": 0, "researching": 1, "thinking": 2, "working": 3, "idle": 4}
    if details:
        best = min(details, key=lambda d: priority.get(d.get("state", "idle"), 5))
        overall = best["state"]
        overall_task = best["task"]
        overall_since = best["since"]
    else:
        overall, overall_task, overall_since = "idle", "în așteptare", time.time()

    last_ledger = read_entries(None, limit=1)
    return {
        "state": overall,
        "task": overall_task,
        "since": overall_since,
        "loops_running": loops_running,
        "details": details,
        "last_decision": last_ledger[0] if last_ledger else None,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


@app.get("/api/{lottery_id}/walkforward")
async def get_lottery_walkforward(lottery_id: str):
    """Return walkforward results for a specific lottery."""
    try:
        config = _config(lottery_id)
    except KeyError:
        return JSONResponse({"error": f"Unknown lottery: {lottery_id}"}, status_code=404)
    result = load_walkforward_result(config)
    if result is None:
        return JSONResponse({"error": f"No walkforward data for lottery '{lottery_id}'"}, status_code=404)
    return result


@app.post("/api/{lottery_id}/run-walkforward")
async def run_walkforward_endpoint(lottery_id: str):
    """Trigger walkforward validation for a lottery using its current best strategy."""
    try:
        config = _config(lottery_id)
    except KeyError:
        return JSONResponse({"error": f"Unknown lottery: {lottery_id}"}, status_code=404)
    try:
        state, _active = auto_loop.load_search_state_for_lottery(lottery_id)
        if state.best_metric <= 0:
            state = auto_loop.load_best_from_results_for_lottery(lottery_id)
        if state.best_metric <= 0:
            return JSONResponse({"error": "No best strategy found. Run optimization first."}, status_code=400)

        auto_loop.set_search_weights()
        try:
            rank_fn = auto_loop.build_rank_fn_from_state(state, config=config)
            result = run_and_save_walkforward(
                lottery_id=lottery_id,
                rank_fn=rank_fn,
                recent_window=state.rw,
                backtest_score=state.best_metric,
                strategy_name=state.best_strategy,
            )
        finally:
            auto_loop.reset_default_weights()

        if result is None:
            return JSONResponse({"error": "Insufficient data for walkforward validation"}, status_code=400)
        return result.to_dict()
    except Exception as exc:
        import traceback
        return JSONResponse({"error": str(exc), "detail": traceback.format_exc()}, status_code=500)



# Multi-Agent System Endpoints
AGENTS_DIR = PROJECT_DIR / "agents"
STATUS_DIR = AGENTS_DIR / "status"
INCIDENTS_DIR = AGENTS_DIR / "incidents"
AGENT_SCRIPT_MAP = {
    "strategy_researcher": "strategy_researcher.py",
    "lottery_manager": "lottery_manager.py",
    "ticket_generator": "ticket_generator.py",
    "system_monitor": "system_monitor.py",
    "experiment_monitor": "experiment_monitor.py",
    "strategy_optimizer": "strategy_optimizer.py",
    "strategy_learner": "strategy_learner.py",
    "ollama_strategist": "ollama_strategist.py",
    "ai_strategy_coder": "ai_strategy_coder.py",
    "ai_ticket_agent": "ai_ticket_agent.py",
    "telegram_agent": "telegram_agent.py",
}
AGENT_IDS = tuple(AGENT_SCRIPT_MAP)
# Serverul e singurul supraveghetor: pornește și repornește toți agenții.
# (system_monitor doar oprește agenții blocați; repornește el însuși numai dacă
# serverul nu răspunde.) Înainte, strategy_researcher/ticket_generator erau porniți
# doar de system_monitor, iar lottery_manager de amândoi → instanțe duble.
SUPERVISED_AGENTS = (
    "system_monitor", "lottery_manager", "experiment_monitor", "strategy_researcher",
    "ticket_generator", "strategy_optimizer", "strategy_learner", "ollama_strategist",
    "ai_strategy_coder", "ai_ticket_agent", "telegram_agent",
)


def _agent_instance_pid(agent_id: str) -> int | None:
    """PID-ul unui proces care rulează chiar scriptul agentului din acest proiect.

    Un PID viu din fișierul de status nu ajunge: Windows reciclează PID-urile, iar
    agentul „părea" pornit (și nu mai era pornit deloc) când PID-ul lui vechi
    ajungea la alt program.
    """
    try:
        import psutil
    except ImportError:
        return None
    script = AGENT_SCRIPT_MAP.get(agent_id, f"{agent_id}.py")
    project = str(PROJECT_DIR).replace("/", "\\").lower()
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = proc.info["cmdline"] or []
            in_project = project in " ".join(cmdline).replace("/", "\\").lower()
            if in_project and any(os.path.basename(arg) == script for arg in cmdline):
                return int(proc.info["pid"])
        except (psutil.Error, OSError, TypeError):
            continue
    return None


# Agenți opriți manual din UI/API — supravegherea nu îi repornește până la Start.
_agents_stopped_by_user: set[str] = set()


def _agent_should_run(agent_id: str) -> bool:
    if agent_id in _agents_stopped_by_user:
        return False
    try:
        config = json.loads((AGENTS_DIR / "config.json").read_text(encoding="utf-8"))
        return bool(config.get("agents", {}).get(agent_id, {}).get("enabled", True))
    except (OSError, ValueError):
        return True


def _get_agent_status(agent_id: str) -> dict:
    """Get status for a specific agent."""
    status_file = STATUS_DIR / f"{agent_id}.json"
    if not status_file.exists():
        return {"id": agent_id, "status": "stopped", "running": False}

    try:
        data, _ = read_json_with_recovery(status_file)
        if not isinstance(data, dict):
            return {"id": agent_id, "status": "unknown", "running": False}

        # Check if process is alive
        pid = data.get("pid")
        running = False
        if pid:
            try:
                if sys.platform == "win32":
                    import ctypes
                    process = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
                    if process:
                        ctypes.windll.kernel32.CloseHandle(process)
                        running = True
                else:
                    os.kill(int(pid), 0)
                    running = True
            except (OSError, ProcessLookupError):
                running = False

        data["running"] = running
        data["id"] = agent_id
        return data
    except Exception:
        return {"id": agent_id, "status": "error", "running": False}


@app.get("/api/agents")
async def get_agents():
    """Get status of all agents."""
    agents = []

    for agent_id in AGENT_IDS:
        agents.append(_get_agent_status(agent_id))

    # Get recent incidents
    incidents = []
    if INCIDENTS_DIR.exists():
        try:
            incident_files = sorted(INCIDENTS_DIR.glob("*.json"), reverse=True)[:5]
            for incident_file in incident_files:
                data, _ = read_json_with_recovery(incident_file)
                if isinstance(data, dict):
                    incidents.append({
                        "timestamp": data.get("timestamp"),
                        "agent": data.get("agent_id"),
                        "type": data.get("type"),
                        "severity": data.get("severity"),
                        "auto_resolved": data.get("auto_resolved", False)
                    })
        except Exception:
            pass

    return {
        "agents": agents,
        "incidents": incidents,
        "system_status": "healthy" if all(a.get("running") for a in agents if a["id"] in ["lottery_manager", "system_monitor"]) else "degraded"
    }


@app.get("/api/agents/status")
async def agents_status_endpoint():
    """Return status of all agents in the multi-agent system."""
    from pathlib import Path
    AGENTS_DIR = PROJECT_DIR / "agents"
    STATUS_DIR = AGENTS_DIR / "status"

    agents = {
        "system_monitor": {"running": False, "status": "unknown", "pid": None},
        "lottery_manager": {"running": False, "status": "unknown", "pid": None},
        "strategy_researcher": {"running": False, "status": "unknown", "pid": None},
        "ticket_generator": {"running": False, "status": "unknown", "pid": None},
        "experiment_monitor": {"running": False, "status": "unknown", "pid": None},
        "strategy_optimizer": {"running": False, "status": "unknown", "pid": None},
        "strategy_learner": {"running": False, "status": "unknown", "pid": None},
    }

    for agent_id in agents:
        status_file = STATUS_DIR / f"{agent_id}.json"
        if status_file.exists():
            try:
                data, _ = read_json_with_recovery(status_file)
                if isinstance(data, dict):
                    agents[agent_id]["status"] = data.get("status", "unknown")
                    agents[agent_id]["current_task"] = data.get("current_task")
                    agents[agent_id]["progress_percent"] = data.get("progress_percent", 0)
                    agents[agent_id]["last_activity"] = data.get("last_activity")

                    pid = data.get("pid")
                    if pid:
                        agents[agent_id]["pid"] = pid
                        # Check if process is alive
                        try:
                            if sys.platform == "win32":
                                import ctypes
                                process = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
                                if process:
                                    ctypes.windll.kernel32.CloseHandle(process)
                                    agents[agent_id]["running"] = True
                            else:
                                os.kill(int(pid), 0)
                                agents[agent_id]["running"] = True
                        except (OSError, ProcessLookupError):
                            pass
            except Exception:
                pass

    return {
        "agents": agents,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_healthy": all(a.get("running", False) for a in agents.values() if a["status"] != "idle")
    }


@app.get("/api/agents/incidents")
async def agents_incidents_endpoint(limit: int = 10):
    """Return recent incidents from the multi-agent system."""
    from pathlib import Path
    AGENTS_DIR = PROJECT_DIR / "agents"
    INCIDENTS_DIR = AGENTS_DIR / "incidents"

    incidents = []

    try:
        incident_files = sorted(INCIDENTS_DIR.glob("*.json"), reverse=True)
        for incident_file in incident_files[:limit]:
            try:
                data, _ = read_json_with_recovery(incident_file)
                if isinstance(data, dict):
                    incidents.append(data)
            except Exception:
                pass
    except Exception:
        pass

    return {
        "incidents": incidents,
        "count": len(incidents),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/agents/strategies/pending")
async def agents_pending_strategies_endpoint():
    """Return pending strategies discovered by the research agent."""
    import json
    from pathlib import Path
    AGENTS_DIR = PROJECT_DIR / "agents"

    # Check both the old location and new locations
    strategies_files = [
        AGENTS_DIR / "strategies" / "pending_review.json",  # Old location
        AGENTS_DIR / "tasks" / "pending_strategies.json",   # New location
    ]

    strategies = []
    for strategies_file in strategies_files:
        try:
            if strategies_file.exists():
                with open(strategies_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        strategies.extend(data)
        except Exception:
            pass

    # Also check strategies generated by strategy_optimizer
    optimizer_strategies_file = AGENTS_DIR / "tasks" / "pending_strategies.json"
    try:
        if optimizer_strategies_file.exists():
            with open(optimizer_strategies_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    # Add source information
                    for strategy in data:
                        strategy["source"] = "strategy_optimizer"
                    strategies.extend(data)
    except Exception:
        pass

    return {
        "strategies": strategies,
        "count": len(strategies),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get detailed status for a specific agent."""
    if agent_id not in AGENT_IDS:
        return JSONResponse({"error": f"Unknown agent: {agent_id}"}, status_code=404)

    return _get_agent_status(agent_id)


@app.post("/api/agents/{agent_id}/start")
async def start_agent(agent_id: str):
    """Start a specific agent."""
    if agent_id not in AGENT_IDS:
        return JSONResponse({"error": f"Unknown agent: {agent_id}"}, status_code=404)

    _agents_stopped_by_user.discard(agent_id)

    # Check if already running
    status = _get_agent_status(agent_id)
    if status.get("running"):
        return {"status": "already_running", "agent": agent_id}

    try:
        script_path = AGENTS_DIR / AGENT_SCRIPT_MAP[agent_id]
        if not script_path.exists():
            return JSONResponse({"error": f"Agent script not found: {script_path}"}, status_code=500)
        if sys.platform == "win32":
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(PROJECT_DIR)
            )
        else:
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                cwd=str(PROJECT_DIR)
            )

        time.sleep(1)  # Wait for agent to start
        new_status = _get_agent_status(agent_id)

        if new_status.get("running"):
            return {"status": "started", "agent": agent_id, "pid": new_status.get("pid")}
        return {"status": "started", "agent": agent_id, "pid": proc.pid, "pending_status": True}

    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/agents/{agent_id}/stop")
async def stop_agent(agent_id: str, force: bool = False):
    """Stop a specific agent."""
    if agent_id not in AGENT_IDS:
        return JSONResponse({"error": f"Unknown agent: {agent_id}"}, status_code=404)

    _agents_stopped_by_user.add(agent_id)

    status = _get_agent_status(agent_id)
    pid = status.get("pid")
    if not status.get("running") and not force:
        return {"status": "not_running", "agent": agent_id}

    # Send stop signal via message file
    command_file = AGENTS_DIR / "messages" / "commands" / f"to_{agent_id}.json"
    command_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        messages = []
        if command_file.exists():
            with open(command_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    messages = data

        messages.append({
            "id": f"stop_{time.time()}",
            "from": "api",
            "to": agent_id,
            "type": "stop_request",
            "payload": {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        with open(command_file, "w", encoding="utf-8") as f:
            json.dump(messages, f)

        force_terminated = False
        if force and pid:
            force_terminated = _terminate_pid(int(pid))

        return {
            "status": "stop_requested",
            "agent": agent_id,
            "pid": pid,
            "force_terminated": force_terminated,
        }

    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/agents/{agent_id}/logs")
async def get_agent_logs(agent_id: str, lines: int = 50):
    """Get recent logs for a specific agent."""
    if agent_id not in AGENT_IDS:
        return JSONResponse({"error": f"Unknown agent: {agent_id}"}, status_code=404)

    log_file = AGENTS_DIR / "logs" / f"{agent_id}.log"
    if not log_file.exists():
        return {"agent": agent_id, "logs": []}

    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
            recent_logs = [line.strip() for line in all_lines[-lines:]]
            return {"agent": agent_id, "logs": recent_logs}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# === Agent Management Functions ===

def _start_required_agents() -> dict:
    """Start required agents when server starts."""
    AGENTS_DIR = PROJECT_DIR / "agents"
    STATUS_DIR = AGENTS_DIR / "status"

    # Agents that should run continuously with the server
    CONTINUOUS_AGENTS = list(SUPERVISED_AGENTS)

    agent_processes = {}

    print("Starting required agents...")

    # Write server PID to a file so agents can check if server is still alive
    server_pid_file = AGENTS_DIR / ".server_pid"
    try:
        with open(server_pid_file, "w") as f:
            f.write(str(os.getpid()))
        print(f"  Server PID ({os.getpid()}) written to {server_pid_file}")
    except Exception as e:
        print(f"Warning: Could not write server PID file: {e}")

    # Ensure status directory exists
    STATUS_DIR.mkdir(parents=True, exist_ok=True)

    for agent_id in CONTINUOUS_AGENTS:
        max_attempts = 3
        attempt = 0

        while attempt < max_attempts:
            try:
                attempt += 1
                print(f"  {agent_id}: checking status (attempt {attempt}/{max_attempts})")

                if not _agent_should_run(agent_id):
                    print(f"  {agent_id}: disabled in agents/config.json, not started")
                    break

                # Check if agent is already running
                if _is_agent_running(agent_id):
                    print(f"  {agent_id}: already running")
                    # Try to get the existing process
                    try:
                        status_file = STATUS_DIR / f"{agent_id}.json"
                        if status_file.exists():
                            with open(status_file, "r") as f:
                                status_data = json.load(f)
                                pid = status_data.get("pid")
                                if pid:
                                    # Try to connect to existing process
                                    try:
                                        if sys.platform == "win32":
                                            import ctypes
                                            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))  # 0x1000 = PROCESS_QUERY_INFORMATION
                                            if handle:
                                                ctypes.windll.kernel32.CloseHandle(handle)
                                                # Process exists, but we don't own it, so we can't track it
                                                print(f"  {agent_id}: found existing process (PID: {pid}) but cannot manage it")
                                                break
                                        else:
                                            os.kill(pid, 0)
                                            print(f"  {agent_id}: found existing process (PID: {pid}) but cannot manage it")
                                            break
                                    except (OSError, ProcessLookupError):
                                        # Process doesn't exist, continue with startup
                                        pass
                    except Exception:
                        pass
                    break

                # Start the agent
                script_path = AGENTS_DIR / f"{agent_id}.py"
                if script_path.exists():
                    print(f"  {agent_id}: starting agent script at {script_path}")

                    if sys.platform == "win32":
                        proc = subprocess.Popen(
                            [sys.executable, str(script_path)],
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            cwd=str(PROJECT_DIR)
                        )
                    else:
                        proc = subprocess.Popen(
                            [sys.executable, str(script_path)],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            start_new_session=True,
                            cwd=str(PROJECT_DIR)
                        )

                    agent_processes[agent_id] = proc
                    print(f"  {agent_id}: started (PID: {proc.pid})")

                    # Verify the process started successfully
                    if proc.poll() is not None:
                        raise Exception(f"Process terminated immediately with return code {proc.returncode}")

                    time.sleep(0.5)  # Small delay between startups
                    break  # Success, break out of retry loop
                else:
                    print(f"  {agent_id}: script not found at {script_path}")
                    break  # No point in retrying if script doesn't exist

            except Exception as e:
                print(f"  {agent_id}: failed to start (attempt {attempt}/{max_attempts}) - {e}")
                if attempt < max_attempts:
                    print(f"  {agent_id}: waiting 2 seconds before retry...")
                    time.sleep(2)
                else:
                    print(f"  {agent_id}: giving up after {max_attempts} attempts")

    print("Agent startup completed.")
    return agent_processes


def _is_server_alive() -> bool:
    """Check if the server process is still alive by verifying the PID in the server PID file."""
    AGENTS_DIR = PROJECT_DIR / "agents"
    server_pid_file = AGENTS_DIR / ".server_pid"

    try:
        if not server_pid_file.exists():
            return False

        with open(server_pid_file, "r") as f:
            pid = int(f.read().strip())

        # Check if process with that PID is still running
        if sys.platform == "win32":
            # On Windows, use tasklist to check if process exists
            result = subprocess.run(
                ["tasklist", "/fi", f"PID eq {pid}"],
                capture_output=True,
                text=True,
                shell=True
            )
            return f" {pid} " in result.stdout
        else:
            # On Unix-like systems, use os.kill with signal 0 to check if process exists
            os.kill(pid, 0)
            return True
    except (ValueError, FileNotFoundError, ProcessLookupError, PermissionError, subprocess.SubprocessError):
        return False


def _monitor_agents(agent_processes: dict) -> None:
    """Periodically check agent health and restart if needed."""
    import threading

    def _health_check():
        global _shutdown_in_progress

        while not _shutdown_in_progress:
            try:
                time.sleep(30)  # Check every 30 seconds

                # Check if shutdown is in progress
                if _shutdown_in_progress:
                    break

                AGENTS_DIR = PROJECT_DIR / "agents"
                STATUS_DIR = AGENTS_DIR / "status"

                for agent_id in SUPERVISED_AGENTS:
                    proc = agent_processes.get(agent_id)
                    try:
                        if proc is not None and proc.poll() is None:
                            continue  # procesul nostru rulează
                        if not _agent_should_run(agent_id):
                            continue  # oprit de utilizator sau dezactivat în config
                        if _agent_instance_pid(agent_id) is not None:
                            continue  # rulează o instanță pe care nu o deținem (ex. copia noastră a ieșit prin garda de instanță unică)
                        if True:
                            previous_pid = proc.pid if proc is not None else None
                            return_code = proc.returncode if proc is not None else None
                            print(f"  {agent_id}: not running (return code: {return_code}), restarting...")

                            # Read status to get last heartbeat
                            status_file = STATUS_DIR / f"{agent_id}.json"
                            last_heartbeat = None
                            last_update = None
                            if status_file.exists():
                                try:
                                    with open(status_file, "r") as f:
                                        status_data = json.load(f)
                                        last_heartbeat = status_data.get("last_heartbeat")
                                        last_update = status_data.get("last_update")
                                except Exception as e:
                                    print(f"  {agent_id}: warning - could not read status file: {e}")

                            # Log incident
                            incident_time = datetime.utcnow().isoformat()
                            try:
                                incident_file = AGENTS_DIR / "incidents" / f"{incident_time.replace(':', '-')}_agent_restart.json"
                                incident_file.parent.mkdir(parents=True, exist_ok=True)
                                with open(incident_file, "w") as f:
                                    json.dump({
                                        "timestamp": incident_time,
                                        "agent_id": agent_id,
                                        "event": "agent_restart",
                                        "previous_pid": previous_pid,
                                        "last_heartbeat": last_heartbeat,
                                        "last_update": last_update,
                                        "return_code": return_code
                                    }, f, indent=2)
                            except Exception as e:
                                print(f"  {agent_id}: warning - could not log restart incident: {e}")

                            # Restart the agent
                            script_path = AGENTS_DIR / f"{agent_id}.py"
                            if script_path.exists():
                                print(f"  {agent_id}: restarting agent script at {script_path}")

                                if sys.platform == "win32":
                                    new_proc = subprocess.Popen(
                                        [sys.executable, str(script_path)],
                                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL,
                                        cwd=str(PROJECT_DIR)
                                    )
                                else:
                                    new_proc = subprocess.Popen(
                                        [sys.executable, str(script_path)],
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL,
                                        start_new_session=True,
                                        cwd=str(PROJECT_DIR)
                                    )

                                agent_processes[agent_id] = new_proc
                                print(f"  {agent_id}: restarted (PID: {new_proc.pid})")
                            else:
                                print(f"  {agent_id}: script not found at {script_path}")
                    except Exception as e:
                        print(f"  Error monitoring {agent_id}: {e}")

            except Exception as e:
                if not _shutdown_in_progress:
                    print(f"  Error in agent health check: {e}")

        print("  Agent monitoring thread stopped.")

    # Start health check in a separate thread
    monitor_thread = threading.Thread(target=_health_check, daemon=True, name="AgentMonitor")
    monitor_thread.start()

    # Store thread reference for potential future use
    global _monitor_thread
    _monitor_thread = monitor_thread


def _stop_agents(agent_processes: dict) -> None:
    """Stop all agents when server shuts down."""
    if not agent_processes:
        print("No agents to stop.")
        return  # Nothing to stop

    print("Stopping agents...")

    # First, send stop signals to agents that support it
    AGENTS_DIR = PROJECT_DIR / "agents"
    COMMANDS_DIR = AGENTS_DIR / "messages" / "commands"

    # Send stop signals to all continuous agents
    CONTINUOUS_AGENTS = list(SUPERVISED_AGENTS)

    print("Sending graceful stop signals to agents...")
    stop_sent_time = time.time()
    for agent_id in CONTINUOUS_AGENTS:
        try:
            command_file = COMMANDS_DIR / f"to_{agent_id}.json"

            # Create stop message
            stop_message = {
                "id": f"stop_{int(time.time())}",
                "from": "server",
                "to": agent_id,
                "type": "stop_request",
                "payload": {},
                "timestamp": datetime.utcnow().isoformat()
            }

            # Read existing messages
            messages = []
            if command_file.exists():
                try:
                    with open(command_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            messages = data
                except (json.JSONError, IOError) as e:
                    print(f"  {agent_id}: warning - could not read existing command file: {e}")
                    messages = []

            # Add stop message
            messages.append(stop_message)

            # Write back
            command_file.parent.mkdir(parents=True, exist_ok=True)
            with open(command_file, "w", encoding="utf-8") as f:
                json.dump(messages, f, indent=2)

            print(f"  {agent_id}: stop signal sent")
        except Exception as e:
            print(f"  {agent_id}: failed to send stop signal - {e}")

    # Wait a bit for graceful shutdown (longer now for better gracefulness)
    graceful_wait_time = 5
    print(f"  Waiting for graceful shutdown ({graceful_wait_time} seconds)...")
    time.sleep(graceful_wait_time)

    # Check which processes are still running and kill them forcefully
    print("Checking for remaining processes...")
    killed_pids = []
    force_kill_start_time = time.time()

    for agent_id, proc in agent_processes.items():
        try:
            if proc.poll() is None:  # Still running
                print(f"  {agent_id}: force terminating (PID: {proc.pid})")
                proc.terminate()
                killed_pids.append(proc.pid)

                # Wait a moment for graceful termination with timeout
                termination_timeout = 3
                start_wait = time.time()
                while proc.poll() is None and (time.time() - start_wait) < termination_timeout:
                    time.sleep(0.1)

                # If still running, force kill
                if proc.poll() is None:
                    print(f"  {agent_id}: force killing (PID: {proc.pid})")
                    proc.kill()

                    # Wait briefly for kill to take effect
                    kill_timeout = 2
                    start_wait = time.time()
                    while proc.poll() is None and (time.time() - start_wait) < kill_timeout:
                        time.sleep(0.1)

                    if proc.poll() is None:
                        print(f"  {agent_id}: warning - process may still be running")
                    else:
                        print(f"  {agent_id}: successfully killed")
                else:
                    print(f"  {agent_id}: successfully terminated")
            else:
                print(f"  {agent_id}: already terminated (return code: {proc.returncode})")
        except Exception as e:
            print(f"  {agent_id}: error terminating - {e}")

    # Also kill any agent processes that might not be in our tracking dict
    print("Cleaning up orphaned agent processes...")
    try:
        import psutil
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        orphan_count = 0
        for child in children:
            try:
                # Check if this is one of our agent processes
                cmdline = child.cmdline()
                if any(agent_script in " ".join(cmdline) for agent_script in ["experiment_monitor.py", "lottery_manager.py", "system_monitor.py", "ticket_generator.py", "strategy_researcher.py"]):
                    if child.pid not in killed_pids:  # Don't double-kill
                        print(f"  Killing orphaned agent process: {child.name()} (PID: {child.pid})")
                        child.terminate()
                        orphan_count += 1

                        # Wait for termination with timeout
                        term_timeout = 3
                        start_wait = time.time()
                        while child.is_running() and (time.time() - start_wait) < term_timeout:
                            time.sleep(0.1)

                        if child.is_running():
                            child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                # Process may have terminated between check and action
                pass
            except Exception as e:
                print(f"  Error checking/killing orphaned process: {e}")

        if orphan_count > 0:
            print(f"  Cleaned up {orphan_count} orphaned agent processes")
        else:
            print("  No orphaned agent processes found")
    except ImportError:
        print("  psutil not available, skipping orphan process cleanup")
        pass  # psutil not available
    except Exception as e:
        print(f"  Error during orphan process cleanup: {e}")

    # Remove server PID file
    try:
        server_pid_file = AGENTS_DIR / ".server_pid"
        if server_pid_file.exists():
            server_pid_file.unlink()
            print("  Server PID file removed")
    except Exception as e:
        print(f"  Warning: Could not remove server PID file: {e}")

    # Log shutdown completion
    total_shutdown_time = time.time() - stop_sent_time
    print(f"Agent shutdown completed in {total_shutdown_time:.2f} seconds.")


def _is_agent_running(agent_id: str) -> bool:
    """Check if an agent is currently running."""
    AGENTS_DIR = PROJECT_DIR / "agents"
    STATUS_DIR = AGENTS_DIR / "status"
    status_file = STATUS_DIR / f"{agent_id}.json"

    try:
        import psutil  # noqa: F401
        return _agent_instance_pid(agent_id) is not None
    except ImportError:
        pass  # fără psutil: verificarea veche, doar după PID

    if not status_file.exists():
        return False

    try:
        with open(status_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        pid = data.get("pid")
        if pid:
            # Check if process exists
            try:
                if sys.platform == "win32":
                    kernel32 = ctypes.windll.kernel32
                    handle = kernel32.OpenProcess(0x1000, False, int(pid))  # 0x1000 = PROCESS_QUERY_INFORMATION
                    if handle:
                        kernel32.CloseHandle(handle)
                        return True
                else:
                    os.kill(pid, 0)
                    return True
            except (OSError, ProcessLookupError):
                pass
    except Exception:
        pass

    return False


@app.get("/api/health")
async def health_check():
    """Health check endpoint for server status."""
    global _shutdown_in_progress
    return {
        "status": "ok" if not _shutdown_in_progress else "shutting_down",
        "server_pid": os.getpid(),
        "uptime": "server uptime tracking not implemented",
        "shutdown_in_progress": _shutdown_in_progress
    }


@app.get("/api/agents/learning/insights")
async def get_learning_insights(lottery_id: str = None, limit: int = 5):
    """Get recent learning insights from the strategy learner agent."""
    from pathlib import Path
    import glob

    LEARNING_DIR = PROJECT_DIR / "agents" / "learning" / "insights"

    insights_list = []

    try:
        if LEARNING_DIR.exists():
            # Find insight files
            pattern = f"*{lottery_id}*" if lottery_id else "*"
            insight_files = list(LEARNING_DIR.glob(f"{pattern}_insights_*.json"))

            # Sort by modification time (newest first)
            insight_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Limit results
            insight_files = insight_files[:limit]

            # Load insights
            for insight_file in insight_files:
                try:
                    with open(insight_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        data["file"] = insight_file.name
                        insights_list.append(data)
                except Exception as e:
                    pass  # Skip files that can't be read

    except Exception as e:
        pass

    return {
        "insights": insights_list,
        "count": len(insights_list),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/agents/incidents/recent")
async def get_recent_incidents(limit: int = 10):
    """Get recent incidents across all agents."""
    incidents = []

    if INCIDENTS_DIR.exists():
        try:
            incident_files = sorted(INCIDENTS_DIR.glob("*.json"), reverse=True)[:limit]
            for incident_file in incident_files:
                data, _ = read_json_with_recovery(incident_file)
                if isinstance(data, dict):
                    incidents.append(data)
        except Exception:
            pass

    return {"incidents": incidents, "count": len(incidents)}


if __name__ == "__main__":
    import argparse
    import sys
    import uvicorn

    _parser = argparse.ArgumentParser(description="AutoResearch Dashboard")
    _parser.add_argument(
        "--port", type=int,
        default=int(os.environ.get("AUTORESEARCH_PORT") or DEFAULT_SERVER_PORT),
    )
    server_port = _parser.parse_args().port
    # agenții porniți de server moștenesc portul (agents/utils.py → health check)
    os.environ["AUTORESEARCH_PORT"] = str(server_port)

    if not _acquire_singleton_lock(SERVER_LOCK_FILE, "server"):
        try:
            existing_pid = int(SERVER_LOCK_FILE.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            existing_pid = None
        print(
            f"AutoResearch Dashboard already running"
            + (f" (PID {existing_pid})" if existing_pid else "")
            + ".",
            file=sys.stderr,
        )
        raise SystemExit(1)

    _register_server_lock_cleanup()
    _set_current_process_priority(getattr(subprocess, "ABOVE_NORMAL_PRIORITY_CLASS", 0))
    if sys.stdout is not None:
        try:
            print(f"AutoResearch Dashboard: http://localhost:{server_port}")
        except OSError:
            pass
    uvicorn.run(app, host="0.0.0.0", port=server_port, log_level="info")
