"""Auto-loop: Narrow search optimization for supported lotteries.

Runs an infinite loop of experiments:
1. Start with current best config (from per-lottery results or defaults)
2. Generate a grid sweep around the best parameters
3. Run all configs, find new best
4. Narrow the search range and refine step size
5. If no improvement found, widen search and try new strategy combos
6. Repeat forever

Usage:
    python auto_loop.py              # Run loop (prints progress, saves per-lottery results.tsv)
    python auto_loop.py --once       # Run one iteration only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).parent

# stdout redirecționat în loop.out.log → pe Windows ar fi cp1252; diacriticele ar crăpa print-urile
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

import ai_strategy_lab
from atomic_io import (
    DEFAULT_ATOMIC_WRITE_RETRIES,
    DEFAULT_ATOMIC_WRITE_RETRY_DELAY,
    read_json_with_recovery,
    recovery_path,
    write_json_atomic,
)
from generic_backtest import (
    BACKTEST_WEIGHTS,
    backtest_strategy,
    balanced_rank as _balanced_rank_generic,
    hot_cold_rank as _hot_cold_rank_generic,
    make_anticorrelation_wrapper,
    make_bayesian_freq_rank,
    make_drift_detector_rank,
    make_jackpot_cycle_rank,
    make_weighted_ensemble,
    top_frequency_rank as _top_frequency_rank_generic,
)
from jackpot_data import load_jackpot_data, load_jackpot_data_with_source
from lock_utils import current_lock_owner, lock_matches_process, read_lock_owner, serialize_lock_owner
from load_data import load_lottery_draws
from lottery_config import LotteryConfig, ensure_lottery_storage, get_lottery_config
from walkforward import run_and_save_walkforward

# Live thoughts integration - broadcast to OpenClawThink dashboard
try:
    from karpathy_live_thoughts import (
        think, analyze, action, solution, error, success, progress, info,
        thought_experiment_started, thought_experiment_complete,
        thought_new_best, thought_discard, thought_commit,
        thought_analyzing_results, thought_strategy_evolving,
        thought_auto_loop_start, thought_auto_loop_stop
    )
    LIVE_THOUGHTS_ENABLED = True
except ImportError:
    LIVE_THOUGHTS_ENABLED = False

ACTIVE_CONFIG = get_lottery_config("multi_multi_pl")
PROGRESS_FILE = ACTIVE_CONFIG.progress_file
RESULTS_FILE = ACTIVE_CONFIG.results_file
LOCK_FILE = ACTIVE_CONFIG.lock_file
LOOP_STATE_FILE = ACTIVE_CONFIG.loop_state_file
STOP_REQUEST_FILE = ACTIVE_CONFIG.stop_request_file

SEARCH_WEIGHTS = {"frequency": 0.75, "recent": 0.05, "delay": 0.20}
DEFAULT_WEIGHTS = {"frequency": 0.45, "recent": 0.35, "delay": 0.20}
CONFIG_TOKEN_RE = re.compile(r"^([a-z]+)(-?\d+(?:\.\d+)?)$")
ATOMIC_WRITE_RETRIES = DEFAULT_ATOMIC_WRITE_RETRIES
ATOMIC_WRITE_RETRY_DELAY = DEFAULT_ATOMIC_WRITE_RETRY_DELAY
PERSISTENCE_WARNING_COOLDOWN_SECONDS = 30.0
_LAST_PERSISTENCE_WARNING_BY_PATH: dict[str, float] = {}
_JACKPOT_DATA: dict = {}
_DRAW_DATES: list = []
PHASED_STRATEGIES = ("jc", "bf", "dd")
BF_PRIOR_CANDIDATES = [5.0, 10.0, 20.0, 50.0, 100.0]
DD_LAMBDA_CANDIDATES = [0.005, 0.01, 0.02, 0.05, 0.1]
AC_RATIO_CANDIDATES = [0.6, 0.7, 0.8, 0.9]
PHASE_ONE_WEIGHT_VALUES = [round(step * 0.5, 3) for step in range(0, 7)]
REQUIRED_NEW_STRATEGIES = ["sb", "jc", "bf", "dd"]


def configure_runtime(lottery_id: str = "multi_multi_pl") -> LotteryConfig:
    global ACTIVE_CONFIG, PROGRESS_FILE, RESULTS_FILE, LOCK_FILE, LOOP_STATE_FILE, STOP_REQUEST_FILE
    global _JACKPOT_DATA, _DRAW_DATES
    ACTIVE_CONFIG = get_lottery_config(lottery_id)
    ensure_lottery_storage(ACTIVE_CONFIG)
    PROGRESS_FILE = ACTIVE_CONFIG.progress_file
    RESULTS_FILE = ACTIVE_CONFIG.results_file
    LOCK_FILE = ACTIVE_CONFIG.lock_file
    LOOP_STATE_FILE = ACTIVE_CONFIG.loop_state_file
    STOP_REQUEST_FILE = ACTIVE_CONFIG.stop_request_file
    _JACKPOT_DATA = {}
    _DRAW_DATES = []
    return ACTIVE_CONFIG


configure_runtime()


def _write_json_atomic(path: Path, data: dict) -> None:
    fallback_path = recovery_path(path)
    try:
        write_json_atomic(
            path,
            data,
            retries=ATOMIC_WRITE_RETRIES,
            retry_delay=ATOMIC_WRITE_RETRY_DELAY,
        )
        try:
            fallback_path.unlink(missing_ok=True)
        except OSError:
            pass
    except OSError as exc:
        try:
            write_json_atomic(
                fallback_path,
                data,
                retries=ATOMIC_WRITE_RETRIES,
                retry_delay=ATOMIC_WRITE_RETRY_DELAY,
            )
        except OSError:
            _warn_persistence_failure(path, exc, fallback_failed=True)
            return
        _warn_persistence_failure(path, exc, fallback_path=fallback_path, fallback_failed=False)


def _read_json(path: Path) -> dict | None:
    data, _source_path = read_json_with_recovery(path)
    if isinstance(data, dict):
        return data
    return None


def _warn_persistence_failure(
    path: Path,
    exc: OSError | None,
    *,
    fallback_path: Path | None = None,
    fallback_failed: bool,
) -> None:
    now = time.time()
    key = str(path)
    last_warning = _LAST_PERSISTENCE_WARNING_BY_PATH.get(key, 0.0)
    if now - last_warning < PERSISTENCE_WARNING_COOLDOWN_SECONDS:
        return
    _LAST_PERSISTENCE_WARNING_BY_PATH[key] = now

    if fallback_failed:
        message = (
            f"[warning] persistence unavailable for {path.name}; continuing without updating "
            f"checkpoint/progress. Last error: {exc!r}"
        )
    else:
        recovery_file = fallback_path or recovery_path(path)
        message = (
            f"[warning] primary persistence locked for {path.name}; wrote recovery state to "
            f"{recovery_file.name} and continuing. Last error: {exc!r}"
        )
    print(message, file=sys.stderr)


def _active_iteration_tested_count(active_iteration: dict) -> int:
    tested_count = active_iteration.get("tested_count")
    if tested_count is not None:
        return int(tested_count)
    all_tested = active_iteration.get("all_tested")
    if isinstance(all_tested, list):
        return len(all_tested)
    weight_index = int(active_iteration.get("weight_index", 0))
    window_index = int(active_iteration.get("window_index", 0))
    if active_iteration.get("phase") == "windows":
        return weight_index + window_index
    return weight_index


def _merge_top_results(top_results: list[dict], entry: dict, limit: int = 15) -> list[dict]:
    updated = [dict(item) for item in top_results]
    updated.append(dict(entry))
    updated.sort(key=lambda item: (-float(item.get("average_hits", 0.0)), item.get("config", "")))
    return updated[:limit]


def _compact_active_iteration(active_iteration: dict) -> dict:
    top_results = active_iteration.get("top_results")
    if not isinstance(top_results, list):
        top_results = []
        for entry in active_iteration.get("all_tested", []):
            top_results = _merge_top_results(top_results, entry)
    compact = {
        "search_state": dict(active_iteration.get("search_state", {})),
        "search_phase": str(active_iteration.get("search_phase", "phase1")),
        "started_at": float(active_iteration.get("started_at", time.time())),
        "phase": str(active_iteration.get("phase", "search")),
        "search_index": int(active_iteration.get("search_index", active_iteration.get("weight_index", 0))),
        "weight_index": int(active_iteration.get("search_index", active_iteration.get("weight_index", 0))),
        "param_index": int(active_iteration.get("param_index", 0)),
        "window_index": int(active_iteration.get("window_index", 0)),
        "total": int(active_iteration.get("total", 0)),
        "tested_count": _active_iteration_tested_count(active_iteration),
        "recent": [dict(item) for item in list(active_iteration.get("recent", []))[-50:]],
        "top_results": [dict(item) for item in list(top_results)[:15]],
        "best_metric": float(active_iteration.get("best_metric", -1.0)),
        "best_config": active_iteration.get("best_config"),
        "best_progress": dict(active_iteration.get("best_progress", {})),
        "best_result": dict(active_iteration["best_result"]) if active_iteration.get("best_result") else None,
        "best_params": dict(active_iteration["best_params"]) if active_iteration.get("best_params") else None,
        "current_config": active_iteration.get("current_config"),
    }
    return compact


def request_stop() -> None:
    STOP_REQUEST_FILE.write_text("stop\n", encoding="utf-8")


def clear_stop_request() -> None:
    try:
        STOP_REQUEST_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def stop_requested() -> bool:
    return STOP_REQUEST_FILE.exists()


def set_search_weights() -> None:
    BACKTEST_WEIGHTS.update(SEARCH_WEIGHTS)


def reset_default_weights() -> None:
    BACKTEST_WEIGHTS.update(DEFAULT_WEIGHTS)


def _pid_alive(pid: int) -> bool:
    """Cross-platform process liveness check."""
    if sys.platform == "win32":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _acquire_lock() -> bool:
    """Acquire an exclusive per-lottery lock using atomic file creation."""
    current_pid = os.getpid()
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock_owner = current_lock_owner(
        script_path=Path(__file__),
        project_dir=PROJECT_DIR,
        kind="autoloop",
        lottery_id=ACTIVE_CONFIG.id,
    )

    for _ in range(2):
        try:
            fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing_owner = read_lock_owner(LOCK_FILE)
            existing_pid = existing_owner.pid
            if existing_pid == current_pid:
                return True
            if existing_pid is not None and _pid_alive(existing_pid):
                if lock_matches_process(existing_owner, script_path=Path(__file__), project_dir=PROJECT_DIR):
                    return False
            try:
                LOCK_FILE.unlink(missing_ok=True)
            except OSError:
                return False
            continue
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(serialize_lock_owner(lock_owner))
                handle.flush()
        except OSError:
            try:
                LOCK_FILE.unlink(missing_ok=True)
            except OSError:
                pass
            return False
        return True
    return False


def _release_lock() -> None:
    owner_pid = read_lock_owner(LOCK_FILE).pid
    if owner_pid != os.getpid():
        return
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except OSError:
        pass


# ─── Strategy definitions (same as train.py) ───


# Generic number domain and base strategies per selected lottery.
def _numbers() -> range:
    return range(1, ACTIVE_CONFIG.number_range + 1)


def top_frequency_rank(counts, recent_deque, last_seen, draw_index, recent_window):
    return _top_frequency_rank_generic(ACTIVE_CONFIG, counts, recent_deque, last_seen, draw_index, recent_window)


def hot_cold_rank(counts, recent_deque, last_seen, draw_index, recent_window):
    return _hot_cold_rank_generic(ACTIVE_CONFIG, counts, recent_deque, last_seen, draw_index, recent_window)


def balanced_rank(counts, recent_deque, last_seen, draw_index, recent_window):
    return _balanced_rank_generic(ACTIVE_CONFIG, counts, recent_deque, last_seen, draw_index, recent_window)


def sector_balanced_rank(counts, recent_deque, last_seen, draw_index, recent_window):
    max_count = max(counts.values()) or 1
    delays = {}
    for n in _numbers():
        s = last_seen.get(n, -1)
        delays[n] = draw_index - s if s >= 0 else draw_index
    max_delay = max(delays.values()) or 1
    base_scores = {}
    for n in _numbers():
        base_scores[n] = 0.70 * (counts.get(n, 0) / max_count) + 0.30 * (delays[n] / max_delay)
    sector_count = max(4, min(10, ACTIVE_CONFIG.number_range // 10 or 1))
    sector_size = max(1, (ACTIVE_CONFIG.number_range + sector_count - 1) // sector_count)
    sectors = {i: [] for i in range(sector_count)}
    for n in _numbers():
        sectors[min(sector_count - 1, (n - 1) // sector_size)].append((base_scores[n], n))
    for s in sectors:
        sectors[s].sort(reverse=True)
    result = []
    picked = set()
    rounds = max(len(values) for values in sectors.values())
    for round_idx in range(rounds):
        for sec in range(sector_count):
            if round_idx < len(sectors[sec]):
                _, n = sectors[sec][round_idx]
                if n not in picked:
                    result.append(n)
                    picked.add(n)
    for n in sorted(_numbers(), key=lambda x: (-base_scores[x], x)):
        if n not in picked:
            result.append(n)
    return result


def delay_weighted_rank(counts, recent_deque, last_seen, draw_index, recent_window):
    max_count = max(counts.values()) or 1
    delays = {}
    for n in _numbers():
        s = last_seen.get(n, -1)
        delays[n] = draw_index - s if s >= 0 else draw_index
    max_delay = max(delays.values()) or 1
    scores = {}
    for n in _numbers():
        freq = counts.get(n, 0) / max_count
        delay = delays[n] / max_delay
        scores[n] = 0.35 * freq + 0.65 * delay
    return sorted(_numbers(), key=lambda n: (-scores[n], n))


def momentum_rank(counts, recent_deque, last_seen, draw_index, recent_window):
    max_count = max(counts.values()) or 1
    recent_list = list(recent_deque)
    half = len(recent_list) // 2
    recent_half = recent_list[half:]
    older_half = recent_list[:half]
    recent_freq = {}
    older_freq = {}
    for n in _numbers():
        recent_freq[n] = sum(1 for d in recent_half if n in d)
        older_freq[n] = sum(1 for d in older_half if n in d)
    max_rf = max(recent_freq.values()) or 1
    max_of = max(older_freq.values()) or 1
    scores = {}
    for n in _numbers():
        freq_norm = counts.get(n, 0) / max_count
        mom = (recent_freq[n] / max_rf) - (older_freq[n] / max_of)
        scores[n] = 0.60 * freq_norm + 0.40 * (mom + 1) / 2
    return sorted(_numbers(), key=lambda n: (-scores[n], n))


def make_weighted_ensemble(fns, weights, config: LotteryConfig | None = None):
    config = config or ACTIVE_CONFIG

    def rank_fn(counts, recent_deque, last_seen, draw_index, recent_window):
        weighted_ranks = {n: 0.0 for n in _config_numbers(config)}
        total_w = sum(weights) or 1.0
        for fn, w in zip(fns, weights):
            ranked = fn(counts, recent_deque, last_seen, draw_index, recent_window)
            for pos, num in enumerate(ranked):
                weighted_ranks[num] += (pos * w) / total_w
        return sorted(_config_numbers(config), key=lambda n: (weighted_ranks[n], n))

    return rank_fn


# ─── Search space ───

BASE_FNS = [top_frequency_rank, hot_cold_rank, balanced_rank]
EXTRA_STRATEGIES = {
    "sb": sector_balanced_rank,
    "delay": delay_weighted_rank,
    "momentum": momentum_rank,
    "jc": None,
    "bf": None,
    "dd": None,
}

# Cheile AI înregistrate acum (plugin-uri promovate din ai_strategies/)
_AI_REGISTERED_KEYS: list = []


def register_ai_strategies() -> list[str]:
    """Include în spațiul de căutare plugin-urile AI promovate ale loteriei active.

    Idempotent și sigur la reapelare (curăță cheile AI vechi înainte).
    """
    global _AI_REGISTERED_KEYS
    for stale in _AI_REGISTERED_KEYS:
        EXTRA_STRATEGIES.pop(stale, None)
    try:
        _AI_REGISTERED_KEYS = ai_strategy_lab.register_promoted_for_config(ACTIVE_CONFIG)
    except Exception:
        _AI_REGISTERED_KEYS = []
    for key in _AI_REGISTERED_KEYS:
        EXTRA_STRATEGIES[key] = "ai"
    return list(_AI_REGISTERED_KEYS)


@dataclass
class SearchState:
    """Current search state — center, range, step."""
    # Core weights (for 4-strategy ensemble: tf, hc, bal, sb)
    tf: float = 3.0
    hc: float = 2.0
    bal: float = 4.0
    sb: float = 1.0
    delay: float = 1.0
    momentum: float = 1.0
    jc: float = 0.0
    bf: float = 0.0
    dd: float = 0.0
    # Windows
    iw: int = 30
    rw: int = 200
    prior_strength: float = 20.0
    ewma_lambda: float = 0.02
    use_anticorrelation: bool = False
    anticorrelation_ratio: float = 0.8
    # Search params
    radius: float = 0.5       # how far from center to search
    step: float = 0.5         # grid step size (coarser initially, narrows over time)
    iw_candidates: list = field(default_factory=lambda: [20, 25, 30, 35, 40, 50])
    rw_candidates: list = field(default_factory=lambda: [150, 200, 250, 300])
    # Extra strategy weights to try
    extra_strategies: list = field(default_factory=lambda: ["sb", "jc", "bf", "dd"])
    # Tracking
    best_metric: float = 0.0
    best_strategy: str = ""
    iteration: int = 0
    no_improve_count: int = 0
    search_phase: str = "phase1"

    def narrow(self):
        """Narrow search after finding improvement."""
        self.radius = max(0.1, self.radius * 0.6)
        self.step = max(0.025, self.step * 0.5)
        self.no_improve_count = 0

    def widen(self):
        """Widen search after no improvement found."""
        self.no_improve_count += 1
        if self.no_improve_count >= 3:
            self.radius = min(2.0, self.radius * 2.0)
            self.step = min(0.5, self.step * 2.0)
            self.no_improve_count = 0
        else:
            self.radius = min(1.5, self.radius * 1.3)
            self.step = max(0.025, self.step * 0.8)

    def weight_range(self, center: float) -> list[float]:
        """Generate weight values around center."""
        low = max(0.1, center - self.radius)
        high = center + self.radius
        vals = []
        v = low
        while v <= high + 1e-9:
            vals.append(round(v, 3))
            v += self.step
        # Always include the center
        center_r = round(center, 3)
        if center_r not in vals:
            vals.append(center_r)
            vals.sort()
        return vals

    def copy(self) -> "SearchState":
        return SearchState.from_dict(self.to_dict())

    def extra_weight(self, name: str) -> float:
        return float(getattr(self, name, 1.0))

    def set_extra_weight(self, name: str, value: float) -> None:
        if name not in self.extra_strategies:
            self.extra_strategies.append(name)
        setattr(self, name, float(value))

    def to_dict(self) -> dict:
        return {
            "tf": self.tf,
            "hc": self.hc,
            "bal": self.bal,
            "sb": self.sb,
            "delay": self.delay,
            "momentum": self.momentum,
            "jc": self.jc,
            "bf": self.bf,
            "dd": self.dd,
            "iw": self.iw,
            "rw": self.rw,
            "prior_strength": self.prior_strength,
            "ewma_lambda": self.ewma_lambda,
            "use_anticorrelation": self.use_anticorrelation,
            "anticorrelation_ratio": self.anticorrelation_ratio,
            "radius": self.radius,
            "step": self.step,
            "iw_candidates": list(self.iw_candidates),
            "rw_candidates": list(self.rw_candidates),
            "extra_strategies": list(self.extra_strategies),
            "best_metric": self.best_metric,
            "best_strategy": self.best_strategy,
            "iteration": self.iteration,
            "no_improve_count": self.no_improve_count,
            "search_phase": self.search_phase,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SearchState":
        return cls(
            tf=float(data.get("tf", 3.0)),
            hc=float(data.get("hc", 2.0)),
            bal=float(data.get("bal", data.get("b", 4.0))),
            sb=float(data.get("sb", 1.0)),
            delay=float(data.get("delay", 1.0)),
            momentum=float(data.get("momentum", 1.0)),
            jc=float(data.get("jc", 0.0)),
            bf=float(data.get("bf", 0.0)),
            dd=float(data.get("dd", 0.0)),
            iw=int(data.get("iw", 30)),
            rw=int(data.get("rw", 200)),
            prior_strength=float(data.get("prior_strength", 20.0)),
            ewma_lambda=float(data.get("ewma_lambda", 0.02)),
            use_anticorrelation=bool(data.get("use_anticorrelation", False)),
            anticorrelation_ratio=float(data.get("anticorrelation_ratio", 0.8)),
            radius=float(data.get("radius", 0.5)),
            step=float(data.get("step", 0.5)),
            iw_candidates=list(data.get("iw_candidates", [20, 25, 30, 35, 40, 50])),
            rw_candidates=list(data.get("rw_candidates", [150, 200, 250, 300])),
            extra_strategies=list(data.get("extra_strategies", ["sb", "jc", "bf", "dd"])),
            best_metric=float(data.get("best_metric", 0.0)),
            best_strategy=str(data.get("best_strategy", "")),
            iteration=int(data.get("iteration", 0)),
            no_improve_count=int(data.get("no_improve_count", 0)),
            search_phase=str(data.get("search_phase", "phase1")),
        )


def _save_loop_state(state: SearchState, active_iteration: dict | None = None) -> None:
    payload = {"version": 1, "search_state": state.to_dict()}
    if active_iteration is not None:
        payload["active_iteration"] = _compact_active_iteration(active_iteration)
    _write_json_atomic(LOOP_STATE_FILE, payload)


def _legacy_reference_payload(state: SearchState) -> dict:
    return {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "best_metric": state.best_metric,
        "best_strategy": state.best_strategy,
        "state": state.to_dict(),
    }


def _write_legacy_reference_if_missing(config: LotteryConfig, state: SearchState) -> None:
    if config.legacy_best_file.exists():
        return
    if state.best_metric <= 0 and not state.best_strategy:
        return
    _write_json_atomic(config.legacy_best_file, _legacy_reference_payload(state))


def _should_upgrade_to_new_strategy_set(state: SearchState) -> bool:
    current = list(dict.fromkeys(state.extra_strategies))
    if all(name in current for name in REQUIRED_NEW_STRATEGIES):
        return False
    return not any(name in current for name in PHASED_STRATEGIES)


def _upgrade_state_for_new_strategies(config: LotteryConfig, state: SearchState) -> tuple[SearchState, bool]:
    if not _should_upgrade_to_new_strategy_set(state):
        return state, False

    _write_legacy_reference_if_missing(config, state)
    state.extra_strategies = list(REQUIRED_NEW_STRATEGIES)
    state.jc = 0.0
    state.bf = 0.0
    state.dd = 0.0
    state.prior_strength = 20.0
    state.ewma_lambda = 0.02
    state.use_anticorrelation = False
    state.anticorrelation_ratio = 0.8
    state.search_phase = "phase1"
    state.no_improve_count = 0
    return state, True


def load_search_state() -> tuple[SearchState, dict | None]:
    data = _read_json(LOOP_STATE_FILE)
    if not data:
        state = load_best_from_results()
        state, _migrated = _upgrade_state_for_new_strategies(ACTIVE_CONFIG, state)
        return state, None
    state = SearchState.from_dict(data.get("search_state", {}))
    active_iteration = data.get("active_iteration")
    state, migrated = _upgrade_state_for_new_strategies(ACTIVE_CONFIG, state)
    if migrated:
        active_iteration = None
        _save_loop_state(state, None)
    return state, active_iteration


def load_search_state_for_lottery(lottery_id: str) -> tuple[SearchState, dict | None]:
    config = get_lottery_config(lottery_id)
    ensure_lottery_storage(config)
    data = _read_json(config.loop_state_file)
    if data:
        state = SearchState.from_dict(data.get("search_state", {}))
        active_iteration = data.get("active_iteration")
        state, migrated = _upgrade_state_for_new_strategies(config, state)
        if migrated:
            payload = {"version": 1, "search_state": state.to_dict()}
            write_json_atomic(config.loop_state_file, payload, retries=ATOMIC_WRITE_RETRIES, retry_delay=ATOMIC_WRITE_RETRY_DELAY)
            active_iteration = None
        return state, active_iteration
    state = load_best_from_results_for_lottery(lottery_id)
    state, _migrated = _upgrade_state_for_new_strategies(config, state)
    return state, None


def _load_best_from_results_file(results_file: Path) -> SearchState:
    """Replay KEEP rows from a results.tsv file to reconstruct the true best state."""
    current = SearchState()
    best = current.copy()
    try:
        lines = results_file.read_text(encoding="utf-8").strip().splitlines()
    except FileNotFoundError:
        return best
    if len(lines) < 2:
        return best
    for line in lines[1:]:
        cols = line.split("\t")
        if len(cols) < 4:
            continue
        status = cols[2].strip()
        strategy = cols[3].strip()
        if status != "KEEP":
            continue
        try:
            metric = float(cols[1])
        except ValueError:
            continue
        _update_state_from_config(current, strategy)
        current.best_metric = metric
        current.best_strategy = strategy
        if metric >= best.best_metric:
            best = current.copy()
            best.best_metric = metric
            best.best_strategy = strategy
    return best


def load_best_from_results() -> SearchState:
    return _load_best_from_results_file(RESULTS_FILE)


def load_best_from_results_for_lottery(lottery_id: str) -> SearchState:
    config = get_lottery_config(lottery_id)
    ensure_lottery_storage(config)
    return _load_best_from_results_file(config.results_file)


# ─── Progress reporting ───

def _write_progress(tested: int, total: int, name: str, state: SearchState,
                    start_time: float, best: dict, recent: list) -> None:
    elapsed = time.time() - start_time
    avg = elapsed / tested if tested > 0 else 0
    remaining = (total - tested) * avg if total > 0 else 0
    data = {
        "status": "running",
        "experiment_id": f"auto_iter{state.iteration}",
        "tested": tested,
        "total": total,
        "percent": round(tested / total * 100, 1) if total > 0 else 0,
        "elapsed_seconds": round(elapsed, 1),
        "eta_seconds": round(remaining, 1),
        "current_config": name,
        "best_so_far": best,
        "recent": recent[-10:],
        "pid": os.getpid(),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "loop_info": {
            "iteration": state.iteration,
            "search_phase": state.search_phase,
            "radius": state.radius,
            "step": state.step,
            "no_improve_streak": state.no_improve_count,
            "strategies": state.extra_strategies,
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
    }
    _write_json_atomic(PROGRESS_FILE, data)


def _write_finished(result: dict) -> None:
    data = {"status": "finished", "result": result,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _write_json_atomic(PROGRESS_FILE, data)


def _write_stopped(state: SearchState, active_iteration: dict) -> None:
    tested = _active_iteration_tested_count(active_iteration)
    total = active_iteration.get("total", tested)
    data = {
        "status": "stopped",
        "experiment_id": f"auto_iter{state.iteration}",
        "tested": tested,
        "total": total,
        "percent": round(tested / total * 100, 1) if total > 0 else 0,
        "current_config": active_iteration.get("current_config"),
        "best_so_far": active_iteration.get("best_progress", {}),
        "recent": active_iteration.get("recent", [])[-10:],
        "pid": os.getpid(),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "loop_info": {
            "iteration": state.iteration,
            "search_phase": state.search_phase,
            "radius": state.radius,
            "step": state.step,
            "no_improve_streak": state.no_improve_count,
            "strategies": state.extra_strategies,
        },
    }
    _write_json_atomic(PROGRESS_FILE, data)


def _append_results_tsv(metric: float, strategy: str, result: dict,
                        elapsed: float, desc: str, best_metric: float) -> None:
    status = "KEEP" if metric > best_metric else "DISCARD"
    row = (f"auto\t{metric}\t{status}\t{strategy}\t"
           f"{result.get('hit_rate_2_plus', 0)}\t{result.get('hit_rate_3_plus', 0)}\t"
           f"{result.get('portfolio_avg', 0)}\t{round(elapsed, 2)}\t{desc}\n")
    with open(ACTIVE_CONFIG.results_file, "a", encoding="utf-8") as f:
        f.write(row)


# ─── Main loop ───

def _params_from_state(state: SearchState) -> dict:
    return {
        "tf": state.tf,
        "hc": state.hc,
        "bal": state.bal,
        "sb": state.sb,
        "delay": state.delay,
        "momentum": state.momentum,
        "jc": state.jc,
        "bf": state.bf,
        "dd": state.dd,
        "iw": state.iw,
        "rw": state.rw,
        "prior_strength": state.prior_strength,
        "ewma_lambda": state.ewma_lambda,
        "use_anticorrelation": state.use_anticorrelation,
        "anticorrelation_ratio": state.anticorrelation_ratio,
        "extra_strategies": list(state.extra_strategies),
    }


def _apply_params_to_state(state: SearchState, params: dict) -> None:
    state.tf = float(params.get("tf", state.tf))
    state.hc = float(params.get("hc", state.hc))
    state.bal = float(params.get("bal", state.bal))
    state.sb = float(params.get("sb", state.sb))
    state.delay = float(params.get("delay", state.delay))
    state.momentum = float(params.get("momentum", state.momentum))
    state.jc = float(params.get("jc", state.jc))
    state.bf = float(params.get("bf", state.bf))
    state.dd = float(params.get("dd", state.dd))
    state.iw = int(params.get("iw", state.iw))
    state.rw = int(params.get("rw", state.rw))
    state.prior_strength = float(params.get("prior_strength", state.prior_strength))
    state.ewma_lambda = float(params.get("ewma_lambda", state.ewma_lambda))
    state.use_anticorrelation = bool(params.get("use_anticorrelation", state.use_anticorrelation))
    state.anticorrelation_ratio = float(params.get("anticorrelation_ratio", state.anticorrelation_ratio))
    state.extra_strategies = list(dict.fromkeys(params.get("extra_strategies", state.extra_strategies)))


def _spec_name_from_params(params: dict) -> str:
    extra_parts = [f"{name}{round(float(params.get(name, 1.0)), 3)}" for name in params.get("extra_strategies", [])]
    if "bf" in params.get("extra_strategies", []) or float(params.get("prior_strength", 20.0)) != 20.0:
        extra_parts.append(f"ps{round(float(params.get('prior_strength', 20.0)), 3)}")
    if "dd" in params.get("extra_strategies", []) or float(params.get("ewma_lambda", 0.02)) != 0.02:
        extra_parts.append(f"el{round(float(params.get('ewma_lambda', 0.02)), 3)}")
    if bool(params.get("use_anticorrelation", False)):
        extra_parts.append("ac1")
        extra_parts.append(f"acr{round(float(params.get('anticorrelation_ratio', 0.8)), 3)}")
    extra_suffix = "_" + "_".join(extra_parts) if extra_parts else ""
    return (
        f"tf{round(float(params['tf']), 3)}_hc{round(float(params['hc']), 3)}_"
        f"b{round(float(params['bal']), 3)}{extra_suffix}"
    )


def _config_numbers(config: LotteryConfig) -> range:
    return range(1, config.number_range + 1)


def _make_extra_strategy(config: LotteryConfig, name: str):
    # Plugin-uri AI promovate (chei "z...") — rezolvate din ai_strategy_lab
    ai_fn = ai_strategy_lab.make_extra(config, name)
    if ai_fn is not None:
        return ai_fn

    if name == "sb":
        def sector_balanced(counts, recent_deque, last_seen, draw_index, recent_window):
            max_count = max(counts.values()) or 1
            delays = {}
            for n in _config_numbers(config):
                s = last_seen.get(n, -1)
                delays[n] = draw_index - s if s >= 0 else draw_index
            max_delay = max(delays.values()) or 1
            base_scores = {}
            for n in _config_numbers(config):
                base_scores[n] = 0.70 * (counts.get(n, 0) / max_count) + 0.30 * (delays[n] / max_delay)
            sector_count = max(4, min(10, config.number_range // 10 or 1))
            sector_size = max(1, (config.number_range + sector_count - 1) // sector_count)
            sectors = {i: [] for i in range(sector_count)}
            for n in _config_numbers(config):
                sectors[min(sector_count - 1, (n - 1) // sector_size)].append((base_scores[n], n))
            for sector in sectors.values():
                sector.sort(reverse=True)
            result = []
            picked = set()
            rounds = max(len(values) for values in sectors.values())
            for round_idx in range(rounds):
                for sec in range(sector_count):
                    if round_idx < len(sectors[sec]):
                        _, n = sectors[sec][round_idx]
                        if n not in picked:
                            result.append(n)
                            picked.add(n)
            for n in sorted(_config_numbers(config), key=lambda x: (-base_scores[x], x)):
                if n not in picked:
                    result.append(n)
            return result

        return sector_balanced

    if name == "delay":
        def delay_weighted(counts, recent_deque, last_seen, draw_index, recent_window):
            max_count = max(counts.values()) or 1
            delays = {}
            for n in _config_numbers(config):
                s = last_seen.get(n, -1)
                delays[n] = draw_index - s if s >= 0 else draw_index
            max_delay = max(delays.values()) or 1
            scores = {}
            for n in _config_numbers(config):
                freq = counts.get(n, 0) / max_count
                delay = delays[n] / max_delay
                scores[n] = 0.35 * freq + 0.65 * delay
            return sorted(_config_numbers(config), key=lambda n: (-scores[n], n))

        return delay_weighted

    if name == "momentum":
        def momentum(counts, recent_deque, last_seen, draw_index, recent_window):
            max_count = max(counts.values()) or 1
            recent_list = list(recent_deque)
            half = len(recent_list) // 2
            recent_half = recent_list[half:]
            older_half = recent_list[:half]
            recent_freq = {}
            older_freq = {}
            for n in _config_numbers(config):
                recent_freq[n] = sum(1 for d in recent_half if n in d)
                older_freq[n] = sum(1 for d in older_half if n in d)
            max_rf = max(recent_freq.values()) or 1
            max_of = max(older_freq.values()) or 1
            scores = {}
            for n in _config_numbers(config):
                freq_norm = counts.get(n, 0) / max_count
                mom = (recent_freq[n] / max_rf) - (older_freq[n] / max_of)
                scores[n] = 0.60 * freq_norm + 0.40 * (mom + 1) / 2
            return sorted(_config_numbers(config), key=lambda n: (-scores[n], n))

        return momentum

    raise KeyError(name)


def _build_rank_fn(state: SearchState, params: dict, config: LotteryConfig | None = None,
                   *, jackpot_data=None, draw_dates=None):
    config = config or ACTIVE_CONFIG
    jackpot_data = _JACKPOT_DATA if jackpot_data is None else jackpot_data
    draw_dates = _DRAW_DATES if draw_dates is None else draw_dates
    weights = dict(SEARCH_WEIGHTS)
    strategy_names = list(dict.fromkeys(params.get("extra_strategies", state.extra_strategies)))
    base_fns = [
        lambda counts, recent_deque, last_seen, draw_index, recent_window: _top_frequency_rank_generic(
            config, counts, recent_deque, last_seen, draw_index, recent_window
        ),
        lambda counts, recent_deque, last_seen, draw_index, recent_window: _hot_cold_rank_generic(
            config, counts, recent_deque, last_seen, draw_index, recent_window
        ),
        lambda counts, recent_deque, last_seen, draw_index, recent_window: _balanced_rank_generic(
            config, counts, recent_deque, last_seen, draw_index, recent_window, weights=weights
        ),
    ]
    extra_fns = []
    active_strategy_names = []
    for name in strategy_names:
        if name == "jc":
            if jackpot_data and draw_dates:
                extra_fns.append(make_jackpot_cycle_rank(jackpot_data, draw_dates, config))
                active_strategy_names.append(name)
            continue
        if name == "bf":
            extra_fns.append(
                make_bayesian_freq_rank(
                    float(params.get("prior_strength", state.prior_strength)),
                    config.expected_rate,
                    config,
                )
            )
            active_strategy_names.append(name)
            continue
        if name == "dd":
            extra_fns.append(
                make_drift_detector_rank(
                    float(params.get("ewma_lambda", state.ewma_lambda)),
                    config,
                )
            )
            active_strategy_names.append(name)
            continue
        if ((name in EXTRA_STRATEGIES and EXTRA_STRATEGIES[name] is not None)
                or name in ai_strategy_lab.registered_keys(config.id)):
            try:
                extra_fns.append(_make_extra_strategy(config, name))
                active_strategy_names.append(name)
            except KeyError:
                continue  # cheie AI neînregistrată / plugin indisponibil → sare
    ensemble_weights = [params["tf"], params["hc"], params["bal"]]
    ensemble_weights.extend(float(params.get(name, state.extra_weight(name))) for name in active_strategy_names)
    ensemble = make_weighted_ensemble(base_fns + extra_fns, ensemble_weights, config=config)
    if bool(params.get("use_anticorrelation", state.use_anticorrelation)):
        ensemble = make_anticorrelation_wrapper(
            ensemble,
            config=config,
            min_cooccurrence_ratio=float(params.get("anticorrelation_ratio", state.anticorrelation_ratio)),
        )
    return ensemble


def build_rank_fn_from_state(state: SearchState, config: LotteryConfig | None = None,
                             *, jackpot_data=None, draw_dates=None):
    config = config or ACTIVE_CONFIG
    promoted_keys = {entry.get("key") for entry in ai_strategy_lab.promoted_plugins(config.id)}
    if promoted_keys != set(ai_strategy_lab.registered_keys(config.id)):
        ai_strategy_lab.register_promoted_for_config(config)
    if jackpot_data is None or draw_dates is None:
        if "jc" in state.extra_strategies:
            draws = load_lottery_draws(config.id)
            jackpot_data = load_jackpot_data(config.id, draws)
            draw_dates = [draw.draw_date for draw in draws]
        else:
            jackpot_data, draw_dates = {}, []
    return _build_rank_fn(state, _params_from_state(state), config=config,
                          jackpot_data=jackpot_data, draw_dates=draw_dates)


def rank_numbers_from_draws(draws, state: SearchState, lottery_id: str | None = None) -> list[int]:
    config = get_lottery_config(lottery_id) if lottery_id else ACTIVE_CONFIG
    counts = {n: 0 for n in _config_numbers(config)}
    recent_deque = deque(maxlen=state.rw)
    last_seen = {}
    for idx, draw in enumerate(draws):
        for n in draw.numbers:
            counts[n] += 1
            last_seen[n] = idx
        recent_deque.append(set(draw.numbers))
    jackpot_data, draw_dates = {}, []
    if "jc" in state.extra_strategies and draws and hasattr(draws[0], "draw_date"):
        jackpot_data = load_jackpot_data(config.id, draws)
        draw_dates = [draw.draw_date for draw in draws]
    rank_fn = build_rank_fn_from_state(state, config=config,
                                       jackpot_data=jackpot_data, draw_dates=draw_dates)
    return rank_fn(counts, recent_deque, last_seen, len(draws), state.rw)


def _bounded_int(center: int, offset: int, minimum: int, maximum: int, step: int = 5) -> int:
    value = center + offset
    value = max(minimum, min(maximum, value))
    if step > 1:
        value = int(round(value / step) * step)
        value = max(minimum, min(maximum, value))
    return value


def _adaptive_iw_candidates(state: SearchState) -> list[int]:
    deltas = [0, -10, -5, 5, 10]
    if state.no_improve_count >= 1:
        deltas.extend([-15, 15])
    if state.no_improve_count >= 2:
        deltas.extend([-20, 20, 30])
    values = list(state.iw_candidates)
    values.extend(_bounded_int(state.iw, delta, 10, 120, step=5) for delta in deltas)
    return sorted(dict.fromkeys(values))


def _adaptive_rw_candidates(state: SearchState) -> list[int]:
    deltas = [0, -50, -25, 25, 50]
    if state.no_improve_count >= 1:
        deltas.extend([-75, 75])
    if state.no_improve_count >= 2:
        deltas.extend([-100, 100, 150])
    values = list(state.rw_candidates)
    values.extend(_bounded_int(state.rw, delta, 75, 600, step=25) for delta in deltas)
    return sorted(dict.fromkeys(values))


def _strategy_variants(state: SearchState) -> list[list[str]]:
    current = list(dict.fromkeys(state.extra_strategies))
    variants = [current]
    missing = [name for name in EXTRA_STRATEGIES if name not in current]
    if state.no_improve_count >= 1:
        for name in missing:
            variants.append(list(dict.fromkeys(current + [name])))
    if state.no_improve_count >= 2:
        for name in missing:
            variants.append([name])
        if missing:
            variants.append(list(dict.fromkeys(current + missing)))
    unique = []
    seen = set()
    for variant in variants:
        key = tuple(variant)
        if key in seen:
            continue
        seen.add(key)
        unique.append(variant)
    return unique


def _weight_range_with_zero(state: SearchState, center: float) -> list[float]:
    values = [0.0]
    values.extend(state.weight_range(center))
    return sorted(dict.fromkeys(round(float(value), 3) for value in values))


def _build_phase_one_search_plan(state: SearchState) -> list[dict]:
    base = _params_from_state(state)
    base["use_anticorrelation"] = False
    specs: list[dict] = []
    seen = set()

    for weight in PHASE_ONE_WEIGHT_VALUES:
        params = dict(base)
        params.update({"jc": weight, "bf": 0.0, "dd": 0.0, "prior_strength": state.prior_strength, "ewma_lambda": state.ewma_lambda})
        name = _spec_name_from_params(params)
        if name not in seen:
            specs.append({"name": name, "params": params})
            seen.add(name)

    for weight in PHASE_ONE_WEIGHT_VALUES:
        for prior_strength in BF_PRIOR_CANDIDATES:
            params = dict(base)
            params.update({"jc": 0.0, "bf": weight, "dd": 0.0, "prior_strength": prior_strength, "ewma_lambda": state.ewma_lambda})
            name = _spec_name_from_params(params)
            if name not in seen:
                specs.append({"name": name, "params": params})
                seen.add(name)

    for weight in PHASE_ONE_WEIGHT_VALUES:
        for ewma_lambda in DD_LAMBDA_CANDIDATES:
            params = dict(base)
            params.update({"jc": 0.0, "bf": 0.0, "dd": weight, "prior_strength": state.prior_strength, "ewma_lambda": ewma_lambda})
            name = _spec_name_from_params(params)
            if name not in seen:
                specs.append({"name": name, "params": params})
                seen.add(name)

    return specs


def _build_phase_two_weight_plan(state: SearchState) -> list[dict]:
    base = _params_from_state(state)
    axes = [
        ("tf", state.tf),
        ("hc", state.hc),
        ("bal", state.bal),
        *((strategy_name, state.extra_weight(strategy_name)) for strategy_name in state.extra_strategies),
    ]
    plan = []
    seen = set()

    center_name = _spec_name_from_params(base)
    plan.append({"name": center_name, "params": dict(base)})
    seen.add(center_name)

    for axis_name, center in axes:
        for value in state.weight_range(center):
            if abs(value - center) < 1e-9:
                continue
            params = dict(base)
            params[axis_name] = value
            name = _spec_name_from_params(params)
            if name not in seen:
                plan.append({"name": name, "params": params})
                seen.add(name)
    return plan


def _build_phase_two_param_plan(state: SearchState) -> list[dict]:
    if state.iteration % 5 != 0:
        return []
    base = _params_from_state(state)
    specs = []
    seen = set()
    for prior_strength in BF_PRIOR_CANDIDATES:
        params = dict(base)
        params["prior_strength"] = prior_strength
        name = _spec_name_from_params(params)
        if name not in seen:
            specs.append({"name": name, "params": params})
            seen.add(name)
    for ewma_lambda in DD_LAMBDA_CANDIDATES:
        params = dict(base)
        params["ewma_lambda"] = ewma_lambda
        name = _spec_name_from_params(params)
        if name not in seen:
            specs.append({"name": name, "params": params})
            seen.add(name)
    return specs


def _build_phase_three_search_plan(state: SearchState) -> list[dict]:
    base = _params_from_state(state)
    specs = []
    seen = set()

    disabled = dict(base)
    disabled["use_anticorrelation"] = False
    disabled_name = _spec_name_from_params(disabled)
    specs.append({"name": disabled_name, "params": disabled})
    seen.add(disabled_name)

    for ratio in AC_RATIO_CANDIDATES:
        params = dict(base)
        params["use_anticorrelation"] = True
        params["anticorrelation_ratio"] = ratio
        name = _spec_name_from_params(params)
        if name not in seen:
            specs.append({"name": name, "params": params})
            seen.add(name)
    return specs


def _build_weight_plan(state: SearchState) -> list[dict]:
    if state.search_phase == "phase1":
        return _build_phase_one_search_plan(state)
    if state.search_phase == "phase3":
        return _build_phase_three_search_plan(state)
    return _build_phase_two_weight_plan(state)


def _build_window_plan(state: SearchState, params: dict) -> list[dict]:
    base = dict(params)
    plan = []
    for iw in _adaptive_iw_candidates(state):
        spec = dict(base)
        spec["iw"] = iw
        plan.append({"name": f"iw{iw}_best", "params": spec})
    for rw in _adaptive_rw_candidates(state):
        spec = dict(base)
        spec["rw"] = rw
        plan.append({"name": f"rw{rw}_best", "params": spec})
    return plan


def _hydrate_active_iteration(state: SearchState, active_iteration: dict) -> dict:
    hydrated = dict(active_iteration)
    hydrated["search_state"] = dict(hydrated.get("search_state", state.to_dict()))
    hydrated["search_phase"] = str(hydrated.get("search_phase", hydrated["search_state"].get("search_phase", state.search_phase)))
    state.search_phase = hydrated["search_phase"]
    hydrated["started_at"] = float(hydrated.get("started_at", time.time()))
    hydrated["phase"] = str(hydrated.get("phase", "search"))
    hydrated["search_index"] = int(hydrated.get("search_index", hydrated.get("weight_index", 0)))
    hydrated["param_index"] = int(hydrated.get("param_index", 0))
    hydrated["window_index"] = int(hydrated.get("window_index", 0))
    hydrated["recent"] = [dict(item) for item in list(hydrated.get("recent", []))[-50:]]

    top_results = hydrated.get("top_results")
    if isinstance(top_results, list):
        hydrated["top_results"] = [dict(item) for item in top_results[:15]]
    else:
        all_tested = hydrated.get("all_tested", [])
        hydrated["top_results"] = []
        for entry in all_tested:
            hydrated["top_results"] = _merge_top_results(hydrated["top_results"], entry)
    hydrated["tested_count"] = _active_iteration_tested_count(hydrated)

    hydrated["search_plan"] = _build_weight_plan(state)
    hydrated["param_plan"] = _build_phase_two_param_plan(state) if state.search_phase == "phase2" else []
    if hydrated["phase"] == "windows":
        best_params = hydrated.get("best_params") or _params_from_state(state)
        hydrated["window_plan"] = _build_window_plan(state, best_params)
    else:
        hydrated["window_plan"] = []
    hydrated["total"] = len(hydrated["search_plan"]) + len(hydrated["param_plan"]) + len(hydrated["window_plan"])
    return hydrated


def run_iteration(state: SearchState, draws, active_iteration: dict | None = None) -> dict | None:
    """Run one iteration of narrow search. Returns result dict."""
    if active_iteration is None:
        state.iteration += 1
        search_plan = _build_weight_plan(state)
        param_plan = _build_phase_two_param_plan(state) if state.search_phase == "phase2" else []
        active_iteration = {
            "search_state": state.to_dict(),
            "search_phase": state.search_phase,
            "started_at": time.time(),
            "phase": "search",
            "search_plan": search_plan,
            "search_index": 0,
            "param_plan": param_plan,
            "param_index": 0,
            "window_plan": [],
            "window_index": 0,
            "total": len(search_plan) + len(param_plan),
            "tested_count": 0,
            "recent": [],
            "top_results": [],
            "best_metric": -1.0,
            "best_config": state.best_strategy,
            "best_progress": {},
            "best_result": None,
            "best_params": None,
            "current_config": None,
        }
        _save_loop_state(state, active_iteration)
    else:
        state = SearchState.from_dict(active_iteration.get("search_state", state.to_dict()))
        active_iteration = _hydrate_active_iteration(state, active_iteration)

    def execute_specs(specs: list[dict], start_index: int, index_key: str) -> tuple[int, bool]:
        index = start_index
        while index < len(specs):
            if stop_requested():
                _save_loop_state(state, active_iteration)
                _write_stopped(state, active_iteration)
                return index, False
            spec = specs[index]
            params = spec["params"]
            active_iteration["current_config"] = spec["name"]
            result_obj = backtest_strategy(
                ACTIVE_CONFIG,
                spec["name"],
                _build_rank_fn(state, params),
                draws,
                ticket_size=ACTIVE_CONFIG.ticket_size,
                recent_window=int(params["rw"]),
                initial_window=int(params["iw"]),
            )
            entry = {
                "config": spec["name"],
                "average_hits": round(result_obj.average_hits, 6),
                "portfolio_avg": round(result_obj.portfolio_average_best_hits, 6),
            }
            active_iteration["tested_count"] = _active_iteration_tested_count(active_iteration) + 1
            active_iteration["recent"].append(entry)
            active_iteration["recent"] = active_iteration["recent"][-50:]
            active_iteration["top_results"] = _merge_top_results(active_iteration.get("top_results", []), entry)
            if result_obj.average_hits > active_iteration["best_metric"]:
                active_iteration["best_metric"] = result_obj.average_hits
                active_iteration["best_config"] = spec["name"]
                active_iteration["best_progress"] = {
                    "config": spec["name"],
                    "average_hits": round(result_obj.average_hits, 6),
                }
                active_iteration["best_result"] = {
                    "hit_rate_2_plus": round(result_obj.hit_rate_2_plus, 4),
                    "hit_rate_3_plus": round(result_obj.hit_rate_3_plus, 4),
                    "portfolio_avg": round(result_obj.portfolio_average_best_hits, 6),
                }
                active_iteration["best_params"] = dict(params)
            index += 1
            active_iteration[index_key] = index
            _save_loop_state(state, active_iteration)
            _write_progress(
                _active_iteration_tested_count(active_iteration),
                active_iteration["total"],
                spec["name"],
                state,
                active_iteration["started_at"],
                active_iteration["best_progress"],
                active_iteration["recent"],
            )
        return index, True

    set_search_weights()
    try:
        if active_iteration["phase"] == "search":
            active_iteration["search_index"], completed = execute_specs(
                active_iteration["search_plan"],
                int(active_iteration.get("search_index", 0)),
                "search_index",
            )
            if not completed:
                return None
            if active_iteration["search_phase"] == "phase2" and active_iteration["param_plan"]:
                active_iteration["phase"] = "params"
                active_iteration["total"] = (
                    len(active_iteration["search_plan"])
                    + len(active_iteration["param_plan"])
                    + len(active_iteration.get("window_plan", []))
                )
                _save_loop_state(state, active_iteration)
            elif active_iteration["search_phase"] in {"phase1", "phase2"}:
                best_params = active_iteration["best_params"] or _params_from_state(state)
                active_iteration["window_plan"] = _build_window_plan(state, best_params)
                active_iteration["phase"] = "windows"
                active_iteration["total"] = (
                    len(active_iteration["search_plan"])
                    + len(active_iteration["param_plan"])
                    + len(active_iteration["window_plan"])
                )
                _save_loop_state(state, active_iteration)

        if active_iteration["phase"] == "params":
            active_iteration["param_index"], completed = execute_specs(
                active_iteration["param_plan"],
                int(active_iteration.get("param_index", 0)),
                "param_index",
            )
            if not completed:
                return None
            best_params = active_iteration["best_params"] or _params_from_state(state)
            active_iteration["window_plan"] = _build_window_plan(state, best_params)
            active_iteration["phase"] = "windows"
            active_iteration["total"] = (
                len(active_iteration["search_plan"])
                + len(active_iteration["param_plan"])
                + len(active_iteration["window_plan"])
            )
            _save_loop_state(state, active_iteration)

        if active_iteration["phase"] == "windows":
            active_iteration["window_index"], completed = execute_specs(
                active_iteration["window_plan"],
                int(active_iteration.get("window_index", 0)),
                "window_index",
            )
            if not completed:
                return None
    finally:
        reset_default_weights()

    elapsed = time.time() - active_iteration["started_at"]
    best_result = active_iteration["best_result"]
    best_metric = active_iteration["best_metric"] if best_result else state.best_metric
    best_config = active_iteration["best_config"] if best_result else state.best_strategy
    result = {
        "metric": round(best_metric, 6) if best_result else round(state.best_metric, 6),
        "strategy": best_config or "no_improvement",
        "best_config": best_config or "no_improvement",
        "hit_rate_2_plus": best_result["hit_rate_2_plus"] if best_result else 0,
        "hit_rate_3_plus": best_result["hit_rate_3_plus"] if best_result else 0,
        "portfolio_avg": best_result["portfolio_avg"] if best_result else 0,
        "total_tested": _active_iteration_tested_count(active_iteration),
        "top_15": list(active_iteration.get("top_results", [])),
        "elapsed_seconds": round(elapsed, 2),
        "iteration": state.iteration,
    }

    prev_best = state.best_metric
    improved = bool(best_result and best_metric > prev_best)
    result["improved"] = improved
    result["status"] = "KEEP" if improved else "DISCARD"

    # Broadcast to OpenClawThink dashboard
    if LIVE_THOUGHTS_ENABLED:
        thought_experiment_complete(
            f"AutoLoop-{state.search_phase}-iter{state.iteration}",
            best_metric if best_result else prev_best,
            "KEEP" if improved else "DISCARD"
        )

    if improved:
        state.best_metric = best_metric
        state.best_strategy = best_config
        _apply_params_to_state(state, active_iteration["best_params"])
        if state.search_phase == "phase2":
            state.narrow()
        else:
            state.no_improve_count = 0
        improvement = ((best_metric - prev_best) / prev_best * 100) if prev_best > 0 else 0
        desc = (f"AutoLoop {state.search_phase} iter{state.iteration} KEEP: {best_config} -> {best_metric:.6f} "
                f"({active_iteration['total']} configs, radius={state.radius:.3f})")
        _append_results_tsv(best_metric, best_config, result, elapsed, desc, prev_best)
        try:
            import forward_test

            forward_test.ensure_snapshot(ACTIVE_CONFIG.id, state)
        except Exception as exc:
            print(f"Forward-test snapshot skipped: {exc}")
        # utilizatorul află pe Telegram când sistemul găsește ceva mai bun
        try:
            import ai_actions as _bus
            import significance as _sig

            _edge = _sig.config_edge_stats(
                ACTIVE_CONFIG, best_metric, max(len(draws) - int(state.iw), 1),
                _sig.count_experiments(ACTIVE_CONFIG.results_file),
            )
            _bus.notify_event(
                f"📈 <b>Strategie îmbunătățită</b> — {ACTIVE_CONFIG.name}\n"
                f"{best_config}\n"
                f"average_hits: <b>{best_metric:.6f}</b> "
                f"(+{improvement:.2f}% vs {prev_best:.6f}; random {ACTIVE_CONFIG.baseline:.3f})\n"
                f"<i>Realitate statistică: z={_edge['z']:+.2f} după {_edge['n_experiments']} configurații "
                f"testate (p ajustat {_edge['p_adjusted']:.3f}) → {_edge['verdict_text']}</i>"
            )
        except Exception:
            pass
        # Run walkforward validation on new best strategy
        try:
            rank_fn = build_rank_fn_from_state(state, config=ACTIVE_CONFIG)
            wf_result = run_and_save_walkforward(
                lottery_id=ACTIVE_CONFIG.id,
                rank_fn=rank_fn,
                recent_window=state.rw,
                backtest_score=best_metric,
                strategy_name=best_config,
            )
            if wf_result:
                print(f"  Walkforward: score={wf_result.wf_score:.6f}, gap={wf_result.gap:.6f}, stability={wf_result.stability:.4f}")
        except Exception as wf_exc:
            print(f"  Walkforward failed: {wf_exc}")
        if LIVE_THOUGHTS_ENABLED:
            thought_new_best(best_config, best_metric, improvement)
            thought_strategy_evolving(best_config, best_metric)
    else:
        if state.search_phase == "phase2":
            state.widen()
        discard_metric = best_metric if best_result else prev_best
        discard_strategy = best_config or "no_improvement"
        if best_result and best_config:
            desc = (
                f"AutoLoop {state.search_phase} iter{state.iteration} DISCARD: "
                f"{best_config} -> {discard_metric:.6f}; no improvement over {prev_best:.6f} "
                f"({active_iteration['total']} configs, radius={state.radius:.3f})"
            )
        else:
            desc = (
                f"AutoLoop {state.search_phase} iter{state.iteration} DISCARD: "
                f"no improvement over {prev_best:.6f} "
                f"({active_iteration['total']} configs, radius={state.radius:.3f})"
            )
        _append_results_tsv(discard_metric, discard_strategy, result, elapsed, desc, prev_best)
        if LIVE_THOUGHTS_ENABLED:
            thought_discard(f"no improvement over {prev_best:.6f}", discard_metric)

    if state.search_phase == "phase1":
        state.search_phase = "phase2"
    elif state.search_phase == "phase2":
        if not improved and state.no_improve_count >= 1:
            state.search_phase = "phase3"
        else:
            state.search_phase = "phase2"
    else:
        state.search_phase = "phase2"

    _save_loop_state(state, None)
    return result


def _update_state_from_config(state: SearchState, config_name: str) -> None:
    """Parse a config name like tf3.25_hc2.1_b4.05_sb1.0 and update state."""
    recognized_extras = []
    for part in config_name.replace("ens4_", "").replace("ft_", "").split("_"):
        try:
            if part.startswith("tf"):
                state.tf = float(part[2:])
            elif part.startswith("hc"):
                state.hc = float(part[2:])
            elif part.startswith("bf"):
                state.bf = float(part[2:])
                recognized_extras.append("bf")
            elif part.startswith("jc"):
                state.jc = float(part[2:])
                recognized_extras.append("jc")
            elif part.startswith("dd"):
                state.dd = float(part[2:])
                recognized_extras.append("dd")
            elif part.startswith("delay"):
                state.delay = float(part[5:])
                recognized_extras.append("delay")
            elif part.startswith("momentum"):
                state.momentum = float(part[8:])
                recognized_extras.append("momentum")
            elif part.startswith("ps"):
                state.prior_strength = float(part[2:])
            elif part.startswith("el"):
                state.ewma_lambda = float(part[2:])
            elif part.startswith("acr"):
                state.anticorrelation_ratio = float(part[3:])
            elif part.startswith("ac"):
                state.use_anticorrelation = part[2:] == "1"
            elif part.startswith("b") and not part.startswith("best"):
                state.bal = float(part[1:])
            elif part.startswith("sb"):
                state.sb = float(part[2:])
                recognized_extras.append("sb")
            elif part.startswith("iw"):
                state.iw = int(part[2:])
            elif part.startswith("rw"):
                state.rw = int(part[2:])
            else:
                match = CONFIG_TOKEN_RE.match(part)
                if match:
                    key, value = match.groups()
                    if key in EXTRA_STRATEGIES:
                        state.set_extra_weight(key, float(value))
                        recognized_extras.append(key)
        except ValueError:
            pass
    if recognized_extras:
        ordered = [name for name in EXTRA_STRATEGIES if name in recognized_extras]
        state.extra_strategies = ordered


def _rebaseline_best(state: SearchState, draws, results_file: Path | None = None) -> tuple[SearchState, bool]:
    """Recalculează scorul best-ului pe istoricul și codul CURENTE.

    best_metric stocat provine din rularea care l-a găsit: între timp istoricul
    crește, iar corecțiile de cod (ex. eliminarea informațiilor viitoare din
    indicatorul jackpot, validarea CSV) schimbă scorul aceleiași configurații.
    Fără recalculare, bucla ar compara candidații cu un prag care nu mai există
    și n-ar mai putea face KEEP. Scrie un rând REBASELINE în results.tsv.
    """
    if state.best_metric <= 0 or not state.best_strategy or not draws:
        return state, False
    try:
        rank_fn = build_rank_fn_from_state(state, config=ACTIVE_CONFIG)
        result = backtest_strategy(
            ACTIVE_CONFIG, "rebaseline", rank_fn, draws,
            ticket_size=ACTIVE_CONFIG.ticket_size,
            recent_window=int(state.rw), initial_window=int(state.iw),
        )
    except Exception as exc:
        print(f"Rebaseline skipped: {exc}")
        return state, False
    fresh = round(result.average_hits, 6)
    if abs(fresh - state.best_metric) < 1e-6:
        return state, False
    previous = state.best_metric
    desc = (f"REBASELINE: {state.best_strategy} {previous:.6f} -> {fresh:.6f} "
            f"(istoric/cod curent, {len(draws)} extrageri)")
    row = (f"auto\t{fresh}\tREBASELINE\t{state.best_strategy}\t"
           f"{round(result.hit_rate_2_plus, 4)}\t{round(result.hit_rate_3_plus, 4)}\t"
           f"{round(result.portfolio_average_best_hits, 6)}\t0\t{desc}\n")
    with open(results_file or ACTIVE_CONFIG.results_file, "a", encoding="utf-8") as handle:
        handle.write(row)
    print(f"Rebaseline: best {previous:.6f} -> {fresh:.6f} on current history/code")
    state.best_metric = fresh
    return state, True


def run_loop(max_iterations: int = 0) -> None:
    """Run the auto-loop. max_iterations=0 means infinite."""
    global _JACKPOT_DATA, _DRAW_DATES
    if not _acquire_lock():
        print("ERROR: Another experiment is already running. Exiting.")
        sys.exit(1)
    clear_stop_request()
    register_ai_strategies()
    print("Loading draws...")
    draws = load_lottery_draws(ACTIVE_CONFIG.id)
    print(f"Loaded {len(draws)} draws")
    _JACKPOT_DATA, jackpot_source = load_jackpot_data_with_source(ACTIVE_CONFIG.id, draws)
    _DRAW_DATES = [draw.draw_date for draw in draws]
    print(f"Loaded jackpot data: {len(_JACKPOT_DATA)} entries ({jackpot_source})")

    state, active_iteration = load_search_state()
    if state.best_metric <= 0:
        state = load_best_from_results()
    state, rebased = _rebaseline_best(state, draws)
    if rebased:
        _save_loop_state(state, active_iteration)
    # forward test: îngheață parametrii curenți (evaluați doar pe extragerile viitoare)
    try:
        import forward_test

        forward_test.ensure_snapshot(ACTIVE_CONFIG.id, state)
    except Exception as exc:
        print(f"Forward-test snapshot skipped: {exc}")
    print(f"Starting from best: {state.best_metric:.6f} "
          f"(tf={state.tf}, hc={state.hc}, bal={state.bal}, sb={state.sb}, "
          f"delay={state.delay}, momentum={state.momentum}, jc={state.jc}, "
          f"bf={state.bf}, dd={state.dd}, iw={state.iw}, rw={state.rw}, "
          f"ps={state.prior_strength}, el={state.ewma_lambda}, ac={state.use_anticorrelation})")
    print(f"Search: phase={state.search_phase}, radius={state.radius}, step={state.step}")
    print("=" * 60)

    # Broadcast loop start
    if LIVE_THOUGHTS_ENABLED:
        thought_auto_loop_start(ACTIVE_CONFIG.id)
        thought_strategy_evolving(state.best_strategy, state.best_metric)

    # Broadcast loop start
    if LIVE_THOUGHTS_ENABLED:
        thought_auto_loop_start(ACTIVE_CONFIG.id)
        info(f"Best: {state.best_metric:.6f} | phase={state.search_phase} radius={state.radius:.3f}")
        thought_strategy_evolving(state.best_strategy, state.best_metric)

    iteration = 0
    try:
        while True:
            if max_iterations > 0 and iteration >= max_iterations:
                break

            if active_iteration:
                print(f"\n--- Resuming iteration {state.iteration} ---")
                print(f"Resume point: search_phase={active_iteration.get('search_phase', state.search_phase)} "
                      f"phase={active_iteration.get('phase')} "
                      f"search={active_iteration.get('search_index', active_iteration.get('weight_index', 0))} "
                      f"params={active_iteration.get('param_index', 0)} "
                      f"windows={active_iteration.get('window_index', 0)}")
            else:
                print(f"\n--- Iteration {state.iteration + 1} ---")
                print(f"Center: phase={state.search_phase} tf={state.tf:.3f} hc={state.hc:.3f} "
                      f"bal={state.bal:.3f} sb={state.sb:.3f} "
                      f"delay={state.delay:.3f} momentum={state.momentum:.3f} "
                      f"jc={state.jc:.3f} bf={state.bf:.3f} dd={state.dd:.3f}")
                print(f"Search: radius={state.radius:.3f} step={state.step:.3f} "
                      f"strategies={state.extra_strategies} "
                      f"ps={state.prior_strength:.3f} el={state.ewma_lambda:.3f} "
                      f"ac={state.use_anticorrelation}:{state.anticorrelation_ratio:.3f}")

            result = run_iteration(state, draws, active_iteration)
            active_iteration = None
            if result is None:
                state, active_iteration = load_search_state()
                print("Loop stop requested. Checkpoint saved.")
                break

            state, _active_unused = load_search_state()
            iteration += 1
            status = result.get("status", "KEEP" if result.get("improved") else "DISCARD")
            print(f"Result: {result['metric']:.6f} ({status}) "
                  f"- {result['best_config']} [{result['total_tested']} configs in "
                  f"{result['elapsed_seconds']:.0f}s]")

            # Broadcast iteration result
            if LIVE_THOUGHTS_ENABLED:
                thought_experiment_complete(f"iter{state.iteration}", result['metric'], status)
                if status == "KEEP":
                    improvement = ((result['metric'] - state.best_metric) / state.best_metric * 100) if state.best_metric > 0 else 0
                    thought_new_best(result['best_config'], result['metric'], improvement)
                else:
                    thought_discard(f"no improvement over {state.best_metric:.6f}", result['metric'])

            if max_iterations > 0 and iteration >= max_iterations:
                _write_finished(result)
                break

            # Brief pause between iterations
            time.sleep(2)

    except KeyboardInterrupt:
        print("\n\nLoop stopped by user.")
        _write_finished({"status": "stopped_by_user", "iteration": iteration,
                         "best_metric": state.best_metric,
                         "best_strategy": state.best_strategy})
    finally:
        _release_lock()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run a single iteration")
    parser.add_argument("--lottery", default="multi_multi_pl", help="Lottery id to optimize")
    args = parser.parse_args()
    configure_runtime(args.lottery)
    run_loop(max_iterations=1 if args.once else 0)
