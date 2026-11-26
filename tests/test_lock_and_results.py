"""Tests for lock mechanism and results.tsv scoping — covers CB-2, CB-4 gaps."""
from __future__ import annotations

import json
import os
from pathlib import Path

import auto_loop
from lock_utils import read_lock_owner


# ---------------------------------------------------------------------------
# Lock mechanism tests (CB-2 fix verification)
# ---------------------------------------------------------------------------

def test_pid_alive_detects_current_process():
    """_pid_alive should return True for the current process."""
    assert auto_loop._pid_alive(os.getpid()) is True


def test_pid_alive_rejects_dead_pid():
    """_pid_alive should return False for a PID that doesn't exist."""
    # Use a very high PID that almost certainly doesn't exist
    assert auto_loop._pid_alive(4_000_000) is False


def test_acquire_lock_succeeds_when_no_lock(tmp_path, monkeypatch):
    lock_file = tmp_path / ".experiment.lock"
    monkeypatch.setattr(auto_loop, "LOCK_FILE", lock_file)

    assert auto_loop._acquire_lock() is True
    assert lock_file.exists()
    owner = read_lock_owner(lock_file)
    assert owner.pid == os.getpid()
    assert owner.kind == "autoloop"
    assert owner.lottery_id == auto_loop.ACTIVE_CONFIG.id


def test_acquire_lock_rejects_running_process(tmp_path, monkeypatch):
    """Lock should be rejected when another live process holds it."""
    lock_file = tmp_path / ".experiment.lock"
    fake_pid = 9999999
    lock_file.write_text(str(fake_pid))
    monkeypatch.setattr(auto_loop, "LOCK_FILE", lock_file)
    # Mock _pid_alive to report fake_pid as alive
    monkeypatch.setattr(auto_loop, "_pid_alive", lambda pid: pid == fake_pid)
    monkeypatch.setattr(auto_loop, "lock_matches_process", lambda owner, **kwargs: owner.pid == fake_pid)

    assert auto_loop._acquire_lock() is False


def test_acquire_lock_steals_stale_lock(tmp_path, monkeypatch):
    """Lock should be acquired when the lock holder is dead (stale lock)."""
    lock_file = tmp_path / ".experiment.lock"
    lock_file.write_text("4000000")  # PID that doesn't exist
    monkeypatch.setattr(auto_loop, "LOCK_FILE", lock_file)
    monkeypatch.setattr(auto_loop, "_pid_alive", lambda pid: False)

    assert auto_loop._acquire_lock() is True
    assert read_lock_owner(lock_file).pid == os.getpid()


def test_acquire_lock_steals_corrupt_lock(tmp_path, monkeypatch):
    """Lock should be acquired when the lock file contains garbage."""
    lock_file = tmp_path / ".experiment.lock"
    lock_file.write_text("not_a_pid")
    monkeypatch.setattr(auto_loop, "LOCK_FILE", lock_file)

    assert auto_loop._acquire_lock() is True


def test_release_lock_removes_file(tmp_path, monkeypatch):
    lock_file = tmp_path / ".experiment.lock"
    lock_file.write_text(str(os.getpid()))
    monkeypatch.setattr(auto_loop, "LOCK_FILE", lock_file)

    auto_loop._release_lock()
    assert not lock_file.exists()


def test_release_lock_keeps_foreign_owner_file(tmp_path, monkeypatch):
    lock_file = tmp_path / ".experiment.lock"
    lock_file.write_text("4321")
    monkeypatch.setattr(auto_loop, "LOCK_FILE", lock_file)

    auto_loop._release_lock()

    assert lock_file.exists()
    assert lock_file.read_text().strip() == "4321"


def test_acquire_lock_steals_live_foreign_process_lock(tmp_path, monkeypatch):
    lock_file = tmp_path / ".experiment.lock"
    lock_file.write_text("9876", encoding="utf-8")
    monkeypatch.setattr(auto_loop, "LOCK_FILE", lock_file)
    monkeypatch.setattr(auto_loop, "_pid_alive", lambda pid: pid == 9876)
    monkeypatch.setattr(auto_loop, "lock_matches_process", lambda owner, **kwargs: False)

    assert auto_loop._acquire_lock() is True
    assert read_lock_owner(lock_file).pid == os.getpid()


def test_write_json_atomic_retries_transient_permission_error(tmp_path, monkeypatch):
    target = tmp_path / "progress.json"
    replace_calls = {"count": 0}
    sleep_calls = []
    original_replace = auto_loop.os.replace

    def flaky_replace(src: str, dst: str):
        replace_calls["count"] += 1
        if replace_calls["count"] < 3:
            raise PermissionError("file temporarily locked")
        return original_replace(src, dst)

    monkeypatch.setattr(auto_loop.os, "replace", flaky_replace)
    monkeypatch.setattr(auto_loop.time, "sleep", lambda delay: sleep_calls.append(delay))

    auto_loop._write_json_atomic(target, {"status": "running", "tested": 12})

    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "running", "tested": 12}
    assert replace_calls["count"] == 3
    assert sleep_calls == [auto_loop.ATOMIC_WRITE_RETRY_DELAY, auto_loop.ATOMIC_WRITE_RETRY_DELAY]
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_json_atomic_falls_back_to_recovery_file_when_primary_stays_locked(tmp_path, monkeypatch):
    target = tmp_path / "progress.json"
    original_replace = auto_loop.os.replace

    def selective_replace(src: str, dst: str):
        if Path(dst) == target:
            raise PermissionError("target remains locked")
        return original_replace(src, dst)

    monkeypatch.setattr(auto_loop.os, "replace", selective_replace)

    auto_loop._write_json_atomic(target, {"status": "running", "tested": 17})

    recovery_file = auto_loop.recovery_path(target)
    assert not target.exists()
    assert json.loads(recovery_file.read_text(encoding="utf-8")) == {"status": "running", "tested": 17}


def test_load_search_state_uses_recovery_file_when_primary_missing(tmp_path, monkeypatch):
    loop_state = tmp_path / "autoloop_state.json"
    recovery_file = auto_loop.recovery_path(loop_state)
    recovery_file.write_text(
        json.dumps(
            {
                "version": 1,
                "search_state": {"tf": 3.4, "hc": 2.1, "bal": 4.6, "best_metric": 2.5, "best_strategy": "fallback_best"},
                "active_iteration": {"phase": "weights", "weight_index": 4},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(auto_loop, "LOOP_STATE_FILE", loop_state)
    monkeypatch.setattr(auto_loop, "load_best_from_results", lambda: auto_loop.SearchState())

    state, active_iteration = auto_loop.load_search_state()

    assert state.tf == 3.4
    assert state.best_metric == 2.5
    assert state.best_strategy == "fallback_best"
    assert active_iteration == {"phase": "weights", "weight_index": 4}


def test_compact_active_iteration_drops_heavy_fields_and_preserves_summary():
    state = auto_loop.SearchState(tf=3.1, hc=2.2, bal=4.3, sb=1.1, best_metric=1.2)
    entries = [
        {"config": f"cfg{i}", "average_hits": round(0.5 + i / 1000, 6), "portfolio_avg": 1.0 + i / 1000}
        for i in range(40)
    ]
    active_iteration = {
        "search_state": state.to_dict(),
        "started_at": 123.0,
        "phase": "weights",
        "weight_plan": [{"name": "w1"}, {"name": "w2"}],
        "weight_index": 7,
        "window_plan": [{"name": "x1"}],
        "window_index": 0,
        "total": 12,
        "all_tested": entries,
        "recent": entries[-12:],
        "best_metric": 1.234,
        "best_config": "cfg39",
        "best_progress": {"config": "cfg39", "average_hits": 0.539},
        "best_result": {"hit_rate_2_plus": 0.4, "hit_rate_3_plus": 0.1, "portfolio_avg": 1.5},
        "best_params": {"tf": 3.1, "hc": 2.2, "bal": 4.3, "iw": 50, "rw": 250},
        "current_config": "cfg39",
    }

    compact = auto_loop._compact_active_iteration(active_iteration)

    assert "all_tested" not in compact
    assert "weight_plan" not in compact
    assert "window_plan" not in compact
    assert compact["tested_count"] == 40
    assert len(compact["recent"]) == 12
    assert len(compact["top_results"]) == 15
    assert compact["top_results"][0]["config"] == "cfg39"


def test_hydrate_active_iteration_rebuilds_plans_from_compact(monkeypatch):
    state = auto_loop.SearchState(tf=3.1, hc=2.2, bal=4.3, sb=1.1, iw=50, rw=250)
    compact = {
        "search_state": state.to_dict(),
        "started_at": 321.0,
        "phase": "windows",
        "weight_index": 9,
        "window_index": 2,
        "tested_count": 11,
        "recent": [{"config": "cfg11", "average_hits": 0.9, "portfolio_avg": 1.4}],
        "top_results": [{"config": "cfg11", "average_hits": 0.9, "portfolio_avg": 1.4}],
        "best_metric": 0.9,
        "best_config": "cfg11",
        "best_progress": {"config": "cfg11", "average_hits": 0.9},
        "best_result": {"hit_rate_2_plus": 0.4, "hit_rate_3_plus": 0.1, "portfolio_avg": 1.4},
        "best_params": {"tf": 3.1, "hc": 2.2, "bal": 4.3, "iw": 40, "rw": 200},
    }

    monkeypatch.setattr(auto_loop, "_build_weight_plan", lambda _state: [{"name": "w1"}, {"name": "w2"}, {"name": "w3"}])
    monkeypatch.setattr(auto_loop, "_build_window_plan", lambda _state, _params: [{"name": "x1"}, {"name": "x2"}])
    monkeypatch.setattr(auto_loop, "_adaptive_iw_candidates", lambda _state: [20, 40])
    monkeypatch.setattr(auto_loop, "_adaptive_rw_candidates", lambda _state: [150, 200, 250])

    hydrated = auto_loop._hydrate_active_iteration(state, compact)

    assert hydrated["search_plan"] == [{"name": "w1"}, {"name": "w2"}, {"name": "w3"}]
    assert hydrated["window_plan"] == [{"name": "x1"}, {"name": "x2"}]
    assert hydrated["total"] == 5
    assert hydrated["tested_count"] == 11
    assert hydrated["top_results"][0]["config"] == "cfg11"


# ---------------------------------------------------------------------------
# Results.tsv scoping tests (CB-4 fix verification)
# ---------------------------------------------------------------------------

def test_append_results_uses_active_config_file(tmp_path, monkeypatch):
    """Results should be written to ACTIVE_CONFIG.results_file, not the module global."""
    from lottery_config import LotteryConfig, RESULTS_HEADER

    # Create a fake config pointing to tmp_path
    config = LotteryConfig(
        id="test_lottery",
        name="Test",
        numbers_drawn=5,
        number_range=42,
        ticket_size=5,
    )
    results_file = tmp_path / "results.tsv"
    results_file.write_text(RESULTS_HEADER, encoding="utf-8")

    # Monkeypatch ACTIVE_CONFIG to use our test config
    monkeypatch.setattr(auto_loop, "ACTIVE_CONFIG", config)
    # Also override the results_file property for the test config
    monkeypatch.setattr(type(config), "results_file", property(lambda self: results_file))

    auto_loop._append_results_tsv(
        metric=1.234,
        strategy="test_strategy",
        result={"hit_rate_2_plus": 0.5, "hit_rate_3_plus": 0.1, "portfolio_avg": 1.3},
        elapsed=10.5,
        desc="test description",
        best_metric=1.0,
    )

    content = results_file.read_text(encoding="utf-8")
    assert "test_strategy" in content
    assert "KEEP" in content  # 1.234 > 1.0


def test_append_results_writes_discard_when_below_best(tmp_path, monkeypatch):
    from lottery_config import LotteryConfig, RESULTS_HEADER

    config = LotteryConfig(
        id="test_lottery",
        name="Test",
        numbers_drawn=5,
        number_range=42,
        ticket_size=5,
    )
    results_file = tmp_path / "results.tsv"
    results_file.write_text(RESULTS_HEADER, encoding="utf-8")

    monkeypatch.setattr(auto_loop, "ACTIVE_CONFIG", config)
    monkeypatch.setattr(type(config), "results_file", property(lambda self: results_file))

    auto_loop._append_results_tsv(
        metric=0.5,
        strategy="bad_strategy",
        result={},
        elapsed=5.0,
        desc="no improvement",
        best_metric=1.0,
    )

    content = results_file.read_text(encoding="utf-8")
    assert "bad_strategy" in content
    assert "DISCARD" in content  # 0.5 <= 1.0
