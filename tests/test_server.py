from __future__ import annotations

import json
import os
import subprocess
import asyncio
from collections import Counter
from datetime import date
from types import SimpleNamespace

from fastapi.testclient import TestClient

import auto_loop
from lock_utils import read_lock_owner
import load_data
import server


def test_get_best_strategy_uses_persisted_autoloop_state(tmp_path, monkeypatch):
    config = SimpleNamespace(
        id="multi_multi_pl",
        loop_state_file=tmp_path / "autoloop_state.json",
        results_file=tmp_path / "results.tsv",
    )
    monkeypatch.setattr(auto_loop, "get_lottery_config", lambda lottery_id: config)
    monkeypatch.setattr(auto_loop, "ensure_lottery_storage", lambda _config: None)

    state = auto_loop.SearchState(tf=3.7, hc=2.2, bal=4.8, sb=1.4, iw=40, rw=250, best_metric=2.61)
    state.best_strategy = "iw40_best"
    config.loop_state_file.write_text(
        json.dumps({"version": 1, "search_state": state.to_dict()}),
        encoding="utf-8",
    )
    config.results_file.write_text(
        "commit\taverage_hits\tstatus\tstrategy\thit_rate_2_plus\thit_rate_3_plus\tportfolio_avg\telapsed_s\tdescription\n",
        encoding="utf-8",
    )

    best = server._get_best_strategy("multi_multi_pl")

    assert best["name"] == "iw40_best"
    assert best["tf"] == 3.7
    assert best["b"] == 4.8
    assert best["iw"] == 40
    assert best["rw"] == 250


def test_generate_numbers_uses_ranked_state(monkeypatch):
    draws = [
        SimpleNamespace(numbers=(1, 2, 3), bonuses=(), draw_date=SimpleNamespace(day=1, month=1, year=2026)),
        SimpleNamespace(numbers=(4, 5, 6), bonuses=(), draw_date=SimpleNamespace(day=2, month=1, year=2026)),
    ]
    state = auto_loop.SearchState(tf=3.2, hc=2.1, bal=4.4, sb=1.3, best_metric=2.55)
    state.best_strategy = "tf3.2_hc2.1_b4.4_sb1.3"

    monkeypatch.setattr(server, "_get_best_strategy", lambda lottery_id: {
        "name": state.best_strategy,
        "metric": state.best_metric,
        "tf": state.tf,
        "hc": state.hc,
        "b": state.bal,
        "sb": state.sb,
        "delay": state.delay,
        "momentum": state.momentum,
        "iw": state.iw,
        "rw": state.rw,
        "extra_strategies": list(state.extra_strategies),
    })
    monkeypatch.setattr(auto_loop, "load_search_state_for_lottery", lambda lottery_id: (state, None))
    monkeypatch.setattr(auto_loop, "load_best_from_results_for_lottery", lambda lottery_id: state)
    monkeypatch.setattr(auto_loop, "rank_numbers_from_draws", lambda _draws, _state, lottery_id=None: list(range(80, 0, -1)))
    monkeypatch.setattr(server, "load_lottery_draws", lambda lottery_id: draws)

    client = TestClient(server.app)
    response = client.get("/api/generate?ticket_count=2&nums_per_ticket=3")

    assert response.status_code == 200
    payload = response.json()
    assert payload["strategy"] == state.best_strategy
    assert payload["weights"]["tf"] == 3.2
    assert payload["windows"]["rw"] == 200
    assert payload["tickets"][0]["main"] == [78, 79, 80]
    assert payload["tickets"][0]["bonus"] == []


def test_generate_numbers_clamps_ticket_size_to_lottery_rule(monkeypatch):
    draws = [
        SimpleNamespace(numbers=(1, 2, 3, 4, 5, 6), bonuses=(), draw_date=SimpleNamespace(day=1, month=1, year=2026)),
        SimpleNamespace(numbers=(7, 8, 9, 10, 11, 12), bonuses=(), draw_date=SimpleNamespace(day=2, month=1, year=2026)),
    ]
    state = auto_loop.SearchState(best_metric=1.23)
    state.best_strategy = "tf3_hc2_b4_sb1"

    monkeypatch.setattr(server, "_get_best_strategy", lambda lottery_id: {
        "name": state.best_strategy,
        "metric": state.best_metric,
        "tf": state.tf,
        "hc": state.hc,
        "b": state.bal,
        "sb": state.sb,
        "delay": state.delay,
        "momentum": state.momentum,
        "iw": state.iw,
        "rw": state.rw,
        "extra_strategies": list(state.extra_strategies),
    })
    monkeypatch.setattr(auto_loop, "load_search_state_for_lottery", lambda lottery_id: (state, None))
    monkeypatch.setattr(auto_loop, "load_best_from_results_for_lottery", lambda lottery_id: state)
    monkeypatch.setattr(
        auto_loop,
        "rank_numbers_from_draws",
        lambda _draws, _state, lottery_id=None: list(range(49, 0, -1)),
    )
    monkeypatch.setattr(server, "load_lottery_draws", lambda lottery_id: draws)

    client = TestClient(server.app)
    response = client.get("/api/lotto_pl/generate?ticket_count=1&nums_per_ticket=10")

    assert response.status_code == 200
    payload = response.json()
    assert payload["lottery"] == "lotto_pl"
    assert len(payload["tickets"][0]["main"]) == 10


def test_generate_numbers_filters_invalid_ranked_numbers_and_builds_bonus(monkeypatch):
    draws = [
        SimpleNamespace(numbers=(1, 2, 3, 4, 5), bonuses=(7, 11), draw_date=SimpleNamespace(day=1, month=1, year=2026)),
        SimpleNamespace(numbers=(6, 7, 8, 9, 10), bonuses=(8, 10), draw_date=SimpleNamespace(day=2, month=1, year=2026)),
    ]
    state = auto_loop.SearchState(best_metric=1.11)
    state.best_strategy = "iw50_best"

    monkeypatch.setattr(server, "_get_best_strategy", lambda lottery_id: {
        "name": state.best_strategy,
        "metric": state.best_metric,
        "tf": state.tf,
        "hc": state.hc,
        "b": state.bal,
        "sb": state.sb,
        "delay": state.delay,
        "momentum": state.momentum,
        "iw": state.iw,
        "rw": state.rw,
        "extra_strategies": list(state.extra_strategies),
    })
    monkeypatch.setattr(auto_loop, "load_search_state_for_lottery", lambda lottery_id: (state, None))
    monkeypatch.setattr(auto_loop, "load_best_from_results_for_lottery", lambda lottery_id: state)
    monkeypatch.setattr(
        auto_loop,
        "rank_numbers_from_draws",
        lambda _draws, _state, lottery_id=None: [55, 54, 53, 52, 51, 50, 49, 48, 47, 46],
    )
    monkeypatch.setattr(server, "load_lottery_draws", lambda lottery_id: draws)

    client = TestClient(server.app)
    response = client.get("/api/eurojackpot/generate?ticket_count=1&nums_per_ticket=5")

    assert response.status_code == 200
    payload = response.json()
    assert payload["tickets"][0]["main"] == [46, 47, 48, 49, 50]
    assert len(payload["tickets"][0]["bonus"]) == 2
    assert all(1 <= number <= 12 for number in payload["tickets"][0]["bonus"])
    assert payload["last_draw_bonus"] == [8, 10]


def test_generate_numbers_supports_eurojackpot_system_ticket_sizes(monkeypatch):
    draws = [
        SimpleNamespace(numbers=(1, 2, 3, 4, 5), bonuses=(7, 11), draw_date=SimpleNamespace(day=1, month=1, year=2026)),
        SimpleNamespace(numbers=(6, 7, 8, 9, 10), bonuses=(8, 10), draw_date=SimpleNamespace(day=2, month=1, year=2026)),
    ]
    state = auto_loop.SearchState(best_metric=1.11)
    state.best_strategy = "rw300_best"

    monkeypatch.setattr(server, "_get_best_strategy", lambda lottery_id: {
        "name": state.best_strategy,
        "metric": state.best_metric,
        "tf": state.tf,
        "hc": state.hc,
        "b": state.bal,
        "sb": state.sb,
        "delay": state.delay,
        "momentum": state.momentum,
        "iw": state.iw,
        "rw": state.rw,
        "extra_strategies": list(state.extra_strategies),
    })
    monkeypatch.setattr(auto_loop, "load_search_state_for_lottery", lambda lottery_id: (state, None))
    monkeypatch.setattr(auto_loop, "load_best_from_results_for_lottery", lambda lottery_id: state)
    monkeypatch.setattr(
        auto_loop,
        "rank_numbers_from_draws",
        lambda _draws, _state, lottery_id=None: list(range(50, 0, -1)),
    )
    monkeypatch.setattr(server, "load_lottery_draws", lambda lottery_id: draws)

    client = TestClient(server.app)
    response = client.get("/api/eurojackpot/generate?ticket_count=1&nums_per_ticket=12&bonus_per_ticket=12")

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "system"
    assert payload["nums_per_ticket"] == 12
    assert payload["bonus_per_ticket"] == 12
    assert len(payload["tickets"][0]["main"]) == 12
    assert len(payload["tickets"][0]["bonus"]) == 12
    assert payload["rules"]["min_bonus_pick_count"] == 2
    assert payload["rules"]["max_bonus_pick_count"] == 12


def test_generate_numbers_varies_bonus_selection_between_calls(monkeypatch):
    draws = [
        SimpleNamespace(numbers=(1, 2, 3, 4, 5), bonuses=(7,), draw_date=SimpleNamespace(day=1, month=1, year=2026)),
        SimpleNamespace(numbers=(6, 7, 8, 9, 10), bonuses=(8,), draw_date=SimpleNamespace(day=2, month=1, year=2026)),
    ]
    state = auto_loop.SearchState(best_metric=0.83)
    state.best_strategy = "tf2.2_hc2.1_b3.0_sb0.9"

    monkeypatch.setattr(server, "_get_best_strategy", lambda lottery_id: {
        "name": state.best_strategy,
        "metric": state.best_metric,
        "tf": state.tf,
        "hc": state.hc,
        "b": state.bal,
        "sb": state.sb,
        "delay": state.delay,
        "momentum": state.momentum,
        "iw": state.iw,
        "rw": state.rw,
        "extra_strategies": list(state.extra_strategies),
    })
    monkeypatch.setattr(auto_loop, "load_search_state_for_lottery", lambda lottery_id: (state, None))
    monkeypatch.setattr(auto_loop, "load_best_from_results_for_lottery", lambda lottery_id: state)
    monkeypatch.setattr(
        auto_loop,
        "rank_numbers_from_draws",
        lambda _draws, _state, lottery_id=None: list(range(70, 0, -1)),
    )
    monkeypatch.setattr(server, "load_lottery_draws", lambda lottery_id: draws)

    call_index = {"value": 0}

    def fake_choices(population, weights=None, k=1):
        idx = call_index["value"] % len(population)
        call_index["value"] += 1
        return [population[idx]]

    monkeypatch.setattr(server.random, "choices", fake_choices)

    client = TestClient(server.app)
    first = client.get("/api/mega_millions_us/generate?ticket_count=1&nums_per_ticket=5&bonus_per_ticket=1")
    second = client.get("/api/mega_millions_us/generate?ticket_count=1&nums_per_ticket=5&bonus_per_ticket=1")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["tickets"][0]["bonus"] != second.json()["tickets"][0]["bonus"]


def test_read_progress_marks_stalled_when_pid_alive_but_file_is_stale(tmp_path, monkeypatch):
    progress_file = tmp_path / "progress.json"
    progress_file.write_text(
        json.dumps({"status": "running", "pid": 1234, "updated_at": "2026-03-27T21:14:35"}),
        encoding="utf-8",
    )
    old_ts = 1_700_000_000
    os.utime(progress_file, (old_ts, old_ts))
    monkeypatch.setattr(server, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(server.time, "time", lambda: old_ts + server.STALE_PROGRESS_SECONDS + 5)

    data = server._read_progress(SimpleNamespace(progress_file=progress_file))

    assert data["status"] == "stalled"
    assert data["pid"] == 1234
    assert "alive" in data["error"]


def test_read_progress_uses_recovery_file_timestamp_when_primary_is_stale(tmp_path, monkeypatch):
    progress_file = tmp_path / "progress.json"
    recovery_file = server.auto_loop.recovery_path(progress_file)
    progress_file.write_text(
        json.dumps({"status": "running", "pid": 1234, "updated_at": "2026-03-27T21:14:35"}),
        encoding="utf-8",
    )
    recovery_file.write_text(
        json.dumps({"status": "running", "pid": 1234, "updated_at": "2026-03-29T06:47:14"}),
        encoding="utf-8",
    )
    old_ts = 1_700_000_000
    fresh_ts = old_ts + server.STALE_PROGRESS_SECONDS - 30
    os.utime(progress_file, (old_ts, old_ts))
    os.utime(recovery_file, (fresh_ts, fresh_ts))
    monkeypatch.setattr(server, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(server.time, "time", lambda: old_ts + server.STALE_PROGRESS_SECONDS + 5)

    data = server._read_progress(SimpleNamespace(progress_file=progress_file))

    assert data["status"] == "running"
    assert "error" not in data


def test_read_progress_synthesizes_running_status_from_live_loop_lock(tmp_path, monkeypatch):
    progress_file = tmp_path / "progress.json"
    loop_state_file = tmp_path / "autoloop_state.json"
    progress_file.write_text(
        json.dumps({"status": "finished", "result": {"metric": 1.23}}),
        encoding="utf-8",
    )
    state = auto_loop.SearchState(best_metric=1.23, best_strategy="tf3_hc2_b4_sb1", search_phase="phase2")
    loop_state_file.write_text(
        json.dumps({"version": 1, "search_state": state.to_dict()}),
        encoding="utf-8",
    )
    config = SimpleNamespace(progress_file=progress_file, loop_state_file=loop_state_file, lock_file=tmp_path / ".experiment.lock")

    monkeypatch.setattr(server, "_lock_file_has_alive_pid", lambda _config: 4321)

    data = server._read_progress(config)

    assert data["status"] == "running"
    assert data["pid"] == 4321
    assert data["waiting_for_next_iteration"] is True
    assert data["loop_info"]["search_phase"] == "phase2"


def test_read_progress_synthesizes_live_auto_loop_progress_without_lock(tmp_path, monkeypatch):
    progress_file = tmp_path / "progress.json"
    loop_state_file = tmp_path / "autoloop_state.json"
    progress_file.write_text(
        json.dumps({"status": "running", "pid": 9876, "experiment_id": "auto_iter16"}),
        encoding="utf-8",
    )
    state = auto_loop.SearchState(best_metric=1.23, best_strategy="tf3_hc2_b4_sb1", search_phase="phase2")
    loop_state_file.write_text(
        json.dumps({"version": 1, "search_state": state.to_dict()}),
        encoding="utf-8",
    )
    config = SimpleNamespace(progress_file=progress_file, loop_state_file=loop_state_file, lock_file=tmp_path / ".experiment.lock")

    monkeypatch.setattr(server, "_lock_file_has_alive_pid", lambda _config: None)
    monkeypatch.setattr(server, "_pid_alive", lambda pid: True)

    data = server._read_progress(config)

    assert data["status"] == "running"
    assert data["pid"] == 9876
    assert data["waiting_for_next_iteration"] is True
    assert data["loop_info"]["search_phase"] == "phase2"


def test_read_progress_uses_fresh_loop_state_timestamp_for_live_auto_loop(tmp_path, monkeypatch):
    progress_file = tmp_path / "progress.json"
    loop_state_file = tmp_path / "autoloop_state.json"
    progress_file.write_text(
        json.dumps({"status": "running", "pid": 9876, "experiment_id": "auto_iter16"}),
        encoding="utf-8",
    )
    state = auto_loop.SearchState(best_metric=1.23, best_strategy="tf3_hc2_b4_sb1", search_phase="phase2")
    loop_state_file.write_text(
        json.dumps({"version": 1, "search_state": state.to_dict()}),
        encoding="utf-8",
    )
    old_ts = 1_700_000_000
    fresh_ts = old_ts + server.STALE_PROGRESS_SECONDS - 30
    os.utime(progress_file, (old_ts, old_ts))
    os.utime(loop_state_file, (fresh_ts, fresh_ts))
    config = SimpleNamespace(progress_file=progress_file, loop_state_file=loop_state_file, lock_file=tmp_path / ".experiment.lock")

    monkeypatch.setattr(server, "_lock_file_has_alive_pid", lambda _config: None)
    monkeypatch.setattr(server, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(server.time, "time", lambda: old_ts + server.STALE_PROGRESS_SECONDS + 5)

    data = server._read_progress(config)

    assert data["status"] == "running"
    assert "error" not in data
    assert data["pid"] == 9876
    assert data["waiting_for_next_iteration"] is True


def test_progress_snapshot_marks_stalled_external_source(tmp_path, monkeypatch):
    progress_file = tmp_path / "progress.json"
    progress_file.write_text(
        json.dumps({"status": "running", "pid": 1234, "updated_at": "2026-03-27T21:14:35"}),
        encoding="utf-8",
    )
    old_ts = 1_700_000_000
    os.utime(progress_file, (old_ts, old_ts))
    monkeypatch.setattr(server, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(server, "_cleanup_processes", lambda: None)
    monkeypatch.setattr(server, "_run_processes", {})
    monkeypatch.setattr(server, "_loop_processes", {})
    monkeypatch.setattr(server.time, "time", lambda: old_ts + server.STALE_PROGRESS_SECONDS + 5)

    data, signature = server._progress_snapshot("multi_multi_pl", SimpleNamespace(progress_file=progress_file))

    assert data["status"] == "stalled"
    assert data["source"] == "external"
    assert signature[2] == "stalled"
    assert signature[3] == "external"


def test_loop_status_uses_lock_file_when_server_did_not_spawn_process(monkeypatch, tmp_path):
    config = SimpleNamespace(
        lock_file=tmp_path / ".experiment.lock",
        progress_file=tmp_path / "progress.json",
        loop_state_file=tmp_path / "autoloop_state.json",
    )
    monkeypatch.setattr(server, "_cleanup_processes", lambda: None)
    monkeypatch.setattr(server, "_loop_processes", {})
    monkeypatch.setattr(server, "_config", lambda lottery_id: config)
    monkeypatch.setattr(server, "_lock_file_has_alive_pid", lambda _config: 2468)
    monkeypatch.setattr(server, "_read_progress", lambda _config: {"status": "idle"})

    payload = asyncio.run(server.loop_status("eurojackpot"))

    assert payload == {"running": True, "pid": 2468}


def test_loop_status_uses_progress_backed_auto_loop_when_lock_is_missing(monkeypatch, tmp_path):
    config = SimpleNamespace(
        lock_file=tmp_path / ".experiment.lock",
        progress_file=tmp_path / "progress.json",
        loop_state_file=tmp_path / "autoloop_state.json",
    )
    monkeypatch.setattr(server, "_cleanup_processes", lambda: None)
    monkeypatch.setattr(server, "_loop_processes", {})
    monkeypatch.setattr(server, "_config", lambda lottery_id: config)
    monkeypatch.setattr(server, "_lock_file_has_alive_pid", lambda _config: None)
    monkeypatch.setattr(server, "_read_progress", lambda _config: {"status": "running", "pid": 1357, "experiment_id": "auto_iter7"})
    monkeypatch.setattr(server, "_progress_auto_loop_pid", lambda _config, data: 1357)

    payload = asyncio.run(server.loop_status("eurojackpot"))

    assert payload == {"running": True, "pid": 1357}


def test_loop_status_prefers_lock_backed_pid_over_server_tracked_process(monkeypatch, tmp_path):
    class DummyProc:
        pid = 1111

        @staticmethod
        def poll():
            return None

    config = SimpleNamespace(
        lock_file=tmp_path / ".experiment.lock",
        progress_file=tmp_path / "progress.json",
        loop_state_file=tmp_path / "autoloop_state.json",
    )
    monkeypatch.setattr(server, "_cleanup_processes", lambda: None)
    monkeypatch.setattr(server, "_loop_processes", {"eurojackpot": DummyProc()})
    monkeypatch.setattr(server, "_config", lambda lottery_id: config)
    monkeypatch.setattr(server, "_lock_file_has_alive_pid", lambda _config: 2468)
    monkeypatch.setattr(server, "_read_progress", lambda _config: {"status": "running", "pid": 1357, "experiment_id": "auto_iter7"})
    monkeypatch.setattr(server, "_progress_auto_loop_pid", lambda _config, data: 1357)

    payload = asyncio.run(server.loop_status("eurojackpot"))

    assert payload == {"running": True, "pid": 2468}


def test_loop_status_uses_starting_progress_pid(monkeypatch, tmp_path):
    config = SimpleNamespace(
        lock_file=tmp_path / ".experiment.lock",
        progress_file=tmp_path / "progress.json",
        loop_state_file=tmp_path / "autoloop_state.json",
    )
    monkeypatch.setattr(server, "_cleanup_processes", lambda: None)
    monkeypatch.setattr(server, "_loop_processes", {})
    monkeypatch.setattr(server, "_config", lambda lottery_id: config)
    monkeypatch.setattr(server, "_read_progress", lambda _config: {"status": "starting", "pid": 7777})
    monkeypatch.setattr(server, "_lock_file_has_alive_pid", lambda _config: None)
    monkeypatch.setattr(server, "_pid_alive", lambda pid: True)

    payload = asyncio.run(server.loop_status("eurojackpot"))

    assert payload == {"running": True, "pid": 7777}


def test_progress_snapshot_prefers_recovery_file_signature(tmp_path, monkeypatch):
    progress_file = tmp_path / "progress.json"
    recovery_file = server.auto_loop.recovery_path(progress_file)
    progress_file.write_text(
        json.dumps({"status": "running", "pid": 1234, "updated_at": "2026-03-27T21:14:35"}),
        encoding="utf-8",
    )
    recovery_file.write_text(
        json.dumps({"status": "running", "pid": 1234, "updated_at": "2026-03-29T06:47:14"}),
        encoding="utf-8",
    )
    old_ts = 1_700_000_000
    fresh_ts = old_ts + server.STALE_PROGRESS_SECONDS - 30
    os.utime(progress_file, (old_ts, old_ts))
    os.utime(recovery_file, (fresh_ts, fresh_ts))
    monkeypatch.setattr(server, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(server, "_cleanup_processes", lambda: None)
    monkeypatch.setattr(server.time, "time", lambda: old_ts + server.STALE_PROGRESS_SECONDS + 5)

    data, signature = server._progress_snapshot("multi_multi_pl", SimpleNamespace(progress_file=progress_file))

    assert data["status"] == "running"
    assert signature[0] == str(recovery_file)


def test_load_ticket_history_reads_recovery_file(tmp_path, monkeypatch):
    history_file = tmp_path / "ticket_history.json"
    recovery_file = server.auto_loop.recovery_path(history_file)
    history_file.write_text("{invalid", encoding="utf-8")
    recovery_file.write_text(json.dumps({"version": 1, "entries": [{"id": "abc", "main": [1, 2, 3]}]}), encoding="utf-8")
    monkeypatch.setattr(server, "ensure_lottery_storage", lambda _config: None)

    entries = server._load_ticket_history(SimpleNamespace(history_file=history_file))

    assert entries == [{"id": "abc", "main": [1, 2, 3]}]


def test_spawn_background_process_writes_to_logs_not_pipes(tmp_path, monkeypatch):
    config = SimpleNamespace(state_dir=tmp_path)
    captured = {}

    class DummyProcess:
        pid = 999

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return DummyProcess()

    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)

    proc = server._spawn_background_process(config, ["python", "auto_loop.py"], "loop")

    assert proc.pid == 999
    assert captured["kwargs"]["stdout"] is not subprocess.PIPE
    assert captured["kwargs"]["stderr"] is not subprocess.PIPE
    assert (tmp_path / "loop.out.log").exists()
    assert (tmp_path / "loop.err.log").exists()


def test_stop_endpoint_can_terminate_external_stalled_process(monkeypatch, tmp_path):
    config = SimpleNamespace(
        progress_file=tmp_path / "progress.json",
        stop_request_file=tmp_path / ".autoloop.stop",
        state_dir=tmp_path,
    )
    config.progress_file.write_text("{}", encoding="utf-8")
    client = TestClient(server.app)
    called = {"terminate": None}

    monkeypatch.setattr(server, "_config", lambda lottery_id: config)
    monkeypatch.setattr(server, "_cleanup_processes", lambda: None)
    monkeypatch.setattr(server, "_run_processes", {})
    monkeypatch.setattr(server, "_read_progress", lambda _config: {"status": "stalled", "pid": 4321})
    monkeypatch.setattr(server, "_pid_alive", lambda pid: True)

    async def fake_wait_for_pid_exit(pid, timeout_s):
        return False

    monkeypatch.setattr(server, "_wait_for_pid_exit", fake_wait_for_pid_exit)
    monkeypatch.setattr(server, "_terminate_pid", lambda pid: called.__setitem__("terminate", pid) or True)
    monkeypatch.setattr(server, "_write_stopped_progress", lambda _config, pid: called.__setitem__("stopped", pid))

    response = client.post("/api/stop")

    assert response.status_code == 200
    assert response.json()["pid"] == 4321
    assert called["terminate"] == 4321
    assert called["stopped"] == 4321
    assert config.stop_request_file.exists()


def test_stop_loop_can_terminate_progress_backed_auto_loop_when_lock_is_missing(monkeypatch, tmp_path):
    config = SimpleNamespace(
        lock_file=tmp_path / ".experiment.lock",
        progress_file=tmp_path / "progress.json",
        loop_state_file=tmp_path / "autoloop_state.json",
        stop_request_file=tmp_path / ".autoloop.stop",
        state_dir=tmp_path,
    )
    client = TestClient(server.app)
    called = {"terminate": None, "stopped": None}

    monkeypatch.setattr(server, "_config", lambda lottery_id: config)
    monkeypatch.setattr(server, "_cleanup_processes", lambda: None)
    monkeypatch.setattr(server, "_loop_processes", {})
    monkeypatch.setattr(server, "_lock_file_has_alive_pid", lambda _config: None)
    monkeypatch.setattr(server, "_read_progress", lambda _config: {"status": "running", "pid": 5555, "experiment_id": "auto_iter12"})
    monkeypatch.setattr(server, "_progress_auto_loop_pid", lambda _config, data: 5555)

    async def fake_wait_for_pid_exit(pid, timeout_s):
        return False

    monkeypatch.setattr(server, "_wait_for_pid_exit", fake_wait_for_pid_exit)
    monkeypatch.setattr(server, "_terminate_pid", lambda pid: called.__setitem__("terminate", pid) or True)
    monkeypatch.setattr(server, "_write_stopped_progress", lambda _config, pid: called.__setitem__("stopped", pid))

    response = client.post("/api/eurojackpot/loop/stop")

    assert response.status_code == 200
    assert response.json()["pid"] == 5555
    assert called["terminate"] == 5555
    assert called["stopped"] == 5555
    assert config.stop_request_file.exists()


def test_stop_loop_terminates_all_detected_loop_pids(monkeypatch, tmp_path):
    config = SimpleNamespace(
        lock_file=tmp_path / ".experiment.lock",
        progress_file=tmp_path / "progress.json",
        loop_state_file=tmp_path / "autoloop_state.json",
        stop_request_file=tmp_path / ".autoloop.stop",
        state_dir=tmp_path,
    )
    client = TestClient(server.app)
    terminated: list[int] = []
    stopped = {"pid": None}

    monkeypatch.setattr(server, "_config", lambda lottery_id: config)
    monkeypatch.setattr(server, "_cleanup_processes", lambda: None)
    monkeypatch.setattr(server, "_loop_processes", {})
    monkeypatch.setattr(server, "_read_progress", lambda _config: {"status": "running", "pid": 5555, "experiment_id": "auto_iter12"})
    monkeypatch.setattr(server, "_active_loop_pids", lambda lottery_id, config=None, progress=None: [5555, 6666])

    async def fake_wait_for_pid_exit(pid, timeout_s):
        return False

    monkeypatch.setattr(server, "_wait_for_pid_exit", fake_wait_for_pid_exit)
    monkeypatch.setattr(server, "_terminate_pid", lambda pid: terminated.append(pid) or True)
    monkeypatch.setattr(server, "_write_stopped_progress", lambda _config, pid: stopped.__setitem__("pid", pid))

    response = client.post("/api/eurojackpot/loop/stop")

    assert response.status_code == 200
    assert response.json()["pid"] == 5555
    assert terminated == [5555, 6666]
    assert stopped["pid"] == 5555
    assert config.stop_request_file.exists()


def test_any_process_running_blocks_starting_progress(monkeypatch, tmp_path):
    config = SimpleNamespace(progress_file=tmp_path / "progress.json")
    monkeypatch.setattr(server, "_cleanup_processes", lambda: None)
    monkeypatch.setattr(server, "_config", lambda lottery_id: config)
    monkeypatch.setattr(server, "_read_progress", lambda _config: {"status": "starting", "pid": 6789})
    monkeypatch.setattr(server, "_canonical_loop_pid", lambda lottery_id, config=None, progress=None: 6789)

    message = server._any_process_running("eurojackpot")

    assert message == "Auto-loop already starting for eurojackpot (PID: 6789)"


def test_start_loop_writes_starting_progress(monkeypatch, tmp_path):
    config = SimpleNamespace(id="eurojackpot", state_dir=tmp_path, progress_file=tmp_path / "progress.json")
    client = TestClient(server.app)
    captured = {"pid": None}

    class DummyProcess:
        pid = 4242

    monkeypatch.setattr(server, "_config", lambda lottery_id: config)
    monkeypatch.setattr(server, "_any_process_running", lambda lottery_id: None)
    monkeypatch.setattr(server, "_spawn_background_process", lambda config, args, log_prefix: DummyProcess())
    monkeypatch.setattr(server, "_write_starting_progress", lambda _config, pid: captured.__setitem__("pid", pid))
    monkeypatch.setattr(server, "_loop_processes", {})

    response = client.post("/api/eurojackpot/loop/start")

    assert response.status_code == 200
    assert response.json()["pid"] == 4242
    assert captured["pid"] == 4242


def test_write_stopped_progress_uses_resilient_json_write(monkeypatch, tmp_path):
    config = SimpleNamespace(progress_file=tmp_path / "progress.json")
    captured = {}

    def fake_write_json_resilient(path, data, ensure_ascii=True):
        captured["path"] = path
        captured["data"] = data
        captured["ensure_ascii"] = ensure_ascii

    monkeypatch.setattr(server, "write_json_resilient", fake_write_json_resilient)
    monkeypatch.setattr(server.time, "strftime", lambda fmt: "2026-03-28T12:34:56")

    server._write_stopped_progress(config, 4321)

    assert captured["path"] == config.progress_file
    assert captured["ensure_ascii"] is True
    assert captured["data"] == {
        "status": "stopped",
        "stopped_pid": 4321,
        "updated_at": "2026-03-28T12:34:56",
    }


def test_write_ticket_history_uses_atomic_json_write(monkeypatch, tmp_path):
    config = SimpleNamespace(history_file=tmp_path / "ticket_history.json")
    captured = {}

    def fake_write_json_atomic(path, data, ensure_ascii=True):
        captured["path"] = path
        captured["data"] = data
        captured["ensure_ascii"] = ensure_ascii

    monkeypatch.setattr(server, "ensure_lottery_storage", lambda _config: None)
    monkeypatch.setattr(server, "write_json_atomic", fake_write_json_atomic)

    entries = [{"ticket_id": "abc", "main": [1, 2, 3, 4, 5, 6]}]
    server._write_ticket_history(config, entries)

    assert captured["path"] == config.history_file
    assert captured["ensure_ascii"] is True
    assert captured["data"] == {"version": 1, "entries": entries}


def test_history_add_list_delete_ticket(monkeypatch, tmp_path):
    config = SimpleNamespace(
        id="lotto_pl",
        history_file=tmp_path / "ticket_history.json",
        min_ticket_numbers=6,
        max_ticket_numbers=12,
        number_range=49,
        bonus_count=0,
        bonus_range=None,
        min_bonus_numbers=0,
        max_bonus_numbers=0,
        next_draw_date=lambda today=None: date(2026, 3, 28),
    )
    client = TestClient(server.app)
    draw = SimpleNamespace(draw_date=date(2026, 3, 28), numbers=(1, 2, 3, 4, 5, 6), bonuses=())

    monkeypatch.setattr(server, "_config", lambda lottery_id: config)
    monkeypatch.setattr(server, "_today_iso", lambda: "2026-03-28")
    monkeypatch.setattr(server, "_now_local", lambda: SimpleNamespace(isoformat=lambda timespec="seconds": "2026-03-28T12:00:00+02:00"))
    monkeypatch.setattr(server, "load_lottery_draws", lambda lottery_id: [draw])
    monkeypatch.setattr(load_data, "load_specific_lottery_draws", lambda lottery_id, target_dates=None, recent_days=30: [draw])
    monkeypatch.setattr(server, "ensure_lottery_storage", lambda _config: None)

    # AI tickets come from the real data/ dir — keep the test isolated
    monkeypatch.setattr(server, "_ai_history_entries", lambda lottery_id: [])
    # Clear LRU cache so stale data from other tests doesn't leak
    server._history_payload_cached.cache_clear()

    save = client.post("/api/lotto_pl/history", json={"main": [1, 2, 3, 4, 5, 6], "bonus": [], "strategy": "best"})
    assert save.status_code == 200
    ticket_id = save.json()["ticket_id"]

    history = client.get("/api/lotto_pl/history")
    assert history.status_code == 200
    payload = history.json()
    assert payload["today_count"] == 1
    assert payload["groups"][0]["winning_ticket_count"] == 1
    assert payload["groups"][0]["tickets"][0]["main_hits"] == 6
    assert payload["groups"][0]["tickets"][0]["won"] is True

    deleted = client.delete(f"/api/lotto_pl/history/{ticket_id}")
    assert deleted.status_code == 200
    assert client.get("/api/lotto_pl/history").json()["groups"] == []


def test_history_enforces_daily_limit(monkeypatch, tmp_path):
    config = SimpleNamespace(
        id="multi_multi_pl",
        history_file=tmp_path / "ticket_history.json",
        min_ticket_numbers=1,
        max_ticket_numbers=10,
        number_range=80,
        bonus_count=0,
        bonus_range=None,
        min_bonus_numbers=0,
        max_bonus_numbers=0,
        next_draw_date=lambda today=None: date(2026, 3, 28),
    )
    entries = []
    for idx in range(server.MAX_HISTORY_TICKETS_PER_DAY):
        entries.append(
            {
                "id": f"t{idx}",
                "saved_at": f"2026-03-28T12:00:{idx:02d}+02:00",
                "saved_date": "2026-03-28",
                "main": [idx + 1],
                "bonus": [],
            }
        )
    config.history_file.write_text(json.dumps({"version": 1, "entries": entries}), encoding="utf-8")

    client = TestClient(server.app)
    monkeypatch.setattr(server, "_config", lambda lottery_id: config)
    monkeypatch.setattr(server, "_today_iso", lambda: "2026-03-28")
    monkeypatch.setattr(server, "ensure_lottery_storage", lambda _config: None)

    response = client.post("/api/multi_multi_pl/history", json={"main": [77], "bonus": []})

    assert response.status_code == 409
    assert "20 tickets" in response.json()["error"]


def test_history_separates_multi_multi_draws_on_same_day(monkeypatch, tmp_path):
    config = SimpleNamespace(
        id="multi_multi_pl",
        history_file=tmp_path / "ticket_history.json",
        min_ticket_numbers=1,
        max_ticket_numbers=10,
        number_range=80,
        bonus_count=0,
        bonus_range=None,
        min_bonus_numbers=0,
        max_bonus_numbers=0,
    )
    entries = [
        {
            "id": "a",
            "saved_at": "2026-03-28T10:00:00+02:00",
            "saved_date": "2026-03-28",
            "target_draw_date": "2026-03-28",
            "target_draw_id": "16684",
            "target_draw_time": None,
            "main": [3, 4, 8, 12],
            "bonus": [],
            "strategy": "iw40_best",
        },
        {
            "id": "b",
            "saved_at": "2026-03-28T18:00:00+02:00",
            "saved_date": "2026-03-28",
            "target_draw_date": "2026-03-28",
            "target_draw_id": "16685",
            "target_draw_time": None,
            "main": [1, 2, 3, 4],
            "bonus": [],
            "strategy": "iw40_best",
        },
    ]
    config.history_file.write_text(json.dumps({"version": 1, "entries": entries}), encoding="utf-8")
    draws = [
        SimpleNamespace(draw_date=date(2026, 3, 28), draw_identifier="16684", draw_time=None, numbers=(3, 4, 8, 12, 13), bonuses=()),
        SimpleNamespace(draw_date=date(2026, 3, 28), draw_identifier="16685", draw_time=None, numbers=(1, 2, 9, 10, 11), bonuses=()),
    ]
    client = TestClient(server.app)

    monkeypatch.setattr(server, "_config", lambda lottery_id: config)
    monkeypatch.setattr(server, "_today_iso", lambda: "2026-03-28")
    monkeypatch.setattr(server, "load_lottery_draws", lambda lottery_id: draws)
    monkeypatch.setattr(load_data, "load_specific_lottery_draws", lambda lottery_id, target_dates=None, recent_days=30: draws)
    monkeypatch.setattr(server, "ensure_lottery_storage", lambda _config: None)

    # AI tickets come from the real data/ dir — keep the test isolated
    monkeypatch.setattr(server, "_ai_history_entries", lambda lottery_id: [])
    # Clear LRU cache so stale data from other tests doesn't leak
    server._history_payload_cached.cache_clear()

    history = client.get("/api/multi_multi_pl/history")

    assert history.status_code == 200
    payload = history.json()
    assert len(payload["groups"]) == 2
    assert {group["target_draw_id"] for group in payload["groups"]} == {"16684", "16685"}
    first_group = next(group for group in payload["groups"] if group["target_draw_id"] == "16684")
    second_group = next(group for group in payload["groups"] if group["target_draw_id"] == "16685")
    assert first_group["tickets"][0]["main_hits"] == 4
    assert first_group["tickets"][0]["won"] is True
    assert second_group["tickets"][0]["main_hits"] == 2
    assert second_group["tickets"][0]["won"] is True


def test_acquire_singleton_lock_rejects_running_server(tmp_path, monkeypatch):
    lock_file = tmp_path / ".server.lock"
    lock_file.write_text("9876", encoding="utf-8")
    monkeypatch.setattr(server, "_pid_alive", lambda pid: pid == 9876)

    assert server._acquire_singleton_lock(lock_file, "server") is False


def test_acquire_singleton_lock_replaces_stale_server_lock(tmp_path, monkeypatch):
    lock_file = tmp_path / ".server.lock"
    lock_file.write_text("9876", encoding="utf-8")
    monkeypatch.setattr(server, "_pid_alive", lambda pid: False)

    assert server._acquire_singleton_lock(lock_file, "server") is True
    owner = read_lock_owner(lock_file)
    assert owner.pid == os.getpid()
    assert owner.kind == "server"


def test_release_singleton_lock_keeps_foreign_owner(tmp_path):
    lock_file = tmp_path / ".server.lock"
    lock_file.write_text("9876", encoding="utf-8")

    server._release_singleton_lock(lock_file)

    assert lock_file.exists()
    assert lock_file.read_text(encoding="utf-8").strip() == "9876"


def test_acquire_singleton_lock_replaces_live_foreign_process_lock(tmp_path, monkeypatch):
    lock_file = tmp_path / ".server.lock"
    lock_file.write_text("9876", encoding="utf-8")
    monkeypatch.setattr(server, "_pid_alive", lambda pid: pid == 9876)
    monkeypatch.setattr(server, "lock_matches_process", lambda owner, **kwargs: False)

    assert server._acquire_singleton_lock(lock_file, "server") is True
    assert read_lock_owner(lock_file).pid == os.getpid()


def test_lock_file_has_alive_pid_ignores_foreign_process(monkeypatch, tmp_path):
    config = SimpleNamespace(lock_file=tmp_path / ".experiment.lock")
    config.lock_file.write_text("1234", encoding="utf-8")
    monkeypatch.setattr(server, "_pid_alive", lambda pid: pid == 1234)
    monkeypatch.setattr(server, "lock_matches_process", lambda owner, **kwargs: False)

    assert server._lock_file_has_alive_pid(config) is None


def test_agent_start_stop_routes_are_unique():
    counts = Counter()
    for route in server.app.routes:
        methods = getattr(route, "methods", set()) or set()
        path = getattr(route, "path", None)
        for method in methods:
            counts[(method, path)] += 1

    assert counts[("POST", "/api/agents/{agent_id}/start")] == 1
    assert counts[("POST", "/api/agents/{agent_id}/stop")] == 1


def test_start_agent_supports_experiment_monitor(monkeypatch, tmp_path):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / "experiment_monitor.py").write_text("print('ok')\n", encoding="utf-8")
    monkeypatch.setattr(server, "AGENTS_DIR", agents_dir)
    monkeypatch.setattr(server.time, "sleep", lambda _seconds: None)

    states = [
        {"running": False},
        {"running": True, "pid": 4321},
    ]
    monkeypatch.setattr(server, "_get_agent_status", lambda _agent_id: states.pop(0))

    class DummyProc:
        pid = 4321

    monkeypatch.setattr(server.subprocess, "Popen", lambda *args, **kwargs: DummyProc())

    payload = asyncio.run(server.start_agent("experiment_monitor"))

    assert payload["status"] == "started"
    assert payload["agent"] == "experiment_monitor"
    assert payload["pid"] == 4321


def test_stop_agent_force_writes_command_and_terminates(monkeypatch, tmp_path):
    agents_dir = tmp_path / "agents"
    commands_dir = agents_dir / "messages" / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(server, "AGENTS_DIR", agents_dir)
    monkeypatch.setattr(server, "_get_agent_status", lambda _agent_id: {"running": True, "pid": 2468})

    terminated = {"pid": None}
    monkeypatch.setattr(server, "_terminate_pid", lambda pid: terminated.__setitem__("pid", pid) or True)

    payload = asyncio.run(server.stop_agent("experiment_monitor", force=True))
    command_file = commands_dir / "to_experiment_monitor.json"
    messages = json.loads(command_file.read_text(encoding="utf-8"))

    assert payload["status"] == "stop_requested"
    assert payload["agent"] == "experiment_monitor"
    assert payload["force_terminated"] is True
    assert terminated["pid"] == 2468
    assert messages[-1]["to"] == "experiment_monitor"
    assert messages[-1]["type"] == "stop_request"


def test_history_cache_per_lottery_invalidation():
    """Per-lottery invalidation should not clear other lotteries' caches."""
    from server import _invalidate_history_cache, _is_history_cache_valid, _clear_history_cache_flag

    # Initially all lotteries are valid
    assert _is_history_cache_valid("multi_multi_pl") is True
    assert _is_history_cache_valid("lotto_pl") is True

    # Invalidate one lottery
    _invalidate_history_cache("multi_multi_pl")
    assert _is_history_cache_valid("multi_multi_pl") is False
    assert _is_history_cache_valid("lotto_pl") is True

    # Clear the flag
    _clear_history_cache_flag("multi_multi_pl")
    assert _is_history_cache_valid("multi_multi_pl") is True


def test_ttl_cache_basic():
    """TTL cache stores and expires entries."""
    from server import _TTLCache

    cache = _TTLCache()
    cache.set("key1", "value1", ttl=10)
    assert cache.get("key1") == "value1"
    assert cache.get("key2") is None

    # Expired entry
    cache._store["expired"] = (0, "old_value")
    assert cache.get("expired") is None


def test_generate_button_has_id():
    """The generate button must have id='btn-generate' for JS selector."""
    html = (server.PROJECT_DIR / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="btn-generate"' in html
