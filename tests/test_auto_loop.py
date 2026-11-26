from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import auto_loop
from lottery_config import LotteryConfig


@pytest.fixture(autouse=True)
def isolate_external_services(monkeypatch):
    import ai_actions

    monkeypatch.setattr(auto_loop, "LIVE_THOUGHTS_ENABLED", False)
    monkeypatch.setattr(ai_actions, "notify_event", lambda *a, **k: None)
    monkeypatch.setattr(auto_loop, "run_and_save_walkforward", lambda *a, **k: None)


def _write_results(path: Path, rows: list[str]) -> None:
    header = "commit\taverage_hits\tstatus\tstrategy\thit_rate_2_plus\thit_rate_3_plus\tportfolio_avg\telapsed_s\tdescription\n"
    path.write_text(header + "".join(rows), encoding="utf-8")


def test_load_best_from_results_replays_keep_history(tmp_path, monkeypatch):
    results_file = tmp_path / "results.tsv"
    _write_results(
        results_file,
        [
            "auto\t2.539376\tKEEP\ttf3.52_hc2.037_b4.57_sb1.45\t0\t0\t0\t1\titer4\n",
            "auto\t2.539571\tKEEP\tiw40_best\t0\t0\t0\t1\titer5\n",
            "auto\t2.539571\tDISCARD\tno_improvement\t0\t0\t0\t1\titer6\n",
        ],
    )
    monkeypatch.setattr(auto_loop, "RESULTS_FILE", results_file)

    state = auto_loop.load_best_from_results()

    assert state.best_metric == 2.539571
    assert state.best_strategy == "iw40_best"
    assert state.tf == 3.52
    assert state.hc == 2.037
    assert state.bal == 4.57
    assert state.sb == 1.45
    assert state.iw == 40
    assert state.rw == 200


def test_run_iteration_saves_checkpoint_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(auto_loop, "PROGRESS_FILE", tmp_path / "progress.json")
    monkeypatch.setattr(auto_loop, "RESULTS_FILE", tmp_path / "results.tsv")
    monkeypatch.setattr(auto_loop, "LOOP_STATE_FILE", tmp_path / "autoloop_state.json")
    config = LotteryConfig(
        id="test_multi_multi",
        name="Test Multi Multi",
        numbers_drawn=20,
        number_range=80,
        ticket_size=10,
        min_pick_count=1,
        max_pick_count=10,
    )
    monkeypatch.setattr(auto_loop, "ACTIVE_CONFIG", config)
    monkeypatch.setattr(type(config), "results_file", property(lambda self: tmp_path / "results.tsv"))

    state = auto_loop.SearchState(radius=0.1, step=0.2, extra_strategies=[])

    counter = {"count": 0}

    def fake_backtest(_config, name, rank_fn, draws, ticket_size, recent_window, initial_window):
        counter["count"] += 1
        metric = 1.0
        if name.startswith("tf2.9_hc1.9_b3.9"):
            metric = 5.0
        if name == "iw40_best":
            metric = 6.0
        return SimpleNamespace(
            average_hits=metric,
            hit_rate_2_plus=0.2,
            hit_rate_3_plus=0.1,
            portfolio_average_best_hits=metric + 1,
        )

    monkeypatch.setattr(auto_loop, "backtest_strategy", fake_backtest)
    monkeypatch.setattr(auto_loop, "stop_requested", lambda: counter["count"] >= 2)

    first = auto_loop.run_iteration(state, draws=[])

    assert first is None
    checkpoint = json.loads((tmp_path / "autoloop_state.json").read_text(encoding="utf-8"))
    active = checkpoint["active_iteration"]
    assert active["weight_index"] == 2
    assert active["phase"] == "search"
    assert active["search_phase"] == "phase1"

    monkeypatch.setattr(auto_loop, "stop_requested", lambda: False)
    second = auto_loop.run_iteration(state, draws=[], active_iteration=active)
    saved_state, active_after = auto_loop.load_search_state()

    assert second is not None
    assert active_after is None
    assert second["best_config"] == "iw40_best"
    assert saved_state.tf == 3.0
    assert saved_state.hc == 2.0
    assert saved_state.bal == 4.0
    assert saved_state.iw == 40
    assert saved_state.best_metric == 6.0
    assert saved_state.search_phase == "phase1"
    assert saved_state.extra_strategies == ["sb", "jc", "bf", "dd"]


def test_run_iteration_records_discard_with_best_tested_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(auto_loop, "PROGRESS_FILE", tmp_path / "progress.json")
    monkeypatch.setattr(auto_loop, "RESULTS_FILE", tmp_path / "results.tsv")
    monkeypatch.setattr(auto_loop, "LOOP_STATE_FILE", tmp_path / "autoloop_state.json")
    config = LotteryConfig(
        id="test_discard",
        name="Test Discard",
        numbers_drawn=20,
        number_range=80,
        ticket_size=10,
        min_pick_count=1,
        max_pick_count=10,
    )
    monkeypatch.setattr(auto_loop, "ACTIVE_CONFIG", config)
    monkeypatch.setattr(type(config), "results_file", property(lambda self: tmp_path / "results.tsv"))

    state = auto_loop.SearchState(
        best_metric=6.0,
        best_strategy="baseline_best",
        search_phase="phase2",
    )
    params = auto_loop._params_from_state(state)

    monkeypatch.setattr(
        auto_loop,
        "_build_weight_plan",
        lambda _state: [{"name": "candidate_cfg", "params": dict(params)}],
    )
    monkeypatch.setattr(auto_loop, "_build_phase_two_param_plan", lambda _state: [])
    monkeypatch.setattr(auto_loop, "_build_window_plan", lambda _state, _params: [])
    monkeypatch.setattr(auto_loop, "_build_rank_fn", lambda _state, _params: None)
    monkeypatch.setattr(auto_loop, "stop_requested", lambda: False)

    def fake_backtest(_config, name, rank_fn, draws, ticket_size, recent_window, initial_window):
        assert name == "candidate_cfg"
        return SimpleNamespace(
            average_hits=5.0,
            hit_rate_2_plus=0.2,
            hit_rate_3_plus=0.1,
            portfolio_average_best_hits=5.5,
        )

    monkeypatch.setattr(auto_loop, "backtest_strategy", fake_backtest)

    result = auto_loop.run_iteration(state, draws=[])

    assert result is not None
    assert result["best_config"] == "candidate_cfg"
    assert result["status"] == "DISCARD"
    assert result["improved"] is False
    lines = (tmp_path / "results.tsv").read_text(encoding="utf-8").splitlines()
    fields = lines[-1].split("\t")
    assert float(fields[1]) == 5.0
    assert fields[2] == "DISCARD"
    assert fields[3] == "candidate_cfg"
    assert "no improvement over 6.000000" in fields[8]


def test_run_iteration_uses_lottery_ticket_size_instead_of_hardcoded_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(auto_loop, "PROGRESS_FILE", tmp_path / "progress.json")
    monkeypatch.setattr(auto_loop, "RESULTS_FILE", tmp_path / "results.tsv")
    monkeypatch.setattr(auto_loop, "LOOP_STATE_FILE", tmp_path / "autoloop_state.json")
    config = LotteryConfig(
        id="test_rules",
        name="Test Rules",
        numbers_drawn=5,
        number_range=50,
        ticket_size=5,
        min_pick_count=5,
        max_pick_count=12,
    )
    monkeypatch.setattr(auto_loop, "ACTIVE_CONFIG", config)
    monkeypatch.setattr(type(config), "results_file", property(lambda self: tmp_path / "results.tsv"))

    state = auto_loop.SearchState(search_phase="phase1")
    params = auto_loop._params_from_state(state)

    monkeypatch.setattr(
        auto_loop,
        "_build_weight_plan",
        lambda _state: [{"name": "candidate_cfg", "params": dict(params)}],
    )
    monkeypatch.setattr(auto_loop, "_build_window_plan", lambda _state, _params: [])
    monkeypatch.setattr(auto_loop, "stop_requested", lambda: False)

    captured = {}

    def fake_backtest(_config, name, rank_fn, draws, ticket_size, recent_window, initial_window):
        captured["ticket_size"] = ticket_size
        return SimpleNamespace(
            average_hits=1.0,
            hit_rate_2_plus=0.2,
            hit_rate_3_plus=0.1,
            portfolio_average_best_hits=1.5,
        )

    monkeypatch.setattr(auto_loop, "backtest_strategy", fake_backtest)

    auto_loop.run_iteration(state, draws=[])

    assert captured["ticket_size"] == 5


def test_rank_numbers_from_draws_uses_state_windows(monkeypatch):
    draws = [
        SimpleNamespace(numbers=(1, 2, 3)),
        SimpleNamespace(numbers=(3, 4, 5)),
    ]
    state = auto_loop.SearchState(rw=5)
    captured = {}

    def fake_rank_fn(counts, recent_deque, last_seen, draw_index, recent_window):
        captured["recent_window"] = recent_window
        captured["draw_index"] = draw_index
        return list(range(1, auto_loop.ACTIVE_CONFIG.number_range + 1))

    monkeypatch.setattr(auto_loop, "build_rank_fn_from_state", lambda _state, **kwargs: fake_rank_fn)

    ranked = auto_loop.rank_numbers_from_draws(draws, state)

    assert ranked[:5] == [1, 2, 3, 4, 5]
    assert captured["recent_window"] == 5
    assert captured["draw_index"] == len(draws)


def test_phase_one_plan_searches_new_strategies_solo():
    state = auto_loop.SearchState(
        tf=3.5,
        hc=2.0,
        bal=4.5,
        sb=1.4,
        radius=0.1,
        step=0.1,
        extra_strategies=["sb", "jc", "bf", "dd"],
        search_phase="phase1",
    )

    plan = auto_loop._build_weight_plan(state)
    names = {spec["name"] for spec in plan}

    assert any("_jc0.5_" in name or name.endswith("_jc0.5") for name in names)
    assert any("_bf0.5_" in name or name.endswith("_bf0.5") for name in names)
    assert any("_dd0.5_" in name or name.endswith("_dd0.5") for name in names)
    assert any("_ps50.0" in name for name in names)
    assert any("_el0.05" in name for name in names)


def test_phase_three_plan_tests_anticorrelation_toggle():
    state = auto_loop.SearchState(search_phase="phase3")

    plan = auto_loop._build_weight_plan(state)
    names = {spec["name"] for spec in plan}

    assert any("ac1" in name for name in names)
    assert any("acr0.7" in name for name in names)


def test_phase_two_plan_uses_local_axis_sweep_instead_of_cartesian_explosion():
    state = auto_loop.SearchState(
        tf=3.0,
        hc=2.0,
        bal=4.0,
        sb=1.0,
        jc=0.5,
        bf=0.5,
        dd=0.5,
        radius=0.1,
        step=0.1,
        extra_strategies=["sb", "jc", "bf", "dd"],
        search_phase="phase2",
    )

    plan = auto_loop._build_weight_plan(state)
    names = {spec["name"] for spec in plan}

    assert len(plan) == 15
    assert "tf3.0_hc2.0_b4.0_sb1.0_jc0.5_bf0.5_dd0.5_ps20.0_el0.02" in names
    assert "tf2.9_hc2.0_b4.0_sb1.0_jc0.5_bf0.5_dd0.5_ps20.0_el0.02" in names
    assert "tf3.0_hc2.0_b4.0_sb1.0_jc0.6_bf0.5_dd0.5_ps20.0_el0.02" in names


def test_window_plan_adapts_around_current_best_windows():
    state = auto_loop.SearchState(iw=40, rw=200, no_improve_count=2)
    params = auto_loop._params_from_state(state)

    plan = auto_loop._build_window_plan(state, params)
    names = {spec["name"] for spec in plan}

    assert "iw45_best" in names
    assert "iw60_best" in names
    assert "rw225_best" in names
    assert "rw350_best" in names


def test_update_state_from_config_parses_new_strategy_tokens():
    state = auto_loop.SearchState()

    auto_loop._update_state_from_config(
        state,
        "tf3.0_hc2.0_b4.0_jc1.5_bf0.5_dd0.7_ps50_el0.05_ac1_acr0.7",
    )

    assert state.jc == 1.5
    assert state.bf == 0.5
    assert state.dd == 0.7
    assert state.prior_strength == 50.0
    assert state.ewma_lambda == 0.05
    assert state.use_anticorrelation is True
    assert state.anticorrelation_ratio == 0.7


def test_upgrade_state_for_new_strategies_preserves_legacy_reference(tmp_path, monkeypatch):
    config = LotteryConfig(
        id="test_upgrade",
        name="Test Upgrade",
        numbers_drawn=20,
        number_range=80,
        ticket_size=10,
        min_pick_count=1,
        max_pick_count=10,
    )
    monkeypatch.setattr(type(config), "legacy_best_file", property(lambda self: tmp_path / "legacy_best_reference.json"))

    state = auto_loop.SearchState(
        sb=1.3,
        delay=2.0,
        extra_strategies=["sb", "delay"],
        best_metric=2.5,
        best_strategy="tf3.0_hc2.0_b4.0_sb1.3_delay2.0",
        search_phase="phase2",
    )

    upgraded, migrated = auto_loop._upgrade_state_for_new_strategies(config, state)

    assert migrated is True
    assert upgraded.extra_strategies == ["sb", "jc", "bf", "dd"]
    assert upgraded.search_phase == "phase1"
    assert upgraded.jc == 0.0
    reference = json.loads((tmp_path / "legacy_best_reference.json").read_text(encoding="utf-8"))
    assert reference["best_metric"] == 2.5
    assert reference["best_strategy"] == "tf3.0_hc2.0_b4.0_sb1.3_delay2.0"


def test_run_loop_does_not_mark_infinite_auto_loop_finished_between_iterations(monkeypatch):
    draws = [SimpleNamespace(draw_date="2026-01-01")]
    state = auto_loop.SearchState(best_metric=1.0, best_strategy="tf3_hc2_b4_sb1")
    finished_calls = []
    run_results = iter([
        {"metric": 1.1, "best_config": "tf3.1_hc2.0_b4.0_sb1.0", "total_tested": 15, "elapsed_seconds": 1.0},
        None,
    ])
    load_state_results = iter([
        (state, None),
        (state, None),
        (state, None),
    ])

    monkeypatch.setattr(auto_loop, "_acquire_lock", lambda: True)
    monkeypatch.setattr(auto_loop, "_release_lock", lambda: None)
    monkeypatch.setattr(auto_loop, "clear_stop_request", lambda: None)
    monkeypatch.setattr(auto_loop, "set_search_weights", lambda: None)
    monkeypatch.setattr(auto_loop, "reset_default_weights", lambda: None)
    monkeypatch.setattr(auto_loop, "load_lottery_draws", lambda lottery_id: draws)
    monkeypatch.setattr(auto_loop, "load_jackpot_data_with_source", lambda lottery_id, _draws: ({}, "proxy"))
    monkeypatch.setattr(auto_loop, "load_search_state", lambda: next(load_state_results))
    monkeypatch.setattr(auto_loop, "load_best_from_results", lambda: state)
    monkeypatch.setattr(auto_loop, "run_iteration", lambda _state, _draws, _active: next(run_results))
    monkeypatch.setattr(auto_loop, "_write_finished", lambda result: finished_calls.append(result))
    monkeypatch.setattr(auto_loop.time, "sleep", lambda seconds: None)

    auto_loop.run_loop(max_iterations=0)

    assert finished_calls == []


def test_run_loop_once_logs_discard_and_skips_final_sleep(monkeypatch, capsys):
    draws = [SimpleNamespace(draw_date="2026-01-01")]
    state = auto_loop.SearchState(best_metric=2.0, best_strategy="baseline_best")
    finished_calls = []
    sleep_calls = []
    result = {
        "metric": 1.9,
        "best_config": "candidate_cfg",
        "total_tested": 12,
        "elapsed_seconds": 1.0,
        "status": "DISCARD",
        "improved": False,
    }
    load_state_results = iter([
        (state, None),
        (state, None),
    ])

    monkeypatch.setattr(auto_loop, "_acquire_lock", lambda: True)
    monkeypatch.setattr(auto_loop, "_release_lock", lambda: None)
    monkeypatch.setattr(auto_loop, "clear_stop_request", lambda: None)
    monkeypatch.setattr(auto_loop, "set_search_weights", lambda: None)
    monkeypatch.setattr(auto_loop, "reset_default_weights", lambda: None)
    monkeypatch.setattr(auto_loop, "load_lottery_draws", lambda lottery_id: draws)
    monkeypatch.setattr(auto_loop, "load_jackpot_data_with_source", lambda lottery_id, _draws: ({}, "proxy"))
    monkeypatch.setattr(auto_loop, "load_search_state", lambda: next(load_state_results))
    monkeypatch.setattr(auto_loop, "load_best_from_results", lambda: state)
    monkeypatch.setattr(auto_loop, "run_iteration", lambda _state, _draws, _active: dict(result))
    monkeypatch.setattr(auto_loop, "_write_finished", lambda payload: finished_calls.append(payload))
    monkeypatch.setattr(auto_loop.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    auto_loop.run_loop(max_iterations=1)

    captured = capsys.readouterr()
    assert "(DISCARD) - candidate_cfg" in captured.out
    assert finished_calls == [result]
    assert sleep_calls == []
