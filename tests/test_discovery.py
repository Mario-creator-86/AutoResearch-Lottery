"""Detectorul de anomalii, consilierul de joc, ciclul de descoperire și autonomia."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
AGENTS_DIR = PROJECT_DIR / "agents"
for _p in (PROJECT_DIR, AGENTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bias_scan
import discovery_cycle
import play_advisor
import watchdog
from lottery_config import get_lottery_config


def _synth(N, K, n, seed, weights=None):
    rng = np.random.default_rng(seed)
    matrix = np.zeros((n, N), dtype=np.int8)
    p = None if weights is None else weights / weights.sum()
    for i in range(n):
        matrix[i, rng.choice(N, K, replace=False, p=p)] = 1
    return matrix


# ---------------------------------------------------------------- bias_scan

def test_uniform_draws_raise_no_alarm():
    verdicts = [bias_scan.scan_matrix(_synth(49, 6, 1500, seed))["verdict"] for seed in range(100, 112)]  # prag proiectat: ~1% alarme false
    assert "significant" not in verdicts


def test_raw_p_values_are_calibrated_under_the_null():
    # sub hazard pur fiecare test trebuie să dea p < 0.05 în ~5% din cazuri
    below = {name: 0 for name in bias_scan.TESTS}
    runs = 120
    for seed in range(runs):
        tests = bias_scan.scan_matrix(_synth(40, 6, 1000, 1000 + seed))["tests"]
        for name in below:
            below[name] += tests[name]["p_value"] < 0.05
    for name, count in below.items():
        assert count / runs < 0.15, (name, count)


def test_heavy_balls_are_detected_and_named():
    weights = np.ones(49)
    biased = [2, 11, 23, 37, 44]
    weights[biased] = 1.3
    result = bias_scan.scan_matrix(_synth(49, 6, 2500, 7, weights))
    assert result["verdict"] == "significant"
    assert result["exploitable"] is True
    found = {item["number"] - 1 for item in result["most_frequent"]}
    assert len(found & set(biased)) >= 4


def test_serial_dependence_is_detected():
    # fiecare extragere păstrează 2 numere din precedenta → testul „repeats"
    rng = np.random.default_rng(3)
    matrix = np.zeros((1500, 49), dtype=np.int8)
    previous = rng.choice(49, 6, replace=False)
    for i in range(1500):
        keep = rng.choice(previous, 2, replace=False)
        rest = rng.choice(np.setdiff1d(np.arange(49), keep), 4, replace=False)
        previous = np.concatenate([keep, rest])
        matrix[i, previous] = 1
    result = bias_scan.scan_matrix(matrix)
    assert result["tests"]["repeats"]["verdict"] == "significant"


def test_too_few_draws_is_insufficient():
    assert bias_scan.scan_matrix(_synth(49, 6, 50, 1))["verdict"] == "insufficient"


def test_family_correction_gets_stricter_with_more_lotteries():
    matrix = _synth(49, 6, 1500, 5)
    one = bias_scan.scan_matrix(matrix, n_lotteries=1)
    many = bias_scan.scan_matrix(matrix, n_lotteries=8)
    assert many["tests_in_family"] == 8 * one["tests_in_family"]
    assert many["p_adjusted"] >= one["p_adjusted"]


def test_draws_matrix_skips_malformed_draws():
    draws = [SimpleNamespace(numbers=(1, 2, 3)), SimpleNamespace(numbers=(1, 2)),
             SimpleNamespace(numbers=(1, 1, 3)), SimpleNamespace(numbers=(1, 2, 99))]
    assert bias_scan.draws_matrix(draws, 10, 3).tolist() == [[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]]


# ------------------------------------------------------------- play_advisor

def test_breakdown_matches_typical_expected_value():
    import prize_estimates as pe

    for lottery_id in pe.TIERS:
        config = get_lottery_config(lottery_id)
        rest, p_top = play_advisor.ev_breakdown(lottery_id, config)
        typical = play_advisor.SPLIT_TOP_TIER.get(lottery_id, {}).get("typical_top", pe.TIERS[lottery_id][0][3])
        assert abs(rest + p_top * typical - pe.expected_value(lottery_id, config)[0]) < 1e-6, lottery_id


def test_loto_5_40_jackpot_needs_first_five_drawn():
    # reportul (Cat. I) = 5 din primele 5 extrase: 1/C(40,5), nu 5 din toate 6
    _rest, p_top = play_advisor.ev_breakdown("loto_5_40_ro", get_lottery_config("loto_5_40_ro"))
    assert round(1 / p_top) == 658_008


def _jackpot_config(tmp_path, lottery_id, rows):
    real = get_lottery_config(lottery_id)
    path = tmp_path / "jackpot_history.csv"
    path.write_text("draw_date,tier1_winners,tier1_amount,is_rollover,total_pool\n" + "\n".join(rows) + "\n",
                    encoding="utf-8")
    return SimpleNamespace(
        name=real.name, number_range=real.number_range, numbers_drawn=real.numbers_drawn,
        ticket_size=real.ticket_size, bonus_count=real.bonus_count, bonus_range=real.bonus_range,
        jackpot_csv=path,
    )


def test_advice_follows_the_jackpot(tmp_path, monkeypatch):
    import lottery_config

    today = date(2026, 11, 10)
    real = get_lottery_config("loto_6_49_ro")
    old = _jackpot_config(tmp_path, "loto_6_49_ro", ["2026-03-29,0,9000000,true,9000000"])
    monkeypatch.setattr(lottery_config, "get_lottery_config", lambda _lid: old)
    base = play_advisor.advise("loto_6_49_ro", today=today)  # report vechi → necunoscut
    assert base["verdict"] == "unknown" and base["breakeven_jackpot"] > 10_000_000
    assert play_advisor.latest_jackpot(old, today) is None

    for amount, expected in ((1_000_000, "unfavorable"), (base["breakeven_jackpot"] * 0.9, "near"),
                             (base["breakeven_jackpot"] * 1.5, "favorable")):
        config = _jackpot_config(tmp_path, "loto_6_49_ro", [f"2026-11-08,0,{amount},true,{amount}"])
        monkeypatch.setattr(lottery_config, "get_lottery_config", lambda _lid, c=config: c)
        advice = play_advisor.advise("loto_6_49_ro", today=today)
        assert advice["verdict"] == expected, (amount, advice)

    # câștigat la ultima extragere → fondul repornește, nu pretindem că știm reportul
    won = _jackpot_config(tmp_path, "loto_6_49_ro", ["2026-11-08,1,9000000,false,9000000"])
    assert play_advisor.latest_jackpot(won, today) is None
    assert real.number_range == 49


def test_fixed_prize_games_have_no_better_moment():
    advice = play_advisor.advise("multi_multi_pl")
    assert advice["verdict"] == "fixed" and "breakeven_jackpot" not in advice
    assert 0 < advice["typical_return_pct"] < 100


# ---------------------------------------------------------- discovery_cycle

def test_alerts_fire_only_on_transitions():
    scan = {"lottery": "kaskada_pl", "name": "Kaskada", "verdict": "significant", "n_draws": 9000,
            "strongest_test": "frequency_recent", "verdict_text": "semnificativ", "exploitable": False,
            "tests": {"frequency_recent": {"p_adjusted": 0.001}}}
    advice = {"lottery": "joker_ro", "name": "Joker", "verdict": "near", "current_jackpot": 1e8,
              "currency": "RON", "verdict_text": "aproape de prag"}

    first = discovery_cycle.find_alerts({}, [scan], [advice])
    assert {a["kind"] for a in first} == {"anomaly", "jackpot"}

    previous = {"anomalies": [scan], "advice": [advice]}
    assert discovery_cycle.find_alerts(previous, [scan], [advice]) == []  # nimic nou → tăcere

    better = dict(advice, verdict="favorable")
    assert [a["kind"] for a in discovery_cycle.find_alerts(previous, [scan], [better])] == ["jackpot"]
    worse = dict(advice, verdict="unfavorable")
    assert discovery_cycle.find_alerts({"anomalies": [scan], "advice": [better]}, [scan], [worse]) == []


def test_summary_is_honest_when_nothing_is_found():
    scans = [{"lottery": "a", "name": "A", "verdict": "noise"}, {"lottery": "b", "name": "B", "verdict": "suggestive"}]
    advice = [{"lottery": "a", "name": "A", "verdict": "unfavorable", "current_return_pct": 44.7}]
    text = discovery_cycle.summary_text(scans, advice)
    assert "Nicio anomalie semnificativă" in text and "B" in text and "45%" in text


def test_run_cycle_saves_state_and_notifies_nothing_without_changes(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(discovery_cycle, "DATA_DIR", tmp_path)
    monkeypatch.setattr(bias_scan, "run_all", lambda save=True: {
        "results": [{"lottery": "joker_ro", "name": "Joker", "verdict": "noise", "n_draws": 2000}],
        "newly_significant": []})
    monkeypatch.setattr(play_advisor, "advise_all", lambda: [
        {"lottery": "joker_ro", "name": "Joker", "verdict": "unfavorable", "current_return_pct": 30.0}])
    import telegram_bot
    monkeypatch.setattr(telegram_bot, "notify", lambda text: sent.append(text) or True)

    state = discovery_cycle.run_cycle(update=False)
    assert state["alerts"] == [] and sent == []
    assert json.loads((tmp_path / "discovery.json").read_text(encoding="utf-8"))["summary"] == state["summary"]


# ------------------------------------------------------------------ autonomy

def test_watchdog_starts_missing_server_and_restarts_hung_one(monkeypatch):
    events = []
    monkeypatch.setattr(watchdog, "log", lambda message: None)
    monkeypatch.setattr(watchdog, "start_server", lambda port: events.append("start") or 4242)
    monkeypatch.setattr(watchdog, "stop_server", lambda: events.append("stop") or 3)

    monkeypatch.setattr(watchdog, "server_healthy", lambda port: True)
    assert watchdog.check_once(8020, 2, 0.0, now=1000.0) == (0, 0.0) and events == []

    monkeypatch.setattr(watchdog, "server_healthy", lambda port: False)
    monkeypatch.setattr(watchdog, "find_processes", lambda script: [])
    assert watchdog.check_once(8020, 0, 0.0, now=1000.0) == (0, 1000.0) and events == ["start"]
    # tocmai pornit: nu îl repornim cât încă se inițializează
    assert watchdog.check_once(8020, 0, 1000.0, now=1060.0) == (0, 1000.0) and events == ["start"]

    # proces viu dar fără răspuns: abia a treia verificare ratată îl repornește
    monkeypatch.setattr(watchdog, "find_processes", lambda script: [object()])
    failures, last_start = 0, 0.0
    for expected in (1, 2):
        failures, last_start = watchdog.check_once(8020, failures, last_start, now=5000.0)
        assert failures == expected and events == ["start"]
    assert watchdog.check_once(8020, failures, last_start, now=5000.0) == (0, 5000.0)
    assert events == ["start", "stop", "start"]


def test_watchdog_only_matches_this_projects_scripts():
    python = "C:/py/python.exe"
    assert watchdog._matches([python, str(PROJECT_DIR / "server.py"), "--port", "8020"], watchdog.SERVER_SCRIPT)
    assert not watchdog._matches([python, "C:/alt-proiect/server.py"], watchdog.SERVER_SCRIPT)
    assert not watchdog._matches([python, str(PROJECT_DIR / "auto_loop.py")], watchdog.SERVER_SCRIPT)


def test_no_other_agent_instance_when_alone():
    import agent_base

    assert agent_base.find_other_instance("agent_care_nu_exista") is None


def test_server_supervises_every_agent_and_respects_stop(monkeypatch, tmp_path):
    import server

    assert set(server.SUPERVISED_AGENTS) == set(server.AGENT_IDS)
    assert server._agent_instance_pid("agent_care_nu_exista") is None

    monkeypatch.setattr(server, "_agents_stopped_by_user", set())
    monkeypatch.setattr(server, "AGENTS_DIR", tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(
        {"agents": {"ticket_generator": {"enabled": False}}}), encoding="utf-8")
    assert server._agent_should_run("lottery_manager") is True
    assert server._agent_should_run("ticket_generator") is False  # dezactivat în config
    server._agents_stopped_by_user.add("lottery_manager")
    assert server._agent_should_run("lottery_manager") is False  # oprit din UI → nu se repornește


def test_system_monitor_leaves_restarts_to_a_live_server(monkeypatch, tmp_path):
    import system_monitor

    monkeypatch.setattr(system_monitor, "STATUS_DIR", tmp_path)
    monitor = system_monitor.SystemMonitorAgent.__new__(system_monitor.SystemMonitorAgent)
    monitor.stale_threshold, monitor.hang_threshold = 120, 1800
    monitor.stale_warned, monitor.down_warned, monitor.last_incident_prune = set(), set(), time.time()
    monitor.MONITORED_AGENTS = ["ticket_generator"]
    restarted, warnings = [], []
    monkeypatch.setattr(monitor, "_restart_agent", lambda agent_id: restarted.append(agent_id), raising=False)
    monkeypatch.setattr(monitor, "log_warning", warnings.append, raising=False)
    monkeypatch.setattr(monitor, "get_config", lambda: {"auto_restart_agents": True}, raising=False)
    monkeypatch.setattr(monitor, "_agent_process", lambda pid, agent_id: None, raising=False)
    (tmp_path / "ticket_generator.json").write_text(json.dumps(
        {"pid": 4242, "status": "running", "last_activity": "2026-01-01T00:00:00"}), encoding="utf-8")

    def supervise():
        try:
            monitor.process()
        except Exception:
            pass  # pașii de după supravegherea agenților nu ne interesează aici

    monkeypatch.setattr(monitor, "_server_supervises", lambda: True, raising=False)
    supervise()
    supervise()
    assert restarted == [] and len(warnings) == 1  # serverul repornește; un singur avertisment

    monkeypatch.setattr(monitor, "_server_supervises", lambda: False, raising=False)
    supervise()
    assert restarted == ["ticket_generator"]  # server căzut → rezervă


def test_incident_pruning_keeps_recent_files(tmp_path):
    import system_monitor

    monitor = system_monitor.SystemMonitorAgent.__new__(system_monitor.SystemMonitorAgent)
    monitor.log_info = lambda message: None
    now = time.time()
    for index in range(8):
        file = tmp_path / f"incident_{index}.json"
        file.write_text("{}", encoding="utf-8")
        age_days = 30 if index < 3 else (8 - index) * 0.01
        os.utime(file, (now - age_days * 86400, now - age_days * 86400))
    removed = monitor._prune_incidents(max_age_days=14, max_files=4, incidents_dir=tmp_path)
    assert removed == 4  # 3 vechi + 1 peste limită (cel mai vechi dintre cele recente)
    assert sorted(f.name for f in tmp_path.iterdir()) == [f"incident_{i}.json" for i in (4, 5, 6, 7)]


def test_forward_test_keeps_longest_window_for_current_params(tmp_path):
    import auto_loop
    import forward_test

    path = tmp_path / "forward_test.json"

    def state(**kw):
        s = auto_loop.SearchState()
        s.best_metric, s.best_strategy, s.extra_strategies = 0.75, "cfg", ["sb"]
        for key, value in kw.items():
            setattr(s, key, value)
        return s

    forward_test.ensure_snapshot("loto_6_49_ro", state(), now=datetime(2025, 1, 10), path=path)
    forward_test.ensure_snapshot("loto_6_49_ro", state(tf=1.23), now=datetime(2025, 3, 10), path=path)
    forward_test.ensure_snapshot("loto_6_49_ro", state(), now=datetime(2025, 6, 10), path=path)  # revenire la A
    out = forward_test.summary("loto_6_49_ro", draws=[], path=path)
    assert out["frozen_after"] == "2025-01-10" and out["snapshots"] == 3
