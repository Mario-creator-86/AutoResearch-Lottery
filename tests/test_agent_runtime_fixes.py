"""Regresii pentru bug-urile găsite în logurile agenților (octombrie 2026)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

PROJECT_DIR = Path(__file__).resolve().parent.parent
AGENTS_DIR = PROJECT_DIR / "agents"
for _p in (PROJECT_DIR, AGENTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import agent_base


def _naive_utc(seconds_ago: float) -> str:
    """Formatul scris de agenți/CLI: datetime.utcnow().isoformat() (naiv, UTC)."""
    moment = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    return moment.replace(tzinfo=None).isoformat()


def test_naive_utc_message_is_fresh_regardless_of_local_timezone():
    # Înainte, .timestamp() pe un datetime naiv îl trata ca oră locală (UTC+3)
    # și orice stop_request trimis de CLI/server părea vechi de 3 ore.
    assert agent_base.is_fresh_message({"timestamp": _naive_utc(10)})
    assert agent_base.is_fresh_message({"timestamp": datetime.now(timezone.utc).isoformat()})
    assert not agent_base.is_fresh_message({"timestamp": _naive_utc(600)})
    assert not agent_base.is_fresh_message({"timestamp": "nu-e-data"})


class _DummyAgent(agent_base.BaseAgent):
    def process(self) -> None:
        raise AssertionError("process() nu trebuie rulat după un stop")


def _dummy_agent(tmp_path, monkeypatch) -> _DummyAgent:
    monkeypatch.setattr(agent_base, "write_json_atomic",
                        lambda path, data: Path(path).write_text(json.dumps(data), encoding="utf-8"))
    agent = _DummyAgent("dummy_agent", poll_interval=300)
    agent.status_file = tmp_path / "status.json"
    agent.log_file = tmp_path / "agent.log"
    agent.command_file = tmp_path / "to_dummy_agent.json"
    monkeypatch.setattr("agents.utils.send_heartbeat", lambda *a, **k: None, raising=False)
    return agent


def test_stop_request_is_honoured_and_consumed(tmp_path, monkeypatch):
    agent = _dummy_agent(tmp_path, monkeypatch)
    agent.command_file.write_text(json.dumps([
        {"type": "stop_request", "from": "cli", "timestamp": _naive_utc(5)},
    ]), encoding="utf-8")

    agent.run()

    assert agent.running is False
    # Consumat → un agent repornit imediat nu se mai oprește singur.
    assert json.loads(agent.command_file.read_text(encoding="utf-8")) == []
    status = json.loads(agent.status_file.read_text(encoding="utf-8"))
    assert status["poll_interval"] == 300


def test_ollama_ignores_stale_stop_request(tmp_path, monkeypatch):
    import ollama_strategist

    monkeypatch.setattr(agent_base, "COMMANDS_DIR", tmp_path)
    command_file = tmp_path / "to_ollama_strategist.json"
    command_file.write_text(json.dumps([
        {"type": "stop_request", "timestamp": "2026-04-10T06:27:57"},
    ]), encoding="utf-8")
    assert ollama_strategist.stop_requested_for("ollama_strategist") is False

    command_file.write_text(json.dumps([
        {"type": "stop_request", "timestamp": _naive_utc(5)},
    ]), encoding="utf-8")
    assert ollama_strategist.stop_requested_for("ollama_strategist") is True


def test_stale_threshold_scales_with_poll_interval(tmp_path, monkeypatch):
    import system_monitor

    monkeypatch.setattr(system_monitor, "STATUS_DIR", tmp_path)
    monitor = system_monitor.SystemMonitorAgent.__new__(system_monitor.SystemMonitorAgent)
    monitor.stale_threshold = 120

    def write_status(agent_id: str, seconds_ago: float, poll_interval: int | None) -> None:
        data = {"last_activity": _naive_utc(seconds_ago)}
        if poll_interval is not None:
            data["poll_interval"] = poll_interval
        (tmp_path / f"{agent_id}.json").write_text(json.dumps(data), encoding="utf-8")

    # ticket_generator doarme 300 s: un heartbeat de acum 200 s e normal.
    write_status("slow", 200, 300)
    assert monitor._check_heartbeat_stale("slow") is False
    write_status("slow_dead", 700, 300)
    assert monitor._check_heartbeat_stale("slow_dead") is True
    # Agenții fără poll_interval păstrează pragul vechi de 120 s.
    write_status("legacy", 200, None)
    assert monitor._check_heartbeat_stale("legacy") is True
    # Timestamp cu fus orar explicit nu mai aruncă TypeError (era raportat stale).
    (tmp_path / "aware.json").write_text(json.dumps(
        {"last_activity": datetime.now(timezone.utc).isoformat(), "poll_interval": 30}), encoding="utf-8")
    assert monitor._check_heartbeat_stale("aware") is False


def test_experiment_monitor_reads_auto_loop_progress_format(tmp_path, monkeypatch):
    import experiment_monitor

    progress_file = tmp_path / "progress.json"
    progress_file.write_text(json.dumps({
        "status": "running",
        "experiment_id": "auto_iter4459",
        "tested": 69,
        "total": 72,
        "best_so_far": {"config": "rw150_best", "average_hits": 0.785652},
        "updated_at": "2026-09-26T20:48:17",
        "loop_info": {"search_phase": "phase2"},
    }), encoding="utf-8")
    monkeypatch.setattr(experiment_monitor, "get_lottery_config",
                        lambda _lid: SimpleNamespace(progress_file=progress_file))

    agent = experiment_monitor.ExperimentMonitorAgent.__new__(experiment_monitor.ExperimentMonitorAgent)
    progress = agent._check_experiment_progress("loto_6_49_ro")

    assert progress["best_metric"] == 0.785652
    assert progress["best_config"] == "rw150_best"
    assert progress["search_phase"] == "phase2"
    assert progress["timestamp"] == "2026-09-26T20:48:17"


def test_experiment_monitor_does_not_announce_first_observation(monkeypatch):
    import experiment_monitor

    agent = experiment_monitor.ExperimentMonitorAgent.__new__(experiment_monitor.ExperimentMonitorAgent)
    agent.experiments_cache = {}
    successes: list[str] = []
    monkeypatch.setattr(agent, "log_success", successes.append, raising=False)

    exp = {"lottery": "loto_6_49_ro", "experiment_id": "it1", "best_metric": 0.78}
    agent._send_notifications([exp])
    assert successes == []  # pornire: starea curentă, nu „record nou”

    agent._send_notifications([{**exp, "best_metric": 0.79}])
    assert len(successes) == 1

    agent._send_notifications([{**exp, "experiment_id": "it2", "best_metric": 0.80}])
    assert len(successes) == 1  # iterație nouă → best_so_far pornește de la zero


def test_agent_log_rotates_when_too_big(tmp_path):
    log_file = tmp_path / "agent.log"
    log_file.write_text("x" * 100, encoding="utf-8")
    agent_base.rotate_log_if_needed(log_file, max_bytes=50)
    assert not log_file.exists()
    assert (tmp_path / "agent.log.1").read_text(encoding="utf-8") == "x" * 100
    agent_base.rotate_log_if_needed(tmp_path / "lipsa.log", max_bytes=50)  # nu aruncă


def _monitor_with_status(tmp_path, monkeypatch, *, seconds_ago, poll_interval, process):
    import system_monitor

    monkeypatch.setattr(system_monitor, "STATUS_DIR", tmp_path)
    monitor = system_monitor.SystemMonitorAgent.__new__(system_monitor.SystemMonitorAgent)
    monitor.stale_threshold = 120
    monitor.hang_threshold = 1800
    monkeypatch.setattr(monitor, "_agent_process", lambda pid, agent_id: process, raising=False)
    (tmp_path / "ticket_generator.json").write_text(json.dumps({
        "pid": 4242, "last_activity": _naive_utc(seconds_ago), "poll_interval": poll_interval,
    }), encoding="utf-8")
    return monitor


def test_busy_agent_is_stale_but_not_hung(tmp_path, monkeypatch):
    monitor = _monitor_with_status(tmp_path, monkeypatch, seconds_ago=900, poll_interval=300, process=object())
    status = monitor._check_agent_status("ticket_generator")
    assert status["alive"] and status["stale"] and not status["hung"]


def test_silent_agent_for_long_is_hung(tmp_path, monkeypatch):
    monitor = _monitor_with_status(tmp_path, monkeypatch, seconds_ago=3 * 3600, poll_interval=300, process=object())
    assert monitor._check_agent_status("ticket_generator")["hung"]


def test_recycled_pid_counts_as_dead(tmp_path, monkeypatch):
    # PID-ul aparține acum altui program → agentul e mort și trebuie repornit.
    monitor = _monitor_with_status(tmp_path, monkeypatch, seconds_ago=3 * 3600, poll_interval=300, process=None)
    status = monitor._check_agent_status("ticket_generator")
    assert not status["alive"] and not status["hung"]


def test_agent_process_rejects_pid_running_other_script():
    import os
    import system_monitor

    monitor = system_monitor.SystemMonitorAgent.__new__(system_monitor.SystemMonitorAgent)
    # Procesul curent e pytest, nu ticket_generator.py.
    assert monitor._agent_process(os.getpid(), "ticket_generator") is None


def test_process_runs_script_checks_command_line(monkeypatch):
    import lock_utils

    monkeypatch.setattr(lock_utils, "_process_identity",
                        lambda pid: ("python.exe", r'python.exe "C:\proj\auto_loop.py" --lottery x'))
    assert lock_utils.process_runs_script(1, Path("auto_loop.py"))
    monkeypatch.setattr(lock_utils, "_process_identity", lambda pid: ("chrome.exe", "chrome.exe --type=renderer"))
    assert not lock_utils.process_runs_script(1, Path("auto_loop.py"))


def test_ledger_ignores_unknown_lottery_ids(tmp_path, monkeypatch):
    import decision_ledger

    monkeypatch.setattr(decision_ledger, "DATA_DIR", tmp_path)
    # ID inventat de model (s-a întâmplat: „loto_pl") și o încercare de ieșire din DATA_DIR.
    decision_ledger.append_entry("ai_action", lottery_id="loto_pl", thought="x")
    decision_ledger.append_entry("ai_action", lottery_id="../evil", thought="y")
    decision_ledger.append_entry("ai_action", lottery_id="joker_ro", thought="z")

    assert sorted(p.name for p in tmp_path.iterdir()) == ["decision_ledger_global.jsonl", "joker_ro"]
    assert not (tmp_path.parent / "evil").exists()
    assert len(decision_ledger.read_entries(None, limit=10)) == 3  # fiecare o singură dată în global
    assert decision_ledger.read_entries("loto_pl", limit=10) == []
