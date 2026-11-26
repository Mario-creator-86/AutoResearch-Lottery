"""Regression tests for the bugs fixed in the September 2026 audit."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

PROJECT_DIR = Path(__file__).resolve().parent.parent
AGENTS_DIR = PROJECT_DIR / "agents"
for _p in (PROJECT_DIR, AGENTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import decision_ledger
import load_data
import server


def test_ledger_update_matches_nested_payload_subset(tmp_path, monkeypatch):
    monkeypatch.setattr(decision_ledger, "DATA_DIR", tmp_path)
    decision_ledger.append_entry(
        "ai_action", lottery_id=None, action="restart_loop",
        payload={"id": "act_1", "risk": "medium", "needs_approval": True},
    )
    ok = decision_ledger.update_latest_result({"payload": {"id": "act_1"}}, {"ok": True})
    assert ok
    assert decision_ledger.read_entries(None, limit=1)[0]["result"] == {"ok": True}
    assert not decision_ledger.update_latest_result({"payload": {"id": "other"}}, {"ok": True})


def test_telegram_agent_ignores_foreign_chats():
    from telegram_agent import TelegramAgent

    mine = {"message": {"chat": {"id": 123}, "text": "/status"}}
    foreign = {"message": {"chat": {"id": 999}, "text": "/status"}}
    callback = {"callback_query": {"data": "approve:act_1", "message": {"chat": {"id": 999}}}}
    assert TelegramAgent._from_allowed_chat(mine, "123")
    assert not TelegramAgent._from_allowed_chat(foreign, "123")
    assert not TelegramAgent._from_allowed_chat(callback, "123")
    assert not TelegramAgent._from_allowed_chat(mine, "")


def test_refresh_history_when_second_daily_draw_missing(monkeypatch):
    import ai_ticket_agent
    import scrape_lotteries

    agent = ai_ticket_agent.AITicketAgent.__new__(ai_ticket_agent.AITicketAgent)
    agent.update_heartbeat = lambda *a, **k: None
    agent.log_success = agent.log_warning = lambda *a, **k: None
    downloaded: list[str] = []
    monkeypatch.setattr(scrape_lotteries, "SCRAPER_MAP", {"kaskada_pl": object()})
    monkeypatch.setattr(scrape_lotteries, "download_lottery", lambda lid: downloaded.append(lid))

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    pending = [{"draw_date": yesterday, "draw_index": 2, "draw_time": "21:40"}]
    only_first_draw = {yesterday: [object()]}
    assert agent._maybe_refresh_history("kaskada_pl", pending, only_first_draw) is True
    assert downloaded == ["kaskada_pl"]

    # both draws already in history → no refresh
    agent._history_refresh_at = {}
    assert agent._maybe_refresh_history("kaskada_pl", pending, {yesterday: [object(), object()]}) is False


def test_history_cache_refreshes_when_draws_file_changes(monkeypatch, tmp_path):
    draws_file = tmp_path / "lotto_pl.csv"
    draws_file.write_text("header\n", encoding="utf-8")
    config = SimpleNamespace(id="lotto_pl", draws_file=draws_file)
    entries = [{
        "id": "a", "saved_at": "2026-03-28T10:00:00", "saved_date": "2026-03-28",
        "target_draw_date": "2026-03-28", "main": [1, 2, 3, 4, 5, 6], "bonus": [],
    }]
    current_draws: list = []
    monkeypatch.setattr(server, "_load_ticket_history", lambda _c: list(entries))
    monkeypatch.setattr(server, "_ai_history_entries", lambda _lid: [])
    monkeypatch.setattr(load_data, "load_specific_lottery_draws", lambda *a, **k: list(current_draws))
    server._history_payload_cached.cache_clear()

    assert server._history_payload(config)["groups"][0]["draw_available"] is False

    current_draws.append(SimpleNamespace(draw_date=date(2026, 3, 28), draw_identifier=None,
                                         draw_time=None, numbers=(1, 2, 3, 4, 5, 6), bonuses=()))
    draws_file.write_text("header\nnew draw row\n", encoding="utf-8")
    assert server._history_payload(config)["groups"][0]["draw_available"] is True


def test_ai_stop_loop_writes_pause_marker(monkeypatch, tmp_path):
    import ai_actions
    import lottery_config

    config = SimpleNamespace(
        state_dir=tmp_path, stop_request_file=tmp_path / ".autoloop.stop",
        lock_file=tmp_path / ".experiment.lock",
    )
    monkeypatch.setattr(lottery_config, "get_lottery_config", lambda _lid: config)
    ai_actions._exec_stop_loop("joker_ro")
    assert (tmp_path / ".autoloop.paused").exists()


def test_agents_use_configured_server_port(monkeypatch):
    from agents import utils

    monkeypatch.delenv("AUTORESEARCH_PORT", raising=False)
    assert utils.server_base_url() == "http://localhost:8020"
    monkeypatch.setenv("AUTORESEARCH_PORT", "9123")
    assert utils.server_base_url() == "http://localhost:9123"


def test_lock_matches_windows_venv_launcher(monkeypatch):
    import lock_utils

    venv_exe = r"C:\proj\.venv\Scripts\python.exe"
    script = r"C:\proj\auto_loop.py"
    owner = lock_utils.LockOwner(pid=4242, executable=venv_exe, script=script, project_dir=r"C:\proj")
    base_exe = r"C:\Python314\python.exe"

    monkeypatch.setattr(lock_utils, "_process_identity",
                        lambda pid: (base_exe, f'"{venv_exe}" "{script}" --lottery joker_ro'))
    assert lock_utils.lock_matches_process(owner)

    # a genuinely different interpreter (PID reuse) is still rejected
    monkeypatch.setattr(lock_utils, "_process_identity",
                        lambda pid: (base_exe, f'"{base_exe}" "{script}" --lottery joker_ro'))
    assert not lock_utils.lock_matches_process(owner)
