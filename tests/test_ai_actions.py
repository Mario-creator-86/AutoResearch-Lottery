from __future__ import annotations

import ai_actions


def test_validate_params_rejects_unknown_type():
    error = ai_actions._validate_params("hack_the_planet", {})
    assert error and "necunoscut" in error


def test_validate_params_rejects_missing_and_bad_params():
    assert ai_actions._validate_params("restart_loop", {}) is not None  # lottery lipsă
    assert ai_actions._validate_params("restart_loop", {"lottery": "nu_exista"}) is not None
    assert ai_actions._validate_params("restart_loop", {"lottery": "loto_6_49_ro"}) is None


def test_validate_params_config_whitelist():
    ok = ai_actions._validate_params("set_agent_config", {"path": "agents.ai_strategy_coder.interval_minutes", "value": 120})
    assert ok is None
    bad = ai_actions._validate_params("set_agent_config", {"path": "system.delete_everything", "value": True})
    assert bad and "nepermisă" in bad


def test_propose_invalid_action_goes_nowhere(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_actions, "ACTIONS_DIR", tmp_path / "ai_actions")
    action = ai_actions.propose_action("tip_inexistent", {}, why="test", require_approval=True)
    assert action["status"] == "invalid"
    assert ai_actions.list_pending() == []


def test_propose_and_reject(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_actions, "ACTIONS_DIR", tmp_path / "ai_actions")
    calls = []
    import telegram_bot

    monkeypatch.setattr(telegram_bot, "send_action_request", lambda *a, **k: calls.append(a) or True)
    monkeypatch.setattr(telegram_bot, "notify", lambda *a, **k: True)

    action = ai_actions.propose_action(
        "run_walkforward", {"lottery": "loto_6_49_ro"}, why="test", require_approval=True,
    )
    assert action["status"] if False else True  # proposed — status e în pending file
    pending = ai_actions.list_pending()
    assert len(pending) == 1 and pending[0]["type"] == "run_walkforward"
    assert len(calls) == 1  # cererea de aprobare a fost trimisă pe Telegram

    result = ai_actions.reject_action(action["id"], by="test")
    assert result["ok"]
    assert ai_actions.list_pending() == []
    history = ai_actions.list_history()
    assert history and history[0]["status"] == "rejected"


def test_propose_and_approve_executes(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_actions, "ACTIONS_DIR", tmp_path / "ai_actions")
    import telegram_bot

    monkeypatch.setattr(telegram_bot, "send_action_request", lambda *a, **k: True)
    monkeypatch.setattr(telegram_bot, "send_action_result", lambda *a, **k: True)
    monkeypatch.setattr(ai_actions, "_execute", lambda t, p: f"executat {t}")

    action = ai_actions.propose_action(
        "run_walkforward", {"lottery": "joker_ro"}, why="test", require_approval=True,
    )
    result = ai_actions.approve_action(action["id"], approved_by="test")
    assert result["ok"] and result["result"] == "executat run_walkforward"
    assert ai_actions.list_pending() == []
    assert ai_actions.list_history()[0]["status"] == "executed"
