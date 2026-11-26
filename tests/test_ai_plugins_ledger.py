from __future__ import annotations

import json
from collections import deque
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import ai_strategy_lab
import decision_ledger
from lottery_config import get_lottery_config


# ─── Decision Ledger ─────────────────────────────────────────────────────────


def test_ledger_append_and_read(tmp_path, monkeypatch):
    monkeypatch.setattr(decision_ledger, "DATA_DIR", tmp_path)
    decision_ledger.append_entry(
        "ai_analysis", lottery_id="loto_6_49_ro",
        thought="gandire", action="numere", why="motiv", payload={"n": [1, 2]},
    )
    decision_ledger.append_entry(
        "strategy_code", lottery_id="loto_6_49_ro",
        thought="t2", action="a2", why="w2", payload={"promoted": True}, status="promoted",
    )
    entries = decision_ledger.read_entries("loto_6_49_ro", limit=10)
    assert len(entries) == 2
    assert entries[0]["type"] == "strategy_code"  # read_entries: cele mai noi primele
    assert entries[0]["payload"]["promoted"] is True
    # flux global conține ambele
    assert len(decision_ledger.read_entries(None, limit=10)) == 2
    # filtrare pe tip
    only_code = decision_ledger.read_entries(None, limit=10, entry_type="strategy_code")
    assert len(only_code) == 1 and only_code[0]["type"] == "strategy_code"


def test_ledger_update_latest_result(tmp_path, monkeypatch):
    monkeypatch.setattr(decision_ledger, "DATA_DIR", tmp_path)
    decision_ledger.append_entry("ai_tickets", lottery_id="joker_ro",
                                 thought="t", action="bilete", why="w")
    ok = decision_ledger.update_latest_result(
        {"type": "ai_tickets", "lottery": "joker_ro"},
        {"total_hits": 3}, lottery_id="joker_ro",
    )
    assert ok
    entries = decision_ledger.read_entries("joker_ro", limit=5)
    assert entries[-1]["result"]["total_hits"] == 3
    assert entries[-1]["status"] == "evaluated"
    # a doua actualizare nu rescrie
    ok2 = decision_ledger.update_latest_result(
        {"type": "ai_tickets", "lottery": "joker_ro"},
        {"total_hits": 99}, lottery_id="joker_ro",
    )
    assert not ok2
    assert decision_ledger.read_entries("joker_ro", limit=5)[-1]["result"]["total_hits"] == 3


def test_ledger_invalid_type_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(decision_ledger, "DATA_DIR", tmp_path)
    entry = decision_ledger.append_entry("hacky_type", lottery_id="joker_ro", thought="x")
    assert entry["type"] == "system"


# ─── AI Strategy Lab ─────────────────────────────────────────────────────────

GOOD_PLUGIN = """
NAME = "zfreqx"
DESCRIPTION = "pure frequency ranking"
SOURCE = "unit-test"

def create(config):
    nums = list(range(1, config.number_range + 1))
    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        return sorted(nums, key=lambda n: (-counts.get(n, 0), n))
    return rank
"""

BAD_PLUGIN = """
NAME = "1badname"
def create(config):
    return lambda *a: []
"""


def _write_plugin(lottery_id: str, code: str, name: str) -> Path:
    path = ai_strategy_lab.strategies_dir(lottery_id) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")
    return path


def test_plugin_validate_and_backtest(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_strategy_lab, "AI_STRATEGIES_DIR", tmp_path / "ai_strategies")
    lottery_id = "loto_5_40_ro"
    path = _write_plugin(lottery_id, GOOD_PLUGIN, "zfreqx_t.py")
    meta = ai_strategy_lab.validate_plugin(path, lottery_id)
    assert meta["name"] == "zfreqx"

    draws = [
        SimpleNamespace(
            draw_date=date(2026, 1, 1),
            numbers=tuple(sorted({((i * 5 + j * 9) % 40) + 1 for j in range(6)})),
            bonuses=(),
        )
        for i in range(80)
    ]
    result = ai_strategy_lab.backtest_plugin(path, lottery_id, draws)
    assert 0 <= result["average_hits"] <= 2.0  # 5 pick x 6/40 => teoretic 0.75, frecvența poate depăși


def test_plugin_rejects_bad_name(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_strategy_lab, "AI_STRATEGIES_DIR", tmp_path / "ai_strategies")
    path = _write_plugin("joker_ro", BAD_PLUGIN, "bad.py")
    try:
        ai_strategy_lab.validate_plugin(path, "joker_ro")
        raise AssertionError("expected validation failure")
    except Exception:
        pass


def test_promote_and_register_for_config(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_strategy_lab, "AI_STRATEGIES_DIR", tmp_path / "ai_strategies")
    lottery_id = "loto_5_40_ro"
    path = _write_plugin(lottery_id, GOOD_PLUGIN, "zfreqx_r.py")
    ai_strategy_lab.validate_plugin(path, lottery_id)
    ai_strategy_lab.promote_plugin(lottery_id, "zfreqx_r.py", "zfreqx", 0.61, 0.6)

    config = get_lottery_config(lottery_id)
    registered = ai_strategy_lab.register_promoted_for_config(config)
    assert "zfreqx" in registered
    fn = ai_strategy_lab.make_extra(config, "zfreqx")
    assert callable(fn)
    # cheie necunoscută → None (fallback grațios în auto_loop)
    assert ai_strategy_lab.make_extra(config, "zghost") is None


def test_unique_key_prefix_and_uniqueness(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_strategy_lab, "AI_STRATEGIES_DIR", tmp_path / "ai_strategies")
    lottery_id = "loto_5_40_ro"
    key1 = ai_strategy_lab.unique_key(lottery_id, "zpair-gap")
    assert key1.startswith("z")
    assert ai_strategy_lab.NAME_RE.match(key1)
    ai_strategy_lab.promote_plugin(lottery_id, "f.py", key1, 0.5, 0.4)
    key2 = ai_strategy_lab.unique_key(lottery_id, "zpair-gap")
    assert key2 != key1
