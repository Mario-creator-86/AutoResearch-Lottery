"""La pornirea buclei, best_metric se recalculează pe istoricul/codul curent."""

from __future__ import annotations

import random
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import auto_loop
from generic_backtest import LotteryDraw


def _draws(n: int = 300) -> list[LotteryDraw]:
    rng = random.Random(5)
    return [LotteryDraw(date(2020, 1, 1) + timedelta(days=i), None, None,
                        tuple(sorted(rng.sample(range(1, 50), 6)))) for i in range(n)]


def test_stale_best_metric_is_rebaselined_and_logged(tmp_path, monkeypatch):
    auto_loop.configure_runtime("loto_6_49_ro")
    monkeypatch.setattr(auto_loop, "_JACKPOT_DATA", {})
    monkeypatch.setattr(auto_loop, "_DRAW_DATES", [])
    state = auto_loop.SearchState()
    state.best_strategy = "stale_best"
    state.best_metric = 9.9  # imposibil: prag „fantomă" dintr-o rulare veche
    results = tmp_path / "results.tsv"

    new_state, changed = auto_loop._rebaseline_best(state, _draws(), results_file=results)

    assert changed
    assert 0.0 < new_state.best_metric < 6.0
    row = results.read_text(encoding="utf-8").strip().split("\t")
    assert row[2] == "REBASELINE" and row[3] == "stale_best"
    assert float(row[1]) == new_state.best_metric


def test_rebaseline_is_noop_when_metric_is_current(tmp_path, monkeypatch):
    auto_loop.configure_runtime("loto_6_49_ro")
    monkeypatch.setattr(auto_loop, "_JACKPOT_DATA", {})
    monkeypatch.setattr(auto_loop, "_DRAW_DATES", [])
    state = auto_loop.SearchState()
    state.best_strategy = "current"
    state.best_metric = 9.9
    draws = _draws()
    state, _ = auto_loop._rebaseline_best(state, draws, results_file=tmp_path / "a.tsv")

    second = tmp_path / "b.tsv"
    _state, changed = auto_loop._rebaseline_best(state, draws, results_file=second)
    assert not changed
    assert not second.exists()


def test_rebaseline_skips_empty_state(tmp_path):
    auto_loop.configure_runtime("loto_6_49_ro")
    state = auto_loop.SearchState()  # best_metric 0 → nimic de recalculat
    _state, changed = auto_loop._rebaseline_best(state, _draws(), results_file=tmp_path / "r.tsv")
    assert not changed
