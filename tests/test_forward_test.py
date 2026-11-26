from __future__ import annotations

import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import auto_loop
import forward_test
from generic_backtest import LotteryDraw


def _draws(n: int = 300) -> list[LotteryDraw]:
    rng = random.Random(9)
    return [LotteryDraw(date(2025, 1, 1) + timedelta(days=i), None, None,
                        tuple(sorted(rng.sample(range(1, 50), 6)))) for i in range(n)]


def _state(**kw) -> auto_loop.SearchState:
    state = auto_loop.SearchState()
    state.best_metric = 0.75
    state.best_strategy = "cfg"
    state.extra_strategies = ["sb"]  # fără jc: evaluarea nu depinde de date jackpot
    for key, value in kw.items():
        setattr(state, key, value)
    return state


def test_snapshot_is_created_once_per_param_set(tmp_path):
    path = tmp_path / "forward_test.json"
    now = datetime(2025, 9, 7, 12, 0)
    first = forward_test.ensure_snapshot("loto_6_49_ro", _state(), now=now, path=path)
    assert first and first["frozen_after"] == "2025-09-07"
    assert forward_test.ensure_snapshot("loto_6_49_ro", _state(), now=now, path=path) is None
    changed = forward_test.ensure_snapshot("loto_6_49_ro", _state(tf=1.23), now=now, path=path)
    assert changed is not None
    assert len(forward_test.load_snapshots("loto_6_49_ro", path)) == 2


def test_empty_state_is_not_snapshotted(tmp_path):
    state = _state(best_metric=0.0)
    assert forward_test.ensure_snapshot("loto_6_49_ro", state, path=tmp_path / "f.json") is None


def test_evaluation_uses_only_draws_after_freeze(tmp_path):
    draws = _draws(300)
    path = tmp_path / "forward_test.json"
    freeze_day = draws[249].draw_date  # extragerile 250..299 sunt „viitoare"
    forward_test.ensure_snapshot("loto_6_49_ro", _state(),
                                 now=datetime.combine(freeze_day, datetime.min.time()), path=path)
    out = forward_test.summary("loto_6_49_ro", draws=draws, path=path)
    assert out["n_draws"] == 50
    assert 0.0 <= out["metric"] <= 6.0
    assert out["verdict"] in {"noise", "suggestive", "significant"}
    assert out["snapshots"] == 1


def test_no_forward_draws_yet(tmp_path):
    draws = _draws(100)
    path = tmp_path / "forward_test.json"
    forward_test.ensure_snapshot("loto_6_49_ro", _state(),
                                 now=datetime.combine(draws[-1].draw_date, datetime.min.time()), path=path)
    out = forward_test.summary("loto_6_49_ro", draws=draws, path=path)
    assert out["n_draws"] == 0 and out["verdict"] == "insufficient"
