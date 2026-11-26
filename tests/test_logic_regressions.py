from datetime import date, timedelta
from pathlib import Path

import pytest

import atomic_io
import walkforward
from generic_backtest import LotteryDraw, _fill_ranked_pool, backtest_strategy
from lottery_config import LotteryConfig


@pytest.fixture
def config():
    return LotteryConfig(id="audit", name="Audit", numbers_drawn=2,
                         number_range=6, ticket_size=2)


def test_failed_atomic_replace_preserves_previous_file(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    target.write_text("old", encoding="utf-8")

    def fail_replace(source, destination):
        assert Path(destination).read_text(encoding="utf-8") == "old"
        raise PermissionError("locked")

    monkeypatch.setattr(atomic_io.os, "replace", fail_replace)
    with pytest.raises(PermissionError):
        atomic_io.write_text_atomic(target, "new", retries=2, retry_delay=0)
    assert target.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_writer_does_not_delete_another_writers_temporary_file(tmp_path):
    target = tmp_path / "state.json"
    other = tmp_path / "state.json.other.tmp"
    other.write_text("pending", encoding="utf-8")
    atomic_io.write_text_atomic(target, "new")
    assert other.read_text(encoding="utf-8") == "pending"
    assert target.read_text(encoding="utf-8") == "new"


@pytest.mark.parametrize("options", [
    {"retries": 0}, {"retries": -1}, {"retries": True}, {"retries": 1.5},
    {"retry_delay": -1}, {"retry_delay": float("nan")},
    {"retry_delay": float("inf")},
])
def test_atomic_writer_rejects_invalid_retry_settings(tmp_path, options):
    with pytest.raises(ValueError):
        atomic_io.write_text_atomic(tmp_path / "state.json", "new", **options)


def test_rank_pool_normalizes_before_deduplication(config):
    ranked = ["3", 3, "3", None, [], float("nan"), float("inf"),
              2.5, True, "invalid", 0, 7, 2]
    assert _fill_ranked_pool(config, ranked) == [3, 2, 1, 4, 5, 6]


@pytest.mark.parametrize("options", [
    {"ticket_size": 0}, {"ticket_size": -1}, {"ticket_size": 7},
    {"recent_window": 0}, {"initial_window": -1}, {"initial_window": 0},
    {"ticket_size": True}, {"recent_window": 1.5},
])
def test_backtest_rejects_invalid_parameters(config, options):
    draws = [LotteryDraw(date(2020, 1, 1), None, None, (1, 2))] * 20
    with pytest.raises(ValueError):
        backtest_strategy(config, "audit", lambda *args: list(range(1, 7)), draws, **options)


def test_walkforward_weights_unequal_folds_by_evaluated_draws(config, monkeypatch):
    # 63 draws -> 37 warmup + folds of 5, 5, 5, 5, 6.
    # Only the last fold has hits, making incorrect equal weighting visible.
    draws = [LotteryDraw(date(2020, 1, 1) + timedelta(days=i), None, str(i),
                         (1, 2) if i >= 57 else (3, 4)) for i in range(63)]
    monkeypatch.setattr(walkforward, "get_lottery_config", lambda _: config)
    monkeypatch.setattr(walkforward, "load_lottery_draws", lambda _: draws)
    result = walkforward.run_walkforward("audit", lambda *args: list(range(1, 7)))
    assert [fold.test_size for fold in result.folds] == [5, 5, 5, 5, 6]
    assert result.wf_score == pytest.approx(12 / 26)
    assert result.baseline_edge == pytest.approx(12 / 26 - 2 * 2 / 6)


def test_backtest_ranker_sees_only_previous_draws(config):
    draws = [LotteryDraw(date(2020, 1, i + 1), None, str(i), numbers)
             for i, numbers in enumerate([(1, 2), (3, 4), (5, 6)])]
    observed = []

    def ranker(counts, recent, last_seen, index, window):
        observed.append((index, dict(counts), list(recent)))
        return list(range(1, 7))

    result = backtest_strategy(config, "audit", ranker, draws, initial_window=1)
    assert observed[0][1][3] == 0
    assert observed[1][1][5] == 0
    assert observed[0][2] == [{1, 2}]
    assert result.evaluated_draws == 2
    assert result.average_hits == 0
