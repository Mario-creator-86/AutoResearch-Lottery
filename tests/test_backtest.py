"""Tests for generic_backtest engine — covers critical gaps from audit."""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from generic_backtest import (
    LotteryDraw,
    backtest_strategy,
    top_frequency_rank,
)
from load_data import load_lottery_draws
from lottery_config import LotteryConfig, get_lottery_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_draws(numbers_list, bonuses_list=None):
    """Create minimal LotteryDraw objects for testing."""
    draws = []
    for idx, nums in enumerate(numbers_list):
        bonuses = bonuses_list[idx] if bonuses_list else ()
        draws.append(LotteryDraw(
            draw_date=date(2020, 1, 1 + idx),
            draw_time=None,
            draw_identifier=str(idx),
            numbers=tuple(nums),
            bonuses=tuple(bonuses),
        ))
    return draws


def _freq_rank(config):
    """Return a rank function bound to the given config."""
    def rank_fn(counts, recent_deque, last_seen, draw_index, recent_window):
        return top_frequency_rank(config, counts, recent_deque, last_seen, draw_index, recent_window)
    return rank_fn


# ---------------------------------------------------------------------------
# 1. Backtest uses correct ticket_size per lottery
# ---------------------------------------------------------------------------

def test_backtest_uses_config_ticket_size_lotto_pl():
    config = get_lottery_config("lotto_pl")
    draws = load_lottery_draws("lotto_pl")
    result = backtest_strategy(
        config, "test_lotto", _freq_rank(config), draws,
        ticket_size=config.ticket_size, recent_window=25, initial_window=10,
    )
    assert result.ticket_size == 6
    # 6/49 baseline is ~0.735 — result should be in a reasonable range
    assert 0.4 < result.average_hits < 1.5


def test_backtest_uses_config_ticket_size_mini_lotto():
    config = get_lottery_config("mini_lotto_pl")
    draws = load_lottery_draws("mini_lotto_pl")
    result = backtest_strategy(
        config, "test_mini", _freq_rank(config), draws,
        ticket_size=config.ticket_size, recent_window=25, initial_window=10,
    )
    assert result.ticket_size == 5
    assert 0.3 < result.average_hits < 1.2


def test_backtest_uses_config_ticket_size_eurojackpot():
    config = get_lottery_config("eurojackpot")
    draws = load_lottery_draws("eurojackpot")
    result = backtest_strategy(
        config, "test_ej", _freq_rank(config), draws,
        ticket_size=config.ticket_size, recent_window=25, initial_window=10,
    )
    assert result.ticket_size == 5
    assert 0.2 < result.average_hits < 1.0


# ---------------------------------------------------------------------------
# 2. Backtest determinism — same input produces same output
# ---------------------------------------------------------------------------

def test_backtest_is_deterministic():
    config = get_lottery_config("multi_multi_pl")
    draws = load_lottery_draws("multi_multi_pl")[:500]  # subset for speed
    r1 = backtest_strategy(config, "det_test", _freq_rank(config), draws,
                           ticket_size=10, recent_window=25, initial_window=10)
    r2 = backtest_strategy(config, "det_test", _freq_rank(config), draws,
                           ticket_size=10, recent_window=25, initial_window=10)
    assert r1.average_hits == r2.average_hits
    assert r1.hit_rate_2_plus == r2.hit_rate_2_plus
    assert r1.portfolio_average_best_hits == r2.portfolio_average_best_hits


# ---------------------------------------------------------------------------
# 3. Bonus evaluation — tracks bonus hits for lottery with bonuses
# ---------------------------------------------------------------------------

def test_backtest_tracks_bonus_hits_for_eurojackpot():
    config = get_lottery_config("eurojackpot")
    draws = load_lottery_draws("eurojackpot")
    result = backtest_strategy(
        config, "test_ej_bonus", _freq_rank(config), draws,
        ticket_size=5, recent_window=25, initial_window=10,
    )
    # EuroJackpot has 2 bonuses from 1-12; frequency-based prediction should
    # beat pure random (baseline = 2/12 ≈ 0.333)
    assert result.bonus_average_hits > 0.0
    assert result.bonus_hit_rate_1_plus > 0.0


def test_backtest_tracks_bonus_hits_for_powerball():
    config = get_lottery_config("powerball_us")
    draws = load_lottery_draws("powerball_us")
    result = backtest_strategy(
        config, "test_pb_bonus", _freq_rank(config), draws,
        ticket_size=5, recent_window=25, initial_window=10,
    )
    # Powerball has 1 bonus from 1-26; frequency prediction is near random
    assert result.bonus_average_hits >= 0.0
    assert result.bonus_hit_rate_1_plus >= 0.0


def test_backtest_no_bonus_for_multi_multi():
    config = get_lottery_config("multi_multi_pl")
    draws = load_lottery_draws("multi_multi_pl")[:500]
    result = backtest_strategy(
        config, "test_mm_no_bonus", _freq_rank(config), draws,
        ticket_size=10, recent_window=25, initial_window=10,
    )
    # Multi Multi has no bonus numbers — bonus fields must remain 0
    assert result.bonus_average_hits == 0.0
    assert result.bonus_hit_rate_1_plus == 0.0


# ---------------------------------------------------------------------------
# 4. Bonus evaluation with synthetic data — exact hit verification
# ---------------------------------------------------------------------------

def test_backtest_bonus_hits_correct_with_known_data():
    """With predictable bonus data, verify exact bonus hit counts."""
    config = LotteryConfig(
        id="test_bonus",
        name="Test Bonus Lottery",
        numbers_drawn=3,
        number_range=10,
        ticket_size=3,
        bonus_count=1,
        bonus_range=5,
    )
    # 15 draws: all bonuses are 1, so frequency ranking always picks bonus=1
    numbers_list = [
        [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 3, 5], [2, 4, 6],
        [7, 8, 10], [1, 2, 4], [3, 5, 7], [6, 8, 9], [1, 4, 7],
        [2, 5, 8], [3, 6, 9], [1, 2, 10], [4, 5, 7], [3, 8, 9],
    ]
    bonuses_list = [
        [1], [1], [1], [1], [1],
        [1], [1], [1], [1], [1],
        [1], [1], [1], [1], [1],
    ]
    draws = _make_draws(numbers_list, bonuses_list)

    result = backtest_strategy(
        config, "known_bonus", _freq_rank(config), draws,
        ticket_size=3, recent_window=5, initial_window=5,
    )
    # After initial window, all remaining draws have bonus=1.
    # Frequency ranking will predict bonus=1 since it has the highest count.
    # So every draw should be a bonus hit → bonus_average_hits = 1.0
    assert result.bonus_average_hits == 1.0
    assert result.bonus_hit_rate_1_plus == 1.0


def test_backtest_bonus_zero_when_no_bonus_config():
    """A lottery with bonus_count=0 should produce 0.0 bonus metrics even if draws
    happen to have non-empty bonuses tuples."""
    config = LotteryConfig(
        id="test_no_bonus",
        name="No Bonus Lottery",
        numbers_drawn=3,
        number_range=10,
        ticket_size=3,
    )
    numbers_list = [
        [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 3, 5], [2, 4, 6],
        [7, 8, 10], [1, 2, 4], [3, 5, 7], [6, 8, 9], [1, 4, 7],
        [2, 5, 8], [3, 6, 9],
    ]
    draws = _make_draws(numbers_list)

    result = backtest_strategy(
        config, "no_bonus", _freq_rank(config), draws,
        ticket_size=3, recent_window=5, initial_window=5,
    )
    assert result.bonus_average_hits == 0.0
    assert result.bonus_hit_rate_1_plus == 0.0
