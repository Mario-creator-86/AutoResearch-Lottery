from __future__ import annotations

from collections import deque
from datetime import date, timedelta

from generic_backtest import (
    LotteryDraw,
    make_anticorrelation_wrapper,
    make_bayesian_freq_rank,
    make_drift_detector_rank,
    make_jackpot_cycle_rank,
)
from jackpot_proxy import JackpotInfo
from lottery_config import LotteryConfig, get_lottery_config


def _make_draws(count: int, number_range: int = 80, numbers_drawn: int = 20) -> list[LotteryDraw]:
    import random

    rng = random.Random(7)
    start = date(2020, 1, 1)
    draws = []
    for index in range(count):
        draws.append(
            LotteryDraw(
                draw_date=start + timedelta(days=index),
                draw_time=None,
                draw_identifier=str(index),
                numbers=tuple(sorted(rng.sample(range(1, number_range + 1), numbers_drawn))),
            )
        )
    return draws


def _build_state(
    draws: list[LotteryDraw],
    config: LotteryConfig,
    *,
    initial_window: int = 10,
    recent_window: int = 25,
):
    counts = {n: 0 for n in range(1, config.number_range + 1)}
    last_seen = {n: -1 for n in range(1, config.number_range + 1)}
    recent_deque: deque[set[int]] = deque(maxlen=recent_window)
    for index, draw in enumerate(draws[:initial_window]):
        for number in draw.numbers:
            counts[number] += 1
            last_seen[number] = index
        recent_deque.append(set(draw.numbers))
    return counts, recent_deque, last_seen


def test_jackpot_cycle_rank_returns_full_domain():
    config = get_lottery_config("multi_multi_pl")
    draws = _make_draws(50)
    jackpot_data = {
        draw.draw_date: JackpotInfo(phase=0.5, draws_since_win=5, is_rollover=True)
        for draw in draws
    }
    rank_fn = make_jackpot_cycle_rank(jackpot_data, [draw.draw_date for draw in draws], config)
    counts, recent_deque, last_seen = _build_state(draws, config)

    ranked = rank_fn(counts, recent_deque, last_seen, 20, 25)

    assert len(ranked) == config.number_range
    assert set(ranked) == set(range(1, config.number_range + 1))


def test_jackpot_cycle_phase_changes_ranking():
    config = get_lottery_config("multi_multi_pl")
    draws = _make_draws(5)
    draw_dates = [draw.draw_date for draw in draws]
    counts = {n: 1 for n in range(1, config.number_range + 1)}
    counts[1] = 10
    counts[2] = 8
    counts[3] = 6
    recent_deque: deque[set[int]] = deque([{3, 4, 5, 6}, {3, 7, 8, 9}], maxlen=25)
    last_seen = {n: 0 for n in range(1, config.number_range + 1)}
    last_seen[1] = 19
    last_seen[2] = 12
    last_seen[3] = 1

    low_phase = {draw.draw_date: JackpotInfo(phase=0.0, draws_since_win=0, is_rollover=False) for draw in draws}
    high_phase = {draw.draw_date: JackpotInfo(phase=1.0, draws_since_win=100, is_rollover=True) for draw in draws}

    rank_low = make_jackpot_cycle_rank(low_phase, draw_dates, config)
    rank_high = make_jackpot_cycle_rank(high_phase, draw_dates, config)

    assert rank_low(counts, recent_deque, last_seen, 20, 25)[:10] != rank_high(counts, recent_deque, last_seen, 20, 25)[:10]


def test_bayesian_freq_rank_strength_affects_ordering():
    config = get_lottery_config("lotto_pl")
    counts = {n: 0 for n in range(1, config.number_range + 1)}
    counts[49] = 4
    counts[1] = 3
    counts[2] = 2
    recent_deque: deque[set[int]] = deque(maxlen=25)
    last_seen = {n: -1 for n in range(1, config.number_range + 1)}

    low_prior = make_bayesian_freq_rank(1.0, config.expected_rate, config)
    high_prior = make_bayesian_freq_rank(200.0, config.expected_rate, config)

    low_rank = low_prior(counts, recent_deque, last_seen, 5, 25)
    high_rank = high_prior(counts, recent_deque, last_seen, 5, 25)

    assert low_rank[0] == 49
    assert high_rank[0] == 49
    assert low_rank[:5] == high_rank[:5]


def test_drift_detector_responds_to_recent_bias():
    config = get_lottery_config("lotto_pl")
    counts = {n: 10 for n in range(1, config.number_range + 1)}
    last_seen = {n: 0 for n in range(1, config.number_range + 1)}
    recent_deque: deque[set[int]] = deque(maxlen=10)
    for _ in range(10):
        recent_deque.append({44, 45, 46, 47, 48, 49})

    slow = make_drift_detector_rank(0.01, config)
    fast = make_drift_detector_rank(0.2, config)

    slow_rank = slow(counts, recent_deque, last_seen, 100, 10)
    fast_rank = fast(counts, recent_deque, last_seen, 100, 10)

    assert set(fast_rank[:6]) == {44, 45, 46, 47, 48, 49}
    assert slow_rank[:10] != fast_rank[:10]


def test_anticorrelation_wrapper_reorders_conflicting_candidates():
    config = LotteryConfig(
        id="anti_corr",
        name="Anti Corr",
        numbers_drawn=5,
        number_range=10,
        ticket_size=5,
    )

    def base_rank_fn(counts, recent_deque, last_seen, draw_index, recent_window):
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    recent_deque: deque[set[int]] = deque(
        [
            {1, 2, 3, 4, 6},
            {1, 2, 3, 4, 6},
            {1, 2, 3, 4, 6},
            {5, 7, 8, 9, 10},
            {5, 7, 8, 9, 10},
            {5, 7, 8, 9, 10},
        ],
        maxlen=6,
    )
    wrapped = make_anticorrelation_wrapper(base_rank_fn, config, min_cooccurrence_ratio=0.9)

    ranked = wrapped({n: 1 for n in range(1, 11)}, recent_deque, {n: 0 for n in range(1, 11)}, 20, 6)

    assert ranked[:5] == [1, 2, 3, 4, 6]
    assert set(ranked) == set(range(1, 11))
