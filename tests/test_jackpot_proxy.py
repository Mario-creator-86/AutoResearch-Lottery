from __future__ import annotations

from datetime import date, timedelta

from generic_backtest import LotteryDraw
from jackpot_proxy import compute_proxy_jackpot
from lottery_config import get_lottery_config


def _make_draws(count: int, number_range: int = 80, numbers_drawn: int = 20) -> list[LotteryDraw]:
    import random

    rng = random.Random(42)
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


def test_compute_proxy_returns_phase_dict():
    config = get_lottery_config("multi_multi_pl")
    draws = _make_draws(200)

    result = compute_proxy_jackpot(draws, config)

    assert isinstance(result, dict)
    assert len(result) == len(draws)
    for info in result.values():
        assert 0.0 <= info.phase <= 1.0
        assert info.draws_since_win >= 0
        assert isinstance(info.is_rollover, bool)


def test_proxy_resets_keep_draws_since_win_non_negative():
    config = get_lottery_config("multi_multi_pl")
    draws = _make_draws(500)

    result = compute_proxy_jackpot(draws, config)
    ordered = [result[draw.draw_date].draws_since_win for draw in draws]

    assert min(ordered) >= 0
    assert max(ordered) > 0
    increments = [ordered[idx + 1] - ordered[idx] for idx in range(len(ordered) - 1)]
    positive_increments = [delta for delta in increments if delta > 0]
    assert positive_increments
    assert all(delta == 1 for delta in positive_increments)


def test_proxy_supports_small_lottery():
    config = get_lottery_config("lotto_pl")
    draws = _make_draws(200, number_range=49, numbers_drawn=6)

    result = compute_proxy_jackpot(draws, config)

    assert len(result) == len(draws)
    assert all(0.0 <= info.phase <= 1.0 for info in result.values())
