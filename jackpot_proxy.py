"""Proxy jackpot cycle estimation from draw history."""

from __future__ import annotations

from bisect import insort
from dataclasses import dataclass
from datetime import date

from generic_backtest import LotteryDraw
from lottery_config import LotteryConfig


DEFAULT_CYCLE_LENGTH = 50
MIN_CALIBRATION_SCORES = 10


@dataclass(frozen=True, slots=True)
class JackpotInfo:
    """Jackpot cycle metadata aligned to a draw date."""

    phase: float
    draws_since_win: int
    is_rollover: bool
    jackpot_amount: float = 0.0


def compute_proxy_jackpot(
    draws: list[LotteryDraw],
    config: LotteryConfig,
    lookback: int = DEFAULT_CYCLE_LENGTH,
    percentile: float = 95.0,
) -> dict[date, JackpotInfo]:
    """Infer jackpot-cycle-like phases from number-distribution surprise."""

    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
        raise ValueError("lookback must be a positive integer")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    expected_count = lookback * config.expected_rate
    frequencies = [0] * (config.number_range + 1)
    sorted_scores: list[float] = []
    completed_cycles: list[int] = []
    last_reset: int | None = None
    draws_since_win = 0
    result: dict[date, JackpotInfo] = {}

    for index, draw in enumerate(draws):
        reset = False
        calibrated = len(sorted_scores) >= MIN_CALIBRATION_SCORES
        if index >= lookback:
            score = sum((frequencies[n] - expected_count) ** 2 / max(expected_count, 0.01)
                        for n in range(1, config.number_range + 1))
            if calibrated:
                threshold_index = int((len(sorted_scores) - 1) * percentile / 100.0)
                reset = score >= sorted_scores[threshold_index]
            insort(sorted_scores, score)

        if reset:
            if last_reset is not None:
                insort(completed_cycles, index - last_reset)
            last_reset = index
            draws_since_win = 0
        elif index:
            draws_since_win += 1
        median_cycle = (completed_cycles[len(completed_cycles) // 2]
                        if completed_cycles else lookback)
        result[draw.draw_date] = JackpotInfo(
            phase=min(draws_since_win / median_cycle, 1.0) if calibrated else 0.5,
            draws_since_win=draws_since_win,
            is_rollover=draws_since_win > 0,
        )

        # Keep only past draws in the next prediction's frequency window.
        for number in draw.numbers:
            frequencies[number] += 1
        if index >= lookback:
            for number in draws[index - lookback].numbers:
                frequencies[number] -= 1

    return result
