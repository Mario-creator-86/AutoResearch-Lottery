from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date
from math import sqrt
from typing import TYPE_CHECKING, Callable

from lottery_config import LotteryConfig

if TYPE_CHECKING:
    from jackpot_proxy import JackpotInfo

BACKTEST_WEIGHTS = {"frequency": 0.75, "recent": 0.05, "delay": 0.20}

RankFn = Callable[[dict[int, int], deque[set[int]], dict[int, int], int, int], list[int]]


@dataclass(frozen=True, slots=True)
class LotteryDraw:
    draw_date: date
    draw_time: str | None
    draw_identifier: str | None
    numbers: tuple[int, ...]
    bonuses: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class StrategyBacktestResult:
    strategy_key: str
    ticket_size: int
    recent_window: int
    initial_window: int
    evaluated_draws: int
    proposed_numbers: tuple[int, ...]
    average_hits: float
    hit_rate_1_plus: float
    hit_rate_2_plus: float
    hit_rate_3_plus: float
    best_hit_count: int
    best_draw_date: str | None
    last_hit_count: int
    last_tested_draw_date: str | None
    portfolio_average_best_hits: float = 0.0
    portfolio_rate_2_plus: float = 0.0
    portfolio_rate_3_plus: float = 0.0
    bonus_average_hits: float = 0.0
    bonus_hit_rate_1_plus: float = 0.0


def number_domain(config: LotteryConfig) -> range:
    return range(1, config.number_range + 1)


def top_frequency_rank(config: LotteryConfig, counts, recent_deque, last_seen, draw_index, recent_window) -> list[int]:
    return sorted(number_domain(config), key=lambda n: (-counts.get(n, 0), n))


def hot_cold_rank(config: LotteryConfig, counts, recent_deque, last_seen, draw_index, recent_window) -> list[int]:
    recent_counts = {n: 0 for n in number_domain(config)}
    for draw in recent_deque:
        for number in draw:
            recent_counts[number] += 1
    max_recent = max(recent_counts.values()) or 1
    delays = {}
    for number in number_domain(config):
        last = last_seen.get(number, -1)
        delays[number] = draw_index - last if last >= 0 else draw_index
    max_delay = max(delays.values()) or 1
    scores = {}
    for number in number_domain(config):
        hotness = recent_counts[number] / max_recent
        coldness = delays[number] / max_delay
        scores[number] = 0.55 * hotness + 0.45 * coldness
    return sorted(number_domain(config), key=lambda n: (-scores[n], n))


def balanced_rank(config: LotteryConfig, counts, recent_deque, last_seen, draw_index, recent_window,
                  *, weights: dict[str, float] | None = None) -> list[int]:
    weights = BACKTEST_WEIGHTS if weights is None else weights
    max_count = max(counts.values()) or 1
    recent_counts = {n: 0 for n in number_domain(config)}
    for draw in recent_deque:
        for number in draw:
            recent_counts[number] += 1
    max_recent = max(recent_counts.values()) or 1
    delays = {}
    for number in number_domain(config):
        last = last_seen.get(number, -1)
        delays[number] = draw_index - last if last >= 0 else draw_index
    max_delay = max(delays.values()) or 1
    scores = {}
    for number in number_domain(config):
        scores[number] = (
            weights["frequency"] * (counts.get(number, 0) / max_count)
            + weights["recent"] * (recent_counts[number] / max_recent)
            + weights["delay"] * (delays[number] / max_delay)
        )
    return sorted(number_domain(config), key=lambda n: (-scores[n], n))


def make_weighted_ensemble(rank_functions: list[RankFn], weights: list[float], config: LotteryConfig) -> RankFn:
    def rank_fn(counts, recent_deque, last_seen, draw_index, recent_window):
        weighted_ranks = {n: 0.0 for n in number_domain(config)}
        total_weight = sum(weights) or 1.0
        for fn, weight in zip(rank_functions, weights):
            ranked = fn(counts, recent_deque, last_seen, draw_index, recent_window)
            for pos, number in enumerate(ranked):
                weighted_ranks[number] += (pos * weight) / total_weight
        return sorted(number_domain(config), key=lambda n: (weighted_ranks[n], n))

    return rank_fn


def make_jackpot_cycle_rank(
    jackpot_data: dict[date, JackpotInfo],
    draw_dates: list[date],
    config: LotteryConfig,
) -> RankFn:
    def rank_fn(counts, recent_deque, last_seen, draw_index, recent_window):
        recent_counts = {n: 0 for n in number_domain(config)}
        for draw in recent_deque:
            for number in draw:
                recent_counts[number] += 1

        max_count = max(counts.values()) or 1
        max_recent = max(recent_counts.values()) or 1
        delays = {}
        for number in number_domain(config):
            last = last_seen.get(number, -1)
            delays[number] = draw_index - last if last >= 0 else draw_index
        max_delay = max(delays.values()) or 1

        lookup_index = min(draw_index - 1, len(draw_dates) - 1)
        if 0 <= draw_index < len(draw_dates):
            # Daily metadata may include later draws from the same date.
            while lookup_index >= 0 and draw_dates[lookup_index] >= draw_dates[draw_index]:
                lookup_index -= 1
        draw_date = draw_dates[lookup_index] if lookup_index >= 0 else None
        info = jackpot_data.get(draw_date) if draw_date is not None else None
        phase = info.phase if info is not None else 0.5

        freq_weight, delay_weight, recent_weight = _phase_weights(phase)
        scores = {}
        for number in number_domain(config):
            scores[number] = (
                freq_weight * (counts.get(number, 0) / max_count)
                + delay_weight * (delays[number] / max_delay)
                + recent_weight * (recent_counts[number] / max_recent)
            )
        return sorted(number_domain(config), key=lambda n: (-scores[n], n))

    return rank_fn


def make_bayesian_freq_rank(
    prior_strength: float,
    expected_rate: float,
    config: LotteryConfig,
) -> RankFn:
    prior_strength = max(float(prior_strength), 1e-6)
    expected_rate = min(max(float(expected_rate), 1e-6), 1 - 1e-6)
    alpha0 = prior_strength * expected_rate
    beta0 = prior_strength * (1.0 - expected_rate)

    def rank_fn(counts, recent_deque, last_seen, draw_index, recent_window):
        total_draws = max(draw_index, 1)
        posterior_mean = {}
        for number in number_domain(config):
            posterior_mean[number] = (alpha0 + counts.get(number, 0)) / (alpha0 + beta0 + total_draws)
        return sorted(number_domain(config), key=lambda n: (-posterior_mean[n], n))

    return rank_fn


def make_drift_detector_rank(
    ewma_lambda: float,
    config: LotteryConfig,
) -> RankFn:
    ewma_lambda = min(max(float(ewma_lambda), 0.001), 0.5)
    ewma = {number: 0.0 for number in number_domain(config)}
    last_processed_index = -1

    def rank_fn(counts, recent_deque, last_seen, draw_index, recent_window):
        nonlocal last_processed_index, ewma

        def apply_draw(draw_numbers: set[int]) -> None:
            for number in number_domain(config):
                ewma[number] *= 1.0 - ewma_lambda
            for number in draw_numbers:
                ewma[number] += ewma_lambda

        if draw_index <= last_processed_index:
            ewma = {number: 0.0 for number in number_domain(config)}
            last_processed_index = -1

        if last_processed_index < 0:
            for draw_numbers in recent_deque:
                apply_draw(draw_numbers)
        elif draw_index == last_processed_index + 1 and recent_deque:
            apply_draw(recent_deque[-1])
        else:
            ewma = {number: 0.0 for number in number_domain(config)}
            for draw_numbers in recent_deque:
                apply_draw(draw_numbers)
        last_processed_index = draw_index

        effective_window = max(len(recent_deque), 1)
        long_term = {
            number: counts.get(number, 0) / max(draw_index, 1)
            for number in number_domain(config)
        }
        scores = {}
        for number in number_domain(config):
            baseline = long_term[number]
            drift = abs(ewma[number] - baseline)
            sigma = sqrt(max(baseline * (1.0 - baseline) / effective_window, 1e-9))
            if drift > 2.0 * sigma:
                scores[number] = ewma[number] + drift
            else:
                scores[number] = baseline
        return sorted(number_domain(config), key=lambda n: (-scores[n], n))

    return rank_fn


def make_anticorrelation_wrapper(
    base_rank_fn: RankFn,
    config: LotteryConfig,
    min_cooccurrence_ratio: float = 0.8,
) -> RankFn:
    threshold = max(float(min_cooccurrence_ratio), 0.0)
    candidate_pool_size = min(config.number_range, max(config.ticket_size * 3, 30))

    def rank_fn(counts, recent_deque, last_seen, draw_index, recent_window):
        ranked = _fill_ranked_pool(config, base_rank_fn(counts, recent_deque, last_seen, draw_index, recent_window))
        candidate_pool = ranked[:candidate_pool_size]
        if len(candidate_pool) <= config.ticket_size or not recent_deque:
            return ranked

        appearance = {number: 0 for number in candidate_pool}
        cooccurrence: dict[tuple[int, int], int] = {}
        window_size = len(recent_deque)
        for draw in recent_deque:
            present = [number for number in candidate_pool if number in draw]
            for number in present:
                appearance[number] += 1
            for idx, left in enumerate(present):
                for right in present[idx + 1:]:
                    pair = (left, right) if left < right else (right, left)
                    cooccurrence[pair] = cooccurrence.get(pair, 0) + 1

        selected: list[int] = []
        deferred: list[int] = []
        for candidate in candidate_pool:
            conflicts = 0
            for chosen in selected:
                pair = (candidate, chosen) if candidate < chosen else (chosen, candidate)
                observed = cooccurrence.get(pair, 0)
                expected = (appearance[candidate] * appearance[chosen]) / max(window_size, 1)
                if expected > 0 and observed < expected * threshold:
                    conflicts += 1
            if conflicts > 2 and len(deferred) < candidate_pool_size:
                deferred.append(candidate)
            else:
                selected.append(candidate)

        reranked = _fill_ranked_pool(config, selected + deferred + ranked[candidate_pool_size:])
        return reranked

    return rank_fn


def _phase_weights(phase: float) -> tuple[float, float, float]:
    phase = min(max(float(phase), 0.0), 1.0)
    anchors = (
        (0.0, (0.80, 0.20, 0.00)),
        (0.5, (0.60, 0.25, 0.15)),
        (1.0, (0.50, 0.40, 0.10)),
    )
    if phase <= anchors[1][0]:
        return _interpolate_phase_triplet(anchors[0], anchors[1], phase)
    return _interpolate_phase_triplet(anchors[1], anchors[2], phase)


def _interpolate_phase_triplet(
    left: tuple[float, tuple[float, float, float]],
    right: tuple[float, tuple[float, float, float]],
    phase: float,
) -> tuple[float, float, float]:
    left_phase, left_values = left
    right_phase, right_values = right
    if right_phase <= left_phase:
        return left_values
    ratio = (phase - left_phase) / (right_phase - left_phase)
    return tuple(
        left_value + (right_value - left_value) * ratio
        for left_value, right_value in zip(left_values, right_values)
    )


def backtest_strategy(
    config: LotteryConfig,
    strategy_key: str,
    rank_fn: RankFn,
    draws: list[LotteryDraw],
    ticket_size: int | None = None,
    recent_window: int = 25,
    initial_window: int = 10,
) -> StrategyBacktestResult:
    if ticket_size is None:
        ticket_size = config.ticket_size
    for name, value in (("ticket_size", ticket_size), ("recent_window", recent_window),
                        ("initial_window", initial_window)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if ticket_size > config.number_range:
        raise ValueError("ticket_size cannot exceed number_range")
    if len(draws) <= initial_window:
        raise ValueError("Prea putine extrageri pentru backtest.")

    counts = {n: 0 for n in number_domain(config)}
    last_seen = {n: -1 for n in number_domain(config)}
    recent_deque: deque[set[int]] = deque(maxlen=recent_window)

    has_bonus = config.bonus_count > 0 and config.bonus_range is not None
    bonus_domain = range(1, (config.bonus_range or 0) + 1)
    bonus_counts: dict[int, int] = {n: 0 for n in bonus_domain} if has_bonus else {}

    for idx, draw in enumerate(draws[:initial_window]):
        draw_set = set(draw.numbers)
        for number in draw.numbers:
            counts[number] += 1
            last_seen[number] = idx
        recent_deque.append(draw_set)
        if has_bonus:
            for bonus in draw.bonuses:
                if bonus in bonus_counts:
                    bonus_counts[bonus] += 1

    hit_counts: list[int] = []
    bonus_hit_counts: list[int] = []
    portfolio_best_hits: list[int] = []
    best_hit_count = -1
    best_draw_date: str | None = None
    last_hit_count = 0
    last_tested_draw_date: str | None = None

    for next_idx in range(initial_window, len(draws)):
        ranked = rank_fn(counts, recent_deque, last_seen, next_idx, recent_window)
        ranked = _fill_ranked_pool(config, ranked)
        tickets = [
            set(ranked[index * ticket_size:(index + 1) * ticket_size])
            for index in range(3)
        ]

        next_draw = draws[next_idx]
        next_set = set(next_draw.numbers)
        ticket_hits = [len(ticket & next_set) for ticket in tickets]

        hit_counts.append(ticket_hits[0])
        portfolio_best_hits.append(max(ticket_hits))

        if has_bonus and next_draw.bonuses:
            bonus_ranked = sorted(bonus_domain, key=lambda n: (-bonus_counts.get(n, 0), n))
            predicted_bonus = set(bonus_ranked[:config.bonus_count])
            actual_bonus = set(next_draw.bonuses)
            bonus_hit_counts.append(len(predicted_bonus & actual_bonus))

        if ticket_hits[0] > best_hit_count:
            best_hit_count = ticket_hits[0]
            best_draw_date = next_draw.draw_date.isoformat()
        last_hit_count = ticket_hits[0]
        last_tested_draw_date = next_draw.draw_date.isoformat()

        draw_set = set(next_draw.numbers)
        for number in next_draw.numbers:
            counts[number] += 1
            last_seen[number] = next_idx
        recent_deque.append(draw_set)
        if has_bonus:
            for bonus in next_draw.bonuses:
                if bonus in bonus_counts:
                    bonus_counts[bonus] += 1

    denom = len(hit_counts) or 1
    bonus_denom = len(bonus_hit_counts) or 1
    proposed = _fill_ranked_pool(config, rank_fn(counts, recent_deque, last_seen, len(draws), recent_window))
    return StrategyBacktestResult(
        strategy_key=strategy_key,
        ticket_size=ticket_size,
        recent_window=recent_window,
        initial_window=initial_window,
        evaluated_draws=len(hit_counts),
        proposed_numbers=tuple(proposed[:ticket_size]),
        average_hits=sum(hit_counts) / denom,
        hit_rate_1_plus=sum(1 for hits in hit_counts if hits >= 1) / denom,
        hit_rate_2_plus=sum(1 for hits in hit_counts if hits >= 2) / denom,
        hit_rate_3_plus=sum(1 for hits in hit_counts if hits >= 3) / denom,
        best_hit_count=max(best_hit_count, 0),
        best_draw_date=best_draw_date,
        last_hit_count=last_hit_count,
        last_tested_draw_date=last_tested_draw_date,
        portfolio_average_best_hits=sum(portfolio_best_hits) / denom,
        portfolio_rate_2_plus=sum(1 for hits in portfolio_best_hits if hits >= 2) / denom,
        portfolio_rate_3_plus=sum(1 for hits in portfolio_best_hits if hits >= 3) / denom,
        bonus_average_hits=sum(bonus_hit_counts) / bonus_denom if bonus_hit_counts else 0.0,
        bonus_hit_rate_1_plus=sum(1 for h in bonus_hit_counts if h >= 1) / bonus_denom if bonus_hit_counts else 0.0,
    )


def _fill_ranked_pool(config: LotteryConfig, ranked: list[int]) -> list[int]:
    unique = []
    seen = set()
    for number in ranked:
        if isinstance(number, bool):
            continue
        try:
            normalized = int(number)
        except (TypeError, ValueError, OverflowError):
            continue
        if not isinstance(number, str) and normalized != number:
            continue
        number = normalized
        if number in seen:
            continue
        if not 1 <= number <= config.number_range:
            continue
        unique.append(number)
        seen.add(number)
    for number in number_domain(config):
        if number not in seen:
            unique.append(number)
    return unique
