"""Walkforward validation for lottery strategy optimization.

Anchored walkforward with 5 folds:
1. Split draws into 60% base train + 40% test (divided into 5 equal folds)
2. For each fold, train window = draw[0 .. fold_start], test = fold draws
3. Evaluate the current best strategy on the unseen test window
4. Aggregate metrics: wf_score, gap, stability, baseline_edge, per-fold breakdown
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from atomic_io import write_text_atomic
from generic_backtest import (
    LotteryDraw,
    StrategyBacktestResult,
    backtest_strategy,
    number_domain,
)
from load_data import load_lottery_draws
from lottery_config import LotteryConfig, get_lottery_config, ensure_lottery_storage

NUM_FOLDS = 5
TRAIN_RATIO = 0.60  # First 60% is minimum train, rest split into folds


@dataclass
class FoldResult:
    fold: int
    train_size: int
    test_size: int
    score: float  # average_hits on test window
    baseline: float  # expected random score on test window
    hit_rate_2plus: float
    hit_rate_3plus: float
    portfolio_avg: float

    def to_dict(self) -> dict:
        return {
            "fold": self.fold,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "score": round(self.score, 6),
            "baseline": round(self.baseline, 6),
            "hit_rate_2plus": round(self.hit_rate_2plus, 4),
            "hit_rate_3plus": round(self.hit_rate_3plus, 4),
            "portfolio_avg": round(self.portfolio_avg, 6),
        }


@dataclass
class WalkforwardResult:
    lottery_id: str
    strategy: str
    backtest_score: float
    wf_score: float
    gap: float
    stability: float
    baseline_avg: float
    baseline_edge: float
    folds: list[FoldResult]
    timestamp: str
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return {
            "lottery_id": self.lottery_id,
            "strategy": self.strategy,
            "backtest_score": round(self.backtest_score, 6),
            "wf_score": round(self.wf_score, 6),
            "gap": round(self.gap, 6),
            "stability": round(self.stability, 4),
            "baseline_avg": round(self.baseline_avg, 6),
            "baseline_edge": round(self.baseline_edge, 6),
            "folds": [f.to_dict() for f in self.folds],
            "timestamp": self.timestamp,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "num_folds": len(self.folds),
        }


def _compute_baseline(config: LotteryConfig) -> float:
    """Expected average hits for a random ticket."""
    return config.ticket_size * config.numbers_drawn / config.number_range


def _backtest_on_window(
    config: LotteryConfig,
    rank_fn,
    train_draws: list[LotteryDraw],
    test_draws: list[LotteryDraw],
    recent_window: int,
) -> StrategyBacktestResult:
    """Run backtest using train_draws as warmup and evaluate only on test_draws.

    We concatenate train + test draws and set initial_window = len(train),
    so backtest_strategy will train on all train_draws and evaluate on test_draws.
    """
    all_draws = list(train_draws) + list(test_draws)
    initial_window = len(train_draws)
    return backtest_strategy(
        config=config,
        strategy_key="walkforward",
        rank_fn=rank_fn,
        draws=all_draws,
        ticket_size=config.ticket_size,
        recent_window=recent_window,
        initial_window=initial_window,
    )


def run_walkforward(
    lottery_id: str,
    rank_fn,
    recent_window: int = 200,
    backtest_score: float = 0.0,
    strategy_name: str = "",
) -> WalkforwardResult | None:
    """Run anchored walkforward validation on a lottery.

    Args:
        lottery_id: Lottery config ID
        rank_fn: The ranking function to evaluate (from build_rank_fn_from_state)
        recent_window: Recent window parameter for the strategy
        backtest_score: The full backtest score (for gap calculation)
        strategy_name: Name of the strategy being evaluated

    Returns:
        WalkforwardResult or None if insufficient data
    """
    config = get_lottery_config(lottery_id)
    draws = load_lottery_draws(lottery_id)

    if not draws or len(draws) < 50:
        return None

    start_time = time.time()
    total = len(draws)

    # Split: 60% minimum train, 40% divided into 5 folds
    train_end = int(total * TRAIN_RATIO)
    test_draws_all = draws[train_end:]
    test_size = len(test_draws_all)

    if test_size < NUM_FOLDS * 5:  # Need at least 5 draws per fold
        return None

    fold_size = test_size // NUM_FOLDS
    baseline = _compute_baseline(config)

    fold_results: list[FoldResult] = []

    for fold_idx in range(NUM_FOLDS):
        fold_start = fold_idx * fold_size
        fold_end = fold_start + fold_size if fold_idx < NUM_FOLDS - 1 else test_size
        test_fold = test_draws_all[fold_start:fold_end]

        # Anchored: train window grows — all draws up to this fold's start
        train_window = draws[:train_end + fold_start]

        if len(train_window) < 10 or len(test_fold) < 2:
            continue

        result = _backtest_on_window(
            config=config,
            rank_fn=rank_fn,
            train_draws=train_window,
            test_draws=test_fold,
            recent_window=recent_window,
        )

        fold_results.append(FoldResult(
            fold=fold_idx + 1,
            train_size=len(train_window),
            test_size=len(test_fold),
            score=result.average_hits,
            baseline=baseline,
            hit_rate_2plus=result.hit_rate_2_plus,
            hit_rate_3plus=result.hit_rate_3_plus,
            portfolio_avg=result.portfolio_average_best_hits,
        ))

    if not fold_results:
        return None

    # Aggregate metrics
    scores = [f.score for f in fold_results]
    evaluated_draws = sum(f.test_size for f in fold_results)
    wf_score = sum(f.score * f.test_size for f in fold_results) / evaluated_draws
    baseline_avg = sum(f.baseline * f.test_size for f in fold_results) / evaluated_draws
    gap = backtest_score - wf_score if backtest_score > 0 else 0.0
    baseline_edge = wf_score - baseline_avg

    # Stability: 1 - CV (coefficient of variation). Higher = more stable.
    if wf_score > 0 and len(scores) > 1:
        fold_mean = sum(scores) / len(scores)
        std_dev = math.sqrt(sum((s - fold_mean) ** 2 for s in scores) / (len(scores) - 1))
        stability = max(0.0, min(1.0, 1.0 - std_dev / fold_mean))
    else:
        stability = 0.0

    elapsed = time.time() - start_time

    return WalkforwardResult(
        lottery_id=lottery_id,
        strategy=strategy_name,
        backtest_score=backtest_score,
        wf_score=wf_score,
        gap=gap,
        stability=stability,
        baseline_avg=baseline_avg,
        baseline_edge=baseline_edge,
        folds=fold_results,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        elapsed_seconds=elapsed,
    )


def save_walkforward_result(config: LotteryConfig, result: WalkforwardResult) -> Path:
    """Save walkforward result to the lottery's data directory."""
    ensure_lottery_storage(config)
    wf_file = config.state_dir / "walkforward.json"
    write_text_atomic(
        wf_file,
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return wf_file


def load_walkforward_result(config: LotteryConfig) -> dict | None:
    """Load walkforward result from the lottery's data directory."""
    wf_file = config.state_dir / "walkforward.json"
    if not wf_file.exists():
        return None
    try:
        return json.loads(wf_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def run_and_save_walkforward(
    lottery_id: str,
    rank_fn,
    recent_window: int = 200,
    backtest_score: float = 0.0,
    strategy_name: str = "",
) -> WalkforwardResult | None:
    """Run walkforward and save results. Convenience wrapper."""
    config = get_lottery_config(lottery_id)
    result = run_walkforward(
        lottery_id=lottery_id,
        rank_fn=rank_fn,
        recent_window=recent_window,
        backtest_score=backtest_score,
        strategy_name=strategy_name,
    )
    if result:
        save_walkforward_result(config, result)
    return result
