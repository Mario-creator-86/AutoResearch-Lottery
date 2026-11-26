"""Run walkforward for multi_multi_pl using the hot_pairs strategy (current best at 2.557726).
This validates whether the hot_pairs edge holds out-of-sample.
"""
import sys
import json
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from walkforward import run_and_save_walkforward
from lottery_config import get_lottery_config

LOTTERY_ID = "multi_multi_pl"
BEST_SCORE = 2.557726
STRATEGY_NAME = "hot_pairs"


def make_hot_pairs_rank(pair_window=100, freq_weight=0.7, pair_weight=0.3):
    """The current best strategy: pair co-occurrence."""
    def rank(c, rd, l, di, rw):
        mc = max(c.values()) if c else 1
        freq_scores = {n: c.get(n, 0) / mc for n in range(1, 81)}

        pw = min(pair_window, len(rd))
        recent_list = list(rd)[-pw:]
        pair_counts = defaultdict(int)
        for draw_set in recent_list:
            nums = sorted(draw_set)
            for i in range(len(nums)):
                for j in range(i+1, len(nums)):
                    pair_counts[(nums[i], nums[j])] += 1

        top_freq = sorted(range(1, 81), key=lambda n: -freq_scores[n])[:20]
        pair_scores = {}
        for n in range(1, 81):
            ps = 0
            for t in top_freq:
                if t == n:
                    continue
                key = (min(n, t), max(n, t))
                ps += pair_counts.get(key, 0)
            pair_scores[n] = ps

        mp = max(pair_scores.values()) if pair_scores else 1
        if mp == 0:
            mp = 1

        scores = {}
        for n in range(1, 81):
            scores[n] = freq_weight * freq_scores[n] + pair_weight * pair_scores[n] / mp

        return sorted(range(1, 81), key=lambda n: -scores[n])
    return rank


def main():
    config = get_lottery_config(LOTTERY_ID)

    print(f"Running walkforward for {LOTTERY_ID} with strategy={STRATEGY_NAME} (backtest={BEST_SCORE})")

    rank_fn = make_hot_pairs_rank(pair_window=100, freq_weight=0.7, pair_weight=0.3)

    result = run_and_save_walkforward(
        lottery_id=LOTTERY_ID,
        rank_fn=rank_fn,
        recent_window=200,
        backtest_score=BEST_SCORE,
        strategy_name=STRATEGY_NAME,
    )

    if result:
        print(f"\nWalkforward results for {STRATEGY_NAME}:")
        print(f"  WF Score: {result.wf_score:.6f}")
        print(f"  Gap (backtest - wf): {result.gap:.6f}")
        print(f"  Baseline edge (wf - baseline): {result.baseline_edge:.6f}")
        print(f"  Stability: {result.stability:.4f}")
        print(f"  Folds:")
        for f in result.folds:
            print(f"    Fold {f.fold}: score={f.score:.6f}, baseline={f.baseline:.4f}, "
                  f"hit2+={f.hit_rate_2plus:.4f}, hit3+={f.hit_rate_3plus:.4f}")
    else:
        print("Walkforward failed!")


if __name__ == "__main__":
    main()