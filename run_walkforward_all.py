"""Run walkforward validation for all lotteries using their current best strategies.

This script loads each lottery's best strategy from autoloop_state.json,
builds the rank function, and runs walkforward validation.
"""
import sys
import json
import time
from pathlib import Path

# Add project root to path
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from auto_loop import SearchState, build_rank_fn_from_state, _params_from_state
from walkforward import run_and_save_walkforward
from load_data import load_lottery_draws
from lottery_config import get_lottery_config, LOTTERIES

def run_all_walkforward():
    results = {}
    for lottery_id in LOTTERIES:
        config = get_lottery_config(lottery_id)
        state_file = config.state_dir / "autoloop_state.json"

        if not state_file.exists():
            print(f"[{lottery_id}] No autoloop state, skipping")
            continue

        state_data = json.loads(state_file.read_text(encoding="utf-8"))
        search_state = SearchState.from_dict(state_data["search_state"])

        best_metric = search_state.best_metric
        best_strategy = search_state.best_strategy
        rw = search_state.rw

        print(f"\n[{lottery_id}] Running walkforward for {best_strategy} (backtest={best_metric:.6f}, rw={rw})")

        # Build rank function from current best state
        rank_fn = build_rank_fn_from_state(search_state, config=config)

        result = run_and_save_walkforward(
            lottery_id=lottery_id,
            rank_fn=rank_fn,
            recent_window=rw,
            backtest_score=best_metric,
            strategy_name=best_strategy,
        )

        if result:
            print(f"[{lottery_id}] wf_score={result.wf_score:.6f}, gap={result.gap:.6f}, "
                  f"baseline_edge={result.baseline_edge:.6f}, stability={result.stability:.4f}")
            results[lottery_id] = result.to_dict()
        else:
            print(f"[{lottery_id}] Walkforward failed (insufficient data)")

    print("\n\n=== WALKFORWARD SUMMARY ===")
    print(f"{'Lottery':<20} {'Backtest':>10} {'WF Score':>10} {'Gap':>8} {'Edge':>8} {'Stab':>6}")
    print("-" * 70)
    for lid, r in sorted(results.items()):
        print(f"{lid:<20} {r['backtest_score']:>10.6f} {r['wf_score']:>10.6f} "
              f"{r['gap']:>8.4f} {r['baseline_edge']:>8.4f} {r['stability']:>6.4f}")

    return results

if __name__ == "__main__":
    run_all_walkforward()