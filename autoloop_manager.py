"""Autoloop Manager - Runs experiments according to CLAUDE.md rules.

Rule: After each experiment:
1. Check if metric > current best in results.tsv
2. If KEEP: append to results.tsv, git commit
3. If DISCARD: append to results.tsv, git commit, revert train.py to last KEEP
4. Rinse, repeat
"""

import subprocess
import json
import sys
import time
import re
from pathlib import Path
from datetime import datetime

# Live thoughts integration - broadcast to OpenClawThink dashboard
try:
    from karpathy_live_thoughts import (
        think, analyze, action, solution, error, success, progress, info,
        thought_experiment_started, thought_experiment_complete,
        thought_new_best, thought_discard, thought_commit,
        thought_analyzing_results, thought_strategy_evolving
    )
    LIVE_THOUGHTS_ENABLED = True
except ImportError:
    LIVE_THOUGHTS_ENABLED = False


def run_command(cmd, cwd=None, timeout=300):
    """Run command and return stdout/stderr."""
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"


def get_current_best():
    """Get best average_hits from results.tsv."""
    results_file = Path("results.tsv")
    if not results_file.exists():
        return 2.500, "none", "none"  # Random baseline
    
    best = 2.500
    best_commit = "none"
    best_desc = "none"
    
    try:
        with open(results_file, "r") as f:
            # Skip header
            lines = f.readlines()[1:]
            for line in lines:
                parts = line.strip().split("\t")
                if len(parts) >= 9:
                    try:
                        metric = float(parts[1])
                        if metric > best:
                            best = metric
                            best_commit = parts[0]
                            best_desc = parts[8] if len(parts) > 8 else ""
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error reading results.tsv: {e}")
    
    return best, best_commit, best_desc


def get_last_keep_commit():
    """Get the last commit hash with KEEP status."""
    results_file = Path("results.tsv")
    if not results_file.exists():
        return None
    
    try:
        with open(results_file, "r") as f:
            lines = f.readlines()[::-1]  # Reverse order
            for line in lines:
                parts = line.strip().split("\t")
                if len(parts) >= 3 and parts[2] == "KEEP":
                    return parts[0]
    except:
        pass
    return None


def git_commit(message):
    """Commit current changes."""
    run_command("git add train.py results.tsv")
    rc, out, err = run_command(f'git commit -m "{message}"')
    if rc == 0:
        # Get commit hash
        rc2, out2, _ = run_command("git rev-parse --short HEAD")
        if rc2 == 0:
            return out2.strip()
    return None


def git_revert_to_commit(commit_hash):
    """Revert train.py to specific commit."""
    if commit_hash:
        rc, _, _ = run_command(f"git checkout {commit_hash} -- train.py")
        return rc == 0
    return False


def git_restore_train_py():
    """Restore train.py from HEAD."""
    rc, _, _ = run_command("git checkout HEAD -- train.py")
    return rc == 0


def run_experiment(experiment_name, experiment_config_fn):
    """Run a single experiment lifecycle."""
    print(f"\n{'='*60}")
    print(f"Starting experiment: {experiment_name}")
    print(f"{'='*60}\n")

    # 0. Broadcast experiment start
    if LIVE_THOUGHTS_ENABLED:
        thought_experiment_started(experiment_name)
        analyze(f"Current best metric from results.tsv")

    # 1. Get current best
    best_metric, best_commit, best_desc = get_current_best()
    print(f"Current best: {best_metric:.6f} from {best_desc[:50]}")
    if LIVE_THOUGHTS_ENABLED:
        info(f"Best so far: {best_metric:.6f}")

    # 2. Apply new experiment config to train.py
    print(f"Applying experiment config...")
    if LIVE_THOUGHTS_ENABLED:
        action(f"Applying experiment config to train.py")
    experiment_config_fn()

    # 3. Run experiment
    print(f"Running train.py (timeout 5 min)...")
    if LIVE_THOUGHTS_ENABLED:
        progress(f"Running train.py... (~5min timeout)")
    start = time.time()
    rc, stdout, stderr = run_command(
        ".venv/Scripts/python.exe train.py", timeout=300
    )
    elapsed = time.time() - start

    if rc != 0:
        print(f"ERROR: Experiment failed with code {rc}")
        print(f"stderr: {stderr[:500]}")
        if LIVE_THOUGHTS_ENABLED:
            error(f"Experiment FAILED with code {rc}")
        return None, "FAILED"

    # 4. Parse JSON output
    try:
        result = json.loads(stdout)
        metric = result.get("metric", 0)
    except json.JSONDecodeError:
        # Try to find JSON in output
        lines = stdout.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    result = json.loads(line)
                    metric = result.get("metric", 0)
                    break
                except:
                    pass
        else:
            print(f"ERROR: Could not parse JSON output")
            print(f"stdout: {stdout[:500]}")
            if LIVE_THOUGHTS_ENABLED:
                error("Could not parse JSON output from train.py")
            return None, "PARSE_ERROR"

    # 5. Determine status
    if metric > best_metric:
        status = "KEEP"
        improvement = ((metric - best_metric) / best_metric * 100) if best_metric > 0 else 0
        print(f"\n✓ NEW BEST! {metric:.6f} > {best_metric:.6f}")
        if LIVE_THOUGHTS_ENABLED:
            thought_new_best(result.get("strategy", experiment_name), metric, improvement)
    else:
        status = "DISCARD"
        reason = f"{metric:.6f} <= {best_metric:.6f}"
        print(f"\n✗ DISCARD: {metric:.6f} <= {best_metric:.6f}")
        if LIVE_THOUGHTS_ENABLED:
            thought_discard(reason, metric)

    # 6. Extract additional metrics
    strategy = result.get("strategy", experiment_name)
    hit2 = result.get("hit_rate_2_plus", 0)
    hit3 = result.get("hit_rate_3_plus", 0)
    portfolio = result.get("portfolio_avg", 0)
    best_config = result.get("best_config", "")

    # Broadcast experiment complete
    if LIVE_THOUGHTS_ENABLED:
        thought_experiment_complete(experiment_name, metric, status)

    # 7. Git commit first to get hash
    commit_msg = f"{experiment_name}: {best_config[:50]}"
    if LIVE_THOUGHTS_ENABLED:
        action(f"Git commit: {commit_msg[:40]}")
    commit_hash = git_commit(commit_msg)
    if commit_hash and LIVE_THOUGHTS_ENABLED:
        thought_commit(commit_hash, commit_msg)
    if not commit_hash:
        commit_hash = "auto"

    # 8. Write to results.tsv
    with open("results.tsv", "a") as f:
        row = f"{commit_hash}\t{metric:.6f}\t{status}\t{strategy}\t{hit2:.4f}\t{hit3:.4f}\t{portfolio:.6f}\t{elapsed:.2f}\t{experiment_name}: {best_config[:80]}"
        f.write(row + "\n")

    # 9. Amend commit with results.tsv
    run_command("git add results.tsv")
    run_command("git commit --amend --no-edit")

    # 10. If DISCARD, revert train.py to last KEEP
    if status == "DISCARD":
        last_keep = get_last_keep_commit()
        if last_keep:
            print(f"Reverting train.py to last KEEP: {last_keep}")
            if LIVE_THOUGHTS_ENABLED:
                info(f"Reverting to KEEP commit: {last_keep[:7]}")
            git_revert_to_commit(last_keep)
            # Commit the revert
            git_commit(f"Revert to {last_keep} after DISCARD")

    return result, status


def create_experiment_21():
    """Experiment 21: Portfolio-first optimization - optimize best-of-3."""
    
    train_content = '''
"""Autoresearch experiment - Portfolio-first optimization."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

LOTO_POLONIA_SRC = Path(os.environ.get("LOTO_POLONIA_SRC", str(Path(__file__).resolve().parent.parent / "Loto Polonia" / "src")))
if str(LOTO_POLONIA_SRC) not in sys.path:
    sys.path.insert(0, str(LOTO_POLONIA_SRC))

from load_data import load_draws
from loto_polonia.analysis.backtest import backtest_strategy


def run_experiment():
    draws = load_draws()
    
    # ========================================================================
    # EXPERIMENT ZONE
    # ========================================================================
    
    # Exp21: Portfolio-first optimization
    # Instead of optimizing average_hits on ticket 1, optimize portfolio_average_best_hits
    # The best-of-3-tickets metric might benefit from different weight combinations
    
    from loto_polonia.generators.top_frequency_generator import rank_pool_from_stats as _top_freq_rank
    from loto_polonia.generators.hot_cold_generator import rank_pool_from_stats as _hot_cold_rank
    from loto_polonia.generators.balanced_generator import rank_pool_from_stats as _bal_rank
    
    def sector_balanced_rank(counts, recent_deque, last_seen, draw_index, recent_window):
        max_count = max(counts.values()) or 1
        delays = {n: draw_index - last_seen.get(n, -1) if last_seen.get(n, -1) >= 0 else draw_index 
                  for n in range(1, 81)}
        max_delay = max(delays.values()) or 1
        base_scores = {n: 0.70 * (counts.get(n, 0) / max_count) + 0.30 * (delays[n] / max_delay) 
                       for n in range(1, 81)}
        sectors = {i: [] for i in range(8)}
        for n in range(1, 81):
            sectors[(n - 1) // 10].append((base_scores[n], n))
        for s in sectors:
            sectors[s].sort(reverse=True)
        result = []
        picked = set()
        for round_idx in range(10):
            for sec in range(8):
                if round_idx < len(sectors[sec]):
                    _, n = sectors[sec][round_idx]
                    if n not in picked:
                        result.append(n)
                        picked.add(n)
        for n in sorted(range(1, 81), key=lambda x: -base_scores[x]):
            if n not in picked:
                result.append(n)
        return result
    
    def make_ensemble(fns, weights):
        def rank_fn(counts, recent_deque, last_seen, draw_index, recent_window):
            weighted_ranks = {n: 0.0 for n in range(1, 81)}
            total_w = sum(weights)
            for fn, w in zip(fns, weights):
                ranked = fn(counts, recent_deque, last_seen, draw_index, recent_window)
                for pos, num in enumerate(ranked):
                    weighted_ranks[num] += (pos * w) / total_w
            return sorted(range(1, 81), key=lambda n: (weighted_ranks[n], n))
        return rank_fn
    
    base_strategies = [_top_freq_rank, _hot_cold_rank, _bal_rank, sector_balanced_rank]
    
    # Best from previous: [3.52, 2.037, 4.57, 1.45] with portfolio_avg ~3.687
    # Try optimizing for portfolio instead
    
    configs = []
    
    # Grid around best weights with focus on portfolio optimization
    import itertools
    best_weights = [3.52, 2.037, 4.57, 1.45]
    
    # Test different combinations specifically for portfolio
    for tf in [3.0, 3.5, 4.0, 4.5]:
        for hc in [1.5, 2.0, 2.5]:
            for bal in [4.0, 4.5, 5.0]:
                for sb in [1.0, 1.5, 2.0]:
                    w = [tf, hc, bal, sb]
                    name = f"port_tf{tf}_hc{hc}_bal{bal}_sb{sb}"
                    configs.append((name, w))
    
    best_portfolio = 0
    best_config = None
    best_result = None
    
    for name, weights in configs:
        rank_fn = make_ensemble(base_strategies, weights)
        r = backtest_strategy(name, rank_fn, draws, ticket_size=10, recent_window=200, initial_window=40)
        
        # Prioritize portfolio_average_best_hits
        portfolio_score = r.portfolio_average_best_hits
        
        if portfolio_score > best_portfolio:
            best_portfolio = portfolio_score
            best_config = name
            best_result = r
    
    # Restore the weights that gave best portfolio to optimize average_hits
    best_metric = best_result.average_hits if best_result else 0
    
    return {
        "metric": round(best_metric, 6),
        "strategy": best_result.strategy_key if best_result else "none",
        "hit_rate_2_plus": round(best_result.hit_rate_2_plus, 4) if best_result else 0,
        "hit_rate_3_plus": round(best_result.hit_rate_3_plus, 4) if best_result else 0,
        "portfolio_avg": round(best_portfolio, 6),
        "evaluated_draws": best_result.evaluated_draws if best_result else 0,
        "best_config": best_config,
    }


if __name__ == "__main__":
    start = time.time()
    result = run_experiment()
    result["elapsed_seconds"] = round(time.time() - start, 2)
    print(json.dumps(result, indent=2))
'''
    
    with open("train.py", "w") as f:
        f.write(train_content.strip())


# Main loop
def main():
    """Run experiments continuously."""
    experiment_num = 21
    
    while True:
        print(f"\n{'#'*60}")
        print(f"EXPERIMENT {experiment_num}")
        print(f"{'#'*60}")
        
        if experiment_num == 21:
            result, status = run_experiment(
                f"Exp{experiment_num}",
                create_experiment_21
            )
        else:
            # For future experiments
            print(f"Experiment {experiment_num} not defined yet")
            break
        
        if result:
            print(f"\nResult: {result}")
            print(f"Status: {status}")
        
        experiment_num += 1
        
        # Small delay between experiments
        time.sleep(2)


if __name__ == "__main__":
    main()
