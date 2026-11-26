from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import significance as sig


def test_hypergeometric_variance_matches_formula():
    # Loto 6/49, bilet de 6
    v = sig.ticket_hits_variance(49, 6, 6)
    assert math.isclose(v, 6 * (6 / 49) * (43 / 49) * (43 / 48), rel_tol=1e-12)
    # Kaskada: 12 din 24, bilet de 12
    assert math.isclose(sig.ticket_hits_variance(24, 12, 12), 12 * 0.5 * 0.5 * (12 / 23))


def test_sidak_adjustment():
    assert math.isclose(sig.sidak_adjust(0.01, 1), 0.01)
    assert math.isclose(sig.sidak_adjust(0.01, 100), 1 - 0.99 ** 100, rel_tol=1e-9)
    assert sig.sidak_adjust(1e-12, 10) < 1e-10
    assert sig.sidak_adjust(1.0, 5) == 1.0


def test_same_edge_becomes_noise_after_many_experiments():
    common = dict(metric=0.7865, baseline=6 * 6 / 49, number_range=49, numbers_drawn=6,
                  ticket_size=6, n_draws=2263)
    single = sig.edge_stats(**common, n_experiments=1)
    searched = sig.edge_stats(**common, n_experiments=50_000)
    assert single["z"] > 3
    assert single["verdict"] == "significant"
    assert searched["verdict"] == "noise"
    assert searched["noise_max_z"] > single["z"]


def test_huge_real_edge_stays_significant():
    stats = sig.edge_stats(metric=0.90, baseline=0.7347, number_range=49, numbers_drawn=6,
                           ticket_size=6, n_draws=5000, n_experiments=100_000)
    assert stats["verdict"] == "significant"


def test_count_experiments_sums_configs(tmp_path):
    f = tmp_path / "results.tsv"
    f.write_text(
        "commit\taverage_hits\tstatus\tstrategy\th2\th3\tpa\te\tdescription\n"
        "auto\t0.7\tKEEP\ts\t0\t0\t0\t1\tAutoLoop phase1 iter1 KEEP: s -> 0.7 (120 configs, radius=0.5)\n"
        "auto\t0.6\tDISCARD\ts\t0\t0\t0\t1\tAutoLoop phase2 iter2 DISCARD: no improvement (30 configs, radius=0.4)\n"
        "abc\t0.5\tDISCARD\tmanual\t0\t0\t0\t1\tmanual experiment\n",
        encoding="utf-8",
    )
    assert sig.count_experiments(f) == 151


def test_prospective_stats_are_insufficient_with_few_draws(monkeypatch):
    import ai_tickets_store
    from lottery_config import get_lottery_config

    store = {"entries": [{"status": "checked", "results": [
        {"main": [1, 2, 3, 4, 5, 6], "hit_count": 1},
        {"main": [7, 8, 9, 10, 11, 12], "hit_count": 0},
    ]}]}
    monkeypatch.setattr(ai_tickets_store, "load_ticket_store", lambda lid: store)
    stats = sig.prospective_ticket_stats("loto_6_49_ro", get_lottery_config("loto_6_49_ro"))
    assert stats["draws_checked"] == 1
    assert stats["mean_hits_per_ticket"] == 0.5
    assert stats["verdict"] == "insufficient"


def test_plugin_holdout_evaluates_only_recent_draws(tmp_path):
    import random
    from datetime import date, timedelta

    import ai_strategy_lab
    from generic_backtest import LotteryDraw

    plugin = tmp_path / "zfreq_test.py"
    plugin.write_text(
        'NAME = "zfreqtest"\nDESCRIPTION = "freq"\nSOURCE = "test"\n\n'
        "def create(config):\n"
        "    def rank(counts, recent_deque, last_seen, draw_index, recent_window):\n"
        "        nums = list(range(1, config.number_range + 1))\n"
        "        return sorted(nums, key=lambda n: (-counts.get(n, 0), n))\n"
        "    return rank\n",
        encoding="utf-8",
    )
    rng = random.Random(1)
    draws = [LotteryDraw(date(2020, 1, 1) + timedelta(days=i), None, None,
                         tuple(sorted(rng.sample(range(1, 50), 6)))) for i in range(200)]
    result = ai_strategy_lab.backtest_plugin_holdout(plugin, "loto_6_49_ro", draws, holdout_fraction=0.25)
    assert result["evaluated_draws"] == 50
    assert 0.0 <= result["average_hits"] <= 6.0
