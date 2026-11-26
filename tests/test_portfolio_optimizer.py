from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import portfolio_optimizer as po


def _cfg(**kw):
    base = dict(number_range=90, numbers_drawn=6, ticket_size=6, bonus_range=None, min_bonus_numbers=0)
    base.update(kw)
    return SimpleNamespace(**base)


def test_min_prize_hits_from_tiers():
    assert po.min_prize_hits("superenalotto_it") == 2
    assert po.min_prize_hits("loto_6_49_ro") == 3
    assert po.min_prize_hits("multi_multi_pl") == 6
    assert po.winning_hits("kaskada_pl", 12) == {8, 9, 10, 11, 12}
    assert po.min_prize_hits("unknown_lottery") is None


def test_simulated_draws_have_exact_size():
    m = po.simulate_draws(49, 6, 500, seed=1)
    assert m.shape == (500, 50)
    assert (m.sum(axis=1) == 6).all()
    assert m[:, 0].sum() == 0


def test_optimizer_improves_concentrated_portfolio_and_keeps_favorite():
    rng = random.Random(3)
    # portofoliu concentrat pe 1..30, ca biletele AI reale la SuperEnalotto
    tickets = [{"main": sorted(rng.sample(range(1, 31), 6)), "bonus": [], "why": "ai"} for _ in range(12)]
    cfg = _cfg()
    out, metrics = po.optimize_ticket_dicts(tickets, cfg, "superenalotto_it", n_samples=1500, eval_samples=6000)

    assert out[0]["main"] == tickets[0]["main"]  # favoritul rămâne
    assert metrics["p_any_prize_after"] > metrics["p_any_prize_before"] + 0.05
    assert metrics["mean_overlap_after"] < metrics["mean_overlap_before"]
    assert metrics["popularity_after"] < metrics["popularity_before"]
    mains = [tuple(t["main"]) for t in out]
    assert len(set(mains)) == len(mains)
    for t in out:
        assert len(set(t["main"])) == 6 and all(1 <= n <= 90 for n in t["main"])


def test_optimizer_is_deterministic():
    tickets = [{"main": [1, 2, 3, 4, 5, 6 + i], "bonus": []} for i in range(5)]
    a, _ = po.optimize_ticket_dicts(tickets, _cfg(number_range=49), "lotto_pl", n_samples=800, eval_samples=2000)
    b, _ = po.optimize_ticket_dicts(tickets, _cfg(number_range=49), "lotto_pl", n_samples=800, eval_samples=2000)
    assert [t["main"] for t in a] == [t["main"] for t in b]


def test_diversify_bonus_spreads_joker_numbers():
    cfg = _cfg(number_range=45, numbers_drawn=5, bonus_range=20, min_bonus_numbers=1)
    tickets = [{"main": [1, 2, 3, 4, 5 + i], "bonus": [7]} for i in range(20)]
    out = po.diversify_bonus(tickets, cfg)
    bonuses = [t["bonus"][0] for t in out]
    assert bonuses[0] == 7
    assert len(set(bonuses)) == 20


def test_unknown_lottery_is_left_untouched():
    tickets = [{"main": [1, 2, 3], "bonus": []}, {"main": [1, 2, 4], "bonus": []}]
    out, metrics = po.optimize_ticket_dicts(tickets, _cfg(number_range=10, numbers_drawn=3), "nope")
    assert out == tickets and metrics == {}
