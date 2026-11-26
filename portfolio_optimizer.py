"""Optimizatorul portofoliului de bilete pentru o extragere.

Adevărul matematic: într-o loterie aleatoare fiecare bilet are ACEEAȘI
probabilitate de câștig, deci nicio alegere de numere nu crește numărul
așteptat de lovituri. Ce SCHIMBĂ alegerea numerelor:

1. Suprapunerea biletelor. Biletele care împart multe numere câștigă sau pierd
   împreună; răspândirea lor crește P(cel puțin un premiu pe extragere) la
   ACELAȘI cost (aceeași valoare așteptată, varianță mai mică).
2. Popularitatea numerelor. Jucătorii aleg excesiv 1–31 (date de naștere), 7 și
   multiplii lui, secvențe consecutive și numerele extragerii anterioare. La
   categoriile cu fond comun (pari-mutuel) premiul se împarte la câștigători →
   numerele nepopulare dau aceleași șanse, dar o parte mai mare din premiu.

Optimizatorul face căutare locală (schimbă câte un număr) maximizând numărul de
extrageri simulate în care măcar un bilet atinge categoria minimă de premiu, cu
o penalizare de popularitate ca obiectiv secundar. Metricele raportate sunt
calculate pe o simulare INDEPENDENTĂ (alt seed), ca îmbunătățirea să nu fie un
artefact al eșantionului folosit la căutare.
"""

from __future__ import annotations

import itertools
from collections import Counter

import numpy as np

from prize_estimates import TIERS, winning_hit_counts

# loterii la care (măcar categoriile mari) se împart între câștigători
PARI_MUTUEL = frozenset({
    "lotto_pl", "mini_lotto_pl", "loto_6_49_ro", "loto_5_40_ro", "joker_ro", "superenalotto_it",
})
SEARCH_SAMPLES = 4000
EVAL_SAMPLES = 20000
DEFAULT_SEED = 20260915


def min_prize_hits(lottery_id: str) -> int | None:
    """Numărul minim de numere principale nimerite care dă un premiu (fără bonus)."""
    hits = [main for main, bonus_req, _label, amount, _cur in TIERS.get(lottery_id, [])
            if amount > 0 and bonus_req == 0 and main > 0]
    return min(hits) if hits else None


def winning_hits(lottery_id: str, ticket_size: int) -> set[int]:
    """Toate valorile de lovituri care dau premiu (ex. Kaskada ≥8 sau ≤4; Multi Multi 0 sau ≥6)."""
    return winning_hit_counts(lottery_id, ticket_size)


def _wins(hits: np.ndarray, win) -> np.ndarray:
    """win = prag întreg (≥ win) sau mulțime de valori câștigătoare."""
    if isinstance(win, (int, np.integer)):
        return hits >= win
    return np.isin(hits, sorted(win))


def simulate_draws(number_range: int, numbers_drawn: int, n_samples: int, seed: int) -> np.ndarray:
    """Matrice (n_samples × number_range+1) cu 1 pe numerele extrase (coloana 0 nefolosită)."""
    rng = np.random.default_rng(seed)
    keys = rng.random((n_samples, number_range))
    picked = np.argpartition(keys, numbers_drawn - 1, axis=1)[:, :numbers_drawn] + 1
    matrix = np.zeros((n_samples, number_range + 1), dtype=np.int16)
    np.put_along_axis(matrix, picked, 1, axis=1)
    return matrix


def popularity_penalty(numbers, number_range: int, last_draw=()) -> float:
    """Cât de „populară" e o combinație (0 = deloc). Normalizat la mărimea biletului."""
    nums = sorted(int(n) for n in numbers)
    if not nums:
        return 0.0
    size = len(nums)
    penalty = 0.0
    if number_range > 31:
        penalty += sum(1 for n in nums if n <= 31) / size
    penalty += 0.5 * sum(1 for n in nums if n % 7 == 0) / size
    penalty += 0.5 * sum(1 for a, b in zip(nums, nums[1:]) if b == a + 1) / size
    if last_draw:
        penalty += 0.5 * len(set(nums) & {int(n) for n in last_draw}) / size
    return penalty


def p_any_prize(tickets, draws_matrix: np.ndarray, min_hits) -> float:
    """Fracțiunea extragerilor simulate în care măcar un bilet câștigă (prag sau mulțime)."""
    if not tickets:
        return 0.0
    hits = np.stack([draws_matrix[:, list(t)].sum(axis=1) for t in tickets], axis=1)
    return float(_wins(hits, min_hits).any(axis=1).mean())


def mean_overlap(tickets) -> float:
    pairs = list(itertools.combinations([set(t) for t in tickets], 2))
    return sum(len(a & b) for a, b in pairs) / len(pairs) if pairs else 0.0


def optimize_portfolio(
    tickets,
    *,
    number_range: int,
    numbers_drawn: int,
    min_hits,
    frozen: int = 1,
    n_samples: int = SEARCH_SAMPLES,
    seed: int = DEFAULT_SEED,
    max_rounds: int = 6,
    last_draw=(),
    popularity_weight: float = 0.0,
) -> list[list[int]]:
    """Căutare locală: pentru fiecare bilet nefixat aplică cel mai bun schimb de un număr.

    Primele `frozen` bilete rămân neschimbate (biletul favorit al strategiei).
    """
    current = [sorted({int(n) for n in t}) for t in tickets]
    if min_hits is None or len(current) <= frozen:
        return current
    draws = simulate_draws(number_range, numbers_drawn, n_samples, seed)
    hits = np.stack([draws[:, t].sum(axis=1) for t in current], axis=1)

    for _round in range(max_rounds):
        improved = False
        for i in range(frozen, len(current)):
            ticket = current[i]
            win_counts = _wins(hits, min_hits).sum(axis=1)
            own = _wins(hits[:, i], min_hits)
            others = (win_counts - own) > 0
            best_score = (float((others | own).sum())
                          - popularity_weight * popularity_penalty(ticket, number_range, last_draw))
            best_move: tuple[int, int] | None = None
            taken = {tuple(t) for j, t in enumerate(current) if j != i}
            in_ticket = set(ticket)
            candidates = np.array([n for n in range(1, number_range + 1) if n not in in_ticket])
            if candidates.size == 0:
                continue
            cand_cols = draws[:, candidates]
            for out in ticket:
                new_hits = hits[:, i][:, None] - draws[:, out][:, None] + cand_cols
                scores = (others[:, None] | _wins(new_hits, min_hits)).sum(axis=0).astype(float)
                stay = [n for n in ticket if n != out]
                if popularity_weight:
                    scores -= popularity_weight * np.array(
                        [popularity_penalty(stay + [int(c)], number_range, last_draw) for c in candidates]
                    )
                for idx in np.argsort(-scores, kind="stable")[:5]:
                    if scores[idx] <= best_score + 1e-9:
                        break
                    if tuple(sorted(stay + [int(candidates[idx])])) in taken:
                        continue  # nu creăm bilete duplicate
                    best_score = float(scores[idx])
                    best_move = (out, int(candidates[idx]))
                    break
            if best_move is not None:
                out, new_number = best_move
                current[i] = sorted([n for n in ticket if n != out] + [new_number])
                hits[:, i] = draws[:, current[i]].sum(axis=1)
                improved = True
        if not improved:
            break
    return current


def diversify_bonus(tickets: list[dict], config) -> list[dict]:
    """Răspândește numerele bonus între bilete (ex. Joker: 20 bilete → 20 bonusuri diferite)."""
    n_bonus = config.min_bonus_numbers if config.bonus_range else 0
    if not n_bonus:
        return tickets
    domain = range(1, int(config.bonus_range) + 1)
    usage: Counter = Counter()
    out: list[dict] = []
    for idx, ticket in enumerate(tickets):
        item = dict(ticket)
        proposed = [int(b) for b in item.get("bonus") or [] if 1 <= int(b) <= config.bonus_range]
        if idx == 0:
            chosen = proposed[:n_bonus]
        else:
            chosen = []
            for b in proposed:
                least = min(usage[x] for x in domain)
                if usage[b] == least and b not in chosen and len(chosen) < n_bonus:
                    chosen.append(b)
            for b in sorted(domain, key=lambda x: (usage[x], x)):
                if len(chosen) >= n_bonus:
                    break
                if b not in chosen:
                    chosen.append(b)
        for b in chosen:
            usage[b] += 1
        item["bonus"] = sorted(chosen)
        out.append(item)
    return out


def optimize_ticket_dicts(
    tickets: list[dict],
    config,
    lottery_id: str,
    *,
    last_draw=(),
    n_samples: int = SEARCH_SAMPLES,
    eval_samples: int = EVAL_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict], dict]:
    """Optimizează biletele agentului (format {"main", "bonus", "why"}) + metrici.

    Primul bilet (favoritul) rămâne neschimbat. Dacă pe simularea independentă
    portofoliul optimizat ar fi mai slab decât cel inițial, se păstrează cel inițial.
    """
    k = min_prize_hits(lottery_id)
    mains = [sorted(int(n) for n in t.get("main", [])) for t in tickets]
    if k is None or len(mains) < 2 or len({len(m) for m in mains}) != 1:
        return tickets, {}
    # Mulțimea exactă de lovituri câștigătoare pentru numerele fiecărui bilet.
    win = winning_hits(lottery_id, len(mains[0])) or k
    # premii fixe (Multi Multi, Kaskada): popularitatea nu schimbă nimic → doar acoperire
    weight = (0.005 * n_samples) if lottery_id in PARI_MUTUEL else 0.0
    optimized = optimize_portfolio(
        mains, number_range=config.number_range, numbers_drawn=config.numbers_drawn,
        min_hits=win, frozen=1, n_samples=n_samples, seed=seed,
        last_draw=last_draw, popularity_weight=weight,
    )
    evaluation = simulate_draws(config.number_range, config.numbers_drawn, eval_samples, seed + 1)
    before = p_any_prize(mains, evaluation, win)
    after = p_any_prize(optimized, evaluation, win)
    if after < before - 0.002:  # plasă de siguranță: nu livrăm un portofoliu mai slab
        optimized, after = mains, before

    out: list[dict] = []
    for ticket, new_main in zip(tickets, optimized):
        item = dict(ticket)
        if sorted(item.get("main", [])) != new_main:
            item["main"] = new_main
            note = "ajustat: acoperire mai bună / numere mai puțin populare"
            item["why"] = f"{item.get('why') or ''} · {note}".strip(" ·")[:400]
        out.append(item)
    out = diversify_bonus(out, config)

    def avg_pop(ts):
        return sum(popularity_penalty(t, config.number_range, last_draw) for t in ts) / len(ts)

    metrics = {
        "min_prize_hits": k,
        "p_any_prize_before": round(before, 4),
        "p_any_prize_after": round(after, 4),
        "mean_overlap_before": round(mean_overlap(mains), 3),
        "mean_overlap_after": round(mean_overlap(optimized), 3),
        "popularity_before": round(avg_pop(mains), 3),
        "popularity_after": round(avg_pop(optimized), 3),
        "simulated_draws": eval_samples,
    }
    return out, metrics
