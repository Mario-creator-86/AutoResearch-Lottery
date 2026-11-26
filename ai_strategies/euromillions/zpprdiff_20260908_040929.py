NAME = "zpprdiff"
DESCRIPTION = 'Personalized-PageRank heat diffusion on the recency-weighted co-occurrence graph of numbers, seeded by fast-decayed recent appearances, with diffusion rate adapted to an entropy-type window concentration and a small window-frequency prior.'
SOURCE = 'model + 2026-09-07'

import math

import numpy as np


def create(config):
    n_range = int(config.number_range)
    k_draw = max(1, int(config.numbers_drawn))

    EPS = 1e-12
    DECAY = 0.965      # recency decay for marginals/pairs (half-life ~20 draws)
    SEED_DECAY = 0.85  # fast decay for the restart seed (half-life ~4 draws)
    SEED_FLOOR = 0.10  # uniform floor mass inside the seed
    SHRINK = 0.60      # shrinkage toward expected co-occurrence mass
    ALPHA_HI = 0.75    # diffusion rate when the window is concentrated
    ALPHA_LO = 0.45    # diffusion rate when the window is uniform
    BETA_FREQ = 0.35   # weight of the standardized window-frequency z-score
    ITERS = 16

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, n_range + 1))
        m = len(numbers)

        def fallback():
            sc = {k: float(counts.get(k, 0)) for k in numbers}
            return sorted(numbers, key=lambda k: (-sc[k], k))

        if m < 3 or k_draw < 2:
            return fallback()

        try:
            raw = list(recent_deque)
        except Exception:
            raw = []
        draws = []
        for d in raw:
            try:
                s = set()
                for x in d:
                    xi = int(x)
                    if 1 <= xi <= n_range:
                        s.add(xi)
            except Exception:
                s = set()
            if s:
                draws.append(s)

        if len(draws) < 2:
            return fallback()

        ages = [len(draws) - 1 - i for i in range(len(draws))]
        w = [DECAY ** a for a in ages]
        wsum = float(sum(w))
        if wsum <= EPS:
            w = [1.0] * len(draws)
            wsum = float(len(draws))
        wsum = max(wsum, EPS)

        idx = {k: i for i, k in enumerate(numbers)}
        c = np.zeros(m, dtype=np.float64)
        for wi, s in zip(w, draws):
            for k in s:
                c[idx[k]] += wi

        tot = float(c.sum())
        conc = 0.0
        if tot > EPS:
            pn = c / tot
            pns = np.sort(pn)[::-1]
            topk = min(10, m)
            share = float(pns[:topk].sum())
            base_share = topk / float(m)
            denom = max(1e-9, 1.0 - base_share)
            conc = min(max((share - base_share) / denom, 0.0), 1.0)

        obs = np.zeros((m, m), dtype=np.float64)
        for wi, s in zip(w, draws):
            lst = sorted(s)
            for a_ in range(len(lst)):
                ia = idx[lst[a_]]
                for b_ in range(a_ + 1, len(lst)):
                    ib = idx[lst[b_]]
                    obs[ia, ib] += wi
                    obs[ib, ia] += wi

        exp_pair = np.outer(c, c) * ((k_draw - 1.0) / (k_draw * wsum))
        base = SHRINK * float(exp_pair.mean())
        aff = (obs + base + SHRINK * exp_pair) / (exp_pair + base + EPS)
        row = aff.sum(axis=1, keepdims=True)
        P = aff / (row + EPS)

        s0 = np.full(m, SEED_FLOOR / m, dtype=np.float64)
        w_seed = [SEED_DECAY ** a for a in ages]
        for wi, s in zip(w_seed, draws):
            add = wi / max(1.0, float(len(s)))
            for k in s:
                s0[idx[k]] += add
        s0 = s0 / float(s0.sum())

        alpha = ALPHA_LO + (ALPHA_HI - ALPHA_LO) * conc
        PT = np.ascontiguousarray(P.T)
        s = s0.copy()
        for _ in range(ITERS):
            s = (1.0 - alpha) * s0 + alpha * (PT @ s)

        mu = float(s.mean())
        sd = float(s.std())
        s_norm = (s - mu) / (sd + EPS)

        p1 = k_draw / float(m)
        var = wsum * p1 * (1.0 - p1)
        z = (c - p1 * wsum) / (math.sqrt(var) + EPS)
        z = np.clip(z, -3.0, 3.0)

        final = s_norm + BETA_FREQ * z
        return sorted(numbers, key=lambda k: (-float(final[idx[k]]), k))

    return rank