import numpy as np
from math import sqrt, tanh

NAME = "zpairgat"
DESCRIPTION = 'Selectie conditionata de perechi: co-ocurenta in fereastra recenta (exclusiv ultima extragere) scorand partenerii extragerii anterioare, amplificati nonlinear de o poarta tanh pe intarziere (gap geometric); adauga fading pe frecventa recenta si tilt negativ pe deviatia globala de frecventa. Toate componentele sunt invariante la orientarea deque-ului.'
SOURCE = 'glm-5.3-flash + 2026-09-27'


def create(config):
    nr = int(config.number_range)
    nd = int(config.numbers_drawn)
    w_pair = float(getattr(config, 'pg_w_pair', 1.0))
    w_due = float(getattr(config, 'pg_w_due', 0.30))
    w_fade = float(getattr(config, 'pg_w_fade', 0.20))
    w_glob = float(getattr(config, 'pg_w_glob', 0.15))
    w_nl = float(getattr(config, 'pg_w_nl', 0.40))

    p = nd / float(nr)
    q = 1.0 - p
    mean_gap = q / p
    sd_gap = sqrt(q) / p

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        nums = list(range(1, nr + 1))
        N = nr
        cnts = np.array([counts.get(n, 0) for n in nums], dtype=float)

        # deviatia globala fata de frecventa asteptata; T auto-calibrat din media contoarelor
        T_eff = cnts.mean() / p
        sd_g = sqrt(max(T_eff * p * q, 1e-12))
        z_glob = (cnts - T_eff * p) / sd_g
        score = -w_glob * z_glob

        prev_idx = int(draw_index) - 1
        prev_set = set()
        if prev_idx >= 0:
            prev_set = set(n for n in nums if last_seen.get(n, -1) == prev_idx)

        # ciclu de intarziere: gap fata de distributia geometrica a asteptarilor
        gaps = np.array([prev_idx - last_seen.get(n, -1) for n in nums], dtype=float)
        due_z = np.clip((gaps - mean_gap) / sd_gap, -3.0, 3.0)
        score = score + w_due * due_z

        if len(recent_deque) > 0:
            rows = [s for s in recent_deque if s != prev_set]
            R = len(rows)
            if R > 0:
                sizes = [len(s) for s in rows]
                cols = np.fromiter((n - 1 for s in rows for n in s), dtype=np.int64, count=sum(sizes))
                ridx = np.repeat(np.arange(R, dtype=np.int64), sizes)
                P = np.zeros((R, N), dtype=float)
                P[ridx, cols] = 1.0
                # fading: penalizeaza supraproducerea recenta
                rsd = sqrt(R * p * q)
                z_rec = (P.sum(axis=0) - R * p) / rsd
                score = score - w_fade * z_rec
                # perechi co-aparute in fereastra, conditonate de extragerea precedenta
                C = P.T @ P
                np.fill_diagonal(C, 0.0)
                if prev_set:
                    pcols = sorted(n - 1 for n in prev_set)
                    pair_raw = C[:, pcols].sum(axis=1)
                    ps = float(pair_raw.std())
                    if ps > 1e-12:
                        pair_z = (pair_raw - pair_raw.mean()) / ps
                        score = score + w_pair * pair_z * (1.0 + w_nl * np.tanh(due_z * 0.5))

        return sorted(nums, key=lambda n: (-float(score[n - 1]), n))

    return rank