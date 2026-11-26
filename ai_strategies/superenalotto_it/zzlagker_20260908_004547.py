# -*- coding: utf-8 -*-
NAME = "zzlagker"
DESCRIPTION = 'Filtru adaptat pe autocorelatiile empirice lag-k (exces de co-aparitie la fiecare lag 1..10, z-standardizat fata de media conditionata), proiectat pe aparitiile recente, plus centralitate eigenvector pe graful de co-aparitie al ferestrei si deviatie z pe termen lung fata de frecventa asteptata 6/90.'
SOURCE = 'model + 2026-09-08'

import numpy as np


def create(config):
    N = int(config.number_range)
    W = int(config.numbers_drawn)
    p = float(W) / float(N)
    CLIP = 2.5
    KMAX_BASE = 10
    W_LAG = 1.0
    W_CENT = 0.35
    W_LONG = 0.20

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        D = max(int(draw_index), 1)
        mu = D * p
        sdb = float(np.sqrt(D * p * (1.0 - p))) + 1e-9
        cn = np.array([float(counts.get(n, 0)) for n in numbers], dtype=float)
        zlong = np.clip((cn - mu) / sdb, -CLIP, CLIP)

        seq = list(recent_deque) if recent_deque is not None else []
        if len(seq) >= 2:
            for target in (draw_index - 1, draw_index):
                prev_nums = {n for n in numbers if last_seen.get(n, -10 ** 9) == target}
                if len(prev_nums) < 3:
                    continue
                s0 = set(seq[0])
                sL = set(seq[-1])
                front = len(prev_nums & s0)
                back = len(prev_nums & sL)
                if max(front, back) >= 3:
                    if front > back:
                        seq = seq[::-1]
                    break

        T = len(seq)
        base = W_LONG * zlong
        if T < 6:
            key = {n: float(base[n - 1]) for n in numbers}
            return sorted(numbers, key=lambda n: (-key[n], n))

        X = np.zeros((T, N), dtype=float)
        for t in range(T):
            st = seq[t]
            for n in st:
                if 1 <= n <= N:
                    X[t, n - 1] = 1.0

        cnt_w = X.sum(axis=0)
        K = min(KMAX_BASE, T)
        lag_score = np.zeros(N, dtype=float)
        denom_pairs = float(max(T * (T - 1), 1))
        for k in range(1, K + 1):
            pair_k = (X[k:] * X[:-k]).sum(axis=0)
            exp_k = (T - k) * cnt_w * np.maximum(cnt_w - 1.0, 0.0) / denom_pairs
            sd_k = np.sqrt(exp_k + 1.0)
            zk = np.clip((pair_k - exp_k) / sd_k, -CLIP, CLIP)
            lag_score += zk * X[T - k, :]
        lag_score /= float(np.sqrt(K))

        A = X.T @ X
        v = np.ones(N, dtype=float)
        for _ in range(40):
            v = A @ v
            s = float(v.sum())
            if (not np.isfinite(s)) or s <= 1e-12:
                v = np.ones(N, dtype=float)
                break
            v = v / s
        cent = np.clip(v / (float(v.mean()) + 1e-12) - 1.0, -1.0, 4.0)

        scores = W_LAG * lag_score + W_CENT * cent + base
        key = {n: float(scores[n - 1]) for n in numbers}
        return sorted(numbers, key=lambda n: (-key[n], n))

    return rank