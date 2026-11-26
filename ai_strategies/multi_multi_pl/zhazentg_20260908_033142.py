NAME = "zhazentg"
DESCRIPTION = 'Hazard empiric pe bucket-uri de gap (rata conditionata de aparitie cu shrinkage Beta spre 0.25), estimat pe fereastra recenta, fuzionat cu EWMA de recenta (half-life 10), kernel de adiacenta pe axa numerelor si deviatia binomiala pe istoricul complet; un gate de dispersie a ferestrei recompara adaptiv momentum vs. reversie.'
SOURCE = 'model + 2026-09-08'

import math
import numpy as np

def create(config):
    P = int(getattr(config, 'number_range', 80))
    K = int(getattr(config, 'numbers_drawn', 20))
    p0 = (K / float(P)) if P > 0 else 0.25

    def _bucket(g):
        if g <= 2:
            return 0
        if g <= 5:
            return 1
        if g <= 10:
            return 2
        if g <= 20:
            return 3
        return 4

    def _zclip(v):
        v = np.asarray(v, dtype=np.float64)
        sd = float(np.std(v))
        if sd < 1e-12:
            return np.zeros_like(v)
        return np.tanh((v - float(np.mean(v))) / sd)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, P + 1))
        draws = list(recent_deque)
        M = len(draws)

        # orientare cronologica (cel mai vechi -> cel mai nou)
        if M >= 2:
            prev_idx = draw_index - 1
            c_last = 0
            for n in draws[-1]:
                if last_seen.get(n, -1) == prev_idx:
                    c_last += 1
            c_first = 0
            for n in draws[0]:
                if last_seen.get(n, -1) == prev_idx:
                    c_first += 1
            if c_first > c_last:
                draws.reverse()
            elif c_first == c_last:
                s_first = 0
                for n in draws[0]:
                    s_first += last_seen.get(n, -1)
                s_last = 0
                for n in draws[-1]:
                    s_last += last_seen.get(n, -1)
                if s_first > s_last:
                    draws.reverse()

        f = min(1.0, M / 60.0) if M > 0 else 0.0

        # gate de regim: raportul de dispersie a numararii pe fereastra
        g_hot = 0.5
        if M > 0:
            flat = np.fromiter((n for s in draws for n in s), dtype=np.int64)
            if flat.size > 0:
                cnt = np.bincount(flat, minlength=P + 1).astype(np.float64)
                v0 = M * p0 * (1.0 - p0)
                if v0 > 1e-12:
                    R = math.sqrt(float(np.var(cnt)) / v0)
                    g_hot = 0.5 + 1.0 * (R - 1.0)
                    if g_hot < 0.0:
                        g_hot = 0.0
                    elif g_hot > 1.0:
                        g_hot = 1.0

        # EWMA de recenta (momentum pe orizont scurt-mediu)
        ew = np.zeros(P + 1, dtype=np.float64)
        if M > 0:
            half = 10.0
            for p in range(M):
                w = 2.0 ** ((p - (M - 1)) / half)
                arr = np.fromiter(draws[p], dtype=np.int64)
                if arr.size > 0:
                    ew += np.bincount(arr, minlength=P + 1) * w
        z_e = _zclip(ew[1:])

        # hazard empiric: rata de aparitie la pasul urmator conditionata de gap
        hits = np.zeros(5, dtype=np.float64)
        trials = np.zeros(5, dtype=np.float64)
        prev_pos = {}
        for p in range(M):
            s = draws[p]
            nxt = draws[p + 1] if p + 1 < M else None
            for n in s:
                q = prev_pos.get(n, -1)
                if q >= 0 and nxt is not None:
                    b = _bucket(p - q)
                    trials[b] += 1.0
                    if n in nxt:
                        hits[b] += 1.0
                prev_pos[n] = p
        a0 = 10.0
        rates = (hits + a0 * p0) / (trials + a0)

        haz_rate = np.full(P + 1, p0, dtype=np.float64)
        if M > 0:
            for n in numbers:
                ls = last_seen.get(n, -1)
                if ls >= 0 and counts.get(n, 0) > 0:
                    haz_rate[n] = rates[_bucket(draw_index - ls)]
        z_h = _zclip(haz_rate[1:])

        # kernel de adiacenta pe axa 1..80 (vecinii directi ai iesirilor recente)
        glow = np.zeros(P + 1, dtype=np.float64)
        if M > 0:
            L = 6 if M >= 6 else M
            for j in range(L):
                s = draws[M - 1 - j]
                wgt = math.exp(-j / 2.5)
                for n in s:
                    if n - 1 >= 1:
                        glow[n - 1] += wgt
                    if n + 1 <= P:
                        glow[n + 1] += wgt
        z_g = _zclip(glow[1:])

        # deviatie binomiala pe tot istoricul (stabilizator)
        z_long = np.zeros(P, dtype=np.float64)
        T = draw_index
        if T > 0:
            sdL = math.sqrt(T * p0 * (1.0 - p0))
            if sdL > 1e-12:
                cl = np.fromiter((counts.get(n, 0) for n in numbers), dtype=np.float64, count=P)
                z_long = np.tanh((cl - T * p0) / sdL)

        score = f * (g_hot * z_e + (1.0 - g_hot) * z_h + 0.35 * z_g) + 0.20 * z_long
        scores = {n: float(score[i]) for i, n in enumerate(numbers)}
        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank