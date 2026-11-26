NAME = "zphasecy"
DESCRIPTION = ('Overdue-cycle phase resonance: each number carries a personal re-appearance cycle (median of its recent gaps) scored with harmonic multiples, amplified when its sector is in a matching overdue phase, all gated by the entropy of the recent window.')
SOURCE = 'cycle-phase research model + 2026-09-08'

import math
from math import comb


def create(config):
    N = int(config.number_range)
    KD = int(config.numbers_drawn)
    p_draw = KD / float(N) if N > 0 else 0.0
    e_gap = (1.0 - p_draw) / p_draw if 0.0 < p_draw < 1.0 else float(N)

    W_H = (1.0, 0.45, 0.22)   # harmonic weights h=1..3
    S0 = 1.5                  # base width of personal harmonics (draws)
    B0 = 0.50                 # base overdue prior amplitude
    SC = 2.4                  # width of base overdue prior
    WS = (1.0, 0.35)          # sector harmonic weights
    SS = 0.9                  # sector resonance width
    L1 = 1.4                  # personal resonance gain
    L2 = 0.9                  # sector locking gain
    E_HI = 0.92               # entropy gate: fully closed at/above this
    E_LO = 0.84               # entropy gate: fully open at/below this
    NSEC = 10

    def _median(vals):
        s = sorted(vals)
        k = len(s)
        if k == 0:
            return None
        if k % 2:
            return float(s[k // 2])
        return 0.5 * (s[k // 2 - 1] + s[k // 2])

    def _clip(x, lo, hi):
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def _sec(n):
        return (n - 1) * NSEC // N

    _sec_egap = {}
    for s in range(NSEC):
        size = sum(1 for n in range(1, N + 1) if _sec(n) == s)
        rest = N - size
        pn = float(comb(rest, KD)) / float(comb(N, KD)) if rest >= KD else 0.0
        _sec_egap[s] = 1.0 / (1.0 - pn) if pn < 1.0 else float(NSEC)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        if not last_seen:
            return list(numbers)

        m = max(last_seen.get(n, -1) for n in numbers)
        if m < 0:
            return list(numbers)

        W = list(recent_deque)
        L = len(W)

        # infer deque orientation deterministically: newest end must equal the last drawn set
        last_set = set(n for n in numbers if last_seen.get(n, -1) == m)
        newest_first = False
        if L > 0:
            if set(W[-1]) == last_set:
                newest_first = False
            elif set(W[0]) == last_set:
                newest_first = True

        def r_of(j):
            return j if newest_first else (L - 1 - j)

        # absolute appearance times per number and per sector
        times = dict((n, []) for n in numbers)
        sec_times = dict((s, []) for s in range(NSEC))
        for j in range(L):
            a = m - r_of(j)
            row = W[j]
            for n in row:
                if 1 <= n <= N:
                    times[n].append(a)
                    sec_times[_sec(n)].append(a)

        # global pooled cycle (median of all inter-appearance gaps)
        pool = []
        for n in numbers:
            t = sorted(times[n])
            for i in range(1, len(t)):
                pool.append(t[i] - t[i - 1])
        C = _median(pool) if pool else e_gap
        C = _clip(C, 3.0, 15.0)

        # personal harmonic resonance per number
        R = {}
        for n in numbers:
            t = sorted(times[n])
            gaps = [t[i] - t[i - 1] for i in range(1, len(t))]
            if not gaps:
                R[n] = 0.0
                continue
            c = _clip(_median(gaps[-3:]), 2.0, 25.0)
            conf = (0.7, 0.9, 1.0)[min(len(gaps), 3) - 1]
            g = float(m - last_seen[n]) if n in last_seen else float(m + 1)
            acc = 0.0
            for h in (1, 2, 3):
                sd = S0 * math.sqrt(h)
                acc += W_H[h - 1] * math.exp(-0.5 * ((g - h * c) / sd) ** 2)
            R[n] = conf * acc

        # sector phase resonance
        RS = {}
        for s in range(NSEC):
            t = sorted(set(sec_times[s]))
            gaps = [t[i] - t[i - 1] for i in range(1, len(t))]
            c = _clip(_median(gaps[-4:]), 1.5, 30.0) if gaps else _sec_egap[s]
            last_t = -1
            for n in numbers:
                if _sec(n) == s:
                    lt = last_seen.get(n, -1)
                    if lt > last_t:
                        last_t = lt
            G = float(m - last_t) if last_t >= 0 else float(m + 1)
            acc = 0.0
            for h in (1, 2):
                sd = SS * math.sqrt(h)
                acc += WS[h - 1] * math.exp(-0.5 * ((G - h * c) / sd) ** 2)
            RS[s] = acc

        # entropy gate over the newest K_ent draws
        gate = 0.5
        K_ent = min(15, L)
        if K_ent > 0:
            cnt = {}
            for r in range(K_ent):
                j = r if newest_first else (L - 1 - r)
                for n in W[j]:
                    if 1 <= n <= N:
                        cnt[n] = cnt.get(n, 0) + 1
            tot = float(KD * K_ent)
            H = 0.0
            for v in cnt.values():
                pp = v / tot
                if pp > 0.0:
                    H -= pp * math.log(pp)
            e = H / math.log(N) if N > 1 else 1.0
            gate = _clip((E_HI - e) / (E_HI - E_LO), 0.0, 1.0)

        scored = []
        for n in numbers:
            g = float(m - last_seen[n]) if n in last_seen else float(m + 1)
            base = 1.0 + B0 * math.exp(-0.5 * ((g - C) / SC) ** 2)
            cyc = L1 * R[n] + L2 * RS[_sec(n)]
            scored.append((-(base * (1.0 + gate * cyc)), n))
        scored.sort()
        return [n for _, n in scored]

    return rank