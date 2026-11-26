import math
from collections import defaultdict

NAME = "zpairflo"
DESCRIPTION = 'Graf de co-ocurenta cu message-passing: afinitatea fiecarui numar cu numerele din ultima extragere plus un pas de propagare a EWMA-ului de frecventa prin muchiile grafului. Combinat non-liniar (tanh) cu presiune de intarziere tip hazard, echilibru pe cele 5 sectoare si un prior slab de frecventa totala.'
SOURCE = 'glm-5.3-flash + 2026-09-08 / EuroMillions 5/50, 98 extrageri'

def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    if K < 1:
        K = 1
    if K > N:
        K = N

    def _zs(vals):
        keys = list(vals.keys())
        m = sum(vals.values()) / len(keys)
        var = sum((vals[k] - m) ** 2 for k in keys) / len(keys)
        sd = math.sqrt(var)
        if sd < 1e-9:
            return dict((k, 0.0) for k in keys)
        return dict((k, (vals[k] - m) / sd) for k in keys)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # conventie: draws[-1] = cea mai recenta extragere din fereastra
        numbers = list(range(1, N + 1))
        draws = list(recent_deque)
        if not draws:
            z_cnt = _zs(dict((n, float(counts.get(n, 0))) for n in numbers))
            return sorted(numbers, key=lambda n: (-z_cnt[n], n))
        L = len(draws)
        exp_gap = N / float(K)

        # --- EWMA hotness, half-life 10 extrageri ---
        lam = math.log(2.0) / 10.0
        ewma = dict((n, 0.0) for n in numbers)
        for i in range(L):
            w = math.exp(-lam * (L - 1 - i))
            for x in draws[i]:
                if 1 <= x <= N:
                    ewma[x] += w
        z_hot = _zs(ewma)

        # --- graf de co-ocurenta pe fereastra scurta (cap 60) ---
        cap = 60
        pd = draws[-cap:] if L > cap else draws
        pl = len(pd)
        pair = defaultdict(float)
        for s in pd:
            ss = sorted(x for x in s if 1 <= x <= N)
            for i in range(len(ss)):
                for j in range(i + 1, len(ss)):
                    pair[(ss[i], ss[j])] += 1.0

        # --- afinitate cu ultima extragere (message-passing pas 1) ---
        lastn = [x for x in draws[-1] if 1 <= x <= N]
        aff = dict((n, 0.0) for n in numbers)
        for n in numbers:
            t = 0.0
            for m in lastn:
                if m != n:
                    a2 = n if n < m else m
                    b2 = m if n < m else n
                    t += pair.get((a2, b2), 0.0)
            aff[n] = t
        z_pair = _zs(aff)

        # --- flow: propagare EWMA prin muchiile grafului (pas 2) ---
        flow = dict((n, 0.0) for n in numbers)
        for (a, b), c in pair.items():
            flow[a] += c * ewma[b]
            flow[b] += c * ewma[a]
        z_flow = _zs(flow)

        # --- echilibru pe 5 sectoare, z hipergeometric ---
        scap = 40
        sdw = draws[-scap:] if L > scap else draws
        sl = len(sdw)
        sec_cnt = [0, 0, 0, 0, 0]
        for s in sdw:
            for x in s:
                if 1 <= x <= N:
                    sec_cnt[(x - 1) * 5 // N] += 1
        hval = (N - K) / float(N - 1) if N > 1 else 1.0
        sec_z = [0.0, 0.0, 0.0, 0.0, 0.0]
        for i in range(5):
            size = (i + 1) * N // 5 - i * N // 5
            if size <= 0:
                continue
            es = sl * K * size / float(N)
            fr = size / float(N)
            pv = K * fr * (1.0 - fr) * hval
            sds = math.sqrt(max(pv * sl, 1e-9))
            sec_z[i] = (sec_cnt[i] - es) / sds

        # --- presiune de intarziere tip hazard ---
        ov = {}
        for n in numbers:
            ls = last_seen.get(n)
            gap = (draw_index - ls) if ls is not None else (draw_index + exp_gap)
            ov[n] = gap / exp_gap

        # --- prior slab de frecventa totala ---
        z_cnt = _zs(dict((n, float(counts.get(n, 0))) for n in numbers))

        # --- combinare non-liniara, fiecare componenta bounded de tanh ---
        scores = {}
        for n in numbers:
            sz = sec_z[(n - 1) * 5 // N]
            scores[n] = (1.00 * math.tanh(0.90 * z_hot[n])
                         + 0.90 * math.tanh(0.90 * z_pair[n])
                         + 0.70 * math.tanh(0.80 * z_flow[n])
                         + 0.50 * math.tanh(0.80 * sz)
                         + 0.60 * math.tanh(0.70 * (ov[n] - 1.0))
                         + 0.20 * z_cnt[n])
        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank