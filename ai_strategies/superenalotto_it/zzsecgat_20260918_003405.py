NAME = "zzsecgat"
DESCRIPTION = 'Ciclu de sector: deviatia z hipergeometrica a loviturilor pe sectoare de SEC_SIZE in fereastra recenta, ponderata de un gate de entropie, combinata liniar cu hazardul de intarziere centrat si plafonat (gap raportat la intervalul teoretic N/K) si cu deviatia z macro pe termen lung; semnele SEC/DEV comuta intre reversion si momentum.'
SOURCE = 'model cantitativ + 2026-09-17'

import math

# grid: SEC_SIZE{4,6,8} x SEC_SIGN{+-1} x DEV_SIGN{+-1} x W_SEC{0.35,0.7,1.05}
#       x W_GAP{0.3,0.5,0.8} x W_DEV{0,0.4,0.8} x GAP_CAP{1.5,2.5,4.0} x GATE_MIX{0.3,0.5,0.7}
SEC_SIZE = 6
SEC_SIGN = 1     # +1 = reversion (sectoare deficitare boost), -1 = momentum (sectoare calde boost)
DEV_SIGN = 1     # +1 = reversion (numere reci boost), -1 = momentum (numere calde boost)
W_SEC = 0.7
W_GAP = 0.5
W_DEV = 0.4
GAP_CAP = 2.5
GATE_MIX = 0.5   # cat de mult se atenueaza W_SEC cand entropia pe sectoare e mica

DIAG = {'calls': 0, 'sec_span_sum': 0.0, 'gap_span_sum': 0.0,
        'dev_span_sum': 0.0, 'entropy_sum': 0.0, 'last': None}


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)

    S = (N + SEC_SIZE - 1) // SEC_SIZE
    sec_of = [0] * (N + 1)
    size = [0] * S
    for n in range(1, N + 1):
        s = (n - 1) // SEC_SIZE
        sec_of[n] = s
        size[s] += 1

    p = K / float(N)
    den = float(N - 1) if N > 1 else 1.0
    share = [size[s] / float(N) for s in range(S)]
    exp_draw = [K * sh for sh in share]
    var_draw = [K * sh * (1.0 - sh) * (N - K) / den for sh in share]

    E_INT = N / float(K)          # intervalul asteptat intre aparitii (90/6 = 15)
    lnS = math.log(S) if S > 1 else 1.0

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        L = len(recent_deque)

        # lovituri pe sector in fereastra recenta (invariant la ordinea deque-ului)
        hits = [0.0] * S
        if L > 0:
            for st in recent_deque:
                for x in st:
                    if 1 <= x <= N:
                        hits[sec_of[x]] += 1.0

        tot = 0.0
        for s in range(S):
            tot += hits[s]

        # entropia normalizata a distributiei loviturilor pe sectoare
        H = 1.0
        if tot > 0.0 and S > 1:
            ent = 0.0
            for s in range(S):
                if hits[s] > 0.0:
                    q = hits[s] / tot
                    ent -= q * math.log(q)
            H = ent / lnS
            if H < 0.0:
                H = 0.0
            elif H > 1.0:
                H = 1.0

        gate = 1.0 - GATE_MIX * (1.0 - H)

        # z-score hipergeometric pe sector: (asteptat - observat) / sd
        Lf = float(L)
        zsec = [0.0] * S
        if L > 0:
            for s in range(S):
                sdw = math.sqrt(Lf * var_draw[s] + 1e-12)
                zsec[s] = (Lf * exp_draw[s] - hits[s]) / sdw

        # deviatia z macro pe termen lung
        T = float(draw_index) if draw_index > 0 else 1.0
        mu = T * p
        sdm = math.sqrt(T * p * (1.0 - p) + 1e-9)

        numbers = list(range(1, N + 1))
        sec_v = [SEC_SIGN * zsec[sec_of[n]] for n in numbers]
        gap_v = [0.0] * N
        dev_v = [0.0] * N
        for i in range(N):
            n = numbers[i]
            ls = last_seen.get(n)
            g = (draw_index - ls) if ls is not None else (draw_index + 1)
            if g < 0:
                g = 0
            r = g / E_INT
            if r > GAP_CAP:
                r = GAP_CAP
            gap_v[i] = r - 1.0
            dz = (mu - counts.get(n, 0)) / sdm
            if dz > 4.0:
                dz = 4.0
            elif dz < -4.0:
                dz = -4.0
            dev_v[i] = DEV_SIGN * dz

        wsg = W_SEC * gate
        scores = [wsg * a + W_GAP * b + W_DEV * c
                  for a, b, c in zip(sec_v, gap_v, dev_v)]

        # diagnostic: span max-min per componenta pentru ablatie
        span_sec = max(sec_v) - min(sec_v)
        span_gap = max(gap_v) - min(gap_v)
        span_dev = max(dev_v) - min(dev_v)
        DIAG['calls'] += 1
        DIAG['sec_span_sum'] += span_sec
        DIAG['gap_span_sum'] += span_gap
        DIAG['dev_span_sum'] += span_dev
        DIAG['entropy_sum'] += H
        DIAG['last'] = {'L': L, 'H': H, 'gate': gate,
                        'sec_span': span_sec, 'gap_span': span_gap,
                        'dev_span': span_dev}

        return sorted(numbers, key=lambda n: (-scores[n - 1], n))

    rank._diag = DIAG
    return rank