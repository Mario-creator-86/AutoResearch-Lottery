NAME = "zzpdpc"
DESCRIPTION = '4 componente z-standardizate combinate liniar: perechi decay-uite ancorate pe ultimele K_TAIL extrageri, deviatie EWMA a aparitiilor, kernel gaussian de vecinatate cu ultima extragere si aliniere de faza a ciclului propriu de recurenta; ponderi la nivel de modul pentru grid-search + audit DIAG orient_flips.'
SOURCE = 'quant-research 2026-09-08 (pair/dev/pos/cad, ablation-ready)'

import math

DIAG = {'calls': 0, 'orient_flips': 0, 'orient_unknown': 0}

# Hyper-parametri (intervale planificate pentru grid-search):
# W_PAIR [0,1.5], W_DEV [-0.8,1.0], W_POS [0,1], W_CAD [0,1],
# GAMMA_P [0.90,0.99], K_TAIL in {1,2,3,5}, SIGMA_POS [4,12]
W_PAIR = 0.50
W_DEV = 0.50
W_POS = 0.20
W_CAD = 0.25
GAMMA_P = 0.94
K_TAIL = 2
SIGMA_POS = 6.0
CAD_SIGMA = 0.5
EPS = 1e-12


def create(config):
    # config: LotteryConfig — folosim doar atributele number_range / numbers_drawn
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    p = (K / float(N)) if N > 0 else 0.0

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        DIAG['calls'] += 1
        numbers = list(range(1, N + 1))
        draws = list(recent_deque)
        if recent_window and recent_window > 0 and len(draws) > recent_window:
            draws = draws[-recent_window:]
        m = len(draws)

        # Audit de orientare: cea mai noua extragere trebuie sa fie la coada (newest_last)
        if m:
            expected_new = set()
            tgt = draw_index - 1
            for n in numbers:
                if last_seen.get(n, -10 ** 9) == tgt:
                    expected_new.add(n)
            if expected_new:
                if set(draws[-1]) == expected_new:
                    pass
                elif set(draws[0]) == expected_new:
                    DIAG['orient_flips'] += 1
                else:
                    DIAG['orient_unknown'] += 1

        # Varste sub orientarea newest_last: index i => varsta m-1-i
        gammas = [GAMMA_P ** (m - 1 - i) for i in range(m)]
        gsum = 0.0
        for g in gammas:
            gsum += g

        # (dev) deviatia decay-uita a aparitiilor fata de medie
        wobs = dict((n, 0.0) for n in numbers)
        for i in range(m):
            g = gammas[i]
            if g <= 0.0:
                continue
            for n in draws[i]:
                if 1 <= n <= N:
                    wobs[n] += g
        wmu = p * gsum
        wsig = math.sqrt(wmu * (1.0 - p)) if wmu > EPS else 0.0
        dev_raw = dict((n, 0.0) for n in numbers)
        if wsig > EPS:
            inv = 1.0 / wsig
            for n in numbers:
                dev_raw[n] = (wobs[n] - wmu) * inv

        # (pair) co-aparitii decay-uite in fereastra recenta
        pmat = dict((n, {}) for n in numbers)
        for i in range(m):
            g = gammas[i]
            if g <= 0.0:
                continue
            ds = sorted(draws[i])
            L = len(ds)
            for ai in range(L):
                a = ds[ai]
                if a < 1 or a > N:
                    continue
                pa = pmat[a]
                for bi in range(ai + 1, L):
                    b = ds[bi]
                    if b < 1 or b > N:
                        continue
                    pa[b] = pa.get(b, 0.0) + g
                    pb = pmat[b]
                    pb[a] = pb.get(a, 0.0) + g
        anchors = {}
        for j in range(min(K_TAIL, m)):
            dj = draws[m - 1 - j]
            wj = GAMMA_P ** j
            for n in dj:
                if 1 <= n <= N:
                    anchors[n] = anchors.get(n, 0.0) + wj
        pair_raw = dict((n, 0.0) for n in numbers)
        if anchors:
            akeys = sorted(anchors)
            for n in numbers:
                if n in anchors:
                    pn = pmat[n]
                    s = 0.0
                    for a in akeys:
                        if a == n:
                            continue
                        s += pn.get(a, 0.0) * anchors[a]
                    pair_raw[n] = s

        # (pos) afinitate gaussiana cu numerele din ultima extragere
        pos_raw = dict((n, 0.0) for n in numbers)
        if m:
            last = draws[-1]
            inv2s2 = 1.0 / (2.0 * SIGMA_POS * SIGMA_POS)
            for n in numbers:
                s = 0.0
                for j in last:
                    if j == n or j < 1 or j > N:
                        continue
                    df = float(n - j)
                    s += math.exp(-df * df * inv2s2)
                pos_raw[n] = s

        # (cad) aliniere de faza cu ciclul propriu de recurenta
        cad_raw = dict((n, 0.0) for n in numbers)
        if draw_index > 0:
            for n in numbers:
                c = counts.get(n, 0)
                if c <= 0:
                    continue
                cycle = draw_index / float(c)
                ls = last_seen.get(n, None)
                gap = draw_index if ls is None else draw_index - ls
                if gap < 0:
                    gap = 0
                rel = (gap - cycle) / cycle
                cad_raw[n] = math.exp(-0.5 * (rel / CAD_SIGMA) ** 2)

        def zscore(d):
            vals = list(d.values())
            mu = sum(vals) / float(len(vals))
            var = 0.0
            for v in vals:
                dv = v - mu
                var += dv * dv
            sd = math.sqrt(var / float(len(vals)))
            if sd < EPS:
                return dict((k, 0.0) for k in d)
            inv = 1.0 / sd
            return dict((k, (v - mu) * inv) for k, v in d.items())

        zp = zscore(pair_raw)
        zd = zscore(dev_raw)
        zpos = zscore(pos_raw)
        zc = zscore(cad_raw)

        scores = {}
        for n in numbers:
            scores[n] = (W_PAIR * zp[n] + W_DEV * zd[n]
                         + W_POS * zpos[n] + W_CAD * zc[n])
        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank