NAME = "zzpairfl"
DESCRIPTION = 'Flux de afinitate pe graful de co-ocurenta: caldura propagata de la ultimele ANCHOR extrageri prin similitudini de perechi (ponderate exponential in fereastra recenta), plus un termen Markov de tranzitie estimat din extragerile consecutive si reversion pe profilul draw-urilor recente (suma si paritate). Toate componentele raw sunt z-normalizate inainte de combinare; frecventa pe termen lung ramane doar ancora slaba.'
SOURCE = 'model cantitativ + 2026-09-16'

import math
from itertools import combinations

DIAG = {'heat_span': 0.0, 'trans_span': 0.0, 'tilt_sum': 0.0, 'tilt_par': 0.0, 'L': 0}

# hiperparametri expusi pentru grid-search ulterior
W_HEAT = 1.00   # greutate flux co-ocurenta (ancorat pe ultimele ANCHOR extrageri)
W_TRANS = 0.65  # greutate tranzitie Markov ultimul draw -> urmatorul
W_PROF = 0.50   # greutate reversion suma
W_PAR = 0.30    # greutate reversion paritate
W_CNT = 0.25    # ancora frecventa pe termen lung
GAMMA_P = 0.97  # decay pentru co-ocurenta / tranzitii (half-life ~23 extrageri)
ANCHOR = 2      # cate extrageri recente sunt surse de caldura
M_PROF = 20     # cate extrageri pentru profilul suma/paritate
ZCAP = 3.0
EPS = 1e-9


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    mid = (N + 1) / 2.0
    E_sum = K * (N + 1) / 2.0
    var_sum = K * (N + 1) * (N - K) / 12.0  # varianta sumei unui draw uniform fara reinlocuire

    def _zc(v):
        return -ZCAP if v < -ZCAP else (ZCAP if v > ZCAP else v)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        L = len(recent_deque)
        DIAG['L'] = L

        # auto-detectarea orientarii deque-ului (aceeasi tehnica robusta)
        def mean_recency(st):
            if not st:
                return float('inf')
            tot = 0.0
            for x in st:
                ls = last_seen.get(x)
                tot += (draw_index - ls) if ls is not None else float(draw_index)
            return tot / float(len(st))

        newest_last = True
        if L >= 2:
            newest_last = mean_recency(recent_deque[-1]) <= mean_recency(recent_deque[0])
        seq = list(recent_deque)
        if newest_last and L >= 2:
            seq.reverse()  # seq[0] = cel mai nou draw

        # marginale ponderate + co-ocurenta de perechi + tranzitii consecutive
        marg = dict.fromkeys(numbers, 0.0)
        co = {}
        tc = {}
        prev = None
        for age in range(L):
            dr = seq[age]
            w = GAMMA_P ** age
            ds = [x for x in dr if 1 <= x <= N]
            for x in ds:
                marg[x] += w
            for a, b in combinations(ds, 2):
                key = (a, b) if a < b else (b, a)
                co[key] = co.get(key, 0.0) + w
            if prev is not None:
                for m in prev:
                    for x in ds:
                        tkey = (m, x)
                        tc[tkey] = tc.get(tkey, 0.0) + w
            prev = ds

        heat_raw = dict.fromkeys(numbers, 0.0)
        trans_raw = dict.fromkeys(numbers, 0.0)
        if L > 0:
            # (a) heat: similitudine normalizata fata de numerele din ultimele ANCHOR draw-uri
            for ai in range(min(ANCHOR, L)):
                adr = seq[ai]
                aw = 1.0 / (ai + 1.0)
                for m in adr:
                    if m < 1 or m > N:
                        continue
                    mm = marg.get(m, 0.0)
                    if mm <= EPS:
                        continue
                    for n in numbers:
                        if n == m:
                            continue
                        mn = marg[n]
                        if mn <= EPS:
                            continue
                        c = co.get((n, m) if n < m else (m, n), 0.0)
                        if c <= 0.0:
                            continue
                        sim = c / math.sqrt(mn * mm)
                        if sim > 1.0:
                            sim = 1.0
                        heat_raw[n] += aw * sim
            # (b) Markov: cat de des a urmat n dupa un draw care continea m (m din ultimul draw)
            last = seq[0]
            for m in last:
                if m < 1 or m > N:
                    continue
                mm = marg.get(m, 0.0)
                if mm <= EPS:
                    continue
                for n in numbers:
                    if n == m:
                        continue
                    t = tc.get((m, n), 0.0)
                    if t > 0.0:
                        trans_raw[n] += t / mm

        # (c) reversion pe profilul draw-urilor recente: suma si paritate
        tilt_sum = 0.0
        tilt_par = 0.0
        if L > 0:
            M = min(M_PROF, L)
            sub = seq[:M]
            ms = 0.0
            odd = 0
            for d in sub:
                ms += sum(d)
                for x in d:
                    if x % 2 == 1:
                        odd += 1
            ms /= float(M)
            sd_mean = math.sqrt(var_sum / M) + EPS
            tilt_sum = math.tanh((E_sum - ms) / sd_mean)
            tot = M * K
            share = odd / float(tot)
            sd_sh = math.sqrt(0.25 / tot) + EPS
            tilt_par = math.tanh((0.5 - share) / (1.5 * sd_sh))

        def znorm(d):
            mu = 0.0
            for n in numbers:
                mu += d[n]
            mu /= float(N)
            var = 0.0
            for n in numbers:
                dv = d[n] - mu
                var += dv * dv
            var /= float(N)
            sd = math.sqrt(var)
            if sd <= EPS:
                return dict.fromkeys(numbers, 0.0), 0.0
            z = {}
            lo = d[numbers[0]]
            hi = lo
            for n in numbers:
                v = d[n]
                if v < lo:
                    lo = v
                if v > hi:
                    hi = v
                z[n] = _zc((v - mu) / sd)
            return z, (hi - lo)

        zc, _ = znorm({n: float(counts.get(n, 0)) for n in numbers})
        zh, span_h = znorm(heat_raw)
        zt, span_t = znorm(trans_raw)
        DIAG['heat_span'] = span_h
        DIAG['trans_span'] = span_t
        DIAG['tilt_sum'] = tilt_sum
        DIAG['tilt_par'] = tilt_par

        out = []
        for n in numbers:
            s = (W_CNT * zc[n]
                 + W_HEAT * zh[n]
                 + W_TRANS * zt[n]
                 + W_PROF * tilt_sum * (n - mid) / mid
                 + W_PAR * (-tilt_par) * (1.0 if n % 2 == 0 else -1.0))
            out.append((-s, n))
        out.sort()
        return [n for _, n in out]

    return rank