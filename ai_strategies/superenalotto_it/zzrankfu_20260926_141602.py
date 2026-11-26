NAME = "zzrankfu"
DESCRIPTION = 'Fuziune de ranguri Borda peste patru semnale: afinitate Markov pe perechi co-aparute in fereastra recenta (raport lift ancorat pe ultimele extrageri), deviatie EWMA fata de p, deviatie z pe termen lung si presiune saturanta de intarziere (Michaelis-Menten pe gap*rate); penalizare optionala pentru repetarea imediata si mod contrarian pe sectoare.'
SOURCE = 'model cantitativ rank-fusion + 2026-09-26'

import math

DIAG = {'orient': None, 'orient_flips': 0}

W_Z = 0.6      # deviatia z pe termen lung
W_MOM = 0.8    # deviatia EWMA fata de p
W_PAIR = 1.0   # afinitate Markov pe perechi (lift)
W_DUE = 0.7    # presiune de intarziere saturanta
W_REP = 0.15   # penalizare repetare din ultima extragere
W_SECT = 0.0   # mod contrarian pe sectoare (0 = dezactivat)
GAMMA = 0.93   # decay EWMA (jumatate de viata ~9 extrageri)
ANCHOR = 2     # cate extrageri recente formeaza setul-ancora Markov
KDUE = 1.0     # semi-saturatia raportului gap*rate


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    p = K / float(N)
    S = (N + 9) // 10

    def _pct(values):
        # percentila determinista in [0,1]; tie-break pe index (= numar crescator)
        m = len(values)
        order = sorted(range(m), key=lambda i: (values[i], i))
        out = [0.0] * m
        denom = float(m - 1) if m > 1 else 1.0
        for pos, i in enumerate(order):
            out[i] = pos / denom
        return out

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        T = float(max(draw_index, 1))
        mu = T * p
        sd = math.sqrt(max(T * p * (1.0 - p), 1e-12)) + 1e-9

        seq = list(recent_deque)
        L = len(seq)

        # auto-detectarea orientarii deque-ului (setul cel mai nou are recența medie minima)
        def mean_rec(st):
            if not st:
                return float('inf')
            tot = 0.0
            for n in st:
                ls = last_seen.get(n)
                tot += (draw_index - ls) if ls is not None else float(draw_index)
            return tot / float(len(st))

        newest_last = True
        if L >= 2:
            newest_last = mean_rec(seq[-1]) <= mean_rec(seq[0])
        orient = 'newest_last' if newest_last else 'newest_first'
        if DIAG['orient'] is not None and DIAG['orient'] != orient:
            DIAG['orient_flips'] += 1
        DIAG['orient'] = orient

        RW = int(recent_window) if recent_window else 0
        if RW > 0 and L > RW:
            seq = seq[-RW:] if newest_last else seq[:RW]
            L = len(seq)

        if newest_last:
            ordered = seq          # vechi -> nou
        else:
            ordered = list(reversed(seq))

        ew = dict((n, 0.0) for n in numbers)
        win = dict((n, 0.0) for n in numbers)
        sec_hits = [0.0] * S
        pc = {}
        if L > 0:
            norm = (1.0 - GAMMA ** L) / (1.0 - GAMMA)
            for i in range(L):
                dr = ordered[i]
                age = L - 1 - i
                w = GAMMA ** age
                ds = []
                for n in dr:
                    if 1 <= n <= N:
                        ds.append(n)
                        ew[n] += w
                        win[n] += 1.0
                        sec_hits[(n - 1) // 10] += 1.0
                a = len(ds)
                for x in range(a):
                    for y in range(x + 1, a):
                        u, v = ds[x], ds[y]
                        if u > v:
                            u, v = v, u
                        key = (u, v)
                        if key in pc:
                            pc[key] += 1
                        else:
                            pc[key] = 1
            for n in numbers:
                ew[n] /= norm
                win[n] /= float(L)

        # setul-ancora: numerele din ultimele ANCHOR extrageri, ponderate 1/(1+varsta)
        anchor_w = {}
        if L > 0:
            for t in range(min(ANCHOR, L)):
                dr = ordered[L - 1 - t]
                for n in dr:
                    if 1 <= n <= N and n not in anchor_w:
                        anchor_w[n] = 1.0 / (1.0 + float(t))

        # afinitate Markov: lift al perechilor (n,m) pentru m in ancora, ponderat de recența lui m
        aff = dict((n, 0.0) for n in numbers)
        if anchor_w:
            items = sorted(anchor_w.items())
            for n in numbers:
                fn = win[n]
                if fn <= 0.0:
                    continue
                acc = 0.0
                for m, wm in items:
                    if m == n:
                        continue
                    key = (n, m) if n < m else (m, n)
                    c = pc.get(key, 0)
                    if c:
                        acc += (float(c) / (fn * win[m] * L + 1e-9)) * wm
                aff[n] = acc

        # presiune de intarziere Michaelis-Menten pe gap*rate (saturanta, doar pozitiva)
        due = {}
        for n in numbers:
            ls = last_seen.get(n)
            gap = draw_index if ls is None else (draw_index - ls)
            if gap < 0:
                gap = 0
            r = max(float(counts.get(n, 0)) / T, 1.0 / T)
            x = gap * r
            due[n] = x / (x + KDUE)

        # mod contrarian pe sectoare: avantajeaza sectoarele sub-reprezentate in fereastra
        sect_contr = [0.0] * S
        if L > 0:
            tot_hits = 0.0
            for s in range(S):
                tot_hits += sec_hits[s]
            if tot_hits > 0:
                for s in range(S):
                    lo = 10 * s + 1
                    hi = min(10 * s + 10, N)
                    size = hi - lo + 1
                    sect_contr[s] = size / float(N) - (sec_hits[s] / tot_hits)

        rz = _pct([(float(counts.get(n, 0)) - mu) / sd for n in numbers])
        rm = _pct([ew[n] - p for n in numbers])
        ra = _pct([aff[n] for n in numbers])
        rd = _pct([due[n] for n in numbers])
        rs = _pct([sect_contr[(n - 1) // 10] for n in numbers])

        last_set = ordered[-1] if L > 0 else set()
        scores = {}
        for idx in range(N):
            n = numbers[idx]
            rep = 1.0 if n in last_set else 0.0
            scores[n] = (W_Z * rz[idx] + W_MOM * rm[idx] + W_PAIR * ra[idx]
                         + W_DUE * rd[idx] + W_SECT * rs[idx] - W_REP * rep)
        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank