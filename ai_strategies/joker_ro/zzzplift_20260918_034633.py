NAME = "zzzplift"
DESCRIPTION = 'Model Markov de tranzitie asupra ultimelor extrageri: lift lag-1 si lag-2 estimat din perechi de extrageri aflate la distanta 1, respectiv 2, in fereastra RG, fiecare celula shrinked spre independenta cu pseudocount GATE_K, surpriza (observat - asteptat) sumata peste membrii ultimei extrageri; plus REP = ecoul ultimei extrageri si EW = EWMA cu half-life KD pe prezentele din fereastra recenta; componente z-standardizate per numar, cu span de contributie per componenta expus in rank._diag si rank._diag_cum.'
SOURCE = 'model + 2026-09-17'

import math

# ---- constante tunabile (grid de ablatie din ledger) ----
# W_L1: greutate lift lag-1, grid {1.1, 2.2, 3.3}
W_L1 = 2.2
# W_L2: greutate lift lag-2; prima ablatie: W_L2 = 0 (posibil redundant cu L1)
W_L2 = 1.1
# W_REP: greutate ecou D1, grid {0, 1.5, 3.0}; redundant cu L1 => 0
W_REP = 1.5
# W_EW: greutate EWMA, grid {0, 0.35, 0.7}; redundant cu lift => 0
W_EW = 0.35
# KD: half-life EWMA in extrageri, grid {12, 20, 30}
KD = 20
# RG: lungimea esantionului de tranzitii, grid {4, 8, 16}
RG = 8
# GATE_K: pseudocount shrinkage spre independenta, grid {0.4, 0.75, 1.2}
GATE_K = 0.75
# LIFT_SIGN: +1 momentum pe tranzitii, -1 reversion (flip doar daca momentum pierde clar)
LIFT_SIGN = 1.0
# EW_SIGN: +1 momentum EWMA, -1 reversion
EW_SIGN = 1.0

COMPS = ('lift1', 'lift2', 'rep', 'ew', 'cnt', 'total')


def create(config):
    N = int(config.number_range)
    KDRAW = int(config.numbers_drawn)
    p = KDRAW / float(N) if N > 0 else 0.0
    kd = max(float(KD), 0.5)
    gate = max(float(GATE_K), 1e-06)
    rg_want = max(2, int(RG))
    neg = -10 ** 9

    per_call = {'spans': dict((c, 0.0) for c in COMPS),
                'span_mean': dict((c, 0.0) for c in COMPS),
                'samples': {},
                'weights': {'W_L1': W_L1, 'W_L2': W_L2, 'W_REP': W_REP, 'W_EW': W_EW,
                            'KD': kd, 'RG': RG, 'GATE_K': gate, 'LIFT_SIGN': LIFT_SIGN,
                            'EW_SIGN': EW_SIGN},
                'orient_checks': 0, 'orient_flips': 0, 'orient_rule': 'lastseen-hit+mean',
                'w_eff': 0.0}
    cum = {'calls': 0, 'pairs1_sum': 0, 'pairs2_sum': 0,
           'span_sum': dict((c, 0.0) for c in COMPS),
           'span_max': {}, 'span_min': {}}

    def _z(vec):
        n = len(vec)
        if n == 0:
            return [0.0]
        m = 0.0
        for v in vec:
            m += v
        m /= n
        var = 0.0
        for v in vec:
            d = v - m
            var += d * d
        sd = math.sqrt(var / n)
        if sd < 1e-12:
            return [0.0] * n
        return [(v - m) / sd for v in vec]

    def _span(vec):
        lo = vec[0]
        hi = vec[0]
        for v in vec:
            if v < lo:
                lo = v
            if v > hi:
                hi = v
        return hi - lo

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        D = int(draw_index)
        numbers = list(range(1, N + 1))
        seq = [set(s) for s in recent_deque]
        Wd = len(seq)

        # orientare deque: seq[-1] trebuie sa fie cea mai noua extragere (last_seen == D-1)
        if Wd >= 2 and D >= 1:
            per_call['orient_checks'] += 1
            newest0 = seq[-1]
            oldest0 = seq[0]
            so = 0
            sf = 0
            for n in newest0:
                if last_seen.get(n, neg) == D - 1:
                    so += 1
            for n in oldest0:
                if last_seen.get(n, neg) == D - 1:
                    sf += 1
            flip = False
            if sf > so:
                flip = True
            elif sf == so:
                mo = None
                mf = None
                acc = 0.0
                c = 0
                for n in newest0:
                    if n in last_seen:
                        acc += last_seen[n]
                        c += 1
                if c > 0:
                    mo = acc / c
                acc = 0.0
                c = 0
                for n in oldest0:
                    if n in last_seen:
                        acc += last_seen[n]
                        c += 1
                if c > 0:
                    mf = acc / c
                if mo is not None and mf is not None and mf > mo:
                    flip = True
            if flip:
                seq.reverse()
                per_call['orient_flips'] += 1

        newest = seq[-1] if Wd > 0 else set()
        members = []
        for n in newest:
            if 1 <= n <= N:
                members.append(n)
        members.sort()

        # esantion de tranzitii: ultimele rg extrageri, vechi -> nou
        rg = min(rg_want, Wd) if Wd > 0 else 0
        tr = seq[-rg:] if rg > 0 else []
        T1 = {}
        R1 = {}
        np1 = 0
        for i in range(len(tr) - 1):
            A = tr[i]
            B = tr[i + 1]
            np1 += 1
            for m in A:
                R1[m] = R1.get(m, 0) + 1
                for n in B:
                    k = (m, n)
                    T1[k] = T1.get(k, 0) + 1
        T2 = {}
        R2 = {}
        np2 = 0
        for i in range(len(tr) - 2):
            A = tr[i]
            B = tr[i + 2]
            np2 += 1
            for m in A:
                R2[m] = R2.get(m, 0) + 1
                for n in B:
                    k = (m, n)
                    T2[k] = T2.get(k, 0) + 1

        # lift lag-1 / lag-2: surpriza de tranzitie cu shrinkage Bayes, sumata pe membrii D1
        s1 = [0.0] * N
        s2 = [0.0] * N
        if p > 0.0:
            for j in range(N):
                num = j + 1
                a1 = 0.0
                for m in members:
                    r1 = R1.get(m, 0)
                    if r1 > 0:
                        t1 = T1.get((m, num), 0)
                        a1 += (t1 - p * r1) / (p * (r1 + gate))
                s1[j] = LIFT_SIGN * a1
                a2 = 0.0
                for m in members:
                    r2 = R2.get(m, 0)
                    if r2 > 0:
                        t2 = T2.get((m, num), 0)
                        a2 += (t2 - p * r2) / (p * (r2 + gate))
                s2[j] = LIFT_SIGN * a2

        # REP: ecoul ultimei extrageri
        rep = [0.0] * N
        for m in members:
            rep[m - 1] = 1.0

        # EW: EWMA cu half-life KD pe prezentele din fereastra recenta
        ew = [0.0] * N
        sw = 0.0
        for i in range(Wd):
            age = Wd - 1 - i
            w = 1.0 if age == 0 else 0.5 ** (age / kd)
            sw += w
            for n in sorted(seq[i]):
                if 1 <= n <= N:
                    ew[n - 1] += w
        per_call['w_eff'] = sw

        z1 = _z(s1)
        z2 = _z(s2)
        zr = _z(rep)
        ze = _z(ew)
        zc = _z([float(counts.get(n, 0)) for n in numbers])

        tot = [0.0] * N
        for j in range(N):
            tot[j] = W_L1 * z1[j] + W_L2 * z2[j] + W_REP * zr[j] + W_EW * ze[j]

        # diag: span de contributie per componenta (max-min peste cele N numere)
        sp = per_call['spans']
        sp['lift1'] = _span(z1)
        sp['lift2'] = _span(z2)
        sp['rep'] = _span(zr)
        sp['ew'] = _span(ze)
        sp['cnt'] = _span(zc)
        sp['total'] = _span(tot)
        per_call['samples'] = {'pairs1': np1, 'pairs2': np2,
                               'r1_mass': sum(R1.values()), 'r2_mass': sum(R2.values()),
                               'newest_size': len(members), 'window': Wd, 'rg_used': rg}
        cum['calls'] += 1
        cum['pairs1_sum'] += np1
        cum['pairs2_sum'] += np2
        for c in COMPS:
            v = sp[c]
            cum['span_sum'][c] += v
            mx = cum['span_max'].get(c)
            mn = cum['span_min'].get(c)
            cum['span_max'][c] = v if mx is None else (v if v > mx else mx)
            cum['span_min'][c] = v if mn is None else (v if v < mn else mn)
            per_call['span_mean'][c] = cum['span_sum'][c] / cum['calls']

        sc = {}
        for j in range(N):
            sc[j + 1] = tot[j]
        return sorted(numbers, key=lambda n: (-sc[n], n))

    rank._diag = per_call
    rank._diag_cum = cum
    return rank