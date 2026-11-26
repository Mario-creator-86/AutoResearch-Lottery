NAME = "zpairgra"
DESCRIPTION = 'Afinitate conditionata pe perechi: matrice de co-occurinta pe ultimele ~20 extrageri (fara cea mai recenta), lift de co-calatorie cu seturile D1/D2, regim de repetare din suprapunerea extragerilor consecutive (E=5) + EWMA scurt si prior global slab.'
SOURCE = 'model + 2026-09-17'

import math


def create(config):
    P = float(config.number_range)
    WD = float(config.numbers_drawn)

    # statistici exacte 20/80
    q_pair = (WD * (WD - 1.0)) / (P * (P - 1.0))          # P(pereche intr-o extragere) ~ 0.0601
    e_ov = WD * WD / P                                     # suprapunere asteptata ~ 5
    sd_ov = math.sqrt(WD * (WD / P) * (1.0 - WD / P))      # sd binomial ~ 1.936

    W_L1 = 2.2     # pondere lift perechi cu D1
    W_L2 = 0.9     # pondere lift perechi cu D2
    W_REP = 1.5    # urmarire/evitare repetari, modulata de regim
    W_EW = 0.35    # momentum scurt EWMA
    W_CNT = 0.05   # prior global de frecventa (tie-break stabil)
    KD = 20        # fereastra pt matricea de perechi (fara cea mai recenta extragere)
    KE = 12        # orizont EWMA
    RG = 8         # nr suprapuneri consecutive pt regim
    LAM = 0.85
    GATE_K = 0.75

    ew_pow = [LAM ** k for k in range(KE)]

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, int(config.number_range) + 1))

        if counts:
            vals = [counts.get(n, 0) for n in numbers]
            mc = sum(vals) / float(len(vals))
            vc = sum((v - mc) * (v - mc) for v in vals) / float(len(vals))
            sc = math.sqrt(vc) if vc > 0 else 1.0
        else:
            mc, sc = 0.0, 1.0

        def fallback():
            return sorted(numbers, key=lambda n: (-counts.get(n, 0), n))

        if not recent_deque or len(recent_deque) < 3:
            return fallback()

        items = list(recent_deque)

        def mean_gap(s):
            tot = 0.0
            for n in s:
                ls = last_seen.get(n)
                tot += (draw_index - ls) if ls is not None else (draw_index + 1.0)
            return tot / max(1, len(s))

        # orientare agnostica: capatul cu gap mediu minim e cel mai recent
        if mean_gap(items[-1]) <= mean_gap(items[0]):
            seq = list(reversed(items))   # seq[0] = cea mai recenta extragere
        else:
            seq = list(items)

        # seturi-ancora din last_seen, independente de ordinea deque-ului
        gmin = None
        gaps = {}
        for n in numbers:
            ls = last_seen.get(n)
            if ls is not None:
                g = draw_index - ls
                gaps[n] = g
                if gmin is None or g < gmin:
                    gmin = g
        if gmin is None:
            return fallback()

        D1 = set(n for n, g in gaps.items() if g == gmin)      # ultima extragere
        D2 = set(n for n, g in gaps.items() if g == gmin + 1)  # penultima

        # regim: suprapuneri consecutive vs E=5
        n_ov = min(RG, len(seq) - 1)
        gate = 0.0
        if n_ov > 0:
            tot = 0
            for k in range(n_ov):
                tot += len(set(seq[k]) & set(seq[k + 1]))
            rbar = tot / float(n_ov)
            gate = math.tanh(GATE_K * (rbar - e_ov) * math.sqrt(n_ov) / sd_ov)

        # matrice de co-occurinta pe KD extrageri, EXCLuzand cea mai recenta (de-confounding)
        KP = min(KD, len(seq) - 1)
        pc = {}
        for k in range(1, KP + 1):
            lst = list(seq[k])
            L = len(lst)
            for i in range(L):
                a = lst[i]
                ai = a * 100
                for j in range(i + 1, L):
                    b = lst[j]
                    key = ai + b if a < b else b * 100 + a
                    pc[key] = pc.get(key, 0) + 1

        # EWMA de prezenta, centrata la valoarea asteptata
        KEFF = min(KE, len(seq))
        ew = {}
        for k in range(KEFF):
            w = ew_pow[k]
            for n in seq[k]:
                ew[n] = ew.get(n, 0.0) + w
        exp_ew = (WD / P) * sum(ew_pow[:KEFF])

        E1 = len(D1) * KP * q_pair
        E2 = len(D2) * KP * q_pair
        if E1 < 1e-9:
            E1 = 1e-9
        if E2 < 1e-9:
            E2 = 1e-9

        scores = {}
        for n in numbers:
            s1 = 0
            for m in D1:
                if m != n:
                    key = n * 100 + m if n < m else m * 100 + n
                    s1 += pc.get(key, 0)
            s2 = 0
            for m in D2:
                if m != n:
                    key = n * 100 + m if n < m else m * 100 + n
                    s2 += pc.get(key, 0)
            rep = 0.0
            if n in D1:
                rep += 1.0
            if n in D2:
                rep -= 0.5
            zc = (counts.get(n, 0) - mc) / sc
            scores[n] = (W_L1 * (s1 / E1) + W_L2 * (s2 / E2)
                         + W_REP * gate * rep
                         + W_EW * (ew.get(n, 0.0) - exp_ew)
                         + W_CNT * zc)

        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank