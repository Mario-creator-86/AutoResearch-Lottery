NAME = "zpairlif"
DESCRIPTION = 'Pair-lift pe cohortele D1/D2: afinitatea fiecarui numar cu numerele din ultima si penultima extragere, masurata ca z-score binomial exact pe co-ocurente in fereastra recenta; plus kernel de recenta exp(-varsta/KD) centrat pe null si persistenta pe fereastra (proxy EWMA standardizat). Un gate tanh, calibrat pe densitatea de repetari din ultimele RG extrageri fata de asteptarea combinatorica exacta, moduleaza blocul momentum si il inverseaza in regim anti-persistent.'
SOURCE = 'model + 2026-09-17'

import math

# Parametri impliciti la nivel de modul (editabili prin patching la grid-run):
W_L1 = 2.2    # pondere pair-lift D1 (cohorta ultimei extrageri)
W_L2 = 0.6    # pondere pair-lift D2 (penultima extragere) - prima candidata la ablare
W_REP = 1.5   # pondere kernel recenta proprie
W_EW = 0.35   # pondere persistenta pe fereastra (ancora mereu activa)
KD = 20.0     # orizontul kernelului de recenta
RG = 8        # fereastra de detectie a regimului (distincte vs asteptare)
GATE_K = 0.75 # castigul gate-ului tanh
ZCAP = 3.0    # clamp pe z-score-urile individuale


def create(config):
    P = int(config.number_range)
    Wd = int(config.numbers_drawn)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, P + 1))
        if last_seen is None:
            last_seen = {}
        cnt_w = dict.fromkeys(numbers, 0)
        co = [[0] * (P + 1) for _ in range(P + 1)]
        rw = 0
        if recent_deque:
            for s in recent_deque:
                if not s:
                    continue
                lst = sorted(x for x in s if 1 <= x <= P)
                if not lst:
                    continue
                rw += 1
                for a in lst:
                    cnt_w[a] += 1
                L = len(lst)
                for i in range(L - 1):
                    row = co[lst[i]]
                    for k in range(i + 1, L):
                        row[lst[k]] += 1
        p1 = (Wd / P) if P > 0 else 0.0
        p2 = (Wd * (Wd - 1) / (P * (P - 1))) if (P > 1 and Wd >= 2) else 0.0
        # persistenta pe fereastra (binomial exact)
        if rw > 0 and 0.0 < p1 < 1.0:
            m1 = rw * p1
            s1 = math.sqrt(rw * p1 * (1.0 - p1))
            zc = {}
            for n in numbers:
                zc[n] = max(-ZCAP, min(ZCAP, (cnt_w[n] - m1) / s1))
        else:
            zc = dict.fromkeys(numbers, 0.0)
        # varste din last_seen (fara a depinde de ordinea din deque)
        ages = {}
        for n in numbers:
            ls = last_seen.get(n)
            ages[n] = None if ls is None else max(draw_index - ls, 0)
        cohort1 = [n for n in numbers if ages[n] == 1]
        if cohort1:
            cohort2 = [n for n in numbers if ages[n] == 2]
            lo_reg = 1
        else:
            # fallback: last_seen indexat pe extragerea curenta
            cohort1 = [n for n in numbers if ages[n] == 0]
            cohort2 = [n for n in numbers if ages[n] == 1]
            lo_reg = 0

        def cohort_lift(coh):
            out = dict.fromkeys(numbers, 0.0)
            if not coh or rw <= 0 or not (0.0 < p2 < 1.0):
                return out
            m2 = rw * p2
            s2 = math.sqrt(rw * p2 * (1.0 - p2))
            for n in numbers:
                acc = 0.0
                for j in coh:
                    if j == n:
                        continue
                    if n < j:
                        z = (co[n][j] - m2) / s2
                    else:
                        z = (co[j][n] - m2) / s2
                    if z > ZCAP:
                        z = ZCAP
                    elif z < -ZCAP:
                        z = -ZCAP
                    acc += z
                out[n] = acc
            return out

        lift1 = cohort_lift(cohort1)
        lift2 = cohort_lift(cohort2)
        # kernel recenta, centrat pe null i.i.d. (varsta ~ geometrica)
        lam = math.exp(-1.0 / KD) if KD > 0 else 0.0
        if 0.0 < p1 < 1.0 and 0.0 < lam < 1.0:
            e1 = p1 * lam / (1.0 - (1.0 - p1) * lam)
            e2 = p1 * lam * lam / (1.0 - (1.0 - p1) * lam * lam)
            srep = math.sqrt(max(e2 - e1 * e1, 1e-12))
        else:
            e1 = 0.0
            srep = 1.0
        rep = {}
        for n in numbers:
            a = ages[n]
            r = (lam ** a) if (a is not None and lam > 0.0) else 0.0
            rep[n] = max(-ZCAP, min(ZCAP, (r - e1) / srep))
        # gate de regim: distincte in ultimele RG extrageri vs asteptare exacta
        if RG and RG > 0:
            m = int(RG)
            qn = (1.0 - p1) ** m
            q2n = (1.0 - 2.0 * p1 + p2) ** m
            q1 = 1.0 - qn
            q2 = 1.0 - 2.0 * qn + q2n
            e_d = P * q1
            var_d = P * q1 * (1.0 - q1) + P * (P - 1) * (q2 - q1 * q1)
            sd_d = math.sqrt(max(var_d, 1e-12))
            H = 0
            for n in numbers:
                a = ages[n]
                if a is not None and lo_reg <= a <= lo_reg + m - 1:
                    H += 1
            s_mom = max(-3.0, min(3.0, (e_d - H) / sd_d))
            g = math.tanh(GATE_K * s_mom)
        else:
            e_d = 0.0
            sd_d = 0.0
            H = 0
            s_mom = 0.0
            g = 0.0
        score = {}
        for n in numbers:
            score[n] = W_EW * zc[n] + g * (W_L1 * lift1[n] + W_L2 * lift2[n] + W_REP * rep[n])
        rank._diag = {
            'gate': g, 's_mom': s_mom, 'H': H, 'E_D': e_d, 'sd_D': sd_d, 'rw': rw,
            'win': recent_window,
            'comp': {'lift1': lift1, 'lift2': lift2, 'rep': rep, 'zc': zc, 'score': score},
        }
        return sorted(numbers, key=lambda n: (-score[n], n))

    return rank