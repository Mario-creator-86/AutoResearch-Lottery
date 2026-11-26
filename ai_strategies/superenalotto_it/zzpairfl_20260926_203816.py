NAME = "zzpairfl"
DESCRIPTION = "Flux de perechi de ordinul doi: co-aparitii cu decay geometric in fereastra recenta, conditionate de membrii ultimei extrageri, combinate prin z-score cu propensity bayesiana de repetare seriala, frecventa in fereastra rapida si baza de frecventa globala; secetele extreme sunt penalizate liniar (anti-revenire la medie)."
SOURCE = "model cantitativ + 2026-09-26"

import math

DIAG = {'last_max_score': 0.0}

# hiperparametri expusi pentru grid-search ulterior
W_PAIR = 1.0        # fluxul de perechi cu membrii ultimei extrageri
W_REP = 0.7         # propensity de repetare seriala (clustering)
W_FAST = 0.6        # frecventa in fereastra rapida
W_BASE = 0.3        # frecventa globala (shrinkage)
W_DROUGHT = 0.45    # penalizare seceta excesiva
DECAY = 0.97        # decay geometric pe vechimea extragerii
CO_CAP = 120        # plafon extrageri pentru statistica de perechi
F_FAST = 12         # marimea ferestrei rapide
PRIOR_A = 6.0       # ponderea priorului Beta in posteriorul de repetare
CAP_DROUGHT = 30.0  # prag de seceta peste care se aplica penalizarea


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    p0 = K / float(N)

    def zmap(vals):
        m = len(vals)
        if m == 0:
            return []
        mu = sum(vals) / float(m)
        var = sum((v - mu) * (v - mu) for v in vals) / float(m)
        sd = math.sqrt(var) if var > 1e-12 else 1.0
        return [(v - mu) / sd for v in vals]

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        sets = [s for s in recent_deque if s]
        if not sets or draw_index <= 0:
            base = {n: counts.get(n, 0) for n in numbers}
            return sorted(numbers, key=lambda n: (-base[n], n))

        # orientare-agnostic: seturile sortate dupa recenza medie; cele mai noi primele
        def set_age(s):
            tot = 0.0
            for n in s:
                ls = last_seen.get(n)
                tot += (draw_index - ls) if ls is not None else float(draw_index)
            return tot / float(len(s))

        order = sorted(range(len(sets)), key=lambda i: (set_age(sets[i]), i))
        capped = [sets[i] for i in order[:CO_CAP]]
        C = len(capped)

        # co-aparitii cu decay geometric pe vechimea relativa
        co = {}
        occ = dict.fromkeys(numbers, 0)
        for age in range(C):
            s = capped[age]
            w = DECAY ** age
            lst = sorted(s)
            for a_i in range(len(lst)):
                u = lst[a_i]
                occ[u] += 1
                for b_i in range(a_i + 1, len(lst)):
                    v = lst[b_i]
                    key = (u, v)
                    co[key] = co.get(key, 0.0) + w

        partners = {}
        for (u, v), w in co.items():
            d = partners.get(u)
            if d is None:
                d = partners[u] = {}
            d[v] = w
            d2 = partners.get(v)
            if d2 is None:
                d2 = partners[v] = {}
            d2[u] = w

        L = capped[0]

        # 1) flux de perechi catre membrii ultimei extrageri
        pair_raw = []
        for n in numbers:
            d = partners.get(n)
            tot = 0.0
            if d:
                for m, w in d.items():
                    if m in L:
                        tot += w
            pair_raw.append(tot)
        z_pair = zmap(pair_raw)

        # 2) propensity de repetare: posterior Beta pe P(n in t+1 | n in t)
        rep_hits = dict.fromkeys(numbers, 0)
        for age in range(C - 1):
            cur = capped[age]
            nxt = capped[age + 1]
            for n in cur:
                if n in nxt:
                    rep_hits[n] += 1
        rep_comp = []
        for n in numbers:
            o = occ[n]
            rep_comp.append((rep_hits[n] + PRIOR_A * p0) / (o + PRIOR_A) - p0)
        z_rep = zmap(rep_comp)

        # 3) frecventa in fereastra rapida (cele mai noi F extrageri)
        F = min(F_FAST, C)
        fast_occ = dict.fromkeys(numbers, 0)
        for s in capped[:F]:
            for n in s:
                fast_occ[n] += 1
        z_fast = zmap([float(fast_occ[n]) for n in numbers])

        # 4) baza de frecventa globala
        z_base = zmap([float(counts.get(n, 0)) for n in numbers])

        # 5) seceta excesiva (anti-revenire la medie, opus hazardului de gap)
        drou = []
        cap4 = int(4 * CAP_DROUGHT)
        for n in numbers:
            ls = last_seen.get(n)
            gap = (draw_index - ls) if ls is not None else min(draw_index + 1, cap4)
            drou.append(max(0.0, gap - CAP_DROUGHT) / CAP_DROUGHT)

        scores = {}
        best = -1e18
        for idx, n in enumerate(numbers):
            s = (W_PAIR * z_pair[idx] + W_REP * z_rep[idx] +
                 W_FAST * z_fast[idx] + W_BASE * z_base[idx] -
                 W_DROUGHT * drou[idx])
            scores[n] = s
            if s > best:
                best = s
        DIAG['last_max_score'] = best
        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank