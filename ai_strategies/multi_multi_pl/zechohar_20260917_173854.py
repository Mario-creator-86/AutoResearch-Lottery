import math

NAME = "zechohar"
DESCRIPTION = 'Fara numaratori globale: deviatie EWMA a ritmului de aparitie vs baseline 20/80 (0.25), ecou de co-aparitie pereche seed-uit pe ultimele 2 extrageri, reversiune de sector (z hipergeometric) si bump dublu-armonic pe ciclul de intarziere (~4 si ~8 extrageri), combinate ne-liniar cu tanh + interactiune momentum×ecou.'
SOURCE = 'model + 2026-09-16'


def create(config):
    P = config.number_range            # ex. 80
    W = config.numbers_drawn           # ex. 20
    base_rate = float(W) / float(P)    # rata asteptata de aparitie per extragere
    gap0 = float(P) / float(W)         # gap-ul asteptat (~4)

    EWMA_WIN = 28
    EWMA_HALF = 8.0
    PAIR_WIN = 20
    PAIR_SEED_DRAWS = 2
    PAIR_DECAY = 0.93
    SEC_WIN = 20
    SEC_SIZE = 10

    SIG1 = 1.8
    SIG2 = 3.5
    I1 = 1.0 / (2.0 * SIG1 * SIG1)
    I2 = 1.0 / (2.0 * SIG2 * SIG2)

    WZ = 2.2    # pondere EWMA z (momentum local)
    WP = 1.5    # pondere ecou pereche
    WS = 0.9    # pondere reversiune sector
    WC = 0.8    # pondere armonici gap
    WI = 1.0    # interactiune ne-liniara tz*tp

    lam = 0.5 ** (1.0 / EWMA_HALF)
    ew_pows = [lam ** a for a in range(EWMA_WIN + 1)]
    pd_pows = [PAIR_DECAY ** a for a in range(PAIR_WIN + 1)]

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, P + 1))
        dq = recent_deque
        L = len(dq)
        if L == 0:
            return sorted(numbers, key=lambda n: (-counts.get(n, 0), n))

        # --- (1) deviatie EWMA a ratei de aparitie, z-score vs base_rate ---
        k1 = min(EWMA_WIN, L)
        den = 0.0
        wsum = 0.0
        wsq = 0.0
        acc = {}
        for a in range(k1):
            s = dq[L - 1 - a]
            wa = ew_pows[a]
            den += wa * len(s)
            wsum += wa
            wsq += wa * wa
            for n in s:
                acc[n] = acc.get(n, 0.0) + wa
        n_eff = (wsum * wsum) / wsq if wsq > 1e-12 else 1.0
        se = math.sqrt(base_rate * (1.0 - base_rate) / n_eff) if n_eff > 1e-9 else 1.0
        if se <= 1e-12:
            se = 1e-12

        # --- (2) ecou de co-aparitie: lift conditionat de seeds (ultimele 2 extrageri) ---
        seeds = []
        seen = set()
        for a in range(min(PAIR_SEED_DRAWS, L)):
            for m in dq[L - 1 - a]:
                if m not in seen:
                    seen.add(m)
                    seeds.append(m)
        nsd = len(seeds)
        pair = dict.fromkeys(numbers, 0.0)
        if nsd > 0:
            k2 = min(PAIR_WIN, L)
            f_seed = dict.fromkeys(seeds, 0.0)
            for a in range(k2):
                wa = pd_pows[a]
                for m in dq[L - 1 - a]:
                    if m in f_seed:
                        f_seed[m] += wa
            # pair_raw[n] = sum_m (1/f(m)) * sum peste extragerile cu m si n, apoi /nsd
            B = {}
            for a in range(k2):
                s = dq[L - 1 - a]
                wa = pd_pows[a]
                A = 0.0
                for m in s:
                    f = f_seed.get(m)
                    if f is not None and f > 0.0:
                        A += 1.0 / f
                if A > 0.0:
                    for n in s:
                        b = B.get(n, 0.0) + wa * A
                        fn = f_seed.get(n)
                        if fn:
                            b -= wa / fn  # elimina auto-coincidenta seed==n
                        B[n] = b
            inv_nsd = 1.0 / nsd
            for n in numbers:
                pair[n] = B.get(n, 0.0) * inv_nsd - base_rate

        # --- (3) reversiune de sector: z hipergeometric per sector pe fereastra scurta ---
        nsec = (P + SEC_SIZE - 1) // SEC_SIZE
        sec_of = [0] * (P + 1)
        sizes = [0] * nsec
        for n in numbers:
            i = (n - 1) // SEC_SIZE
            sec_of[n] = i
            sizes[i] += 1
        k3 = min(SEC_WIN, L)
        obs = [0.0] * nsec
        for a in range(k3):
            for n in dq[L - 1 - a]:
                obs[sec_of[n]] += 1.0
        kd = float(k3) if k3 > 0 else 1.0
        hyper = float(P - W) / float(P - 1) if P > 1 else 1.0
        sec_z = [0.0] * nsec
        for i in range(nsec):
            q = float(sizes[i]) / float(P)
            mu = W * q * kd
            var = W * q * (1.0 - q) * hyper * kd
            sd = math.sqrt(var) if var > 1e-12 else 1e-12
            sec_z[i] = (obs[i] - mu) / sd

        # --- (4) combinare ne-liniara ---
        scores = {}
        for n in numbers:
            rate = (acc.get(n, 0.0) / den) if den > 0 else 0.0
            tz = math.tanh(0.5 * (rate - base_rate) / se)
            tp = math.tanh(4.0 * pair[n])
            ts = math.tanh(-0.5 * sec_z[sec_of[n]])
            ls = last_seen.get(n)
            gap = float(draw_index - ls) if ls is not None else float(draw_index + 1)
            c1 = math.exp(-((gap - gap0) ** 2) * I1)
            c2 = 0.5 * math.exp(-((gap - 2.0 * gap0) ** 2) * I2)
            tc = math.tanh(0.9 * (c1 + c2))
            scores[n] = WZ * tz + WP * tp + WS * ts + WC * tc + WI * tz * tp
        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank