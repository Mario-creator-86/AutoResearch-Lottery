import math
from collections import defaultdict

NAME = "zzgapent"
DESCRIPTION = ("Scor hibrid neliniar: EWMA pe frecvente recente, z-score binomial vs asteptat, "
               "bump pe gap-ratio (fereastra due la 1-2x gap asteptat), momentum de perechi "
               "co-aparute, totul modulat de entropia Shannon a ferestrei recente.")
SOURCE = "model + web research (lottery mathematics, gambler's fallacy) + 2026-09-07"


def create(config):
    P = config.number_range          # ex. 42
    WD = config.numbers_drawn        # ex. 5
    p_mean = WD / P
    exp_gap = P / WD                 # gap asteptat per numar

    PAIR_WINDOW = 60                 # sub-fereastra pentru co-aparire
    EWMA_HALF = 12                   # halflife EWMA in extrageri
    W_MOM = 1.0                      # pondere momentum (z recent)
    W_DUE = 0.9                      # pondere due-window
    W_PAIR = 0.8                     # pondere perechi
    W_GLOB = 0.2                     # pondere deviatie globala (mica)
    DECAY = math.log(2.0) / EWMA_HALF

    def bump(x, lo, hi):
        if x <= lo or x >= hi:
            return 0.0
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        return 1.0 - abs(x - mid) / half

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, P + 1))
        W = len(recent_deque)
        if W == 0:
            return sorted(numbers, key=lambda n: (-counts.get(n, 0), n))

        # --- (1) EWMA pe frecventele recente, ponderi exponentiale dupa vechime ---
        ewma = {n: 0.0 for n in numbers}
        denom = 0.0
        for idx, draw in enumerate(recent_deque):
            w = math.exp(-DECAY * ((W - 1) - idx))
            denom += w
            for x in draw:
                if 1 <= x <= P:
                    ewma[x] += w

        # --- (2) entropie Shannon normalizata a ferestrei ---
        H = 0.0
        for n in numbers:
            pn = ewma[n] / denom
            if pn > 1e-12:
                H -= pn * math.log(pn)
        Hn = H / math.log(P)
        conc = 1.0 - Hn   # 0 = uniform, 1 = concentrat

        # --- (3) momentum de perechi co-aparute, pondere 1/(1+age) ---
        pw = list(recent_deque)[-PAIR_WINDOW:]
        m = len(pw)
        pair_sum = {n: 0.0 for n in numbers}
        if m:
            for idx, draw in enumerate(pw):
                wgt = 1.0 / (1.0 + (m - 1 - idx))
                d = sorted(draw)
                L = len(d)
                for i in range(L):
                    ai = d[i]
                    for j in range(i + 1, L):
                        bj = d[j]
                        pair_sum[ai] += wgt
                        pair_sum[bj] += wgt
        pmax = max(pair_sum.values()) or 1.0

        # --- deviatie globala (counts vs media lor) ---
        vals = [counts.get(n, 0) for n in numbers]
        avg_c = sum(vals) / len(vals)
        var_c = sum((v - avg_c) ** 2 for v in vals) / len(vals)
        sd_c = math.sqrt(var_c) or 1.0

        # --- z-score recent binomial ---
        mu = W * p_mean
        sd = math.sqrt(W * p_mean * (1.0 - p_mean)) or 1.0

        scores = {}
        for n in numbers:
            z = (ewma[n] - mu) / sd
            mom = math.tanh(1.5 * z) * (0.5 + conc)

            if n in last_seen:
                gap = draw_index - last_seen[n]
            else:
                gap = draw_index + 1   # niciodata aparut = gap maxim
            ratio = gap / exp_gap
            due = bump(ratio, 0.5, 2.5) * (1.3 - 0.5 * conc)

            pair_term = (pair_sum[n] / pmax) * (0.6 + 0.8 * conc)

            zg = (counts.get(n, 0) - avg_c) / sd_c
            glob = math.tanh(zg / 2.0)

            scores[n] = W_MOM * mom + W_DUE * due + W_PAIR * pair_term + W_GLOB * glob

        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank