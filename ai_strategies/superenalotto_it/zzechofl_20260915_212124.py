NAME = "zzechofl"
DESCRIPTION = 'Difuzie de afinitate pe graf de co-aparitie: activitatea medie a vecinilor de co-aparitie din fereastra recenta (ponderata exponential pe varsta draw-ului, normalizata pe grad), combinata cu divergenta dual-orizont EWMA (HL 8/32) si cu o inclinare spectrala pe axa numerelor data de abaterea z a sumelor recente fata de media combinatorica K(N+1)/2. Componente z-standardizate, combinare liniara ponderata.'
SOURCE = 'model cantitativ + 2026-09-15'

import math

DIAG = {'orient': None, 'orient_flips': 0}

# hiperparametri expusi pentru grid-search / ablatie
W_BASE = 0.30   # rata de baza pe tot istoricul
W_ECHO = 1.00   # difuzie de co-aparitie (camp de contagion)
W_X = 0.55      # divergenta dual-orizont fast-slow
W_SUM = 0.18    # inclinare spectrala pe suma draw-ului

GAMMA_A = 0.93  # decay afinitate (HL ~9.5 extrageri)
HL_FAST = 8.0
HL_SLOW = 32.0
SUM_CAP = 2.0


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    GF = 0.5 ** (1.0 / HL_FAST)
    GS = 0.5 ** (1.0 / HL_SLOW)
    half_span = 0.5 * (N + 1)

    def zscore_map(d, numbers):
        vals = [d[n] for n in numbers]
        m = len(vals)
        mu = sum(vals) / float(m)
        var = sum((v - mu) * (v - mu) for v in vals) / float(m)
        sd = math.sqrt(var)
        if sd < 1e-12:
            return dict.fromkeys(numbers, 0.0)
        return {n: (d[n] - mu) / sd for n in numbers}

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        L = len(recent_deque)

        # fallback determinist: fereastra prea subtire -> doar rata de baza
        if L < 2 or draw_index <= 0:
            base = {n: counts.get(n, 0) / float(max(draw_index, 1)) for n in numbers}
            zb = zscore_map(base, numbers)
            total = {n: W_BASE * zb[n] for n in numbers}
            return sorted(numbers, key=lambda n: (-total[n], n))

        # auto-detectie orientare deque: setul cel mai nou are recenta medie minima
        def mean_recency(st):
            if not st:
                return 1e18
            tot = 0.0
            for n in st:
                ls = last_seen.get(n)
                tot += (draw_index - ls) if ls is not None else float(draw_index)
            return tot / float(len(st))

        newest_last = mean_recency(recent_deque[-1]) <= mean_recency(recent_deque[0])
        orient = 'newest_last' if newest_last else 'newest_first'
        if DIAG['orient'] is not None and DIAG['orient'] != orient:
            DIAG['orient_flips'] += 1
        DIAG['orient'] = orient

        seq = list(recent_deque)
        if newest_last:
            seq.reverse()  # seq[0] = cea mai noua extragere (age 0)

        # camp de activitate dual-orizont pe fereastra
        mass_f = 0.0
        mass_s = 0.0
        for a in range(L):
            mass_f += GF ** a
            mass_s += GS ** a

        fast = dict.fromkeys(numbers, 0.0)
        slow = dict.fromkeys(numbers, 0.0)
        sum_tot = 0
        for a in range(L):
            wf = GF ** a
            ws = GS ** a
            st = seq[a]
            s = 0
            for n in st:
                if 1 <= n <= N:
                    fast[n] += wf
                    slow[n] += ws
                    s += n
            sum_tot += s

        fast = {n: fast[n] / mass_f for n in numbers}
        slow = {n: slow[n] / mass_s for n in numbers}

        # graf de co-aparitie ponderat exponential pe varsta (dict-of-dicts)
        nbr = {}
        for a in range(L):
            w = GAMMA_A ** a
            st = sorted(x for x in seq[a] if 1 <= x <= N)
            mx = len(st)
            for i in range(mx):
                x = st[i]
                dx = nbr.get(x)
                if dx is None:
                    dx = {}
                    nbr[x] = dx
                for j in range(i + 1, mx):
                    y = st[j]
                    dx[y] = dx.get(y, 0.0) + w
                    dy = nbr.get(y)
                    if dy is None:
                        dy = {}
                        nbr[y] = dy
                    dy[x] = dy.get(x, 0.0) + w

        # difuzie un pas: activitatea medie a vecinilor, normalizata pe grad
        echo = {}
        for n in numbers:
            d = nbr.get(n)
            if not d:
                echo[n] = 0.0
                continue
            num = 0.0
            den = 0.0
            for m2, w in d.items():
                num += w * fast[m2]
                den += w
            echo[n] = num / den if den > 1e-9 else 0.0

        # inclinare spectrala: sumele recente peste medie -> favorizeaza numere mici
        mean_S = K * (N + 1) / 2.0
        sd_S = math.sqrt(K * (N + 1) * (N - K) / 12.0)
        se = sd_S / math.sqrt(L) + 1e-9
        delta = (sum_tot / float(L) - mean_S) / se
        if delta > SUM_CAP:
            delta = SUM_CAP
        elif delta < -SUM_CAP:
            delta = -SUM_CAP
        tilt = {n: -delta * (n - half_span) / half_span for n in numbers}

        base = {n: counts.get(n, 0) / float(max(draw_index, 1)) for n in numbers}
        zb = zscore_map(base, numbers)
        ze = zscore_map(echo, numbers)
        zx = zscore_map({n: fast[n] - slow[n] for n in numbers}, numbers)
        zt = zscore_map(tilt, numbers)

        total = {n: W_BASE * zb[n] + W_ECHO * ze[n] + W_X * zx[n] + W_SUM * zt[n]
                 for n in numbers}
        return sorted(numbers, key=lambda n: (-total[n], n))

    return rank