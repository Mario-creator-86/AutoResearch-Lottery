import math

NAME = "zzborda"
DESCRIPTION = 'Fuziune Borda a 4 semnale: frecventa in fereastra recenta, re-entrantare la gap-ul modal (ciclu geometric p=numbers_drawn/number_range), centralitate masa-de-parteneri-calzi din co-aparitiile ferestrei, si z-scor al frecventei totale. Ordine finala = suma ponderata a punctajelor Borda; determinist, order-free fata de deque.'
SOURCE = 'glm + 2026-09-23'

WEIGHTS = {'f': 1.0, 'gap': 0.9, 'cent': 0.8, 'z': 0.5}


def create(config):
    n_range = config.number_range
    nd = config.numbers_drawn
    p = nd / float(n_range) if n_range else 0.25
    mean_gap = (1.0 - p) / p if p > 0 else 3.0
    gap_mode = mean_gap + 0.5
    gap_sigma = max(2.0, 0.75 * mean_gap)
    numbers = tuple(range(1, n_range + 1))
    N = n_range

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # S1: frecventa in fereastra (order-free, independenta de orientarea deque-ului)
        f = dict.fromkeys(numbers, 0)
        for d in recent_deque:
            for n in d:
                f[n] += 1

        # S3: centralitate = masa medie de hotness a co-aparitilor, per aparitie
        cent = dict.fromkeys(numbers, 0.0)
        for d in recent_deque:
            s = 0
            for n in d:
                s += f[n]
            for n in d:
                cent[n] += (s - f[n])
        for n in numbers:
            fn = f[n]
            cent[n] = (cent[n] / fn) if fn else 0.0

        # S2: fit pe ciclu de gap (re-entrantare in jurul mediei inter-aparitiei)
        gap_sig = {}
        for n in numbers:
            ls = last_seen.get(n)
            g = (draw_index - ls) if ls is not None else 10 ** 6
            gap_sig[n] = math.exp(-((g - gap_mode) ** 2) / (2.0 * gap_sigma * gap_sigma))

        # S4: bias pe termen lung (z pe totaluri)
        vals = [counts.get(n, 0) for n in numbers]
        mu = sum(vals) / float(N) if N else 0.0
        var = sum((v - mu) ** 2 for v in vals) / N if N else 0.0
        sd = math.sqrt(var) or 1.0
        z_sig = {}
        for n in numbers:
            z_sig[n] = (counts.get(n, 0) - mu) / sd

        def borda(sig):
            order = sorted(numbers, key=lambda x: (-sig[x], x))
            return {n: N - i for i, n in enumerate(order)}

        b_f = borda(f)
        b_gap = borda(gap_sig)
        b_cent = borda(cent)
        b_z = borda(z_sig)
        w = WEIGHTS
        total = {
            n: w['f'] * b_f[n] + w['gap'] * b_gap[n]
                 + w['cent'] * b_cent[n] + w['z'] * b_z[n]
            for n in numbers
        }
        return sorted(numbers, key=lambda x: (-total[x], x))

    return rank