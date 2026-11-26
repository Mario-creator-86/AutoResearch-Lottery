import math

NAME = "zzsector"
DESCRIPTION = ('Ciclu de intarziere pe sectoare (deficit normalizat pe blocuri de 8 in fereastra recenta) '
               '+ deviatie macro fata de frecventa asteptata (z-score pe tot istoricul, semn contrarian) '
               '+ rezonanta armonica a intervalului propriu (gap curent vs 1x/2x/3x din gap-ul istoric mediu). '
               'Combinare liniara, apoi sortare descrescatoare cu tie-break pe valoarea numerica.')
SOURCE = 'model + 2026-09-16'

# ---- constante de ablatie (editeaza aici, nu in cod) ----
W_SEC = 0.7            # grid {0.35, 0.7, 1.05}
SECTOR_SIZE = 8        # grid {5, 7, 9}
W_DEV = 0.4            # grid {0, 0.4, 0.8}
DEV_CONTRARIAN = True  # True = boost numere reci (reversiune la medie); False = momentum
W_GAP = 0.3            # grid {0.3, 0.5, 0.8}
GAP_SIGMA = 0.6        # latimea nucleului gaussian al rezonantei
GAP_HARMONICS = (1.0, 2.0, 3.0)


def create(config):
    NR = config.number_range
    ND = config.numbers_drawn

    n_sec = -(-NR // SECTOR_SIZE)
    sec_of = [(n - 1) // SECTOR_SIZE for n in range(NR + 1)]
    sec_size = [min(SECTOR_SIZE, NR - j * SECTOR_SIZE) for j in range(n_sec)]
    numbers = list(range(1, NR + 1))

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        D = draw_index + 1
        p = ND / NR
        exp_cnt = D * p
        var = D * p * (1.0 - p)
        std = var ** 0.5 if var > 1e-12 else 1e-9

        # (1) sector-cycle: deficit pe sector in fereastra recenta
        wlen = len(recent_deque)
        obs = [0] * n_sec
        if wlen:
            for draw in recent_deque:
                for n in draw:
                    j = sec_of[n]
                    if 0 <= j < n_sec:
                        obs[j] += 1
        pressure = [0.0] * n_sec
        for j in range(n_sec):
            exp_j = wlen * ND * sec_size[j] / NR
            if exp_j > 0:
                pressure[j] = (exp_j - obs[j]) / (exp_j ** 0.5 + 1e-9)

        base_gap = NR / ND
        scores = {}
        for n in numbers:
            c = counts.get(n, 0)

            # (2) deviatie macro (z-score), semn contrarian pentru reversiune la medie
            z = (c - exp_cnt) / std
            s = W_SEC * pressure[sec_of[n]] + W_DEV * (-z if DEV_CONTRARIAN else z)

            # (3) rezonanta armonica a intervalului propriu
            ls = last_seen.get(n)
            cur_gap = (draw_index - ls) if ls is not None else (draw_index + 1)
            mean_gap = base_gap if c <= 0 else (D + 1.0) / (c + 1.0)
            if mean_gap <= 0:
                mean_gap = base_gap
            ratio = cur_gap / mean_gap
            if ratio > 60.0:
                ratio = 60.0
            g = 0.0
            for k in GAP_HARMONICS:
                d = (ratio - k) / GAP_SIGMA
                v = math.exp(-0.5 * d * d)
                if v > g:
                    g = v
            scores[n] = s + W_GAP * g

        return sorted(numbers, key=lambda nn: (-scores[nn], nn))

    return rank