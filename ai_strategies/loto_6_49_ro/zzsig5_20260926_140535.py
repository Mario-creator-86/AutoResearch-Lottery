import math

NAME = "zzsig5"
DESCRIPTION = 'Blend de 6 semnale standardizate: reversie pe z-score frecventa totala, momentum pe fereastra recenta, kernel exponential de recenta din last_seen (independent de ordinea deque-ului), presiune de intarziere geometrica, deviatie pe 5 sectoare si interactie recent-hot x overdue.'
SOURCE = 'qstrat+zsig5 + 2026-09-26'

W_LONG = -0.7      # reversie pe frecventa totala (z-score)
W_RECENT = 2.2     # momentum in fereastra recenta
W_RECENCY = 1.3    # kernel exponential de recenta (din last_seen)
W_GAP = 0.8        # presiune overdue (gap standardizat geometric)
W_SECTOR = 0.7     # momentum la nivel de sector
W_INTERACT = 0.6   # interactie z_recent * z_gap
TAU = 12.0         # constanta de decay a kernelului de recenta
CLIP = 2.5         # clipping pentru z-gap
N_SECT = 5         # numar de sectoare


def create(config):
    N = config.number_range
    K = config.numbers_drawn

    # index de sector precalculat (dimensiuni aproximativ egale)
    sec_idx = [min((n - 1) * N_SECT // N, N_SECT - 1) for n in range(1, N + 1)]
    sec_sizes = [0] * N_SECT
    for si in sec_idx:
        sec_sizes[si] += 1

    p = K / float(N)
    mu_g = (1.0 - p) / p            # E[gap] sub geometric
    sd_g = math.sqrt(1.0 - p) / p   # SD[gap] sub geometric

    def zstd(vals):
        m = sum(vals) / float(len(vals))
        var = sum((v - m) ** 2 for v in vals) / float(len(vals))
        s = math.sqrt(var)
        if s <= 1e-12:
            s = 1.0
        return [(v - m) / s for v in vals]

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))

        # counts in fereastra recenta + counts pe sectoare (independent de ordine)
        r = dict.fromkeys(numbers, 0)
        obs = [0] * N_SECT
        wd = 0
        for draw in recent_deque:
            wd += 1
            for n in draw:
                if 1 <= n <= N:
                    si = sec_idx[n - 1]
                    r[n] += 1
                    obs[si] += 1

        # deviatie standardizata pe sector (Poisson-style)
        dev = [0.0] * N_SECT
        for si in range(N_SECT):
            exp_si = wd * K * (sec_sizes[si] / float(N))
            if exp_si > 0:
                dev[si] = (obs[si] - exp_si) / math.sqrt(exp_si)

        z_long = zstd([counts.get(n, 0) for n in numbers])
        z_rec = zstd([r[n] for n in numbers])

        rho = []
        gz = []
        for n in numbers:
            ls = last_seen.get(n)
            g = (draw_index - ls) if ls is not None else (draw_index + 1)
            if g < 0:
                g = 0
            rho.append(math.exp(-g / TAU))
            z = (g - mu_g) / sd_g
            if z > CLIP:
                z = CLIP
            elif z < -CLIP:
                z = -CLIP
            gz.append(z)
        z_rho = zstd(rho)

        scored = []
        for i, n in enumerate(numbers):
            sc = (W_LONG * z_long[i]
                  + W_RECENT * z_rec[i]
                  + W_RECENCY * z_rho[i]
                  + W_GAP * gz[i]
                  + W_SECTOR * dev[sec_idx[n - 1]]
                  + W_INTERACT * z_rec[i] * gz[i])
            scored.append((-sc, n))
        scored.sort()
        return [n for _, n in scored]

    return rank