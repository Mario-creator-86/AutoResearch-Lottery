import math

NAME = "zregimsi"
DESCRIPTION = (
    'Sinteza multi-componenta: tilt global (z-score frecvente istorice), momentum pe fereastra '
    'recenta modulat de gate de dispersie (chi2/NR), hazard de intarziere tanh(gap vs NR/ND), '
    'sinergie de perechi co-aparute in fereastra si deficit pe 7 sectoare x 7 numere. '
    'Fara stare, doar apartenenta + last_seen => invarianta la orientarea deque-ului si determinista.'
)
SOURCE = 'model + 2026-11-12'

W_GLOB = 0.30
W_WIN = 0.80
W_HAZ = 0.30
W_PAIR = 0.55
W_SECT = 0.18
GATE_GAIN = 1.5
NSECT = 7


def create(config):
    NR = int(config.number_range)
    ND = int(config.numbers_drawn)
    K = max(1, NR // NSECT)
    p_num = ND / float(NR)
    EG = NR / float(ND)
    E_partners_draw = ND * (ND - 1) / float(NR)
    mean_sec_draw = ND * K / float(NR)
    var_sec_draw = ND * (K / float(NR)) * (1.0 - K / float(NR)) * (NR - ND) / float(max(1, NR - 1))

    sector_of = {n: (n - 1) // K for n in range(1, NR + 1)}
    numbers = list(range(1, NR + 1))

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # 1) tilt global
        mu = 0.0
        for n in numbers:
            mu += counts.get(n, 0)
        mu /= NR
        var = 0.0
        for n in numbers:
            d = counts.get(n, 0) - mu
            var += d * d
        var /= NR
        sd = math.sqrt(var)
        zg = {}
        if sd > 1e-12:
            inv = 1.0 / sd
            for n in numbers:
                zg[n] = (counts.get(n, 0) - mu) * inv
        else:
            for n in numbers:
                zg[n] = 0.0

        W = len(recent_deque)
        zw = dict.fromkeys(numbers, 0.0)
        zp = dict.fromkeys(numbers, 0.0)
        zs = dict.fromkeys(numbers, 0.0)
        mod = 1.0

        if W > 0:
            wcnt = dict.fromkeys(numbers, 0)
            sect = [0] * NSECT
            pairs = {}
            for s in recent_deque:
                members = [x for x in s if 1 <= x <= NR]
                for x in members:
                    wcnt[x] += 1
                    sect[sector_of[x]] += 1
                ml = len(members)
                for i in range(ml):
                    a = members[i]
                    for j in range(i + 1, ml):
                        b = members[j]
                        key = (a, b) if a < b else (b, a)
                        pairs[key] = pairs.get(key, 0) + 1

            mu_w = W * p_num
            sd_w = math.sqrt(W * p_num * (1.0 - p_num))
            if sd_w > 1e-12:
                denom = W * p_num * (1.0 - p_num)
                x2 = 0.0
                invw = 1.0 / sd_w
                for n in numbers:
                    d = wcnt[n] - mu_w
                    x2 += d * d
                    zw[n] = d * invw
                d_ratio = (x2 / denom) / float(NR)
                mod = 1.0 + math.tanh(GATE_GAIN * (1.0 - d_ratio))

            if pairs:
                ptot = dict.fromkeys(numbers, 0)
                for (a, b), c in pairs.items():
                    ptot[a] += c
                    ptot[b] += c
                E_P = W * E_partners_draw
                invp = 1.0 / math.sqrt(max(E_P, 1.0))
                for n in numbers:
                    zp[n] = (ptot[n] - E_P) * invp

            sd_s = math.sqrt(max(W * var_sec_draw, 1e-12))
            mean_s = W * mean_sec_draw
            if sd_s > 1e-12:
                invs = 1.0 / sd_s
                for n in numbers:
                    zs[n] = (sect[sector_of[n]] - mean_s) * invs

        scores = {}
        for n in numbers:
            ls = last_seen.get(n)
            gap = draw_index if ls is None else draw_index - ls
            if gap < 0:
                gap = 0
            hz = math.tanh((gap - EG) / (2.0 * EG))
            scores[n] = (W_GLOB * zg[n]
                         + W_WIN * mod * zw[n]
                         + W_HAZ * hz
                         + W_PAIR * zp[n]
                         + W_SECT * zs[n])

        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank