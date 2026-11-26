import math

NAME = "zewma2ha"
DESCRIPTION = ('Scor compozit determinist: doua EWMA pe indicatorii de aparitie (half-life 22 si 110), '
               'deviatie z pe termen lung fata de frecventa asteptata, hazard de intarziere normalizat '
               'la (1-p)/p, afinitate de co-occurienta cu ultima extragere, deficit pe sectoare (zeci) '
               'si interactie ne-liniara frecventa-lunga x gap. Toate componentele clip-uite la ±2.')
SOURCE = 'model + 2026-09-08'


def create(config):
    P = float(config.number_range)
    K = float(config.numbers_drawn)
    p = K / P
    q = 1.0 - p

    H1 = 22.0
    H2 = 110.0
    LMAX = 90
    LPAIR = 60
    WSEC = 30

    W_DEV = 0.30
    W_EW1 = 0.85
    W_EW2 = 0.55
    W_GAP = 0.60
    W_PAIR = 0.30
    W_SEC = 0.25
    W_INT = 0.20

    d1 = 0.5 ** (1.0 / H1)
    d2 = 0.5 ** (1.0 / H2)
    sec_size = (10, 10, 10, 10, 9)

    def _clip(x):
        if x < -2.0:
            return -2.0
        if x > 2.0:
            return 2.0
        return x

    def _z_emp(vals):
        m = sum(vals) / len(vals)
        var = sum((v - m) * (v - m) for v in vals) / len(vals)
        sd = math.sqrt(var) + 1e-9
        return [_clip((v - m) / sd) for v in vals]

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, int(P) + 1))
        NN = len(numbers)
        N = draw_index if draw_index > 0 else 1
        L = len(recent_deque)

        mu = N * p
        sd_lt = math.sqrt(N * p * q) + 1e-9
        z_lt = [_clip((counts.get(n, 0) - mu) / sd_lt) for n in numbers]

        gaps = []
        for n in numbers:
            if n in last_seen:
                gaps.append(float(draw_index - last_seen[n]))
            else:
                gaps.append(float(draw_index + 1))
        zg = _z_emp(gaps)

        z1 = [0.0] * NN
        z2 = [0.0] * NN
        zp = [0.0] * NN
        zsec = [0.0] * NN

        if L > 0:
            step = 1
            if L >= 2:
                sa = max((last_seen.get(n, -1) for n in recent_deque[0]), default=-1)
                sb = max((last_seen.get(n, -1) for n in recent_deque[-1]), default=-1)
                step = 1 if sb >= sa else -1

            eff = min(L, LMAX)
            last_draw = recent_deque[-1] if step == 1 else recent_deque[0]

            ew1 = dict.fromkeys(numbers, 0.0)
            ew2 = dict.fromkeys(numbers, 0.0)
            sec_cnt = [0, 0, 0, 0, 0]
            wsum1 = 0.0
            wsum2 = 0.0
            for age in range(eff):
                idx = (L - 1 - age) if step == 1 else age
                dr = recent_deque[idx]
                w1 = d1 ** age
                w2 = d2 ** age
                wsum1 += w1
                wsum2 += w2
                for n in dr:
                    ew1[n] += w1
                    ew2[n] += w2
                    if age < WSEC:
                        si = (n - 1) // 10
                        if si > 4:
                            si = 4
                        sec_cnt[si] += 1

            e1 = p * wsum1
            e2 = p * wsum2
            v1 = p * q * (1.0 - d1 ** (2 * eff)) / (1.0 - d1 * d1)
            v2 = p * q * (1.0 - d2 ** (2 * eff)) / (1.0 - d2 * d2)
            s1 = math.sqrt(v1) + 1e-9
            s2 = math.sqrt(v2) + 1e-9
            z1 = [_clip((ew1[n] - e1) / s1) for n in numbers]
            z2 = [_clip((ew2[n] - e2) / s2) for n in numbers]

            aff = dict.fromkeys(numbers, 0.0)
            for j in range(1, min(LPAIR, L - 1) + 1):
                idx = (L - 1 - j) if step == 1 else j
                dr = recent_deque[idx]
                inter = dr & last_draw
                k = len(inter)
                if k:
                    for n in dr:
                        if n in inter:
                            aff[n] += k - 1
                        else:
                            aff[n] += k
            zp = _z_emp([aff[n] for n in numbers])

            wsec = min(WSEC, eff)
            if wsec > 0:
                for si in range(5):
                    exp_s = wsec * K * (sec_size[si] / P)
                    act = float(sec_cnt[si])
                    zd = _clip((exp_s - act) / math.sqrt(exp_s + 1.0))
                    base = si * 10
                    for n in range(base + 1, base + sec_size[si] + 1):
                        zsec[n - 1] = zd

        smap = {}
        for i in range(NN):
            s = (W_DEV * z_lt[i] + W_EW1 * z1[i] + W_EW2 * z2[i]
                 + W_GAP * zg[i] + W_PAIR * zp[i] + W_SEC * zsec[i]
                 + W_INT * (z_lt[i] * zg[i]))
            smap[numbers[i]] = s

        return sorted(numbers, key=lambda n: (-smap[n], n))

    return rank