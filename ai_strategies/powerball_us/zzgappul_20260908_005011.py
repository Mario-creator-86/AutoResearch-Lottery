import math
from itertools import combinations

NAME = "zzgappul"
DESCRIPTION = 'Scor hibrid determinist: z-score frecventa lunga + kernel gaussian pe gap-ratio (hazard-renewal, centru 0.35, latime 0.742, cap 2.5*E[gap]) + momentum EWMA fast-slow + exces z pe perechi co-aparute + due saturant (cap 3*E[gap]) + deviatie pe fereastra recenta, toate modulate de un gate adaptiv pe raportul de suprdispersie chi2/df clip-uit in [0.65, 1.35].'
SOURCE = 'quant kernel-momentum-gate model + 2026-09-07'


def create(config):
    P = float(config.number_range)
    Wn = float(config.numbers_drawn)
    p0 = Wn / P
    E_GAP = 1.0 / p0
    P_PAIR = (Wn * (Wn - 1.0)) / (P * (P - 1.0))

    K_CENTER = 0.35
    K_WIDTH = 0.742
    R_CAP = 2.5
    DUE_CAP = 3.0 * E_GAP

    AF = 2.0 / (10.0 + 1.0)
    AS = 2.0 / (80.0 + 1.0)
    PAIR_LAG = 12

    W_Z, W_HAZ, W_MOM, W_PAIR, W_DUE, W_DEV = 1.0, 0.55, 0.45, 0.55, 0.25, 0.30
    G_MOM, G_REV = 0.5, 0.4
    BAND_LO, BAND_HI = 0.65, 1.35

    def clip(v, lo, hi):
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, int(P) + 1))
        sum_counts = 0.0
        for n in numbers:
            sum_counts += counts.get(n, 0)
        D = sum_counts / Wn if Wn > 0 else 0.0
        mean_c = D * p0
        std_c = math.sqrt(max(D * p0 * (1.0 - p0), 1e-9))

        draws = list(recent_deque)
        L = len(draws)
        pos = dict((n, []) for n in numbers)
        for i in range(L):
            for n in draws[i]:
                pos[n].append(i)

        exp_c = L * p0
        dev_std = math.sqrt(max(exp_c * (1.0 - p0), 1e-9))
        if L > 0 and exp_c > 1e-12:
            chi = 0.0
            for n in numbers:
                d = len(pos[n]) - exp_c
                chi += d * d / exp_c
            R = chi / max(int(P) - 1, 1)
        else:
            R = 1.0
        Rc = clip(R, BAND_LO, BAND_HI)
        gate_mom = 1.0 + G_MOM * (Rc - 1.0)
        gate_z = 1.0 - G_REV * (Rc - 1.0)

        pf = (1.0 - AF) ** L
        ps = (1.0 - AS) ** L
        powf = [1.0] * (L + 1)
        pows = [1.0] * (L + 1)
        for k in range(1, L + 1):
            powf[k] = powf[k - 1] * (1.0 - AF)
            pows[k] = pows[k - 1] * (1.0 - AS)

        pl = PAIR_LAG
        if recent_window > 0 and recent_window < pl:
            pl = recent_window
        if L < pl:
            pl = L
        tail = draws[L - pl:] if pl > 0 else []
        pc = {}
        for s in tail:
            ss = sorted(s)
            for a, b in combinations(ss, 2):
                key = (a, b)
                if key in pc:
                    pc[key] += 1
                else:
                    pc[key] = 1
        n_tail = float(len(tail))
        exp_pair = n_tail * P_PAIR
        pair_sd = math.sqrt(max(exp_pair * (1.0 - P_PAIR), 1e-9))
        pairz = dict((n, 0.0) for n in numbers)
        for key, c in pc.items():
            if c >= 2:
                z = clip((c - exp_pair) / pair_sd, 0.0, 3.0)
                pairz[key[0]] += z
                pairz[key[1]] += z
        for n in numbers:
            pairz[n] = clip(pairz[n], -3.0, 3.0)

        scores = {}
        for n in numbers:
            z = (counts.get(n, 0) - mean_c) / std_c
            z = clip(z, -3.0, 3.0)

            g = float(draw_index - last_seen.get(n, -1))
            if g < 0.0:
                g = 0.0
            rc = g / E_GAP
            if rc > R_CAP:
                rc = R_CAP
            kern = math.exp(-0.5 * ((rc - K_CENTER) / K_WIDTH) ** 2)
            due = g / DUE_CAP
            if due > 1.0:
                due = 1.0

            ef = pf * p0
            es = ps * p0
            pn = pos[n]
            for i in pn:
                ef += AF * powf[L - 1 - i]
                es += AS * pows[L - 1 - i]
            momz = clip((ef - es) / p0, -2.0, 2.0)

            if L > 0:
                devz = clip((len(pn) - exp_c) / dev_std, -2.0, 2.0)
            else:
                devz = 0.0

            scores[n] = (W_Z * gate_z * z
                         + W_HAZ * kern
                         + W_MOM * gate_mom * momz
                         + W_PAIR * gate_mom * pairz[n]
                         + W_DUE * due
                         + W_DEV * gate_mom * devz)

        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank