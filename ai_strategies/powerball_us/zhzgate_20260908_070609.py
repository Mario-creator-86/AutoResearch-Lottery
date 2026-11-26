import math

NAME = "zhzgate"
DESCRIPTION = 'Hazard-gated multi-signal: deviatie bayesiana z-tanh pe frecventa totala, EWMA (half-life 30) pe fereastra recenta, risc de intirziere log1p(gap/ciclu), cicluri pe sectoare si afinitate cu nucleul fierbinte recent, combinate printr-o poarta adaptiva de dispersie chi2 (momentum vs. reversie) plus interactiune non-liniara z_long x z_recent.'
SOURCE = 'glm research 2026-09-08 hzgate v1'

def create(config):
    K = int(config.number_range)
    D = int(config.numbers_drawn) or 5
    p = D / float(K) if K > 0 else 0.0725
    cycle = (K / float(D)) if D > 0 else 13.8

    A_LONG = 1.10
    B_REC = 1.50
    C_GAP = 0.85
    D_PAIR = 0.70
    E_SEC = 0.35
    I_INT = 0.25
    HALF_LIFE = 30.0
    PAIR_HL = 50.0
    MAX_DEPTH = 150
    ENT_WIN = 30
    PAIR_WIN = 100
    HOT_K = 5
    GAP_CAP = 8.0 * cycle
    ZCLIP = 3.0

    sec_of = [0] * (K + 1)
    sec_size = [0] * 5
    for n in range(1, K + 1):
        j = (n - 1) * 5 // K
        if j > 4:
            j = 4
        sec_of[n] = j
        sec_size[j] += 1

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, K + 1))

        total_appear = 0
        for n in numbers:
            total_appear += counts.get(n, 0)
        Nf = total_appear / float(D) if D > 0 else 0.0
        mu = Nf * p
        sigma = math.sqrt(Nf * p * (1.0 - p)) if Nf > 1.0 else 1.0

        draws = list(recent_deque)
        L = len(draws)
        depth = L if L < MAX_DEPTH else MAX_DEPTH
        recent = draws[-depth:] if depth > 0 else []

        ew = dict.fromkeys(numbers, 0.0)
        wsum = 0.0
        m = len(recent)
        for i in range(m):
            w = 0.5 ** ((m - 1 - i) / HALF_LIFE)
            wsum += w
            for x in recent[i]:
                if 1 <= x <= K:
                    ew[x] += w
        inv = (1.0 / (wsum * p)) if wsum > 1e-9 else 0.0

        ewin = draws[-ENT_WIN:] if L > 0 else []
        ec = {}
        for s in ewin:
            for x in s:
                if 1 <= x <= K:
                    ec[x] = ec.get(x, 0) + 1
        tot_e = len(ewin) * D
        t_gate = 0.0
        if tot_e > 0:
            e_exp = tot_e / float(K)
            chi2 = 0.0
            for n in numbers:
                o = ec.get(n, 0)
                d = o - e_exp
                chi2 += d * d / e_exp
            t_gate = (chi2 / (K - 1.0) - 1.0) * 2.0
            if t_gate < 0.0:
                t_gate = 0.0
            elif t_gate > 1.0:
                t_gate = 1.0
        mom_gate = 1.0 + 0.8 * t_gate
        rev_gate = 1.0 + 0.8 * (1.0 - t_gate)

        obs_sec = [0, 0, 0, 0, 0]
        for s in ewin:
            for x in s:
                if 1 <= x <= K:
                    obs_sec[sec_of[x]] += 1
        sec_z = [0.0] * 5
        for j in range(5):
            ee = tot_e * sec_size[j] / float(K)
            if ee > 0:
                z = (ee - obs_sec[j]) / math.sqrt(ee + 1.0)
                if z > 1.5:
                    z = 1.5
                elif z < -1.5:
                    z = -1.5
                sec_z[j] = z

        hotset = []
        if wsum > 1e-9:
            order = sorted(numbers, key=lambda n: (-ew[n], n))
            hotset = order[:HOT_K]

        pair = dict.fromkeys(numbers, 0.0)
        if hotset and L > 0:
            pwin = draws[-PAIR_WIN:]
            pw = len(pwin)
            for i in range(pw):
                w = 0.5 ** ((pw - 1 - i) / PAIR_HL)
                s = pwin[i]
                ov = 0
                for h in hotset:
                    if h in s:
                        ov += 1
                if ov:
                    for x in s:
                        if 1 <= x <= K:
                            pair[x] += ov * w
        pmax = 0.0
        for n in numbers:
            if pair[n] > pmax:
                pmax = pair[n]
        pinv = (1.0 / pmax) if pmax > 1e-9 else 0.0

        scores = {}
        for n in numbers:
            c = counts.get(n, 0)
            zl = (c - mu) / sigma
            if zl > ZCLIP:
                zl = ZCLIP
            elif zl < -ZCLIP:
                zl = -ZCLIP
            long_t = math.tanh(zl / 2.5)

            if wsum > 1e-9:
                zr = (ew[n] * inv - 1.0) / 0.42
            else:
                zr = 0.0
            if zr > 3.0:
                zr = 3.0
            elif zr < -2.0:
                zr = -2.0

            ls = last_seen.get(n)
            if ls is None:
                g = Nf + 1.0
            else:
                g = draw_index - ls
            if g < 0.0:
                g = 0.0
            elif g > GAP_CAP:
                g = GAP_CAP
            gap_t = math.log1p(g / cycle)

            sz = sec_z[sec_of[n]]
            pt = pair[n] * pinv

            scores[n] = (A_LONG * long_t
                         + B_REC * zr * mom_gate
                         + C_GAP * gap_t * rev_gate
                         + E_SEC * sz
                         + D_PAIR * pt
                         + I_INT * long_t * zr)

        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank