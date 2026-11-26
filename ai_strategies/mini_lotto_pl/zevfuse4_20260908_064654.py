import math

NAME = "zevfuse4"
DESCRIPTION = ('Log-probability evidence fusion for 5/42: all-time frequency deviation, EWMA frequency '
               '(half-life 12), pair echo from the 3 most recent draws, geometric gap hazard per number '
               'and per sector (6x7); momentum/echo vs due weights adaptively modulated by window '
               'over-dispersion and entropy.')
SOURCE = 'glm-5.3-flash + 2026-09-08'


def create(config):
    R = int(config.number_range)
    K = int(config.numbers_drawn)
    p = K / R
    N_SEC = 6
    sz = (R + N_SEC - 1) // N_SEC
    sector_of = [0] * (R + 1)
    sec_size = [0] * N_SEC
    for n in range(1, R + 1):
        si = min((n - 1) // sz, N_SEC - 1)
        sector_of[n] = si
        sec_size[si] += 1
    q_sec = []
    for si in range(N_SEC):
        rest = R - sec_size[si]
        q_sec.append(1.0 if rest < K else 1.0 - math.comb(rest, K) / math.comb(R, K))
    mu_sec = [(1.0 - q) / q for q in q_sec]
    sd_sec = [math.sqrt(max(1.0 - q, 1e-12)) / q for q in q_sec]

    HALF_LIFE = 12.0
    ECHO_LAGS = 3
    ECHO_DECAY = 0.5
    SEC_LOOKBACK = 12
    W_ALL, W_EW, W_ECHO, W_GAP, W_SEC = 1.0, 1.2, 0.9, 0.8, 0.55
    ENT_BETA = 6.0
    GAP_CAP = 80
    T_CAP = 6.0

    decay = 0.5 ** (1.0 / HALF_LIFE)
    ew_mean = p / (1.0 - decay)
    ew_sd = math.sqrt(p * (1.0 - p) / (1.0 - decay * decay))
    gap_fac = math.log10(1.0 / (1.0 - p))
    SQ2 = math.sqrt(2.0)
    lnR = math.log(R)

    def tail(z):
        t = 0.5 * math.erfc(z / SQ2)
        if t < 1e-300:
            t = 1e-300
        v = -math.log10(t)
        return v if v < T_CAP else T_CAP

    _ck = [None]
    _cr = [None]

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        key = (draw_index, recent_window)
        if _ck[0] == key and _cr[0] is not None:
            return list(_cr[0])

        W = len(recent_deque)
        N = draw_index if draw_index > 0 else 1

        crec = [0] * (R + 1)
        for s in recent_deque:
            for n in s:
                if 1 <= n <= R:
                    crec[n] += 1
        tot = sum(crec[1:])

        m = tot / R
        var = sum((x - m) ** 2 for x in crec[1:]) / R
        sd_w = math.sqrt(W * p * (1.0 - p))
        if W > 0 and sd_w > 0:
            disp = math.sqrt(var) / sd_w
        else:
            disp = 1.0
        Hn = 0.0
        if tot > 0:
            h = 0.0
            for x in crec[1:]:
                if x > 0:
                    pn = x / tot
                    h -= pn * math.log(pn)
            Hn = h / lnR
        if W == 0 or tot == 0:
            gate_mom = 1.0
            gate_due = 1.0
        else:
            gate_mom = max(0.4, min(2.5, disp)) * (1.0 + ENT_BETA * (1.0 - Hn))
            gate_due = max(0.4, min(2.5, 2.0 - disp))

        ew = [0.0] * (R + 1)
        if W:
            pw = [1.0] * W
            for k in range(1, W):
                pw[k] = pw[k - 1] * decay
            for i, s in enumerate(recent_deque):
                wgt = pw[W - 1 - i]
                for n in s:
                    if 1 <= n <= R:
                        ew[n] += wgt

        P = [None] * (R + 1)
        for n in range(1, R + 1):
            P[n] = [0] * (R + 1)
        for s in recent_deque:
            ns = [n for n in s if 1 <= n <= R]
            L = len(ns)
            for i in range(L - 1):
                a = ns[i]
                Pa = P[a]
                for j in range(i + 1, L):
                    b = ns[j]
                    Pa[b] += 1
                    P[b][a] += 1

        echo = [0.0] * (R + 1)
        for lag in range(1, ECHO_LAGS + 1):
            if lag > W:
                break
            s = recent_deque[-lag]
            wl = ECHO_DECAY ** (lag - 1)
            for n in range(1, R + 1):
                row = P[n]
                acc = 0
                for mb in s:
                    if 1 <= mb <= R and mb != n:
                        acc += row[mb]
                echo[n] += wl * acc
        me = sum(echo[1:]) / R
        ve = sum((x - me) ** 2 for x in echo[1:]) / R
        se = math.sqrt(ve)

        gsec = [SEC_LOOKBACK] * N_SEC
        found = [False] * N_SEC
        for lag in range(1, min(SEC_LOOKBACK, W) + 1):
            for n in recent_deque[-lag]:
                if 1 <= n <= R:
                    si = sector_of[n]
                    if not found[si]:
                        gsec[si] = lag - 1
                        found[si] = True

        mu_all = N * p
        sd_all = math.sqrt(N * p * (1.0 - p))
        sc = [0.0] * (R + 1)
        for n in range(1, R + 1):
            v = W_ALL * tail((counts.get(n, 0) - mu_all) / sd_all)
            v += gate_mom * W_EW * tail((ew[n] - ew_mean) / ew_sd)
            if se > 0:
                v += gate_mom * W_ECHO * tail((echo[n] - me) / se)
            g = draw_index - last_seen.get(n, -1)
            if g < 1:
                g = 1
            elif g > GAP_CAP:
                g = GAP_CAP
            v += gate_due * W_GAP * (g - 1) * gap_fac
            si = sector_of[n]
            v += gate_due * W_SEC * tail((gsec[si] - mu_sec[si]) / sd_sec[si])
            sc[n] = v

        order = sorted(range(1, R + 1), key=lambda n: (-sc[n], n))
        _ck[0] = key
        _cr[0] = order
        return list(order)

    return rank