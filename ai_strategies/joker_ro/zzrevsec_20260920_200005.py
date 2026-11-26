NAME = "zzrevsec"
DESCRIPTION = 'Reversie la medie pe z-score-ul aparitiilor din fereastra recenta, modulata de un gate de dispersie (chi2/df al distributiei ferestrei) care amplifica fade-ul extremelor; presiune de cicluri de intarziere pe sectoare de 5 numere (gap standardizat sub geometrica cu p-hit ipergeometric), alocata membrilor proportional cu deficitul recent; interactiune neliniara relu(tanh(z-lifetime)) x (-z-recent) care directioneaza reversia spre numerele calde istoric si face fade momentum-ului structurally-hot. Diag per componenta in rank._diag.'
SOURCE = 'model + 2026-09-20'

import math

# ---- constante tunabile (grila de ablatie) ----
W_REV = 1.00    # greutate reversie recenta (inmultita cu gate-ul de dispersie)
W_SECT = 0.60   # greutate presiune sector (ancora, nu e gate-uita)
W_INT = 0.40    # interactiune rece-recent x cald-lifetime
GATE_LO = 0.70  # scaling rev la dispersie ~1
GATE_HI = 1.60  # scaling rev la dispersie ridicata
DISP_K = 0.60   # normalizare (chi2/df - 1) / DISP_K
SECTOR = 5      # marimea sectorului

DIAG = {'calls': 0, 'orient_checks': 0, 'orient_flips': 0,
        'contrib': {'rev': 0.0, 'sect': 0.0, 'int': 0.0},
        'x_mean': 0.0, 'gap_max': 0, 'last': {}}


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    p0 = K / float(N) if N > 0 else 0.0
    q0 = 1.0 - p0
    NUMS = tuple(range(1, N + 1))

    n_sec = (N + SECTOR - 1) // SECTOR
    sectors = []
    sec_of = [0] * (N + 1)
    for s in range(n_sec):
        mem = [n for n in NUMS if (n - 1) // SECTOR == s]
        sectors.append(mem)
        for m in mem:
            sec_of[m] = s

    # p-hit: un sector apare intr-o extragere <=> 1 - C(N-S, K)/C(N, K)
    if 0.0 < p0 < 1.0 and N - SECTOR >= K:
        try:
            p_no = math.comb(N - SECTOR, K) / float(math.comb(N, K))
        except (ValueError, OverflowError):
            p_no = (1.0 - p0) ** K
    else:
        p_no = (1.0 - p0) ** K
    p_hit = 1.0 - p_no
    if p_hit <= 1e-12:
        p_hit = 1e-12
    mu_g = (1.0 - p_hit) / p_hit
    sd_g = math.sqrt(1.0 - p_hit) / p_hit

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        D = int(draw_index)

        # --- statistici lifetime ---
        total = 0
        for n in NUMS:
            total += counts.get(n, 0)
        D_hist = total / float(K) if K > 0 else 0.0
        mu_L = D_hist * p0
        sd_L = math.sqrt(D_hist * p0 * q0) if D_hist > 0.0 else 0.0

        # --- orientare deque (verificare: seq[-1] = cel mai nou draw) ---
        W = len(recent_deque)
        flip = False
        if W >= 2 and D >= 1:
            DIAG['orient_checks'] += 1
            so = 0
            sf = 0
            for n in recent_deque[-1]:
                if last_seen.get(n, -10 ** 9) == D - 1:
                    so += 1
            for n in recent_deque[0]:
                if last_seen.get(n, -10 ** 9) == D - 1:
                    sf += 1
            if sf > so:
                flip = True
                DIAG['orient_flips'] += 1
        src = reversed(recent_deque) if flip else recent_deque

        # --- counts in fereastra ---
        c_rec = [0] * (N + 1)
        slots = 0
        for st in src:
            for n in st:
                if 1 <= n <= N:
                    c_rec[n] += 1
                    slots += 1

        # --- z recent + dispersie chi2/df ---
        z_rec = [0.0] * (N + 1)
        x = 1.0
        if W > 0 and 0.0 < p0 < 1.0:
            var_r = W * p0 * q0
            if var_r > 1e-12:
                sd_r = math.sqrt(var_r)
                exp_r = W * p0
                chi2 = 0.0
                E_s = slots / float(N) if (slots > 0 and N > 0) else 0.0
                for n in NUMS:
                    z_rec[n] = (c_rec[n] - exp_r) / sd_r
                    if E_s > 0.0:
                        d = c_rec[n] - E_s
                        chi2 += d * d / E_s
                if slots > 0 and N > 1:
                    x = chi2 / float(N - 1)

        disp = (x - 1.0) / DISP_K
        if disp < 0.0:
            disp = 0.0
        elif disp > 1.0:
            disp = 1.0
        g_rev = GATE_LO + (GATE_HI - GATE_LO) * disp

        # --- gap-uri pe sectoare ---
        z_sec = [0.0] * n_sec
        gap_max = 0
        for s in range(n_sec):
            last = -1
            for m in sectors[s]:
                ls = last_seen.get(m, -1)
                if ls > last:
                    last = ls
            g_s = (D - 1 - last) if last >= 0 else D
            if g_s < 0:
                g_s = 0
            if g_s > gap_max:
                gap_max = g_s
            z_sec[s] = (g_s - mu_g) / sd_g

        # --- alocarea presiunii sectorului: proportional cu deficitul recent ---
        cold = [0.0] * (N + 1)
        cmass = [0.0] * n_sec
        for n in NUMS:
            d = -z_rec[n]
            if d > 0.0:
                cold[n] = d
                cmass[sec_of[n]] += d
        share = [0.0] * (N + 1)
        for n in NUMS:
            mc = cmass[sec_of[n]]
            if cold[n] > 0.0 and mc > 1e-9:
                share[n] = cold[n] / mc
            else:
                share[n] = 1.0 / len(sectors[sec_of[n]])

        # --- relu(tanh(z_lifetime)): doar partea calda istoric ---
        hotL = [0.0] * (N + 1)
        if sd_L > 1e-12:
            for n in NUMS:
                t = (counts.get(n, 0) - mu_L) / sd_L
                if t > 0.0:
                    hotL[n] = math.tanh(t)

        # --- compunere ---
        scores = [0.0] * (N + 1)
        a_rev = 0.0
        a_sec = 0.0
        a_int = 0.0
        for n in NUMS:
            s = sec_of[n]
            rev = W_REV * g_rev * (-z_rec[n])
            sec_t = W_SECT * z_sec[s] * share[n]
            int_t = W_INT * (-z_rec[n]) * hotL[n]
            scores[n] = rev + sec_t + int_t
            a_rev += -rev if rev < 0.0 else rev
            a_sec += -sec_t if sec_t < 0.0 else sec_t
            a_int += -int_t if int_t < 0.0 else int_t

        # --- diag: medii incrementale per componenta ---
        DIAG['calls'] += 1
        c = DIAG['calls']
        DIAG['contrib']['rev'] += (a_rev / N - DIAG['contrib']['rev']) / c
        DIAG['contrib']['sect'] += (a_sec / N - DIAG['contrib']['sect']) / c
        DIAG['contrib']['int'] += (a_int / N - DIAG['contrib']['int']) / c
        DIAG['x_mean'] += (x - DIAG['x_mean']) / c
        if gap_max > DIAG['gap_max']:
            DIAG['gap_max'] = gap_max
        DIAG['last'] = {'x': x, 'g_rev': g_rev, 'disp': disp}

        return sorted(NUMS, key=lambda n: (-scores[n], n))

    rank._diag = DIAG
    return rank