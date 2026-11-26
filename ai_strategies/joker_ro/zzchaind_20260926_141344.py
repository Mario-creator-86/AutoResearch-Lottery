NAME = "zzchaind"
DESCRIPTION = 'Selectare secventiala tip DPP: scor marginal din EWMA standardizat pe fereastra recenta, z-score lifetime autocalibrat, hazard de gap log-standardizat, overdue pe sectoare si echo Markov al ultimei extrageri. Permutarea se construieste greedy: fiecare numar ales aduce catre ramasi bonusuri din matricea de tranzitii (perechi din extrageri consecutive din fereastra) si o penalizare de repulsie pe sector. Orientarea deque-ului este verificata fata de last_seen si corectata automat.'
SOURCE = 'model + 2026-09-26'

import math

# ---- constante tunabile (grile de scan pentru ablatie) ----
W_EWMA = 1.0
LAMBDA = 0.90
W_Z = 0.60
W_DUE = 0.50
W_SECT = 0.40
W_ECHO = 0.40
PAIR_W = 0.60
REP_W = 0.30
SECT_SIZE = 5

DIAG = {'orient_checks': 0, 'orient_flips': 0}


def create(config):
    N = int(config.number_range)
    KD = int(config.numbers_drawn)
    p = KD / float(N) if N > 0 else 0.0
    q = 1.0 - p

    n_sect = (N + SECT_SIZE - 1) // SECT_SIZE
    if n_sect < 1:
        n_sect = 1

    sect_of = [0] * (N + 1)
    for n in range(1, N + 1):
        si = (n - 1) // SECT_SIZE
        if si > n_sect - 1:
            si = n_sect - 1
        sect_of[n] = si

    sect_members = []
    sexp = [0.0] * n_sect
    for si in range(n_sect):
        lo = si * SECT_SIZE + 1
        hi = lo + SECT_SIZE - 1
        if hi > N:
            hi = N
        sect_members.append(list(range(lo, hi + 1)))
        size = hi - lo + 1
        if size > 0 and 0.0 <= p < 1.0:
            sexp[si] = 1.0 - ((1.0 - p) ** size)

    numbers = list(range(1, N + 1))

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        D = int(draw_index)
        seq = [set(s) for s in recent_deque]
        W = len(seq)

        # --- orientare: seq[-1] ar trebui sa fie cel mai nou draw; validare last_seen == D-1 ---
        if W >= 2 and D >= 1:
            DIAG['orient_checks'] += 1
            new_hits = 0
            old_hits = 0
            for n in seq[W - 1]:
                if last_seen.get(n, -10 ** 9) == D - 1:
                    new_hits += 1
            for n in seq[0]:
                if last_seen.get(n, -10 ** 9) == D - 1:
                    old_hits += 1
            if old_hits > new_hits:
                seq.reverse()
                DIAG['orient_flips'] += 1
        L = seq[W - 1] if W > 0 else set()

        # --- ponderi de recenza: age 0 = cel mai nou ---
        weights = [1.0] * W
        for t in range(W - 2, -1, -1):
            weights[t] = weights[t + 1] * LAMBDA

        # --- z-score lifetime, autocalibrat din masa totala de counts ---
        z = [0.0] * (N + 1)
        if KD > 0:
            tot = 0
            for n in numbers:
                tot += counts.get(n, 0)
            if tot > 0:
                D_hist = tot / float(KD)
                if D_hist >= 1.0 and 0.0 < p < 1.0:
                    mu = D_hist * p
                    sd = math.sqrt(D_hist * p * q)
                    inv = 1.0 / sd
                    for n in numbers:
                        z[n] = (counts.get(n, 0) - mu) * inv

        # --- EWMA pe fereastra, standardizat peste numere ---
        r = [0.0] * (N + 1)
        if W > 0:
            wsum = 0.0
            for t in range(W):
                wsum += weights[t]
            if wsum > 0:
                inv = 1.0 / wsum
                for t in range(W):
                    wt = weights[t] * inv
                    if wt <= 0.0:
                        continue
                    for n in seq[t]:
                        r[n] += wt
                mean_r = 0.0
                for n in numbers:
                    mean_r += r[n]
                mean_r /= N
                var_r = 0.0
                for n in numbers:
                    d = r[n] - mean_r
                    var_r += d * d
                sd_r = math.sqrt(var_r / N)
                if sd_r < 1e-9:
                    sd_r = 1.0
                inv = 1.0 / sd_r
                for n in numbers:
                    r[n] = (r[n] - mean_r) * inv

        # --- due: log-gap standardizat peste numere ---
        gaps = [0.0] * (N + 1)
        logs = [0.0] * N
        for i, n in enumerate(numbers):
            ls = last_seen.get(n, None)
            g = float(D + 1) if ls is None else float(D - ls)
            if g < 0.0:
                g = 0.0
            gaps[n] = g
            logs[i] = math.log1p(g)
        m = 0.0
        for v in logs:
            m += v
        m /= N
        var = 0.0
        for v in logs:
            d = v - m
            var += d * d
        sdg = math.sqrt(var / N) if N > 1 else 1.0
        if sdg < 1e-9:
            sdg = 1.0
        due = [0.0] * (N + 1)
        for i, n in enumerate(numbers):
            due[n] = (logs[i] - m) / sdg

        # --- overdue pe sectoare: deficit fata de rata asteptata 1-(1-p)^size ---
        rate = [0.0] * n_sect
        if W > 0:
            s_hits = [0] * n_sect
            for t in range(W):
                seen = [False] * n_sect
                for n in seq[t]:
                    seen[sect_of[n]] = True
                for si in range(n_sect):
                    if seen[si]:
                        s_hits[si] += 1
            for si in range(n_sect):
                rate[si] = s_hits[si] / float(W)

        # --- tranzitii Markov consecutive in fereastra: T[x][y] = pondere x->y ---
        have_T = W >= 2
        T = [dict() for _ in range(N + 1)]
        if have_T:
            for t in range(W - 1):
                w = weights[t + 1]
                if w <= 0.0:
                    continue
                a = seq[t]
                b = seq[t + 1]
                for x in a:
                    Tx = T[x]
                    for y in b:
                        if y != x:
                            Tx[y] = Tx.get(y, 0.0) + w

        # --- echo: membrii ultimei extrageri + tranzitiile lor catre restul ---
        echo = [0.0] * (N + 1)
        for n in L:
            echo[n] += 1.0
        if have_T:
            for x in L:
                Tx = T[x]
                for y, w in Tx.items():
                    echo[y] += w

        # --- scoruri marginale ---
        base = [0.0] * (N + 1)
        for n in numbers:
            si = sect_of[n]
            e = echo[n]
            e_term = (e / (1.0 + e)) if e > 0.0 else 0.0
            base[n] = (W_EWMA * r[n] + W_Z * z[n] + W_DUE * due[n]
                       + W_SECT * (sexp[si] - rate[si]) + W_ECHO * e_term)

        # --- constructie greedy cu interactiuni (DPP-like): ordinea = permutarea ---
        acc = [0.0] * (N + 1)
        cur = [0.0] * (N + 1)
        for n in numbers:
            cur[n] = base[n]
        in_picked = [False] * (N + 1)
        sec_cnt = [0] * n_sect
        picked = []

        for _ in range(N):
            best_n = -1
            best_s = -1e18
            for n in numbers:
                if in_picked[n]:
                    continue
                s = cur[n]
                if s > best_s:
                    best_s = s
                    best_n = n
            if best_n < 0:
                break
            picked.append(best_n)
            in_picked[best_n] = True
            si = sect_of[best_n]
            sec_cnt[si] += 1
            if have_T:
                for y, w in T[best_n].items():
                    if not in_picked[y]:
                        a2 = acc[y] + w
                        acc[y] = a2
                        sy = sect_of[y]
                        cur[y] = base[y] + PAIR_W * (a2 / (1.0 + a2)) - REP_W * sec_cnt[sy]
            rep = -REP_W * sec_cnt[si]
            for y in sect_members[si]:
                if not in_picked[y]:
                    a2 = acc[y]
                    pt = PAIR_W * (a2 / (1.0 + a2)) if a2 > 0.0 else 0.0
                    cur[y] = base[y] + pt + rep

        if len(picked) < N:
            for n in numbers:
                if not in_picked[n]:
                    picked.append(n)
        return picked

    return rank