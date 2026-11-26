NAME = "zzhepta7"
DESCRIPTION = 'Model compozit din 7 semnale standardizate: z pe frecventa lifetime (autocalibrat), momentum pe fereastra recenta, echo-ul ultimei extrageri, msg2 = perechi co-aparute in fereastra legate de extragerea cea mai noua, due = gap^HAZ_EXP standardizat sub geometrica(p=5/45), aff1 = kernel recursiv 1/(age+K)^GAMMA pe propria prezenta, aff2 = acelasi kernel pe co-occurrenta cu membrii ultimei extrageri; orientarea deque-ului este verificata si corectata automat fata de last_seen.'
SOURCE = 'model + 2026-09-08'

import math

# ---- constante tunabile (grile de scan pentru ablatie) ----
W_Z = 0.50
W_MOM = 0.80
W_ECHO = 0.35
W_MSG2 = 0.45
W_DUE = 0.25
W_AFF1 = 0.55
W_AFF2 = 0.40
K_AFF = 3
GAMMA_AFF = 0.90
HAZ_EXP = 1.0

DIAG = {'orient_checks': 0, 'orient_flips': 0, 'orient_rule': 'none'}


def create(config):
    N = int(config.number_range)
    KD = int(config.numbers_drawn)
    p = KD / float(N) if N > 0 else 0.0
    q = 1.0 - p
    p2 = (KD * (KD - 1)) / float(N * (N - 1)) if (N > 1 and KD >= 2) else 0.0

    # momente numerice pentru due: G ~ geometrica(q), due_raw = G^HAZ_EXP
    mu_h = 0.0
    m2_h = 0.0
    for t in range(0, 600):
        wgt = p * (q ** t) if (0.0 < p < 1.0) else 0.0
        if t > 50 and wgt < 1e-15:
            break
        v = (float(t) ** HAZ_EXP) if t > 0 else 0.0
        mu_h += wgt * v
        m2_h += wgt * v * v
    sd_h = math.sqrt(max(m2_h - mu_h * mu_h, 1e-12))

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        D = int(draw_index)
        numbers = list(range(1, N + 1))

        # (1) z-score lifetime, autocalibrat din masa totala de counts
        total_slots = 0
        for n in numbers:
            total_slots += counts.get(n, 0)
        z = [0.0] * N
        if KD > 0:
            D_hist = total_slots / float(KD)
            if D_hist >= 1.0 and 0.0 < p < 1.0:
                mu = D_hist * p
                sd = math.sqrt(D_hist * p * q)
                for i, n in enumerate(numbers):
                    z[i] = (counts.get(n, 0) - mu) / sd

        seq = [set(s) for s in recent_deque]
        W = len(seq)
        newest = seq[-1] if W > 0 else set()

        # orientare: seq[-1] ar trebui sa fie cel mai nou draw; validare cu last_seen == D-1
        if W >= 2 and D >= 1:
            DIAG['orient_checks'] += 1
            oldest = seq[0]
            so = 0
            sf = 0
            for n in newest:
                if last_seen.get(n, -10 ** 9) == D - 1:
                    so += 1
            for n in oldest:
                if last_seen.get(n, -10 ** 9) == D - 1:
                    sf += 1
            flip = False
            rule = 'lastseen-hit'
            if sf > so:
                flip = True
            elif sf == so and len(newest) > 0:
                mo = 0.0
                mf = 0.0
                for n in newest:
                    mo += last_seen.get(n, -10 ** 9)
                for n in oldest:
                    mf += last_seen.get(n, -10 ** 9)
                if mf > mo:
                    flip = True
                    rule = 'mean-last_seen'
            if flip:
                seq.reverse()
                newest = seq[-1]
                DIAG['orient_flips'] += 1
            DIAG['orient_rule'] = rule

        mom = [0.0] * N
        echo = [0.0] * N
        msg2 = [0.0] * N
        aff1 = [0.0] * N
        aff2 = [0.0] * N
        due = [0.0] * N

        if W > 0 and 0.0 < p < 1.0:
            # (2) momentum uniform pe fereastra, standardizat
            rc = [0] * N
            for s in seq:
                for n in s:
                    if 1 <= n <= N:
                        rc[n - 1] += 1
            mu_m = W * p
            sd_m = math.sqrt(W * p * q)
            for i in range(N):
                mom[i] = (rc[i] - mu_m) / sd_m

            # (3) echo: apartenenta la cea mai noua extragere
            sd_e = math.sqrt(p * q)
            for i in range(N):
                e = 1.0 if (i + 1) in newest else 0.0
                echo[i] = (e - p) / sd_e

            # (4) msg2: perechi co-aparute in fereastra, afinitate catre newest
            if KD >= 2 and N > 1:
                pc = {}
                base = N + 1
                for s in seq:
                    sl = sorted(x for x in s if 1 <= x <= N)
                    for ai in range(len(sl)):
                        ka = sl[ai] * base
                        for bi in range(ai + 1, len(sl)):
                            key = ka + sl[bi]
                            pc[key] = pc.get(key, 0) + 1
                for i in range(N):
                    n = i + 1
                    m_n = KD - (1 if n in newest else 0)
                    tot = 0
                    for m in newest:
                        if m == n:
                            continue
                        if n < m:
                            tot += pc.get(n * base + m, 0)
                        else:
                            tot += pc.get(m * base + n, 0)
                    mu_p = W * p2 * m_n
                    sd_p = math.sqrt(max(W * p2 * (1.0 - p2) * m_n, 1e-12))
                    msg2[i] = (tot - mu_p) / sd_p

            # (6)+(7) nuclee de recenenta 1/(age+K)^GAMMA_AFF
            wts = [0.0] * W
            wsum1 = 0.0
            wsum2 = 0.0
            for age in range(1, W + 1):
                wv = 1.0 / ((age + K_AFF) ** GAMMA_AFF)
                wts[age - 1] = wv
                wsum1 += wv
                wsum2 += wv * wv
            acc1 = [0.0] * N
            acc2 = [0.0] * N
            for idx in range(W):
                s = seq[idx]
                age = W - idx
                wv = wts[age - 1]
                ov_full = len(s & newest)
                for n in s:
                    if 1 <= n <= N:
                        acc1[n - 1] += wv
                        if ov_full > 0:
                            acc2[n - 1] += wv * (ov_full - (1 if n in newest else 0))
            sd_a = math.sqrt(max(p * q * wsum2, 1e-12))
            use_pairs = (KD >= 2 and N > 1)
            for i in range(N):
                aff1[i] = (acc1[i] - p * wsum1) / sd_a
                if use_pairs:
                    m_n = KD - (1 if (i + 1) in newest else 0)
                    mu_b = wsum1 * p2 * m_n
                    sd_b = math.sqrt(max(wsum2 * p2 * (1.0 - p2) * m_n, 1e-12))
                    aff2[i] = (acc2[i] - mu_b) / sd_b

        # (5) due: gap^HAZ_EXP standardizat sub geometrica
        if sd_h > 0:
            for i in range(N):
                ls = last_seen.get(i + 1, -1)
                g = (D - 1) - ls
                if g < 0:
                    g = 0
                due[i] = ((float(g) ** HAZ_EXP) - mu_h) / sd_h

        score = [0.0] * N
        for i in range(N):
            score[i] = (W_Z * z[i] + W_MOM * mom[i] + W_ECHO * echo[i]
                        + W_MSG2 * msg2[i] + W_DUE * due[i]
                        + W_AFF1 * aff1[i] + W_AFF2 * aff2[i])
        order = sorted(range(N), key=lambda i: (-score[i], i))
        return [i + 1 for i in order]

    return rank