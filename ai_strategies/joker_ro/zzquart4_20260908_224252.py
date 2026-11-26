NAME = "zzquart4"
DESCRIPTION = 'Cvartet de semnale standardizate combinate liniar: (1) deviatia binomiala a fiecarui numar fata de frecventa asteptata pe fereastra recenta (flip-abil prin semnul lui W_DEV), (2) masa de co-occurrere conditionata de numerele-semina din ultimele K_TAIL extrageri, cu decay recenT GAMMA_P, (3) camp KDE gaussian pe axa numerelor din histograma recency-weighted, corectat la margini, cu latime SIGMA_POS, (4) ciclu de cadenta: bump gaussian cand varsta curenta de la ultima aparitie coincide cu intervalul mediu inter-aparitii, shrinkat spre 1/p. Orientarea deque-ului (newest_last) este verificata fata de last_seen si corectata automat.'
SOURCE = 'model + 2026-09-09'

import math

# ---- constante tunabile (grile de scan pentru ablatie) ----
W_PAIR = 0.90          # [0, 1.5]
W_DEV = 0.70           # [-0.8, 1.0]; negativ = flip spre anti-momentum
W_POS = 0.55           # [0, 1]
W_CAD = 0.35           # [0, 1]
GAMMA_P = 0.97         # [0.90, 0.99] decay recenT pt co-occurrere/KDE
K_TAIL = 2             # {1, 2, 3, 5} extrageri noi care formeaza semintele
SIGMA_POS = 8.0        # [4, 12] latimea kernelului KDE
CAD_SHRINK = 3.0       # shrinkage al mean-gapului spre 1/p
CAD_WIDTH_FRAC = 0.45  # latimea bumpului de cadenta ca fractie din mean-gap

DIAG = {'orient_checks': 0, 'orient_flips': 0, 'orient_rule': 'lastseen-hit-then-sum'}


def create(config):
    N = int(config.number_range)
    KD = int(config.numbers_drawn)
    p = KD / float(N) if N > 0 else 0.0
    q = 1.0 - p
    base_gap = (1.0 / p) if (0.0 < p < 1.0) else float(N)
    sigma = max(float(SIGMA_POS), 0.5)

    # kernel gaussian pe distanta + corectia de margine per numar
    ker = [math.exp(-0.5 * (d / sigma) ** 2) for d in range(N)]
    edge = [0.0] * (N + 1)
    for n in range(1, N + 1):
        acc = 0.0
        for m in range(1, N + 1):
            acc += ker[abs(n - m)]
        edge[n] = acc

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        seq = [set(s) for s in recent_deque]
        W = len(seq)
        D = int(draw_index)

        # ---- orientare: seq[-1] trebuie sa fie cel mai nou draw ----
        if W >= 2 and D >= 1:
            DIAG['orient_checks'] += 1
            hits_new = 0
            hits_old = 0
            for n in seq[-1]:
                if last_seen.get(n, -10 ** 9) == D - 1:
                    hits_new += 1
            for n in seq[0]:
                if last_seen.get(n, -10 ** 9) == D - 1:
                    hits_old += 1
            flip = False
            if hits_old > hits_new:
                flip = True
            elif hits_old == hits_new:
                mo = 0.0
                mf = 0.0
                for n in seq[-1]:
                    mo += last_seen.get(n, -10 ** 9)
                for n in seq[0]:
                    mf += last_seen.get(n, -10 ** 9)
                if mf > mo:
                    flip = True
            if flip:
                seq.reverse()
                DIAG['orient_flips'] += 1

        wts = [GAMMA_P ** (W - 1 - i) for i in range(W)]

        kt = int(K_TAIL)
        if kt > W:
            kt = W
        seeds = set()
        if kt > 0:
            for i in range(W - kt, W):
                seeds |= seq[i]
        seed_zone = W - kt  # draw-urile cu i >= seed_zone sunt blocul de seminte

        h = [0.0] * (N + 1)         # histograma recency-weighted (pt KDE)
        ccnt = [0] * (N + 1)        # counts brute pe fereastra (pt dev)
        pairmass = [0.0] * (N + 1)  # masa de co-occurrere cu semintele
        ages = [[] for _ in range(N + 1)]
        for i in range(W):
            dr = seq[i]
            wi = wts[i]
            age = W - 1 - i
            so = 0
            if i < seed_zone:
                for m in dr:
                    if m in seeds:
                        so += 1
            for m in dr:
                h[m] += wi
                ccnt[m] += 1
                ages[m].append(age)
                if so > 0:
                    pairmass[m] += wi * so

        def zscale(vec):
            mean = 0.0
            for n in range(1, N + 1):
                mean += vec[n]
            mean /= float(N)
            var = 0.0
            for n in range(1, N + 1):
                d = vec[n] - mean
                var += d * d
            sd = math.sqrt(var / float(N))
            if sd < 1e-12:
                return None
            out = [0.0] * (N + 1)
            for n in range(1, N + 1):
                out[n] = (vec[n] - mean) / sd
            return out

        # (1) deviatia binomiala pe fereastra
        devz = [0.0] * (N + 1)
        if W > 0 and 0.0 < p < 1.0:
            mu = W * p
            sdw = math.sqrt(W * p * q)
            for n in range(1, N + 1):
                devz[n] = (ccnt[n] - mu) / sdw

        # (3) KDE pe axa numerelor, corectata la margine
        pos = [0.0] * (N + 1)
        for n in range(1, N + 1):
            acc = 0.0
            for m in range(1, N + 1):
                hm = h[m]
                if hm > 0.0:
                    acc += hm * ker[abs(n - m)]
            pos[n] = acc / edge[n] if edge[n] > 0.0 else 0.0

        # (4) cadenta: bump la age_now ~= mean-gap shrinkat spre 1/p
        cad = [0.0] * (N + 1)
        for n in range(1, N + 1):
            lst = ages[n]
            k = len(lst)
            if k == 0:
                continue
            lst.sort()
            if k >= 2:
                ng = k - 1
                mg_raw = 0.0
                for j in range(ng):
                    mg_raw += lst[j + 1] - lst[j]
                mg_raw /= float(ng)
                mg = (ng * mg_raw + CAD_SHRINK * base_gap) / (ng + CAD_SHRINK)
            else:
                mg = base_gap
            age_now = lst[0]
            denom = CAD_WIDTH_FRAC * mg
            if denom < 0.75:
                denom = 0.75
            zz = (age_now - mg) / denom
            if zz > 40.0:
                cad[n] = 0.0
            else:
                cad[n] = math.exp(-0.5 * zz * zz)

        # ---- combinare standardizata ----
        total = [0.0] * (N + 1)
        zp = zscale(pairmass)
        zpo = zscale(pos)
        zc = zscale(cad)
        if zp is not None:
            for n in range(1, N + 1):
                total[n] += W_PAIR * zp[n]
        if W > 0 and 0.0 < p < 1.0:
            for n in range(1, N + 1):
                total[n] += W_DEV * devz[n]
        if zpo is not None:
            for n in range(1, N + 1):
                total[n] += W_POS * zpo[n]
        if zc is not None:
            for n in range(1, N + 1):
                total[n] += W_CAD * zc[n]

        return sorted(numbers, key=lambda n: (-total[n], -counts.get(n, 0), n))

    return rank