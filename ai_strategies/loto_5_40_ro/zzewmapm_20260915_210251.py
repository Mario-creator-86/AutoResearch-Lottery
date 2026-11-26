import math
from collections import defaultdict

NAME = "zzewmapm"
DESCRIPTION = "Compozit de z-score-uri normalizate statistic: frecventa globala z, rata EWMA recenta z (half-life 40) cu poarta de entropie, centralitate PMI pe graf de perechi si hazard de intarziere z; shrinkage non-liniar tanh pe toate componentele."
SOURCE = "glm-5.3-flash + 2026-09-15"


def create(config):
    P = config.number_range          # ex. 40
    W = config.numbers_drawn         # ex. 6
    p = W / float(P)                 # prob. de aparitie per numar per extragere
    HALF_LIFE = 40.0
    LAM = 0.5 ** (1.0 / HALF_LIFE)
    W_G = 0.55                       # pondere frecventa globala (mica: e zgomotos)
    W_R = 1.00                       # pondere recenta EWMA (semnalul principal)
    W_PMI = 0.40                     # pondere perechi PMI
    W_O = 0.12                       # pondere intarziere
    PAIR_CAP = 120

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, P + 1))
        N = max(int(draw_index) + 1, 1)
        sd_g = math.sqrt(max(N * p * (1.0 - p), 1e-12))

        # --- z global (frecventa totala) ---
        z_g = {}
        for n in numbers:
            c = counts.get(n, 0)
            z_g[n] = (c - N * p) / sd_g

        # --- EWMA recency z + poarta de entropie ---
        draws = list(recent_deque)
        m = len(draws)
        z_r = {n: 0.0 for n in numbers}
        q = {n: 0.0 for n in numbers}
        w_r, w_g = W_R, W_G
        if m > 0:
            rnum = {n: 0.0 for n in numbers}
            S = 0.0
            for i in range(m):
                wgt = LAM ** (m - 1 - i)
                S += wgt
                for n in draws[i]:
                    rnum[n] += wgt
            S2 = (1.0 - LAM ** (2 * m)) / (1.0 - LAM * LAM)
            var_r = (p * (1.0 - p)) * S2 / (S * S)
            sd_r = math.sqrt(var_r) if var_r > 1e-18 else 1e-9
            for n in numbers:
                r_n = rnum[n] / S
                q[n] = r_n
                z_r[n] = (r_n - p) / sd_r
            # entropia distributiei per-numar din fereastra -> gate adaptiv
            tot = sum(q.values())
            if tot > 0:
                H = 0.0
                for n in numbers:
                    pn = q[n] / tot
                    if pn > 0.0:
                        H -= pn * math.log(pn)
                t = max(0.0, (math.log(P) - H) / math.log(P))  # 0 = uniform
                w_r = W_R * (0.75 + 1.25 * t)
                w_g = W_G * (1.15 - 0.30 * t)

        # --- PMI perechi (fereastra recenta, cap la PAIR_CAP) ---
        pmi_z = {n: 0.0 for n in numbers}
        if m >= 2:
            pw = draws[-PAIR_CAP:] if m > PAIR_CAP else draws
            mw = len(pw)
            wcnt = {n: 0 for n in numbers}
            pcnt = defaultdict(int)
            for dset in pw:
                ds = sorted(dset)
                for n in ds:
                    wcnt[n] += 1
                for i in range(len(ds)):
                    a = ds[i]
                    for j in range(i + 1, len(ds)):
                        pcnt[(a, ds[j])] += 1
            num = {n: 0.0 for n in numbers}
            den = {n: 0.0 for n in numbers}
            for (a, b), c in pcnt.items():
                exp_c = mw * (wcnt[a] / float(mw)) * (wcnt[b] / float(mw))
                if exp_c <= 1e-12:
                    continue
                val = c * math.log(c / exp_c)
                num[a] += val
                den[a] += c
                num[b] += val
                den[b] += c
            vals = []
            for n in numbers:
                v = num[n] / den[n] if den[n] > 1e-12 else 0.0
                pmi_z[n] = v
                vals.append(v)
            mu = sum(vals) / float(P)
            sd_p = math.sqrt(sum((v - mu) ** 2 for v in vals) / float(P))
            if sd_p < 1e-12:
                sd_p = 1.0
            for n in numbers:
                pmi_z[n] = (pmi_z[n] - mu) / sd_p

        # --- intarziere z (gap vs. media geometrica) ---
        sd_o = math.sqrt(1.0 - p) / p
        z_o = {}
        for n in numbers:
            g = draw_index - last_seen.get(n, -1)
            if g > 50:
                g = 50
            z_o[n] = (g - 1.0 / p) / sd_o

        scores = {}
        for n in numbers:
            s = (w_g * math.tanh(z_g[n] / 2.5)
                 + w_r * math.tanh(z_r[n] / 2.5)
                 + W_PMI * math.tanh(pmi_z[n] / 2.0)
                 + W_O * math.tanh(z_o[n] / 2.5))
            scores[n] = s
        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank