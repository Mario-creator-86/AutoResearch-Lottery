import math

NAME = "zzentrop"
DESCRIPTION = (
    'Strategie adaptiva de regim: un chi2 pe fereastra recenta (proxy de entropie) '
    'moduleaza combinarea ne-liniara dintre momentum EWMA (half-life 10), z-score-ul '
    'istoric, intarzierea normalizata prin hazard (gap/2, cap 3) si un hub de perechi co-aparute.'
)
SOURCE = 'model + 2026-09-07'


def create(config):
    P = int(config.number_range)          # 24
    KD = int(config.numbers_drawn)        # 12
    PH = KD / float(P)                    # p aparitie per extragere = 0.5
    INV_G = 1.0 / PH                      # gap mediu asteptat = 2
    LAM = 0.5 ** (1.0 / 10.0)             # half-life EWMA = 10 extrageri
    PAIR_WIN = 24                         # fereastra pt co-aparitii
    GAP_CAP = 3.0                         # cap gap-ratio

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, P + 1))
        M = len(numbers)
        N = float(max(int(draw_index), 1))
        W = max(1, min(int(recent_window), len(recent_deque)))
        draws = list(recent_deque)[-W:]
        L = len(draws)

        # 1) EWMA momentum + contoare plate pe fereastra
        wsum = 0.0
        w2 = 0.0
        ew = dict.fromkeys(numbers, 0.0)
        r = dict.fromkeys(numbers, 0)
        for i in range(L):
            w = LAM ** (L - 1 - i)
            wsum += w
            w2 += w * w
            d = draws[i]
            for n in d:
                if 1 <= n <= P:
                    ew[n] += w
                    r[n] += 1
        inv_w = 1.0 / wsum if wsum > 0 else 1.0
        sd_ew = math.sqrt(PH * (1.0 - PH) * (w2 / (wsum * wsum))) if wsum > 0 else 1.0
        sd_ew = max(sd_ew, 1e-6)

        # 2) poarta de regim: chi2 pe fereastra vs uniformitate
        mu_r = W * PH
        x2 = 0.0
        for n in numbers:
            x2 += (r[n] - mu_r) ** 2
        x2 /= max(mu_r, 1e-9)
        dof = M - 1
        gate = math.tanh((x2 - dof) / math.sqrt(2.0 * dof))

        # 3) z istoric + top-12 recent pt hub
        mu_l = N * PH
        sd_l = math.sqrt(N * PH * (1.0 - PH)) + 1e-9
        hot = sorted(numbers, key=lambda n: -r[n])[:KD]

        # 4) hub de perechi co-aparute pe ultimele PAIR_WIN extrageri
        pw = draws[-min(L, PAIR_WIN):] if L > 0 else []
        pc = {}
        for d in pw:
            ds = sorted(d)
            ln = len(ds)
            for i in range(ln):
                a = ds[i]
                if a < 1 or a > P:
                    continue
                for j in range(i + 1, ln):
                    b = ds[j]
                    if b < 1 or b > P:
                        continue
                    key = (a, b)
                    if key in pc:
                        pc[key] += 1
                    else:
                        pc[key] = 1
        hub = {}
        for n in numbers:
            s = 0.0
            for h in hot:
                if h == n:
                    continue
                s += pc.get((n, h), 0) if n < h else pc.get((h, n), 0)
            hub[n] = s
        mh = sum(hub.values()) / float(M)
        vh = sum((hub[n] - mh) ** 2 for n in numbers) / float(M)
        sh = math.sqrt(vh) + 1e-9

        # 5) combinare ne-liniara modulata de regim
        w_mom = 0.9 + 0.5 * gate
        w_due = 0.4 - 0.3 * gate
        scores = {}
        for n in numbers:
            m_n = math.tanh((ew[n] * inv_w - PH) / sd_ew)
            long_n = math.tanh((counts.get(n, 0) - mu_l) / sd_l)
            ls = last_seen.get(n)
            g = float(draw_index - ls) if ls is not None else float(draw_index) + 100.0
            due_n = math.tanh(0.8 * (min(g / INV_G, GAP_CAP) - 1.0))
            pair_n = math.tanh((hub[n] - mh) / sh)
            scores[n] = w_mom * m_n + 0.5 * long_n + w_due * due_n + 0.45 * pair_n

        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank