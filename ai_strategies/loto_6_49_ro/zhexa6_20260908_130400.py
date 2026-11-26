NAME = "zhexa6"
DESCRIPTION = ('Scor compozit standardizat din 7 semnale: z-score al frecventei pe fereastra recenta '
               '(shrinkage k), momentum EWMA, echo ultima extragere, mesaj penultima extragere, '
               'hazard de intarziere (due, semn scanabil) si afinitate de co-aparitie propagata din '
               'ultimele 1-2 extrageri cu decadere in lege de putere. Orientarea deque-ului este '
               'revalidata la fiecare apel fata de last_seen; sortare determinista.')
SOURCE = 'model + 2026-09-08'

import math

# ---- hipermarametri (tinte de scan pentru meta-optimizator) ----
W_Z = 0.30        # z-score fereastra (scan 0-0.6)
W_MOM = 0.40      # momentum EWMA
W_ECHO = 0.50     # echo ultima extragere (scan 0.2-1.0)
W_MSG2 = 0.30     # penultima extragere (scan 0-1.5)
W_DUE = 0.25      # due; semn scanabil (+/-0.25)
W_AFF1 = 0.40     # afinitate seed=ultima extragere (scan 0-0.8)
W_AFF2 = 0.20     # afinitate seed=penultima extragere
K = 3             # shrinkage {1,3,5,8}
LAM = 0.02        # decay EWMA momentum (scan 0.005-0.1)
GAMMA_AFF = 0.90  # decay putere afinitate (scan 0.8-0.97)
HAZ_EXP = 1.0     # panta hazardului due {0.5, 1.0, 1.5}
DUE_CLAMP = 2.5


def create(config):
    N = int(config.number_range)
    D = int(config.numbers_drawn)
    p = float(D) / float(N)

    def _zstd(d):
        vals = list(d.values())
        mu = sum(vals) / len(vals)
        var = sum((v - mu) * (v - mu) for v in vals) / len(vals)
        sd = math.sqrt(var)
        if sd < 1e-12:
            return dict.fromkeys(d, 0.0)
        return {k: (v - mu) / sd for k, v in d.items()}

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        seq = [set(s) for s in recent_deque]
        m = len(seq)

        # --- validare orientare: ipoteza deque[-1] = cea mai noua extragere ---
        newest_ls = set()
        prev = draw_index - 1
        for n in numbers:
            ls = last_seen.get(n)
            if ls is not None and ls == prev:
                newest_ls.add(n)
        if m and newest_ls and (len(newest_ls & seq[0]) > len(newest_ls & seq[-1])):
            seq.reverse()
        seq.reverse()  # normalizat intern: seq[0] = cea mai noua extragere
        if recent_window and m > int(recent_window):
            seq = seq[:int(recent_window)]
            m = len(seq)
        newest = seq[0] if m else newest_ls
        second = seq[1] if m >= 2 else set()

        # 1) z-score al frecventei pe fereastra, cu shrinkage
        cw = dict.fromkeys(numbers, 0)
        for s in seq:
            for n in s:
                cw[n] += 1
        exp_c = m * p
        sd_c = math.sqrt(exp_c * (1.0 - p)) if m else 1.0
        shrink = (m / (m + K)) if m else 0.0
        z = {n: ((cw[n] - exp_c) / sd_c) * shrink for n in numbers}

        # 2) momentum: rata EWMA / p - 1
        ew = dict.fromkeys(numbers, 0.0)
        wsum = 0.0
        for a in range(m):
            w = math.exp(-LAM * a)
            wsum += w
            for n in seq[a]:
                ew[n] += w
        mom = {}
        for n in numbers:
            rate = (ew[n] / wsum) if wsum > 1e-12 else 0.0
            mom[n] = (rate / p - 1.0) * shrink

        # 3) echo + msg2: indicatori de apartenenta
        echo = {n: (1.0 if n in newest else 0.0) for n in numbers}
        msg2 = {n: (1.0 if n in second else 0.0) for n in numbers}

        # 4) due: hazard de intarziere fata de gap-ul asteptat (geometric)
        mean_gap = (1.0 - p) / p
        due = {}
        for n in numbers:
            ls = last_seen.get(n)
            g = (draw_index - ls) if ls is not None else draw_index
            if g < 0:
                g = 0
            hz = ((g / mean_gap) ** HAZ_EXP) - 1.0
            due[n] = max(-DUE_CLAMP, min(DUE_CLAMP, hz))

        # 5)+6) afinitate de co-aparitie, seed = ultima / penultima extragere
        def _affinity(seed, self_pos):
            acc = dict.fromkeys(numbers, 0.0)
            norm = 0.0
            for a in range(m):
                if a == self_pos:
                    continue
                s = seq[a]
                hit = len(s & seed)
                if hit:
                    w = (1.0 + a) ** (-GAMMA_AFF)
                    norm += w
                    for n in s:
                        acc[n] += w * hit
            if norm <= 1e-12:
                return None
            return {n: acc[n] / norm for n in numbers}

        aff1 = _affinity(newest, 0) if newest else None
        aff2 = _affinity(second, 1) if second else None

        zs = _zstd(z)
        ms = _zstd(mom)
        es = _zstd(echo)
        gs = _zstd(msg2)
        ds = _zstd(due)
        a1 = _zstd(aff1) if aff1 else None
        a2 = _zstd(aff2) if aff2 else None

        out = []
        for n in numbers:
            sc = (W_Z * zs[n] + W_MOM * ms[n] + W_ECHO * es[n]
                  + W_MSG2 * gs[n] + W_DUE * ds[n])
            if a1 is not None:
                sc += W_AFF1 * a1[n]
            if a2 is not None:
                sc += W_AFF2 * a2[n]
            out.append((sc, n))
        out.sort(key=lambda t: (-t[0], t[1]))
        return [n for _, n in out]

    return rank