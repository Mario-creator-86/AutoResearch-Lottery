NAME = "zzcycgat"
DESCRIPTION = 'Cicluri de intarziere pe sectoare de latime configurabila (SEC_SIZE), normalizate la ciclul de asteptare teoretic al sectorului, combinate cu hazard de gap per-numar plafonat (GAP_CAP) si deviatia z macro pe termen lung; semnalele per-numar trec printr-un gate liniar (GATE_MIX) alimentat de presiunea de ciclu a sectorului. Semnele reversion/momentum sunt comutabile prin SEC_SIGN/DEV_SIGN, iar mecanismul este invariant la orientarea deque-ului.'
SOURCE = 'model cantitativ + decision ledger 2026-09-17'

import math

DIAG = {'calls': 0, 'sec_span': 0.0, 'gap_span': 0.0, 'dev_span': 0.0,
        'gate_mean': 0.0, 'L': 0, 'draw_index': -1}

# hiperparametri expusi pentru grid-search (patchabili la nivel de modul)
SEC_SIZE = 6      # {4, 6, 8} latimea sectorului
SEC_SIGN = 1      # +1 = reversion (sectoare intarziate promote), -1 = momentum
DEV_SIGN = 1      # +1 = reversion (reci promote), -1 = momentum (calde promote)
W_SEC = 0.7       # {0.35, 0.7, 1.05} pondere presiune sector
W_GAP = 0.5       # {0.3, 0.5, 0.8} pondere hazard gap per-numar
W_DEV = 0.4       # {0, 0.4, 0.8} pondere deviatie z macro
GAP_CAP = 2.5     # {1.5, 2.5, 4.0} plafon pe raportul gap/asteptat
GATE_MIX = 0.5    # {0.3, 0.5, 0.7} intensitatea gate-ului sectorial


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    q = K / float(N)

    cache = {}

    def build(size):
        if size in cache:
            return cache[size]
        S = (N + size - 1) // size
        sec_of = [0] * (N + 1)
        members = []
        for s in range(S):
            lo = s * size + 1
            hi = min((s + 1) * size, N)
            members.append(tuple(range(lo, hi + 1)))
            for n in range(lo, hi + 1):
                sec_of[n] = s
        p_sec = []
        for s in range(S):
            m = len(members[s])
            miss = 1.0
            for i in range(K):
                num = N - m - i
                if num <= 0:
                    miss = 0.0
                    break
                miss *= float(num) / float(N - i)
            p_sec.append(1.0 - miss)
        cache[size] = (S, sec_of, members, p_sec)
        return cache[size]

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        S, sec_of, members, p_sec = build(SEC_SIZE)
        T = float(draw_index) if draw_index > 0 else 1.0
        mu = T * q
        sd = math.sqrt(T * q * (1.0 - q))
        if sd < 1e-9:
            sd = 1e-9

        # per-numar: hazard de gap plafonat + deviatie z macro (clip la 3 sigma)
        gap_h = [0.0] * (N + 1)
        dev_s = [0.0] * (N + 1)
        for n in range(1, N + 1):
            ls = last_seen.get(n)
            gap = (draw_index - ls) if ls is not None else (draw_index + 1)
            if gap < 0:
                gap = 0
            r = gap * q
            if r > GAP_CAP:
                r = GAP_CAP
            gap_h[n] = r / GAP_CAP
            z = (counts.get(n, 0) - mu) / sd
            if z > 3.0:
                z = 3.0
            elif z < -3.0:
                z = -3.0
            dev_s[n] = -DEV_SIGN * (z / 3.0)

        # per-sector: ciclu de intarziere normalizat + gate liniar
        sec_sc = [0.0] * S
        mult = [0.0] * S
        for s in range(S):
            gap_s = draw_index + 1
            for n in members[s]:
                ls = last_seen.get(n)
                if ls is not None:
                    g = draw_index - ls
                    if g < gap_s:
                        gap_s = g
            v = (gap_s * p_sec[s] - 1.0) * 0.5
            if v > 1.0:
                v = 1.0
            elif v < -1.0:
                v = -1.0
            v *= SEC_SIGN
            sec_sc[s] = v
            mult[s] = (1.0 - GATE_MIX) + GATE_MIX * (0.5 + 0.5 * v)

        scores = [0.0] * (N + 1)
        for n in range(1, N + 1):
            s = sec_of[n]
            scores[n] = W_SEC * sec_sc[s] + mult[s] * (W_GAP * gap_h[n] + W_DEV * dev_s[n])

        # diagnostic pentru ablation: span per componenta + stats gate
        DIAG['sec_span'] = max(sec_sc) - min(sec_sc)
        DIAG['gap_span'] = max(gap_h[1:]) - min(gap_h[1:])
        DIAG['dev_span'] = max(dev_s[1:]) - min(dev_s[1:])
        DIAG['gate_mean'] = sum(mult) / float(S)
        DIAG['L'] = len(recent_deque)
        DIAG['draw_index'] = draw_index
        DIAG['calls'] += 1

        return sorted(range(1, N + 1), key=lambda n: (-scores[n], n))

    rank._diag = DIAG
    return rank