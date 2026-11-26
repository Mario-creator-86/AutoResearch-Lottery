NAME = "zcycpgen"
DESCRIPTION = ("Amestec geometric (multiplicativ) de patru semnale: faza ciclica a intarzierii "
               "cos(2*pi*gap/interval), centralitate damped pe graful de co-aparitii din fereastra "
               "recenta, deviatia z a EWMA-ului de frecvente si deviatia z pe termen lung; ponderea "
               "semnalului recent este modulata de entropia Shannon a distributiei empirice din fereastra.")
SOURCE = "model GLM strategist + 2026-09-18"

import math
import numpy as np

EW_HALFLIFE = 45.0
PAIR_DAMP = 0.55
PAIR_ITERS = 2
GAP_CAP_MULT = 6.0
WL = 0.08
WR = 0.25
WC = 0.10
WP = 0.12
MIX_BASE = 0.5
MIX_GAIN = 25.0


def create(config):
    P = int(config.number_range)
    Wn = int(config.numbers_drawn)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, P + 1))
        N = float(draw_index) if draw_index and draw_index > 0 else 1.0
        p = Wn / float(P)
        draws = [s for s in recent_deque if s]
        D = len(draws)
        half = math.tanh

        # (1) deviatie pe termen lung: z binomial pe count-urile totale
        mu = N * p
        sdL = math.sqrt(max(N * p * (1.0 - p), 1e-12))
        tL = [half(0.5 * (counts.get(n, 0) - mu) / sdL) for n in numbers]

        # ponderi EWMA pe fereastra (presupun cea mai noua extragere la coada deque-ului)
        hl = max(4.0, min(EW_HALFLIFE, max(int(recent_window), 1) / 3.0))
        ws = [2.0 ** (-(D - 1 - i) / hl) for i in range(D)]
        S1 = math.fsum(ws)
        S2 = math.fsum(w * w for w in ws)
        ew = [0.0] * P
        cw = [0] * P
        for i in range(D):
            w = ws[i]
            for n in draws[i]:
                j = int(n) - 1
                if 0 <= j < P:
                    ew[j] += w
                    cw[j] += 1
        sdR = math.sqrt(max(S2 * p * (1.0 - p), 1e-12))
        zR = [(ew[j] - S1 * p) / sdR for j in range(P)]

        # (2) entropie Shannon normalizata -> modulator de regim pentru semnalul recent
        tot = float(sum(cw))
        if tot > 0.0:
            H = 0.0
            for c in cw:
                if c > 0:
                    q = c / tot
                    H -= q * math.log(q)
            Hn = H / math.log(P)
        else:
            Hn = 1.0
        mixR = min(1.0, max(0.0, MIX_BASE + MIX_GAIN * (1.0 - Hn)))

        # (3) graf de co-aparitii ponderat temporal; centralitate prin propagare damped
        A = [[0.0] * P for _ in range(P)]
        for i in range(D):
            w = ws[i]
            idx = []
            for n in draws[i]:
                j = int(n) - 1
                if 0 <= j < P:
                    idx.append(j)
            L = len(idx)
            if L < 2:
                continue
            for a in range(L - 1):
                ia = idx[a]
                rowa = A[ia]
                for b in range(a + 1, L):
                    ib = idx[b]
                    rowa[ib] += w
                    A[ib][ia] += w
        Am = np.array(A, dtype=float)
        rs = Am.sum(axis=1, keepdims=True)
        An = Am / np.maximum(rs, 1e-9)
        x0 = np.array([0.5 * (1.0 + half(0.5 * zR[j])) for j in range(P)])
        x = x0.copy()
        for _ in range(PAIR_ITERS):
            x = (1.0 - PAIR_DAMP) * x0 + PAIR_DAMP * An.dot(x)
        xm = float(x.mean())
        xs = float(x.std())
        if xs < 1e-12:
            xs = 1.0
        tC = [half(0.5 * (float(x[j]) - xm) / xs) for j in range(P)]

        # (4) faza ciclica a intarzierii: cos(2*pi*gap/interval), gap plafonat
        iv = P / float(Wn)
        cap = GAP_CAP_MULT * iv
        cyc = []
        for n in numbers:
            ls = last_seen.get(n)
            gap = draw_index if ls is None else draw_index - int(ls)
            if gap < 0:
                gap = 0
            elif gap > cap:
                gap = cap
            cyc.append(math.cos(2.0 * math.pi * (gap / iv)))

        # (5) combinare multiplicativa (interactii conjunctive), tie-break determinist
        scores = {}
        for j in range(P):
            n = numbers[j]
            f = ((1.0 + WL * tL[j])
                 * (1.0 + WR * mixR * half(0.5 * zR[j]))
                 * (1.0 + WC * tC[j])
                 * (1.0 + WP * cyc[j]))
            scores[n] = f
        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank