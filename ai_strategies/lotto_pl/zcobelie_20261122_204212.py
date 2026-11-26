NAME = "zcobelie"
DESCRIPTION = ('Ierarhie 1..49 construita greedy (submodular) sub un model de credinta conditionat: '
               'marjinale shrink-uite global+fereastra, boost conditionat din afinitatile tanh-z ale '
               'perechilor din fereastra recenta si modelare multiplicativa a sumei biletului.')
SOURCE = 'greedy conditional-belief quant model, 2026-11-12'

from math import tanh, exp

DEFAULTS = {
    'KAPPA_G': 2500.0,
    'KAPPA_W': 8.0,
    'BETA': 0.35,
    'THETA': 0.75,
    'SUM_LAM': 0.35,
    'ECHO_W': 0.0,
    'ZSCALE': 3.0,
}


def create(config, params=None):
    P = dict(DEFAULTS)
    if params:
        P.update(params)
    NR = int(config.number_range)
    ND = int(config.numbers_drawn)
    TS = int(getattr(config, 'ticket_size', 0) or ND) or ND
    pbar = ND / float(NR)
    s_star = TS * (NR + 1) / 2.0
    p_pair = ND * (ND - 1.0) / (NR * (NR - 1.0))
    theta = float(P['THETA'])
    slam = float(P['SUM_LAM'])
    echo = float(P['ECHO_W'])
    beta = float(P['BETA'])
    kg = float(P['KAPPA_G'])
    kw = float(P['KAPPA_W'])
    zsc = float(P['ZSCALE'])

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, NR + 1))
        draws = list(recent_deque)
        W = len(draws)

        # membri in fereastra + perechi co-aparute (canonice min<max;
        # doar apartenenta => invariant la orientarea deque-ului)
        cw = [0] * (NR + 1)
        pairc = {}
        for s in draws:
            L = sorted(s)
            for n in L:
                if 1 <= n <= NR:
                    cw[n] += 1
            for i in range(len(L) - 1):
                a = L[i]
                for j in range(i + 1, len(L)):
                    k = (a, L[j])
                    pairc[k] = pairc.get(k, 0) + 1

        # marjinale shrink-uite: global + fereastra
        T = 0.0
        for n in numbers:
            T += counts.get(n, 0)
        qv = [0.0] * (NR + 1)
        for n in numbers:
            cg = counts.get(n, 0)
            qg = (cg + kg * pbar) / (T + kg) if T > 0 else pbar
            qw = (cw[n] + kw * pbar) / (W + kw) if W > 0 else pbar
            qv[n] = (1.0 - beta) * qg + beta * qw

        # echo de recency pe marjinal (implicit OPRIT -> flip-safe total)
        if echo > 0.0:
            tau = max(2.0, (recent_window or 40) * 0.05)
            horizon = draw_index + tau * 8.0
            for n in numbers:
                ls = last_seen.get(n, -1)
                gap = (draw_index - ls) if ls >= 0 else horizon
                qv[n] *= 1.0 + echo * exp(-gap / tau)

        # matrice de afinitate tanh(z Poisson) pe perechi; E pe pereche specifica = W*p_pair
        expc = W * p_pair
        sig = (expc if expc > 0.5 else 0.5) ** 0.5
        M = [[0.0] * (NR + 1) for _ in range(NR + 1)]
        for k, c in pairc.items():
            a, b = k
            if 1 <= a <= NR and 1 <= b <= NR:
                aff = tanh(((c - expc) / sig) / zsc)
                M[a][b] = aff
                M[b][a] = aff

        # greedy conditionat pe tot spatiul => permutare completa
        remaining = list(numbers)
        sums = [0.0] * (NR + 1)
        partial = 0.0
        half = NR / 2.0
        order = []
        while remaining:
            kp = len(order)
            btarget = None
            if slam > 0.0 and kp < TS:
                btarget = (s_star - partial) / (TS - kp)
            best_n = None
            best_g = None
            for n in remaining:
                g = qv[n]
                if kp:
                    cond = 1.0 + theta * (sums[n] / kp)
                    if cond < 0.05:
                        cond = 0.05
                    g *= cond
                if btarget is not None:
                    rel = (n - btarget) / half
                    g *= exp(-slam * rel * rel)
                if best_g is None or g > best_g:
                    best_g = g
                    best_n = n
            remaining.remove(best_n)
            order.append(best_n)
            partial += best_n
            row = M[best_n]
            for m in remaining:
                sums[m] += row[m]
        return order

    return rank