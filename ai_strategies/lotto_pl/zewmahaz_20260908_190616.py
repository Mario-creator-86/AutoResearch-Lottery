import math

NAME = "zewmahaz"
DESCRIPTION = 'Scor pe 4 semnale combinate neliniar: z-score al frecventei globale, EWMA decadata pe fereastra recenta (moment), hazard logaritmic de intarziere raportat la media geometrica (N-K)/K si deficitul ponderat al sectorului (7 sectoare x 7 numere), cu o poarta de entropie care intensifica rebalansarea pe sectoare cand distributia recenta e concentrata.'
SOURCE = 'glm-5.3-flash quant plugin, 2026-09-08'


def create(config):
    N = config.number_range
    K = config.numbers_drawn
    numbers = list(range(1, N + 1))

    GAMMA = 0.97
    W_Z = 0.90
    W_EW = 0.70
    W_HAZ = 0.35
    W_SEC = 0.30
    SEC = 7
    RCAP = 6.0
    KAPPA = 0.25
    GATE = 0.50

    NS = (N + SEC - 1) // SEC
    sec_of = [0] * (N + 1)
    sec_size = [0] * NS
    for n in numbers:
        s = (n - 1) // SEC
        sec_of[n] = s
        sec_size[s] += 1
    MU = (N - K) / K

    st = {'sig': None, 'order': None}
    pw = {'L': -1, 'p': None}

    def _powers(L):
        if pw['L'] != L:
            p = [1.0] * L
            for k in range(1, L):
                p[k] = p[k - 1] * GAMMA
            pw['L'] = L
            pw['p'] = p
        return pw['p']

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        L = len(recent_deque)
        d0 = recent_deque[0] if L else None
        dL = recent_deque[-1] if L else None
        sig = (draw_index, L, frozenset(d0) if d0 is not None else None, frozenset(dL) if dL is not None else None)
        if sig == st['sig'] and st['order'] is not None:
            return list(st['order'])

        p = _powers(L)
        cnt = [counts.get(n, 0) for n in numbers]
        mc = sum(cnt) / N
        var = sum((v - mc) * (v - mc) for v in cnt) / N
        sd = math.sqrt(var) or 1.0

        ew = [0.0] * N
        sew = [0.0] * NS
        wsum = 0.0
        pos = 0
        for d in recent_deque:
            w = p[L - 1 - pos]
            wsum += w
            for n in d:
                j = n - 1
                if 0 <= j < N:
                    ew[j] += w
                    sew[sec_of[n]] += w
            pos += 1

        if wsum > 0.0:
            ev = [ew[j] / wsum for j in range(N)]
            me = sum(ev) / N
            ve = sum((v - me) * (v - me) for v in ev) / N
            se = math.sqrt(ve) or 1.0
            ps = [sv / wsum for sv in sew if sv > 0.0]
            if ps:
                h = -sum(q * math.log(q) for q in ps)
                hn = h / math.log(NS) if NS > 1 else 1.0
                if hn < 0.0:
                    hn = 0.0
                elif hn > 1.0:
                    hn = 1.0
            else:
                hn = 1.0
            gate = 1.0 + GATE * (1.0 - hn)
            defz = [0.0] * NS
            for s in range(NS):
                exp_s = wsum * K * sec_size[s] / N
                defz[s] = (exp_s - sew[s]) / math.sqrt(exp_s + 1.0)
        else:
            ev = [0.0] * N
            me = 0.0
            se = 1.0
            gate = 1.0
            defz = [0.0] * NS

        cap = RCAP * MU
        sc = [0.0] * N
        for j in range(N):
            n = j + 1
            ls = last_seen.get(n)
            gap = draw_index + 1 if ls is None else draw_index - ls
            if gap < 0:
                gap = 0
            elif gap > cap:
                gap = cap
            haz = math.log1p(gap / MU)
            z = (cnt[j] - mc) / sd
            ze = (ev[j] - me) / se
            sc[j] = W_Z * z + W_EW * ze * (1.0 + KAPPA * math.tanh(z)) + W_HAZ * haz + W_SEC * gate * defz[sec_of[n]]

        order = sorted(numbers, key=lambda t: (-sc[t - 1], t))
        st['sig'] = sig
        st['order'] = order
        return order

    return rank