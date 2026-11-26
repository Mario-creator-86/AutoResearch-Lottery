import math
from collections import defaultdict

NAME = "zpairfol"
DESCRIPTION = ('Rezonanta Markov de tip follow: puncteaza numerele care au urmat istoric '
               'numerele din ultima extragere (tranzitii lag-1 ponderate recent + perechi '
               'co-aparute semi-ponderate), combinat neliniar prin entropy-gating cu '
               'z-score de frecventa globala (momentum) si hazard de intarziere (overdue).')
SOURCE = 'quant-model 2026-09-17'


def create(config):
    NR = config.number_range
    ND = config.numbers_drawn
    P = ND / float(NR)      # probabilitatea de includere a unui numar intr-o extragere
    DECAY = 0.97            # ponderare exponentiala a tranzitiilor recente
    TRANS_WIN = 40          # fereastra maxima pentru structura de tranzitii
    CO_W = 0.5              # ponderea perechilor simetrice fata de tranzitii directionale
    W_HOT = 0.55            # pondere de baza momentum (frecventa globala)
    W_GAP = 0.80            # pondere de baza overdue
    W_RES = 1.10            # pondere de baza rezonanta follow
    ZCAP = 3.0

    def _zscore(d):
        vals = list(d.values())
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        sd = math.sqrt(var)
        if sd < 1e-12:
            return {k: 0.0 for k in d}
        return {k: (v - m) / sd for k, v in d.items()}

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, NR + 1))

        # 1) momentum: z-score fata de frecventa asteptata N*p
        n_total = (sum(counts.values()) / float(ND)) if counts else 0.0
        if n_total > 1.0:
            sd_hot = math.sqrt(n_total * P * (1.0 - P))
            hotz = {n: (counts.get(n, 0) - n_total * P) / sd_hot for n in numbers}
        else:
            hotz = {n: 0.0 for n in numbers}

        # 2) overdue: hazard pe intarziere fata de gap-ul asteptat 1/p
        mu = 1.0 / P
        sg = math.sqrt(1.0 - P) / P
        gapz = {}
        for n in numbers:
            ls = last_seen.get(n)
            gap = (draw_index - ls) if ls is not None else (draw_index + 1)
            gapz[n] = max(-ZCAP, min(ZCAP, (gap - mu) / sg))

        # orientarea deque-ului: detectam capatul 'newest' folosind last_seen
        seq = list(recent_deque)
        prev_set = set(n for n in numbers if last_seen.get(n) == draw_index - 1)
        if len(seq) >= 2 and prev_set and seq[0] == prev_set:
            seq = seq[::-1]
        win = seq[-TRANS_WIN:]
        T = len(win)

        # 3) rezonanta follow: tranzitii lag-1 + perechi co-aparute, ponderate recent
        trans = defaultdict(float)
        pairw = defaultdict(float)
        for i in range(T):
            w = DECAY ** (T - 1 - i)
            ds = sorted(win[i])
            for a in range(len(ds)):
                for b in range(a + 1, len(ds)):
                    pairw[(ds[a], ds[b])] += w
            if i > 0:
                for m in win[i - 1]:
                    for n in win[i]:
                        if n != m:
                            trans[(m, n)] += w
        last_set = prev_set if prev_set else (set(win[-1]) if win else set())
        res_raw = {}
        for n in numbers:
            s = 0.0
            for m in last_set:
                if m == n:
                    continue
                s += trans.get((m, n), 0.0)
                key = (n, m) if n < m else (m, n)
                s += CO_W * pairw.get(key, 0.0)
            res_raw[n] = s

        # 4) entropia Shannon normalizata a ferestrei recente -> gating intre follow si overdue
        if T >= 5:
            c = defaultdict(int)
            for d in win:
                for n in d:
                    c[n] += 1
            tot = float(ND * T)
            H = 0.0
            for n in numbers:
                q = c.get(n, 0) / tot
                if q > 0.0:
                    H -= q * math.log(q)
            Hn = max(0.0, min(1.0, H / math.log(NR)))
        else:
            Hn = 0.5

        hz = _zscore(hotz)
        gz = _zscore(gapz)
        rz = _zscore(res_raw)
        w_res = W_RES * (0.30 + 0.70 * (1.0 - Hn))
        w_gap = W_GAP * (0.30 + 0.70 * Hn)
        scores = {n: W_HOT * hz[n] + w_gap * gz[n] + w_res * rz[n] for n in numbers}
        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank