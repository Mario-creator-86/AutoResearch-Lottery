import math

NAME = "zhzmsg7"
DESCRIPTION = 'Hazard-EWMA cu message-passing pe graf de co-occurinta: z-score termen lung, EWMA duala (half-life 8/45), hazard de intarziere vs gap mediu 49/6, bonus de repetitie pe ultimele 3 extrageri si un pas de propagare de mesaje peste perechi co-aparute ponderate de recency; termenii recenti sunt modulati de entropia ferestrei si combinati non-liniar (interactiune lt x fast).'
SOURCE = 'model + 2026-09-08'


def create(config):
    N = config.number_range
    K = config.numbers_drawn

    H_FAST = 8.0
    H_SLOW = 45.0
    H_PAIR = 20.0
    PAIR_CAP = 3.0
    W_LT = 1.00
    W_FAST = 1.25
    W_SLOW = 0.70
    W_MSG = 0.90
    W_OD = 0.35
    W_REP = 0.30
    W_INTER = 0.25

    _cache = {'key': None, 'scores': {}}

    def _compute(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        tot = 0
        for n in numbers:
            tot += counts.get(n, 0)
        D = max(1.0, tot / float(K))
        p = K / float(N)
        mean_lt = D * p
        sd_lt = max(1e-9, math.sqrt(D * p * (1.0 - p)))

        draws = list(recent_deque)
        L = len(draws)

        wf = [0.0] * (N + 1)
        ws = [0.0] * (N + 1)
        twf = 0.0
        tws = 0.0

        wp = [None] * (N + 1)
        for n in numbers:
            wp[n] = [0.0] * (N + 1)

        lf = 0.5 ** (1.0 / H_FAST)
        ls = 0.5 ** (1.0 / H_SLOW)
        lp = 0.5 ** (1.0 / H_PAIR)

        for i in range(L):
            age = L - 1 - i
            ef = lf ** age
            es = ls ** age
            ep = lp ** age
            twf += ef
            tws += es
            sn = list(draws[i])
            for n in sn:
                wf[n] += ef
                ws[n] += es
            m = len(sn)
            for a in range(m - 1):
                na = sn[a]
                rowa = wp[na]
                for b in range(a + 1, m):
                    nb = sn[b]
                    va = rowa[nb] + ep
                    rowa[nb] = va if va < PAIR_CAP else PAIR_CAP
                    vb = wp[nb][na] + ep
                    wp[nb][na] = vb if vb < PAIR_CAP else PAIR_CAP

        rowsum = [0.0] * (N + 1)
        for n in numbers:
            row = wp[n]
            s0 = 0.0
            for m2 in range(1, N + 1):
                s0 += row[m2]
            rowsum[n] = s0

        exp_f = twf * p
        exp_s = tws * p
        zf = [0.0] * (N + 1)
        zs = [0.0] * (N + 1)
        base = [0.0] * (N + 1)
        bmean = 0.0
        bsd = 1.0
        if twf > 1e-9:
            sf = math.sqrt(exp_f) + 1.0
            ss = math.sqrt(exp_s) + 1.0
            acc = 0.0
            for n in numbers:
                zfv = (wf[n] - exp_f) / sf
                zsv = (ws[n] - exp_s) / ss
                zf[n] = zfv
                zs[n] = zsv
                az = zfv if zfv >= 0 else -zfv
                bv = zfv / (1.0 + az)
                base[n] = bv
                acc += bv
            bmean = acc / N
            bvar = 0.0
            for n in numbers:
                d0 = base[n] - bmean
                bvar += d0 * d0
            bsd = max(1e-6, math.sqrt(bvar / N))

        msg = [0.0] * (N + 1)
        for n in numbers:
            den = rowsum[n]
            if den > 1e-9:
                row = wp[n]
                a2 = 0.0
                for m2 in range(1, N + 1):
                    wnm = row[m2]
                    if wnm > 0.0:
                        a2 += wnm * base[m2]
                mv = (a2 / den - bmean) / bsd
                if mv > 2.0:
                    mv = 2.0
                elif mv < -2.0:
                    mv = -2.0
                msg[n] = mv

        mod = 1.4
        if twf > 1e-9:
            denom = twf * K
            H = 0.0
            for n in numbers:
                pn = wf[n] / denom
                if pn > 1e-12:
                    H -= pn * math.log(pn)
            if H > 0.0:
                mod = 0.6 + 0.8 * (H / math.log(N))

        mgap = 1.0 / p
        rep = [0.0] * (N + 1)
        if L >= 1:
            for n in draws[-1]:
                rep[n] += 1.0
        if L >= 2:
            for n in draws[-2]:
                rep[n] += 0.5
        if L >= 3:
            for n in draws[-3]:
                rep[n] += 0.25

        scores = {}
        for n in numbers:
            zlt = (counts.get(n, 0) - mean_lt) / sd_lt
            lt = math.tanh(zlt / 2.0)
            gap = draw_index - last_seen.get(n, -1)
            od = math.tanh((gap - mgap) / (1.5 * mgap))
            recent = W_FAST * zf[n] + W_SLOW * zs[n] + W_MSG * msg[n]
            scores[n] = W_LT * lt + W_OD * od + W_REP * rep[n] + mod * recent + W_INTER * lt * zf[n]
        return scores

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        tot = 0
        for n in range(1, N + 1):
            tot += counts.get(n, 0)
        key = (draw_index, recent_window, len(recent_deque), tot)
        if _cache['key'] != key:
            _cache['key'] = key
            _cache['scores'] = _compute(counts, recent_deque, last_seen, draw_index, recent_window)
        scores = _cache['scores']
        numbers = list(range(1, N + 1))
        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank