import math

NAME = "zmrbalan"
DESCRIPTION = 'Mean-reversion balancer: penalizeaza numerele supra-reprezentate recent (tanh saturat) si istoric (z-score negativ), stimuleaza numerele intarziate (gap log-scalat) si sectoarele sub-reprezentate; termenul anti-hot e modulat de supradirpersiona frecventelor din fereastra.'
SOURCE = 'model + 2026-09-08'


def create(config):
    R = config.number_range            # 24
    K = config.numbers_drawn           # 12
    p = float(K) / float(R)            # probabilitatea de aparitie a unui numar per extragere
    NSEC = 4
    SECSZ = max(R // NSEC, 1)
    W_HOT = 1.0                        # anti-hot recent (mean reversion scurt)
    W_SEC = 0.4                        # rotatie de sector (deficit recent standardizat)
    W_OD = 0.5                         # intarziere (overdue)
    W_Z = 0.35                         # fade al frecventei istorice
    GCAP = 8.0
    LOG_GCAP = math.log(1.0 + GCAP)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, R + 1))
        W = len(recent_deque)
        r = dict.fromkeys(numbers, 0)
        for s in recent_deque:
            for n in s:
                if 1 <= n <= R:
                    r[n] += 1
        e = W * p
        sdw = math.sqrt(max(e * (1.0 - p), 1e-9))
        rs = list(r.values())
        mr = sum(rs) / R
        vr = sum((x - mr) ** 2 for x in rs) / R
        rho = math.sqrt(vr) / sdw      # supradirpersion: fereastra spiky => reversion mai puternica
        if rho > 1.4:
            rho = 1.4
        elif rho < 0.6:
            rho = 0.6
        D = draw_index if draw_index > 0 else 1
        mu = D * p
        sg = math.sqrt(max(D * p * (1.0 - p), 1e-12))
        obs = [0.0] * NSEC
        for n in numbers:
            i = (n - 1) // SECSZ
            if i >= NSEC:
                i = NSEC - 1
            obs[i] += r[n]
        exp_s = SECSZ * p * W
        if exp_s > 0:
            defs = [(exp_s - o) / exp_s for o in obs]
        else:
            defs = [0.0] * NSEC
        md = sum(defs) / NSEC
        vd = sum((d - md) ** 2 for d in defs) / NSEC
        sstd = math.sqrt(vd) + 1e-9
        scores = {}
        for n in numbers:
            rn = r[n]
            hot = -math.tanh((rn - e) / sdw)
            zt = -((counts.get(n, 0) - mu) / sg)
            ls = last_seen.get(n)
            g = (draw_index - ls) if ls is not None else (draw_index + 1)
            if g < 0:
                g = 0
            gg = g if g < GCAP else GCAP
            od = math.log(1.0 + gg) / LOG_GCAP
            i = (n - 1) // SECSZ
            if i >= NSEC:
                i = NSEC - 1
            sz = (defs[i] - md) / sstd
            scores[n] = W_HOT * rho * hot + W_SEC * sz + W_OD * od + W_Z * zt
        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank