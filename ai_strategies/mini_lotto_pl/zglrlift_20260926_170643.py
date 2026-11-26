import math

NAME = "zglrlift"
DESCRIPTION = "Log-lift entropic (PMI pe numar: rata din fereastra recenta / rata globala) combinat multiplicativ cu z-scorul deviatiei de la frecventa asteptata si cu hazard-ul de gap normalizat Poisson fata de rata proprie; interactiune non-liniara intre semnale."
SOURCE = "model + 2026-09-26"


def create(config):
    N = config.number_range
    K = config.numbers_drawn

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        p_base = K / float(N)
        T = sum(counts.values()) / float(K) if counts else float(draw_index)
        if T <= 0.0:
            T = 1.0
        W = len(recent_deque)

        # aparitii in fereastra recenta (independent de orientarea deque-ului)
        r = dict.fromkeys(numbers, 0)
        for s in recent_deque:
            for n in s:
                if n in r:
                    r[n] += 1

        exp_freq = T * p_base
        sd_freq = math.sqrt(max(T * p_base * (1.0 - p_base), 1e-9))

        scores = {}
        for n in numbers:
            c = counts.get(n, 0)
            # rata globala cu shrinkage spre uniform
            p0 = (c + 2.0) / (T + 2.0 * N)
            p0 = min(max(p0, 1e-6), 1.0 - 1e-6)

            # 1) log-lift entropic semnat pe fereastra recenta (PMI)
            if W > 0:
                p1 = (r[n] + 0.5) / (W + 1.0)
                lift = min(max(p1 / p0, 1e-6), 1e6)
                s1 = math.tanh(math.log(lift) / 1.5)
            else:
                s1 = 0.0

            # 2) deviatie fata de frecventa asteptata (z-scor binomial global)
            s2 = math.tanh(((c - exp_freq) / sd_freq) / 2.0)

            # 3) hazard de gap: gap-ul curent vs propria rata (proces de reinnoire)
            if n in last_seen:
                gap = draw_index - last_seen[n]
                mu = T / c if c > 0 else float(T)
            else:
                gap = draw_index + 1
                mu = float(T)
            gz = (gap - mu) / math.sqrt(max(mu, 1e-9))
            s3 = math.tanh(gz / 6.0)

            # combinare multiplicativa (toti factorii > 0)
            scores[n] = (1.0 + 1.00 * s1) * (1.0 + 0.35 * s2) * (1.0 + 0.60 * s3)

        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank