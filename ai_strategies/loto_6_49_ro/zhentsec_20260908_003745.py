import math

NAME = "zhentsec"
DESCRIPTION = ("Reversie conditionata de entropie: deviatiile standardizate ale ferestrei recente "
               "penalizeaza numerele supra-reprezentate (anti-momentum), cu doza scalata de "
               "entropia Shannon a ferestrei; multiplicator de acoperire pe 3 megasectoare "
               "standardizat hipergeometric si bump saturant de ciclu de intarziere, centrat.")
SOURCE = "model + 2026-09-07"

def create(config):
    N = config.number_range
    K = config.numbers_drawn
    p = K / float(N)
    exp_gap = 1.0 / p
    bounds = [(1, 16), (17, 32), (33, N)]
    sizes = [hi - lo + 1 for (lo, hi) in bounds]
    per_draw_mean = [K * s / float(N) for s in sizes]
    per_draw_var = [K * (s / float(N)) * (1.0 - s / float(N)) * ((N - K) / float(N - 1)) for s in sizes]

    def sector_of(n):
        for i in range(len(bounds) - 1):
            if n <= bounds[i][1]:
                return i
        return len(bounds) - 1

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        W = len(recent_deque)
        if W <= 0:
            base = {n: counts.get(n, 0) for n in numbers}
            return sorted(numbers, key=lambda n: (-base[n], n))
        r = dict.fromkeys(numbers, 0)
        for s in recent_deque:
            for n in s:
                if 1 <= n <= N:
                    r[n] += 1
        e = W * p
        denom = math.sqrt(e * (1.0 - p))
        if denom <= 0:
            denom = 1.0
        total = float(sum(r.values()))
        H = 0.0
        if total > 0:
            for n in numbers:
                if r[n] > 0:
                    q = r[n] / total
                    H -= q * math.log(q)
        Hn = H / math.log(N)
        if Hn < 0.0:
            Hn = 0.0
        elif Hn > 1.0:
            Hn = 1.0
        conc = 1.0 - Hn
        z_s = []
        for i in range(len(bounds)):
            exp_i = per_draw_mean[i] * W
            lo, hi = bounds[i]
            act_i = 0.0
            for n in range(lo, hi + 1):
                act_i += r[n]
            sd = math.sqrt(per_draw_var[i] * W)
            z_s.append((exp_i - act_i) / sd if sd > 0 else 0.0)
        a_rev = 0.35 + 0.55 * Hn
        a_stick = 0.15 * conc
        a_gap = 0.06
        scores = {}
        for n in numbers:
            z = (r[n] - e) / denom
            base = -math.tanh(z) * a_rev + math.tanh(0.5 * z) * a_stick
            i = sector_of(n)
            mult = 1.0 + 0.18 * math.tanh(z_s[i])
            if n in last_seen:
                gap = draw_index - last_seen[n]
            else:
                gap = draw_index + 1
            if gap < 0:
                gap = 0
            gnorm = gap / exp_gap
            g = gnorm / (1.0 + gnorm)
            scores[n] = (base + a_gap * (g - 0.5)) * mult
        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank