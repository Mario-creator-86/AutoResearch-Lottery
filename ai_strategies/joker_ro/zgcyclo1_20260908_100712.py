import math
from collections import defaultdict

NAME = "zgcyclo1"
DESCRIPTION = "Gap-cycle renewal model: estimates an empirical gap distribution (mean/std) per number from the recent window, shrunk toward the global 1/p baseline, and scores each number with a Gaussian echo bump centered at its mean gap plus a per-number normalized overdue term (z-gap); blends window momentum, long-run frequency z and a first-order Markov affinity lift from the last draw's numbers."
SOURCE = "quant research / 2026-09-08"

def create(config):
    P = int(config.number_range)
    K = int(config.numbers_drawn)
    p0 = (K / P) if P > 0 else 0.0

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, P + 1))
        D = max(int(draw_index), 1)
        w = len(recent_deque)
        if w == 0 or p0 <= 0.0:
            base = {n: counts.get(n, 0) for n in numbers}
            return sorted(numbers, key=lambda n: (-base[n], n))

        draws = [set(s) for s in recent_deque]
        last_draw = draws[-1]
        # orient: verify deque[-1] is the most recent draw via last_seen
        reversed_order = False
        for n in sorted(last_draw):
            ls = last_seen.get(n)
            if ls is None:
                continue
            if ls == draw_index - w:
                reversed_order = True
            break
        if reversed_order:
            draws.reverse()
            last_draw = draws[-1]

        # window appearance counts
        wc = defaultdict(int)
        for S in draws:
            for n in S:
                wc[n] += 1

        # per-number gap statistics within the window (oldest -> newest)
        g_sum = defaultdict(float)
        g_sq = defaultdict(float)
        g_cnt = defaultdict(int)
        lastpos = {}
        for i, S in enumerate(draws):
            for n in S:
                lp = lastpos.get(n)
                if lp is not None:
                    g = i - lp
                    g_sum[n] += g
                    g_sq[n] += g * g
                    g_cnt[n] += 1
                lastpos[n] = i

        # first-order Markov affinity: co-occurrence with last draw's anchors
        anchors = sorted(last_draw)
        co = defaultdict(int)
        if anchors:
            for S in draws:
                inter = S & last_draw
                if not inter:
                    continue
                for m in inter:
                    for n in S:
                        co[(m, n)] += 1

        mu_g = 1.0 / p0
        sig_g = mu_g * 0.85
        den_lt = math.sqrt(D * p0 * (1.0 - p0)) or 1.0
        den_r = math.sqrt(w * p0 * (1.0 - p0)) or 1.0

        scored = []
        for n in numbers:
            c = counts.get(n, 0)
            z_lt = (c - D * p0) / den_lt
            z_r = (wc.get(n, 0) - w * p0) / den_r
            mom = math.tanh(z_r / 2.0)

            cnt = g_cnt.get(n, 0)
            if cnt > 0:
                mu_w = g_sum[n] / cnt
                var = g_sq[n] / cnt - mu_w * mu_w
                sig_w = math.sqrt(var) if var > 0.0 else sig_g
                kk = 3.0
                mu = (cnt * mu_w + kk * mu_g) / (cnt + kk)
                sig = (cnt * sig_w + kk * sig_g) / (cnt + kk)
            else:
                mu = mu_g
                sig = sig_g
            mu = max(mu, 1.5)
            sig = max(sig, 1.0)

            ls = last_seen.get(n)
            t = (draw_index - 1 - ls) if ls is not None else D
            if t < 0:
                t = 0
            zgap = (t - mu) / sig
            echo = math.exp(-0.5 * zgap * zgap)
            due = math.tanh(zgap / 3.0)

            aff = 0.0
            if anchors:
                pn = min(max(c / D, 0.02), 0.6)
                acc = 0.0
                for m in anchors:
                    pm = min(max(counts.get(m, 0) / D, 0.02), 0.6)
                    ratio = (co.get((m, n), 0) + 1.0) / (w * pn * pm + 1.0)
                    acc += math.log(ratio)
                aff = math.tanh((acc / len(anchors)) / 1.2)

            score = (0.50 * echo + 0.25 * due + 0.30 * mom
                     + 0.20 * (z_lt / 3.0) + 0.35 * aff)
            scored.append((n, score))

        scored.sort(key=lambda x: (-x[1], x[0]))
        return [n for n, _ in scored]

    return rank