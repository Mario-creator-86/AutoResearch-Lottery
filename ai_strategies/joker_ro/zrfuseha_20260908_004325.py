import math

NAME = "zrfuseha"
DESCRIPTION = 'Rank-fusion ensemble: long-run frequency z-score, log-normal renewal-hazard phase peaking near 1.42x mean gap, and recent-window momentum; weights adapt to window overdispersion, plus saturating pair-affinity lift.'
SOURCE = 'model + 2026-09-07'

def create(config):
    P_TOTAL = config.number_range
    K = config.numbers_drawn
    p_hit = K / float(P_TOTAL)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, P_TOTAL + 1))
        N = draw_index if (draw_index and draw_index > 0) else 0

        # component 1: long-run frequency z-score
        mu = N * p_hit
        sd = math.sqrt(max(mu * (1.0 - p_hit), 1e-12))
        z = {}
        for n in numbers:
            v = (counts.get(n, 0) - mu) / sd
            z[n] = -3.0 if v < -3.0 else (3.0 if v > 3.0 else v)

        # recent window stats (order-independent: deque NOT assumed LIFO/FIFO)
        W = len(recent_deque)
        r = dict.fromkeys(numbers, 0)
        pair = {}
        for S in recent_deque:
            for n in S:
                r[n] += 1
            sl = sorted(S)
            for i in range(len(sl) - 1):
                a = sl[i]
                for j in range(i + 1, len(sl)):
                    key = (a, sl[j])
                    pair[key] = pair.get(key, 0) + 1

        # component 3: recency z-score + overdispersion gate
        if W > 0:
            mu_r = W * p_hit
            sd_r = math.sqrt(max(mu_r * (1.0 - p_hit), 1e-12))
            zr = {}
            for n in numbers:
                v = (r[n] - mu_r) / sd_r
                zr[n] = -3.0 if v < -3.0 else (3.0 if v > 3.0 else v)
            mean_r = sum(r.values()) / float(P_TOTAL)
            var_r = 0.0
            for n in numbers:
                var_r += (r[n] - mean_r) ** 2
            var_r /= float(P_TOTAL)
            cv = math.sqrt(var_r) / max(mean_r, 1e-12)
            cv_th = math.sqrt((1.0 - p_hit) / max(W * p_hit, 1e-12))
            R = cv / max(cv_th, 1e-12)
        else:
            zr = dict.fromkeys(numbers, 0.0)
            R = 1.0
        t = (R - 0.5) / 1.5
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0

        # component 2: renewal hazard, log-normal kernel on gap ratio h = gap / mean_gap
        # peak at h = exp(0.35) ~ 1.42, soft saturation before h = 3
        hz = {}
        for n in numbers:
            c = counts.get(n, 0)
            if c <= 0:
                hz[n] = 0.0
                continue
            m = (N + 1.0) / (c + 1.0)
            g = N - last_seen.get(n, 0)
            h = (g / m) if m > 0 else 0.0
            if h <= 0.0:
                hz[n] = 0.0
            else:
                d = (math.log(h) - 0.35) / 0.742
                hz[n] = math.exp(-0.5 * d * d)

        def rank01(dd):
            order = sorted(numbers, key=lambda n: (dd[n], n))
            out = {}
            M = float(len(numbers) - 1)
            for i, n in enumerate(order):
                out[n] = i / M
            return out

        rz = rank01(z)
        rh = rank01(hz)
        rr = rank01(zr)

        # adaptive weights: clustered window -> more recency; diffuse -> more long-run + hazard
        w_r = 0.15 + 0.20 * t
        w_h = 0.30 + 0.10 * (1.0 - t)
        w_z = 1.0 - w_r - w_h

        # pair-affinity: co-occurrences weighted by partner popularity, own-frequency normalized
        A = dict.fromkeys(numbers, 0.0)
        for (i, j), cnt in pair.items():
            A[i] += cnt * r[j]
            A[j] += cnt * r[i]
        Pp = {}
        for n in numbers:
            Pp[n] = A[n] / (1.0 + r[n])
        mx = max(Pp.values())
        lift_amp = 0.25 + 0.20 * t

        scores = {}
        for n in numbers:
            base = w_z * rz[n] + w_h * rh[n] + w_r * rr[n]
            if mx > 0:
                base *= (1.0 + lift_amp * Pp[n] / mx)
            scores[n] = base

        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank