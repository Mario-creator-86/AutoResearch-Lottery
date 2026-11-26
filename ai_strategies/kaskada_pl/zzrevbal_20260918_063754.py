import math

NAME = "zzrevbal"
DESCRIPTION = 'Reversion compusa anti-deviatie: z-score negativ pe frecventa long-run si recenta, deficit de gap geometric (numere intarziate, cap 12) si reechilibrare pe sectoare (4 sferturi + paritate) in fereastra recenta. Determinist, O(P+fereastra), auto-calibrat pe media empirica.'
SOURCE = 'glm-quant 2026-09-18'

def create(config):
    P = int(getattr(config, 'number_range', 24))
    W = int(getattr(config, 'numbers_drawn', 12))
    p = float(W) / float(P)
    q = 1.0 - p

    WZ = 1.00
    WR = 0.35
    WG = 0.50
    WQ = 0.60
    WP = 0.30
    GAP_CAP = 12.0
    NQ = 4

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, P + 1))

        total_all = 0
        for n in numbers:
            total_all += counts.get(n, 0)
        mu_g = total_all / float(P)
        sd_g = math.sqrt(max(mu_g * q, 1e-12))
        use_g = mu_g > 5.0

        recent_sets = list(recent_deque)
        R = len(recent_sets)
        rc = dict.fromkeys(numbers, 0)
        total_r = 0
        for st in recent_sets:
            for n in st:
                if 1 <= n <= P:
                    rc[n] += 1
                    total_r += 1
        mu_r = (total_r / float(P)) if R > 0 else 0.0
        sd_r = math.sqrt(max(mu_r * q, 1e-12))
        use_r = (R >= 5) and (mu_r > 2.0)

        sec_id = {}
        sec_size = [0] * NQ
        for n in numbers:
            k = (n - 1) * NQ // P
            sec_id[n] = k
            sec_size[k] += 1

        z_pe = 0.0
        sec_cnt = [0] * NQ
        exp_q = [0.0] * NQ
        sd_q = [0.0] * NQ
        if use_r:
            for n in numbers:
                sec_cnt[sec_id[n]] += rc[n]
            for k in range(NQ):
                exp_q[k] = sec_size[k] * mu_r
                sd_q[k] = math.sqrt(max(exp_q[k] * q, 1e-12))
            n_even = 0
            for n in numbers:
                if n % 2 == 0:
                    n_even += 1
            exp_pe = n_even * mu_r
            sd_pe = math.sqrt(max(exp_pe * q, 1e-12))
            even_cnt = 0
            for n in numbers:
                if n % 2 == 0:
                    even_cnt += rc[n]
            if sd_pe > 1e-9:
                z_pe = (even_cnt - exp_pe) / sd_pe

        scores = {}
        for n in numbers:
            s = 0.0
            if use_g:
                s -= WZ * (counts.get(n, 0) - mu_g) / sd_g
            if use_r:
                s -= WR * (rc[n] - mu_r) / sd_r
            if n in last_seen:
                gap = float(draw_index - last_seen[n])
            else:
                gap = float(draw_index) + 1.0
            if gap < 0.0:
                gap = 0.0
            elif gap > GAP_CAP:
                gap = GAP_CAP
            s += WG * gap * p
            if use_r:
                k = sec_id[n]
                if sd_q[k] > 1e-9:
                    s -= WQ * (sec_cnt[k] - exp_q[k]) / sd_q[k]
                if n % 2 == 0:
                    s -= WP * z_pe
                else:
                    s += WP * z_pe
            scores[n] = s

        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank