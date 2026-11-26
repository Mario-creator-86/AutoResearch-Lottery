import math

NAME = "zphasecy"
DESCRIPTION = 'Presiune de faza adaptiva: gap normalizat per numar cu interval propriu shrink-uit catre media teoretica, bonus pentru sectoarele sub-reprezentine in fereastra recenta si momentum conditionat de entropia distributiei recente.'
SOURCE = 'model + 2026-09-26'


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    base_interval = N / float(K)
    sec_size = max(1, N // 7)
    n_sectors = (N + sec_size - 1) // sec_size

    W_PHASE = 1.0
    W_SEC = 0.45
    W_ENT = 0.55
    SHRINK_K = 8.0
    I_MIN = 3.0
    I_MAX = 30.0

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))

        sector_obs = [0] * n_sectors
        num_recent = {}
        total_recent = 0
        if recent_window and recent_deque:
            for s in recent_deque:
                for x in s:
                    if 1 <= x <= N:
                        k = (x - 1) // sec_size
                        if k >= n_sectors:
                            k = n_sectors - 1
                        sector_obs[k] += 1
                        num_recent[x] = num_recent.get(x, 0) + 1
                        total_recent += 1

        ent_gate = 0.0
        if total_recent > 0:
            H = 0.0
            for xn in sorted(num_recent):
                p = num_recent[xn] / float(total_recent)
                H -= p * math.log(p)
            u = H / math.log(N) if N > 1 else 0.0
            ent_gate = 1.0 - u
            if ent_gate < 0.0:
                ent_gate = 0.0

        exp_sec = sec_size * total_recent / float(N) if total_recent > 0 else 0.0
        sec_score = [0.0] * n_sectors
        if exp_sec > 0.0:
            sd = math.sqrt(exp_sec)
            for k in range(n_sectors):
                z = (sector_obs[k] - exp_sec) / sd
                if z > 2.0:
                    z = 2.0
                elif z < -2.0:
                    z = -2.0
                sec_score[k] = -z / 2.0

        mean_r = total_recent / float(N) if total_recent > 0 else 0.0

        scores = {}
        for n in numbers:
            c = counts.get(n, 0)
            if c > 0 and draw_index > 0:
                raw = draw_index / float(c)
                interval = (c * raw + SHRINK_K * base_interval) / (c + SHRINK_K)
            else:
                interval = base_interval
            if interval < I_MIN:
                interval = I_MIN
            elif interval > I_MAX:
                interval = I_MAX

            ls = last_seen.get(n)
            if ls is None:
                elapsed = draw_index + 1
            else:
                elapsed = draw_index - ls
            if elapsed < 0:
                elapsed = 0

            phi = elapsed / (0.8 * interval)
            p_press = 1.0 - math.exp(-phi)

            k = (n - 1) // sec_size
            s_sec = sec_score[k] if k < n_sectors else 0.0

            e_sc = 0.0
            if ent_gate > 0.0 and mean_r > 0.0:
                r = num_recent.get(n, 0)
                rz = (r - mean_r) / math.sqrt(mean_r)
                if rz > 2.0:
                    rz = 2.0
                elif rz < -2.0:
                    rz = -2.0
                e_sc = ent_gate * (rz / 2.0)

            scores[n] = W_PHASE * p_press + W_SEC * s_sec + W_ENT * e_sc

        return sorted(numbers, key=lambda m: (-scores[m], m))

    return rank