import math

NAME = "zregimee"
DESCRIPTION = 'Scorer multi-orizont adaptiv de regim: z-score al frecventei globale, recenta EWMA (half-life 12), presiune de intarziere (gap-hazard), eco de co-ocurenta cu ultima extragere si tilt de deficit pe 9 sectoare; un gate chi-patrat pe dispersia ferestrei recente reajusteaza ponderile momentum vs intarziere.'
SOURCE = 'quant model + 2026-09-08'


def create(config):
    n_max = int(config.number_range)
    k = int(config.numbers_drawn)
    p = k / n_max
    expected_gap = n_max / k
    n_sectors = 9
    span = n_max / n_sectors
    numbers = list(range(1, n_max + 1))
    sec_of = {}
    for n in numbers:
        s = int((n - 1) / span)
        if s >= n_sectors:
            s = n_sectors - 1
        sec_of[n] = s

    half_life = 12.0
    decay = 0.5 ** (1.0 / half_life)
    weight_cache = {}

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        total_hits = 0
        for n in numbers:
            total_hits += counts.get(n, 0)
        draws_done = total_hits / k if k > 0 else 1.0
        if draws_done < 1.0:
            draws_done = 1.0
        mu = draws_done * p
        sigma = math.sqrt(draws_done * p * (1.0 - p)) + 1e-9

        L = len(recent_deque)
        if L not in weight_cache:
            weight_cache[L] = [decay ** (L - 1 - i) for i in range(L)]
        ws = weight_cache[L]

        ema = dict.fromkeys(numbers, 0.0)
        obs = dict.fromkeys(numbers, 0)
        aff = dict.fromkeys(numbers, 0.0)
        sec_obs = [0] * n_sectors

        aff_start = L - 40 if L > 40 else 0
        last_set = recent_deque[-1] if L > 0 else set()
        wsum = 0.0
        i = 0
        for dset in recent_deque:
            w = ws[i]
            wsum += w
            overlap = 0
            if i >= aff_start:
                overlap = len(dset & last_set)
            for n in dset:
                ema[n] += w
                obs[n] += 1
                if overlap:
                    aff[n] += overlap * w
                sec_obs[sec_of[n]] += 1
            i += 1

        e_per = (L * k) / n_max if L > 0 else 1.0
        x2 = 0.0
        for n in numbers:
            d = obs[n] - e_per
            x2 += d * d / e_per
        df = n_max - 1 if n_max > 1 else 1
        ratio = x2 / df
        gate = (ratio - 1.0) / 0.30
        if gate < 0.0:
            gate = 0.0
        elif gate > 1.0:
            gate = 1.0

        w_z = 2.6
        w_rec = 0.8 + 1.5 * gate
        w_gap = 1.2 - 0.9 * gate
        w_echo = 0.5 + 1.1 * gate
        w_sec = 0.6

        exp_ema = wsum * p if wsum > 0 else 1.0
        rec_sd = math.sqrt(exp_ema * (1.0 - p)) + 1e-9
        e_sec = (L * k) / n_sectors if L > 0 else 1.0

        scores = {}
        for n in numbers:
            z = (counts.get(n, 0) - mu) / sigma
            r = (ema[n] - exp_ema) / rec_sd
            gap = draw_index - last_seen.get(n, -1) - 1
            if gap < 0:
                gap = 0
            u = (gap - expected_gap) / (1.5 * expected_gap)
            a = aff[n] / 3.0
            dsec = (e_sec - sec_obs[sec_of[n]]) / (math.sqrt(e_sec) + 1.0)
            scores[n] = (w_z * math.tanh(z / 2.0)
                         + w_rec * math.tanh(r / 1.5)
                         + w_gap * math.tanh(u)
                         + w_echo * math.tanh(a / 2.0)
                         + w_sec * math.tanh(dsec / 2.0))

        return sorted(numbers, key=lambda m: (-scores[m], m))

    return rank