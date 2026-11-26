NAME = "zhazents"
DESCRIPTION = 'Gap-hazard log-saturat (mean-reversion pe intarziere) + reechilibrare pe 7 sectoare in fereastra recenta + poarta de regim bazata pe Herfindahl; combinare neliniara tanh.'
SOURCE = 'glm-5.3-flash quant model + 2026-09-08'

import math


def create(config):
    N = int(getattr(config, 'number_range', 49))
    K = int(getattr(config, 'numbers_drawn', 6))
    NS = 7  # sectoare de dimensiune egala (49/7)
    numbers = list(range(1, N + 1))
    sec_of = {n: (n - 1) * NS // N for n in numbers}
    p_one = K / float(N)
    mean_gap = 1.0 / p_one
    shift = math.log1p(1.0)  # valoarea hazardului la gap == mean_gap

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        di = draw_index if draw_index else 0
        total = float(di) if di > 0 else 1.0
        mu = total * p_one
        sd = math.sqrt(total * p_one * (1.0 - p_one))
        if sd <= 0.0:
            sd = 1.0

        # 1) gap-hazard saturat logaritmic (due cu saturație)
        haz = {}
        for n in numbers:
            ls = last_seen.get(n)
            g = di - ls if ls is not None else di + 1
            if g < 1:
                g = 1
            haz[n] = math.log1p(g / mean_gap) - shift

        # 2) statistici pe fereastra recenta: frecvente + contoare pe sector
        w = len(recent_deque)
        freq = dict.fromkeys(numbers, 0)
        obs_sec = [0] * NS
        tot_app = 0
        for d in recent_deque:
            for n in d:
                if 1 <= n <= N:
                    freq[n] += 1
                    obs_sec[sec_of[n]] += 1
                    tot_app += 1

        if tot_app > 0 and w > 0:
            # 3) poarta de regim: Herfindahl observat vs asteptat analitic
            hh = 0.0
            inv_t = 1.0 / tot_app
            for n in numbers:
                q = freq[n] * inv_t
                hh += q * q
            e_hh = 1.0 / N + (1.0 - p_one) / (w * K)
            r = hh / e_hh if e_hh > 0 else 1.0
            if r < 0.75:
                r = 0.75
            elif r > 1.35:
                r = 1.35
            gate = (r - 0.75) / 0.60
            # 4) deficit pe sector (varianta cu varianta redusa a 'due'-ului)
            exp_sec = tot_app / float(NS)
            sd_sec = math.sqrt(exp_sec) if exp_sec > 0 else 1.0
            sec_sc = {}
            for n in numbers:
                dsec = exp_sec - obs_sec[sec_of[n]]
                if dsec < 0.0:
                    dsec = 0.0
                sec_sc[n] = math.tanh(dsec / (1.5 * sd_sec))
            w_haz = 0.30 + 0.35 * gate
            w_sec = 0.20 + 0.20 * gate
            w_z = 0.40 - 0.30 * gate
        else:
            sec_sc = dict.fromkeys(numbers, 0.0)
            w_haz, w_sec, w_z = 0.45, 0.25, 0.35

        # 5) combinare neliniara saturata
        scores = {}
        inv2 = 1.0 / (2.5 * sd)
        for n in numbers:
            scores[n] = (w_z * math.tanh((counts.get(n, 0) - mu) * inv2)
                         + w_haz * math.tanh(1.2 * haz[n])
                         + w_sec * sec_sc[n])

        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank