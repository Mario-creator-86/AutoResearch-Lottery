import math

NAME = "zzgapent"
DESCRIPTION = ('Regime-switching gap-cycle: amestec adaptiv intre momentum pe fereastra scurta, '
               'intarziere (gap) raportata la intervalul istoric propriu al fiecarui numar, '
               'z-score pe sectoare si deviatia de frecventa pe termen lung. Poarta de regim '
               'este un sigmoid pe chi2 al distributiei sectoarelor din fereastra recenta: '
               'over-dispersie => tilt catre mean-reversion, under-dispersie => momentum.')
SOURCE = 'model + 2026-09-16'

DIAG = {'calls': 0, 'gate_min': 1.0, 'gate_max': 0.0, 'gate_sum': 0.0,
        'gate_flips': 0, '_last_side': 0}


def create(config):
    n_num = config.number_range
    drawn = config.numbers_drawn

    k_mom = 8        # sub-fereastra de momentum
    w_mom = 0.90     # greutate momentum z (regim cald)
    w_gap = 1.10     # greutate gap-cycle (regim reversiune)
    w_sect = 0.55    # greutate sector z, semn negativ (regim reversiune)
    w_freq = 0.35    # greutate deviatie frecventa lunga (regim cald)
    gap_cap = 5.0
    sec_size = 7
    min_sec_win = 5

    n_sec = (n_num + sec_size - 1) // sec_size

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, n_num + 1))
        p = drawn / float(n_num)
        inv_p = 1.0 / p

        dq = list(recent_deque)
        weff = len(dq)
        k = k_mom if weff > k_mom else weff

        mom_counts = [0] * (n_num + 1)
        sec_counts = [0] * n_sec
        gate = 0.5
        exp_sec = 0.0

        if weff:
            mom_start = weff - k
            for i in range(weff):
                in_mom = i >= mom_start
                for x in dq[i]:
                    if 1 <= x <= n_num:
                        sec_counts[(x - 1) // sec_size] += 1
                        if in_mom:
                            mom_counts[x] += 1
            exp_sec = weff * drawn * sec_size / float(n_num)
            if weff >= min_sec_win and exp_sec > 0:
                chi2 = 0.0
                for c in sec_counts:
                    d = c - exp_sec
                    chi2 += d * d / exp_sec
                dfree = n_sec - 1
                z = (chi2 - dfree) / math.sqrt(2.0 * dfree)
                if z > 30.0:
                    z = 30.0
                elif z < -30.0:
                    z = -30.0
                gate = 1.0 / (1.0 + math.exp(-z))

        dg = DIAG
        dg['calls'] += 1
        if gate < dg['gate_min']:
            dg['gate_min'] = gate
        if gate > dg['gate_max']:
            dg['gate_max'] = gate
        dg['gate_sum'] += gate
        side = 1 if gate >= 0.5 else -1
        if dg['_last_side'] and side != dg['_last_side']:
            dg['gate_flips'] += 1
        dg['_last_side'] = side

        exp_mom = k * p
        sd_mom = math.sqrt(exp_mom * (1.0 - p)) if exp_mom > 0 else 1.0
        exp_tot = draw_index * p
        sd_tot = math.sqrt(exp_tot * (1.0 - p)) if exp_tot > 0 else 1.0

        var_draw = 0.0
        if n_num > 1:
            var_draw = (drawn * (sec_size / float(n_num))
                        * ((n_num - sec_size) / float(n_num))
                        * ((n_num - drawn) / float(n_num - 1)))
        sd_sec = math.sqrt(weff * var_draw) if (weff * var_draw) > 0 else 1.0

        scores = [0.0] * (n_num + 1)
        for n in numbers:
            c = counts.get(n, 0)
            fz = (c - exp_tot) / sd_tot
            mz = (mom_counts[n] - exp_mom) / sd_mom
            ls = last_seen.get(n)
            cur_gap = (draw_index - ls) if ls is not None else (draw_index + 1)
            if cur_gap < 0:
                cur_gap = 0
            mean_gap = (draw_index / float(c)) if c > 0 else inv_p
            r = (cur_gap / mean_gap) if mean_gap > 0 else 0.0
            if r > gap_cap:
                r = gap_cap
            gn = math.log1p(r)
            sz = (sec_counts[(n - 1) // sec_size] - exp_sec) / sd_sec
            scores[n] = ((1.0 - gate) * (w_mom * mz + w_freq * fz)
                         + gate * (w_gap * gn - w_sect * sz))

        return sorted(numbers, key=lambda m: (-scores[m], m))

    return rank