NAME = "zzseccyc"
DESCRIPTION = 'Sector-cycle reversion gated by recent sector-balance entropy, blended with per-number macro frequency deviation and a capped overdue gap; reversion/momentum signs auto-orient on window halves.'
SOURCE = 'quant research 2026-09-17'

import math


def create(config):
    P = int(config.number_range)
    Wd = int(config.numbers_drawn)

    def _hp():
        return (
            int(getattr(config, 'SEC_SIZE', 6)),
            float(getattr(config, 'SEC_SIGN', 1.0)),
            float(getattr(config, 'DEV_SIGN', 1.0)),
            float(getattr(config, 'W_SEC', 0.7)),
            float(getattr(config, 'W_GAP', 0.5)),
            float(getattr(config, 'W_DEV', 0.4)),
            float(getattr(config, 'GAP_CAP', 2.5)),
            float(getattr(config, 'GATE_MIX', 0.5)),
            int(getattr(config, 'L_CAP', 150)),
            int(getattr(config, 'ORIENT_AUTO', 1)),
            float(getattr(config, 'MOM_THRESH', 0.2)),
        )

    cache = {}

    def _corr(xs, ys):
        n = len(xs)
        if n < 3:
            return 0.0
        mx = sum(xs) / n
        my = sum(ys) / n
        sxy = sxx = syy = 0.0
        for x, y in zip(xs, ys):
            dx = x - mx
            dy = y - my
            sxy += dx * dy
            sxx += dx * dx
            syy += dy * dy
        if sxx <= 1e-12 or syy <= 1e-12:
            return 0.0
        return sxy / math.sqrt(sxx * syy)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        (SEC_SIZE, SEC_SIGN, DEV_SIGN, W_SEC, W_GAP, W_DEV,
         GAP_CAP, GATE_MIX, L_CAP, ORIENT_AUTO, MOM_THRESH) = _hp()

        numbers = list(range(1, P + 1))

        draws = list(recent_deque) if recent_deque is not None else []
        if L_CAP > 0 and len(draws) > L_CAP:
            draws = draws[-L_CAP:]
        if len(draws) >= 2:
            a = max((last_seen.get(n, -1) for n in draws[0]), default=-1)
            b = max((last_seen.get(n, -1) for n in draws[-1]), default=-1)
            if a > b:
                draws.reverse()
        L = len(draws)

        if L < 8:
            exp_gap = P / float(Wd)
            cap = max(1.0, 2.5 * exp_gap)
            order = sorted(
                numbers,
                key=lambda n: (
                    -min((draw_index - last_seen[n]) if n in last_seen else draw_index + 1, cap),
                    -counts.get(n, 0),
                    n,
                ),
            )
            rank._diag = {'sector_span': 0.0, 'dev_span': 0.0, 'gap_span': 0.0,
                          'entropy_norm': 1.0, 'gate_factor': GATE_MIX,
                          'orient_flips': 0, 'pers_z': 0.0, 'L': L}
            return order

        lsig = 0
        for n in numbers:
            lsig += last_seen.get(n, -1)
        key = (draw_index, L,
               hash(frozenset(draws[0])), hash(frozenset(draws[-1])), lsig,
               (SEC_SIZE, SEC_SIGN, DEV_SIGN, W_SEC, W_GAP, W_DEV,
                GAP_CAP, GATE_MIX, ORIENT_AUTO, MOM_THRESH))
        hit = cache.get(key)
        if hit is not None:
            rank._diag = hit[1]
            return list(hit[0])

        n_sec = (P + SEC_SIZE - 1) // SEC_SIZE
        if n_sec < 1:
            n_sec = 1
        sec_of = {}
        for s in range(n_sec):
            lo = 1 + s * SEC_SIZE
            hi = min(P, lo + SEC_SIZE - 1)
            for n in range(lo, hi + 1):
                sec_of[n] = s
        for n in numbers:
            if n not in sec_of:
                sec_of[n] = n_sec - 1

        obs = [0] * (P + 2)
        obs_old = [0] * (P + 2)
        obs_new = [0] * (P + 2)
        mass = [0.0] * n_sec
        mass_old = [0.0] * n_sec
        mass_new = [0.0] * n_sec
        half = L // 2
        p_num = Wd / float(P)
        p_sec = SEC_SIZE / float(P)
        for t, d in enumerate(draws):
            newer = t >= half
            for n in d:
                if 1 <= n <= P:
                    obs[n] += 1
                    s = sec_of[n]
                    mass[s] += 1.0
                    if newer:
                        obs_new[n] += 1
                        mass_new[s] += 1.0
                    else:
                        obs_old[n] += 1
                        mass_old[s] += 1.0

        e_sec = L * SEC_SIZE * p_num
        sd_sec = math.sqrt(max(1e-9, L * Wd * p_sec * (1.0 - p_sec)))
        z_sec = [(mass[s] - e_sec) / sd_sec for s in range(n_sec)]
        span_sz = max(z_sec) - min(z_sec)
        inv_s = 1.0 / span_sz if span_sz > 1e-9 else 0.0

        e_num = L * p_num
        sd_num = math.sqrt(max(1e-9, L * p_num * (1.0 - p_num)))
        z_num = [(obs[n] - e_num) / sd_num for n in numbers]
        span_nz = max(z_num) - min(z_num)
        inv_n = 1.0 / span_nz if span_nz > 1e-9 else 0.0

        tot = sum(mass)
        H = 0.0
        if tot > 0:
            for s in range(n_sec):
                if mass[s] > 0:
                    ps = mass[s] / tot
                    H -= ps * math.log(ps)
        Hn = H / math.log(n_sec) if n_sec > 1 else 1.0
        if Hn < 0.0:
            Hn = 0.0
        elif Hn > 1.0:
            Hn = 1.0
        gate = GATE_MIX + (1.0 - GATE_MIX) * (1.0 - Hn)

        flips = 0
        sgn_sec = SEC_SIGN
        sgn_dev = DEV_SIGN
        pers_z = 0.0
        if ORIENT_AUTO:
            c_sec = _corr(mass_old, mass_new)
            c_num = _corr([obs_old[n] for n in numbers], [obs_new[n] for n in numbers])
            pers_z = c_num * math.sqrt(max(1.0, P - 2))
            if c_sec > MOM_THRESH:
                sgn_sec = -SEC_SIGN
                flips += 1
            if c_num > MOM_THRESH:
                sgn_dev = -DEV_SIGN
                flips += 1

        exp_gap = P / float(Wd)
        cap = max(1.0, GAP_CAP * exp_gap)
        ws = gate * W_SEC * sgn_sec
        wdv = W_DEV * sgn_dev
        wgp = W_GAP * sgn_dev
        sec_val = [0.0] * (P + 1)
        dev_val = [0.0] * (P + 1)
        gap_val = [0.0] * (P + 1)
        scores = [0.0] * (P + 1)
        for n in numbers:
            sec_c = -z_sec[sec_of[n]] * inv_s
            dev_c = -z_num[n - 1] * inv_n
            ls = last_seen.get(n)
            g = (draw_index - ls) if ls is not None else (draw_index + 1)
            gap_c = 2.0 * (min(g, cap) / cap) - 1.0
            sv = ws * sec_c
            dv = wdv * dev_c
            gv = wgp * gap_c
            sec_val[n] = sv
            dev_val[n] = dv
            gap_val[n] = gv
            scores[n] = sv + dv + gv

        order = sorted(numbers, key=lambda n: (-scores[n], -counts.get(n, 0), n))

        diag = {
            'sector_span': max(sec_val[1:]) - min(sec_val[1:]),
            'dev_span': max(dev_val[1:]) - min(dev_val[1:]),
            'gap_span': max(gap_val[1:]) - min(gap_val[1:]),
            'entropy_norm': Hn,
            'gate_factor': gate,
            'orient_flips': flips,
            'pers_z': pers_z,
            'L': L,
        }
        if len(cache) > 8192:
            cache.clear()
        cache[key] = (tuple(order), diag)
        rank._diag = diag
        return order

    return rank