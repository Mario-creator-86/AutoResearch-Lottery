import math

NAME = "zsecgapr"
DESCRIPTION = 'Sector-cycle reversion with entropy gate: standardized hypergeometric sector tilt over the recent window, capped overdue-gap score vs expected cycle P/K, and long-run deviation z-score vs expected frequency. Consecutive-draw overlap persistence z auto-flips reversion/momentum signs (ORIENT_AUTO) and entropy scales component weights (GATE_MIX). Exposes rank._diag with sector_span, dev_span, gap_span, entropy_norm, gate_factor, orient_flips, pers_z, L.'
SOURCE = 'model + 2026-09-17'

PARAMS = {
    'REC_W': 60,
    'SEC_SIZE': 6,
    'SEC_SIGN': -1,
    'DEV_SIGN': -1,
    'GAP_SIGN': 1,
    'W_SEC': 0.7,
    'W_GAP': 0.5,
    'W_DEV': 0.4,
    'GAP_CAP': 2.5,
    'GATE_MIX': 0.5,
    'ORIENT_AUTO': 1,
    'PERS_THR': 0.5,
    'CLAMP_Z': 3.0,
}


def create(config):
    P = int(config.number_range)
    K = int(config.numbers_drawn)
    if P < 2:
        P = 2
    if K < 1:
        K = 1
    elif K > P:
        K = P

    # moments of |A cap B| for two independent K-subsets of P (persistence baseline)
    p_i = K * (K - 1) / (P * (P - 1)) if P > 1 else 0.0
    if K >= 4 and P > 3:
        p_ij = p_i * ((K - 2.0) / (P - 2.0)) * ((K - 3.0) / (P - 3.0))
    else:
        p_ij = p_i * p_i
    var_pair = P * p_i * (1.0 - p_i) + P * (P - 1) * (p_ij - p_i * p_i)
    if var_pair <= 1e-12:
        var_pair = 1.0

    def _get(name):
        v = None
        try:
            v = getattr(config, name)
        except Exception:
            v = None
        if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
            v = PARAMS[name]
        return v

    state = {'diag': {}}

    def diag():
        return dict(state['diag'])

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        rec_w = int(_get('REC_W'))
        if recent_window is not None and int(recent_window) > 0:
            rec_w = min(rec_w, int(recent_window))
        sec_size = max(2, int(_get('SEC_SIZE')))
        clamp_z = float(_get('CLAMP_Z'))

        numbers = list(range(1, P + 1))
        draws = []
        if rec_w > 0 and recent_deque:
            draws = [set(d) for d in list(recent_deque)[-rec_w:]]
        Wd = len(draws)

        n_sec = (P + sec_size - 1) // sec_size
        sec_m = [0] * n_sec
        for n in numbers:
            sec_m[(n - 1) // sec_size] += 1

        sec_count = [0] * n_sec
        for d in draws:
            for n in d:
                if 1 <= n <= P:
                    sec_count[(n - 1) // sec_size] += 1

        tilt = [0.0] * n_sec
        if Wd > 0:
            fpc = (P - K) / float(P - 1) if P > 1 else 1.0
            for s in range(n_sec):
                share = sec_m[s] / float(P)
                exp_s = K * Wd * share
                var_s = Wd * K * share * (1.0 - share) * fpc
                std_s = math.sqrt(var_s) if var_s > 1e-12 else 1.0
                z = (sec_count[s] - exp_s) / std_s
                if z > clamp_z:
                    z = clamp_z
                elif z < -clamp_z:
                    z = -clamp_z
                tilt[s] = z

        tot = K * Wd
        entropy_norm = 0.0
        if tot > 0:
            H = 0.0
            for s in range(n_sec):
                if sec_count[s] > 0:
                    q = sec_count[s] / float(tot)
                    H -= q * math.log(q)
            Hmax = 0.0
            for s in range(n_sec):
                q = sec_m[s] / float(P)
                if q > 0:
                    Hmax -= q * math.log(q)
            if Hmax > 1e-12:
                entropy_norm = H / Hmax
                if entropy_norm < 0.0:
                    entropy_norm = 0.0
                elif entropy_norm > 1.0:
                    entropy_norm = 1.0

        pers_z = 0.0
        if Wd > 1:
            obs = 0
            for i in range(Wd - 1):
                obs += len(draws[i] & draws[i + 1])
            mu_p = (Wd - 1) * K * (K - 1) / float(P - 1) if P > 1 else 0.0
            pers_z = (obs - mu_p) / math.sqrt((Wd - 1) * var_pair)

        orient_flips = 0
        flip = 1.0
        if int(_get('ORIENT_AUTO')) == 1 and pers_z > float(_get('PERS_THR')):
            flip = -1.0
            orient_flips = 1
        s_sec = float(_get('SEC_SIGN')) * flip
        s_dev = float(_get('DEV_SIGN')) * flip
        s_gap = float(_get('GAP_SIGN')) * flip

        g = float(_get('GATE_MIX')) * entropy_norm
        if g < 0.0:
            g = 0.0
        elif g > 1.0:
            g = 1.0
        w_sec = float(_get('W_SEC')) * (1.0 - g)
        w_gap = float(_get('W_GAP')) * (1.0 + g)
        w_dev = float(_get('W_DEV')) * (1.0 - 0.5 * g)

        D = float(draw_index) if draw_index is not None and draw_index > 0 else 1.0
        p1 = K / float(P)
        mu_d = D * p1
        sd_d = math.sqrt(D * p1 * (1.0 - p1))
        if sd_d <= 1e-12:
            sd_d = 1.0

        Eg = P / float(K) if K > 0 else 1.0
        gap_cap = float(_get('GAP_CAP'))

        c_sec_v = [0.0] * P
        c_gap_v = [0.0] * P
        c_dev_v = [0.0] * P
        scores = [0.0] * P
        for n in numbers:
            ls = None
            if last_seen is not None:
                ls = last_seen.get(n)
            gap_n = (draw_index - ls) if ls is not None else (draw_index + 1)
            if gap_n < 0:
                gap_n = 0
            gap_frac = min(gap_n / Eg, gap_cap) - 1.0
            dz = (counts.get(n, 0) - mu_d) / sd_d
            if dz > clamp_z:
                dz = clamp_z
            elif dz < -clamp_z:
                dz = -clamp_z
            i = n - 1
            a = s_sec * w_sec * tilt[i // sec_size]
            b = s_gap * w_gap * gap_frac
            c = s_dev * w_dev * dz
            c_sec_v[i] = a
            c_gap_v[i] = b
            c_dev_v[i] = c
            scores[i] = a + b + c

        L = 0
        for w in (w_sec, w_gap, w_dev):
            if w > 1e-9:
                L += 1
        state['diag'] = {
            'sector_span': max(c_sec_v) - min(c_sec_v),
            'dev_span': max(c_dev_v) - min(c_dev_v),
            'gap_span': max(c_gap_v) - min(c_gap_v),
            'entropy_norm': entropy_norm,
            'gate_factor': g,
            'orient_flips': orient_flips,
            'pers_z': pers_z,
            'L': L,
            'Wd': Wd,
            'sec_sign_eff': s_sec,
            'dev_sign_eff': s_dev,
            'gap_sign_eff': s_gap,
        }

        return sorted(numbers, key=lambda n: (-scores[n - 1], n))

    rank._diag = diag
    rank.diag = diag
    return rank