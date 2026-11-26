NAME = "zmhr4"
DESCRIPTION = ('Reversie multi-orizont 6/49: deviatie z pe termen lung (favorizeaza numerele reci), '
               'momentum EWMA pe fereastra recenta (favorizeaza numerele calde), hazard de intarziere '
               'cap-at si cicluri de deficit pe 7 sectoare; termen neliniar de emergenta ewma_z*(-z). '
               'Permutare completa 1..N, determinista, doar stdlib.')
SOURCE = 'plugin cantitativ mhr4 v1, grid-search ready, 2026-09-08'

import math

# Intervale de scanare pentru grid-search: seteaza atribute pe config (w_z, w_ew, w_haz, w_sec,
# gamma, rcap, cyc, kappa) sau patch-uie PARAM_GRID[cheie]['default'] inainte de create(config).
PARAM_GRID = {
    'w_z':   {'interval': [0.0, 1.0],   'default': 0.35, 'role': 'deviatie z long-run, favor rece; ablare: z'},
    'w_ew':  {'interval': [0.0, 0.8],   'default': 0.30, 'role': 'momentum EWMA recent; ablare: mom'},
    'w_haz': {'interval': [0.0, 1.5],   'default': 0.60, 'role': 'hazard overdue cap-at (echiv W_OV); ablare: haz'},
    'w_sec': {'interval': [0.0, 1.0],   'default': 0.25, 'role': 'deficit sectorial (cicluri); ablare: sec'},
    'gamma': {'interval': [0.90, 0.99], 'default': 0.95, 'role': 'decay EWMA'},
    'rcap':  {'interval': [3.0, 8.0],   'default': 5.0,  'role': 'cap hazard in unitati de 1/p (~8.17)'},
    'cyc':   {'interval': [0.0, 3.0],   'default': 1.0,  'role': 'exponent convexitate deficit sectorial'},
    'kappa': {'interval': [0.0, 0.5],   'default': 0.10, 'role': 'interactiune neliniara ewma_z * (-z)'},
}

DIAG = {'orient_flips': 0, 'orientation_checks': 0, 'calls': 0, 'last_window_len': 0}


def _cfg(config, key):
    try:
        return float(getattr(config, key))
    except Exception:
        return float(PARAM_GRID[key]['default'])


def create(config):
    R = int(config.number_range)
    D = int(config.numbers_drawn)

    w_z = _cfg(config, 'w_z')
    w_ew = _cfg(config, 'w_ew')
    w_haz = _cfg(config, 'w_haz')
    w_sec = _cfg(config, 'w_sec')
    gamma = _cfg(config, 'gamma')
    rcap = _cfg(config, 'rcap')
    cyc = _cfg(config, 'cyc')
    kappa = _cfg(config, 'kappa')

    p = D / float(R)
    cap_gap = rcap / p
    n_sec = 7
    sec_size = -(-R // n_sec)

    sec_of = [0] * (R + 1)
    sec_exp = [0.0] * n_sec
    sec_var = [1e-9] * n_sec
    for d in range(n_sec):
        lo = d * sec_size + 1
        hi = min(R, (d + 1) * sec_size)
        K = hi - lo + 1
        if K <= 0:
            continue
        for n in range(lo, hi + 1):
            sec_of[n] = d
        share = K / float(R)
        sec_exp[d] = D * share
        var = D * share * (1.0 - share) * ((R - D) / float(max(R - 1, 1)))
        sec_var[d] = max(var, 1e-9)

    global DIAG
    diag = {'orient_flips': 0, 'orientation_checks': 0, 'calls': 0, 'last_window_len': 0}
    DIAG = diag

    _wpow = {}

    def _weights(L):
        got = _wpow.get(L)
        if got is None:
            w = [gamma ** (L - 1 - i) for i in range(L)]
            wsum = 0.0
            w2 = 0.0
            for v in w:
                wsum += v
                w2 += v * v
            got = (w, wsum, w2)
            _wpow[L] = got
        return got

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        diag['calls'] += 1
        numbers = list(range(1, R + 1))
        N = draw_index if draw_index > 0 else 1
        exp_total = N * p
        sd_total = max(math.sqrt(N * p * (1.0 - p)), 1e-9)

        lst = list(recent_deque)
        L = len(lst)
        diag['last_window_len'] = L

        # DIAG orientare: cel mai recent draw (last_seen == draw_index-1) trebuie sa fie la COADA deque-ului (newest_last).
        if L >= 2:
            prev = draw_index - 1
            fs = 0
            for x in lst[0]:
                if last_seen.get(x, -1) == prev:
                    fs += 1
            bs = 0
            for x in lst[-1]:
                if last_seen.get(x, -1) == prev:
                    bs += 1
            diag['orientation_checks'] += 1
            if fs >= 3 and fs > bs:
                lst.reverse()
                diag['orient_flips'] += 1

        w, wsum, w2 = _weights(L)
        ewma = [0.0] * (R + 1)
        sec_hits = [0] * n_sec
        for wi, s in zip(w, lst):
            if wi <= 0.0:
                continue
            for x in s:
                if 1 <= x <= R:
                    ewma[x] += wi
                    sec_hits[sec_of[x]] += 1
        sd_ew = max(math.sqrt(p * (1.0 - p) * w2), 1e-9) if w2 > 0.0 else 1e-9

        u_sec = [0.0] * n_sec
        for d in range(n_sec):
            e = L * sec_exp[d]
            sd = math.sqrt(max(L * sec_var[d], 1e-9))
            u_sec[d] = (e - sec_hits[d]) / sd

        scores = [0.0] * (R + 1)
        for n in numbers:
            c = counts.get(n, 0)
            z = (c - exp_total) / sd_total
            ewz = (ewma[n] - p * wsum) / sd_ew
            gap = draw_index - last_seen.get(n, -1)
            haz = gap / cap_gap
            if haz < 0.0:
                haz = 0.0
            elif haz > 1.0:
                haz = 1.0
            u = u_sec[sec_of[n]]
            if cyc > 0.0:
                g = abs(u) ** cyc
                if u < 0.0:
                    g = -g
            else:
                g = 1.0 if u > 0.0 else (-1.0 if u < 0.0 else 0.0)
            inter = ewz * (-z)
            if inter > 6.0:
                inter = 6.0
            elif inter < -6.0:
                inter = -6.0
            scores[n] = (w_z * (-z) + w_ew * ewz + w_haz * haz + w_sec * g + kappa * inter)

        return sorted(numbers, key=lambda nn: (-scores[nn], nn))

    rank.DIAG = diag
    return rank