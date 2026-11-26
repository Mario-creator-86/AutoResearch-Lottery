NAME = "zzsewqua"
DESCRIPTION = 'Scor multi-factor determinist: z-score frecventa istorica, momentum EWMA (GAMMA) pe fereastra recenta, acceleratie de ordinul doi (EWMA rapid - EWMA lent), hazard de intarziere capuit la RCAP si ciclu pe sectoare, combinate liniar plus interactiune KAPPA. Detecteaza automat orientarea deque-ului (newest_last/newest_first) si publica DIAG cu orient_flips.'
SOURCE = 'model + 2026-09-08'

import math

PARAMS = {
    'W_Z': 0.50,
    'W_EW': 0.35,
    'W_HAZ': 0.60,
    'W_SEC': 0.25,
    'W_CYC': 0.15,
    'GAMMA': 0.95,
    'RCAP': 5.0,
    'KAPPA': 0.10,
}

DIAG = {'orient_flips': 0, 'orientation': 'unknown', 'calls': 0}

_ORIENT_STATE = {'prev': None}
_POW_CACHE = {}


def _gamma_powers(gamma, length):
    key = round(float(gamma), 12)
    lst = _POW_CACHE.get(key)
    if lst is None:
        lst = [1.0]
        _POW_CACHE[key] = lst
    while len(lst) < length:
        lst.append(lst[-1] * gamma)
    return lst


def _detect_orientation(recent_deque, last_seen):
    L = len(recent_deque)
    if L < 2:
        return None
    try:
        first = recent_deque[0]
        last = recent_deque[L - 1]
        if len(first) == 0 or len(last) == 0:
            return None
        m0 = 0.0
        for n in first:
            m0 += last_seen.get(n, -1)
        m1 = 0.0
        for n in last:
            m1 += last_seen.get(n, -1)
        m0 /= len(first)
        m1 /= len(last)
        if m1 > m0 + 1e-9:
            return 'newest_last'
        if m0 > m1 + 1e-9:
            return 'newest_first'
    except Exception:
        return None
    return None


def _standardize(numbers, raw):
    total = 0.0
    for n in numbers:
        total += raw[n]
    mean = total / len(numbers)
    var = 0.0
    for n in numbers:
        d = raw[n] - mean
        var += d * d
    sd = math.sqrt(var / len(numbers))
    if sd < 1e-12:
        return dict.fromkeys(numbers, 0.0)
    inv = 1.0 / sd
    return {n: (raw[n] - mean) * inv for n in numbers}


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    p = (K / N) if N > 0 else 0.0

    n_sectors = 4
    sector_size = max(1, N // n_sectors)
    sector_of = {}
    for n in range(1, N + 1):
        s = (n - 1) // sector_size
        if s > n_sectors - 1:
            s = n_sectors - 1
        sector_of[n] = s

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        DIAG['calls'] += 1
        gamma = float(PARAMS['GAMMA'])
        rcap = float(PARAMS['RCAP'])
        kappa = float(PARAMS['KAPPA'])
        w_z = float(PARAMS['W_Z'])
        w_ew = float(PARAMS['W_EW'])
        w_hz = float(PARAMS['W_HAZ'])
        w_sc = float(PARAMS['W_SEC'])
        w_cy = float(PARAMS['W_CYC'])

        numbers = list(range(1, N + 1))

        orient = _detect_orientation(recent_deque, last_seen)
        if orient is not None:
            prev = _ORIENT_STATE['prev']
            if prev is not None and prev != orient:
                DIAG['orient_flips'] += 1
            _ORIENT_STATE['prev'] = orient
        DIAG['orientation'] = orient if orient is not None else 'assumed_newest_last'
        newest_last = orient != 'newest_first'

        tot = 0
        for n in numbers:
            tot += counts.get(n, 0)
        T = (tot / K) if K > 0 else 0.0
        mu = T * p
        sd = math.sqrt(max(T * p * (1.0 - p), 0.0))
        z_raw = dict.fromkeys(numbers, 0.0)
        if sd > 1e-12:
            inv = 1.0 / sd
            for n in numbers:
                z_raw[n] = (counts.get(n, 0) - mu) * inv

        L = len(recent_deque)
        ew = dict.fromkeys(numbers, 0.0)
        ewf = dict.fromkeys(numbers, 0.0)
        sec_ew = [0.0] * n_sectors
        pw = _gamma_powers(gamma, L)
        pwf = _gamma_powers(gamma * gamma * gamma, L)
        norm = 0.0
        normf = 0.0
        for i, dset in enumerate(recent_deque):
            age = (L - 1 - i) if newest_last else i
            w = pw[age]
            wf = pwf[age]
            norm += w
            normf += wf
            for n in dset:
                if 1 <= n <= N:
                    ew[n] += w
                    ewf[n] += wf
                    sec_ew[sector_of[n]] += w

        ew_raw = dict.fromkeys(numbers, 0.0)
        acc_raw = dict.fromkeys(numbers, 0.0)
        cyc_raw = dict.fromkeys(numbers, 0.0)
        if norm > 1e-12 and normf > 1e-12 and p > 0.0:
            mu_s = sector_size * p
            sec_dev = [0.0] * n_sectors
            for s in range(n_sectors):
                if mu_s > 1e-9:
                    sec_dev[s] = (sec_ew[s] / norm - mu_s) / mu_s
            for n in numbers:
                dev = ew[n] / norm - p
                ew_raw[n] = dev
                acc_raw[n] = (ewf[n] / normf - p) - dev
                cyc_raw[n] = -sec_dev[sector_of[n]]

        if last_seen:
            M = max(last_seen.values())
            never_gap = (M - min(last_seen.values())) + 1
        else:
            M = -1
            never_gap = 1
        haz_raw = dict.fromkeys(numbers, 0.0)
        for n in numbers:
            ls = last_seen.get(n)
            gap = (M - ls) if ls is not None else never_gap
            if gap < 0:
                gap = 0
            r = gap * p
            if r > rcap:
                r = rcap
            haz_raw[n] = r

        z_s = _standardize(numbers, z_raw)
        ew_s = _standardize(numbers, ew_raw)
        acc_s = _standardize(numbers, acc_raw)
        haz_s = _standardize(numbers, haz_raw)
        cyc_s = _standardize(numbers, cyc_raw)

        score = {}
        for n in numbers:
            score[n] = (w_z * z_s[n] + w_ew * ew_s[n] + w_hz * haz_s[n]
                        + w_sc * acc_s[n] + w_cy * cyc_s[n]
                        + kappa * (ew_s[n] * haz_s[n]))

        return sorted(numbers, key=lambda n: (-score[n], n))

    rank.DIAG = DIAG
    return rank