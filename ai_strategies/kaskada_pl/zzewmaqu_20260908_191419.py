NAME = "zzewmaqu"
DESCRIPTION = 'Scor cu 4 componente ortogonale standardizate cross-sectiunal: z-score global de frecventa (sigma binomial), EWMA pe recenta in fereastra (gamma), hazard de intarziere capatuit si deviatia z pe 4 sectoare in fereastra recenta. Orientarea deque-ului (newest_last) e detectata la runtime, flip-urile sunt contorizate in DIAG.'
SOURCE = 'quant-model + 2026-09-08'

DIAG = {'orient_flips': 0, 'calls': 0, 'orientation': 'assumed_newest_last'}

# Tinte grid-search (suprascribile in prompt-overrides):
# W_Z in [0,1], W_EW in [0,0.8], W_HAZ in [0,1.5], W_SEC in [0,1],
# GAMMA in [0.90,0.99], RCAP in [3,8]
PARAMS = {
    'W_Z': 0.35,
    'W_EW': 0.55,
    'W_HAZ': 0.15,
    'W_SEC': 0.30,
    'GAMMA': 0.94,
    'RCAP': 5.0,
}


def _zcross(vals):
    n = len(vals)
    if n == 0:
        return {}
    mean = sum(vals.values()) / float(n)
    var = sum((v - mean) * (v - mean) for v in vals.values()) / float(n)
    sd = var ** 0.5
    if sd < 1e-12:
        return dict((k, 0.0) for k in vals)
    return dict((k, (v - mean) / sd) for k, v in vals.items())


def _detect_orientation(recent_deque, last_seen):
    if not last_seen or not recent_deque:
        return None
    mx = max(last_seen.values())
    target = frozenset(n for n, idx in last_seen.items() if idx == mx)
    try:
        first = frozenset(recent_deque[0])
        last = frozenset(recent_deque[-1])
    except TypeError:
        return None
    if last == target:
        return 'newest_last'
    if first == target:
        return 'newest_first'
    return None


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    numbers = list(range(1, N + 1))
    p = (K / float(N)) if N > 0 else 0.5
    g_exp = (N / float(K)) if K > 0 else 2.0
    sec_len = max(1, N // 4)
    nsec = (N + sec_len - 1) // sec_len
    sec_of = dict((n, (n - 1) // sec_len) for n in numbers)
    sizes = [0] * nsec
    for n in numbers:
        sizes[sec_of[n]] += 1

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        DIAG['calls'] += 1
        counts = counts if counts else {}
        last_seen = last_seen if last_seen else {}

        total = 0
        for k, v in counts.items():
            if 1 <= k <= N:
                total += v
        D = (total / float(K)) if (K > 0 and total > 0) else 0.0
        mean_hist = D * p
        sd_hist = (D * p * (1.0 - p)) ** 0.5 if D > 0 else 0.0

        zvals = {}
        if sd_hist > 1e-9:
            inv = 1.0 / sd_hist
            for n in numbers:
                zvals[n] = (counts.get(n, 0) - mean_hist) * inv
        else:
            for n in numbers:
                zvals[n] = 0.0

        orient = _detect_orientation(recent_deque, last_seen)
        if orient == 'newest_first':
            DIAG['orient_flips'] += 1
            DIAG['orientation'] = 'newest_first'
        elif orient == 'newest_last':
            DIAG['orientation'] = 'newest_last'

        seq_list = list(recent_deque)
        if orient == 'newest_first':
            seq_list.reverse()
        seq = [frozenset(s) for s in seq_list]
        Wlen = len(seq)

        gamma = PARAMS['GAMMA']
        ew = dict((n, 0.0) for n in numbers)
        obs = [0] * nsec
        for i in range(Wlen):
            d = Wlen - 1 - i
            w = gamma ** d
            s = seq[i]
            for n in s:
                if 1 <= n <= N:
                    ew[n] += w
                    obs[sec_of[n]] += 1

        rcap = PARAMS['RCAP']
        haz = {}
        for n in numbers:
            ls = last_seen.get(n)
            gap = (draw_index - 1 - ls) if ls is not None else draw_index
            if gap < 0:
                gap = 0
            h = (gap / g_exp) if g_exp > 0 else 0.0
            if h > rcap:
                h = rcap
            haz[n] = h

        secz_map = {}
        if Wlen > 0 and K > 0 and N > 0:
            draws_eff = Wlen * K
            for si in range(nsec):
                q = sizes[si] / float(N)
                exp_si = draws_eff * q
                var_si = draws_eff * q * (1.0 - q)
                z_si = 0.0
                if var_si > 1e-12:
                    z_si = (obs[si] - exp_si) / (var_si ** 0.5)
                secz_map[si] = z_si
        secv = dict((n, secz_map.get(sec_of[n], 0.0)) for n in numbers)

        Zz = _zcross(zvals)
        Zw = _zcross(ew)
        Zh = _zcross(haz)
        Zs = _zcross(secv)

        wz = PARAMS['W_Z']
        we = PARAMS['W_EW']
        wh = PARAMS['W_HAZ']
        ws = PARAMS['W_SEC']
        score = {}
        for n in numbers:
            score[n] = wz * Zz[n] + we * Zw[n] + wh * Zh[n] + ws * Zs[n]
        return sorted(numbers, key=lambda n: (-score[n], -counts.get(n, 0), n))

    return rank