NAME = "zpairort"
DESCRIPTION = 'Afinitate de perechi de ordinul doi: reziduuri de co-ocurenta fata de independenta (pc - c_a*c_b/L) ponderate cu deviatia marginelelor a partenerilor, proiectate ortogonal (Gram-Schmidt) pe marginele si momentum; plus momentum de recency exponential pe varsta ultimei aparitii, gap saturat Michaelis-Menten (half-sat = 14, exact gap-ul geometric mediu la 6/90) cu band-gate pe zona moderat-intarziata, ancora slaba de z pe istoricul complet si etaj de entropie care moduleaza ponderea perechilor. Agregare exact-integer pe fereastra, cheie de cache canonica sortata (independenta de ordine si de recent_window), deci flip-check-ul deque-inversat-permutare-identica e garantat structural.'
SOURCE = 'model cantitativ + 2026-11-22'

import math

DIAG = {'corr_glue_z': None, 'corr_mom_z': None, 'corr_gap_mom': None,
        'uniformity': None, 'ent_mult': None, 'boost_q': None,
        'gap_sat_q': None, 'calls': 0}

# hiperparametri pt sweep-ul prescris:
# W_PAIR{0,0.15,0.3,0.6} x W_GAP{0.2,0.5,0.9} x ENT_FLOOR{0,0.35,0.7}; ablatie W_PAIR=0
W_Z = 0.60        # marginelele pe fereastra recenta (semnal primar)
W_LZ = 0.10       # ancora slaba: deviatie z pe tot istoricul
W_PAIR = 0.30     # glue de perechi rezidualizat (ablatie la 0)
W_MOM = 0.25      # momentum de recency
W_GAP = 0.50      # gap saturat cu band-gate
ENT_FLOOR = 0.35  # etaj de entropie pe ponderea perechilor
ENT_SPREAD = 14.0 # calibrare: (1-u)*SPREAD ~= 0.5 la zgomotul de esantionare pt rw~200
T_MOM = 10.0      # constanta de timp momentum
GATE_HALF = 14.0  # (1-p)/p = 14 = gap geometric mediu la 6/90
GATE_LO = 0.35    # band-gate jos pe sat (g ~ 7.5)
GATE_HI = 0.85    # band-gate sus pe sat (g ~ 79)
GATE_E = 0.07     # latimea rampelor gate

_WCACHE = {}
_WCACHE_MAX = 32


def _canon(recent_deque, N):
    items = []
    for d in (recent_deque or []):
        try:
            it = [x for x in d if 1 <= x <= N]
        except TypeError:
            continue
        if it:
            it.sort()
            items.append(tuple(it))
    items.sort()
    return tuple(items)


def _stats(canon, N, K):
    c = [0] * (N + 1)
    pairs = {}
    for draw in canon:
        m = len(draw)
        for x in draw:
            c[x] += 1
        for i in range(m):
            a = draw[i]
            for j in range(i + 1, m):
                kk = (a, draw[j])
                pairs[kk] = pairs.get(kk, 0) + 1
    L = len(canon)
    pr = K / float(N)
    mu = L * pr
    sd = math.sqrt(max(mu * (1.0 - pr), 1e-12))
    return (L, mu, sd, c, sorted(pairs.items()))


def _std(v):
    n = len(v)
    if n == 0:
        return []
    m = 0.0
    for x in v:
        m += x
    m /= float(n)
    var = 0.0
    for x in v:
        var += (x - m) * (x - m)
    var /= float(n)
    if var < 1e-18:
        return [0.0] * n
    s = math.sqrt(var)
    return [(x - m) / s for x in v]


def _corrstd(a, b):
    if not a:
        return 0.0
    num = 0.0
    for x, y in zip(a, b):
        num += x * y
    c = num / float(len(a))
    if c > 1.0:
        c = 1.0
    elif c < -1.0:
        c = -1.0
    return c


def _ortho(v, basis):
    out = list(v)
    for b in basis:
        nb = 0.0
        for x in b:
            nb += x * x
        if nb < 1e-12:
            continue
        d = 0.0
        for x, y in zip(out, b):
            d += x * y
        f = d / nb
        out = [x - f * y for x, y in zip(out, b)]
    return _std(out)


def _gate(s):
    lo = GATE_LO
    hi = GATE_HI
    e = GATE_E
    if s <= lo - e or s >= hi + e:
        return 0.0
    if s < lo:
        return (s - (lo - e)) / e
    if s > hi:
        return 1.0 - (s - hi) / e
    return 1.0


def _quant(vals_sorted, q):
    n = len(vals_sorted)
    if n == 0:
        return 0.0
    pos = q * (n - 1)
    lo = int(pos)
    if lo >= n - 1:
        return vals_sorted[-1]
    fr = pos - lo
    return vals_sorted[lo] * (1.0 - fr) + vals_sorted[lo + 1] * fr


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    p = K / float(N)
    lnn = math.log(float(N))

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        DIAG['calls'] += 1
        canon = _canon(recent_deque, N)
        key = (N, K, canon)
        st = _WCACHE.get(key)
        if st is None:
            st = _stats(canon, N, K)
            if len(_WCACHE) >= _WCACHE_MAX:
                _WCACHE.clear()
            _WCACHE[key] = st
        L, mu, sd, c, plist = st

        names = list(range(1, N + 1))
        nn = N

        # semnal primar: deviatia marginelelor pe fereastra recenta
        if L > 0:
            zs = _std([(c[n] - mu) / sd for n in names])
        else:
            zs = [0.0] * nn

        # ancora slaba pe istoricul complet (nu depinde de fereastra)
        T = float(draw_index) if draw_index > 0 else 1.0
        mul = T * p
        sdl = math.sqrt(max(mul * (1.0 - p), 1e-12))
        if counts:
            lzs = _std([((counts.get(n, 0) or 0) - mul) / sdl for n in names])
        else:
            lzs = [0.0] * nn

        # momentum de recency + gap saturat din varsta ultimei aparitii
        mv = [0.0] * nn
        satv = [0.0] * nn
        gtv = [0.0] * nn
        for i, n in enumerate(names):
            ls = last_seen.get(n) if last_seen else None
            if ls is None:
                age = draw_index + 1
            else:
                age = draw_index - ls
                if age < 0:
                    age = 0
            mv[i] = math.exp(-age / T_MOM)
            s = age / (age + GATE_HALF)
            satv[i] = s
            gtv[i] = s * _gate(s)

        # etaj de entropie pe ponderea perechilor
        mult = ENT_FLOOR
        DIAG['uniformity'] = None
        if L > 0:
            tot = float(K * L)
            H = 0.0
            for n in names:
                qn = c[n] / tot
                if qn > 0.0:
                    H -= qn * math.log(qn)
            u = H / lnn
            DIAG['uniformity'] = u
            conc = (1.0 - u) * ENT_SPREAD
            if conc < 0.0:
                conc = 0.0
            elif conc > 1.0:
                conc = 1.0
            mult = ENT_FLOOR + (1.0 - ENT_FLOOR) * conc
        DIAG['ent_mult'] = mult

        # glue de perechi: reziduu de co-ocurenta ponderat cu z-ul partenerului
        gl = [0.0] * nn
        if L > 0 and plist:
            FL = float(L)
            for (a, b), k in plist:
                cc = (c[a] * c[b]) / FL
                r = (k - cc) / math.sqrt(cc + 1.0)
                ia = a - 1
                ib = b - 1
                zb = zs[ib]
                if zb != 0.0:
                    gl[ia] += r * zb
                za = zs[ia]
                if za != 0.0:
                    gl[ib] += r * za
        gs = _std(gl)
        ms = _std(mv)

        # diagnostice pentru sweep (masurate INAINTE de ortogonalizare)
        DIAG['corr_glue_z'] = _corrstd(gs, zs)
        DIAG['corr_mom_z'] = _corrstd(ms, zs)
        DIAG['corr_gap_mom'] = _corrstd(_std(satv), ms)
        svals = sorted(satv)
        DIAG['gap_sat_q'] = (_quant(svals, 0.25), _quant(svals, 0.5),
                             _quant(svals, 0.75), _quant(svals, 0.9))
        act = [s for s in satv if _gate(s) >= 0.5]
        act.sort()
        DIAG['boost_q'] = (_quant(act, 0.5), _quant(act, 0.9),
                           (act[-1] if act else 0.0))

        # ortogonalizare Gram-Schmidt: forteaza structural corr=0 fata de z
        ms_p = _ortho(ms, [zs])
        gs_p = _ortho(gs, [zs, ms_p])
        gts = _std(gtv)

        wp = W_PAIR * mult
        score = [0.0] * nn
        for i in range(nn):
            score[i] = (W_Z * zs[i] + W_LZ * lzs[i] + W_MOM * ms_p[i]
                        + wp * gs_p[i] + W_GAP * gts[i])
        order = sorted(range(nn), key=lambda i: (-score[i], names[i]))
        return [names[i] for i in order]

    return rank