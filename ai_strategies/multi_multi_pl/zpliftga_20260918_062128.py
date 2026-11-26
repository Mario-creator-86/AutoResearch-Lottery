import math
from collections import deque

NAME = "zpliftga"
DESCRIPTION = ('Pair-lift conditionat cu gate de persistenta + momentum recent. Scor per numar = '
               'gate*(W_EW*z_EWMA(half-life KD) + W_REP*z_recency(ultimele RG extrageri)) + '
               'LIFT_SIGN*(W_L1*z_lift fereastra KD + W_L2*z_lift fereastra KD2). Liftul numarului n '
               'masura co-occurinta cu hub-urile fierbinti fata de asteptarea conditionata '
               'c_n*S_hot/W; gate amplifica blocul de momentum doar cand liftul agregat al '
               'hub-urilor > 1. Ignora counts (frecventa pe termen lung) si overdue; expune '
               'rank.PARAMS, rank._diag si rank._diag_cum pentru ablatie.')
SOURCE = 'glm-5.3-flash:cloud + 2026-09-18'

PARAMS = {
    'W_L1': 2.2,   # pondere pair-lift fereastra scurta KD
    'W_L2': 1.1,   # pondere pair-lift fereastra lunga KD2
    'W_REP': 1.5,  # pondere recency (aparitii in ultimele RG)
    'W_EW': 0.35,  # pondere EWMA momentum
    'KD': 20,      # half-life EWMA / fereastra lift scurta
    'KD2': 80,     # fereastra lift lunga
    'RG': 8,       # fereastra recency
    'GATE_K': 0.75,# intensitate gate pe momentum
    'LIFT_SIGN': 1.0,  # +1 momentum, -1 reversion
    'HOT_K': 20,   # marimea setului de hub-uri
}


def create(config):
    P = int(config.number_range)
    DR = int(config.numbers_drawn)
    p = float(DR) / float(P)

    W_L1 = float(PARAMS['W_L1'])
    W_L2 = float(PARAMS['W_L2'])
    W_REP = float(PARAMS['W_REP'])
    W_EW = float(PARAMS['W_EW'])
    KD = max(2, int(PARAMS['KD']))
    KD2 = max(KD, int(PARAMS['KD2']))
    RG = max(1, int(PARAMS['RG']))
    GATE_K = float(PARAMS['GATE_K'])
    LIFT_SIGN = float(PARAMS['LIFT_SIGN'])
    HOT_K = min(P, max(2, int(PARAMS['HOT_K'])))

    decay = 0.5 ** (1.0 / KD)
    mu_ew = p / (1.0 - decay)
    sd_ew = math.sqrt(max(1e-12, p * (1.0 - p) / (1.0 - decay * decay)))
    lnP = math.log(P)

    sizes = (KD, KD2, RG)
    mirrors = (deque(), deque(), deque())   # 0: KD, 1: KD2, 2: RG
    numcnt = ({}, {}, {})                   # aparitii per numar in fiecare fereastra
    paircnt = ({}, {})                      # contoare perechi pentru ferestre 0 si 1
    ewma = {}
    st_idx = None
    st_len = 0
    st_rw = None
    st_obj = None
    cum_n = 0
    cum = {}

    def _pairs_add(wi, s):
        d = paircnt[wi]
        ns = sorted(s)
        L = len(ns)
        for i in range(L):
            a = ns[i]
            for j in range(i + 1, L):
                k = (a, ns[j])
                d[k] = d.get(k, 0) + 1

    def _pairs_sub(wi, s):
        d = paircnt[wi]
        ns = sorted(s)
        L = len(ns)
        for i in range(L):
            a = ns[i]
            for j in range(i + 1, L):
                k = (a, ns[j])
                c = d.get(k, 0) - 1
                if c > 0:
                    d[k] = c
                elif k in d:
                    del d[k]

    def _win_add(wi, s):
        nc = numcnt[wi]
        for n in s:
            nc[n] = nc.get(n, 0) + 1
        mirrors[wi].append(s)
        if wi < 2:
            _pairs_add(wi, s)

    def _win_trim(wi):
        mir = mirrors[wi]
        while len(mir) > sizes[wi]:
            old = mir.popleft()
            nc = numcnt[wi]
            for n in old:
                c = nc.get(n, 0) - 1
                if c > 0:
                    nc[n] = c
                elif n in nc:
                    del nc[n]
            if wi < 2:
                _pairs_sub(wi, old)

    def _ewma_reset():
        for n in range(1, P + 1):
            ewma[n] = mu_ew

    def _ewma_step(s):
        for n in ewma:
            ewma[n] *= decay
        for n in s:
            ewma[n] += 1.0

    def _rebuild(recent_deque):
        for mir in mirrors:
            mir.clear()
        for nc in numcnt:
            nc.clear()
        for d in paircnt:
            d.clear()
        _ewma_reset()
        dq = list(recent_deque)
        L = len(dq)
        for s in dq:
            _ewma_step(s)
        for wi in (0, 1, 2):
            sz = min(sizes[wi], L)
            if sz > 0:
                for i in range(L - sz, L):
                    _win_add(wi, dq[i])

    def _stats(d):
        vals = list(d.values())
        m = len(vals)
        if m == 0:
            return {'avg': 0.0, 'min': 0.0, 'max': 0.0}
        tot = 0.0
        mn = vals[0]
        mx = vals[0]
        for v in vals:
            tot += v
            if v < mn:
                mn = v
            if v > mx:
                mx = v
        return {'avg': tot / m, 'min': mn, 'max': mx}

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        nonlocal st_idx, st_len, st_rw, st_obj, cum_n
        dq_len = len(recent_deque)
        s_top = recent_deque[-1] if dq_len > 0 else None
        incremental = (
            dq_len > 0
            and st_idx is not None
            and draw_index == st_idx + 1
            and st_rw == recent_window
            and dq_len == min(recent_window, st_len + 1)
            and (st_obj is None or s_top is not st_obj)
        )
        if incremental:
            _ewma_step(s_top)
            for wi in (0, 1, 2):
                _win_add(wi, s_top)
                _win_trim(wi)
        else:
            _rebuild(recent_deque)
        st_idx = draw_index
        st_len = dq_len
        st_rw = recent_window
        st_obj = s_top

        c1, c2, cR = numcnt
        W1 = len(mirrors[0]) or 1
        W2 = len(mirrors[1]) or 1
        WR = len(mirrors[2]) or 1

        nums = range(1, P + 1)

        z_ew = {}
        for n in nums:
            z_ew[n] = (ewma[n] - mu_ew) / sd_ew

        H1 = sorted(nums, key=lambda n: (-ewma[n], n))[:HOT_K]
        H2 = sorted(nums, key=lambda n: (-c2.get(n, 0), n))[:HOT_K]
        h1set = set(H1)
        h2set = set(H2)

        muR = WR * p
        sdR = math.sqrt(max(1e-12, WR * p * (1.0 - p)))

        pc1 = paircnt[0]
        pc2 = paircnt[1]
        S1 = 0
        for m in H1:
            S1 += c1.get(m, 0)
        S2 = 0
        for m in H2:
            S2 += c2.get(m, 0)

        z_l1 = {}
        z_l2 = {}
        z_rep = {}
        aff1_hot = 0
        exp1_hot = 0.0
        fw1 = float(W1)
        fw2 = float(W2)
        for n in nums:
            cn1 = c1.get(n, 0)
            rest1 = S1 - (cn1 if n in h1set else 0)
            exp1 = cn1 * rest1 / fw1
            aff = 0
            for m in H1:
                if m == n:
                    continue
                k = (n, m) if n < m else (m, n)
                v = pc1.get(k)
                if v is not None:
                    aff += v
            z_l1[n] = (aff - exp1) / math.sqrt(exp1 + 1.0)
            if n in h1set:
                aff1_hot += aff
                exp1_hot += exp1

            cn2 = c2.get(n, 0)
            rest2 = S2 - (cn2 if n in h2set else 0)
            exp2 = cn2 * rest2 / fw2
            aff = 0
            for m in H2:
                if m == n:
                    continue
                k = (n, m) if n < m else (m, n)
                v = pc2.get(k)
                if v is not None:
                    aff += v
            z_l2[n] = (aff - exp2) / math.sqrt(exp2 + 1.0)

            z_rep[n] = (cR.get(n, 0) - muR) / sdR

        gate = 1.0
        if exp1_hot > 0.0:
            ratio = aff1_hot / exp1_hot
            exc = ratio - 1.0
            if exc > 0.0:
                gate = 1.0 + LIFT_SIGN * GATE_K * exc
            if gate < 0.25:
                gate = 0.25
            gcap = 1.0 + 2.0 * abs(GATE_K)
            if gate > gcap:
                gate = gcap

        cL1 = {}
        cL2 = {}
        cREP = {}
        cEW = {}
        scores = {}
        for n in nums:
            a1 = LIFT_SIGN * W_L1 * z_l1[n]
            a2 = LIFT_SIGN * W_L2 * z_l2[n]
            ar = gate * W_REP * z_rep[n]
            ae = gate * W_EW * z_ew[n]
            cL1[n] = a1
            cL2[n] = a2
            cREP[n] = ar
            cEW[n] = ae
            scores[n] = a1 + a2 + ar + ae

        order = sorted(nums, key=lambda n: (-scores[n], n))

        tot = 0.0
        for n in nums:
            tot += ewma[n]
        H = 0.0
        if tot > 0:
            for n in nums:
                q = ewma[n] / tot
                if q > 0:
                    H -= q * math.log(q)
            H /= lnP

        diag = {
            'draw_index': draw_index,
            'L1': _stats(cL1),
            'L2': _stats(cL2),
            'REP': _stats(cREP),
            'EW': _stats(cEW),
            'avg_gate': gate,
            'avg_H': H,
        }
        rank._diag = diag
        cum_n += 1
        for key in ('L1', 'L2', 'REP', 'EW'):
            d = diag[key]
            for st in ('avg', 'min', 'max'):
                k2 = key + '.' + st
                cum[k2] = cum.get(k2, 0.0) + d[st]
        cum['avg_gate'] = cum.get('avg_gate', 0.0) + gate
        cum['avg_H'] = cum.get('avg_H', 0.0) + H
        cd = {'n': cum_n}
        for k2, v in cum.items():
            cd[k2] = v / cum_n
        rank._diag_cum = cd

        return order

    rank.PARAMS = dict(PARAMS)
    rank._diag = None
    rank._diag_cum = None
    return rank