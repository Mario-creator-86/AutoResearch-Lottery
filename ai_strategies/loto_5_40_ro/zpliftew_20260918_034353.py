import math
from collections import deque

import numpy as np

NAME = "zpliftew"
DESCRIPTION = (
    'Lift de perechi pe tranzitii lag-1 (ce a urmat istoric dupa numerele din ultima extragere), '
    'lift-chain in 2 pasi si indicator de repetitie lag-1, modulate de un gate de regim dat de '
    'suprapunerea extragerilor consecutive; adauga EWMA pe frecventele recente, deviatia pe termen '
    'lung si un gate de diversitate pe suporteri (demoteaza aglomerarea top-ului pe acelasi sponsor).'
)
SOURCE = 'model + 2026-09-17 (Loto 5/40 RO, plugin)'

# ---- parametri (tinte de grid: W_L1{1.1,2.2,3.3} x W_REP{0,1.5,3.0} x W_EW{0,0.35,0.7} x KD{12,20,30} x RG{4,8,16} x GATE_K{0.4,0.75,1.2}) ----
W_L1 = 2.2
W_L2 = 0.8
W_REP = 1.5
W_EW = 0.35
W_CNT = 0.0
KD = 20
RG = 8
GATE_K = 0.75
SHRINK = 2.0
GATE_LO = 0.25
GATE_HI = 2.5
EW_MAX = 100


class _Perm(list):
    pass


def create(config):
    N = int(config.number_range)
    D = int(config.numbers_drawn)
    p_hit = float(D) / float(N)
    base_ov = float(D * D) / float(N)

    # stare incrementala: matricea de tranzitii lag-1 pe fereastra KD
    st = {'idx': None, 'newest': None,
          'T1': np.zeros((N + 1, N + 1), dtype=np.float64),
          'OUT': np.zeros(N + 1, dtype=np.float64),
          'tr': deque()}
    cum = {'calls': 0,
           'span_sum': {'lift1': 0.0, 'lift2': 0.0, 'rep': 0.0, 'ew': 0.0, 'cnt': 0.0},
           'recent': deque(maxlen=64)}

    def _add_trans(src, tgt, sign):
        s = np.array(sorted(src), dtype=np.intp)
        t = np.array(sorted(tgt), dtype=np.intp)
        st['T1'][s[:, None], t[None, :]] += sign
        st['OUT'][s] += sign
        return (s, t)

    def _drop_trans(pair):
        s, t = pair
        st['T1'][s[:, None], t[None, :]] -= 1.0
        st['OUT'][s] -= 1.0

    def _rebuild(sets):
        st['T1'][:, :] = 0.0
        st['OUT'][:] = 0.0
        st['tr'].clear()
        Lx = len(sets)
        lo = max(0, Lx - 1 - KD)
        for i in range(lo, Lx - 1):
            st['tr'].append(_add_trans(sets[i], sets[i + 1], 1.0))

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        dq = list(recent_deque)
        L = len(dq)
        sets = [s if isinstance(s, (set, frozenset)) else set(s) for s in dq]
        newest = frozenset(sets[-1]) if L >= 1 else None
        prev = frozenset(sets[-2]) if L >= 2 else None

        # update incremental al tranziitiilor (idempotent la apeluri repetate, rebuild la desincronizare)
        if st['idx'] is not None and st['idx'] == draw_index and st['newest'] == newest:
            pass
        elif (st['idx'] is not None and draw_index == st['idx'] + 1
              and prev is not None and st['newest'] == prev):
            st['tr'].append(_add_trans(st['newest'], sets[-1], 1.0))
            if len(st['tr']) > KD:
                _drop_trans(st['tr'].popleft())
            st['idx'] = draw_index
            st['newest'] = newest
        else:
            _rebuild(sets)
            st['idx'] = draw_index
            st['newest'] = newest

        def _emit(order, spans, extra):
            diag = {'draw': draw_index, 'spans': spans,
                    'params': {'W_L1': W_L1, 'W_L2': W_L2, 'W_REP': W_REP, 'W_EW': W_EW,
                               'W_CNT': W_CNT, 'KD': KD, 'RG': RG, 'GATE_K': GATE_K},
                    'top6': list(order[:6])}
            diag.update(extra)
            cum['calls'] += 1
            for k in spans:
                cum['span_sum'][k] += float(spans.get(k, 0.0))
            cum['recent'].append((draw_index, list(order[:6])))
            perm = _Perm(order)
            perm._diag = diag
            rank._diag = diag
            rank._diag_cum = {'calls': cum['calls'],
                              'span_avg': {k: v / float(max(cum['calls'], 1)) for k, v in cum['span_sum'].items()},
                              'recent': list(cum['recent'])}
            return perm

        # fallback istoric prea scurt
        if L < 4 or not newest:
            order = sorted(numbers, key=lambda n: (-counts.get(n, 0), n))
            zero = {'lift1': 0.0, 'lift2': 0.0, 'rep': 0.0, 'ew': 0.0, 'cnt': 0.0}
            return _emit(order, zero, {'mode': 'fallback_counts'})

        rows = np.array(sorted(newest), dtype=np.intp)
        T1 = st['T1']
        OUTv = st['OUT']
        T1m = T1[rows, :]

        # LIFT1: obs vs asteptat pe tranzitii lag-1 cu sursa in D1
        obs1 = T1m.sum(axis=0)
        exp1 = p_hit * float(OUTv[rows].sum())
        lift1 = (obs1 - exp1) / math.sqrt(exp1 + SHRINK)

        # LIFT2: propagare in 2 pasi prin P1 = T1/(OUT+1)
        P1 = T1 / (OUTv[:, None] + 1.0)
        chain = P1[rows, :] @ P1
        s2 = chain.sum(axis=0)
        lift2 = (s2 - len(rows) * p_hit) / math.sqrt(len(rows) * p_hit + SHRINK)

        # REP: indicator D1 centrat la p_hit
        rep = np.zeros(N + 1, dtype=np.float64)
        rep[rows] = 1.0
        rep -= p_hit

        # EWMA pe frecventele recente, standardizata cross-sectional
        ew = np.zeros(N + 1, dtype=np.float64)
        M = min(L, EW_MAX)
        dec = 1.0 - 2.0 / (KD + 1.0)
        Z = 0.0
        w = 1.0
        for i in range(M):
            sset = sets[L - 1 - i]
            for n in sset:
                ew[n] += w
            Z += w
            w *= dec
        ew /= max(Z, 1e-12)
        ewm = float(ew[1:].mean())
        ews = float(ew[1:].std())
        if ews > 1e-9:
            ewz = (ew - ewm) / ews
        else:
            ewz = np.zeros(N + 1, dtype=np.float64)

        # CNT: z pe frecventa istorica (diagnostic, W_CNT=0 default)
        vals = np.array([counts.get(n, 0) for n in range(1, N + 1)], dtype=np.float64)
        mu = float(vals.mean())
        sdv = math.sqrt(max(mu * (1.0 - p_hit), 1e-9))
        cnt = np.zeros(N + 1, dtype=np.float64)
        cnt[1:] = (vals - mu) / sdv

        # gate de regim: suprapunerea medie a extragerilor consecutive vs baza D^2/N
        Kg = min(RG, L - 1)
        if Kg > 0:
            avg_ov = sum(len(sets[-1 - k] & sets[-k]) for k in range(1, Kg + 1)) / float(Kg)
        else:
            avg_ov = base_ov
        gate = min(GATE_HI, max(GATE_LO, avg_ov / base_ov))

        raw = (W_L1 * gate) * lift1 + W_L2 * lift2 + (W_REP * gate) * rep + W_EW * ewz + W_CNT * cnt

        # gate de diversitate pe suporteri: penalizez numerele al carui sponsor principal a mai fost folosit
        c1 = (W_L1 * gate) * lift1
        sd1 = float(c1[1:].std())
        sup_num = rows[np.argmax(T1m, axis=0)]
        has_sup = T1m.max(axis=0) > 0.0

        pass1 = sorted(range(1, N + 1), key=lambda n: (-raw[n], n))
        usage = {}
        fin = np.zeros(N + 1, dtype=np.float64)
        for n in pass1:
            pen = 0.0
            if has_sup[n] and sd1 > 0.0:
                ms = int(sup_num[n])
                u = usage.get(ms, 0)
                pen = GATE_K * sd1 * u
                usage[ms] = u + 1
            fin[n] = raw[n] - pen

        order = sorted(range(1, N + 1), key=lambda n: (-fin[n], n))

        def _span(a):
            return float(a[1:].max() - a[1:].min())

        spans = {'lift1': _span(lift1), 'lift2': _span(lift2), 'rep': _span(rep),
                 'ew': _span(ewz), 'cnt': _span(cnt)}
        contrib = {'lift1': _span(c1), 'lift2': _span(W_L2 * lift2),
                   'rep': _span((W_REP * gate) * rep), 'ew': _span(W_EW * ewz),
                   'cnt': _span(W_CNT * cnt)}
        extra = {'mode': 'full', 'gate': float(gate), 'avg_ov': float(avg_ov),
                 'sd1': sd1, 'contrib': contrib}
        return _emit(order, spans, extra)

    return rank