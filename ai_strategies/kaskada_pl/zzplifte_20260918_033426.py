import math
import numpy as np

NAME = "zzplifte"
DESCRIPTION = ('Momentum pur, flip complet fata de strategia reversionista precedenta: pair-lift '
               'per numar fata de membrii extragerii D1 si D2 (co-occurinte in fereastra recenta), '
               'apartenenta la D1 (REP), EWMA pe frecvente (KD) si frecventa istorica (CNT); '
               'componentele de momentum sunt modulate de un gate de regim de persistenta '
               '(overlap mediu intre extrageri consecutive vs 6 asteptat).')
SOURCE = 'model + 2026-09-18'

W_L1 = 2.2
W_L2 = 1.1
W_REP = 1.5
W_EW = 0.35
W_CNT = 0.5
KD = 20
GATE_K = 0.75
GATE_MIN = 0.25
GATE_MAX = 2.5


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    T = int(config.ticket_size)

    p_pair = (K / N) * ((K - 1.0) / (N - 1.0)) if N > 1 else 0.0
    pr = K / float(N)
    exp_ov = K * K / float(N)
    var_ov = N * pr * pr * (1.0 - pr * pr)
    std_ov = math.sqrt(var_ov) if var_ov > 0 else 1.0
    alpha = 1.0 / KD

    diag = {
        'calls': 0,
        'span_mean': {'lift1': 0.0, 'lift2': 0.0, 'rep': 0.0, 'ew': 0.0, 'cnt': 0.0, 'total': 0.0},
        'gate_mean': 0.0,
        'gate_min': float('inf'),
        'gate_max': float('-inf'),
        'tickets': [],
        'params': {'W_L1': W_L1, 'W_L2': W_L2, 'W_REP': W_REP, 'W_EW': W_EW,
                   'W_CNT': W_CNT, 'KD': KD, 'GATE_K': GATE_K},
    }

    def _z(v):
        arr = np.asarray(v, dtype=np.float64)
        s = arr.std()
        if s < 1e-12:
            return np.zeros_like(arr)
        return (arr - arr.mean()) / s

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        if counts is None:
            counts = {}
        if last_seen is None:
            last_seen = {}
        deque_len = len(recent_deque) if recent_deque is not None else 0
        numbers = list(range(1, N + 1))

        d1 = [n for n in numbers if last_seen.get(n) == draw_index - 1]
        d2 = [n for n in numbers if last_seen.get(n) == draw_index - 2]
        d1set = set(d1)

        lift1 = np.zeros(N)
        lift2 = np.zeros(N)
        ew = np.zeros(N)
        gate = 1.0

        if deque_len > 0:
            M = np.zeros((deque_len, N), dtype=np.float64)
            for i, d in enumerate(recent_deque):
                row = [n - 1 for n in d if 1 <= n <= N]
                if row:
                    M[i, row] = 1.0
            if d1set and set(recent_deque[0]) == d1set:
                M = M[::-1].copy()

            if W_EW > 0:
                w = np.power(1.0 - alpha, np.arange(deque_len - 1, -1, -1))
                ew = (M * w[:, None]).sum(axis=0) / max(float(w.sum()), 1e-12)

            if deque_len >= 2:
                ov = (M[:-1] * M[1:]).sum(axis=1)
                m = ov.shape[0]
                zg = (float(ov.mean()) - exp_ov) / (std_ov / math.sqrt(m))
                zg = max(-3.0, min(3.0, zg))
                gate = max(GATE_MIN, min(GATE_MAX, 1.0 + GATE_K * zg))

            E = deque_len * p_pair
            sd = math.sqrt(max(deque_len * p_pair * (1.0 - p_pair), 1e-12))

            if W_L1 > 0 and d1:
                idx = [n - 1 for n in d1]
                Z = M[:, idx].T @ M
                Zs = (Z - E) / sd
                lift1 = _z(Zs.sum(axis=0) - Zs[np.arange(len(idx)), idx])

            if W_L2 > 0 and d2:
                idx2 = [n - 1 for n in d2]
                Z2 = M[:, idx2].T @ M
                Zs2 = (Z2 - E) / sd
                lift2 = _z(Zs2.sum(axis=0) - Zs2[np.arange(len(idx2)), idx2])

        rep = np.array([1.0 if n in d1set else 0.0 for n in numbers])
        cnt = np.array([float(counts.get(n, 0)) for n in numbers])

        zrep = _z(rep) if W_REP > 0 else np.zeros(N)
        zew = _z(ew) if W_EW > 0 else np.zeros(N)
        zcnt = _z(cnt) if W_CNT > 0 else np.zeros(N)

        score = gate * (W_L1 * lift1 + W_L2 * lift2 + W_REP * zrep) + W_EW * zew + W_CNT * zcnt

        order = sorted(range(N), key=lambda i: (-score[i], i))
        perm = [numbers[i] for i in order]

        c = diag['calls']
        spans = {
            'lift1': float(lift1.max() - lift1.min()),
            'lift2': float(lift2.max() - lift2.min()),
            'rep': float(zrep.max() - zrep.min()),
            'ew': float(zew.max() - zew.min()),
            'cnt': float(zcnt.max() - zcnt.min()),
            'total': float(score.max() - score.min()),
        }
        for k, s in spans.items():
            diag['span_mean'][k] = (diag['span_mean'][k] * c + s) / (c + 1)
        diag['gate_mean'] = (diag['gate_mean'] * c + gate) / (c + 1)
        if gate < diag['gate_min']:
            diag['gate_min'] = gate
        if gate > diag['gate_max']:
            diag['gate_max'] = gate
        diag['calls'] = c + 1
        tk = diag['tickets']
        tk.append((int(draw_index), tuple(perm[:T])))
        if len(tk) > 300:
            del tk[:-300]

        return perm

    rank._diag = diag
    return rank