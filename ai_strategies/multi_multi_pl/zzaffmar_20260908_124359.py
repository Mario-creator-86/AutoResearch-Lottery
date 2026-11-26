import numpy as np

NAME = "zzaffmar"
DESCRIPTION = 'Decayed co-occurrence graph with Markov message passing: EWMA appearance rate from the diagonal of a decayed pair matrix, one- and two-step affinity from the last draws via row-normalized co-occurrence transitions, plus a clipped overdue-gap z-score and a tiny long-run frequency z-score.'
SOURCE = 'glm-5.3-flash strategist, 2026-09-08'

def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)

    DECAY = 0.9975
    W_RATE = 1.00
    W_AFF1 = 0.85
    W_AFF2 = 0.40
    W_GAP = 0.10
    W_LONG = 0.05
    p = K / float(N)
    GAP_MEAN = 1.0 / p
    GAP_SD = ((1.0 - p) ** 0.5) / p

    M = np.zeros((N, N), dtype=np.float64)
    seen = set()
    Tcache = None
    dirty = True

    def _z(x):
        return (x - x.mean()) / (x.std() + 1e-12)

    def _ingest(fs):
        nonlocal dirty
        idx = np.fromiter((x - 1 for x in fs), dtype=np.intp, count=len(fs))
        M *= DECAY
        M[np.ix_(idx, idx)] += 1.0
        dirty = True

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        nonlocal Tcache, dirty
        n_items = len(recent_deque)
        if n_items and frozenset(recent_deque[-1]) not in seen:
            new_list = []
            i = n_items - 1
            while i >= 0:
                fs = frozenset(recent_deque[i])
                if fs in seen:
                    break
                seen.add(fs)
                new_list.append(fs)
                i -= 1
            for fs in reversed(new_list):
                _ingest(fs)
        if dirty or Tcache is None:
            T = M.copy()
            np.fill_diagonal(T, 0.0)
            rs = T.sum(axis=1)
            rs[rs <= 1e-9] = 1.0
            T /= rs[:, None]
            Tcache = T
            dirty = False
        T = Tcache

        diag = np.diagonal(M)
        rate_z = _z(diag) if diag.sum() > 0 else np.zeros(N)

        v = np.zeros(N, dtype=np.float64)
        if n_items:
            for x in recent_deque[-1]:
                v[x - 1] += 0.65
            if n_items >= 2:
                for x in recent_deque[-2]:
                    v[x - 1] += 0.35
        vs = v.sum()
        if vs > 0:
            v /= vs

        a1 = v @ T
        a2 = a1 @ T
        aff1_z = _z(a1)
        aff2_z = _z(a2)

        gz = np.zeros(N, dtype=np.float64)
        for n in range(1, N + 1):
            ls = last_seen.get(n)
            if ls is not None:
                z = (draw_index - ls - GAP_MEAN) / GAP_SD
                gz[n - 1] = 3.0 if z > 3.0 else (-2.0 if z < -2.0 else z)

        cvec = np.fromiter((counts.get(n, 0) for n in range(1, N + 1)), dtype=np.float64, count=N)
        long_z = _z(cvec) if cvec.sum() > 0 else np.zeros(N)

        score = (W_RATE * rate_z + W_AFF1 * aff1_z + W_AFF2 * aff2_z
                 + W_GAP * gz + W_LONG * long_z)
        return sorted(range(1, N + 1), key=lambda n: (-score[n - 1], n))

    return rank