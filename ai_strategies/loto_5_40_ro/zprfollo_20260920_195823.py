NAME = "zprfollo"
DESCRIPTION = 'PageRank personalizat pe graful de co-aparitie al numerelor din fereastra recenta: random walk cu restart, personalizare = ultima extragere + EWMA + ancora lifetime; scor final = combinatie de z-scoreuri (graf, EWMA, lifetime), sortare determinista.'
SOURCE = 'glm-5.3-flash | 2026-09-20 | personalized PageRank pe co-occurrence graph'

import numpy as np

PARAMS = {
    'W_PR': 1.0,
    'W_EW': 0.5,
    'W_LIFE': 0.35,
    'LAM_LAST': 0.35,
    'EW_DECAY': 0.94,
    'PR_ALPHA': 0.85,
    'PR_ITERS': 64,
}


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)

    def _z(v):
        v = np.asarray(v, dtype=float)
        s = v.std()
        if s <= 1e-12:
            return np.zeros_like(v)
        return (v - v.mean()) / s

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        P = PARAMS
        numbers = list(range(1, N + 1))
        draws = list(recent_deque)
        rw = int(recent_window) if recent_window else 0
        if rw and len(draws) > rw:
            draws = draws[-rw:]
        T = len(draws)

        # graf de co-aparitie: muchie intre numerele extrase impreuna
        A = np.zeros((N, N), dtype=float)
        for s in draws:
            idx = [n - 1 for n in s if 1 <= n <= N]
            if len(idx) >= 2:
                ii = np.asarray(idx, dtype=int)
                A[np.ix_(ii, ii)] += 1.0
        rs = A.sum(axis=1, keepdims=True)
        Ah = np.divide(A, rs, out=np.zeros_like(A), where=rs > 1e-12)
        AhT = Ah.T.copy()

        # personalizare: restart din ultima extragere + EWMA + ancora lifetime
        lam = float(P['LAM_LAST'])
        p = np.zeros(N, dtype=float)
        if draws:
            lidx = [n - 1 for n in draws[-1] if 1 <= n <= N]
            if lidx:
                p[np.asarray(lidx, dtype=int)] += lam / len(lidx)

        z_ew = np.zeros(N, dtype=float)
        if T > 0:
            X = np.zeros((T, N), dtype=float)
            for t, s in enumerate(draws):
                for n in s:
                    if 1 <= n <= N:
                        X[t, n - 1] = 1.0
            w = float(P['EW_DECAY']) ** np.arange(T - 1, -1, -1)
            ew = w @ X
            if ew.sum() > 1e-12:
                p += (1.0 - lam) * 0.75 * (ew / ew.sum())
            if w.sum() > 1e-12:
                z_ew = _z(ew / w.sum())

        life = np.array([counts.get(n, 0) for n in numbers], dtype=float)
        if life.sum() > 1e-12:
            p += (1.0 - lam) * 0.25 * (life / life.sum())
        z_life = _z(life)

        ps = p.sum()
        p = p / ps if ps > 1e-12 else np.full(N, 1.0 / N)

        alpha = float(P['PR_ALPHA'])
        pr = p.copy()
        for _ in range(int(P['PR_ITERS'])):
            pr = alpha * (AhT @ pr) + (1.0 - alpha) * p
        z_pr = _z(pr)

        scores = float(P['W_PR']) * z_pr + float(P['W_EW']) * z_ew + float(P['W_LIFE']) * z_life

        order = sorted(range(N), key=lambda i: (-scores[i], numbers[i]))
        ranked = [numbers[i] for i in order]

        rank._diag = {
            'contrib': {
                'pr': float(np.abs(float(P['W_PR']) * z_pr).mean()),
                'ew': float(np.abs(float(P['W_EW']) * z_ew).mean()),
                'life': float(np.abs(float(P['W_LIFE']) * z_life).mean()),
            },
            'z_pr_top_mean': float(z_pr[order[:K]].mean()),
            'z_ew_top_mean': float(z_ew[order[:K]].mean()),
            'edges': int((A > 0).sum() // 2),
            'window_used': T,
        }
        return ranked

    rank._diag = {}
    return rank