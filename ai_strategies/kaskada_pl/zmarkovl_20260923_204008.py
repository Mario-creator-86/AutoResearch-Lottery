import numpy as np

NAME = "zmarkovl"
DESCRIPTION = "Linear blend of z-standardized signals: Markov co-occurrence lift with the latest draw, EWMA momentum of recent frequencies, overdue gap pressure, recent-window deviation and all-history frequency deviation."
SOURCE = "glm-5.3-flash:cloud + 2026-09-23"

def create(config):
    R = int(config.number_range)
    p = float(config.numbers_drawn) / float(R) if R > 0 else 0.5

    DECAY = 0.94
    W_MARKOV = 1.0
    W_EWMA = 0.60
    W_OVERDUE = 0.45
    W_RECENT = 0.35
    W_TOTAL = 0.25
    GAP_CAP = 8.0

    def _z(v):
        a = np.asarray(v, dtype=np.float64)
        s = float(a.std())
        if s < 1e-12:
            return np.zeros_like(a)
        return (a - float(a.mean())) / s

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, R + 1))
        draws = list(recent_deque)
        if recent_window and len(draws) > recent_window:
            draws = draws[-recent_window:]
        W = len(draws)

        c = np.array([float(counts.get(n, 0)) for n in numbers], dtype=np.float64)
        z_total = _z(c)

        if W == 0:
            order = np.argsort(-z_total, kind="stable")
            return [numbers[i] for i in order]

        bms = np.zeros(W, dtype=np.int64)
        for i, S in enumerate(draws):
            b = 0
            for n in S:
                if 1 <= n <= R:
                    b |= 1 << (n - 1)
            bms[i] = b
        shift = np.arange(R, dtype=np.int64)
        X = ((bms[:, None] >> shift[None, :]) & 1).astype(np.float64)

        rc = X.sum(axis=0)
        z_recent = _z(rc)

        ages = np.arange(W - 1, -1, -1, dtype=np.float64)
        wts = DECAY ** ages
        ew = wts @ X
        var = p * (1.0 - p) * float(np.dot(wts, wts))
        z_ewma = (ew - p * float(wts.sum())) / (np.sqrt(var) + 1e-12)

        P = X.T @ X
        latest = X[-1]
        sL = float((rc * latest).sum())
        co = P @ latest
        lift = co / (rc * (sL / float(W)) + 1.0)
        z_markov = _z(lift)

        gaps = np.array([float(draw_index - last_seen.get(n, -1)) for n in numbers], dtype=np.float64)
        z_overdue = _z(np.minimum(gaps, GAP_CAP) / GAP_CAP)

        score = (W_MARKOV * z_markov + W_EWMA * z_ewma + W_OVERDUE * z_overdue
                 + W_RECENT * z_recent + W_TOTAL * z_total)
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

        order = np.argsort(-score, kind="stable")
        return [numbers[i] for i in order]

    return rank