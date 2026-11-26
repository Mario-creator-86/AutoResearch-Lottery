import numpy as np

NAME = "zppraffi"
DESCRIPTION = ('Graf de co-aparitii + Personalized PageRank: din fereastra recenta, cu ponderi '
               'exp-decay (half-life 20 extrageri), construieste o matrice de afinitate pe perechi, '
               'o amesteca cu un floor uniform si ruleaza PPR exact (solve liniar). Vectorul de '
               'restart amesteca frecventa EWMA (0.66), rampa overdue (0.20) si deviatia-z fata de '
               'frecventa globala (0.14). Un gate de entropie adapteaza damping-ul alpha.')
SOURCE = 'model + 2026-09-07'


def create(config):
    N = int(config.number_range)
    K = max(int(config.numbers_drawn), 1)
    DECAY = 0.5 ** (1.0 / 20.0)
    LAM = 0.15
    W_F, W_O, W_Z = 0.66, 0.20, 0.14
    OV_START, OV_SPAN = 0.6, 2.5

    def _ls(d, n):
        x = d.get(n, -1)
        return -1 if x is None else x

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        draws = [s for s in recent_deque if s]
        L = len(draws)

        # detecteaza capatul 'cel mai recent' al deque-ului folosind last_seen
        li = draw_index - 1
        last_draw = {n for n in numbers if _ls(last_seen, n) == li} if draw_index > 0 else set()
        newest_right = True
        if L >= 2 and last_draw:
            if len(draws[0] & last_draw) > len(draws[-1] & last_draw):
                newest_right = False
        if not newest_right:
            draws = draws[::-1]
        try:
            rw = int(recent_window) if recent_window else 0
        except (TypeError, ValueError):
            rw = 0
        if rw > 0 and L > rw:
            draws = draws[-rw:] if newest_right else draws[:rw]
            L = len(draws)

        # ponderi temporale: cel mai vechi -> DECAY^(L-1), cel mai nou -> 1
        w = DECAY ** np.arange(L - 1, -1, -1, dtype=np.float64)
        X = np.zeros((L, N), dtype=np.float64)
        rows = [list(s) for s in draws]
        if L and len({len(t) for t in rows}) == 1:
            arr = np.clip(np.asarray(rows, dtype=np.intp) - 1, 0, N - 1)
            X[np.arange(L)[:, None], arr] = 1.0
        else:
            for d, t in enumerate(rows):
                if t:
                    X[d, np.clip(np.asarray(t, dtype=np.intp) - 1, 0, N - 1)] = 1.0

        f = w @ X
        fsum = float(f.sum())
        fhat = f / fsum if fsum > 1e-12 else np.full(N, 1.0 / N)

        C = (X * w[:, None]).T @ X
        np.fill_diagonal(C, 0.0)
        rowS = C.sum(axis=1)
        Chat = np.divide(C, rowS[:, None], out=np.zeros_like(C), where=rowS[:, None] > 1e-12)
        P = (1.0 - LAM) * Chat + (LAM / N)
        dead = rowS <= 1e-12
        if dead.any():
            P[dead, :] = 1.0 / N

        # gate de entropie pe distributia EWMA
        p = np.clip(fhat, 1e-12, None)
        Hn = float(-(p * np.log(p)).sum() / np.log(N))
        Hn = min(max(Hn, 0.0), 1.0)
        alpha = 0.78 + 0.15 * (1.0 - Hn)
        if L < 12:
            alpha = min(alpha, 0.80)

        # componentele vectorului de restart
        E = N / float(K)
        ls = np.array([_ls(last_seen, n) for n in numbers], dtype=np.float64)
        gap = (draw_index - 1) - ls
        u = np.clip((gap - OV_START * E) / (OV_SPAN * E), 0.0, 1.0)
        usum = float(u.sum())
        uhat = u / usum if usum > 1e-12 else np.full(N, 1.0 / N)

        cv = np.array([(counts.get(n, 0) or 0) for n in numbers], dtype=np.float64)
        sd = float(cv.std())
        if sd > 1e-12:
            z = np.clip((cv - cv.mean()) / sd, -2.0, 2.0)
            zr = np.exp(0.4 * z)
            zhat = zr / zr.sum()
        else:
            zhat = np.full(N, 1.0 / N)

        v = W_F * fhat + W_O * uhat + W_Z * zhat
        v = v / v.sum()

        # PPR exact: (I - alpha*P^T) r = (1-alpha) v
        PT = np.ascontiguousarray(P.T)
        try:
            r = np.linalg.solve(np.eye(N) - alpha * PT, (1.0 - alpha) * v)
        except np.linalg.LinAlgError:
            r = v.copy()
            for _ in range(200):
                r = alpha * (PT @ r) + (1.0 - alpha) * v

        scores = {numbers[i]: float(r[i]) for i in range(N)}
        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank