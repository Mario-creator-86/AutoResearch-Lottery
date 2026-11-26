import math
import numpy as np

NAME = "zzprcicl"
DESCRIPTION = ('PageRank personalizat pe graful de co-occcurente al extragerilor recente '
               '(ponderi exponentiale, teleport pe frecventa decay-uita), combinat neliniar '
               'in log-spasiu cu faza ciclului personal de recurenta si cu un tilt mic '
               'spre frecventa pe termen lung.')
SOURCE = 'spectral centrality (power iteration) + gap-cycle phase; 2026-09-26'


def create(config):
    # config: LotteryConfig - folosesc doar atributele, nu iterez peste config
    N = int(config.number_range)
    K = int(config.numbers_drawn)

    GAMMA = 0.93      # decay exponential pe vechimea draw-ului in graf
    WIN = 60          # cate extrageri intru in graf
    DAMP = 0.85       # damping PageRank
    ITERS = 25        # iteratii power method
    S_PR = 0.55       # pondere z PageRank
    S_CYC = 0.50      # pondere z ciclu personal
    S_FREQ = 0.12     # pondere z frecventa lunga
    SIGMA = 0.55      # toleranta fazei ciclului

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        idx = {n: i for i, n in enumerate(numbers)}

        draws = list(recent_deque)
        if len(draws) > WIN:
            draws = draws[-WIN:]

        # 1) graf de co-occcurente cu decay exponential; diagonala = aparitii ponderate
        A = np.zeros((N, N), dtype=np.float64)
        w = 1.0
        for d in reversed(draws):  # deque: cel mai recent e ultimul
            ids = list({idx[n] for n in d if n in idx})
            if ids:
                A[np.ix_(ids, ids)] += w
            w *= GAMMA

        diag = np.diag(A).copy()
        tot = diag.sum()
        tele = diag / tot if tot > 0 else np.full(N, 1.0 / N)

        # 2) PageRank personalizat: v = (1-d)*tele + d * P^T v
        B = A.copy()
        np.fill_diagonal(B, 0.0)
        deg = B.sum(axis=1)
        Prow = B / np.maximum(deg, 1e-12)[:, None]
        v = tele.copy()
        for _ in range(ITERS):
            v = (1.0 - DAMP) * tele + DAMP * (Prow.T @ v)
            s = v.sum()
            if s > 0:
                v /= s
        cent = v

        # 3) faza ciclului personal de recurenta
        T = len(draws)
        memb = np.zeros((T, N), dtype=bool)
        for t in range(T):
            for n in draws[t]:
                j = idx.get(n)
                if j is not None:
                    memb[t, j] = True
        base_gap = K / float(N)
        fit = np.zeros(N, dtype=np.float64)
        for i, n in enumerate(numbers):
            ls = last_seen.get(n)
            if ls is None:
                fit[i] = 0.35
                continue
            gap = draw_index - ls
            if gap < 0:
                gap = 0
            rows = np.nonzero(memb[:, i])[0]
            if len(rows) >= 2:
                g = float(np.diff(rows).mean())
            else:
                cnt = counts.get(n, 0)
                g = (draw_index + 1.0) / cnt if cnt > 0 else base_gap
            if g < 1.0:
                g = 1.0
            phase = gap / g
            fit[i] = math.exp(-((phase - 1.0) * (phase - 1.0)) / (2.0 * SIGMA * SIGMA))

        # 4) combinare neliniara: suma ponderata de z-score-uri, sortata descrescator
        arr = np.array([counts.get(n, 0) for n in numbers], dtype=np.float64)
        z_fr = (arr - arr.mean()) / (arr.std() + 1e-9)
        z_pr = (cent - cent.mean()) / (cent.std() + 1e-12)
        z_cy = (fit - fit.mean()) / (fit.std() + 1e-12)
        score = S_PR * z_pr + S_CYC * z_cy + S_FREQ * z_fr

        order = sorted(range(N), key=lambda i: (-score[i], numbers[i]))
        return [numbers[i] for i in order]

    return rank