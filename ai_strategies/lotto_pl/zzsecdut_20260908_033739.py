import math
from itertools import islice
import numpy as np

NAME = "zzsecdut"
DESCRIPTION = 'Fuziune Borda pe percentile a 4 semnale: z-score global de frecventa (dominant), z-score al ferestrei recente, eco de co-aparitie cu numerele ultimei extrageri si datoria de intarziere pe sectoare de 7 numere; bonus de concordanta r_all*r_win.'
SOURCE = 'model + 2026-09-08'


def create(config):
    K = int(getattr(config, 'number_range', 49) or 49)
    WD = int(getattr(config, 'numbers_drawn', 6) or 6)
    if K < 1:
        K = 49
    if WD < 1 or WD > K:
        WD = max(1, min(6, K))
    p = WD / float(K)
    MAXW = 600

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        if counts is None:
            counts = {}
        numbers = list(range(1, K + 1))
        n = len(numbers)
        idx = np.arange(1, n + 1, dtype=np.float64)
        c = np.fromiter((float(counts.get(v, 0)) for v in numbers), dtype=np.float64, count=n)

        # 1) frecventa globala: z-score pe tot istoricul disponibil (N = sum(counts)/WD)
        total = float(c.sum())
        N = total / float(WD) if WD > 0 else 0.0
        var = N * p * (1.0 - p)
        sd = math.sqrt(var) if var > 0 else 1.0
        z_all = (c - N * p) / sd

        # 2) fereastra recenta: conteuri (agnostic la orientarea deque-ului)
        wc = np.zeros(n, dtype=np.float64)
        win_sets = []
        w_used = 0
        if recent_deque is not None:
            src = recent_deque
            try:
                if len(recent_deque) > MAXW:
                    src = islice(reversed(recent_deque), MAXW)
            except TypeError:
                src = recent_deque
            for s in src:
                try:
                    st = set(s)
                except TypeError:
                    st = set()
                if st:
                    win_sets.append(st)
                    for v in st:
                        if 1 <= v <= n:
                            wc[v - 1] += 1.0
                    w_used += 1
        den_w = math.sqrt(max(w_used, 1) * p * (1.0 - p))
        if den_w <= 0:
            den_w = 1.0
        z_win = (wc - w_used * p) / den_w

        # ultima extragere = multimea cu gap minim (0 sau 1, robust la conventia last_seen)
        di = int(draw_index) if draw_index is not None else 0
        gaps = {}
        min_gap = None
        if last_seen:
            for v in numbers:
                ls = last_seen.get(v)
                if ls is None:
                    continue
                try:
                    g = di - int(ls)
                except (TypeError, ValueError):
                    continue
                if 0 <= g <= 1000000000:
                    gaps[v] = g
                    if min_gap is None or g < min_gap:
                        min_gap = g
        last_draw = set(v for v, g in gaps.items() if g == min_gap) if min_gap is not None else set()

        # 3) eco de co-aparitie: numere aparute alaturi de cele din ultima extragere
        echo = np.zeros(n, dtype=np.float64)
        if last_draw:
            for st in win_sets:
                ov = len(st & last_draw)
                if ov:
                    boost = 1.0 + 0.25 * (ov - 1)
                    for v in st:
                        if v not in last_draw and 1 <= v <= n:
                            echo[v - 1] += boost

        # 4) datoria de intarziere pe sectoare de cate 7 numere
        sec = max(1, n // 7)
        nsec = (n + sec - 1) // sec
        obs = np.zeros(nsec, dtype=np.float64)
        for i in range(n):
            obs[i // sec] += wc[i]
        exp_s = w_used * p * sec
        den_s = math.sqrt(exp_s) if exp_s > 0 else 1.0
        debt_s = np.tanh((exp_s - obs) / den_s)
        debt = debt_s[np.arange(n) // sec]

        # fuziune: percentile ordinale deterministe + bonus de concordanta
        def pct(x):
            x = np.asarray(x, dtype=np.float64)
            if n > 1 and float(x.max() - x.min()) > 0.0:
                order = np.lexsort((idx, x))
                rk = np.empty(n, dtype=np.float64)
                rk[order] = np.arange(n, dtype=np.float64)
                rk /= float(n - 1)
                return rk
            return np.full(n, 0.5, dtype=np.float64)

        r_all = pct(z_all)
        r_win = pct(z_win)
        r_echo = pct(echo)
        r_debt = pct(debt)
        score = 1.20 * r_all + 0.40 * r_win + 0.35 * r_echo + 0.30 * r_debt + 0.30 * (r_all * r_win)

        order = np.lexsort((idx, -score))
        return [numbers[i] for i in order]

    return rank