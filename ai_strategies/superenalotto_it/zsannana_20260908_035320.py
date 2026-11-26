import numpy as np

NAME = "zsannana"
DESCRIPTION = ('Self-analog k-nearest-neighbour forecasting: slides over the recent window, '
               'finds the historical draw-blocks most similar to the current multi-draw pattern '
               '(recency-weighted overlap) and scores each number by its gamma-decayed appearance '
               'rate in the draws that FOLLOWED those analog blocks. Two block scales (L=2, L=5) '
               'are fused with a nonlinear consensus bonus, plus a mild long-run binomial-z anchor.')
SOURCE = 'model + 2026-09-08'


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    P_BASE = float(K) / float(N)

    def _zclip(x):
        x = np.asarray(x, dtype=float)
        sd = x.std()
        if sd < 1e-12:
            return np.zeros_like(x)
        return np.clip((x - x.mean()) / sd, -2.5, 2.5)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        try:
            seq = list(recent_deque)
        except Exception:
            seq = []
        T = len(seq)

        counts_vec = np.array([float(counts.get(n, 0)) for n in numbers])
        denom = float(counts_vec.sum())
        total_draws = max(1.0, denom / float(K)) if denom > 0 else 1.0
        sd_long = np.sqrt(P_BASE * (1.0 - P_BASE) / total_draws)
        if sd_long > 1e-12:
            z_long = np.clip((counts_vec / total_draws - P_BASE) / sd_long, -2.5, 2.5)
        else:
            z_long = np.zeros(N)

        if T < 6:
            order = sorted(range(N), key=lambda i: (-z_long[i], numbers[i]))
            return [numbers[i] for i in order]

        # --- orientarea deque-ului: capatul cu last_seen mai mari este cel mai nou
        def _ls_sum(draw):
            return sum(float(last_seen.get(n, -1)) for n in draw)

        newest_is_last = _ls_sum(seq[-1]) >= _ls_sum(seq[0])
        ordered = seq if newest_is_last else seq[::-1]

        M = np.zeros((T, N), dtype=float)
        for i, s in enumerate(ordered):
            for n in s:
                if 1 <= n <= N:
                    M[i, n - 1] = 1.0

        f_win = M.sum(axis=0) / float(T)
        sd_win = np.sqrt(P_BASE * (1.0 - P_BASE) / float(T))
        if sd_win > 1e-12:
            z_win = np.clip((f_win - P_BASE) / sd_win, -2.5, 2.5)
        else:
            z_win = np.zeros(N)

        if T >= 16:
            scales = [(2, 2, 0.55), (5, 3, 0.45)]
        elif T >= 10:
            scales = [(2, 2, 1.0)]
        else:
            scales = [(1, 2, 1.0)]

        z_scales = []
        for (L, H, wsc) in scales:
            L = min(int(L), T - 1)
            if L < 1:
                continue
            ncand = T - L
            Q = M[T - L: T]
            qw = np.array([0.72 ** (L - 1 - j) for j in range(L)])
            qw /= qw.sum()
            sims = np.zeros(ncand)
            for j in range(L):
                sims += qw[j] * (Q[j] @ M[j: j + ncand].T)
            # rata de aparitie in viitorul fiecarui candidat, cu decay
            F = np.zeros((ncand, N))
            FN = np.zeros(ncand)
            gam = 0.7
            for h in range(H):
                lo = L + h
                if lo >= T:
                    break
                cnt = min(ncand, T - lo)
                F[:cnt] += (gam ** h) * M[lo: lo + cnt]
                FN[:cnt] += gam ** h
            Ktop = min(30, ncand)
            cand = sorted(range(ncand), key=lambda t: (-sims[t], -t))[:Ktop]
            A = np.zeros(N)
            wsum = 0.0
            for t in cand:
                if FN[t] <= 1e-9 or sims[t] <= 1e-9:
                    continue
                A += sims[t] * (F[t] / FN[t])
                wsum += sims[t]
            if wsum <= 1e-9:
                continue
            z_scales.append((wsc, _zclip(A / wsum)))

        if z_scales:
            if len(z_scales) >= 2:
                (w1, z1), (w2, z2) = z_scales[0], z_scales[1]
                z_an = (w1 * z1 + w2 * z2) / (w1 + w2) + 0.25 * np.minimum(z1, z2)
            else:
                z_an = z_scales[0][1]
            score = z_an + 0.30 * z_long + 0.20 * z_win
        else:
            score = z_long + 0.30 * z_win

        order = sorted(range(N), key=lambda i: (-score[i], numbers[i]))
        return [numbers[i] for i in order]

    return rank