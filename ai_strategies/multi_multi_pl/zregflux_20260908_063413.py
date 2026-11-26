import math
import numpy as np

NAME = "zregflux"
DESCRIPTION = (
    'Regime-switching ranker: measures short-range persistence via consecutive-draw '
    'overlap vs its hypergeometric expectation, then blends momentum (EWMA frequency '
    '+ co-occurrence pair-field pull) in sticky regimes with gap-based mean reversion '
    'in choppy regimes, over a mild long-run frequency tilt.'
)
SOURCE = 'quant strategist (glm), 2026-09-08'


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    p = K / float(N) if N > 0 else 0.25
    CAP = 120          # max draws used for regime detection / pair field
    H = 40             # momentum horizon (draws)
    HALF_LIFE = 15.0   # EWMA half-life (draws)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        draws = list(recent_deque)
        if len(draws) > CAP:
            draws = draws[-CAP:]
        W = len(draws)

        cnt = [float(counts.get(n, 0)) for n in numbers]
        T = (sum(cnt) / K) if K > 0 else 0.0
        if T <= 0.0:
            T = 1.0
        if W == 0:
            return sorted(numbers, key=lambda n: (-counts.get(n, 0), n))

        M = np.zeros((W, N), dtype=np.float64)
        for r, s in enumerate(draws):
            idx = [n - 1 for n in s if 1 <= n <= N]
            if idx:
                M[r, idx] = 1.0

        # --- regime detector: overlap of consecutive draws ---
        exp_ov = K * p  # = K*K/N
        var1 = K * p * (1.0 - p) * ((N - K) / float(N - 1)) if N > 1 else 1.0
        z_ov = 0.0
        if W >= 2:
            ov = (M[:-1] * M[1:]).sum(axis=1)
            kp = ov.shape[0]
            se = math.sqrt(max(var1, 1e-12) / kp)
            z_ov = (float(ov.mean()) - exp_ov) / se
        reg = math.tanh(z_ov / 2.0)
        mom_w = 0.5 * (1.0 + reg)
        rev_w = 0.5 * (1.0 - reg)

        # --- recency weights over window (newest = age 0) ---
        ages = np.arange(W - 1, -1, -1, dtype=np.float64)
        wts = np.exp(-(math.log(2.0) / HALF_LIFE) * ages)
        wts /= wts.sum()
        f_w = wts @ M

        # --- short-horizon momentum z ---
        Hh = min(H, W)
        f_h = M[-Hh:].mean(axis=0)
        se_h = math.sqrt(max(p * (1.0 - p) / Hh, 1e-12))
        zS = np.clip((f_h - p) / se_h, -4.0, 4.0)

        # --- long-run frequency z (mild tilt) ---
        mu = p * T
        sd = math.sqrt(max(T * p * (1.0 - p), 1e-12))
        zL = np.clip((np.asarray(cnt) - mu) / sd, -4.0, 4.0)

        # --- pair-field pull: co-occurrence with recency-hot partners ---
        Co = M.T @ M
        np.fill_diagonal(Co, 0.0)
        pull = Co @ f_w
        ps = float(pull.std())
        if ps > 1e-9:
            pull_z = np.clip((pull - pull.mean()) / ps, -4.0, 4.0)
        else:
            pull_z = np.zeros(N, dtype=np.float64)

        # --- gap-based due z (geometric: mean 1/p, sd sqrt(1-p)/p) ---
        gap_mean = 1.0 / p
        gap_sd = math.sqrt(max(1.0 - p, 1e-12)) / p
        due = np.empty(N, dtype=np.float64)
        for i, n in enumerate(numbers):
            ls = last_seen.get(n)
            g = (draw_index - ls) if ls is not None else (draw_index + 5)
            due[i] = max(-2.5, min(2.5, (g - gap_mean) / gap_sd))

        score = (0.30 * zL
                 + mom_w * (1.00 * zS + 0.60 * pull_z)
                 + rev_w * 0.80 * due)

        order = sorted(range(N), key=lambda i: (-float(score[i]), numbers[i]))
        return [numbers[i] for i in order]

    return rank