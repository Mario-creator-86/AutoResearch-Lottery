import numpy as np

NAME = "zzstreak"
DESCRIPTION = 'Serial-dependence streak model: per-number lag-1/lag-2 appearance autocorrelations (empirical-Bayes shrunk) conditioned on presence in the last two draws, blended with an EWMA recent rate, long-run frequency z-score and a mild freshness gap penalty into a logit score.'
SOURCE = 'quant research plugin 2026-09-26, Kaskada 12/24'

BETA_R = 0.9
BETA_1 = 1.5
BETA_2 = 0.7
SHRINK_K = 60.0
EWMA_D = 0.97
W_Z = 0.05
WGAP = 0.015
GAP_CAP = 12
_BITS = np.arange(64, dtype=np.int64)


def create(config):
    N = int(config.number_range)
    base = float(config.numbers_drawn) / float(N)
    bits_idx = _BITS[:N]
    ewma_cache = {}

    def _ewma(L):
        w = ewma_cache.get(L)
        if w is None:
            w = EWMA_D ** np.arange(L - 1, -1, -1, dtype=np.float64)
            ewma_cache[L] = w
        return w

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        nums = list(range(1, N + 1))
        L = len(recent_deque)
        cnt = np.array([counts.get(n, 0) for n in nums], dtype=np.float64)
        sd = float(cnt.std())
        z = (cnt - cnt.mean()) / (sd if sd > 1e-9 else 1.0)
        gap = np.array(
            [min(max(draw_index - last_seen.get(n, draw_index + 1), 0), GAP_CAP) for n in nums],
            dtype=np.float64)

        if L >= 8:
            masks = np.fromiter(
                (sum(1 << (n - 1) for n in s) for s in recent_deque),
                dtype=np.int64, count=L)
            b = ((masks[:, None] >> bits_idx[None, :]) & 1).astype(np.float64)
            r = b.mean(axis=0)
            w = _ewma(L)
            w = w / w.sum()
            r_ew = (b * w[:, None]).sum(axis=0)
            x1 = b[-1]
            x2 = b[-2]
            c1 = (b[1:] * b[:-1]).sum(axis=0) / (L - 1)
            c2 = (b[2:] * b[:-2]).sum(axis=0) / (L - 2)
            phi1 = (c1 - r * r) * ((L - 1) / (L - 1 + SHRINK_K))
            phi2 = (c2 - r * r) * ((L - 2) / (L - 2 + SHRINK_K))
            p = (base
                 + BETA_R * (r_ew - base)
                 + BETA_1 * phi1 * (x1 - base)
                 + BETA_2 * phi2 * (x2 - base)
                 + W_Z * z)
            p = np.clip(p, 0.03, 0.97)
            score = np.log(p / (1.0 - p)) - WGAP * gap
        else:
            score = 0.10 * z - WGAP * gap

        order = sorted(range(N), key=lambda i: (-float(score[i]), i + 1))
        return [nums[i] for i in order]

    return rank