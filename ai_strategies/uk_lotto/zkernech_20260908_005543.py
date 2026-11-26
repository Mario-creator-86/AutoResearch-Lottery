import math
import numpy as np

NAME = "zkernech"
DESCRIPTION = 'Regime-switching persistence strategy: hump-shaped recency kernel over gap ratios (peak ~0.35x expected gap), EWMA frequency momentum, full-history frequency z-scores, co-occurrence echo with the latest draw, and sector-pressure deviations, blended by an overlap/diversity regime gate.'
SOURCE = 'model + 2026-09-07'

def create(config):
    P = int(config.number_range)
    W = int(config.numbers_drawn)
    p = W / float(P)
    e_gap = (1.0 - p) / p
    lam = 0.5 ** (1.0 / 8.0)
    ker_mu = 0.35
    ker_sd = 0.742
    sector_ids = (np.arange(1, P + 1) - 1) // 10
    n_sec = int(sector_ids.max()) + 1
    sizes = np.bincount(sector_ids, minlength=n_sec).astype(float)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = np.arange(1, P + 1)
        draws = list(recent_deque)
        if recent_window and len(draws) > int(recent_window):
            draws = draws[-int(recent_window):]
        m = len(draws)
        m_eff = min(m, 50)
        win = draws[-m_eff:] if m_eff else []

        cnt = np.array([float(counts.get(int(n), 0)) for n in numbers])
        mu = cnt.mean()
        sd = cnt.std()
        z_freq = (cnt - mu) / sd if sd > 1e-9 else np.zeros(P)

        mom = np.zeros(P)
        aff = np.zeros(P)
        if m_eff:
            M = np.zeros((m_eff, P))
            for j, s in enumerate(win):
                for n in s:
                    M[j, n - 1] = 1.0
            wts = lam ** np.arange(m_eff - 1, -1, -1)
            sw = float(wts.sum())
            sw2 = float(np.dot(wts, wts))
            sd_w = math.sqrt(p * (1.0 - p) * sw2)
            if sd_w > 1e-12:
                mom = np.clip((np.dot(wts, M) - p * sw) / sd_w, -3.0, 3.0)
            co = M.T @ M
            np.fill_diagonal(co, 0.0)
            a = co @ M[-1]
            a_sd = a.std()
            if a_sd > 1e-9:
                aff = np.clip((a - a.mean()) / a_sd, -3.0, 3.0)
            Ms = M[-min(m_eff, 12):]
        else:
            Ms = np.zeros((0, P))

        gaps = np.array([float(draw_index - last_seen.get(int(n), -1)) for n in numbers])
        g = np.clip(gaps / e_gap, 0.0, 3.0)
        rec = np.exp(-0.5 * ((g - ker_mu) / ker_sd) ** 2)
        rec = rec + 0.18 * np.clip((g - 1.5) / 1.5, 0.0, 1.0)

        obs = Ms.sum(axis=0)
        L = Ms.shape[0]
        exp_s = L * sizes * p
        var_s = np.maximum(L * sizes * p * (1.0 - p), 1e-9)
        dev = (np.bincount(sector_ids, weights=obs, minlength=n_sec) - exp_s) / np.sqrt(var_s)
        sec_term = dev[sector_ids]

        k = len(draws[-1] & draws[-2]) if m >= 2 else 0
        ge = min(k, 3) / 3.0
        if m >= 1:
            div = float(len({(int(n) - 1) // 10 for n in draws[-1]}))
            gd = max(0.0, (float(n_sec) - div) / float(max(n_sec - 1, 1)))
        else:
            gd = 0.0
        r = 0.6 * ge + 0.4 * gd

        scores = (0.30 * z_freq
                  + (0.48 - 0.22 * r) * mom
                  + (0.35 + 0.50 * r) * aff
                  + 0.60 * rec
                  + 0.20 * sec_term)

        order = np.lexsort((numbers, -scores))
        return [int(numbers[i]) for i in order]

    return rank