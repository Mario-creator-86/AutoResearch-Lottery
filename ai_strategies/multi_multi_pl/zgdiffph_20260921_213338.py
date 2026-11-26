import math
import numpy as np

NAME = "zgdiffph"
DESCRIPTION = 'Difuzie de momentum pe graful de co-ocurenta recent (random walk row-normalizat, 2 pasi), orthogonalizata fata de semnalul hot, combinata cu faza de ciclu propriu (gap/interval asteptat per numar), deviatia long-run fata de frecventa asteptata si un fade padratic pe numerele supraextinse recent.'
SOURCE = 'model + 2026-09-20'

DECAY = 0.96
HOT_CAP = 100
PAIR_DRAWS = 40
DIFF_ALPHA = 0.45
DIFF_ITERS = 2
W_HOT = 1.00
W_DIFF = 0.55
W_GAP = 0.50
W_DEV = 0.12
W_FADE = 0.60
FADE_Z0 = 2.0
FADE_WIN = 24
PHASE_MAX = 6.0


def create(config):
    n_num = int(config.number_range)
    n_dr = int(config.numbers_drawn)
    p = (n_dr / float(n_num)) if n_num > 0 else 0.25
    tau_theory = (1.0 / p) if p > 0 else 4.0

    def znorm(v):
        sd = v.std()
        if sd < 1e-12:
            return np.zeros_like(v)
        return (v - v.mean()) / sd

    def unit(v):
        lo = float(v.min())
        hi = float(v.max())
        if hi - lo < 1e-12:
            return np.zeros_like(v)
        return (v - lo) / (hi - lo)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        arr = np.arange(n_num, dtype=np.int64)
        total = float(draw_index + 1)
        dq = recent_deque
        L = len(dq)

        # 1) EWMA momentum (newest-last assumption: dq[L-1] = ultima extragere)
        hot = np.zeros(n_num, dtype=np.float64)
        cap = HOT_CAP if L > HOT_CAP else L
        w = 1.0
        for k in range(cap):
            s = dq[L - 1 - k]
            if s:
                idx = np.fromiter(s, dtype=np.int64, count=len(s)) - 1
                hot[idx] += w
            w *= DECAY

        # 2) graf de co-ocurenta din ultimele PAIR_DRAWS + difuzie
        pd = PAIR_DRAWS if L > PAIR_DRAWS else L
        diff = np.zeros(n_num, dtype=np.float64)
        if pd >= 2:
            X = np.zeros((pd, n_num), dtype=np.float64)
            for i in range(pd):
                s = dq[L - 1 - i]
                if s:
                    X[i, np.fromiter(s, dtype=np.int64, count=len(s)) - 1] = 1.0
            Co = X.T @ X
            np.fill_diagonal(Co, 0.0)
            rs = Co.sum(axis=1)
            Wm = Co / (rs[:, None] + 1e-9)
            h0 = unit(hot)
            r = h0
            for _ in range(DIFF_ITERS):
                r = (1.0 - DIFF_ALPHA) * h0 + DIFF_ALPHA * (Wm @ r)
            diff = r

        # 3) faza de ciclu propriu: u = gap / tau(n), kernel u*exp(1-u), varf la u=1
        gap = np.empty(n_num, dtype=np.float64)
        for j in range(n_num):
            c = counts.get(j + 1, 0)
            ls = last_seen.get(j + 1)
            g = (draw_index + 1) if ls is None else (draw_index - ls)
            tau = (total / c) if c > 0 else tau_theory
            if tau < 0.5 * tau_theory:
                tau = 0.5 * tau_theory
            elif tau > 3.0 * tau_theory:
                tau = 3.0 * tau_theory
            u = g / tau
            if u > PHASE_MAX:
                u = PHASE_MAX
            elif u < 0.0:
                u = 0.0
            gap[j] = u * math.exp(1.0 - u)

        # 4) deviatie long-run fata de frecventa asteptata
        cnt = np.fromiter((counts.get(j + 1, 0) for j in range(n_num)), dtype=np.float64, count=n_num)
        exp_mean = total * p
        sd_tot = math.sqrt(exp_mean * (1.0 - p)) if exp_mean > 0 else 1.0
        if sd_tot < 1e-9:
            sd_tot = 1.0
        dev = (cnt - exp_mean) / sd_tot

        # 5) fade patratic pe numerele supraextinse in fereastra scurta
        zf = np.zeros(n_num, dtype=np.float64)
        fw = FADE_WIN if L > FADE_WIN else L
        if fw > 0:
            frec = np.zeros(n_num, dtype=np.float64)
            for i in range(fw):
                s = dq[L - 1 - i]
                if s:
                    frec[np.fromiter(s, dtype=np.int64, count=len(s)) - 1] += 1.0
            exp_w = fw * p
            sd_w = math.sqrt(exp_w * (1.0 - p)) if exp_w > 0 else 1.0
            if sd_w < 1e-9:
                sd_w = 1.0
            zr = (frec - exp_w) / sd_w
            zf = np.maximum(zr - FADE_Z0, 0.0) ** 2

        z_hot = znorm(hot)
        z_diff = znorm(diff)
        nh = float(z_hot @ z_hot)
        if nh > 1e-9:
            z_diff = z_diff - (float(z_diff @ z_hot) / nh) * z_hot
            z_diff = znorm(z_diff)

        c_hot = W_HOT * z_hot
        c_diff = W_DIFF * z_diff
        c_gap = W_GAP * znorm(gap)
        c_dev = W_DEV * znorm(dev)
        c_fade = -W_FADE * zf
        sc = c_hot + c_diff + c_gap + c_dev + c_fade

        order = np.lexsort((arr, -sc))
        try:
            top_final = set(order[:20].tolist())
            ov = {}
            for nm, cv in (('hot', c_hot), ('diff', c_diff), ('gap', c_gap), ('dev', c_dev), ('fade', c_fade)):
                o2 = np.argsort(-cv, kind='stable')[:20]
                ov[nm] = int(len(top_final.intersection(o2.tolist())))
            rank._diag = {'contrib': ov}
        except Exception:
            pass

        return [int(x) + 1 for x in order]

    return rank