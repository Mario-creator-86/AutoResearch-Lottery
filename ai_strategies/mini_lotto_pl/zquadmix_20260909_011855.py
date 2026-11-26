NAME = "zquadmix"
DESCRIPTION = ('Strategie cu 4 componente z-normalizate combinate liniar: interactiune '
               'co-occurventa x deviatie (ponderata exponential GAMMA_P), deviatie binomiala '
               'fata de frecventa asteptata, kernel gaussian de recenta (SIGMA_POS) si hazard '
               'saturant de intarziere (K_TAIL). Ponderile w_pair/w_dev/w_pos/w_cad sunt expuse '
               'ca parametri pentru grid-search si ablative; DIAG[orient_flips] valideaza '
               'orientarea newest_last a deque-ului.')
SOURCE = 'qstrat-quant + 2026-09-09'

import math
import numpy as np

DIAG = {'orient_flips': 0, 'calls': 0}


def _param(cfg, name, default):
    v = getattr(cfg, name, None)
    if v is None:
        for holder in ('strategy_params', 'params', 'hyperparams'):
            d = getattr(cfg, holder, None)
            if isinstance(d, dict) and name in d:
                v = d[name]
                break
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _z(v):
    sd = float(v.std())
    if sd < 1e-12:
        return np.zeros_like(v)
    return (v - v.mean()) / sd


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)

    W_PAIR = _param(config, 'w_pair', 0.60)
    W_DEV = _param(config, 'w_dev', -0.40)
    W_POS = _param(config, 'w_pos', 0.70)
    W_CAD = _param(config, 'w_cad', 0.50)
    GAMMA_P = min(max(_param(config, 'gamma_p', 0.95), 0.0), 0.999)
    K_TAIL = max(_param(config, 'k_tail', 2.0), 0.25)
    SIGMA_POS = max(_param(config, 'sigma_pos', 6.0), 0.5)

    p_exp = K / float(N)                                   # P: numarul iese intr-o extragere
    q_abs = (N - K) / float(N) if N > K else 0.0           # P: numarul nu iese
    g_mean = (1.0 / (1.0 - q_abs)) if q_abs < 1.0 else 1.0  # gap mediu asteptat (5/42: 6.0)

    need_pair = (W_PAIR != 0.0)
    need_dev = (W_DEV != 0.0) or need_pair
    need_pos = (W_POS != 0.0)
    need_cad = (W_CAD != 0.0)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        DIAG['calls'] += 1
        D = int(draw_index)
        score = np.zeros(N, dtype=np.float64)
        dev = None

        # DEV: deviatia fata de frecventa asteptata pe istoricul complet (z binomial)
        if need_dev:
            cnt = np.fromiter((counts.get(n, 0) for n in range(1, N + 1)),
                              dtype=np.float64, count=N)
            var = D * p_exp * (1.0 - p_exp)
            dev = ((cnt - D * p_exp) / math.sqrt(var)) if var > 0.0 else np.zeros(N)

        # orientarea deque-ului: presupunem newest_last; validare prin suma last_seen
        seq = list(recent_deque)
        if len(seq) >= 2:
            f0 = sum(last_seen.get(n, -1) for n in seq[0])
            fl = sum(last_seen.get(n, -1) for n in seq[-1])
            if f0 > fl:
                DIAG['orient_flips'] += 1   # deque era newest_first
            else:
                seq.reverse()               # newest_last -> iteratie newest-first
        Wn = len(seq)

        rf = np.zeros(N + 1)
        pos = np.zeros(N + 1)
        pg = np.zeros(N + 1)
        if Wn > 0 and (need_pair or need_pos):
            ages = np.arange(Wn, dtype=np.float64)      # seq[0] = cel mai nou (age 0)
            sizes = [len(s) for s in seq]
            total = sum(sizes)
            if total > 0:
                idx = np.fromiter((m for s in seq for m in s),
                                  dtype=np.int64, count=total)
                if need_pos:
                    kern = np.exp(-(ages * ages) / (2.0 * SIGMA_POS * SIGMA_POS))
                    pos = np.bincount(idx, weights=np.repeat(kern, sizes),
                                      minlength=N + 1)
                if need_pair:
                    w = np.power(GAMMA_P, ages)
                    dev_ext = np.zeros(N + 1)
                    dev_ext[1:] = dev
                    dev_m = dev_ext[idx]
                    if sizes[0] == K and all(s == K for s in sizes):
                        G = dev_m.reshape(Wn, K).sum(axis=1)
                    else:
                        G = np.empty(Wn)
                        o = 0
                        for i in range(Wn):
                            G[i] = dev_m[o:o + sizes[i]].sum()
                            o += sizes[i]
                    wg = w * G
                    rf = np.bincount(idx, weights=np.repeat(w, sizes),
                                     minlength=N + 1)
                    pg = np.bincount(idx, weights=np.repeat(wg, sizes),
                                     minlength=N + 1)

        if need_pair:
            # pair[n] = sum over drawurile care l-au continut pe n de w_d*(G_d - dev_n)
            pair = pg[1:] - dev * rf[1:]
            score += W_PAIR * _z(pair)
        if need_dev:
            score += W_DEV * _z(dev)
        if need_pos:
            score += W_POS * _z(pos[1:])
        if need_cad:
            gap = np.fromiter((D - 1 - last_seen.get(n, -1) for n in range(1, N + 1)),
                              dtype=np.float64, count=N)
            np.maximum(gap, 0.0, out=gap)
            cad = 1.0 - np.exp(-gap / (K_TAIL * g_mean))
            score += W_CAD * _z(cad)

        return sorted(range(1, N + 1), key=lambda n: (-float(score[n - 1]), n))

    return rank