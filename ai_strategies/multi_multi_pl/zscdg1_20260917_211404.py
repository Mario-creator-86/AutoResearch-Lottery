NAME = "zscdg1"
DESCRIPTION = 'Sector-cycle + deviatie macro (z recent + z long-run) + ciclu de intarziere (gap-bump gaussian), cu poarta de entropie si auto-orientare reversion/momentum pe persistenta suprapunerilor consecutive. Determinist, doar stdlib + numpy.'
SOURCE = 'model cantitativ propriu (plugin quant) + 2026-09-17'

import math
import numpy as np

CONFIG = {
    'SEC_SIZE': 8,
    'SEC_SIGN': 1,
    'DEV_SIGN': 1,
    'W_SEC': 0.7,
    'W_GAP': 0.5,
    'W_DEV': 0.4,
    'GAP_CAP': 2.5,
    'GATE_MIX': 0.5,
    'ORIENT_AUTO': 1,
    'PERS_THR': 0.6,
    'PERS_EWMA': 0.92,
    'DEV_LW': 0.35,
    'GATE_SCALE': 20.0,
}

_LAST_DIAG = {}


def diag():
    return dict(_LAST_DIAG)


def create(config):
    P = int(config.number_range)
    WD = int(config.numbers_drawn)
    p0 = WD / float(P)
    mu_gap = P / float(WD)
    sec_size = int(CONFIG['SEC_SIZE'])
    n_sec = (P + sec_size - 1) // sec_size
    sec_ids = ((np.arange(1, P + 1) - 1) // sec_size).astype(np.int64)
    k_sec = np.bincount(sec_ids, minlength=n_sec).astype(np.float64)
    numbers = list(range(1, P + 1))
    lnP = math.log(P)
    q = WD / float(P)
    pair_mean = WD * q
    pair_var = WD * q * (1.0 - q) * ((P - WD) / float(P - 1)) if P > 1 else 0.0
    pair_sd = math.sqrt(pair_var) if pair_var > 0 else 1.0
    state = {'orient': 1, 'flips': 0, 'pers': 0.0}

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        L = len(recent_deque)
        if L > 0:
            flat = [x for s in recent_deque for x in s]
            c = np.bincount(np.asarray(flat, dtype=np.int64), minlength=P + 1)[1:P + 1].astype(np.float64)
        else:
            c = np.zeros(P, dtype=np.float64)

        tot = float(c.sum())
        if tot > 0:
            pr = c / tot
            nz = pr > 0
            H = float(-(pr[nz] * np.log(pr[nz])).sum())
            entropy_norm = H / lnP
        else:
            entropy_norm = 1.0
        gate = min(2.0, max(0.6, 1.0 + CONFIG['GATE_MIX'] * (1.0 - entropy_norm) * CONFIG['GATE_SCALE']))

        if L >= 8:
            pairs = list(recent_deque)
            n_pairs = L - 1
            ov = 0
            for i in range(n_pairs):
                ov += len(pairs[i] & pairs[i + 1])
            pers_z = (ov / float(n_pairs) - pair_mean) / (pair_sd / math.sqrt(n_pairs))
            e = CONFIG['PERS_EWMA']
            state['pers'] = e * state['pers'] + (1.0 - e) * pers_z
            if CONFIG['ORIENT_AUTO']:
                if state['pers'] > CONFIG['PERS_THR'] and state['orient'] != -1:
                    state['orient'] = -1
                    state['flips'] += 1
                elif state['pers'] < -CONFIG['PERS_THR'] and state['orient'] != 1:
                    state['orient'] = 1
                    state['flips'] += 1
        orient = state['orient']

        if L > 0 and 0.0 < p0 < 1.0:
            sd_s = math.sqrt(L * p0 * (1.0 - p0))
            z_short = (c - L * p0) / sd_s
        else:
            z_short = np.zeros(P, dtype=np.float64)
        cnt = np.array([float(counts.get(n, 0)) for n in numbers])
        mc = float(cnt.mean())
        sc = float(cnt.std())
        z_long = (cnt - mc) / sc if sc > 0 else np.zeros(P, dtype=np.float64)
        dev_n = CONFIG['DEV_LW'] * z_long + z_short
        dev_contrib = -orient * CONFIG['DEV_SIGN'] * CONFIG['W_DEV'] * gate * dev_n

        Hs = np.bincount(sec_ids, weights=c, minlength=n_sec)
        exp_s = L * k_sec * p0
        var_sec = np.maximum(L * k_sec * p0 * (1.0 - p0), 1e-12)
        sec_z = (Hs - exp_s) / np.sqrt(var_sec)
        sec_contrib = -orient * CONFIG['SEC_SIGN'] * CONFIG['W_SEC'] * gate * sec_z[sec_ids]

        g = np.array([float(draw_index - last_seen.get(n, -1)) for n in numbers])
        center = mu_gap * CONFIG['GAP_CAP']
        bump = np.exp(-0.5 * ((g - center) / mu_gap) ** 2)
        gap_contrib = CONFIG['W_GAP'] * bump

        score = sec_contrib + dev_contrib + gap_contrib
        order = np.argsort(-score, kind='stable')

        d = {
            'draw_index': int(draw_index),
            'sector_span': float(sec_contrib.max() - sec_contrib.min()),
            'dev_span': float(dev_contrib.max() - dev_contrib.min()),
            'gap_span': float(gap_contrib.max() - gap_contrib.min()),
            'entropy_norm': float(entropy_norm),
            'gate_factor': float(gate),
            'orient_flips': int(state['flips']),
            'orient_state': int(orient),
            'pers_z': float(state['pers']),
            'L': int(L),
        }
        _LAST_DIAG.clear()
        _LAST_DIAG.update(d)
        try:
            rank._diag = d
        except Exception:
            pass

        return [numbers[i] for i in order]

    return rank