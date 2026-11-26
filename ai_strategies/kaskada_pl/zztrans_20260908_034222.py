import math

import numpy as np

NAME = "zztrans"
DESCRIPTION = 'Markov de ordin 1 pe indicatorii de aparitie: matrice de tranzitie P(n la t+1 | m la t) estimata din fereastra recenta si shrink-uita catre 0.5, conditionata pe ultima extragere; combinata cu persistenta lag-1 pe diagonala (scalata de stare), cu z-score de frecventa pe tot istoricul si cu un mic tilt de echilibru pe 4 sectoare.'
SOURCE = 'model + 2026-09-08'


def create(config):
    P = int(config.number_range)
    K = int(config.numbers_drawn)
    try:
        TK = int(getattr(config, 'ticket_size', 0)) or K
    except Exception:
        TK = K
    n_sect = 4
    sect_of = np.array([((n - 1) * n_sect) // P for n in range(1, P + 1)], dtype=int)
    exp_sect = float(TK) / n_sect
    w_trans = 0.55
    w_stick = 0.40
    w_freq = 0.60
    w_bal = 0.08
    alpha = 20.0

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        score = np.zeros(P, dtype=float)
        D = [set(s) for s in recent_deque]
        m = len(D)
        if m >= 2 and last_seen:
            mx = max(last_seen.values())
            last_set = set(n for n, v in last_seen.items() if v == mx)
            if last_set and last_set != D[-1] and last_set == D[0]:
                D.reverse()
        if m >= 6:
            W = np.zeros((m, P), dtype=float)
            for i, s in enumerate(D):
                for n in s:
                    if 1 <= n <= P:
                        W[i, n - 1] = 1.0
            Cm = W[:-1].T @ W[1:]
            r = W[:-1].sum(axis=0)
            Mh = (Cm + alpha * 0.5) / (r[:, None] + alpha)
            l = W[-1]
            off = Mh - 0.5
            diagv = np.diag(off).copy()
            A = (l @ off) - l * diagv
            sdA = float(A.std())
            a_sig = np.tanh(A / (sdA + 0.02))
            sdD = float(diagv.std())
            st = np.where(l > 0.5, 1.0, -1.0)
            s_sig = np.tanh(diagv / (sdD + 0.02)) * st
            score += w_trans * a_sig + w_stick * s_sig
        total = 0
        for v in counts.values():
            total += v
        N = (float(total) / K) if K > 0 else 0.0
        if N >= 30.0:
            cnt = np.array([float(counts.get(n, 0)) for n in range(1, P + 1)])
            z = (cnt / N - 0.5) * 2.0 * math.sqrt(N)
            score += w_freq * np.tanh(z / 3.0)
        if TK >= 2 and m >= 1:
            prov = np.argsort(-score, kind='stable')[:TK]
            cc = np.bincount(sect_of[prov], minlength=n_sect).astype(float)
            deficit = np.clip((exp_sect - cc) / max(exp_sect, 1.0), -1.0, 1.0)
            score = score + w_bal * deficit[sect_of]
        order = np.argsort(-score, kind='stable')
        return [int(i) + 1 for i in order]

    return rank