import os
import numpy as np

NAME = "zzquad4"
DESCRIPTION = ('Scorer liniar z-score pe 4 componente: PAIR = co-aparitii ponderate EWMA gamma^varsta '
               'in fereastra recenta; DEV = deviatia fata de frecventa asteptata (pondere permisa negativa '
               '=> mean-reversion); POS = recența kernel gaussian pe pozitia ultimei aparitii in deque; '
               'CAD = cadență hazard (gap curent x rata locala Laplace din ultimele K_TAIL extrageri). '
               'Grid-search prin env: W_PAIR in [0,1.5], W_DEV in [-0.8,1.0], W_POS in [0,1], W_CAD in [0,1], '
               'GAMMA_P in [0.90,0.99], K_TAIL in {1,2,3,5}, SIGMA_POS in [4,12]. '
               "Verifica rank.DIAG['orient_flips']==0 inainte de rulare (audita orientarii newest_last a deque-ului).")
SOURCE = 'model + 2026-09-09 (kaskada_pl 12/24)'


def _env_float(name, default, lo, hi):
    try:
        v = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        v = float(default)
    return min(max(v, lo), hi)


def _z(x, cap=3.0):
    x = np.asarray(x, dtype=float)
    sd = x.std()
    if sd < 1e-12:
        return np.zeros_like(x)
    return np.clip((x - x.mean()) / sd, -cap, cap)


def create(config):
    nr = int(config.number_range)
    nd = int(config.numbers_drawn)

    W_PAIR = _env_float('W_PAIR', 0.80, 0.0, 1.5)
    W_DEV = _env_float('W_DEV', 0.10, -0.8, 1.0)
    W_POS = _env_float('W_POS', 0.50, 0.0, 1.0)
    W_CAD = _env_float('W_CAD', 0.50, 0.0, 1.0)
    GAMMA_P = _env_float('GAMMA_P', 0.945, 0.90, 0.99)
    SIGMA_POS = _env_float('SIGMA_POS', 8.0, 4.0, 12.0)
    try:
        K_TAIL = int(os.environ.get('K_TAIL', 2))
    except (TypeError, ValueError):
        K_TAIL = 2
    if K_TAIL not in (1, 2, 3, 5):
        K_TAIL = 2

    DIAG = {'orient_flips': 0, 'calls': 0, 'last_L': 0}

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, nr + 1))
        L = len(recent_deque)
        DIAG['calls'] += 1
        DIAG['last_L'] = L

        zp = np.zeros(nr)
        zd = np.zeros(nr)
        zpo = np.zeros(nr)
        zc = np.zeros(nr)

        if L > 0:
            X = np.zeros((L, nr), dtype=float)
            for j, s in enumerate(recent_deque):
                if s:
                    X[j, [n - 1 for n in s if 1 <= n <= nr]] = 1.0
            # orientare presupusa: newest_last (deque[-1] = extragerea cea mai recenta)
            ages = (L - 1) - np.arange(L)
            w = GAMMA_P ** ages
            Xrev = X[::-1]
            any_mask = Xrev.any(axis=0)
            maxpos = (L - 1) - np.argmax(Xrev, axis=0)

            # audit orientare: last_seen mai recent => maxpos nu mai mic; violatiile cresc orient_flips
            present_known = [n for n in numbers if any_mask[n - 1] and n in last_seen]
            if len(present_known) >= 2:
                present_known.sort(key=lambda n: -last_seen[n])
                seq = [int(maxpos[n - 1]) for n in present_known]
                bad = sum(1 for a, b in zip(seq, seq[1:]) if b > a)
                if bad:
                    DIAG['orient_flips'] += bad

            if W_PAIR != 0.0:
                Cw = (X * w[:, None]).T @ X
                pair_raw = Cw.sum(axis=1) - np.diag(Cw)
                zp = _z(pair_raw)

            if W_POS != 0.0:
                d = np.where(any_mask, (L - 1) - maxpos, float(L))
                zpo = _z(np.exp(-(d * d) / (2.0 * SIGMA_POS * SIGMA_POS)))

            if W_CAD != 0.0:
                k = min(K_TAIL, L)
                tail_hits = X[L - k:].sum(axis=0)
                rate = (tail_hits + 1.0) / (k + 2.0)
                max_gap = 8.0 * (float(nr) / max(nd, 1))
                g = np.array([
                    draw_index - last_seen[n] if n in last_seen else draw_index + 1
                    for n in numbers
                ], dtype=float)
                g = np.clip(g, 0.0, max_gap)
                zc = _z(g * rate - 1.0)

        if W_DEV != 0.0:
            total = float(sum(counts.values()))
            draws_so_far = total / nd if nd > 0 else 0.0
            exp_ct = draws_so_far * (float(nd) / nr)
            cnt = np.array([counts.get(n, 0) for n in numbers], dtype=float)
            zd = _z(cnt - exp_ct)

        total_score = W_PAIR * zp + W_DEV * zd + W_POS * zpo + W_CAD * zc
        cnt_arr = np.array([counts.get(n, 0) for n in numbers], dtype=float)
        order = sorted(range(nr), key=lambda i: (-total_score[i], -cnt_arr[i], i + 1))
        return [numbers[i] for i in order]

    rank.DIAG = DIAG
    rank.GRID = {
        'W_PAIR': [0.0, 1.5], 'W_DEV': [-0.8, 1.0], 'W_POS': [0.0, 1.0],
        'W_CAD': [0.0, 1.0], 'GAMMA_P': [0.90, 0.99], 'K_TAIL': [1, 2, 3, 5],
        'SIGMA_POS': [4.0, 12.0],
    }
    return rank