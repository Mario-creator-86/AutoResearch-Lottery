NAME = "zzsectpa"
DESCRIPTION = 'Fuzioneaza patru semnale structurale: deficit de lovituri pe 8 sectoare a cate 10 numere (reversie agregata), tractiune pe graful de co-aparitie cu ancore din ultimele 3 extrageri, deficit de frecventa pe tot istoricul si gol de aparitie (gap). Scoruri z standardizate cross-sectiunal, limitate cu tanh, combinate liniar; sortare determinista.'
SOURCE = 'sector-reversal + pair-graph traction + history deficit fusion, 2026-09-08'

import numpy as np


def create(config):
    N = int(config.number_range)      # 80
    K = int(config.numbers_drawn)     # 20 extrase per extragere
    p = K / float(N)                  # 0.25 probabilitate per numar

    numbers = list(range(1, N + 1))
    n_sect = 8
    sector_of = np.array([(n - 1) * n_sect // N for n in numbers], dtype=np.int64)
    ii, jj = np.triu_indices(K, k=1)  # 190 perechi de pozitii intr-o extragere

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # fereastra recenta ca matrice (W, K); nu depinde de ordinea din deque
        draws = [s for s in recent_deque if len(s) == K]
        W = len(draws)
        cnt_win = np.zeros(N, dtype=np.float64)
        co = np.zeros((N, N), dtype=np.float64)
        if W > 0:
            flat = np.fromiter((x for d in draws for x in d), dtype=np.int64, count=W * K)
            arr = flat.reshape(W, K)
            cnt_win = np.bincount(arr.ravel(), minlength=N + 1)[1:].astype(np.float64)
            a = arr[:, ii].ravel() - 1
            b = arr[:, jj].ravel() - 1
            cof = np.bincount(a * N + b, minlength=N * N).astype(np.float64).reshape(N, N)
            co = cof + cof.T

        gaps = np.array([draw_index - last_seen.get(n, -1) for n in numbers], dtype=np.float64)

        # semnal 1: deficit pe sectoare, standardizat cross-sectiunal
        obs = np.bincount(sector_of, weights=cnt_win, minlength=n_sect)
        def_s = p * (N / float(n_sect)) * W - obs
        z_sec = (def_s - def_s.mean()) / (def_s.std() + 1e-9)
        s1 = z_sec[sector_of]

        # semnal 2: tractiune de perechi cu ancore din ultimele 3 extrageri
        anchor = (gaps >= 1) & (gaps <= 3)
        row_mass = co.sum(axis=1)
        if bool(anchor.any()):
            a_mass = co[:, anchor].sum(axis=1)
        else:
            a_mass = np.zeros(N, dtype=np.float64)
        tract = a_mass / (row_mass + 1e-9)
        s2 = (tract - tract.mean()) / (tract.std() + 1e-9)

        # semnal 3: deficit de frecventa pe tot istoricul (z binomial + tanh)
        T_tot = float(sum(counts.values())) / float(K)
        exp_c = p * T_tot
        std_c = np.sqrt(max(T_tot, 1.0) * p * (1.0 - p))
        cnt_full = np.array([counts.get(n, 0) for n in numbers], dtype=np.float64)
        s3 = np.tanh((exp_c - cnt_full) / (std_c + 1e-9))

        # semnal 4: gap / intarziere (reversie)
        s4 = np.tanh((gaps - gaps.mean()) / (gaps.std() + 1e-9))

        ramp = min(1.0, W / 50.0) if W > 0 else 0.0
        score = ramp * (s1 + s2) + 0.7 * s3 + 0.5 * s4
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

        order = sorted(range(N), key=lambda i: (-float(score[i]), numbers[i]))
        return [numbers[i] for i in order]

    return rank