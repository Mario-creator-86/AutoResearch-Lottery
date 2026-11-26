NAME = "zpairzse"
DESCRIPTION = "Combinație de deviație z, EWMA recent, frecvență în perechi și ciclu pe sectoare pentru a evalua toate cele 90 de numere."
SOURCE = "model cantitativ + 2026-09-08"

import math
from collections import defaultdict

# hiperparametri expuși pentru grid‑search ulterior
W_Z = 0.55        # pondere deviație z
W_EW = 0.30       # pondere EWMA recent
W_PAIR = 0.40    # pondere perechi co‑occurrence
W_SEC = 0.50      # pondere ciclu pe sectoare
GAMMA = 0.94      # factor de decădere EWMA (halftime ~ 11 extrageri)
TK = 0.5          # scalare pentru tanh

def create(config):
    N = int(config.number_range)          # 90 pentru SuperEnalotto
    K = int(config.numbers_drawn)        # 6
    p = K / float(N)                     # probabilitatea teoretică pe un număr
    sector_size = 10
    n_sectors = (N + sector_size - 1) // sector_size

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        T = max(draw_index, 1)            # număr de extrageri deja efectuate
        expected = T * p                  # așteptarea binomială pe fiecare număr
        sd = math.sqrt(expected * (1 - p) + 1e-12)

        # -------- 1. deviație z (normalizată prin tanh) --------
        z_score = {}
        for n in numbers:
            cnt = counts.get(n, 0)
            raw_z = (cnt - expected) / sd
            # limităm extremele și aplicăm tanh pentru netezire
            z_score[n] = math.tanh(TK * raw_z)

        # -------- 2. EWMA pe fereastra recentă --------
        ewma = dict.fromkeys(numbers, 0.0)
        L = len(recent_deque)
        for age, draw_set in enumerate(recent_deque):
            weight = GAMMA ** age
            for n in draw_set:
                if 1 <= n <= N:
                    ewma[n] += weight
        # normalizare la suma totală a greutăților
        norm_factor = (1 - GAMMA ** L) / (1 - GAMMA) if L > 0 else 1.0
        for n in numbers:
            ewma[n] = math.tanh(TK * (ewma[n] / norm_factor))

        # -------- 3. Frecvență în perechi (co‑occurrence) --------
        pair_counts = defaultdict(float)   # scor pereche
        for age, draw_set in enumerate(recent_deque):
            w = GAMMA ** age               # aceeași decădere ca EWMA
            draw_list = list(draw_set)
            for i in range(len(draw_list)):
                for j in range(i + 1, len(draw_list)):
                    a, b = draw_list[i], draw_list[j]
                    if a > b:
                        a, b = b, a
                    pair_counts[(a, b)] += w
        # agregăm la nivel de număr
        pair_score = dict.fromkeys(numbers, 0.0)
        for (a, b), val in pair_counts.items():
            pair_score[a] += val
            pair_score[b] += val
        # netezim prin tanh
        for n in numbers:
            pair_score[n] = math.tanh(TK * pair_score[n])

        # -------- 4. Ciclu pe sectoare (gap‑sector) --------
        # Pentru fiecare sector calculăm cel mai mic gap (ultimul hit) dintre numerele sale
        sector_gap = [float('inf')] * n_sectors
        for n in numbers:
            sector_idx = (n - 1) // sector_size
            ls = last_seen.get(n)
            gap = draw_index - ls if ls is not None else draw_index
            if gap < sector_gap[sector_idx]:
                sector_gap[sector_idx] = gap
        # Transformăm gap‑ul în scor (gap mare => scor mare) şi normalizăm
        max_gap = max(sector_gap) if sector_gap else 1
        sector_score = [0.0] * n_sectors
        for i, g in enumerate(sector_gap):
            # scaling linear între 0 şi 1, apoi tanh
            norm = g / max_gap
            sector_score[i] = math.tanh(TK * norm)
        # fiecare număr primește scorul sectorului său
        sec_score = {}
        for n in numbers:
            sec_idx = (n - 1) // sector_size
            sec_score[n] = sector_score[sec_idx]

        # -------- combinație ponderată --------
        final_score = {}
        for n in numbers:
            s = (W_Z * z_score[n] +
                 W_EW * ewma[n] +
                 W_PAIR * pair_score[n] +
                 W_SEC * sec_score[n])
            final_score[n] = s

        # sortăm descrescător după scor, cu tie‑break pe număr (mai mic = prioritate)
        sorted_nums = sorted(numbers, key=lambda n: (-final_score[n], n))
        return sorted_nums

    return rank