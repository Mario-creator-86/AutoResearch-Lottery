NAME = "zpairsec"
DESCRIPTION = "Combinație de EWMA recent, forță perechi recent, entropie sectorială și deviație z pentru ranking complet al numerelor 1..90."
SOURCE = "quant model + 2026-09-23"

import math
import numpy as np
from collections import defaultdict

# Diagnostic pentru orientarea deque‑ului (nou → vechi sau invers)
DIAG = {"orient": None, "orient_flips": 0}

# hiperparametri (pot fi supuși grid‑search ulterior)
W_EWMA = 0.5   # pondere EWMA recentă
W_PAIR = 0.4   # pondere forță perechi
W_ENT = 0.3    # pondere entropie sectorială
W_Z = 0.2      # pondere deviație z (penalizează extreme)
DECAY_EWMA = 0.94  # factor de decay pentru EWMA (half‑life ≈ 12 drawuri)

def create(config):
    """Returnează funcţia rank conform interfeţei cerute.
    config: LotteryConfig cu .number_range, .numbers_drawn, .ticket_size
    """
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    p_theory = K / float(N)
    sector_size = 10
    num_sectors = (N + sector_size - 1) // sector_size

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # -----------------------------------------------------------------
        # Detectare orientare deque (newest_last vs newest_first)
        # -----------------------------------------------------------------
        def mean_recency(draw_set):
            if not draw_set:
                return float('inf')
            total = 0.0
            for n in draw_set:
                ls = last_seen.get(n)
                total += (draw_index - ls) if ls is not None else float(draw_index)
            return total / len(draw_set)

        newest_last = True  # presupunem implicit
        if len(recent_deque) >= 2:
            newest_last = mean_recency(recent_deque[-1]) <= mean_recency(recent_deque[0])
        orient = 'newest_last' if newest_last else 'newest_first'
        if DIAG["orient"] is not None and DIAG["orient"] != orient:
            DIAG["orient_flips"] += 1
        DIAG["orient"] = orient

        # Dacă deque‑ul este în ordine newest_last, inversăm pentru a avea
        # indexul 0 = cel mai vechi și indexul L‑1 = cel mai recent.
        seq = list(recent_deque)
        if newest_last:
            seq = list(reversed(seq))
        L = len(seq)

        numbers = list(range(1, N + 1))

        # -----------------------------------------------------------------
        # 1) EWMA recentă
        # -----------------------------------------------------------------
        ewma_score = np.zeros(N + 1)  # 1‑indexed
        for age, draw in enumerate(seq):
            weight = DECAY_EWMA ** age
            for n in draw:
                if 1 <= n <= N:
                    ewma_score[n] += weight
        # Normalizare
        ew_min, ew_max = ewma_score[1:].min(), ewma_score[1:].max()
        if ew_max > ew_min:
            ewma_norm = (ewma_score - ew_min) / (ew_max - ew_min)
        else:
            ewma_norm = np.zeros(N + 1)

        # -----------------------------------------------------------------
        # 2) Forța perechilor în fereastra recentă
        # -----------------------------------------------------------------
        pair_counts = defaultdict(int)
        for draw in seq:
            draw_list = list(draw)
            for i in range(len(draw_list)):
                a = draw_list[i]
                for j in range(i + 1, len(draw_list)):
                    b = draw_list[j]
                    if a > b:
                        a, b = b, a
                    pair_counts[(a, b)] += 1
        pair_score = np.zeros(N + 1)
        for (a, b), cnt in pair_counts.items():
            pair_score[a] += cnt
            pair_score[b] += cnt
        # Normalizare
        p_min, p_max = pair_score[1:].min(), pair_score[1:].max()
        if p_max > p_min:
            pair_norm = (pair_score - p_min) / (p_max - p_min)
        else:
            pair_norm = np.zeros(N + 1)

        # -----------------------------------------------------------------
        # 3) Entropia sectorială (bazată pe EWMA ponderată pe sectoare)
        # -----------------------------------------------------------------
        sector_hits = np.zeros(num_sectors)
        sector_total_weight = np.zeros(num_sectors)
        for age, draw in enumerate(seq):
            w = DECAY_EWMA ** age
            for n in draw:
                if 1 <= n <= N:
                    sec = (n - 1) // sector_size
                    sector_hits[sec] += w
                    sector_total_weight[sec] += w
        # Probabilităţi sectoriale
        sector_prob = np.where(sector_total_weight > 0, sector_hits / sector_total_weight, 0.0)
        # Entropie Shannon (ln)
        entropy = -np.sum(np.where(sector_prob > 0, sector_prob * np.log(sector_prob), 0.0))
        # Scor sectorial pentru fiecare număr = entropia (identic pentru toate numerele din acelaşi sector)
        ent_score = np.zeros(N + 1)
        for sec in range(num_sectors):
            # Normalizăm entropia la [0,1] folosind log(num_sectors) ca maxim teoretic
            ent_norm = entropy / math.log(num_sectors) if num_sectors > 1 else 0.0
            lo = sec * sector_size + 1
            hi = min((sec + 1) * sector_size, N)
            ent_score[lo:hi + 1] = ent_norm
        # Entropia este deja în intervalul [0,1]
        ent_norm = ent_score

        # -----------------------------------------------------------------
        # 4) Z‑score față de frecvenţa teoretică (penalizare pentru deviatii mari)
        # -----------------------------------------------------------------
        mu = draw_index * p_theory
        sigma = math.sqrt(draw_index * p_theory * (1 - p_theory) + 1e-9)
        z_score = np.zeros(N + 1)
        for n in numbers:
            cnt = counts.get(n, 0)
            z = (cnt - mu) / sigma
            # penalizăm abaterea absolută; cu cât este mai mică, cu atât e mai bun
            z_score[n] = -abs(z)
        # Normalizare la [0,1]
        z_min, z_max = z_score[1:].min(), z_score[1:].max()
        if z_max > z_min:
            z_norm = (z_score - z_min) / (z_max - z_min)
        else:
            z_norm = np.zeros(N + 1)

        # -----------------------------------------------------------------
        # Combinare greutate
        # -----------------------------------------------------------------
        final_score = (
            W_EWMA * ewma_norm +
            W_PAIR * pair_norm +
            W_ENT * ent_norm +
            W_Z * z_norm
        )

        # Sortare descrescătoare după scor, tie‑break pe număr (crescător)
        sorted_numbers = sorted(numbers, key=lambda n: (-final_score[n], n))
        return sorted_numbers

    return rank