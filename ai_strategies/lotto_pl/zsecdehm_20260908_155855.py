NAME = "zsecdehm"
DESCRIPTION = "Strategie combinată sector‑deficit, EWMA recent și hazard pe gap. Prioritizează numere din sectoare sub‑reprezentate, puțin văzute recent și cu gap mare, pentru a captura reveria medie."
SOURCE = "ChatGPT 2026-09-08"

import numpy as np
from collections import deque

def create(config):
    # Config parameters – nu le modificăm, le folosim doar pentru a determina range‑ul.
    number_range = config.number_range  # 49
    numbers_drawn = config.numbers_drawn  # 6
    # Fixed hyper‑parameters (deterministice)
    GAMMA = 0.95          # EWMA smoothing factor
    W_S = 1.5            # weight for sector deficit
    W_E = 2.0            # weight for EWMA (inverse)
    W_H = 1.0            # weight for hazard (gap)
    SECTORS = 7          # 7 sectoare de câte 7 numere
    SECTOR_SIZE = number_range // SECTORS  # 7

    # Pre‑compute sector mapping for rapid lookup
    sector_of = {n: (n - 1) // SECTOR_SIZE for n in range(1, number_range + 1)}

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # ------------------------------------------------------------------
        # 1. Sector deficit scores
        # ------------------------------------------------------------------
        # Expected total draws for each number (historical) = counts total / number_range
        total_draws = sum(counts.values())
        expected_per_number = total_draws / number_range
        expected_per_sector = expected_per_number * SECTOR_SIZE
        # Actual sector counts (sum of counts of numbers belonging to the sector)
        sector_counts = np.zeros(SECTORS, dtype=float)
        for n, cnt in counts.items():
            sector_counts[sector_of[n]] += cnt
        sector_deficit = expected_per_sector - sector_counts  # positive -> sector under‑represented
        # Normalizare la [0,1]
        if sector_deficit.max() > 0:
            sector_deficit_norm = (sector_deficit - sector_deficit.min()) / (sector_deficit.max() - sector_deficit.min())
        else:
            sector_deficit_norm = np.zeros(SECTORS)
        # Map each number to its sector deficit score
        sector_score = {n: sector_deficit_norm[sector_of[n]] for n in range(1, number_range + 1)}

        # ------------------------------------------------------------------
        # 2. EWMA recent score (lower EWMA => higher priority)
        # ------------------------------------------------------------------
        # Initialize EWMA array
        ewma = np.zeros(number_range + 1, dtype=float)  # index 0 unused
        # recent_deque is ordered from oldest (left) to newest (right)
        for draw_set in recent_deque:
            presence = np.zeros(number_range + 1, dtype=float)
            for n in draw_set:
                if 1 <= n <= number_range:
                    presence[n] = 1.0
            ewma = GAMMA * ewma + (1.0 - GAMMA) * presence
        # Inverse EWMA to favor numbers with low recent presence
        ewma_score = {n: 1.0 - ewma[n] for n in range(1, number_range + 1)}

        # ------------------------------------------------------------------
        # 3. Hazard (gap) score
        # ------------------------------------------------------------------
        gaps = []
        gap_dict = {}
        for n in range(1, number_range + 1):
            if n in last_seen:
                g = draw_index - last_seen[n]
            else:
                # never seen → use a large gap (treat as max possible)
                g = draw_index + 1
            gaps.append(g)
            gap_dict[n] = g
        gaps_arr = np.array(gaps, dtype=float)
        avg_gap = gaps_arr.mean()
        std_gap = gaps_arr.std() if gaps_arr.std() > 0 else 1.0
        # Standardized gap, then tanh to bound between -1 and 1
        hazard_raw = (gaps_arr - avg_gap) / std_gap
        hazard = np.tanh(hazard_raw)
        hazard_score = {n: hazard[i] for i, n in enumerate(range(1, number_range + 1))}

        # ------------------------------------------------------------------
        # 4. Combine scores
        # ------------------------------------------------------------------
        combined = {}
        for n in range(1, number_range + 1):
            s = (W_S * sector_score[n] +
                 W_E * ewma_score[n] +
                 W_H * hazard_score[n])
            combined[n] = s

        # ------------------------------------------------------------------
        # 5. Return ordering: highest combined score first, break ties by number
        # ------------------------------------------------------------------
        ordered = sorted(range(1, number_range + 1), key=lambda x: (-combined[x], x))
        return ordered

    return rank