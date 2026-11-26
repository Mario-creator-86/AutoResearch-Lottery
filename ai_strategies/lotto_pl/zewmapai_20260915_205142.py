NAME = "zewmapai"
DESCRIPTION = "Combine EWMA recent frequency, pair co‑occurrence strength and overdue gap to rank numbers"
SOURCE = "generated + 2026-09-15"

import numpy as np
from collections import deque

def create(config):
    # Hyper‑parameters (fixed, chosen empirically but kept constant for determinism)
    GAMMA = 0.96          # decay for EWMA recent window
    W_EWMA = 0.45         # weight for EWMA recent frequency
    W_PAIR = 0.35         # weight for pair co‑occurrence score
    W_GAP  = 0.20         # weight for overdue gap score

    numbers = list(range(1, config.number_range + 1))

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # ---------- 1. EWMA recent frequency ----------
        # recent_deque[-1] is the most recent draw (newest_last orientation)
        ewma_scores = {n: 0.0 for n in numbers}
        weight = 1.0
        # iterate from newest to oldest in the deque
        for draw in reversed(recent_deque):
            for n in draw:
                ewma_scores[n] += weight
            weight *= GAMMA
        # normalise by sum of weights to keep scale comparable
        total_weight = (1 - GAMMA ** len(recent_deque)) / (1 - GAMMA) if recent_deque else 0.0
        if total_weight > 0:
            for n in ewma_scores:
                ewma_scores[n] /= total_weight

        # ---------- 2. Pair co‑occurrence strength ----------
        # Build a co‑occurrence matrix for the recent window (symmetric)
        pair_matrix = np.zeros((config.number_range + 1, config.number_range + 1), dtype=float)
        for draw in recent_deque:
            draw_list = list(draw)
            for i in range(len(draw_list)):
                a = draw_list[i]
                for j in range(i + 1, len(draw_list)):
                    b = draw_list[j]
                    pair_matrix[a, b] += 1.0
                    pair_matrix[b, a] += 1.0
        # Pair score for a number = sum of its co‑occurrences (again EWMA‑weighted by recency)
        # We reuse the same EWMA weights as above to give more importance to recent pairs.
        pair_scores = {n: 0.0 for n in numbers}
        weight = 1.0
        for draw in reversed(recent_deque):
            draw_list = list(draw)
            for i in range(len(draw_list)):
                a = draw_list[i]
                for j in range(i + 1, len(draw_list)):
                    b = draw_list[j]
                    pair_scores[a] += weight
                    pair_scores[b] += weight
            weight *= GAMMA
        # Normalise pair scores by the maximum possible (2 * recent_window) to keep them in [0,1]
        max_pair = 2 * recent_window if recent_window > 0 else 1
        for n in pair_scores:
            pair_scores[n] /= max_pair

        # ---------- 3. Overdue gap score ----------
        # draws since last appearance (larger -> more overdue). We cap at recent_window*2 for normalisation.
        max_gap = recent_window * 2 if recent_window > 0 else 1
        gap_scores = {}
        for n in numbers:
            last = last_seen.get(n, -1)
            if last == -1:
                gap = draw_index  # never seen yet, treat as maximum overdue
            else:
                gap = draw_index - last
            # Transform to a score that grows with gap, bounded to 1
            gap_scores[n] = min(gap / max_gap, 1.0)

        # ---------- 4. Combine all components ----------
        combined_scores = {}
        for n in numbers:
            combined = (
                W_EWMA * ewma_scores.get(n, 0.0) +
                W_PAIR * pair_scores.get(n, 0.0) +
                W_GAP  * gap_scores.get(n, 0.0)
            )
            combined_scores[n] = combined

        # ---------- 5. Return ranking ----------
        # Higher combined score = higher rank; tie‑break by numeric order.
        return sorted(numbers, key=lambda n: (-combined_scores[n], n))

    return rank