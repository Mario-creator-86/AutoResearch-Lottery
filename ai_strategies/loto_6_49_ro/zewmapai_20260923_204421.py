NAME = "zewmapai"
DESCRIPTION = "Combines recent EWMA frequency, recent co‑occurrence pair strength, and overdue gap to rank numbers 1‑49."
SOURCE = "generated + 2026-09-23"

import numpy as np
from collections import deque


def create(config):
    # Weights for the three components (can be tuned later)
    W_EWMA = 0.5
    W_PAIR = 0.3
    W_OVERDUE = 0.2
    # EWMA blending factor (recent vs historic)
    ALPHA = 0.6  # recent weight inside EWMA component

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        """Return a full permutation of numbers 1..config.number_range.

        counts: dict {num: total appearances in the whole history}
        recent_deque: collections.deque of set(int) – the last `recent_window` draws
        last_seen: dict {num: draw_index of last appearance; missing => never seen}
        draw_index: int – index of the current draw (0‑based)
        recent_window: int – size of the recent window (may be larger than len(recent_deque) at start)
        """
        numbers = list(range(1, config.number_range + 1))
        # ---- 1. recent frequencies (EWMA part) ----
        recent_counts = {n: 0 for n in numbers}
        for draw_set in recent_deque:
            for n in draw_set:
                recent_counts[n] += 1
        recent_draws = len(recent_deque) if recent_deque else 0
        # Avoid division by zero
        recent_frac = {n: recent_counts[n] / recent_draws if recent_draws else 0.0 for n in numbers}
        total_draws_sofar = max(draw_index, 1)  # at least 1 to avoid zero
        historic_frac = {n: counts.get(n, 0) / total_draws_sofar for n in numbers}
        ewma_score = {n: ALPHA * recent_frac[n] + (1 - ALPHA) * historic_frac[n] for n in numbers}
        # Normalise EWMA scores to [0,1]
        max_ewma = max(ewma_score.values()) if ewma_score else 1.0
        ewma_norm = {n: ewma_score[n] / max_ewma for n in numbers}

        # ---- 2. pair co‑occurrence strength (within recent window) ----
        pair_raw = {n: 0 for n in numbers}
        for draw_set in recent_deque:
            size = len(draw_set)
            if size < 2:
                continue
            for n in draw_set:
                # each partner contributes 1
                pair_raw[n] += size - 1
        # Normalise pair scores
        max_pair = max(pair_raw.values()) if pair_raw else 1
        pair_norm = {n: pair_raw[n] / max_pair for n in numbers}

        # ---- 3. overdue bonus ----
        overdue_raw = {}
        for n in numbers:
            last = last_seen.get(n, -1)
            gap = draw_index - last if last >= 0 else draw_index + 1
            overdue_raw[n] = gap
        max_gap = max(overdue_raw.values()) if overdue_raw else 1
        overdue_norm = {n: overdue_raw[n] / max_gap for n in numbers}

        # ---- 4. combine ----
        total_score = {}
        for n in numbers:
            total_score[n] = (
                W_EWMA * ewma_norm[n] +
                W_PAIR * pair_norm[n] +
                W_OVERDUE * overdue_norm[n]
            )
        # Return numbers sorted by descending total_score, then by number ascending for reproducibility
        return sorted(numbers, key=lambda n: (-total_score[n], n))

    return rank