NAME = "zewmaove"
DESCRIPTION = "Combines recent EWMA frequency, overdue bonus, and entropy penalty to rank numbers for Mini Lotto (5/42)."
SOURCE = "generated + 2026-09-26"

import numpy as np
from collections import Counter

def create(config):
    # Fixed weights for the three components
    w_recent = 0.5   # weight for recent frequency (EWMA‑like)
    w_overdue = 0.3  # weight for overdue (time since last seen)
    w_entropy = -0.2  # negative weight for entropy deviation (penalty)

    # Small epsilon to avoid division by zero
    eps = 1e-12

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # List of all possible numbers
        numbers = list(range(1, config.number_range + 1))

        # ---- Recent frequency (simple EWMA) ----
        # Count appearances in the recent window
        recent_counter = Counter()
        for draw in recent_deque:
            recent_counter.update(draw)
        # Normalized recent frequency p_recent = recent_count / recent_window
        if recent_window > 0:
            p_recent = {n: recent_counter.get(n, 0) / recent_window for n in numbers}
        else:
            p_recent = {n: 0.0 for n in numbers}

        # ---- Overdue component ----
        # Gap = number of draws since last appearance (or draw_index+1 if never seen)
        max_gap = draw_index + 1  # biggest possible gap at current draw
        gap = {n: (draw_index - last_seen.get(n, -1)) for n in numbers}
        # Normalized overdue (0..1)
        p_overdue = {n: gap[n] / max_gap for n in numbers}

        # ---- Entropy penalty ----
        # Expected uniform probability for each number in a 5/42 draw
        expected_p = config.numbers_drawn / config.number_range
        # Deviation from expected recent probability
        dev = {n: abs(p_recent[n] - expected_p) for n in numbers}
        # Normalize deviation (0..1)
        max_dev = max(dev.values()) if dev else 1.0
        p_entropy = {n: dev[n] / (max_dev + eps) for n in numbers}

        # ---- Composite score ----
        scores = {}
        for n in numbers:
            scores[n] = (
                w_recent * p_recent[n] +
                w_overdue * p_overdue[n] +
                w_entropy * p_entropy[n]
            )

        # Return numbers sorted by descending score, tie‑break by natural order
        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank