NAME = "zpairewm"
DESCRIPTION = "Combines total frequency, recent EWMA, overdue gap and co‑occurrence with hot numbers to rank 1‑24."
SOURCE = "custom_strategy 2026-09-08"

import numpy as np
from collections import deque

def create(config):
    # config: LotteryConfig — are .number_range, .numbers_drawn, .ticket_size
    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # numbers 1..R
        numbers = np.arange(1, config.number_range + 1)
        R = config.number_range

        # 1) total historic counts (baseline)
        total = np.array([counts.get(int(n), 0) for n in numbers], dtype=float)

        # 2) recent raw counts (within recent_window draws)
        recent_counts = np.zeros(R, dtype=float)
        for draw in recent_deque:
            for num in draw:
                recent_counts[num - 1] += 1.0

        # 3) EWMA over recent draws (more weight to newer draws)
        gamma = 0.93  # decay factor, can be tuned
        ewma = np.zeros(R, dtype=float)
        weight_sum = 0.0
        # recent_deque[0] = oldest, recent_deque[-1] = newest → iterate reversed for newest first
        for i, draw in enumerate(reversed(recent_deque)):
            w = gamma ** i
            weight_sum += w
            for num in draw:
                ewma[num - 1] += w
        if weight_sum > 0:
            ewma /= weight_sum

        # 4) overdue (draws since last appearance)
        overdue = np.array([draw_index - last_seen.get(int(n), -1) for n in numbers], dtype=float)

        # 5) co‑occurrence with globally hot numbers (top‑5 by total count)
        top_k = 5
        hot_idx = np.argsort(-total)[:top_k]
        hot_set = set(int(numbers[i]) for i in hot_idx)
        pair_score = np.zeros(R, dtype=float)
        for draw in recent_deque:
            draw_set = set(draw)
            intersect = draw_set & hot_set
            if intersect:
                # each number in the draw gets a boost proportional to how many hot numbers are present
                boost = len(intersect)
                for num in draw_set:
                    pair_score[num - 1] += boost

        # Normalisation to 0‑1 range for each component
        def _norm(arr):
            m = arr.max()
            return arr / m if m > 0 else arr

        total_n = _norm(total)
        recent_n = _norm(recent_counts)
        ewma_n = _norm(ewma)
        overdue_n = _norm(overdue)
        pair_n = _norm(pair_score)

        # Weighted sum (weights chosen empirically, can be tuned later)
        scores = (
            1.0 * total_n +
            0.8 * recent_n +
            0.6 * ewma_n +
            0.4 * overdue_n +
            0.5 * pair_n
        )

        # Sort numbers by descending score, break ties by ascending numeric value
        # np.lexsort uses last key as primary, so we give -scores first, then numbers
        order = np.lexsort((numbers, -scores))
        ranked = numbers[order].tolist()
        return ranked
    return rank