NAME = "zrromix3"
DESCRIPTION = "Combines EWMA hotness, overdue gap, recent anchor‑pair co‑occurrence and global z‑score, then applies a greedy sector‑diversity penalty."
SOURCE = "generated 2026-09-16"

import numpy as np
from collections import deque

def create(config):
    # config: LotteryConfig with .number_range, .numbers_drawn, .ticket_size
    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # numbers 1..N
        numbers = list(range(1, config.number_range + 1))
        p0 = config.numbers_drawn / config.number_range  # base probability per number

        # ---------- EWMA hotness ----------
        gamma = 0.96
        ewma_scores = {}
        # recent_deque holds the most recent draws, oldest first; we need most recent first for decay
        rev_deque = list(recent_deque)[::-1]
        for n in numbers:
            ew = 0.0
            wsum = 0.0
            for i, draw_set in enumerate(rev_deque):
                w = gamma ** i
                wsum += w
                if n in draw_set:
                    ew += w
            ewma_scores[n] = ew / wsum if wsum > 0 else 0.0

        # ---------- GAP (overdue) ----------
        expected_gap = 1.0 / p0
        gap_scores = {}
        for n in numbers:
            last = last_seen.get(n, None)
            gap = (draw_index - last) if last is not None else (draw_index + 1)
            gap_scores[n] = (gap - expected_gap) / expected_gap

        # ---------- PAIR (anchor co‑occurrence) ----------
        anchor_k = 3  # number of recent draws used as anchors
        anchor_numbers = set()
        recent_list = list(recent_deque)
        for draw_set in recent_list[-anchor_k:]:
            anchor_numbers.update(draw_set)
        pair_scores = {}
        for n in numbers:
            cnt = 0
            for draw_set in recent_deque:
                if n in draw_set and draw_set.intersection(anchor_numbers):
                    cnt += 1
            pair_scores[n] = cnt

        # ---------- GLOBAL FREQUENCY Z‑score ----------
        var = draw_index * p0 * (1 - p0)
        std = np.sqrt(var) if var > 0 else 1.0
        freq_scores = {}
        for n in numbers:
            freq_scores[n] = (counts.get(n, 0) - draw_index * p0) / std

        # ---------- WEIGHTED COMBINATION ----------
        W_EWMA, W_GAP, W_PAIR, W_FREQ = 1.5, 0.8, 1.0, 0.5
        raw_scores = {}
        for n in numbers:
            raw = (W_EWMA * ewma_scores[n] +
                   W_GAP * gap_scores[n] +
                   W_PAIR * pair_scores[n] +
                   W_FREQ * freq_scores[n])
            raw_scores[n] = raw

        # ---------- GREEDY SECTOR DIVERSITY PENALTY ----------
        W_DIV = 0.3
        sector_counts = [0] * 10  # 80 numbers / 8 per sector
        final_order = []
        remaining = set(numbers)
        while remaining:
            # pick number maximising (raw_score - penalty)
            best = max(remaining, key=lambda n: raw_scores[n] - W_DIV * sector_counts[(n - 1) // 8])
            final_order.append(best)
            sector_counts[(best - 1) // 8] += 1
            remaining.remove(best)
        return final_order
    return rank