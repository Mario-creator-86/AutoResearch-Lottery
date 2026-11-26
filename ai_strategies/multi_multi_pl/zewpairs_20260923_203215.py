NAME = "zewpairs"
DESCRIPTION = "Combines recent frequency, sector overdue and pair co‑occurrence with hot numbers to rank 1..80."
SOURCE = "ewpairsec_strategy + 2026‑09‑23"

import numpy as np
from collections import deque

def create(config):
    # config: LotteryConfig – .number_range, .numbers_drawn, .ticket_size
    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # numbers list
        numbers = np.arange(1, config.number_range + 1)
        # ---- 1. Recent frequency (within recent_window) ----
        recent_counts = np.zeros(config.number_range + 1, dtype=int)  # 1‑based index
        for draw_set in recent_deque:
            for n in draw_set:
                recent_counts[n] += 1
        recent_freq = recent_counts[1:]  # shape (N,)
        # normalize to [0,1]
        if recent_window > 0:
            recent_freq_norm = recent_freq / recent_window
        else:
            recent_freq_norm = np.zeros_like(recent_freq, dtype=float)
        # ---- 2. Sector overdue score ----
        sector_size = 10
        n_sectors = (config.number_range + sector_size - 1) // sector_size
        # compute most recent appearance per sector
        sector_last = np.full(n_sectors, -1, dtype=int)
        for n in range(1, config.number_range + 1):
            sector_idx = (n - 1) // sector_size
            ls = last_seen.get(n, -1)
            if ls > sector_last[sector_idx]:
                sector_last[sector_idx] = ls
        # overdue = draw_index - sector_last
        sector_overdue = draw_index - sector_last  # larger = more overdue
        # transform to score (inverse, +1 to avoid div‑0)
        sector_score = 1.0 / (sector_overdue + 1)
        # map each number to its sector score
        sector_score_per_num = np.empty(config.number_range, dtype=float)
        for n in range(1, config.number_range + 1):
            sector_idx = (n - 1) // sector_size
            sector_score_per_num[n-1] = sector_score[sector_idx]
        # ---- 3. Pair co‑occurrence with top hot numbers ----
        # Determine top K hot numbers by total counts
        K = 15
        total_counts_arr = np.array([counts.get(i, 0) for i in range(1, config.number_range + 1)])
        top_k_indices = np.argpartition(-total_counts_arr, K-1)[:K]
        top_k_set = set(int(i+1) for i in top_k_indices)  # convert to actual numbers
        pair_score = np.zeros(config.number_range, dtype=float)
        for draw_set in recent_deque:
            intersect_hot = draw_set & top_k_set
            if not intersect_hot:
                continue
            for n in draw_set:
                pair_score[n-1] += len(intersect_hot)
        # Normalize pair_score
        if recent_window > 0:
            pair_score_norm = pair_score / recent_window
        else:
            pair_score_norm = np.zeros_like(pair_score, dtype=float)
        # ---- 4. Combine scores with predetermined weights ----
        w_recent = 0.5
        w_sector = 0.3
        w_pair   = 0.2
        total_score = (w_recent * recent_freq_norm) + (w_sector * sector_score_per_num) + (w_pair * pair_score_norm)
        # ---- 5. Sort numbers by descending score, tie‑break by numeric order ----
        sorted_idx = np.lexsort((numbers, -total_score))  # negative for descending
        ranked_numbers = numbers[sorted_idx].tolist()
        return ranked_numbers
    return rank