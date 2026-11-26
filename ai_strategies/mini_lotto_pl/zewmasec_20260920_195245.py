NAME = "zewmasec"
DESCRIPTION = "Combines global frequency, EWMA of recent draws and sector age to rank numbers"
SOURCE = "generated 2026-09-20"

import numpy as np
from collections import deque

def create(config):
    # config: LotteryConfig with .number_range, .numbers_drawn, .ticket_size
    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        """Return a permutation of numbers 1..config.number_range.
        
        counts: dict {num: total appearances in full history}
        recent_deque: deque of set() with the numbers of the last recent_window draws
        last_seen: dict {num: draw index of last appearance (or -1 if never)}
        draw_index: current draw index (integer)
        recent_window: size of the recent window (int)
        """
        # 1. Prepare list of numbers
        numbers = list(range(1, config.number_range + 1))
        total_counts = sum(counts.values()) if counts else 0
        # Avoid division by zero
        total_counts = max(total_counts, 1)

        # 2. Global normalized frequency score (weight 0.4)
        freq_score = {n: counts.get(n, 0) / total_counts for n in numbers}

        # 3. EWMA over recent draws (weight 0.4)
        #   EWMA_n = sum_{k=0}^{L-1} alpha * (1-alpha)^k * I[n in draw_{-k}]
        #   where draw_{0} is the most recent draw.
        alpha = 0.3
        ewma_score = {n: 0.0 for n in numbers}
        # recent_deque is ordered from oldest (left) to newest (right) by default;
        # we iterate reversed to have k=0 for the most recent draw.
        for k, draw_set in enumerate(reversed(recent_deque)):
            weight = alpha * ((1 - alpha) ** k)
            for n in draw_set:
                ewma_score[n] += weight
        # Normalize EWMA to [0,1] (max possible weight = alpha * (1 - (1-alpha)^L) / (1-(1-alpha)))
        max_ewma = alpha * (1 - (1 - alpha) ** len(recent_deque)) / (1 - (1 - alpha))
        max_ewma = max(max_ewma, 1e-12)  # safety
        for n in numbers:
            ewma_score[n] /= max_ewma

        # 4. Sector age score (weight 0.2)
        # Divide the range into 6 equal sectors (size 7). Compute for each sector the most recent appearance.
        sector_age = {}
        sector_size = 7
        for n in numbers:
            sector = (n - 1) // sector_size
            sector_age.setdefault(sector, -1)
        # Find latest draw index per sector
        latest_in_sector = {s: -1 for s in sector_age}
        for n in numbers:
            sector = (n - 1) // sector_size
            ls = last_seen.get(n, -1)
            if ls > latest_in_sector[sector]:
                latest_in_sector[sector] = ls
        # Age = draw_index - latest_seen_in_sector (bigger age => sector neglected)
        for n in numbers:
            sector = (n - 1) // sector_size
            age = draw_index - latest_in_sector[sector]
            # Normalise by current draw_index+1 to keep within [0,1]
            sector_age[n] = age / (draw_index + 1)

        # 5. Combine scores with fixed weights
        w_freq, w_ewma, w_sector = 0.4, 0.4, 0.2
        combined = {}
        for n in numbers:
            combined[n] = (
                w_freq * freq_score[n]
                + w_ewma * ewma_score[n]
                + w_sector * sector_age[n]
            )
        # 6. Return numbers sorted by descending combined score, break ties by natural order
        return sorted(numbers, key=lambda n: (-combined[n], n))
    return rank