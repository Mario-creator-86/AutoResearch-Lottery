NAME = "zewmasec"
DESCRIPTION = "Îmbină EWMA al frecvenţei totale, hot‑recent, penalizare pentru perechi populare și echilibrare sectorială pentru a genera o ordine de numere cu diversificare şi potenţial de hit crescut."
SOURCE = "ChatGPT 2026-09-16"

import numpy as np
from collections import defaultdict


def create(config):
    """Return a ranking function for Lotto 6/49.
    
    Parameters
    ----------
    config : LotteryConfig
        .number_range (int) – e.g. 49
        .numbers_drawn (int) – e.g. 6
        .ticket_size (int) – not used here
    """

    # Hyper‑parameters – chosen to be deterministic and fast
    GAMMA = 0.3                 # decay factor for EWMA‑like effect (used on total counts)
    W_EWMA = 1.0                # weight for EWMA / deviation score
    W_RECENT = 0.8              # weight for recent‑window hits
    W_PAIR_PEN = 0.6            # weight for pair penalty (negative)
    W_SECTOR = 0.4              # weight for sector balance (positive for under‑hot sectors)
    SECTOR_SIZE = 7            # we split 1..49 into 7 equal sectors of 7 numbers

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, config.number_range + 1))

        # ---- 1. Global frequency statistics ----
        total_counts = np.array([counts.get(n, 0) for n in numbers], dtype=float)
        global_mean = total_counts.mean()
        global_std = total_counts.std(ddof=0) if total_counts.size > 0 else 1.0
        # EWMA‑like deviation (hot‑cold) – centred and scaled
        ewma_score = (total_counts - global_mean) / (global_std if global_std != 0 else 1.0)

        # ---- 2. Recent hotness (appearances in the sliding window) ----
        # recent_deque contains sets of numbers from the last `recent_window` draws
        recent_counts = np.zeros(config.number_range, dtype=float)
        for draw in recent_deque:
            for n in draw:
                recent_counts[n - 1] += 1  # index shift
        # Normalise by window size to keep comparable across windows
        recent_score = recent_counts / max(1, recent_window)

        # ---- 3. Pair penalty (avoid overly frequent pairs) ----
        pair_freq = defaultdict(int)
        for draw in recent_deque:
            draw_list = list(draw)
            for i in range(len(draw_list)):
                for j in range(i + 1, len(draw_list)):
                    a, b = sorted((draw_list[i], draw_list[j]))
                    pair_freq[(a, b)] += 1
        # For each number compute the sum of frequencies of pairs it participates in
        pair_penalty = np.zeros(config.number_range, dtype=float)
        for (a, b), cnt in pair_freq.items():
            pair_penalty[a - 1] += cnt
            pair_penalty[b - 1] += cnt
        # Higher sum => stronger penalty, normalise
        if pair_penalty.max() > 0:
            pair_penalty = pair_penalty / pair_penalty.max()
        pair_penalty = -pair_penalty  # negative because we want to subtract this score

        # ---- 4. Sector balance ----
        # Compute sector mean hotness (average EWMA score per sector)
        sector_scores = np.zeros((config.number_range // SECTOR_SIZE,))
        for sec_idx in range(sector_scores.shape[0]):
            start = sec_idx * SECTOR_SIZE
            end = start + SECTOR_SIZE
            sector_scores[sec_idx] = ewma_score[start:end].mean()
        # Global sector mean (should be close to 0 after centring) – we reward numbers from sectors below 0
        sector_bonus = np.zeros(config.number_range, dtype=float)
        for n in numbers:
            sec_idx = (n - 1) // SECTOR_SIZE
            # If sector mean is negative (cold sector) give a positive bonus proportional to its coldness
            sector_bonus[n - 1] = -sector_scores[sec_idx] if sector_scores[sec_idx] < 0 else 0.0

        # ---- 5. Combine scores ----
        total_score = (
            W_EWMA * ewma_score +
            W_RECENT * recent_score +
            W_PAIR_PEN * pair_penalty +
            W_SECTOR * sector_bonus
        )

        # Build a dict for sorting; break ties by the natural number order (ascending)
        score_dict = {n: total_score[n - 1] for n in numbers}
        ranked = sorted(numbers, key=lambda n: (-score_dict[n], n))
        return ranked

    return rank