NAME = "zewepair"
DESCRIPTION = "Combines EWMA recent frequency, recent pair co‑occurrence, deviation from uniform recent frequency and a long‑term z‑score to rank numbers 1..42."
SOURCE = "ewepair_strategy.py @ 2026-09-08"

import numpy as np
from collections import deque

def create(config):
    # Hyper‑parameters (can be tuned later via external scans)
    GAMMA = 0.85          # decay factor for EWMA / recent weighting
    ALPHA = 1.0           # weight for EWMA recent frequency
    BETA = 0.7            # weight for recent pair co‑occurrence
    DELTA = 0.3           # weight for deviation from uniform recent frequency
    Z_WEIGHT = 0.2        # weight for long‑term z‑score

    # Pre‑computed constants
    N_RANGE = config.number_range          # 42 for Mini Lotto
    K_DRAW = config.numbers_drawn          # 5 numbers per draw

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        """Return a full permutation of numbers 1..N ordered by the strategy score.

        counts: dict {num: total appearances in whole history}
        recent_deque: collections.deque of sets, length = recent_window (oldest→newest)
        last_seen: dict {num: draw index of the most recent appearance}
        draw_index: int, index of the draw we are about to predict (unused here)
        recent_window: int, size of recent_deque (provided for safety)
        """
        # ------- 1. Prepare arrays for fast vectorised work -------
        nums = np.arange(1, N_RANGE + 1)
        # total counts vector
        total_counts = np.array([counts.get(int(n), 0) for n in nums], dtype=float)

        # ------- 2. EWMA recent frequency -------
        # recent_deque is ordered oldest -> newest, we want weight gamma^age where age=0 for newest
        ewma = np.zeros(N_RANGE, dtype=float)
        pair_score = np.zeros(N_RANGE, dtype=float)
        recent_counts = np.zeros(N_RANGE, dtype=float)  # simple recent frequency (for delta)
        weight = 1.0  # weight for newest draw (age 0)
        # iterate from newest to oldest
        for draw in reversed(recent_deque):
            arr = np.zeros(N_RANGE, dtype=int)
            for n in draw:
                idx = n - 1
                arr[idx] = 1
                # pair contribution: each other number in the same draw adds weight
                pair_score[idx] += weight * (len(draw) - 1)
            ewma += weight * arr
            recent_counts += arr
            weight *= GAMMA  # decay for older draw
        # Normalise EWMA by sum of weights to keep scale comparable
        sum_weights = (1 - GAMMA**len(recent_deque)) / (1 - GAMMA) if GAMMA != 1 else len(recent_deque)
        if sum_weights > 0:
            ewma /= sum_weights
        # Normalise pair_score similarly (weights already applied)
        # No further normalisation – its magnitude is already comparable to ewma

        # ------- 3. Deviation from uniform recent frequency (delta) -------
        # Expected proportion per number in a single draw = K_DRAW / N_RANGE
        expected_recent = K_DRAW / N_RANGE
        recent_freq = recent_counts / max(1, recent_window)  # avoid div by 0
        delta = recent_freq - expected_recent

        # ------- 4. Long‑term z‑score -------
        mean_counts = total_counts.mean() if total_counts.size else 0.0
        std_counts = total_counts.std(ddof=0) if total_counts.size else 0.0
        if std_counts == 0:
            z_score = np.zeros(N_RANGE)
        else:
            z_score = (total_counts - mean_counts) / std_counts

        # ------- 5. Composite score -------
        score = (
            ALPHA * ewma +
            BETA * pair_score +
            DELTA * delta +
            Z_WEIGHT * z_score
        )

        # ------- 6. Produce deterministic ranking -------
        # argsort on negative score gives descending order, tie‑break by smaller number
        order = np.lexsort((nums, -score))  # first key = -score, second = nums (ascending)
        ranked_numbers = nums[order].tolist()
        return ranked_numbers

    return rank