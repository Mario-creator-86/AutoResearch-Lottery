NAME = "zezmix01"
DESCRIPTION = "Combines long‑term frequency, EWMA of recent draws and overdue penalty to rank numbers 1‑40 for Loto 5/40."
SOURCE = "generated 2026-09-08"

import numpy as np


def create(config):
    # Config fields we need: number_range, numbers_drawn, ticket_size (unused here)
    # Fixed hyper‑parameters – chosen to be simple and deterministic
    ALPHA = 0.8          # decay for EWMA (more recent draws matter more)
    W_FREQ = 0.5         # weight for long‑term frequency
    W_EWMA = 0.3         # weight for EWMA recent activity
    W_OVERDUE = 0.2      # weight for overdue (draws since last appearance)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        """Return a full permutation of numbers 1..N ordered by score.

        Parameters
        ----------
        counts: dict[int, int]
            Total occurrences of each number in the whole history.
        recent_deque: collections.deque[set[int]]
            Sets of numbers from the last ``recent_window`` draws, newest at the right end.
        last_seen: dict[int, int]
            Index of the most recent draw where the number appeared (‑1 if never).
        draw_index: int
            Index of the draw for which we are ranking numbers.
        recent_window: int
            Size of the recent window (matches len(recent_deque)).
        """
        # Prepare the list of all possible numbers
        numbers = list(range(1, config.number_range + 1))

        # ---------- Long‑term frequency component ----------
        # Normalise by the maximum count to keep the scale 0‑1
        max_count = max(counts.values()) if counts else 1
        freq_score = {n: counts.get(n, 0) / max_count for n in numbers}

        # ---------- EWMA of recent appearances ----------
        # Compute weighted sum of appearances in recent_deque with exponential decay
        ewma_score = {}
        # Pre‑compute decay factors for positions 0 .. recent_window‑1 (0 = most recent draw)
        decay_factors = [ALPHA ** i for i in range(len(recent_deque))]
        for n in numbers:
            weighted_sum = 0.0
            for i, draw_set in enumerate(reversed(recent_deque)):
                if n in draw_set:
                    weighted_sum += decay_factors[i]
            # Normalise by the sum of decay factors (constant for a given window size)
            ewma_score[n] = weighted_sum / sum(decay_factors) if decay_factors else 0.0

        # ---------- Overdue component ----------
        # Number of draws since last appearance; larger value -> larger boost
        # Normalise by the current draw index (or recent_window) to keep 0‑1 range
        overdue_score = {}
        # Use draw_index as denominator; if draw_index == 0 avoid division by zero
        denominator = max(draw_index, 1)
        for n in numbers:
            last = last_seen.get(n, -1)
            gap = draw_index - last if last >= 0 else denominator
            overdue_score[n] = gap / denominator

        # ---------- Combine components ----------
        combined_score = {}
        for n in numbers:
            combined_score[n] = (
                W_FREQ * freq_score[n] +
                W_EWMA * ewma_score[n] +
                W_OVERDUE * overdue_score[n]
            )

        # Sort numbers by descending combined score, break ties by the natural order
        return sorted(numbers, key=lambda n: (-combined_score[n], n))

    return rank