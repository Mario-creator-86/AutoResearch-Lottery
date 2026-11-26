import collections
import numpy as np

NAME = "zewmaove"
DESCRIPTION = "Combines long‑term frequencies, EWMA recent frequencies, overdue age and co‑occurrence with historically hot numbers to rank all 49 balls."
SOURCE = "custom_strategy_v1 + 2026-09-26"

def create(config):
    """Return a ranking function for Lotto PL (6/49).

    Parameters
    ----------
    config: LotteryConfig
        Must expose .number_range, .numbers_drawn and .ticket_size.
    """
    # parameters (can be tuned later via walk‑forward)
    alpha = 0.8               # EWMA decay
    w_counts = 0.4            # weight for total frequency
    w_ewma = 0.3              # weight for EWMA recent frequency
    w_overdue = 0.2           # weight for overdue factor
    w_pair = 0.1              # weight for co‑occurrence with hot numbers
    hot_ratio = 0.3           # top % of numbers considered hot

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, config.number_range + 1))
        total_counts = sum(counts.values()) if counts else 0
        if total_counts == 0:
            # fallback: uniform scores
            return numbers

        # 1) Long‑term frequency component (normalized)
        freq = np.array([counts.get(n, 0) / total_counts for n in numbers])
        # 2) EWMA of recent draws
        ewma_raw = np.zeros(config.number_range)
        weight_sum = 0.0
        for i, draw_set in enumerate(reversed(recent_deque)):
            w = alpha ** i
            weight_sum += w
            for n in draw_set:
                ewma_raw[n - 1] += w
        ewma = ewma_raw / weight_sum if weight_sum > 0 else ewma_raw
        # normalize EWMA to [0,1]
        if ewma.max() > 0:
            ewma = ewma / ewma.max()
        # 3) Overdue factor – larger age ⇒ larger score (log‑scaled)
        overdue_raw = np.zeros(config.number_range)
        for idx, n in enumerate(numbers):
            last = last_seen.get(n, -1)
            age = draw_index - last if last != -1 else draw_index + 1
            # log1p to keep growth sub‑linear
            overdue_raw[idx] = np.log1p(age)
        # normalize overdue
        overdue = overdue_raw / overdue_raw.max()
        # 4) Pair/co‑occurrence with hot numbers
        # Determine hot set based on overall counts
        k_hot = max(1, int(config.number_range * hot_ratio))
        hot_numbers = set(sorted(counts, key=lambda x: counts[x], reverse=True)[:k_hot])
        pair_raw = np.zeros(config.number_range)
        for draw_set in recent_deque:
            intersect_hot = draw_set & hot_numbers
            for n in draw_set:
                # number of hot companions (exclude itself if hot)
                companions = len(intersect_hot - {n})
                pair_raw[n - 1] += companions
        # normalize pair component
        if pair_raw.max() > 0:
            pair = pair_raw / pair_raw.max()
        else:
            pair = pair_raw
        # Combine all components
        final_score = (
            w_counts * freq +
            w_ewma * ewma +
            w_overdue * overdue +
            w_pair * pair
        )
        # Return numbers sorted by descending final_score, tie‑break by numeric order
        sorted_numbers = sorted(numbers, key=lambda n: (-final_score[n - 1], n))
        return sorted_numbers

    return rank