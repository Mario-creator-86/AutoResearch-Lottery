NAME = "zmultisi"
DESCRIPTION = "Combină frecvență totală, recentă, overdue, balans pe sectoare și entropie pentru ranking"
SOURCE = "custom_strategy_2024-09-26"

import numpy as np
from collections import deque

def create(config):
    # Config attributes used: number_range, numbers_drawn, ticket_size (unused)
    # constants – pot fi ajustate prin sweep‑uri viitoare
    w_freq = 1.0      # greutate frecvență totală
    w_recent = 0.8    # greutate frecvență recentă
    w_overdue = 0.5   # greutate overdue (invers)
    w_sector = 0.3    # greutate corecție sector
    w_entropy = 0.2   # greutate entropie (favorizează distribuții echilibrate)

    # Pre‑calc sector map (1‑40 → 0‑3)
    sector_of = {n: (n - 1) // 10 for n in range(1, config.number_range + 1)}

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # numbers list
        numbers = list(range(1, config.number_range + 1))
        total_draws = draw_index  # draws processed so far (draw_index starts at 0)
        # total tickets drawn = draw_index * numbers_drawn (not exact but proportional)
        # 1) total frequency (normalized)
        if total_draws > 0:
            norm_counts = {n: counts.get(n, 0) / total_draws for n in numbers}
        else:
            norm_counts = {n: 0.0 for n in numbers}

        # 2) recent frequency (absolute count in recent_deque)
        recent_counts = {n: 0 for n in numbers}
        for draw_set in recent_deque:
            for n in draw_set:
                recent_counts[n] += 1

        # 3) overdue score (inverse of time since last appearance)
        overdue_scores = {}
        for n in numbers:
            last = last_seen.get(n, -1)
            gap = draw_index - last if last != -1 else draw_index + 1
            overdue_scores[n] = 1.0 / (gap + 1)

        # 4) sector imbalance – compute recent hits per sector
        sector_recent_hits = {s: 0 for s in range(4)}
        for n, rc in recent_counts.items():
            sector_recent_hits[sector_of[n]] += rc
        sector_scores = {}
        for n in numbers:
            sector = sector_of[n]
            sector_scores[n] = 1.0 / (sector_recent_hits[sector] + 1)

        # 5) entropy‑based score – encourage numbers with low recent probability
        # flatten recent numbers
        total_recent_numbers = recent_window * config.numbers_drawn
        if total_recent_numbers > 0:
            probs = {n: recent_counts[n] / total_recent_numbers for n in numbers}
            # avoid log(0) by substituting a tiny epsilon
            eps = 1e-12
            entropy_scores = {n: - (p if p > 0 else eps) * np.log(p if p > 0 else eps) for n, p in probs.items()}
        else:
            entropy_scores = {n: 0.0 for n in numbers}

        # Composite score
        scores = {}
        for n in numbers:
            scores[n] = (
                w_freq * norm_counts[n]
                + w_recent * recent_counts[n]
                + w_overdue * overdue_scores[n]
                + w_sector * sector_scores[n]
                + w_entropy * entropy_scores[n]
            )

        # Return numbers sorted descending by score, tie‑breaker by natural order
        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank