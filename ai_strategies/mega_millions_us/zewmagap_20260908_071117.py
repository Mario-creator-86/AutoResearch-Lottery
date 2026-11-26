NAME = "zewmagap"
DESCRIPTION = "Îmbină EWMA a frecvenței recente, penalizare pe baza gap‑ului de la ultima apariție și bonusul de co‑apariție cu ultimele draw‑uri pentru a genera o rangare completă a numerelor 1‑70."
SOURCE = "generativ_strategy_v1_2026-09-08"

import numpy as np

def create(config):
    # config: LotteryConfig — are .number_range, .numbers_drawn, .ticket_size
    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        """Return a full permutation of numbers 1..config.number_range.
        
        Scoring components:
        1. EWMA of recent frequency vs overall frequency.
        2. Recency factor based on gap since last appearance.
        3. Pair‑score from co‑occurrences in the most recent few draws.
        """
        alpha = 0.6  # weight for recent frequency in EWMA
        # Pre‑compute overall frequency (probability) for each number
        overall_freq = {n: counts.get(n, 0) / max(draw_index, 1) for n in range(1, config.number_range + 1)}
        # Recent frequency in the sliding window
        recent_counts = {n: 0 for n in range(1, config.number_range + 1)}
        for draw in recent_deque:
            for n in draw:
                recent_counts[n] += 1
        recent_freq = {n: recent_counts[n] / max(recent_window, 1) for n in recent_counts}
        # EWMA score per number
        ewma_score = {n: alpha * recent_freq[n] + (1 - alpha) * overall_freq[n] for n in recent_counts}
        # Recency factor (larger when gap is larger)
        recency_factor = {}
        for n in range(1, config.number_range + 1):
            last = last_seen.get(n, -1)
            gap = draw_index - last
            recency_factor[n] = 1.0 / (gap + 1)
        # Pair‑score: co‑occurrence with numbers in the most recent draws (up to 5 draws)
        pair_score = {n: 0 for n in range(1, config.number_range + 1)}
        recent_draws_list = list(recent_deque)[-5:] if len(recent_deque) >= 5 else list(recent_deque)
        # Build a set of numbers that appeared in each recent draw for quick lookup
        for draw in recent_draws_list:
            for n in draw:
                # each co‑occurrence counts once per draw (excluding self)
                pair_score[n] += len(draw) - 1
        # Composite score
        composite = {}
        for n in range(1, config.number_range + 1):
            composite[n] = ewma_score[n] * recency_factor[n] + 0.01 * pair_score[n]
        # Return numbers sorted by descending composite score, tie‑break by numeric order
        numbers = list(range(1, config.number_range + 1))
        numbers.sort(key=lambda x: (-composite[x], x))
        return numbers
    return rank