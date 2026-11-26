NAME = "zewmasec"
DESCRIPTION = "Combines EWMA recent frequency, overdue boost, sector imbalance, and recent pair popularity to rank numbers 1‑49."
SOURCE = "generated 2026-09-08"

import numpy as np
from collections import Counter

def create(config):
    # config: .number_range, .numbers_drawn, .ticket_size (unused here)
    number_range = config.number_range
    sector_size = 7  # 49 -> 7 sectors of 7 numbers each
    gamma = 0.95  # EWMA decay factor
    w_ewma = 0.4
    w_overdue = 0.2
    w_sector = 0.2
    w_pair = 0.2

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # ---- Preparare date ----
        numbers = np.arange(1, number_range + 1)
        # 1) EWMA din frecvența recentă
        # recent_deque conține seturi, ordinea este oldest -> newest (standard deque)
        ewma_weights = []
        ewma_counts = np.zeros(number_range, dtype=float)
        weight = 1.0
        for draw_set in reversed(recent_deque):  # newest first
            for n in draw_set:
                ewma_counts[n - 1] += weight
            ewma_weights.append(weight)
            weight *= gamma
        if ewma_weights:
            total_weight = sum(ewma_weights)
            ewma_scores = ewma_counts / total_weight
        else:
            ewma_scores = np.zeros(number_range)

        # 2) Overdue – numerele cu cel mai lung timp fără apariție
        overdue_days = np.array([
            draw_index - last_seen.get(n, -1) for n in numbers
        ], dtype=float)
        # Normalize to max overdue in current window (or overall)
        max_overdue = overdue_days.max() if overdue_days.max() > 0 else 1.0
        overdue_scores = overdue_days / max_overdue

        # 3) Sector imbalance în fereastra recentă
        sector_counts = np.zeros(sector_size, dtype=int)
        for draw_set in recent_deque:
            for n in draw_set:
                sector_counts[(n - 1) // sector_size] += 1
        # Expected count per sector in recent window (each draw has 6 numbers)
        expected_per_sector = recent_window * 6 / sector_size if recent_window > 0 else 0
        sector_imbalance = expected_per_sector - sector_counts  # pozitiv = sub‑reprezentat
        # Normalizare la interval [-1,1]
        if sector_imbalance.ptp() == 0:
            sector_norm = np.zeros(sector_size)
        else:
            sector_norm = (sector_imbalance - sector_imbalance.min()) / sector_imbalance.ptp() * 2 - 1
        sector_scores = np.array([sector_norm[(n - 1) // sector_size] for n in numbers])

        # 4) Popularitate perechi în fereastra recentă
        pair_counter = Counter()
        for draw_set in recent_deque:
            draw_list = list(draw_set)
            for i in range(len(draw_list)):
                for j in range(i + 1, len(draw_list)):
                    a, b = sorted((draw_list[i], draw_list[j]))
                    pair_counter[(a, b)] += 1
        pair_scores = np.zeros(number_range, dtype=float)
        if pair_counter:
            max_pair_cnt = max(pair_counter.values())
            for (a, b), cnt in pair_counter.items():
                pair_scores[a - 1] += cnt / max_pair_cnt
                pair_scores[b - 1] += cnt / max_pair_cnt
        # Normalizare simplă (max already 1)

        # ---- Combinație scoruri ----
        total_score = (
            w_ewma * ewma_scores +
            w_overdue * overdue_scores +
            w_sector * sector_scores +
            w_pair * pair_scores
        )
        # În caz de egalitate, păstrează ordinea naturală (număr mai mic prima)
        ranking = sorted(numbers, key=lambda n: (-total_score[n - 1], n))
        return ranking

    return rank