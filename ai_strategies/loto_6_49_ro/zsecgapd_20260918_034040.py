NAME = "zsecgapd"
DESCRIPTION = "Sector deficit + overdue gap + deviation scoring"
SOURCE = "secgapdev_2024-09-18"

import numpy as np
from collections import defaultdict

def create(config):
    # Config: .number_range (e.g., 49), .numbers_drawn (e.g., 6), .ticket_size (unused here)
    sector_size = 7  # 49 / 7 = 7 sectors, each of size 7 numbers
    num_sectors = (config.number_range + sector_size - 1) // sector_size

    # Weights – modest, tunable via external grid, but fixed here for determinism
    W_SEC = 0.6   # sector deficit weight
    W_GAP = 0.3   # overdue gap weight
    W_DEV = 0.1   # long‑term deviation weight

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # ----- Preparare valori globale -----
        numbers = list(range(1, config.number_range + 1))
        total_draws = sum(counts.values()) // config.numbers_drawn  # aproximare a numărului de extrageri totale
        expected_total_per_num = total_draws * config.numbers_drawn / config.number_range
        # Deviatia standard a distribuției totale (folosim numpy pentru viteză)
        freq_array = np.array([counts.get(n, 0) for n in numbers], dtype=float)
        mean_freq = freq_array.mean()
        std_freq = freq_array.std() if freq_array.std() > 0 else 1.0

        # ----- 1. Deficit sectorial recent -----
        # Contorizăm apariţiile în fereastra recentă pentru fiecare număr
        recent_counts = defaultdict(int)
        for draw in recent_deque:
            for n in draw:
                recent_counts[n] += 1
        # Numărul așteptat de apariţii în fereastra recentă pentru un sector
        exp_sector_hits = recent_window * config.numbers_drawn / config.number_range * sector_size
        sector_deficit = np.zeros(num_sectors, dtype=float)
        for s in range(num_sectors):
            start = s * sector_size + 1
            end = min(start + sector_size, config.number_range + 1)
            actual = sum(recent_counts.get(n, 0) for n in range(start, end))
            deficit = max(0.0, exp_sector_hits - actual)  # doar sub‑reprezentarea contează
            sector_deficit[s] = deficit / exp_sector_hits if exp_sector_hits > 0 else 0.0

        # ----- 2. Overdue gap normalizat -----
        max_gap = config.number_range  # gap maxim teoretic
        gap_norm = {}
        for n in numbers:
            last = last_seen.get(n, -1)
            gap = draw_index - last if last >= 0 else draw_index + 1
            gap_norm[n] = gap / max_gap

        # ----- 3. Deviation față de frecvența așteptată (reversiune) -----
        dev_score = {}
        for n in numbers:
            # z‑score; numerele sub‑reprezentate (z negativ) primesc scor pozitiv
            z = (counts.get(n, 0) - expected_total_per_num) / std_freq
            dev_score[n] = -z  # inversăm semnul pentru re‑versiune

        # ----- Combinare scoruri -----
        final_scores = {}
        for n in numbers:
            sector_idx = (n - 1) // sector_size
            score = (
                W_SEC * sector_deficit[sector_idx] +
                W_GAP * gap_norm[n] +
                W_DEV * dev_score[n]
            )
            final_scores[n] = score

        # Returnare permutare completă, sortată descrescător după scor, tie‑break pe număr
        return sorted(numbers, key=lambda n: (-final_scores[n], n))

    return rank