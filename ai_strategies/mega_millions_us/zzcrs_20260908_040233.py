NAME = "zzcrs"
DESCRIPTION = "Strategie care combină deviația pe termen lung, frecvența recentă și un factor de recență exponențială pentru a ordona numerele din 1..70."
SOURCE = "generated 2026-09-08"

import math
from collections import defaultdict


def create(config):
    # config: LotteryConfig — .number_range, .numbers_drawn, .ticket_size
    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # 1. Pregătire valori de bază
        numbers = list(range(1, config.number_range + 1))
        total_counts = sum(counts.values())  # total draw * numbers_drawn până la momentul curent
        p = config.numbers_drawn / config.number_range  # probabilitatea teoretică pe un număr
        expected_per_num = total_counts * p
        variance = total_counts * p * (1 - p)
        sqrt_var = math.sqrt(variance) if variance > 0 else 1.0

        # 2. Calculăm semnalele pentru fiecare număr
        #    a) z_long – deviație standardizată pe termen lung
        #    b) freq_recent – câte apariții în fereastra recentă
        #    c) recency_score – exp(-gap / tau)
        tau = max(1, recent_window * 2)  # scală pentru factorul exponențial

        # Precalculăm frecvențele în fereastra recentă (O(70 * recent_window))
        recent_freq = defaultdict(int)
        for draw in recent_deque:
            for n in draw:
                recent_freq[n] += 1

        scores = {}
        for n in numbers:
            cnt = counts.get(n, 0)
            # a) z_long
            z_long = (cnt - expected_per_num) / sqrt_var
            # b) frecvența recentă (valorile absolute, nu standardizate)
            freq_rec = recent_freq.get(n, 0)
            # c) recency_score
            gap = draw_index - last_seen.get(n, -10**9)  # dacă nu a apărut niciodată, gap foarte mare
            recency_score = math.exp(-gap / tau)
            # combinăm cu ponderi fixe
            score = 0.6 * z_long + 0.3 * freq_rec + 0.1 * recency_score
            scores[n] = score

        # 3. Sortăm descrescător după scor, tie‑break pe valoarea numerică
        ranked = sorted(numbers, key=lambda n: (-scores[n], n))
        return ranked
    return rank