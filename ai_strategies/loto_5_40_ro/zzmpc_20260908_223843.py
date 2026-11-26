NAME = "zzmpc"
DESCRIPTION = "Combinație de z‑score, momentum EWMA, hazard de întârziere și corecție pe sector pentru a genera o ordine completă a numerelor 1‑40."
SOURCE = "ChatGPT 2026‑09‑08"

import numpy as np
from collections import deque

# ponderi fixe ale componentelor – pot fi supurate prin grid‑search ulterior
W_Z = 1.0      # z‑score
W_MOM = 1.0    # momentum (EWMA recent)
W_HAZ = 0.5    # hazard (gap)
W_SEC = 0.8    # corecție sector
RCAP = 8       # plafon pentru hazard

def create(config):
    """Return a rank function compatible cu framework‑ul.
    
    config must expose:
      - number_range (ex: 40)
      - numbers_drawn (ex: 6)
      - ticket_size (not used here)
    """
    numbers = np.arange(1, config.number_range + 1)
    sector_ids = (numbers - 1) // 10  # 0‑3 pentru 1‑10,11‑20,…
    expected_prob = config.numbers_drawn / config.number_range

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # ---------- 1. date brute ----------
        total_draws = sum(counts.values())
        count_arr = np.array([counts.get(int(n), 0) for n in numbers], dtype=float)

        # ---------- 2. z‑score ----------
        exp_counts = total_draws * expected_prob
        std_counts = np.sqrt(total_draws * expected_prob * (1 - expected_prob))
        # evităm diviziunea prin zero dacă total_draws este foarte mic
        z_score = (count_arr - exp_counts) / (std_counts + 1e-12)

        # ---------- 3. momentum pe fereastra recentă ----------
        recent_counts = np.zeros_like(count_arr)
        for draw_set in recent_deque:
            for n in draw_set:
                recent_counts[int(n) - 1] += 1
        recent_total = recent_window * config.numbers_drawn
        # protecție împotriva divider zero (fereastra poate fi 0 în teste unitare)
        if recent_total == 0:
            recent_freq = np.zeros_like(count_arr)
        else:
            recent_freq = recent_counts / recent_total
        momentum = recent_freq - expected_prob

        # ---------- 4. hazard (gap de la ultima apariție) ----------
        gap_arr = np.empty_like(count_arr)
        for idx, n in enumerate(numbers):
            last = last_seen.get(int(n), -1)
            gap_arr[idx] = draw_index - last
        # cap la RCAP pentru a nu domina celelalte componente
        hazard = np.minimum(gap_arr, RCAP)

        # ---------- 5. corecție sector ----------
        # frecvența recentă pe sector (sumă pe toate numerele din sector)
        sector_recent_counts = np.zeros(4, dtype=float)
        for s in range(4):
            sector_recent_counts[s] = recent_counts[sector_ids == s].sum()
        if recent_total == 0:
            sector_recent_freq = np.zeros(4)
        else:
            sector_recent_freq = sector_recent_counts / recent_total
        # sector_weight = (expected - sector_freq) –‑> pozitivi pentru sectoare reci
        sector_weight_per_num = expected_prob - sector_recent_freq[sector_ids]

        # ---------- 6. combinare liniară ----------
        score = (W_Z * z_score) + (W_MOM * momentum) + (W_HAZ * hazard) + (W_SEC * sector_weight_per_num)

        # ---------- 7. returnare ordine ----------
        # Sortăm descrescător după scor, la egalitate numerotăm crescător
        sorted_numbers = sorted(numbers, key=lambda n: (-score[int(n) - 1], n))
        return sorted_numbers

    return rank