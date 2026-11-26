NAME = "zpairewm"
DESCRIPTION = "Strategie hibridă care combină frecvența totală, penalizare pentru numerele întârziate, bonus pentru co‑apariții recenţi și deviația față de așteptările binomiale."
SOURCE = "ChatGPT 2024‑09‑08"

import math
import numpy as np
from collections import deque

def create(config):
    # config: LotteryConfig — are .number_range, .numbers_drawn, .ticket_size
    # Greutăţi fixe pentru fiecare componentă a scorului
    W_FREQ = 0.4
    W_OVERDUE = 0.3
    W_PAIR = 0.2
    W_Z = 0.1
    GAMMA_EWMA = 0.9  # nu utilizăm EWMA stocat, dar îl menţinem pentru extensibilitate

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # numbers are 1..config.number_range
        numbers = list(range(1, config.number_range + 1))
        total_draws_sofar = draw_index  # draw_index este indexul curent (0‑based), deci numărul de extrageri anterioare
        # 1️⃣ frecvenţa totală (normalizată)
        max_count = max(counts.values()) if counts else 1
        freq_score = {n: counts.get(n, 0) / max_count for n in numbers}

        # 2️⃣ penalizare overdue – cu cât a durat mai mult să apară, cu atât scorul creşte (bonus)
        overdue_score = {}
        for n in numbers:
            last = last_seen.get(n, -1)
            gap = draw_index - last if last != -1 else draw_index + 1
            overdue_score[n] = gap / (total_draws_sofar + 1)  # normalizare

        # 3️⃣ scor de co‑apariție în fereastra recentă
        # pentru fiecare număr, adunăm (dimensiunea extragerii - 1) pentru fiecare apariție recentă
        pair_score = {n: 0 for n in numbers}
        for draw_set in recent_deque:
            draw_size = len(draw_set)
            for n in draw_set:
                pair_score[n] += (draw_size - 1)
        # normalizare la maximul observat
        max_pair = max(pair_score.values()) if pair_score else 1
        pair_score = {n: pair_score[n] / max_pair for n in numbers}

        # 4️⃣ deviaţia (z‑score) față de valoarea așteptată (binomială)
        # așteptarea pentru fiecare număr: (draw_index * numbers_drawn) / number_range
        expected = (total_draws_sofar * config.numbers_drawn) / config.number_range if total_draws_sofar > 0 else 0
        # varianţa binomială aproximativă
        p = config.numbers_drawn / config.number_range
        var = total_draws_sofar * p * (1 - p) if total_draws_sofar > 0 else 1
        std = math.sqrt(var)
        z_score = {}
        for n in numbers:
            cnt = counts.get(n, 0)
            z = (cnt - expected) / std if std > 0 else 0.0
            # transformăm z‑score într-un bonus pozitiv: valori mari pozitiv sunt mai frecvente
            # și valori foarte negative (sub‑reprezentate) primesc un mic bonus
            z_score[n] = (z + abs(z)) / (2 * max(1.0, abs(z)))  # mapare în [0,1]

        # combinaţia finală a scorurilor
        final_score = {}
        for n in numbers:
            final_score[n] = (
                W_FREQ * freq_score[n] +
                W_OVERDUE * overdue_score[n] +
                W_PAIR * pair_score[n] +
                W_Z * z_score[n]
            )

        # sortare descrescător după scor, apoi după număr pentru stabilitate
        sorted_numbers = sorted(numbers, key=lambda n: (-final_score[n], n))
        return sorted_numbers

    return rank