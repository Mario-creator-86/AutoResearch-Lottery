NAME = "zewmaove"
DESCRIPTION = "Combină frecvența totală, EWMA pe fereastră recentă și overdue pentru a ranka numerele"
SOURCE = "model + 2026-09-08"

import numpy as np
from collections import deque

def create(config):
    # Configurable hyper‑parameters (pot fi ajustate ulterior)
    GAMMA = 0.94          # factor de decădere pentru EWMA
    W_COUNT = 0.4         # pondere pentru frecvența totală
    W_EWMA = 0.4          # pondere pentru EWMA recentă
    W_OVERDUE = 0.2      # pondere pentru overdue

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # Lista completă de numere din 1..N
        numbers = list(range(1, config.number_range + 1))

        # 1️⃣ Normalizare frecvență totală
        max_count = max(counts.get(n, 0) for n in numbers) or 1
        count_score = {n: counts.get(n, 0) / max_count for n in numbers}

        # 2️⃣ EWMA pe fereastra recentă (cel mai recent are greutate maximă)
        ewma_score = {n: 0.0 for n in numbers}
        # iterăm de la cel mai recent la cel mai vechi
        for age, draw_set in enumerate(reversed(recent_deque)):
            weight = GAMMA ** age
            for n in draw_set:
                ewma_score[n] += weight
        # normalizare EWMA
        max_ewma = max(ewma_score.values()) or 1.0
        for n in numbers:
            ewma_score[n] /= max_ewma

        # 3️⃣ Overdue – numărul de extrageri de la ultima apariție
        overdue_score = {}
        max_overdue = draw_index + 1  # cel mai mare posibil
        for n in numbers:
            last = last_seen.get(n, -1)
            overdue = draw_index - last if last != -1 else max_overdue
            overdue_score[n] = overdue / max_overdue

        # Combina scorurile cu ponderi
        combined = {}
        for n in numbers:
            combined[n] = (
                W_COUNT * count_score[n] +
                W_EWMA * ewma_score[n] +
                W_OVERDUE * overdue_score[n]
            )

        # Returnează numerele ordonate descrescător după scor combinat, apoi numeric
        return sorted(numbers, key=lambda n: (-combined[n], n))

    return rank