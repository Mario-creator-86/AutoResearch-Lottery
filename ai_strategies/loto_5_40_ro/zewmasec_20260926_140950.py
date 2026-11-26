NAME = "zewmasec"
DESCRIPTION = "Combines EWMA of recent frequencies, sector momentum, pair co‑occurrence and a mild overdue bias into a single ranking for Loto 5/40."
SOURCE = "generated 2026-09-26"

import numpy as np

def create(config):
    decay = 0.85                # factor de decădere pentru EWMA
    sector_size = 8             # 40 numere → 5 sectoare de câte 8
    w_ewma = 1.0
    w_recent = 0.5
    w_sector = 0.3
    w_pair = 0.4
    w_over = 0.6

    def normalize(arr):
        m = arr.max()
        return arr / m if m > 0 else arr

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # vector cu toate numerele posibile
        numbers = np.arange(1, config.number_range + 1)
        n = numbers.size

        # ---- EWMA al apariţiilor ----
        ewma = np.zeros(n, dtype=float)
        decay_pow = 1.0
        for draw in reversed(recent_deque):
            mask = np.fromiter((1 if num in draw else 0 for num in numbers), dtype=int)
            ewma += decay_pow * mask
            decay_pow *= decay

        # ---- Numărare simplă în fereastra recentă ----
        recent_counts = np.zeros(n, dtype=int)
        for draw in recent_deque:
            mask = np.fromiter((1 if num in draw else 0 for num in numbers), dtype=int)
            recent_counts += mask

        # ---- EWMA pe nivel de sector ----
        num_sectors = (config.number_range + sector_size - 1) // sector_size
        sector_ewma = np.zeros(num_sectors, dtype=float)
        decay_pow = 1.0
        for draw in reversed(recent_deque):
            for num in draw:
                sid = (num - 1) // sector_size
                sector_ewma[sid] += decay_pow
            decay_pow *= decay
        sector_score = np.array([sector_ewma[(num - 1) // sector_size] for num in numbers])

        # ---- Scor co‑apariție perechi (proxy rapid) ----
        pair_score = recent_counts * (config.numbers_drawn - 1)

        # ---- Scor "overdue" (inverse al gap-ului) ----
        gaps = np.array([draw_index - last_seen.get(int(num), -1) for num in numbers])
        overdue_score = 1.0 / (gaps + 1.0)

        # ---- Normalizare şi sumă ponderată ----
        total = (
            w_ewma   * normalize(ewma) +
            w_recent * normalize(recent_counts) +
            w_sector * normalize(sector_score) +
            w_pair   * normalize(pair_score) +
            w_over   * normalize(overdue_score)
        )

        # Returnează o permutare completă, sortată descrescător după scorul total
        ordered = sorted(numbers, key=lambda x: (-total[x - 1], x))
        return ordered

    return rank