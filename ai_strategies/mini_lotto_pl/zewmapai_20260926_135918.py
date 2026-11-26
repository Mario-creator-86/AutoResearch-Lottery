NAME = "zewmapai"
DESCRIPTION = "Combina frecvența globală, EWMA pe fereastra recentă, co‑apariția cu numerele hot şi recurenţa pe sectoare pentru a genera o permutare completă a numerelor 1..42."
SOURCE = "custom_strategy_2026-09-26"

import numpy as np

def create(config):
    # greutăţi fixe; suma = 1.0
    w_counts = 0.4
    w_ewma = 0.3
    w_pair = 0.2
    w_sector = 0.1
    decay = 0.94                # factor de decădere pentru EWMA
    sector_size = 7             # 42 / 6 sectoare

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # vectorul cu toate numerele 1..N
        numbers = np.arange(1, config.number_range + 1)

        # 1️⃣  Frecvența globală
        cnt = np.array([counts.get(int(n), 0) for n in numbers], dtype=float)

        # 2️⃣  EWMA pe fereastra recentă
        ewma = np.zeros_like(cnt)
        weight = 1.0
        # luăm ultimele `recent_window` extrageri (cel mai recent primește greutatea maximă)
        for draw in reversed(list(recent_deque)[-recent_window:]):
            for n in draw:
                ewma[int(n) - 1] += weight
            weight *= decay

        # 3️⃣  Co‑apariție cu cele 5 numere hot (pe baza frecvenței globale)
        top_k = 5
        hot_idx = np.argsort(-cnt)[:top_k]
        hot_numbers = set(int(numbers[i]) for i in hot_idx)
        pair_score = np.zeros_like(cnt)
        for draw in recent_deque:
            inter = draw & hot_numbers
            if not inter:
                continue
            for n in draw:
                pair_score[int(n) - 1] += len(inter)

        # 4️⃣  Recency pe sector (prioritizează sectoarele mai „întârziate”)
        sector_score = np.zeros_like(cnt)
        for start in range(0, config.number_range, sector_size):
            sector_nums = range(start + 1, min(start + sector_size + 1, config.number_range + 1))
            ages = []
            for n in sector_nums:
                ls = last_seen.get(n, -1)
                if ls >= 0:
                    ages.append(draw_index - ls)
                else:
                    ages.append(draw_index)  # niciodată văzut → maxim
            sector_age = np.mean(ages) if ages else 0
            for n in sector_nums:
                sector_score[int(n) - 1] = sector_age

        # normalizare în [0,1] pentru fiecare componentă
        def norm(arr):
            if arr.max() == arr.min():
                return np.zeros_like(arr)
            return (arr - arr.min()) / (arr.max() - arr.min())

        cnt_n = norm(cnt)
        ewma_n = norm(ewma)
        pair_n = norm(pair_score)
        sector_n = norm(sector_score)

        # scorul final combinat
        final = (
            w_counts * cnt_n +
            w_ewma * ewma_n +
            w_pair * pair_n +
            w_sector * sector_n
        )

        # sortare descrescătoare după scor, apoi ascendentă după număr pentru egalităţi
        sorted_idx = np.lexsort((numbers, -final))
        return list(numbers[sorted_idx])

    return rank