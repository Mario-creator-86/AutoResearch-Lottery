import numpy as np
from collections import defaultdict

NAME = "zhybride"
DESCRIPTION = "Combinație de frecvență totală, recentă, co‑aparții și entropie sectorială"
SOURCE = "ChatGPT + 2026-09-16"

def create(config):
    # config: LotteryConfig — are .number_range, .numbers_drawn, .ticket_size
    sector_size = 10  # 80 / 8 = 10 numere per sector
    sector_bounds = [(i, i + sector_size - 1) for i in range(1, config.number_range + 1, sector_size)]

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, config.number_range + 1))

        # 1️⃣ Frecvenţa totală (raw counts)
        total_counts = np.array([counts.get(n, 0) for n in numbers], dtype=float)
        # Normalizare 0‑1
        if total_counts.max() > 0:
            norm_total = (total_counts - total_counts.min()) / (total_counts.max() - total_counts.min())
        else:
            norm_total = np.zeros_like(total_counts)

        # 2️⃣ Frecvenţa recentă în fereastra specificată
        recent_counts = defaultdict(int)
        for draw_set in recent_deque:
            for n in draw_set:
                recent_counts[n] += 1
        recent_arr = np.array([recent_counts.get(n, 0) for n in numbers], dtype=float)
        if recent_arr.max() > 0:
            norm_recent = (recent_arr - recent_arr.min()) / (recent_arr.max() - recent_arr.min())
        else:
            norm_recent = np.zeros_like(recent_arr)

        # 3️⃣ Co‑aparţiile (maxim co‑apariție pentru fiecare număr)
        co_counts = {n: defaultdict(int) for n in numbers}
        for draw_set in recent_deque:
            for n in draw_set:
                for m in draw_set:
                    if m != n:
                        co_counts[n][m] += 1
        max_co = np.array([max(co_counts[n].values()) if co_counts[n] else 0 for n in numbers], dtype=float)
        if max_co.max() > 0:
            norm_pair = (max_co - max_co.min()) / (max_co.max() - max_co.min())
        else:
            norm_pair = np.zeros_like(max_co)

        # 4️⃣ Entropia sectorială (calculată pe frecvenţa recentă)
        entropy_scores = []
        for n in numbers:
            # Identify sector index
            sector_idx = (n - 1) // sector_size
            start, end = sector_bounds[sector_idx]
            sector_nums = list(range(start, end + 1))
            sector_counts = np.array([recent_counts.get(s, 0) for s in sector_nums], dtype=float)
            total_sector = sector_counts.sum()
            if total_sector > 0:
                probs = sector_counts / total_sector
                ent = -np.sum(probs * np.log2(probs + 1e-12))  # Shannon entropy
            else:
                ent = 0.0
            entropy_scores.append(ent)
        entropy_arr = np.array(entropy_scores, dtype=float)
        if entropy_arr.max() > 0:
            norm_entropy = (entropy_arr - entropy_arr.min()) / (entropy_arr.max() - entropy_arr.min())
        else:
            norm_entropy = np.zeros_like(entropy_arr)

        # Ponderi (ajustabile prin config în viitor)
        w_total = 0.30
        w_recent = 0.30
        w_pair = 0.20
        w_entropy = 0.20

        final_score = (
            w_total * norm_total +
            w_recent * norm_recent +
            w_pair * norm_pair +
            w_entropy * norm_entropy
        )

        # Return numbers sortate descrescător după scor, tie‑break pe valoarea numerică
        sorted_numbers = sorted(numbers, key=lambda n: (-final_score[n - 1], n))
        return sorted_numbers

    return rank