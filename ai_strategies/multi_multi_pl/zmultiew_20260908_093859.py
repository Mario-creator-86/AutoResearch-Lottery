NAME = "zmultiew"
DESCRIPTION = "Combines total frequency, EWMA recent hits, overdue boost, recent pair co‑occurrence, sector activity, and entropy‑based rarity to rank numbers 1‑80."
SOURCE = "generated + 2026-09-08"

import numpy as np
from collections import defaultdict


def create(config):
    # config: .number_range, .numbers_drawn, .ticket_size (unused here)
    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # ------------------------------------------------------------------
        # Preparare liste de bază
        # ------------------------------------------------------------------
        numbers = list(range(1, config.number_range + 1))
        N = len(numbers)

        # ------------------------------------------------------------------
        # 1. Frecvenţa totală (normalizată)
        # ------------------------------------------------------------------
        total_counts = np.array([counts.get(n, 0) for n in numbers], dtype=float)
        # evităm diviziunea prin zero – dacă toate sunt zero, lăsăm un vector de 0
        max_tc = total_counts.max() if total_counts.max() > 0 else 1.0
        freq_score = total_counts / max_tc

        # ------------------------------------------------------------------
        # 2. EWMA pe fereastra recentă
        # ------------------------------------------------------------------
        alpha = 0.30  # factor de descreştere exponenţială
        win_len = len(recent_deque)
        if win_len == 0:
            ewma_score = np.zeros(N)
        else:
            # greutăţi exponenţiale, cele mai noi au greutate mai mare
            weights = np.array([alpha * ((1 - alpha) ** i) for i in range(win_len)], dtype=float)
            weights /= weights.sum()
            # matrice (win_len x N) cu 1 dacă numărul apare în draw‑ul i‑lea
            recent_matrix = np.zeros((win_len, N), dtype=float)
            for i, draw_set in enumerate(recent_deque):
                for n in draw_set:
                    if 1 <= n <= config.number_range:
                        recent_matrix[i, n - 1] = 1.0
            ewma_score = weights @ recent_matrix  # dimensiune (N,)

        # ------------------------------------------------------------------
        # 3. Overdue – număr de extrageri de la ultima apariție
        # ------------------------------------------------------------------
        overdue_vals = np.empty(N, dtype=float)
        for idx, n in enumerate(numbers):
            last = last_seen.get(n, -1)
            draws_since = draw_index - last if last != -1 else draw_index + 1
            overdue_vals[idx] = np.log1p(draws_since)  # log(1 + d) pentru stabilitate
        # normalizare
        overdue_score = overdue_vals / (overdue_vals.max() if overdue_vals.max() > 0 else 1.0)

        # ------------------------------------------------------------------
        # 4. Scor de perechi în fereastra recentă
        # ------------------------------------------------------------------
        # Construim un dicţionar de co‑ocurenţă pentru numerele din fereastra recentă
        pair_counts = defaultdict(lambda: defaultdict(int))
        for draw_set in recent_deque:
            lst = list(draw_set)
            for i in range(len(lst)):
                a = lst[i]
                for j in range(i + 1, len(lst)):
                    b = lst[j]
                    if 1 <= a <= config.number_range and 1 <= b <= config.number_range:
                        pair_counts[a][b] += 1
                        pair_counts[b][a] += 1
        pair_score = np.zeros(N, dtype=float)
        for idx, n in enumerate(numbers):
            total_pair = sum(pair_counts[n].values())
            pair_score[idx] = total_pair
        # normalizare
        max_pair = pair_score.max() if pair_score.max() > 0 else 1.0
        pair_score /= max_pair

        # ------------------------------------------------------------------
        # 5. Intensitate sectorială (sectors de 10 numere)
        # ------------------------------------------------------------------
        sector_size = 10
        sector_cnt = np.zeros((config.number_range // sector_size + 1), dtype=float)
        for draw_set in recent_deque:
            for n in draw_set:
                sector = (n - 1) // sector_size
                sector_cnt[sector] += 1
        # normalizăm pe sectorul fiecărui număr
        sector_score = np.empty(N, dtype=float)
        for idx, n in enumerate(numbers):
            sector = (n - 1) // sector_size
            sector_score[idx] = sector_cnt[sector]
        sector_score /= sector_score.max() if sector_score.max() > 0 else 1.0

        # ------------------------------------------------------------------
        # 6. Entropie / raritate în fereastra recentă (bonus pentru numere puţin frecvente)
        # ------------------------------------------------------------------
        # Frecvenţa recentă a fiecărui număr
        recent_hits = recent_matrix.sum(axis=0) if win_len > 0 else np.zeros(N, dtype=float)
        total_recent_numbers = recent_window * config.numbers_drawn
        prob = recent_hits / total_recent_numbers
        # evităm log(0)
        info_content = -np.log(prob + 1e-12)
        # Normalizare la [0,1]
        info_score = info_content / (info_content.max() if info_content.max() > 0 else 1.0)

        # ------------------------------------------------------------------
        # Combinație ponderată a tuturor componentelor
        # ------------------------------------------------------------------
        w_freq = 0.20
        w_ewma = 0.30
        w_over = 0.15
        w_pair = 0.15
        w_sector = 0.10
        w_info = 0.10
        total_score = (
            w_freq * freq_score +
            w_ewma * ewma_score +
            w_over * overdue_score +
            w_pair * pair_score +
            w_sector * sector_score +
            w_info * info_score
        )
        # Returnăm sortarea descrescătoare a scorurilor; în caz de egalitate, numărul mic primul
        sorted_numbers = sorted(numbers, key=lambda n: (-total_score[n - 1], n))
        return sorted_numbers
    return rank