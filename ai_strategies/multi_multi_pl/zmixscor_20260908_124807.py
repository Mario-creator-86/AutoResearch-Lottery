NAME = "zmixscor"
DESCRIPTION = "Clasifică numerele prin scor mixt: frecvență totală, overdue, EWMA recentă și co‑apariții."
SOURCE = "mixscore_v1 + 2026-09-08"

import numpy as np
from collections import deque

def create(config):
    # config: .number_range, .numbers_drawn, .ticket_size (nu le iterăm)
    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # ------------------------------------------------------------
        # 1. Pregătirea listelor și a constantelor de ponderi
        # ------------------------------------------------------------
        numbers = np.arange(1, config.number_range + 1)
        # ponderi pentru fiecare componentă (ajustabile prin back‑test)
        w_total = 0.40   # frecvența totală
        w_overdue = 0.15 # penalizare overdue (gap)
        w_ewma = 0.30    # EWMA recentă
        w_pair = 0.15    # co‑apariție recentă
        alpha = 0.6      # factor de decădere EWMA
        # ------------------------------------------------------------
        # 2. Scorul bazat pe frecvența totală (normalizat)
        # ------------------------------------------------------------
        total_counts = np.array([counts.get(int(n), 0) for n in numbers], dtype=float)
        if total_counts.max() > 0:
            norm_total = total_counts / total_counts.max()
        else:
            norm_total = np.zeros_like(total_counts)
        # ------------------------------------------------------------
        # 3. Scorul overdue (gap) – cu valori mari adăugăm puţin
        # ------------------------------------------------------------
        gaps = np.array([
            (draw_index - last_seen.get(int(n), -1) - 1) if n in last_seen else draw_index
            for n in numbers
        ], dtype=float)
        # normalizare la [0,1]
        if gaps.max() > 0:
            norm_gap = gaps / gaps.max()
        else:
            norm_gap = np.zeros_like(gaps)
        # Overdue este benefic, dar cu impact moderat
        overdue_score = norm_gap
        # ------------------------------------------------------------
        # 4. EWMA pe fereastra recentă
        # ------------------------------------------------------------
        ewma_scores = np.zeros_like(total_counts)
        decay = 1.0
        for draw_set in reversed(recent_deque):
            # draw_set este un set cu numerele dintr‑o extragere recentă
            present = np.isin(numbers, list(draw_set)).astype(float)
            ewma_scores += decay * present
            decay *= alpha
        # normalizare
        if ewma_scores.max() > 0:
            norm_ewma = ewma_scores / ewma_scores.max()
        else:
            norm_ewma = np.zeros_like(ewma_scores)
        # ------------------------------------------------------------
        # 5. Scor de co‑apariție în ultimele extrageri
        #    Pentru fiecare număr, numărăm câţi parteneri a avut în set‑urile recente.
        # ------------------------------------------------------------
        pair_scores = np.zeros_like(total_counts)
        for draw_set in recent_deque:
            for n in draw_set:
                # fiecare alt element din acelaşi set este un partener
                pair_scores[int(n)-1] += (len(draw_set) - 1)
        if pair_scores.max() > 0:
            norm_pair = pair_scores / pair_scores.max()
        else:
            norm_pair = np.zeros_like(pair_scores)
        # ------------------------------------------------------------
        # 6. Combinaţia finală de scoruri
        # ------------------------------------------------------------
        final_score = (
            w_total * norm_total +
            w_overdue * overdue_score +
            w_ewma * norm_ewma +
            w_pair * norm_pair
        )
        # ------------------------------------------------------------
        # 7. Returnează o permutare sortată descrescător după scor, apoi număr
        # ------------------------------------------------------------
        sorted_indices = np.argsort(-final_score, kind='stable')  # stable păstrează ordinea numerică la egalitate
        ranked_numbers = numbers[sorted_indices].tolist()
        return ranked_numbers
    return rank