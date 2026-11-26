NAME = "zewpasec"
DESCRIPTION = "Combina EWMA a frecvenţei recente, scorul de co‑apariţie în perechi şi gap‑sector pentru a genera o ordine de numere cu potenţial de depăşire a mediei de 6.0 hits per ticket."
SOURCE = "ChatGPT 2026‑09‑26"

import numpy as np
from collections import deque

def create(config):
    # Config attributes: number_range, numbers_drawn, ticket_size (not used here)
    # Fixed hyper‑parameters (could be exposed later for grid‑search)
    DECAY_EWMA = 0.9          # Exponential decay for EWMA
    W_EWMA = 1.0
    W_PAIR = 0.8
    W_SECT = 0.6
    SECT_SIZE = 6            # 24 numbers → 4 sectoare de 6 numere fiecare

    sector_of = lambda n: (n - 1) // SECT_SIZE  # 0‑based sector index

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # -------------------------------------------------------------------
        # 1️⃣ EWMA de prezenţă recentă (prezenţă = 1 dacă numărul apare în extragere)
        # -------------------------------------------------------------------
        # recent_deque conţine seturi cu numerele din cele `recent_window` extrageri
        # Cel mai recent set este la capătul stâng (index 0) – presupunem așa.
        numbers = list(range(1, config.number_range + 1))
        ewma_scores = {n: 0.0 for n in numbers}
        total_weight = 0.0
        for i, draw_set in enumerate(recent_deque):
            weight = DECAY_EWMA ** i  # i = 0 → cel mai recent, greutate maximă
            total_weight += weight
            for n in draw_set:
                ewma_scores[n] += weight
        if total_weight > 0:
            for n in numbers:
                ewma_scores[n] /= total_weight  # normalizare la [0,1]

        # -------------------------------------------------------------------
        # 2️⃣ Pair‑score – numărul de co‑apariţii cu alte numere în fereastra recentă
        # -------------------------------------------------------------------
        pair_counts = {n: 0 for n in numbers}
        for draw_set in recent_deque:
            for n in draw_set:
                # fiecare apariţie contribuie cu (len(draw_set)-1) co‑apariţii
                pair_counts[n] += len(draw_set) - 1
        # Normalizare la intervalul [0,1]
        max_pair = max(pair_counts.values()) if pair_counts else 1
        pair_scores = {n: pair_counts[n] / max_pair for n in numbers}

        # -------------------------------------------------------------------
        # 3️⃣ Sector‑gap – cât de mult așteaptă sectorul să apară din nou
        # -------------------------------------------------------------------
        # Pentru fiecare sector găsim cea mai recentă apariţie a oricărui număr din sector.
        sector_last = {}
        for n in numbers:
            sec = sector_of(n)
            if n in last_seen:
                sector_last.setdefault(sec, []).append(last_seen[n])
        # Gap = draw_index - max(last_seen) pentru sectorul respectiv
        sector_gap_score = {}
        for sec in range((config.number_range + SECT_SIZE - 1) // SECT_SIZE):
            if sec in sector_last and sector_last[sec]:
                gap = draw_index - max(sector_last[sec])
                # Normalizăm pe baza unei ferestre pragmatică (ex. 100 extrageri)
                norm_gap = min(gap / 100.0, 1.0)
            else:
                # Sector never seen → maximă
                norm_gap = 1.0
            # toate numerele din sector primesc acelaşi scor
            for n in numbers:
                if sector_of(n) == sec:
                    sector_gap_score[n] = norm_gap

        # -------------------------------------------------------------------
        # 4️⃣ Combinarea scorurilor
        # -------------------------------------------------------------------
        final_scores = {}
        for n in numbers:
            score = (
                W_EWMA * ewma_scores.get(n, 0.0) +
                W_PAIR * pair_scores.get(n, 0.0) +
                W_SECT * sector_gap_score.get(n, 0.0)
            )
            final_scores[n] = score

        # -------------------------------------------------------------------
        # 5️⃣ Returnare ordine completă 1..24, sortată descrescător după scor
        # -------------------------------------------------------------------
        ordered = sorted(numbers, key=lambda n: (-final_scores[n], n))
        return ordered

    return rank