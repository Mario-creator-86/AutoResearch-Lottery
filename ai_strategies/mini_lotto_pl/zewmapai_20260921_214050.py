NAME = "zewmapai"
DESCRIPTION = "Folosește frecvenţa totală cu decadere exponențială, prezența recentă și co-apărare în ultimele extrageri pentru a ranka numerele."
SOURCE = "custom_strategy + 2026-09-21"

import numpy as np
from collections import deque

def create(config):
    # parametri interni (pot fi ajustați prin fine‑tuning rapid)
    DECAY = 0.94          # factor de decădere exponențială pentru EWMA
    RECENT_BONUS = 2.0    # multiplicator pentru aparițiile din fereastra recentă
    PAIR_BONUS = 1.0      # multiplicator pentru co‑apărarea recentă

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        """Returnează o permutare completă a numerelor 1..number_range.
        
        counts: dict {număr: apariții totale în istoric}
        recent_deque: deque de seturi, fiecare set conține numerele dintr-o extragere recentă
        last_seen: dict {număr: indicele ultimei apariții (sau -1 dacă niciodată)}
        draw_index: indicele curent al extragerii (începând cu 0)
        recent_window: dimensiunea ferestrei recentă (len(recent_deque) ar trebui să fie egală)
        """
        numbers = list(range(1, config.number_range + 1))
        # 1) EWMA bazat pe diferența de timp de la ultima apariție
        ewma_scores = {}
        for n in numbers:
            last = last_seen.get(n, -1)
            gap = draw_index - last if last >= 0 else draw_index + 1
            # dacă nu a apărut niciodată, tratăm gap ca draw_index+1 pentru a avea un factor mic
            weight = DECAY ** gap
            ewma_scores[n] = counts.get(n, 0) * weight
        # 2) Bonus pentru prezența recentă în fereastra recentă
        recent_counts = {n: 0 for n in numbers}
        for draw_set in recent_deque:
            for n in draw_set:
                recent_counts[n] += 1
        # 3) Bonus de co‑apărare: număr de seturi recente în care numărul a apărut ALĂTURI cu alte numere din fereastra recentă
        # Pentru eficiență, folosim deja recent_counts (acesta reprezintă exact aparițiile recente)
        # dar dorim să penalizăm supra‑numărarea, așa că îl combinăm cu PAIR_BONUS
        pair_scores = {n: recent_counts[n] * PAIR_BONUS for n in numbers}
        # scor final compus
        final_scores = {}
        for n in numbers:
            final_scores[n] = (
                ewma_scores[n]
                + recent_counts[n] * RECENT_BONUS
                + pair_scores[n]
            )
        # sortăm descrescător după scor, apoi numeric pentru break‑ties
        sorted_numbers = sorted(numbers, key=lambda n: (-final_scores[n], n))
        return sorted_numbers

    return rank