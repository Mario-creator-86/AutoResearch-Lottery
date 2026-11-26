NAME = "zmultisi"
DESCRIPTION = "Strategie multi‑semnal ce combină deviaţia pe termen lung, momentum recent, overdue și cicluri pe sectoare pentru a evalua fiecare număr din 1..49."
SOURCE = "model + 2026-09-08"

import math
import numpy as np
from collections import deque

# parametri default – pot fi suprascrise prin grid‑search în mediul de back‑test
W_Z = 1.0          # greutate pentru z‑score pe termen lung
W_MOM = 1.0        # greutate pentru momentum recent
W_HAZ = 1.0        # greutate pentru hazard (overdue)
W_SEC = 1.0        # greutate pentru semnalul sectorial
GAMMA = 0.95       # factor de decădere pentru EWMA (nu este folosit explicit, dar lăsat pentru extensie)
RCAP = 6           # plafonul de overdue exprimat în multiplii de media așteptată
SECTORS = 7        # număr de sectoare (7×7 = 49)

def create(config):
    """Returnează funcția rank conform interfeței așteptate.
    
    config trebuie să aibă atributele:
        - number_range (ex. 49)
        - numbers_drawn (ex. 6)
        - ticket_size (nu este utilizat direct)
    """
    n_range = config.number_range
    k_draw = config.numbers_drawn
    prob_global = k_draw / n_range

    # pregătim maparea sector -> list of numbers pentru acces rapid
    sector_size = n_range // SECTORS
    sector_map = {}
    for s in range(SECTORS):
        start = s * sector_size + 1
        end = start + sector_size
        sector_map[s] = list(range(start, end))

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        """Calculează scorurile și returnează permutarea completă a numerelor 1..49.
        
        - counts: dict {număr: apariții totale în istoricul complet}
        - recent_deque: collections.deque de set() cu ultimele `recent_window` extrageri
        - last_seen: dict {număr: indexul ultimei apariții (0‑based)}
        - draw_index: indexul curent al extragerii (0‑based)
        - recent_window: dimensiunea ferestrei recentă (coincide cu len(recent_deque))
        """
        numbers = np.arange(1, n_range + 1)
        # ---------- Componenta Z (deviatia pe termen lung) ----------
        total_draws = draw_index  # numărul de extrageri deja procesate
        expected_counts = total_draws * prob_global
        var_counts = total_draws * prob_global * (1 - prob_global)
        # evităm diviziunea prin zero în primele extrageri
        std_counts = math.sqrt(var_counts) if var_counts > 0 else 1.0
        z_long = np.array([ (counts.get(int(num), 0) - expected_counts) / std_counts for num in numbers ])

        # ---------- Componenta Momentum recent (frecvenţa recentă) ----------
        # construim vectorul de apariții în fereastra recentă
        recent_counts = np.zeros(n_range, dtype=int)
        for draw_set in recent_deque:
            for num in draw_set:
                if 1 <= num <= n_range:
                    recent_counts[num - 1] += 1
        momentum = recent_counts / recent_window  # valori între 0 şi 1

        # ---------- Componenta Hazard (overdue) ----------
        # gap = draw_index - last_seen[num]
        gap = np.array([ draw_index - last_seen.get(int(num), -1) for num in numbers ])
        # media așteptată a gap‑ului pentru distribuție uniformă este 1/prob_global ≈ n_range/k_draw
        cap = RCAP * (n_range / k_draw)  # plafon în termeni de medii
        hazard = np.minimum(gap / cap, 1.0)  # normalizat în [0,1]

        # ---------- Componenta Sectorială ----------
        # calculăm câte apariții recentare are fiecare sector
        sector_hits = np.zeros(SECTORS, dtype=int)
        for s, nums in sector_map.items():
            sector_hits[s] = recent_counts[[n - 1 for n in nums]].sum()
        # așteptarea sectorială pe fereastra recentă
        prob_sector = (sector_size * k_draw) / n_range  # probabilitatea ca o extragere să conţină un număr din sector
        exp_sector_hits = recent_window * prob_sector
        var_sector = recent_window * prob_sector * (1 - prob_sector)
        std_sector = math.sqrt(var_sector) if var_sector > 0 else 1.0
        sector_z = (sector_hits - exp_sector_hits) / std_sector
        # atribuim acelaşi sector_z fiecărui număr din sectorul respectiv
        sector_z_per_num = np.empty(n_range)
        for s, nums in sector_map.items():
            sector_z_per_num[[n - 1 for n in nums]] = sector_z[s]

        # ---------- Scor final ----------
        total_score = (W_Z * z_long) + (W_MOM * momentum) + (W_HAZ * hazard) + (W_SEC * sector_z_per_num)
        # sortăm descrescător după scor, cu tiebreak pe număr mic
        ordered = numbers[np.argsort((-total_score, numbers))]
        return list(ordered)

    return rank