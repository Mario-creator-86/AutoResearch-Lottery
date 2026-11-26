NAME = "zewmaovs"
DESCRIPTION = "Combinează frecvența recentă (EWMA), gradul de \"overdue\" și scorul pe sectoare de 7 numere pentru a ranka toate numerele 1..42."
SOURCE = "custom_strategy_2024-09-08"

import numpy as np
from collections import deque

# hiperparametri – pot fi supuși unei căutări grid‑search în viitor
def _hyperparams():
    return {
        "W_EWMA": 0.5,   # greutate pentru frecvența recentă
        "W_OV":   0.3,   # greutate pentru overdue (inverse)
        "W_SEC":  0.2,   # greutate pentru scorul pe sector
        "SECTOR_SIZE": 7 # dimensiunea fiecărui sector
    }

def create(config):
    # config are .number_range, .numbers_drawn, .ticket_size (nu le folosim direct aici)
    params = _hyperparams()
    W_EWMA = params["W_EWMA"]
    W_OV   = params["W_OV"]
    W_SEC  = params["W_SEC"]
    SECTOR_SIZE = params["SECTOR_SIZE"]

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # Lista completă a numerelor
        numbers = np.arange(1, config.number_range + 1)
        # 1) scor EWMA – în absența unui EWMA recursiv utilizăm media pe fereastra recentă
        recent_counts = np.zeros_like(numbers, dtype=float)
        # recent_deque conține seturi de numere pentru ultimele `recent_window` extrageri
        for draw_set in recent_deque:
            for n in draw_set:
                if 1 <= n <= config.number_range:
                    recent_counts[n-1] += 1.0
        # frecvența pe fereastră recentă (împărțim la dimensiunea ferestrei pentru a normaliza)
        if recent_window > 0:
            recent_freq = recent_counts / recent_window
        else:
            recent_freq = np.zeros_like(numbers, dtype=float)
        # 2) scor overdue – inversul timpului de la ultima apariție (cu +1 pentru a evita divizare cu zero)
        overdue = np.empty_like(numbers, dtype=float)
        for idx, n in enumerate(numbers, start=1):
            last_idx = last_seen.get(int(n), -1)
            days_since = draw_index - last_idx
            overdue[idx-1] = 1.0 / (days_since + 1.0)
        # 3) scor pe sector – fiecare sector are dimensiunea SECTOR_SIZE
        sector_id = ((numbers - 1) // SECTOR_SIZE)
        # calculăm media overdue pe fiecare sector
        sector_overdue_avg = np.zeros(sector_id.max() + 1, dtype=float)
        for s in range(sector_overdue_avg.size):
            mask = sector_id == s
            if np.any(mask):
                sector_overdue_avg[s] = overdue[mask].mean()
        # apoi transformăm în inverse pentru a avea un scor crescător când sectorul este "overdue"
        sector_score = 1.0 / (sector_overdue_avg[sector_id] + 1.0)
        # 4) scor compus
        composite = (W_EWMA * recent_freq) + (W_OV * overdue) + (W_SEC * sector_score)
        # ordonăm descrescător după scor, tie‑break numeric ascendent
        sorted_idx = np.lexsort((numbers, -composite))  # -composite pentru descrescător
        ranked_numbers = numbers[sorted_idx].tolist()
        return ranked_numbers

    return rank