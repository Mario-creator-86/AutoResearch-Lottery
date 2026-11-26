NAME = "zzewpair"
DESCRIPTION = "Strategie compusă din EWMA pe frecvența recentă, scor de co‑apariție cu ultimele numere, deviație față de așteptarea binomială și ponderarea sectorului de întârzieri."
SOURCE = "model + 2026-09-08"

import math
import numpy as np

# ==================== CONFIGURABLE WEIGHTS (fixate pentru back‑test) ====================
W_EWMA = 0.40
W_PAIR = 0.30
W_DEV  = 0.20
W_SEC  = 0.10
GAMMA_EWMA = 0.85  # exponent de uitare pentru EWMA

def create(config):
    """Returnează funcția rank specifică acestei strategii.
    
    config este un LotteryConfig cu atributele:
        number_range   – N (ex. 45)
        numbers_drawn – K (ex. 5)
        ticket_size    – dimensiunea biletului (nu e folosită aici)
    """
    N = config.number_range
    K = config.numbers_drawn
    p = K / float(N) if N else 0.0  # probabilitatea unei apariții într‑un singur bilet

    # ---------------------------------------------------------------------
    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # -----------------------------------------------------------------
        numbers = list(range(1, N + 1))
        W = len(recent_deque)  # ar trebui să fie recent_window, dar nu ne bazăm pe acesta
        # 1️⃣ EWMA pe prezența recentă
        ewma_raw = np.zeros(N)
        norm = 0.0
        for i, draw in enumerate(reversed(recent_deque)):
            weight = GAMMA_EWMA ** i
            norm += weight
            for n in draw:
                ewma_raw[n - 1] += weight
        if norm > 0:
            ewma_raw /= norm
        # 2️⃣ Pair‑score cu ultimele numere (noua extragere)
        newest = set(recent_deque[-1]) if recent_deque else set()
        pair_raw = np.zeros(N)
        # căutăm în toate extragerile anterioare ale ferestrei
        for draw in list(recent_deque)[:-1]:
            if not newest.intersection(draw):
                continue
            for n in draw:
                pair_raw[n - 1] += 1
        # 3️⃣ Deviatia față de așteptarea binomială (z‑score)
        dev_raw = np.zeros(N)
        # expected count = draw_index * p (draw_index = număr de extrageri deja efectuate)
        exp_cnt = draw_index * p
        var_cnt = exp_cnt * (1 - p) if exp_cnt > 0 else 1.0
        sd_cnt = math.sqrt(var_cnt)
        for n in numbers:
            obs = counts.get(n, 0)
            dev_raw[n - 1] = (obs - exp_cnt) / sd_cnt if sd_cnt > 0 else 0.0
        # 4️⃣ Sector overdue (media gap‑urilor pe sector de 5 numere)
        sector_raw = np.zeros(N)
        # calculăm gap‑ul pentru fiecare număr
        gaps = np.array([draw_index - last_seen.get(n, -1) for n in numbers], dtype=float)
        # grupăm în 9 sectoare (1‑5, 6‑10, …, 41‑45)
        sector_means = []
        for s in range(0, N, 5):
            sector_means.append(gaps[s:s+5].mean())
        sector_means = np.array(sector_means)
        for idx, n in enumerate(numbers):
            sector_idx = (n - 1) // 5
            sector_raw[idx] = sector_means[sector_idx]
        # -----------------------------------------------------------------
        # Standardizăm fiecare componentă (z‑score pe setul de 45 de numere)
        def standardize(arr):
            m = arr.mean()
            sd = arr.std(ddof=0)
            if sd < 1e-12:
                return np.zeros_like(arr)
            return (arr - m) / sd

        ewma_z = standardize(ewma_raw)
        pair_z = standardize(pair_raw)
        dev_z  = standardize(dev_raw)
        sec_z  = standardize(sector_raw)
        # -----------------------------------------------------------------
        # Agregăm cu ponderi
        final_score = (W_EWMA * ewma_z +
                        W_PAIR * pair_z +
                        W_DEV  * dev_z  +
                        W_SEC  * sec_z)
        # Construim dicționarul de scoruri
        score_dict = {n: final_score[n - 1] for n in numbers}
        # Returnăm ordonarea descrescătoare a scorurilor (în caz de egalitate, crescător pe număr)
        return sorted(numbers, key=lambda n: (-score_dict[n], n))

    return rank