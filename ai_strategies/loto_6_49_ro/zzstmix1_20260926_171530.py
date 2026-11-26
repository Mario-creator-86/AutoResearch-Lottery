NAME = "zzstmix1"
DESCRIPTION = "Combinație de semnale standardizate (reversie z‑score, momentum recent, recență exponențială, gap standardizat, deviație sectorială și interacțiune) pentru a ranka toate 49 de numere."
SOURCE = "ChatGPT + 2026‑09‑26"

import collections
import numpy as np

def create(config):
    # parametri interni (pot fi ajustaţi prin auto‑învăţare ulterioară)
    TAU = 12                     # scală de recență exponenţială
    EXPECTED_GAP = 7.17          # medie geometrică a gap‑ului pentru o numerotecă
    STD_GAP = 7.65               # devierea standard a gap‑ului
    GAP_CLIP = 2.5               # prag de tăiere a standardizării gap‑ului
    SECTOR_SIZE = 7              # 49/7 = 7 sectoare
    # greutăţi ale semnalelor – alese modest pentru a evita dominarea unei singure componente
    W_ZREV = 0.5
    W_MOM = 0.3
    W_REC = 0.2
    W_GAP = 0.4
    W_SECT = 0.3
    W_INT = 0.5

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, config.number_range + 1))

        # 1. statistici globale (z‑score negativ pentru reversie)
        total_counts = np.array([counts.get(n, 0) for n in numbers], dtype=float)
        mean_tot = total_counts.mean()
        std_tot = total_counts.std(ddof=0) if total_counts.std(ddof=0) > 0 else 1.0
        z_score = (total_counts - mean_tot) / std_tot
        score_zrev = -z_score * W_ZREV  # semn negativ = reversie

        # 2. momentum recent (frecvenţa în fereastra recentă)
        recent_counts = np.zeros(config.number_range, dtype=float)
        for draw_set in recent_deque:
            for n in draw_set:
                recent_counts[n - 1] += 1
        recent_freq = recent_counts / recent_window
        # așteptarea teoretică în fereastra recentă
        exp_recent = recent_window * config.numbers_drawn / config.number_range
        exp_recent_freq = exp_recent / recent_window
        score_mom = recent_freq * W_MOM

        # 3. recență exponențială bazată pe gap‑ul ultimei apariţii
        gaps = np.array([
            draw_index - last_seen.get(n, -1)
            for n in numbers
        ], dtype=float)
        rho = np.exp(-gaps / TAU)
        score_rec = rho * W_REC

        # 4. standardizarea gap‑ului (overdue vs. over‑hit)
        gap_std = (gaps - EXPECTED_GAP) / STD_GAP
        gap_std = np.clip(gap_std, -GAP_CLIP, GAP_CLIP)
        score_gap = gap_std * W_GAP

        # 5. deviație sectorială (Poisson‑like) pe fereastra recentă
        # determinăm sectorul fiecărui număr (1‑7, 8‑14, ...)
        sector_idx = np.array([( (n-1) // SECTOR_SIZE ) for n in numbers])
        # contor sector în fereastra recentă
        sector_counts = np.zeros(49 // SECTOR_SIZE, dtype=float)
        for draw_set in recent_deque:
            for n in draw_set:
                sector_counts[(n-1)//SECTOR_SIZE] += 1
        # așteptarea sectorială în fereastra recentă
        exp_sector = recent_window * config.numbers_drawd / config.number_range * SECTOR_SIZE
        sector_z = (sector_counts - exp_sector) / np.sqrt(exp_sector + 1e-9)
        score_sect = sector_z[sector_idx] * W_SECT

        # 6. interacţiune non‑liniară între momentum recent (standardizat) şi gap‑std
        # standardizăm recent_freq similar cu sector_z pentru a obţine recent_z
        recent_z = (recent_counts - exp_recent) / np.sqrt(exp_recent + 1e-9)
        score_int = (recent_z * gap_std) * W_INT

        # scor total
        total_score = score_zrev + score_mom + score_rec + score_gap + score_sect + score_int
        score_dict = {n: total_score[i] for i, n in enumerate(numbers)}
        # sortare descrescător după scor, apoi număr ascendent pentru tie‑break determinist
        return sorted(numbers, key=lambda n: (-score_dict[n], n))

    return rank