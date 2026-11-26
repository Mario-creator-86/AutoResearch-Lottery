NAME = "zzewmase"
DESCRIPTION = "Strategie combinată: z‑score pe viață, EWMA recent, întârziere sectorială și entropie pentru Joker 5/45"
SOURCE = "model + 2026-09-26"

import math
import numpy as np
from collections import deque

# ----- constante tunabile (pentru sweep‑uri viitoare) -----
W_Z = 0.5          # greutate z‑score lifetime
W_EWMA = 1.0       # greutate EWMA recent
W_SECT = 0.7       # greutate sectorială (overdue)
W_ENT = -0.3       # greutate entropie (penalizare)
ALPHA_EWMA = 0.35  # factor de deșărtire pentru EWMA (0 = uniform, 1 = foarte recent)

DIAG = {
    "orientation_checks": 0,
    "orientation_flips": 0,
    "orientation_rule": "none",
}

def create(config):
    N = int(config.number_range)   # 45 pentru Joker
    K = int(config.numbers_drawn)  # 5 pentru Joker
    p = K / float(N) if N > 0 else 0.0
    q = 1.0 - p

    # pre‑calculăm valori folosite în scorul z‑standardizat
    def lifetime_z(counts):
        # media și deviaţia pentru un număr în setul total de extrageri
        total_draws = sum(counts.values()) / K if K > 0 else 0.0
        if total_draws < 1:
            return {n: 0.0 for n in range(1, N + 1)}
        mu = total_draws * p
        sd = math.sqrt(total_draws * p * q) if total_draws * p * q > 0 else 1.0
        return {n: (counts.get(n, 0) - mu) / sd for n in range(1, N + 1)}

    def ewma_recent(recent_deque):
        # recent_deque conţine seturi cu numerele din ultimele `recent_window` extrageri
        # EWMA: w_i = ALPHA_EWMA * (1-ALPHA_EWMA)^{i} ; i=0 este cea mai recentă
        ewma = {n: 0.0 for n in range(1, N + 1)}
        decay = 1.0 - ALPHA_EWMA
        factor = ALPHA_EWMA
        for i, draw_set in enumerate(reversed(recent_deque)):
            for n in draw_set:
                ewma[n] += factor
            factor *= decay
        return ewma

    def sector_overdue(last_seen, draw_idx):
        # Împărţim numerele în 5 sectoare (9 numere fiecare, ultimul are 9 (45) )
        sector_score = {s: 0.0 for s in range(5)}
        sector_count = {s: 0 for s in range(5)}
        for n in range(1, N + 1):
            sector = (n - 1) // 9
            # dacă nu a apărut niciodată, considerăm overdue maxim = draw_idx+1
            overdue = draw_idx - last_seen.get(n, -1)
            sector_score[sector] += overdue
            sector_count[sector] += 1
        # medie overdue per sector
        avg_overdue = {s: sector_score[s] / sector_count[s] for s in sector_score}
        # transformăm în dicţionar mapare număr -> overdue sectorial mediu
        return {n: avg_overdue[(n - 1) // 9] for n in range(1, N + 1)}

    def entropy_recent(recent_deque, window):
        # p = apariţii în fereastra recentă / window
        freq = {n: 0 for n in range(1, N + 1)}
        for draw_set in recent_deque:
            for n in draw_set:
                freq[n] += 1
        ent = {}
        for n in range(1, N + 1):
            p_n = freq[n] / float(window) if window > 0 else 0.0
            if p_n > 0.0:
                ent[n] = -p_n * math.log(p_n)
            else:
                ent[n] = 0.0
        return ent

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # orientare: verificăm că ultimul element al deque‑ului corespunde draw_index‑1
        if recent_deque:
            DIAG["orientation_checks"] += 1
            newest = recent_deque[-1]
            # numerele care au apărut în draw‑ul curent‑1 ar trebui să fie în newest
            if any(last_seen.get(n, -1) == draw_index - 1 for n in newest):
                pass  # ok
            else:
                DIAG["orientation_flips"] += 1
        # 1) z‑score lifetime
        z_scores = lifetime_z(counts)
        # 2) EWMA recent
        ewma_scores = ewma_recent(recent_deque)
        # 3) Sector overdue
        sect_scores = sector_overdue(last_seen, draw_index)
        # 4) Entropie recentă
        ent_scores = entropy_recent(recent_deque, recent_window)
        # calculăm scorul combinat
        final_scores = {}
        for n in range(1, N + 1):
            final = (
                W_Z * z_scores[n]
                + W_EWMA * ewma_scores[n]
                + W_SECT * sect_scores[n]
                + W_ENT * ent_scores[n]
            )
            final_scores[n] = final
        # returnăm o permutare completă sortată descrescător după scor, cu tie‑break pe număr
        numbers = list(range(1, N + 1))
        numbers.sort(key=lambda x: (-final_scores[x], x))
        return numbers

    return rank