NAME = "zdppgree"
DESCRIPTION = "Greedy DPP‑like selector using z‑score, EWMA, Markov transitions and sector diversification."
SOURCE = "model + 2026-09-26"

import math
from collections import defaultdict

# ---- parametri tunabili (pentru ablatie) ----
W_Z = 1.0          # greutate z‑score lifetime
W_EWMA = 0.8       # greutate EWMA recent
W_PAIR = 0.6       # greutate scor tranziție Markov
W_SECT = 0.4       # penalizare sector
ALPHA_EWMA = 0.6   # factor de deces pentru EWMA
SECTOR_SIZE = 5    # 45/9 = 5 numere per sector

DIAG = {"orient_checks": 0, "orient_flips": 0, "orient_rule": "none"}


def create(config):
    N = int(config.number_range)          # 45
    K = int(config.numbers_drawn)         # 5
    p = K / float(N) if N > 0 else 0.0
    q = 1.0 - p

    sector_of = lambda n: (n - 1) // SECTOR_SIZE  # 0..8

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # ------------------- pre‑calculări -------------------
        numbers = list(range(1, N + 1))

        # 1) z‑score lifetime (standardizare empirică)
        vals = [counts.get(n, 0) for n in numbers]
        mu = sum(vals) / float(N) if N else 0.0
        var = sum((v - mu) ** 2 for v in vals) / float(N) if N else 0.0
        sigma = math.sqrt(var) if var > 1e-12 else 1.0
        z_score = {n: (counts.get(n, 0) - mu) / sigma for n in numbers}

        # 2) EWMA pe fereastra recentă
        ewma = defaultdict(float)
        for age, draw in enumerate(reversed(list(recent_deque))):
            w = ALPHA_EWMA ** age
            for n in draw:
                ewma[n] += w
        # normalizare simplă (nu e vitală)
        max_ew = max(ewma.values()) if ewma else 1.0
        ewma_norm = {n: ewma.get(n, 0.0) / max_ew for n in numbers}

        # 3) Matrice tranziție Markov (consecutive draws în fereastra recentă)
        trans = defaultdict(lambda: defaultdict(int))
        recent_list = list(recent_deque)
        for i in range(len(recent_list) - 1):
            older = recent_list[i]
            newer = recent_list[i + 1]
            for a in older:
                for b in newer:
                    trans[a][b] += 1
                    trans[b][a] += 1  # păstrez simetria pentru comoditate

        # 4) Funcții auxiliare pentru penalizare sector
        sector_counts = [0] * 9  # 9 sectoare de 5 numere
        def sector_penalty(sec_idx, add=0):
            cur = sector_counts[sec_idx]
            new = cur + add
            # penalizare marginală: (new/size)^2 - (cur/size)^2
            ss = SECTOR_SIZE
            return (new / ss) ** 2 - (cur / ss) ** 2

        selected = []
        remaining = set(numbers)

        # ------------------- selecție greedy -------------------
        while len(selected) < K and remaining:
            best_n = None
            best_score = -float('inf')
            for n in remaining:
                # scor marginal compus
                s = (W_Z * z_score[n] +
                     W_EWMA * ewma_norm[n] +
                     W_PAIR * sum(trans[n].get(sel, 0) for sel in selected) -
                     W_SECT * sector_penalty(sector_of(n), add=1))
                if s > best_score:
                    best_score = s
                    best_n = n
            if best_n is None:
                break
            selected.append(best_n)
            remaining.remove(best_n)
            sector_counts[sector_of(best_n)] += 1

        # ------------------- ordine pentru restul -------------------
        # Pentru numerele ne‑selectate, folosim doar semnalele de bază
        rest_order = sorted(remaining,
                            key=lambda n: (-(W_Z * z_score[n] + W_EWMA * ewma_norm[n]), n))

        # Îmbinăm: primul K numere sunt cele selectate în ordine greedy, apoi restul
        final_order = selected + rest_order
        return final_order

    return rank