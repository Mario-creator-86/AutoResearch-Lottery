NAME = "zzcombo5"
DESCRIPTION = "Strategie combinată: z‑score, EWMA recentă, frecvența de perechi și penalizare pentru numerele overdue."
SOURCE = "model + 2026-09-08"

import math
import numpy as np
from collections import deque

# ---- constante tunabile (pentru grid‑search/ablatie) ----
W_Z = 0.30          # pondere z‑score
W_EWMA = 0.30       # pondere EWMA recentă
W_PAIR = 0.20       # pondere frecvență de perechi
W_OVERDUE = 0.20    # pondere overdue (gap)
GAMMA_EWMA = 0.93    # factor de declin exponențial pentru EWMA

DIAG = {"orient_checks": 0, "orient_flips": 0, "orient_rule": "none"}


def create(config):
    """Return a rank function compatible with the platform.
    config.number_range – N (ex. 45)
    config.numbers_drawn – K (ex. 5)
    """
    N = int(config.number_range)
    K = int(config.numbers_drawn)

    # pre‑compute a list of all possible numbers for speed
    ALL_NUMS = list(range(1, N + 1))

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # ------------------------------------------------------------------
        # 1️⃣  Z‑score pe frecvența totală (lifetime)
        # ------------------------------------------------------------------
        total_hits = sum(counts.get(n, 0) for n in ALL_NUMS)
        avg_hits = total_hits / float(N) if N else 0.0
        # estimare simplă a deviației standard (Poisson‑like)
        std_hits = math.sqrt(avg_hits) if avg_hits > 0 else 1.0
        z_scores = {}
        for n in ALL_NUMS:
            z_scores[n] = (counts.get(n, 0) - avg_hits) / std_hits
        # normalizare la [0,1]
        z_vals = np.array([z_scores[n] for n in ALL_NUMS])
        z_min, z_max = z_vals.min(), z_vals.max()
        if z_max - z_min > 0:
            z_norm = (z_vals - z_min) / (z_max - z_min)
        else:
            z_norm = np.zeros_like(z_vals)

        # ------------------------------------------------------------------
        # 2️⃣  EWMA pe fereastra recentă (momentum)
        # ------------------------------------------------------------------
        ewma_scores = np.zeros(N)
        if recent_deque:
            L = len(recent_deque)
            for idx, draw_set in enumerate(recent_deque):
                # weight = gamma^(distance from newest)
                weight = GAMMA_EWMA ** (L - idx - 1)
                for n in draw_set:
                    ewma_scores[n - 1] += weight
            # normalizare
            if ewma_scores.max() > 0:
                ewma_norm = ewma_scores / ewma_scores.max()
            else:
                ewma_norm = np.zeros_like(ewma_scores)
        else:
            ewma_norm = np.zeros_like(ewma_scores)

        # ------------------------------------------------------------------
        # 3️⃣  Frecvența de apariție în perechi (co‑ocurențe) pe fereastra recentă
        # ------------------------------------------------------------------
        pair_counts = np.zeros(N)
        if recent_deque:
            for draw_set in recent_deque:
                size = len(draw_set)
                if size < 2:
                    continue
                incr = size - 1  # fiecare număr are (size‑1) parteneri în acea extragere
                for n in draw_set:
                    pair_counts[n - 1] += incr
            if pair_counts.max() > 0:
                pair_norm = pair_counts / pair_counts.max()
            else:
                pair_norm = np.zeros_like(pair_counts)
        else:
            pair_norm = np.zeros_like(pair_counts)

        # ------------------------------------------------------------------
        # 4️⃣  Penalizare overdue (gap de la ultima apariție) – cu cât gap‑ul e mai mare, cu atât scorul crește
        # ------------------------------------------------------------------
        overdue_scores = np.zeros(N)
        max_gap = 0
        for n in ALL_NUMS:
            gap = draw_index - last_seen.get(n, -1)
            overdue_scores[n - 1] = gap
            if gap > max_gap:
                max_gap = gap
        if max_gap > 0:
            overdue_norm = overdue_scores / float(max_gap)
        else:
            overdue_norm = np.zeros_like(overdue_scores)

        # ------------------------------------------------------------------
        # 5️⃣  Combinație ponderată
        # ------------------------------------------------------------------
        combined = (
            W_Z * z_norm +
            W_EWMA * ewma_norm +
            W_PAIR * pair_norm +
            W_OVERDUE * overdue_norm
        )

        # ------------------------------------------------------------------
        # 6️⃣  Verificare orientare deque (debug, nu afectează scorul)
        # ------------------------------------------------------------------
        if recent_deque:
            DIAG["orient_checks"] += 1
            newest = recent_deque[-1]
            oldest = recent_deque[0]
            # count how many numbers in newest/oldest coincide with last_seen index D-1
            newest_hits = sum(1 for n in newest if last_seen.get(n, -1) == draw_index - 1)
            oldest_hits = sum(1 for n in oldest if last_seen.get(n, -1) == draw_index - 1)
            if oldest_hits > newest_hits:
                DIAG["orient_flips"] += 1

        # ------------------------------------------------------------------
        # 7️⃣  Returnare rangare completă (descrescător după scor, apoi id‑numeric)
        # ------------------------------------------------------------------
        score_dict = {n: combined[n - 1] for n in ALL_NUMS}
        ranked = sorted(ALL_NUMS, key=lambda n: (-score_dict[n], n))
        return ranked

    return rank