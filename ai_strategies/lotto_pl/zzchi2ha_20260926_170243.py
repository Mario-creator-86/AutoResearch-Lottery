NAME = "zzchi2ha"
DESCRIPTION = "Combinație de EWMA a frecvenței recente, halo de co‑apariție, recurență pe sectoare și surpriză de entropie, toate ponderate liniar și sortate pentru a produce o permutare completă a numerelor 1..49."
SOURCE = "zchi2halo_2026-09-26.py"

import math
import numpy as np
from collections import deque

def create(config):
    """Return a rank function compatible with the optimisation framework.

    Parameters
    ----------
    config: LotteryConfig
        Must expose .number_range (e.g., 49) and .numbers_drawn (e.g., 6).
    """
    N = config.number_range
    decay = 0.9  # exponential decay factor for recent draws
    sector_size = 7  # 49 numbers → 7 sectors of 7 numbers each
    # Fixed weights for the four scoring blocks
    w_ewma = 0.6
    w_halo = 0.8
    w_sector = 0.4
    w_surprise = 0.3

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # ------------------------------------------------------------
        # 1. Detect orientation of recent_deque (newest‑first vs oldest‑first)
        # ------------------------------------------------------------
        dq = recent_deque
        L = len(dq)
        if L == 0:
            newest_first = True
        else:
            # sum of last_seen indices for the first and last element
            sum_first = sum(last_seen.get(n, -1) for n in dq[0])
            sum_last = sum(last_seen.get(n, -1) for n in dq[-1])
            newest_first = sum_first < sum_last  # newer draws contain larger indices

        # ------------------------------------------------------------
        # 2. EWMA of raw frequencies using the recent window only
        # ------------------------------------------------------------
        ewma_counts = {n: 0.0 for n in range(1, N + 1)}
        for idx, draw_set in enumerate(dq):
            age = (L - idx - 1) if newest_first else idx
            weight = decay ** age
            for n in draw_set:
                ewma_counts[n] += weight

        # ------------------------------------------------------------
        # 3. Halo score – sum of weighted co‑occurrences within the recent window
        # ------------------------------------------------------------
        halo_scores = {n: 0.0 for n in range(1, N + 1)}
        for idx, draw_set in enumerate(dq):
            age = (L - idx - 1) if newest_first else idx
            weight = decay ** age
            draw_list = list(draw_set)
            m = len(draw_list)
            for i in range(m):
                a = draw_list[i]
                for j in range(i + 1, m):
                    b = draw_list[j]
                    halo_scores[a] += weight
                    halo_scores[b] += weight

        # ------------------------------------------------------------
        # 4. Sector recency – inverse of the gap since the sector last appeared
        # ------------------------------------------------------------
        # Determine the most recent appearance of any number within each sector
        sector_last = {}
        for n in range(1, N + 1):
            sec = (n - 1) // sector_size
            ls = last_seen.get(n, -1)
            if ls > sector_last.get(sec, -1):
                sector_last[sec] = ls
        sector_score = {n: 0.0 for n in range(1, N + 1)}
        for sec, ls in sector_last.items():
            gap = draw_index - ls if ls >= 0 else recent_window + 1
            value = 1.0 / (gap + 1)
            start = sec * sector_size + 1
            end = min(N, (sec + 1) * sector_size)
            for n in range(start, end + 1):
                sector_score[n] = value

        # ------------------------------------------------------------
        # 5. Surprise (entropy‑like) based on recent empirical distribution
        # ------------------------------------------------------------
        recent_counts = {}
        total_recent = 0
        for draw_set in dq:
            for n in draw_set:
                recent_counts[n] = recent_counts.get(n, 0) + 1
                total_recent += 1
        surprise = {}
        eps = 1e-12
        for n in range(1, N + 1):
            p = recent_counts.get(n, 0) / total_recent if total_recent > 0 else 0.0
            surprise[n] = -math.log(p + eps)

        # ------------------------------------------------------------
        # 6. Combine all blocks linearly
        # ------------------------------------------------------------
        final_scores = {}
        for n in range(1, N + 1):
            score = (
                w_ewma * ewma_counts.get(n, 0.0)
                + w_halo * halo_scores.get(n, 0.0)
                + w_sector * sector_score.get(n, 0.0)
                + w_surprise * surprise.get(n, 0.0)
            )
            final_scores[n] = score

        # ------------------------------------------------------------
        # 7. Return a full permutation ordered by descending score
        # ------------------------------------------------------------
        return sorted(range(1, N + 1), key=lambda n: (-final_scores[n], n))

    return rank