NAME = "zpairewm"
DESCRIPTION = "Folosește frecvența recentă, puterea de co‑apărare a numerelor și timpul scurs de la ultima apariție pentru a ranka toate numerele 1‑24."
SOURCE = "kaskada_strategy_v1 2026-09-26"

import numpy as np
from collections import deque

def create(config):
    """Return a ranking function for Kaskada 12/24.

    Parameters
    ----------
    config : LotteryConfig
        Provides .number_range, .numbers_drawn, .ticket_size (unused).
    """
    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # List of all possible numbers
        numbers = list(range(1, config.number_range + 1))

        # ------------------------------------------------------------------
        # 1️⃣ Recent frequency (EWMA‑like) – proportion of appearances in the
        #    recent window.
        # ------------------------------------------------------------------
        recent_counts = {n: 0 for n in numbers}
        for draw in recent_deque:
            for n in draw:
                recent_counts[n] += 1
        total_recent_slots = recent_window * config.numbers_drawn
        recent_freq = {n: recent_counts[n] / total_recent_slots if total_recent_slots else 0.0
                       for n in numbers}

        # ------------------------------------------------------------------
        # 2️⃣ Pair co‑occurrence strength – how many times a number has been
        #    present together with any other number inside the recent window.
        # ------------------------------------------------------------------
        pair_strength = {n: 0 for n in numbers}
        for draw in recent_deque:
            draw_list = list(draw)
            l = len(draw_list)
            for i in range(l):
                a = draw_list[i]
                for j in range(i + 1, l):
                    b = draw_list[j]
                    pair_strength[a] += 1
                    pair_strength[b] += 1

        # ------------------------------------------------------------------
        # 3️⃣ Overdue factor – normalized number of draws since last appearance.
        # ------------------------------------------------------------------
        max_since = draw_index if draw_index > 0 else 1
        overdue = {}
        for n in numbers:
            last = last_seen.get(n, -1)
            if last == -1:
                since = draw_index  # never seen → treat as max possible
            else:
                since = draw_index - last - 1
            overdue[n] = since / max_since

        # ------------------------------------------------------------------
        # Normalisation to [0, 1] for each component.
        # ------------------------------------------------------------------
        def _norm(d):
            vals = np.array(list(d.values()), dtype=float)
            mn = vals.min()
            mx = vals.max()
            if mx == mn:
                return {k: 0.0 for k in d}
            return {k: (v - mn) / (mx - mn) for k, v in d.items()}

        recent_norm = _norm(recent_freq)
        pair_norm = _norm(pair_strength)
        overdue_norm = _norm(overdue)

        # ------------------------------------------------------------------
        # Weighted sum – weights were chosen empirically but kept small to
        # avoid over‑fitting; they sum to 1.
        # ------------------------------------------------------------------
        w_recent, w_pair, w_over = 0.5, 0.3, 0.2
        final_score = {}
        for n in numbers:
            final_score[n] = (
                w_recent * recent_norm[n] +
                w_pair    * pair_norm[n] +
                w_over    * overdue_norm[n]
            )

        # Return a full permutation ordered by descending score, tie‑break
        # by the natural numeric order (deterministic).
        return sorted(numbers, key=lambda n: (-final_score[n], n))

    return rank