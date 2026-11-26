NAME = "zzzpairg"
DESCRIPTION = "Combines recent pair co‑occurrence lift with a Michaelis‑Menten gap saturation and a small repeat penalty, using Borda‑style aggregation of normalized scores."
SOURCE = "quantitative model + 2026-09-26"

import math

# hyper‑parameters (tune via grid‑search later)
W_PAIR = 0.6          # weight of the pair‑lift signal
W_GAP = 0.3          # weight of the gap‑saturation signal
W_PEN = 0.1          # weight of the immediate‑repeat penalty
GAMMA = 0.9          # recency decay for pair lift
MM_K = 100.0         # Michaelis‑Menten constant for gap saturation
EPS = 1e-12

DIAG = {"orient": None, "orient_flips": 0}

def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))

        # ----- orientation detection (optional) -----
        if recent_deque:
            def mean_recency(st):
                tot = 0.0
                for n in st:
                    ls = last_seen.get(n)
                    age = draw_index - ls if ls is not None else draw_index
                    tot += age
                return tot / len(st) if st else float('inf')
            newest_first = mean_recency(recent_deque[-1]) < mean_recency(recent_deque[0])
            orient = "newest_first" if newest_first else "newest_last"
            if DIAG["orient"] is not None and DIAG["orient"] != orient:
                DIAG["orient_flips"] += 1
            DIAG["orient"] = orient

        # work with draws ordered from newest to oldest
        seq = list(recent_deque)[::-1]   # newest first

        # ----- 1) Pair‑lift signal -----
        pair_score = {n: 0.0 for n in numbers}
        # count co‑occurrences inside the recent window
        co_counts = {}
        for draw in seq:
            draw_list = list(draw)
            L = len(draw_list)
            for i in range(L):
                for j in range(i + 1, L):
                    a, b = draw_list[i], draw_list[j]
                    if a > b:
                        a, b = b, a
                    co_counts[(a, b)] = co_counts.get((a, b), 0) + 1

        # compute lift and distribute it with recency weight
        for age, draw in enumerate(seq):
            w = GAMMA ** age
            draw_list = list(draw)
            L = len(draw_list)
            for i in range(L):
                for j in range(i + 1, L):
                    a, b = draw_list[i], draw_list[j]
                    if a > b:
                        a, b = b, a
                    co = co_counts.get((a, b), 0)
                    freq_a = counts.get(a, 0) + EPS
                    freq_b = counts.get(b, 0) + EPS
                    lift = co / (freq_a * freq_b)
                    pair_score[a] += w * lift
                    pair_score[b] += w * lift

        # ----- 2) Gap‑saturation signal -----
        gap_score = {}
        for n in numbers:
            last = last_seen.get(n)
            gap = draw_index - last if last is not None else draw_index
            gap_score[n] = gap / (MM_K + gap)   # Michaelis‑Menten saturation

        # ----- 3) Immediate repeat penalty -----
        newest_draw = recent_deque[-1] if recent_deque else set()
        repeat_pen = {n: (1.0 if n in newest_draw else 0.0) for n in numbers}

        # ----- Normalise each component to [0,1] -----
        def normalize(d):
            vals = list(d.values())
            mn = min(vals)
            mx = max(vals)
            span = mx - mn + EPS
            return {k: (v - mn) / span for k, v in d.items()}

        pair_norm = normalize(pair_score)
        gap_norm = normalize(gap_score)

        # ----- Aggregate with Borda‑style weighted sum -----
        final_score = {}
        for n in numbers:
            final_score[n] = (
                W_PAIR * pair_norm[n] +
                W_GAP  * gap_norm[n] -
                W_PEN  * repeat_pen[n]
            )

        # sort numbers by descending score, break ties by numerical order
        return sorted(numbers, key=lambda n: (-final_score[n], n))

    return rank