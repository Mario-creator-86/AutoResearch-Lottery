import numpy as np

NAME = "zoverdue"
DESCRIPTION = "Combines overdue time, recent co‑occurrence pairs and entropy gain to rank numbers."
SOURCE = "generated 2026-09-16"

def create(config):
    # config: LotteryConfig – only .number_range, .numbers_drawn, .ticket_size are used
    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # Prepare base list of numbers
        numbers = list(range(1, config.number_range + 1))

        # 1️⃣ Overdue score (higher = longer since last appearance)
        overdue_raw = {}
        for n in numbers:
            last = last_seen.get(n, -1)
            # If never seen, treat as maximum overdue (draw_index + 1)
            overdue_raw[n] = draw_index - last if last != -1 else draw_index + 1
        # Normalise overdue to [0,1]
        overdue_vals = np.array(list(overdue_raw.values()), dtype=float)
        overdue_min, overdue_max = overdue_vals.min(), overdue_vals.max()
        if overdue_max == overdue_min:
            overdue_norm = {n: 0.0 for n in numbers}
        else:
            overdue_norm = {n: (overdue_raw[n] - overdue_min) / (overdue_max - overdue_min) for n in numbers}

        # 2️⃣ Recent pair co‑occurrence score
        pair_raw = {n: 0 for n in numbers}
        for draw_set in recent_deque:
            k = len(draw_set)
            if k < 2:
                continue
            for n in draw_set:
                pair_raw[n] += k - 1  # count of other numbers in the same draw
        pair_vals = np.array(list(pair_raw.values()), dtype=float)
        pair_min, pair_max = pair_vals.min(), pair_vals.max()
        if pair_max == pair_min:
            pair_norm = {n: 0.0 for n in numbers}
        else:
            pair_norm = {n: (pair_raw[n] - pair_min) / (pair_max - pair_min) for n in numbers}

        # 3️⃣ Entropy gain if the number were added to the recent window
        # Compute current frequencies in the recent window
        freq = {n: 0 for n in numbers}
        total_counts = 0
        for draw_set in recent_deque:
            for n in draw_set:
                freq[n] += 1
                total_counts += 1
        # Edge case: empty recent window
        if total_counts == 0:
            # No entropy information – give zero gain to all numbers
            entropy_gain_norm = {n: 0.0 for n in numbers}
        else:
            # Current probability distribution
            probs = np.array([freq[n] / total_counts for n in numbers], dtype=float)
            # Avoid log(0) by masking zeros
            mask = probs > 0
            current_entropy = -np.sum(probs[mask] * np.log(probs[mask]))

            entropy_gain_raw = {}
            for n in numbers:
                new_total = total_counts + 1
                new_freq_n = freq[n] + 1
                new_probs = np.array([
                    (new_freq_n if i == n else freq[i]) / new_total for i in numbers
                ], dtype=float)
                new_mask = new_probs > 0
                new_entropy = -np.sum(new_probs[new_mask] * np.log(new_probs[new_mask]))
                entropy_gain_raw[n] = new_entropy - current_entropy
            # Normalise entropy gain (can be negative or positive)
            gain_vals = np.array(list(entropy_gain_raw.values()), dtype=float)
            gain_min, gain_max = gain_vals.min(), gain_vals.max()
            if gain_max == gain_min:
                entropy_gain_norm = {n: 0.0 for n in numbers}
            else:
                entropy_gain_norm = {n: (entropy_gain_raw[n] - gain_min) / (gain_max - gain_min) for n in numbers}

        # Weighted combination (weights sum to 1.0)
        w_overdue = 0.5
        w_pair = 0.3
        w_entropy = 0.2
        scores = {}
        for n in numbers:
            scores[n] = (
                w_overdue * overdue_norm[n]
                + w_pair * pair_norm[n]
                + w_entropy * entropy_gain_norm[n]
            )

        # Return full permutation sorted by descending score, tie‑break on numeric order
        return sorted(numbers, key=lambda n: (-scores[n], n))
    return rank