NAME = "zpairhaz"
DESCRIPTION = "Combină z‑score, EWMA, hazard over‑due și afinitatea de pereche pentru a ordona numerele"
SOURCE = "custom 2026-09-08"

def create(config):
    import math
    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # Listă completă a numerelor
        numbers = list(range(1, config.number_range + 1))
        # ----- statistici de bază -----
        total_counts = [counts.get(n, 0) for n in numbers]
        mean_counts = sum(total_counts) / len(total_counts)
        var = sum((c - mean_counts) ** 2 for c in total_counts) / len(total_counts)
        std_counts = math.sqrt(var) if var > 0 else 1.0
        # ----- hiper‑parametri -----
        w_z = 0.8       # greutate z‑score
        w_ewma = 0.5    # greutate EWMA
        w_due = 0.6     # greutate hazard over‑due
        w_pair = 0.4    # greutate afinitate de pereche
        gamma = 0.7     # factor de decădere pentru EWMA
        # ----- 1. Z‑score normalizat în [0,1] -----
        z_scores = {}
        for n in numbers:
            z = (counts.get(n, 0) - mean_counts) / std_counts
            # presupunem că z ∈ [-3, 3]; reproducem în intervalul [0,1]
            z_norm = (z + 3) / 6.0
            z_norm = min(max(z_norm, 0.0), 1.0)
            z_scores[n] = z_norm
        # ----- 2. EWMA pe fereastra recentă -----
        ewma_raw = {n: 0.0 for n in numbers}
        for i, draw in enumerate(reversed(recent_deque)):
            weight = gamma ** i
            for n in draw:
                ewma_raw[n] += weight
        max_ewma = max(ewma_raw.values()) if ewma_raw else 1.0
        ewma_scores = {n: (ewma_raw[n] / max_ewma) for n in numbers}
        # ----- 3. Hazard over‑due (log‑scale) -----
        due_raw = {}
        for n in numbers:
            gap = draw_index - last_seen.get(n, -1)  # dacă nu a apărut niciodată, gap = draw_index+1
            due_raw[n] = math.log(gap + 1)
        max_due = max(due_raw.values()) if due_raw else 1.0
        due_scores = {n: (due_raw[n] / max_due) for n in numbers}
        # ----- 4. Afinitate de pereche cu ultima extragere -----
        last_draw = recent_deque[-1] if recent_deque else set()
        pair_raw = {n: 0 for n in numbers}
        for draw in recent_deque:
            if not draw:
                continue
            if draw & last_draw:  # există cel puțin un număr comun cu ultima extragere
                for n in draw:
                    pair_raw[n] += 1
        max_pair = max(pair_raw.values()) if pair_raw else 1.0
        pair_scores = {n: (pair_raw[n] / max_pair) for n in numbers}
        # ----- scor final agregat -----
        final_scores = {}
        for n in numbers:
            final_scores[n] = (
                w_z * z_scores[n] +
                w_ewma * ewma_scores[n] +
                w_due * due_scores[n] +
                w_pair * pair_scores[n]
            )
        # Returnează permutarea completă, sortată descrescător după scor, apoi numeric
        return sorted(numbers, key=lambda n: (-final_scores[n], n))
    return rank