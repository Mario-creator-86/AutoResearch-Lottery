import numpy as np
from collections import deque

NAME = "zzechrio"
DESCRIPTION = "Strategie multicomponentă ce combină z‑score, gap‑overdue, EWMA și afinitate Markov pentru SuperEnalotto"
SOURCE = "zechrior_2026-09-08"

def create(config):
    """Return the ranking function for SuperEnalotto.

    Parameters
    ----------
    config: LotteryConfig – obiect cu atributele .number_range, .numbers_drawn, .ticket_size
    """
    # ponderi implicite – pot fi supuse scanării ulterioare
    W_Z = 0.6      # z‑score
    W_GAP = 0.4    # overdue
    W_MOM = 0.5    # momentum EWMA
    W_AFF = 0.3    # afinitate Markov
    ALPHA = 0.10   # factor EWMA
    K_HOT = 0.90   # percentile pentru a defini setul de numere hot

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # ------------------------------------------------------------
        # 1. pregătirea listei numerelor
        # ------------------------------------------------------------
        numbers = list(range(1, config.number_range + 1))
        # ------------------------------------------------------------------
        # 2. componente statistice generale
        # ------------------------------------------------------------------
        cnt_values = np.array([counts.get(n, 0) for n in numbers], dtype=float)
        mean_cnt = cnt_values.mean()
        std_cnt = cnt_values.std(ddof=0) if cnt_values.std(ddof=0) > 0 else 1.0
        # z‑score normalizat în [0,1] cu limitare la ±3σ
        z_raw = (cnt_values - mean_cnt) / std_cnt
        z_clipped = np.clip(z_raw, -3, 3)
        score_z = (z_clipped + 3) / 6.0  # mapare la [0,1]
        # ------------------------------------------------------------------
        # 3. gap (draws since last appearance) – normalizat
        # ------------------------------------------------------------------
        max_gap = draw_index if draw_index > 0 else 1
        gap_array = np.empty(len(numbers), dtype=float)
        for i, n in enumerate(numbers):
            last = last_seen.get(n, -1)
            gap = draw_index - last if last != -1 else draw_index
            gap_array[i] = gap / max_gap
        score_gap = gap_array  # deja în [0,1]
        # ------------------------------------------------------------------
        # 4. momentum EWMA din fereastra recentă (recent_deque conţine seturi)
        # ------------------------------------------------------------------
        recent_counts = np.zeros(len(numbers), dtype=float)
        if recent_window > 0 and recent_deque:
            for draw_set in recent_deque:
                for n in draw_set:
                    if 1 <= n <= config.number_range:
                        recent_counts[n-1] += 1
            recent_freq = recent_counts / recent_window  # proporţie apariţii
        else:
            recent_freq = np.zeros(len(numbers))
        # EWMA simplificat: score = α * recent_freq + (1‑α) * previous_score (≈0 la început)
        score_mom = ALPHA * recent_freq  # rang [0, α]
        # ------------------------------------------------------------------
        # 5. afinitate Markov – co‑apariţii cu numere hot în fereastra recentă
        # ------------------------------------------------------------------
        # definim setul de numere hot pe baza percentilelor din counts
        hot_threshold = np.percentile(cnt_values, K_HOT * 100)
        hot_set = set([n for n, c in zip(numbers, cnt_values) if c >= hot_threshold])
        aff_scores = np.zeros(len(numbers), dtype=float)
        # acumulăm co‑apariţii în fereastra recentă
        for draw_set in recent_deque:
            intersect_hot = draw_set & hot_set
            if not intersect_hot:
                continue
            for n in draw_set:
                if 1 <= n <= config.number_range:
                    # fiecare apariţie cu fiecare hot (excluzând pe sine)
                    aff_scores[n-1] += len(intersect_hot) - (1 if n in intersect_hot else 0)
        # normalizare
        max_aff = aff_scores.max() if aff_scores.max() > 0 else 1.0
        score_aff = aff_scores / max_aff
        # ------------------------------------------------------------------
        # 6. agregare ponderată a scorurilor
        # ------------------------------------------------------------------
        total_score = (
            W_Z * score_z +
            W_GAP * score_gap +
            W_MOM * score_mom +
            W_AFF * score_aff
        )
        # ------------------------------------------------------------------
        # 7. întoarcerea rangului – sortare descrescătoare prin scor, tie‑break numeric
        # ------------------------------------------------------------------
        # construim dicţionar pentru acces rapid în lambda
        score_dict = {n: total_score[i] for i, n in enumerate(numbers)}
        ranked = sorted(numbers, key=lambda n: (-score_dict[n], n))
        return ranked

    return rank