import math
import numpy as np

NAME = "zecombo5"
DESCRIPTION = "Strategie hibridă care combină EWMA, co‑apariție, întârziere pe sectoare, entropie și corecție pentru sub‑reprezentare."
SOURCE = "model + 2026-09-15"

# ---- ponderi pentru semnale (ablatie ulterioară) ----
W_EWMA = 0.35   # EWMA pe frecvența recentă
W_PAIR = 0.30   # co‑apariție cu ultima extragere
W_SEC  = 0.15   # întârziere pe sector
W_ENT  = 0.10   # entropie (uniformitate recentă)
W_DEV  = 0.10   # deviație față de frecvența așteptată

HALF_LIFE = 20  # număr de extrageri pentru EWMA


def create(config):
    # config: LotteryConfig cu .number_range, .numbers_drawn, .ticket_size
    N = config.number_range
    K = config.numbers_drawn
    lambda_ = math.log(2) / HALF_LIFE if HALF_LIFE > 0 else 0.0

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # --- pregătim date de bază ---
        numbers = list(range(1, N + 1))
        total_occurrences = sum(counts.get(n, 0) for n in numbers)
        expected_mean = total_occurrences / float(N) if N else 0.0

        # 1) EWMA pe frecvența recentă
        ewma_score = {n: 0.0 for n in numbers}
        deque_len = len(recent_deque)
        for idx, draw_set in enumerate(recent_deque):
            age = deque_len - 1 - idx  # 0 pentru cea mai recentă
            weight = math.exp(-lambda_ * age)
            for n in draw_set:
                ewma_score[n] += weight
        # normalizare (optional, nu strict necesară)
        max_ewma = max(ewma_score.values()) if ewma_score else 1.0
        if max_ewma > 0:
            for n in numbers:
                ewma_score[n] /= max_ewma

        # 2) Co‑apariție cu ultima extragere (draw-ul cel mai recent din deque)
        pair_score = {n: 0.0 for n in numbers}
        if deque_len > 0:
            latest_set = recent_deque[-1]
            # construim contor de perechi în fereastra recentă
            pair_counts = {}
            for s in recent_deque:
                lst = list(s)
                for i in range(len(lst)):
                    for j in range(i + 1, len(lst)):
                        a, b = lst[i], lst[j]
                        key = (a, b) if a < b else (b, a)
                        pair_counts[key] = pair_counts.get(key, 0) + 1
            # scor pentru fiecare număr = sumă contor perechi cu elemente din latest_set
            for n in numbers:
                score = 0.0
                for m in latest_set:
                    if n == m:
                        continue
                    key = (n, m) if n < m else (m, n)
                    score += pair_counts.get(key, 0)
                pair_score[n] = score
            # normalizare
            max_pair = max(pair_score.values()) if pair_score else 1.0
            if max_pair > 0:
                for n in numbers:
                    pair_score[n] /= max_pair

        # 3) Întârziere pe sector (5 sectoare de câte 9 numere)
        sec_score = {n: 0.0 for n in numbers}
        sector_age_sum = [0.0] * 5
        sector_cnt = [0] * 5
        for n in numbers:
            sector = (n - 1) // 9  # 0‑4
            last = last_seen.get(n, -1)
            age = draw_index - last if last >= 0 else draw_index + 1
            sector_age_sum[sector] += age
            sector_cnt[sector] += 1
        sector_avg = [sector_age_sum[i] / sector_cnt[i] if sector_cnt[i] else 0.0 for i in range(5)]
        # fiecare număr primește scorul mediu al propriului sector
        for n in numbers:
            sector = (n - 1) // 9
            sec_score[n] = sector_avg[sector]
        max_sec = max(sec_score.values()) if sec_score else 1.0
        if max_sec > 0:
            for n in numbers:
                sec_score[n] /= max_sec

        # 4) Entropie (uniformitate recentă)
        ent_score = {n: 0.0 for n in numbers}
        recent_counts = {n: 0 for n in numbers}
        for s in recent_deque:
            for n in s:
                recent_counts[n] += 1
        total_recent_occ = sum(recent_counts.values())
        avg_recent = total_recent_occ / float(N) if N else 0.0
        for n in numbers:
            dev = abs(recent_counts[n] - avg_recent)
            ent_score[n] = -dev  # penalizăm deviația de la uniform
        # normalizare (valori negative, aducem la interval 0‑1)
        min_ent = min(ent_score.values())
        max_ent = max(ent_score.values())
        range_ent = max_ent - min_ent if max_ent != min_ent else 1.0
        for n in numbers:
            ent_score[n] = (ent_score[n] - min_ent) / range_ent

        # 5) deviație față de frecvența așteptată (favorizează sub‑reprezentarea)
        dev_score = {n: 0.0 for n in numbers}
        for n in numbers:
            dev = counts.get(n, 0) - expected_mean
            dev_score[n] = -dev
        max_dev = max(dev_score.values()) if dev_score else 1.0
        if max_dev > 0:
            for n in numbers:
                dev_score[n] /= max_dev

        # --- combinăm semnalele ---
        final_score = {}
        for n in numbers:
            total = (
                W_EWMA * ewma_score.get(n, 0.0) +
                W_PAIR * pair_score.get(n, 0.0) +
                W_SEC  * sec_score.get(n, 0.0) +
                W_ENT  * ent_score.get(n, 0.0) +
                W_DEV  * dev_score.get(n, 0.0)
            )
            final_score[n] = total

        # sortăm descrescător după scor, tie‑break pe număr mic
        sorted_numbers = sorted(numbers, key=lambda n: (-final_score[n], n))
        return sorted_numbers

    return rank