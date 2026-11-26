NAME = "zzpair2e"
DESCRIPTION = "Combină z‑score al frecvenței pe viață, EWMA recentă, co‑apariție cu ultima extragere, sector‑overdue și entropie moderată pentru a genera un ranking al numerelor."
SOURCE = "generated + 2026-11-22"

import math
import numpy as np

# ponderi (ajustate manual, pot fi supusă sweep‑ului ulterior)
W_Z = 0.40
W_EWMA = 0.30
W_PAIR = 0.20
W_SECTOR = 0.07
W_ENT = 0.03

ALPHA_EWMA = 0.35  # factor de amortizare pentru EWMA

def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    p_draw = K / float(N) if N else 0.0

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        total_counts = sum(counts.get(n, 0) for n in numbers)
        # numărul total de extrageri din istoric (fiecare extragere are K numere)
        total_draws = total_counts // K if K else 0
        # parametrii distribuției binomiale pentru frecvența pe viață
        mu = total_draws * p_draw
        sigma = math.sqrt(total_draws * p_draw * (1.0 - p_draw)) if total_draws > 0 else 1.0

        # prelucrări pentru semnalul sector‑overdue
        sector_gap = {s: 0.0 for s in range(5)}  # 5 sectoare de câte 9 numere
        sector_cnt = {s: 0 for s in range(5)}
        for n in numbers:
            sector = (n - 1) // 9
            gap = draw_index - last_seen.get(n, -1)
            sector_gap[sector] += gap
            sector_cnt[sector] += 1
        sector_score = {s: (sector_gap[s] / sector_cnt[s] if sector_cnt[s] else 0.0) for s in sector_gap}

        # ultima extragere (dacă există)
        latest_set = set()
        if recent_deque:
            latest_set = recent_deque[-1]

        # pre‑calculăm prezența recentă pentru EWMA și entropie
        recent_len = max(len(recent_deque), 1)
        recent_presence = {n: 0 for n in numbers}
        for draw in recent_deque:
            for n in draw:
                recent_presence[n] += 1
        ewma_score = {}
        entropy_score = {}
        for n in numbers:
            # EWMA (exponential decay pe fereastra recentă)
            w = 0.0
            for i, draw in enumerate(reversed(recent_deque)):
                if n in draw:
                    w += ALPHA_EWMA * ((1.0 - ALPHA_EWMA) ** i)
            ewma_score[n] = w
            # entropie bazată pe frecvența recentă
            p_n = recent_presence[n] / float(recent_len)
            if p_n in (0.0, 1.0):
                ent = 0.0
            else:
                ent = - (p_n * math.log(p_n) + (1.0 - p_n) * math.log(1.0 - p_n))
            # transformăm astfel încât p_n≈0.5 să aibă scor maxim
            entropy_score[n] = ent

        # pair‑score: numărul de extrageri din fereastra recentă în care n a apărut împreună cu VREUN număr din latest_set
        pair_score = {n: 0 for n in numbers}
        if latest_set:
            for draw in recent_deque:
                if len(draw & latest_set) >= 2:  # există cel puțin două numere comune, deci co‑apariție posibilă
                    for n in draw:
                        if n in latest_set:
                            continue
                        if len(draw & latest_set) >= 2:
                            pair_score[n] += 1

        # scor final ponderat
        scores = {}
        for n in numbers:
            # z‑score live
            z = (counts.get(n, 0) - mu) / sigma if sigma != 0 else 0.0
            sec = sector_score[(n - 1) // 9]
            scores[n] = (
                W_Z * z +
                W_EWMA * ewma_score[n] +
                W_PAIR * pair_score[n] +
                W_SECTOR * sec +
                W_ENT * entropy_score[n]
            )

        # sortăm descrescător după scor și, ca tie‑breaker, pe număr ascendent
        return sorted(numbers, key=lambda x: (-scores[x], x))

    return rank