NAME = "zewmasec"
DESCRIPTION = "Scorizeaza fiecare numar pe baza frecventei totale, EWMA a aparitiilor recente, intarzierea sectorului, prezenta in cea mai frecventa pereche recenta si contributia la entropia recenta. Returneaza lista sortata descrescator dupa scor."
SOURCE = "strategist:chatgpt_2024-09-21"

import collections
import numpy as np
import math

def create(config):
    # Config attributes: number_range, numbers_drawn, ticket_size
    # Greutati fixe pentru fiecare componenta
    W_FREQ = 1.0      # frecventa totala
    W_EWMA = 1.2      # trend recent EWMA
    W_SECT = 0.8      # penalizare sector subreprezentat
    W_PAIR = 1.5      # bonus pentru numerele din cea mai frecventa pereche recenta
    W_ENT  = 0.5      # bonus de entropie (favorizeaza numere rare recent)
    DECAY = 0.96      # factor de decadere pentru EWMA
    SECT_SIZE = 7     # 49 -> 7 sectoare de cate 7 numere

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # numbers 1..N
        numbers = np.arange(1, config.number_range + 1)
        # ------------------------------------------------------------
        # 1) frecventa totala (counts dict)
        total_counts = np.vectorize(lambda n: counts.get(int(n), 0))(numbers).astype(float)
        score_freq = total_counts * W_FREQ

        # ------------------------------------------------------------
        # 2) EWMA pe fereastra recenta
        # recent_deque este deque de seturi (numere din fiecare extractie recentă)
        ewma = np.zeros(config.number_range + 1, dtype=float)  # indexat de la 1
        # iteram de la cea mai veche la cea mai noua
        n_draws = len(recent_deque)
        for i, draw_set in enumerate(recent_deque):
            weight = DECAY ** (n_draws - i - 1)
            for num in draw_set:
                ewma[num] += weight
        ewma_scores = ewma[1:] * W_EWMA

        # ------------------------------------------------------------
        # 3) sector delay (cate extrageri din fiecare sector au aparut recent)
        sector_counts = np.zeros((config.number_range // SECT_SIZE) + 1, dtype=int)
        for draw_set in recent_deque:
            for num in draw_set:
                sector = (num - 1) // SECT_SIZE
                sector_counts[sector] += 1
        # intarziere = max_possible - actual
        max_draws_in_sector = recent_window * config.numbers_drawn
        sector_delay = max_draws_in_sector - sector_counts
        # fiecare numar primeste delay-ul sectorului lui
        sector_delay_per_num = np.array([sector_delay[(n-1)//SECT_SIZE] for n in numbers])
        score_sect = sector_delay_per_num * W_SECT

        # ------------------------------------------------------------
        # 4) cea mai frecventa pereche din fereastra recenta
        pair_counter = collections.Counter()
        for draw_set in recent_deque:
            draw_list = list(draw_set)
            for i in range(len(draw_list)):
                for j in range(i + 1, len(draw_list)):
                    a, b = sorted((draw_list[i], draw_list[j]))
                    pair_counter[(a, b)] += 1
        if pair_counter:
            top_pair = pair_counter.most_common(1)[0][0]
            top_pair_set = set(top_pair)
        else:
            top_pair_set = set()
        score_pair = np.array([W_PAIR if n in top_pair_set else 0.0 for n in numbers])

        # ------------------------------------------------------------
        # 5) bonus de entropie – penalizeaza numerele deja frecvente recent
        # frecventa relativa in fereastra recenta
        recent_total = recent_window * config.numbers_drawn
        recent_counts = np.zeros(config.number_range + 1, dtype=int)
        for draw_set in recent_deque:
            for num in draw_set:
                recent_counts[num] += 1
        # contributia la entropie daca am adauga numarul (simplificata ca 1/(freq+1))
        entropy_bonus = 1.0 / (recent_counts[1:].astype(float) + 1.0)
        score_ent = entropy_bonus * W_ENT

        # ------------------------------------------------------------
        # total score
        total_score = (
            score_freq +
            ewma_scores +
            score_sect +
            score_pair +
            score_ent
        )
        # Sortare descrescatoare dupa scor, la egalie crescator numar
        sorted_numbers = sorted(
            range(1, config.number_range + 1),
            key=lambda n: (-total_score[n-1], n)
        )
        return sorted_numbers

    return rank