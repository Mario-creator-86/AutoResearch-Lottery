NAME = "zrrfanch"
DESCRIPTION = 'Fuziune RRF a 3 mecanisme order-agnostic: deviatie fata de frecventa asteptata (memorie lunga), recenta exponentiala pe last_seen (GAMMA fix 0.96) si afinitatea de perechi recente cu ancorele de top (non-liniar, RRF_K fix 60); reordonare finala greedy cu bonus de echilibru pe sectoare.'
SOURCE = 'glm-5.3-flash + 2026-09-15'

import math

GAMMA = 0.96
RRF_K = 60.0
W_DEV = 0.50
W_REC = 0.70
W_PAIR = 0.40
W_DIV = 0.15
ANCHOR_K = 3


def create(config):
    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        N = config.number_range
        numbers = list(range(1, N + 1))

        # M1: deviatie fata de frecventa asteptata (z-score empiric pe tot istoricul)
        vals = [counts.get(n, 0) for n in numbers]
        mean_c = sum(vals) / float(len(vals))
        var_c = sum((v - mean_c) * (v - mean_c) for v in vals) / float(len(vals))
        std_c = math.sqrt(var_c)
        if std_c < 1e-12:
            std_c = 1.0
        dev = {n: (counts.get(n, 0) - mean_c) / std_c for n in numbers}

        # M2: recenta exponentiala, doar din last_seen si draw_index (order-agnostic)
        rec = {}
        for n in numbers:
            ls = last_seen.get(n)
            age = (draw_index - ls) if ls is not None else (draw_index + 1)
            if age < 0:
                age = 0
            rec[n] = GAMMA ** age

        def mech_ranks(smap, descending):
            if descending:
                order = sorted(numbers, key=lambda x: (-smap[x], x))
            else:
                order = sorted(numbers, key=lambda x: (smap[x], x))
            return {x: i + 1 for i, x in enumerate(order)}

        r_dev = mech_ranks(dev, True)
        r_rec = mech_ranks(rec, True)
        base = {n: W_DEV / (RRF_K + r_dev[n]) + W_REC / (RRF_K + r_rec[n]) for n in numbers}

        # M3: co-aparitii in fereastra recenta (seturi sortate -> independent de ordinea deque-ului)
        pc = {}
        for s in recent_deque:
            items = sorted(set(s))
            L = len(items)
            for i in range(L):
                ai = items[i] * 100
                for j in range(i + 1, L):
                    k = ai + items[j]
                    pc[k] = pc.get(k, 0) + 1

        # ancorare non-liniara: top-K dupa scorul de baza, apoi afinitatea recenta cu ancorele
        anchor_order = sorted(numbers, key=lambda x: (-base[x], x))[:ANCHOR_K]
        pair_score = {}
        for n in numbers:
            t = 0
            for a in anchor_order:
                if a == n:
                    continue
                k = (a * 100 + n) if a < n else (n * 100 + a)
                t += pc.get(k, 0)
            pair_score[n] = t
        r_pair = mech_ranks(pair_score, True)

        # fuziune RRF
        score = {n: base[n] + W_PAIR / (RRF_K + r_pair[n]) for n in numbers}
        max_score = (W_DEV + W_REC + W_PAIR) / (RRF_K + 1.0)
        if max_score < 1e-12:
            max_score = 1.0

        # reordonare greedy: scor normalizat + bonus de echilibru pe sectoare
        n_sec = (N - 1) // 10 + 1
        target = config.ticket_size / float(n_sec)
        remaining = sorted(numbers, key=lambda x: (-score[x], x))
        picked = []
        sec_count = [0] * n_sec
        while remaining:
            best_i = 0
            best_key = -1.0e18
            for i, n in enumerate(remaining):
                gap = target - sec_count[(n - 1) // 10]
                bonus = W_DIV * gap if gap > 0.0 else 0.0
                key = score[n] / max_score + bonus
                if key > best_key:
                    best_key = key
                    best_i = i
            n = remaining.pop(best_i)
            picked.append(n)
            sec_count[(n - 1) // 10] += 1
        return picked

    return rank