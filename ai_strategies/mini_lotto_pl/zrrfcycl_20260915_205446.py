import math
from collections import defaultdict

NAME = "zrrfcycl"
DESCRIPTION = ('Fuziune neliniara RRF (k=60) a patru ordonari: (1) afinitatea de perechi co-aparute cu '
               'numerele extragerii celei mai recente (ancora derivata doar din last_seen, independenta '
               'de ordinea deque-ului), (2) alinierea gap-ului curent la ciclul propriu de inter-arrivale '
               'al fiecarui numar (media gap-urilor din fereastra recenta), (3) intarzierea medie pe '
               'sectoare de 6 numere fata de asteptarea 8.4 extrageri, (4) reversia deviatiei de frecventa '
               'pe termen lung (z-score al numaratorii totale). Combinare de tip reciprocal-rank, fara '
               'parametri calibrati pe istoric.')
SOURCE = 'glm-5.3-flash + 2026-09-15'


def create(config):
    R = int(config.number_range)
    K = int(config.numbers_drawn)
    RRF_K = 60.0
    W_PAIR = 1.0
    W_CYCLE = 0.7
    W_SECTOR = 0.5
    W_DEV = 0.4
    N_SECT = 7

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, R + 1))
        p = K / float(R)
        exp_gap = R / float(K)

        # Ancora: numerele celei mai recente extrageri, doar din last_seen (orientare-agnostic)
        if last_seen:
            mx = max(last_seen.values())
            anchor = set(n for n in numbers if last_seen.get(n, -10 ** 9) == mx)
        else:
            anchor = set()

        # (1) Co-occcurente perechi cu ancora, in fereastra recenta (fara a depinde de ordine)
        co = defaultdict(int)
        for s in recent_deque:
            if not (s & anchor):
                continue
            sl = sorted(s)
            for i in range(len(sl)):
                for j in range(i + 1, len(sl)):
                    a, b = sl[i], sl[j]
                    if a in anchor or b in anchor:
                        co[(a, b)] += 1
        aff = {}
        for n in numbers:
            a = 0
            for m in anchor:
                if m != n:
                    a += co.get((min(n, m), max(n, m)), 0)
            aff[n] = a

        # (2) Ciclul propriu de inter-arrivale: offseturile sortate fac gapurile invariante la orientare
        pos = defaultdict(list)
        for idx, s in enumerate(recent_deque):
            for n in s:
                pos[n].append(idx)
        gap_dist = {}
        for n in numbers:
            ps = sorted(pos.get(n, ()))
            if len(ps) >= 2:
                gaps = [ps[i + 1] - ps[i] for i in range(len(ps) - 1)]
                cyc = sum(gaps) / float(len(gaps))
            else:
                cyc = exp_gap
            g = (draw_index - last_seen[n]) if n in last_seen else int(3 * exp_gap)
            gap_dist[n] = abs(g - cyc)

        # (3) Intarziere medie pe sectoare de cate 6 numere
        per = max(1, R // N_SECT)
        nsec = (R + per - 1) // per
        ssum = [0.0] * nsec
        scnt = [0] * nsec
        for n in numbers:
            si = (n - 1) // per
            d = (draw_index - last_seen[n]) if n in last_seen else int(10 * exp_gap)
            ssum[si] += d
            scnt[si] += 1
        soff = {n: (ssum[(n - 1) // per] / scnt[(n - 1) // per]) - exp_gap for n in numbers}

        # (4) Deviatie long-run fata de asteptat; reversie la medie (coldest long-run mai sus)
        T = float(draw_index) if draw_index > 0 else 1.0
        exp_cnt = T * p
        sd = math.sqrt(T * p * (1.0 - p)) + 1e-9
        zdev = {n: (counts.get(n, 0) - exp_cnt) / sd for n in numbers}

        # Reciprocal Rank Fusion (non-liniar, invariant la scala)
        rp = {n: i for i, n in enumerate(sorted(numbers, key=lambda x: (-aff[x], x)))}
        rg = {n: i for i, n in enumerate(sorted(numbers, key=lambda x: (gap_dist[x], x)))}
        rsec = {n: i for i, n in enumerate(sorted(numbers, key=lambda x: (-soff[x], x)))}
        rz = {n: i for i, n in enumerate(sorted(numbers, key=lambda x: (zdev[x], x)))}

        score = {n: (W_PAIR / (RRF_K + rp[n])
                     + W_CYCLE / (RRF_K + rg[n])
                     + W_SECTOR / (RRF_K + rsec[n])
                     + W_DEV / (RRF_K + rz[n]))
                 for n in numbers}
        return sorted(numbers, key=lambda n: (-score[n], n))

    return rank