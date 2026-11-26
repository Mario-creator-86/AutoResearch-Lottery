NAME = "zpcdvqua"
DESCRIPTION = 'Scor liniar din 4 componente standardizate (z): afinitate de perechi co-aparute (lift Bayesian x prezenta EWMA a partenerului), deviatia semnata a ratei EWMA recente fata de p=K/N (pondere flippabila momentum/reversiune), raport de densitate positional pe sloturile sortate fata de centrele a priori ale statisticii de ordine, si cadenta proprie (gapul curent raportat la media ultimelor K_TAIL intergape). Permutare completa 1..N, determinista, doar stdlib.'
SOURCE = 'model + 2026-09-08'

import math
from itertools import combinations

PARAMS = {
    'W_PAIR': 1.0,
    'W_DEV': 0.3,
    'W_POS': 0.5,
    'W_CAD': 0.5,
    'GAMMA_P': 0.97,
    'K_TAIL': 3,
    'SIGMA_POS': 6.0,
}

DIAG = {'orientation': 'unknown', 'orient_flips': 0, 'calls': 0}

_CAD_SIG = 0.8
_PRIOR_W = 8.0


def _zscore(vals):
    n = len(vals)
    if n == 0:
        return []
    m = sum(vals) / n
    var = sum((v - m) * (v - m) for v in vals) / n
    s = math.sqrt(var)
    if s < 1e-12:
        return [0.0] * n
    return [(v - m) / s for v in vals]


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    numbers = list(range(1, N + 1))
    prior_gap = (N / K) if K > 0 else 1.0
    e_slot = [(j + 1) * (N + 1) / (K + 1) for j in range(K)]

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        DIAG['calls'] += 1
        draws = list(recent_deque)
        L = len(draws)

        # orientare deque: capatul cu media last_seen mai mare este cel mai nou
        if L >= 2:
            def _mls(s):
                v = [last_seen[n] for n in s if n in last_seen]
                return (sum(v) / len(v)) if v else -1.0
            ml = _mls(draws[0])
            mr = _mls(draws[-1])
            if mr > ml + 1e-9:
                orient = 'newest_last'
            elif ml > mr + 1e-9:
                orient = 'newest_first'
            else:
                orient = DIAG['orientation'] if DIAG['orientation'] != 'unknown' else 'newest_last'
        else:
            orient = DIAG['orientation'] if DIAG['orientation'] != 'unknown' else 'newest_last'
        if DIAG['orientation'] != 'unknown' and DIAG['orientation'] != orient:
            DIAG['orient_flips'] += 1
        DIAG['orientation'] = orient
        if orient == 'newest_first':
            draws.reverse()

        # aliniere varste cu last_seen: offset = draw_index - index-ul ultimei extrageri
        if L > 0:
            newest = draws[-1]
            idxs = [last_seen[n] for n in newest if n in last_seen]
            offset = (draw_index - max(idxs)) if idxs else 1
            if offset < 0:
                offset = 0
        else:
            offset = 1

        p = (K / N) if N > 0 else 0.0
        gamma = float(PARAMS.get('GAMMA_P', 0.97))
        gamma = min(0.999, max(0.5, gamma))
        k_tail = int(PARAMS.get('K_TAIL', 3))
        if k_tail < 1:
            k_tail = 1
        sigma_pos = float(PARAMS.get('SIGMA_POS', 6.0))
        if sigma_pos <= 0:
            sigma_pos = 6.0
        w_pair = float(PARAMS.get('W_PAIR', 1.0))
        w_dev = float(PARAMS.get('W_DEV', 0.3))
        w_pos = float(PARAMS.get('W_POS', 0.5))
        w_cad = float(PARAMS.get('W_CAD', 0.5))

        ew_present = dict.fromkeys(numbers, 0.0)
        cnt_w = dict.fromkeys(numbers, 0)
        ages_by_n = {}
        for n in numbers:
            ages_by_n[n] = []
        slot_ew = [0.0] * K
        wsum = 0.0
        pair_cnt = {}

        for i in range(L):
            dr = draws[i]
            age = offset + (L - 1 - i)
            w = gamma ** age
            wsum += w
            srt = sorted(dr)
            nk = min(K, len(srt))
            for j in range(nk):
                slot_ew[j] += w * srt[j]
            for n in dr:
                if n in ew_present:
                    ew_present[n] += w
                    cnt_w[n] += 1
                    ages_by_n[n].append(age)
            for a, b in combinations(srt, 2):
                pair_cnt[(a, b)] = pair_cnt.get((a, b), 0) + 1

        # DEV: rata EW recenta minus p (semnul e controlat de W_DEV)
        if wsum > 0:
            dev_raw = [(ew_present[n] / wsum) - p for n in numbers]
        else:
            dev_raw = [0.0] * N

        # POS: raport de densitati sloturi-recente vs prior uniform
        eff = (wsum / (wsum + _PRIOR_W)) if wsum > 0 else 0.0
        targets = []
        for j in range(K):
            mj = (slot_ew[j] / wsum) if wsum > 0 else e_slot[j]
            targets.append(eff * mj + (1.0 - eff) * e_slot[j])
        inv2s2 = 1.0 / (2.0 * sigma_pos * sigma_pos)
        pos_raw = []
        for n in numbers:
            num = 0.0
            for t in targets:
                d = n - t
                num += math.exp(-d * d * inv2s2)
            den = 0.0
            for t in e_slot:
                d = n - t
                den += math.exp(-d * d * inv2s2)
            pos_raw.append(num / (den + 1e-9))

        # PAIR: lift de co-aparitie x prezenta EW a partenerului
        pair_raw = [0.0] * N
        if wsum > 0 and L >= 2:
            for ai in range(N):
                a = ai + 1
                ca = cnt_w[a]
                acc = 0.0
                for bi in range(N):
                    if bi == ai:
                        continue
                    b = bi + 1
                    key = (a, b) if a < b else (b, a)
                    c = pair_cnt.get(key, 0)
                    exp_ab = (ca * cnt_w[b]) / L
                    acc += ((c + 0.5) / (exp_ab + 0.5)) * ew_present[b]
                pair_raw[ai] = acc

        # CAD: adresarea gapului curent la cadenta proprie (ultimele K_TAIL gapuri)
        cad_raw = [0.0] * N
        for i in range(N):
            n = numbers[i]
            if n in last_seen:
                g0 = draw_index - last_seen[n]
            else:
                g0 = draw_index + 1
            if g0 < 0:
                g0 = 0
            ages = ages_by_n[n]
            if len(ages) >= 2:
                gaps = [ages[t + 1] - ages[t] for t in range(len(ages) - 1)]
                tail = gaps[-k_tail:]
                mu = sum(tail) / float(len(tail))
            else:
                mu = prior_gap
            x = math.log((g0 + 1.0) / (mu + 1.0))
            cad_raw[i] = math.exp(-0.5 * (x / _CAD_SIG) ** 2)

        z_pair = _zscore(pair_raw)
        z_dev = _zscore(dev_raw)
        z_pos = _zscore(pos_raw)
        z_cad = _zscore(cad_raw)

        scores = [
            w_pair * z_pair[i] + w_dev * z_dev[i] + w_pos * z_pos[i] + w_cad * z_cad[i]
            for i in range(N)
        ]
        return sorted(numbers, key=lambda n: (-scores[n - 1], n))

    return rank