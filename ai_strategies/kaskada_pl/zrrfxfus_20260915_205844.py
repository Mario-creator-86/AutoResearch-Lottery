NAME = "zrrfxfus"
DESCRIPTION = 'Fuziune non-liniara Reciprocal Rank Fusion (K=60) a 4 semnale ortogonale: (1) multiplicitatea aparitiilor in fereastra recenta atenuata exponential de gap (rc*GAMMA^g, interactiune ne-liniara), (2) overdue/ciclu de intarziere fata de media P/W, (3) deviatia globala de frecventa, (4) deficitul de diversitate pe 4 sectoare. Agregarea pe ranguri (nu pe scoruri) este robusta la scale heterogene si complet determinista.'
SOURCE = 'glm-5.3-flash quant plugin, conform Decision Ledger (RRF_K=60, GAMMA=0.96, grid W_*), 2026-09-15'

from itertools import chain
from collections import Counter

_RRF_K = 60.0
_GAMMA = 0.96
_GAP_CAP = 60
_W_HOT = 1.2
_W_GAP = 0.8
_W_FREQ = 0.3
_W_DIV = 0.25
_N_SEC = 4


def create(config):
    # config: LotteryConfig — foloseste doar atributele; NU itera/config dict
    P = config.number_range
    sec_size = (P + _N_SEC - 1) // _N_SEC
    numbers = list(range(1, P + 1))
    sec_of = {n: (n - 1) // sec_size for n in numbers}

    def _ranks(vals):
        # rang 1 = cel mai bun; tie-break determinist pe indice
        order = sorted(range(len(vals)), key=lambda i: (-vals[i], i))
        r = [0] * len(vals)
        for pos, i in enumerate(order):
            r[i] = pos + 1
        return r

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # 1) counts in fereastra recenta — order-agnostic (orient_flips-proof):
        #    deque-ul e consumat doar prin chain.from_iterable, ordinea e irelevanta
        if recent_deque:
            rc = Counter(chain.from_iterable(recent_deque))
        else:
            rc = {}

        hot = [0.0] * P
        gap = [0] * P
        for i, n in enumerate(numbers):
            ls = last_seen.get(n)
            if ls is None:
                g = draw_index  # niciodata aparut = overdue maxim
            else:
                g = draw_index - ls - 1
                if g < 0:
                    g = 0
            if g > _GAP_CAP:
                g = _GAP_CAP
            gap[i] = g
            # interactiune ne-liniara: multiplicitate recenta x decay exponenential
            # al recenței ultimei aparitii (GAMMA=0.96 fix)
            hot[i] = rc.get(n, 0) * (_GAMMA ** g)

        r_hot = _ranks(hot)
        r_gap = _ranks(gap)
        r_freq = _ranks([counts.get(n, 0) for n in numbers])

        # 4) deficit de sector: share asteptat 0.25 per sector; sub-reprezentate primul
        sec_tot = [0] * _N_SEC
        for n in numbers:
            sec_tot[sec_of[n]] += rc.get(n, 0)
        tot = float(sum(sec_tot)) or 1.0
        deficit = [0.25 - sec_tot[sec_of[n]] / tot for n in numbers]
        r_div = _ranks(deficit)

        weights = (_W_HOT, _W_GAP, _W_FREQ, _W_DIV)
        ranks = (r_hot, r_gap, r_freq, r_div)
        score = [0.0] * P
        for i in range(P):
            s = 0.0
            for w_, r_ in zip(weights, ranks):
                s += w_ / (_RRF_K + r_[i])
            score[i] = s

        # permutare completă 1..24, deterministă
        return sorted(numbers, key=lambda n: (-score[n - 1], n))

    return rank