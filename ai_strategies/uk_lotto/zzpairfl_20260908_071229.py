NAME = "zzpairfl"
DESCRIPTION = 'Co-occurrence affinity flow: each number is scored by the exponentially-weighted recency of the numbers it co-appeared with in the recent window (one-step association flow on the pair graph), plus a saturating self-recency term; no frequency, gap or sector priors.'
SOURCE = 'pair-affinity flow model + 2026-09-08'

import math


def create(config):
    N = int(config.number_range)
    KD = int(config.numbers_drawn)
    HL = 9.0        # half-life (draws) of association memory
    AF_W = 1.0      # affinity-flow weight
    RC_W = 0.45     # self-recency weight

    def _mean(v):
        return sum(v) / len(v) if v else 0.0

    def _std(v):
        if len(v) < 2:
            return 0.0
        mu = _mean(v)
        return math.sqrt(sum((x - mu) * (x - mu) for x in v) / len(v))

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        if not last_seen or not recent_deque:
            return sorted(numbers, key=lambda n: (-counts.get(n, 0), n))

        W = list(recent_deque)
        L = len(W)
        if L == 0:
            return sorted(numbers, key=lambda n: (-counts.get(n, 0), n))

        m = max(last_seen.get(n, -1) for n in numbers)
        if m < 0:
            return sorted(numbers, key=lambda n: (-counts.get(n, 0), n))

        # infer deque orientation deterministically: newest end must equal last drawn set
        last_set = set(n for n in numbers if last_seen.get(n, -1) == m)
        newest_last = True
        if set(W[-1]) == last_set:
            newest_last = True
        elif set(W[0]) == last_set:
            newest_last = False

        # exponential memory weights by age within the window
        wts = [0.0] * L
        for j in range(L):
            r = (L - 1 - j) if newest_last else j
            wts[j] = 0.5 ** (r / HL)

        # EWMA recency per number
        rec = dict((n, 0.0) for n in numbers)
        for j in range(L):
            wj = wts[j]
            if wj <= 0.0:
                continue
            for n in W[j]:
                if 1 <= n <= N:
                    rec[n] += wj

        # affinity flow: A(n) = sum over draws containing n of wj * mean recency of co-members
        aff = dict((n, 0.0) for n in numbers)
        for j in range(L):
            wj = wts[j]
            if wj <= 0.0:
                continue
            row = [n for n in W[j] if 1 <= n <= N]
            k = len(row)
            if k < 2:
                continue
            s = 0.0
            for b in row:
                s += rec[b]
            for n in row:
                aff[n] += wj * (s - rec[n]) / (k - 1)

        # robust standardization + tanh saturation
        mu_r = _mean([rec[n] for n in numbers])
        sd_r = _std([rec[n] for n in numbers])
        mu_a = _mean([aff[n] for n in numbers])
        sd_a = _std([aff[n] for n in numbers])
        sdr = sd_r if sd_r > 1e-9 else 1.0
        sda = sd_a if sd_a > 1e-9 else 1.0

        scores = {}
        for n in numbers:
            zr = (rec[n] - mu_r) / sdr
            za = (aff[n] - mu_a) / sda
            scores[n] = AF_W * math.tanh(za) + RC_W * math.tanh(zr)

        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank