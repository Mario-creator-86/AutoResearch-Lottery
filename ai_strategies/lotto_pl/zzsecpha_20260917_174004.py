NAME = "zzsecpha"
DESCRIPTION = 'Faza de ciclu pe sectoare: deficit de acoperire in fereastra scurta + gap normalizat fata de ultima aparitie a sectorului, combinat cu reversiune z-score macro pe frecventele totale si ecou de perechi ponderat de similaritatea fiecarei extrageri cu extragerea anterioara.'
SOURCE = 'model + 2026-09-16'

import math
from collections import defaultdict

SECTOR_SIZE = 7      # sector size (1..49 -> 7 sectors); also read from config.sector_size if present
COV_WIN = 36         # short window (draws) for sector coverage deficit
W_SEC = 0.70         # weight: sector coverage deficit
W_GAP = 0.50         # weight: normalized sector gap
W_DEV = 0.40         # weight: macro z-score deviation
DEV_SIGN = -1.0      # -1 = reversion (favor cold numbers); +1 = momentum
W_ECHO = 0.60        # weight: pair echo with previous draw
GAP_CAP = 5.0        # cap for gap_norm in units of expected gap

DIAG = {}


def _no_hit_prob(n, k, d):
    # P(no number from a k-sized sector appears in d draws) = C(n-k, d) / C(n, d)
    p = 1.0
    for i in range(d):
        p *= (n - k - i) / float(n - i)
    return p


def create(config):
    N = config.number_range
    D = config.numbers_drawn
    K = getattr(config, 'sector_size', SECTOR_SIZE)
    try:
        K = int(K)
    except Exception:
        K = SECTOR_SIZE
    if not K or K < 1:
        K = SECTOR_SIZE
    K = min(K, N)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))

        # 1) macro deviation: mean/std over total counts
        vals = [counts.get(n, 0) for n in numbers]
        mean = sum(vals) / float(len(vals))
        var = sum((v - mean) ** 2 for v in vals) / float(len(vals))
        std = math.sqrt(var)
        if std < 1e-12:
            std = 1.0

        # 2) sectors: sizes, coverage on short window, gap since last sector hit
        n_sec = (N + K - 1) // K
        sector_of = {}
        k_of = {}
        for n in numbers:
            s = (n - 1) // K
            sector_of[n] = s
            k_of[s] = k_of.get(s, 0) + 1

        cov_list = list(recent_deque)
        if len(cov_list) > COV_WIN:
            cov_list = cov_list[-COV_WIN:]
        lc = len(cov_list)
        actual = defaultdict(int)
        for d in cov_list:
            for m in d:
                s = sector_of.get(m)
                if s is not None:
                    actual[s] += 1

        deficit = {}
        for s in range(n_sec):
            exp_s = lc * D * k_of[s] / float(N)
            deficit[s] = (exp_s - actual.get(s, 0)) / max(exp_s, 1.0)

        gap_norm = {}
        for s in range(n_sec):
            p_hit = 1.0 - _no_hit_prob(N, k_of[s], D)
            e_gap = 1.0 / p_hit if p_hit > 1e-9 else 10.0
            last_hit = -1
            for n in numbers:
                if sector_of[n] == s:
                    ls = last_seen.get(n)
                    if ls is not None and ls > last_hit:
                        last_hit = ls
            gap = float(draw_index - last_hit) if last_hit >= 0 else float(draw_index + 1)
            if gap < 0.0:
                gap = 0.0
            gap_norm[s] = min(gap, GAP_CAP * e_gap) / e_gap

        # 3) previous draw, robust to deque ordering (rebuild from last_seen)
        prev_idx = draw_index - 1
        last_set = set(n for n in numbers if last_seen.get(n) == prev_idx)
        if not last_set and len(recent_deque) > 0:
            last_set = set(recent_deque[-1])

        # 4) pair echo: co-occurrence weighted by |draw intersect last_set|
        weights = []
        total_w = 0.0
        for d in recent_deque:
            w = float(len(set(d) & last_set))
            weights.append(w)
            total_w += w
        echo = dict((n, 0.0) for n in numbers)
        if total_w > 1e-12:
            for d, w in zip(recent_deque, weights):
                if w > 0.0:
                    for m in d:
                        if m in echo:
                            echo[m] += w
            inv = 1.0 / total_w
            for n in echo:
                echo[n] *= inv

        # 5) composite score and final permutation
        scores = {}
        for n in numbers:
            s = sector_of[n]
            scores[n] = (W_SEC * deficit[s]
                         + W_GAP * gap_norm[s]
                         + DEV_SIGN * W_DEV * ((counts.get(n, 0) - mean) / std)
                         + W_ECHO * echo[n])

        dmin = min(deficit.values())
        dmax = max(deficit.values())
        gmin = min(gap_norm.values())
        gmax = max(gap_norm.values())
        emin = min(echo.values())
        emax = max(echo.values())
        DIAG.update({
            'deficit_span': dmax - dmin,
            'gap_norm_span': gmax - gmin,
            'echo_span': emax - emin,
            'echo_min': emin,
            'echo_max': emax,
            'z_min': (min(vals) - mean) / std,
            'z_max': (max(vals) - mean) / std,
            'cov_len': lc,
        })

        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank