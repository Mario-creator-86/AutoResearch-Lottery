NAME = "zhzfusio"
DESCRIPTION = 'Fuziune neliniara de cinci semnale z-score: EWMA pe frecvente recente (half-life 34), hazard de intarziere normalizat geometric, cicluri de intarziere pe 8 sectoare, tractiune de perechi co-aparute din fereastra recenta si deviatie contrarian pe termen lung; semnalul EWMA e modulat de entropia distributiei ponderate, orientarea deque-ului e detectata automat din last_seen.'
SOURCE = 'AI strategist, 2026-09-24'

import math
from math import comb

H_HALF_LIFE = 34.0
W_HOT = 1.15
W_GAP = 0.85
W_SECTOR = 0.55
W_PAIR = 0.80
W_DEV = -0.45
PAIR_LOOKBACK = 110
N_SECTORS = 8
SECTOR_SCAN_CAP = 90


def _zscore(vals):
    n = len(vals)
    if n == 0:
        return list(vals)
    m = sum(vals) / float(n)
    var = sum((v - m) * (v - m) for v in vals) / float(n)
    if var <= 1e-12:
        return [0.0] * n
    sd = math.sqrt(var)
    return [(v - m) / sd for v in vals]


def create(config):
    total = config.number_range
    drawn = config.numbers_drawn
    sector_size = max(1, total // N_SECTORS)
    mean_gap = total / float(drawn)
    log_mean_gap = math.log1p(mean_gap)
    p_sec = 1.0 - comb(total - sector_size, drawn) / float(comb(total, drawn))
    exp_sector_gap = 1.0 / p_sec if p_sec > 0 else 3.0
    log_esg = math.log1p(exp_sector_gap)
    decay = 0.5 ** (1.0 / H_HALF_LIFE)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, total + 1))
        draws = list(recent_deque)
        L = len(draws)
        big = 10 ** 9

        def min_gap(s):
            g = big
            for x in s:
                ls = last_seen.get(x)
                if ls is not None:
                    d = draw_index - ls
                    if d < g:
                        g = d
            return g

        newest_at_end = True
        if L >= 2:
            newest_at_end = min_gap(draws[-1]) <= min_gap(draws[0])

        hot = dict.fromkeys(numbers, 0.0)
        for i in range(L):
            age = (L - 1 - i) if newest_at_end else i
            w = decay ** age
            for x in draws[i]:
                hot[x] += w
        hot_list = [hot[n] for n in numbers]
        z_hot = _zscore(hot_list) if L else [0.0] * total
        tot_hot = sum(hot_list)
        gain = 1.0
        if tot_hot > 0.0:
            Hs = 0.0
            for v in hot_list:
                if v > 0.0:
                    p = v / tot_hot
                    Hs -= p * math.log(p)
            gain = 0.35 + 0.95 * (Hs / math.log(total))

        gap_cap = int(4 * mean_gap)
        gap_raw = []
        for n in numbers:
            ls = last_seen.get(n)
            gap = (draw_index - ls) if ls is not None else int(3 * mean_gap)
            if gap < 0:
                gap = 0
            elif gap > gap_cap:
                gap = gap_cap
            gap_raw.append(math.log1p(gap / mean_gap) / log_mean_gap)
        z_gap = _zscore(gap_raw)

        sec_gap = [None] * N_SECTORS
        found = 0
        order = range(L - 1, -1, -1) if newest_at_end else range(L)
        for i in order:
            age = (L - 1 - i) if newest_at_end else i
            if age > SECTOR_SCAN_CAP:
                break
            for x in draws[i]:
                si = (x - 1) // sector_size
                if si >= N_SECTORS:
                    si = N_SECTORS - 1
                if sec_gap[si] is None:
                    sec_gap[si] = age
                    found += 1
            if found >= N_SECTORS:
                break
        sec_raw = []
        never_gap = int(3 * exp_sector_gap)
        for n in numbers:
            si = (n - 1) // sector_size
            if si >= N_SECTORS:
                si = N_SECTORS - 1
            g = sec_gap[si]
            if g is None:
                g = never_gap
            if g > 12:
                g = 12
            sec_raw.append(math.log1p(g / exp_sector_gap) / log_esg)
        z_sec = _zscore(sec_raw)

        pair_pull = [0.0] * total
        if L >= 2:
            if newest_at_end:
                anchor = sorted(draws[-1])
                hist = draws[:-1]
                window = hist[-PAIR_LOOKBACK:] if len(hist) > PAIR_LOOKBACK else hist
            else:
                anchor = sorted(draws[0])
                hist = draws[1:]
                window = hist[:PAIR_LOOKBACK]
            pcount = {}
            for s in window:
                lst = sorted(s)
                ln = len(lst)
                for ai in range(ln):
                    base = lst[ai] * 64
                    for bi in range(ai + 1, ln):
                        k = base + lst[bi]
                        pcount[k] = pcount.get(k, 0) + 1
            for idx in range(total):
                n = numbers[idx]
                t = 0
                for m in anchor:
                    if m == n:
                        continue
                    k = (n * 64 + m) if n < m else (m * 64 + n)
                    c = pcount.get(k, 0)
                    if c:
                        t += c
                pair_pull[idx] = float(t)
        z_pair = _zscore(pair_pull)

        cs = [counts.get(n, 0) for n in numbers]
        m_all = sum(cs) / float(total)
        var_all = sum((c - m_all) * (c - m_all) for c in cs) / float(total)
        sd_all = math.sqrt(var_all) if var_all > 1e-12 else 1.0
        z_dev = [(c - m_all) / sd_all for c in cs]

        w_hot = W_HOT * gain
        scores = [0.0] * total
        for i in range(total):
            scores[i] = (w_hot * z_hot[i] + W_GAP * z_gap[i]
                         + W_SECTOR * z_sec[i] + W_PAIR * z_pair[i]
                         + W_DEV * z_dev[i])
        idx_order = sorted(range(total), key=lambda i: (-scores[i], numbers[i]))
        result = [numbers[i] for i in idx_order]

        rank._diag = {
            'newest_first': (not newest_at_end),
            'gain': round(gain, 4),
            'contrib': {
                'hot': round(sum(abs(w_hot * z) for z in z_hot) / total, 5),
                'gap': round(sum(abs(W_GAP * z) for z in z_gap) / total, 5),
                'sector': round(sum(abs(W_SECTOR * z) for z in z_sec) / total, 5),
                'pair': round(sum(abs(W_PAIR * z) for z in z_pair) / total, 5),
                'dev': round(sum(abs(W_DEV * z) for z in z_dev) / total, 5)
            }
        }
        return result

    return rank