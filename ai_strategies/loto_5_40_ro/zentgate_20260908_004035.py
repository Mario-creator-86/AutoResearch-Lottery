import math
from collections import defaultdict

NAME = "zentgate"
DESCRIPTION = ('Regime-gated dual-horizon scoring: fast (HL=8) and slow (HL=30) EWMA appearance '
               'z-scores compressed by tanh form the momentum core, whose weight is modulated by a '
               'chi-square concentration gate on the recent window; contrarian full-history z-reversion, '
               'Poisson-excess pair affinity and saturating per-number/block gap bumps complete the score.')
SOURCE = 'model + 2026-09-07'


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    p = (K / float(N)) if N > 0 else 0.0
    q = (K * (K - 1) / float(N * (N - 1))) if N > 1 else 0.0
    p_ok = 1e-9 < p < 1.0 - 1e-9
    exp_num_gap = (float(N) / K) if K > 0 else float(N if N > 0 else 1)
    B = max(1, N // 10)
    miss = 1.0
    for i in range(K):
        num = N - B - i
        if num <= 0:
            miss = 0.0
            break
        miss *= num / float(N - i)
    p_block = 1.0 - miss if 0.0 < miss < 1.0 else (0.0 if miss <= 0.0 else 1.0)
    exp_block_gap = (1.0 / p_block) if p_block > 1e-9 else float(N if N > 0 else 1)
    due_cap = 2.5 * exp_num_gap
    bg_cap = 3.0 * exp_block_gap
    WF = [0.5 ** (d / 8.0) for d in range(120)]
    WS = [0.5 ** (d / 30.0) for d in range(120)]

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        if not numbers:
            return []
        counts = counts if counts else {}
        last_seen = last_seen if last_seen else {}
        total_counts = 0
        for n in numbers:
            total_counts += counts.get(n, 0)
        n_draws = (total_counts / float(K)) if K > 0 else 0.0
        expected = n_draws * p
        sigma_t = math.sqrt(max(expected * (1.0 - p), 0.25))

        wlen = 0
        if recent_deque:
            try:
                wlen = min(int(recent_window), len(recent_deque))
            except (TypeError, ValueError):
                wlen = len(recent_deque)
        newest_first = []
        if wlen > 0:
            by_gap = sorted(numbers, key=lambda n: (draw_index - last_seen.get(n, -1), n))
            gset = set(by_gap[:K]) if K > 0 else set()
            best_i = 0
            best_ov = -1
            for i in range(wlen):
                ov = len(recent_deque[i] & gset)
                if ov > best_ov:
                    best_ov = ov
                    best_i = i
            if best_i == wlen - 1:
                newest_first = [recent_deque[i] for i in range(wlen - 1, -1, -1)]
            else:
                newest_first = [recent_deque[i] for i in range(wlen)]

        depth_e = min(wlen, 120)
        wc = defaultdict(int)
        for d in range(depth_e):
            for n in newest_first[d]:
                wc[n] += 1
        conc = 0.0
        if depth_e > 0 and p_ok:
            e_c = depth_e * p
            sd_w = math.sqrt(depth_e * p * (1.0 - p))
            s2 = 0.0
            for n in numbers:
                z = (wc.get(n, 0) - e_c) / sd_w
                s2 += z * z
            chi2n = s2 / float(N)
            if chi2n >= 1.35:
                conc = 1.0
            elif chi2n <= 0.65:
                conc = -1.0
            else:
                conc = (chi2n - 1.0) / 0.35
        m_mult = 1.0 + 0.5 * conc
        r_mult = 1.0 - 0.5 * conc
        pair_mult = 1.0 + 0.4 * conc
        gap_mult = 1.0 - 0.4 * conc

        acc_f = defaultdict(float)
        acc_s = defaultdict(float)
        sw_f = 0.0
        sw_s = 0.0
        for d in range(depth_e):
            wf = WF[d]
            ws = WS[d]
            sw_f += wf
            sw_s += ws
            for n in newest_first[d]:
                acc_f[n] += wf
                acc_s[n] += ws

        depth_p = min(wlen, 60)
        cooc = defaultdict(int)
        for d in range(depth_p):
            sl = sorted(newest_first[d])
            for i in range(len(sl) - 1):
                a = sl[i]
                for j in range(i + 1, len(sl)):
                    cooc[(a, sl[j])] += 1
        exp_pair = depth_p * q
        pa = {}
        for n in numbers:
            v = 0.0
            for m in numbers:
                if m == n:
                    continue
                key = (n, m) if n < m else (m, n)
                c = cooc.get(key, 0)
                if c > exp_pair:
                    v += c - exp_pair
            pa[n] = v
        pmax = max(pa.values()) if pa else 0.0
        if pmax > 1e-9:
            inv = 1.0 / pmax
            for n in numbers:
                pa[n] *= inv
        else:
            for n in numbers:
                pa[n] = 0.0

        block_last = {}
        for n in numbers:
            b = (n - 1) // B
            ls = last_seen.get(n, -1)
            if ls > block_last.get(b, -1):
                block_last[b] = ls

        scores = {}
        for n in numbers:
            if p_ok and sw_f > 1e-9:
                zf = (acc_f.get(n, 0.0) / sw_f - p) / math.sqrt(p * (1.0 - p) / sw_f)
            else:
                zf = 0.0
            if p_ok and sw_s > 1e-9:
                zs = (acc_s.get(n, 0.0) / sw_s - p) / math.sqrt(p * (1.0 - p) / sw_s)
            else:
                zs = 0.0
            zt = (counts.get(n, 0) - expected) / sigma_t
            gap = draw_index - last_seen.get(n, -1)
            due = gap / due_cap if due_cap > 0 else 0.0
            if due > 1.0:
                due = 1.0
            b = (n - 1) // B
            bg = (draw_index - block_last.get(b, -1)) / bg_cap if bg_cap > 0 else 0.0
            if bg > 1.0:
                bg = 1.0
            s = (m_mult * (math.tanh(zf) + 0.55 * math.tanh(zs))
                 - r_mult * 0.45 * math.tanh(zt)
                 + pair_mult * 0.55 * pa[n]
                 + gap_mult * (0.25 * bg + 0.30 * due))
            scores[n] = s
        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank