import math

NAME = "zechomar"
DESCRIPTION = 'Five standardized signals blended per number: decaying short-horizon echo of recent appearances, overdue gap pressure, EWMA momentum vs the long-run rate, all-time frequency z-score, and shrinkage-regularized Markov affinity propagated from the last draw over the windowed co-occurrence graph, plus an optional order-2 message pass. Components are z-scored across 1..N each call and combined with fixed, individually ablatable weights.'
SOURCE = 'model + 2026-09-08'


def create(config):
    N = config.number_range
    K = config.numbers_drawn

    # Fixed hyper-parameters (ablation / grid-scan targets)
    W_ECHO = 0.50      # ablation base: only echo; scan 0.2-1.0
    W_DUE = 0.20       # sign scan +/- 0.25
    W_MOM = 0.60       # + EWMA momentum
    W_ZLT = 0.80       # + long-run z
    W_AFF = 0.40       # + Markov affinity; scan 0-0.8
    W_MSG2 = 0.30      # order-2 message pass; scan 0-1.5
    K_SHRINK = 3       # affinity shrinkage k in {1, 3, 5, 8}
    H_ECHO = 5         # echo horizon (draws)
    ECHO_DECAY = 0.55
    EWMA_A = 0.10      # EWMA alpha
    EWMA_LEN = 50      # EWMA truncation (draws)
    H_COOC = 90        # co-occurrence window (draws)
    BRIDGE_TOP = 20    # bridge nodes for the order-2 pass
    P = K / float(N)

    def _zscore(vals):
        n = len(vals)
        m = 0.0
        for v in vals:
            m += v
        m /= n
        var = 0.0
        for v in vals:
            d = v - m
            var += d * d
        sd = math.sqrt(var / n) if n > 0 else 0.0
        if sd < 1e-12:
            return [0.0] * n
        inv = 1.0 / sd
        return [(v - m) * inv for v in vals]

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        t = float(draw_index) if draw_index > 0 else 1.0
        exp_gap = 1.0 / P
        cap = 4.0 * exp_gap
        window = list(recent_deque)
        nw = len(window)

        # --- echo + due (ordering-free: derived from last_seen) ---
        echo = [0.0] * N
        due = [0.0] * N
        last_set = set()
        for i in range(N):
            ls = last_seen.get(i + 1)
            if ls is None:
                gap = draw_index + 1
            else:
                gap = draw_index - ls
                if gap <= 1:
                    last_set.add(i + 1)
            ge = gap if gap > 1 else 1
            if ls is not None and 1 <= ge <= H_ECHO:
                echo[i] = ECHO_DECAY ** (ge - 1)
            g = gap if gap < cap else cap
            due[i] = g / exp_gap - 1.0

        # --- momentum: EWMA of appearance indicators vs long-run rate ---
        mom = [0.0] * N
        wsum = 0.0
        dec = 1.0 - EWMA_A
        for j in range(min(nw, EWMA_LEN)):
            s = window[nw - 1 - j]
            w = dec ** j
            wsum += w
            for x in s:
                if 1 <= x <= N:
                    mom[x - 1] += w
        exp_mom = P * wsum
        if exp_mom > 1e-12:
            inv = 1.0 / exp_mom
            mom = [v * inv - 1.0 for v in mom]

        # --- long-run z with +/-3 cap ---
        exp_cnt = t * P
        sd_cnt = math.sqrt(t * P * (1.0 - P))
        if sd_cnt < 1e-12:
            sd_cnt = 1.0
        zlt = [0.0] * N
        for i in range(N):
            z = (counts.get(i + 1, 0) - exp_cnt) / sd_cnt
            if z > 3.0:
                z = 3.0
            elif z < -3.0:
                z = -3.0
            zlt[i] = z

        # --- Markov affinity (pass 1) + order-2 message pass (pass 2) ---
        aff = [0.0] * N
        aff2 = [0.0] * N
        if (W_AFF != 0.0 or W_MSG2 != 0.0) and nw > 0 and last_set:
            cooc = {}
            cnt_w = {}
            for j in range(min(nw, H_COOC)):
                s = window[nw - 1 - j]
                for m in s:
                    if m < 1 or m > N:
                        continue
                    cnt_w[m] = cnt_w.get(m, 0) + 1
                    row = cooc.get(m)
                    if row is None:
                        row = cooc[m] = {}
                    for u in s:
                        if u != m and 1 <= u <= N:
                            row[u] = row.get(u, 0) + 1
            for m in last_set:
                denom = cnt_w.get(m, 0) + K_SHRINK
                row = cooc.get(m)
                if not row:
                    continue
                idm = 1.0 / denom
                for u, c in row.items():
                    aff[u - 1] += c * idm
            if W_MSG2 != 0.0:
                order = sorted(range(N), key=lambda i: -aff[i])
                mmax = aff[order[0]]
                if mmax > 1e-12:
                    for b in order[:BRIDGE_TOP]:
                        w = aff[b] / mmax
                        if w < 0.05:
                            break
                        denom = cnt_w.get(b + 1, 0) + K_SHRINK
                        row = cooc.get(b + 1)
                        if not row:
                            continue
                        idm = w / denom
                        for u, c in row.items():
                            aff2[u - 1] += c * idm

        # --- z-score each component, blend with fixed weights ---
        zs = (_zscore(echo), _zscore(due), _zscore(mom), _zscore(zlt), _zscore(aff), _zscore(aff2))
        ws = (W_ECHO, W_DUE, W_MOM, W_ZLT, W_AFF, W_MSG2)
        scores = [0.0] * N
        for i in range(N):
            s = 0.0
            for k in range(6):
                w = ws[k]
                if w != 0.0:
                    s += w * zs[k][i]
            scores[i] = s

        return sorted(numbers, key=lambda i: (-scores[i - 1], i))

    return rank