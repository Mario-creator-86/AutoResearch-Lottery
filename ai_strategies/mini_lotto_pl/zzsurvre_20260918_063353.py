import math

NAME = "zzsurvre"
DESCRIPTION = (
    'Strategie de mean-reversion multi-orizont pentru 5/42: (1) hazard de gap individual '
    'H = 1-(1-p)^g - numerele intarziate primesc un scor de urgenta saturant derivat din '
    'supravietuirea geometrica a gap-ului, cu cap la 3x gap-ul asteptat; (2) deficit de sector - '
    'suma gap-urilor numerelor din cele 6 sectoare contigue comparata cu media asteptata '
    'corectata de cap; sectoarele intarziate (gap total peste medie) sunt boost-uite cu un '
    'raport clampat in [0,1]; (3) reversia deviatiei de frecventa pe termen lung (z clampat, '
    'semn inversat - numerele istoric reci sunt favorizate). Nu parcurge fereastra recenta: '
    'last_seen este statistica suficienta pentru recenta, deci rank ruleaza in O(N) si suporta '
    'mii de backteste in timp rezonabil.'
)
SOURCE = 'glm-5.3-flash:cloud | 2026-09-18 | gap-survival hazard + sector gap deficit + long-run reversion'

W_H = 1.0           # hazard individual (mecanism principal)
W_S = 0.60          # deficit de sector, raport clampat [0,1]
W_L = 0.15          # reversie long-run
Z_CAP = 2.5         # clamp pentru z-score
SECTORS = 6         # numar de sectoare contigue
GAP_CAP_MULT = 3.0  # cap pe gap: 3 x gap asteptat (1/p)


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    try:
        TS = int(config.ticket_size)
    except Exception:
        TS = K
    if TS <= 0:
        TS = K

    p = K / float(N)
    q = 1.0 - p
    log_q = math.log(q)
    gap_cap = GAP_CAP_MULT / p
    gap_cap_i = int(gap_cap)
    # E[min(g, gap_cap)] pentru geometrica P(g=k) = p * q^(k-1), k >= 1
    e_min = 0.0
    for k in range(1, gap_cap_i + 1):
        e_min += k * p * (q ** (k - 1))
    e_min += gap_cap * (q ** gap_cap_i)

    n_sec = SECTORS
    base = N // n_sec
    rem = N - base * n_sec
    sec_of = [0] * N
    sec_size = [0] * n_sec
    idx = 0
    for k in range(n_sec):
        sz = base + (1 if k < rem else 0)
        sec_size[k] = sz
        for _ in range(sz):
            sec_of[idx] = k
            idx += 1
    mu_sec = [sz * e_min for sz in sec_size]

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        exp_cnt = (draw_index + 1) * p
        var_lr = (draw_index + 1) * p * q
        sd_lr = math.sqrt(var_lr) if var_lr > 0.0 else 1.0

        h_vals = [0.0] * N
        gaps = [0.0] * N
        sec_gap = [0.0] * n_sec
        di = float(draw_index)
        for i in range(N):
            ls = last_seen.get(i + 1)
            if ls is None:
                g = di + 1.0
            else:
                g = di - float(ls)
            if g < 0.0:
                g = 0.0
            elif g > gap_cap:
                g = gap_cap
            gaps[i] = g
            h_vals[i] = 1.0 - math.exp(log_q * g)
            sec_gap[sec_of[i]] += g

        sec_def = [0.0] * n_sec
        for k in range(n_sec):
            if mu_sec[k] <= 0.0:
                continue
            d = (sec_gap[k] - mu_sec[k]) / mu_sec[k]
            if d < 0.0:
                d = 0.0
            elif d > 1.0:
                d = 1.0
            sec_def[k] = d

        scores = [0.0] * N
        h_sum = 0.0
        h_min = 1e18
        h_max = -1e18
        s_sum = 0.0
        s_max = 0.0
        l_sum = 0.0
        l_min = 1e18
        l_max = -1e18
        g_sum = 0.0
        g_max = 0.0
        for i in range(N):
            H = h_vals[i]
            S = sec_def[sec_of[i]]
            zc = (counts.get(i + 1, 0) - exp_cnt) / sd_lr
            if zc > Z_CAP:
                zc = Z_CAP
            elif zc < -Z_CAP:
                zc = -Z_CAP
            L = -zc
            scores[i] = W_H * H + W_S * S + W_L * L
            h_sum += H
            if H < h_min:
                h_min = H
            if H > h_max:
                h_max = H
            s_sum += S
            if S > s_max:
                s_max = S
            l_sum += L
            if L < l_min:
                l_min = L
            if L > l_max:
                l_max = L
            g = gaps[i]
            g_sum += g
            if g > g_max:
                g_max = g

        order = sorted(range(1, N + 1), key=lambda n: (-scores[n - 1], n))

        inv_n = 1.0 / float(N)
        top_sc = sorted(scores, reverse=True)[:TS]
        top_avg = sum(top_sc) / float(len(top_sc)) if top_sc else 0.0
        rank._diag = {
            'draw_index': draw_index,
            'H_avg': h_sum * inv_n, 'H_min': h_min, 'H_max': h_max,
            'S_avg': s_sum * inv_n, 'S_max': s_max,
            'L_avg': l_sum * inv_n, 'L_min': l_min, 'L_max': l_max,
            'gap_avg': g_sum * inv_n, 'gap_max': g_max,
            'topTS_score_avg': top_avg,
            'sec_def': list(sec_def)
        }
        c = rank._diag_cum
        c['n'] = c.get('n', 0) + 1
        c['H_sum'] = c.get('H_sum', 0.0) + h_sum * inv_n
        c['S_sum'] = c.get('S_sum', 0.0) + s_sum * inv_n
        c['L_sum'] = c.get('L_sum', 0.0) + l_sum * inv_n
        c['gap_sum'] = c.get('gap_sum', 0.0) + g_sum * inv_n
        c['topTS_sum'] = c.get('topTS_sum', 0.0) + top_avg
        return order

    rank._diag = {}
    rank._diag_cum = {}
    return rank