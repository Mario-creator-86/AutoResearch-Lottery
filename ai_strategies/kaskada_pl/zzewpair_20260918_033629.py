NAME = "zzewpair"
DESCRIPTION = 'Momentum adaptiv: pair-lift lag-1 (co-aparitii ponderate cu membrii ultimei extrageri) + EWMA pe frecventele recente, amestecate non-liniar printr-o poarta de entropie Shannon cu deviatia z binomiala pe termen lung; entropie mica -> momentum scurt, entropie mare -> deviatie macro. Semne momentum, conform ledger-ului.'
SOURCE = 'model + 2026-09-18'

import math


def create(config):
    N = int(config.number_range)
    ND = int(getattr(config, 'numbers_drawn', max(1, N // 2)) or max(1, N // 2))
    P0 = (float(ND) / float(N)) if N > 0 else 0.5
    K_EW = 20      # fereastra EWMA (extrageri, inclusiv ultima)
    K_PL = 20      # fereastra pair-lift (extrageri anterioare ultimei)
    DECAY = 0.90   # decay per extragere
    W_EW = 0.35    # pondere EWMA (punct din grid)
    W_PL = 2.2     # pondere pair-lift lag-1 = W_L1 (punct din grid)
    W_DV = 0.35    # pondere deviatie macro
    GATE_K = 0.75  # panta portii de entropie (punct din grid)
    LN_N = math.log(N) if N > 1 else 1.0

    state = {'gate': 0.5, 'span_ew': 0.0, 'span_lift': 0.0,
             'span_dev': 0.0, 'draw_index': -1}

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        if counts is None:
            counts = {}
        if not recent_deque:
            return numbers

        def edge_mean_rec(s):
            tot = 0
            cnt = 0
            for n in s:
                tot += (last_seen.get(n, -1) if last_seen else -1)
                cnt += 1
            return (tot / float(cnt)) if cnt else -1.0

        # detectie orientare: capatul cu media last_seen mai mare e cel mai nou
        first_d = recent_deque[0]
        last_d = recent_deque[-1]
        right_newest = edge_mean_rec(last_d) >= edge_mean_rec(first_d)
        last_set = last_d if right_newest else first_d
        if not isinstance(last_set, (set, frozenset)):
            last_set = set(last_set)
        M = len(last_set)
        if M == 0:
            return numbers

        ew = dict.fromkeys(numbers, 0.0)
        C = dict.fromkeys(numbers, 0.0)
        pm_keys = sorted(last_set)
        PM = dict.fromkeys(pm_keys, 0.0)
        tot_ew = 0.0
        S = 0.0
        L = max(K_EW, K_PL + 1)
        w = 1.0
        i = 0
        it = reversed(recent_deque) if right_newest else iter(recent_deque)
        for d in it:
            if i >= L:
                break
            ds = d if isinstance(d, (set, frozenset)) else set(d)
            if i < K_EW:
                for n in ds:
                    ew[n] += w
                tot_ew += w
            if 1 <= i <= K_PL:
                S += w
                ov = len(ds & last_set)
                if ov:
                    wf = w * ov
                    for n in ds:
                        C[n] += wf
                for m in pm_keys:
                    if m in ds:
                        PM[m] += w
            i += 1
            w *= DECAY

        if tot_ew <= 0.0:
            return numbers

        inv_ew = 1.0 / tot_ew
        p = {}
        for n in numbers:
            p[n] = ew[n] * inv_ew

        have_pair = S > 0.0
        a_par = 0.0
        base_pair = 0.0
        if have_pair:
            invS = 1.0 / S
            invM = 1.0 / float(M)
            a_par = invS * invM
            q_sum = 0.0
            for m in pm_keys:
                q_sum += PM[m] * invS
            base_pair = q_sum * invM

        # entropia Shannon a ferestrei -> poarta non-liniara
        H = 0.0
        for n in numbers:
            pn = p[n]
            if pn > 0.0:
                H -= pn * math.log(pn)
        Hn = H / LN_N
        if Hn < 0.0:
            Hn = 0.0
        elif Hn > 1.0:
            Hn = 1.0
        gate = 0.5 + GATE_K * (Hn - 0.5)
        if gate < 0.0:
            gate = 0.0
        elif gate > 1.0:
            gate = 1.0

        # deviatie macro z binomial, cu garde de consistenta counts vs draw_index
        Deff = float(draw_index) if (draw_index is not None and draw_index > 0) else 0.0
        mx = 0
        for n in numbers:
            c = counts.get(n, 0)
            if c > mx:
                mx = c
        if mx > Deff + 1.0:
            Deff = float(mx)
        sd = math.sqrt(Deff * P0 * (1.0 - P0)) if Deff > 0.0 else 0.0
        inv_z = (1.0 / sd) if sd > 0.0 else 0.0

        scores = {}
        lo_e = hi_e = lo_l = hi_l = lo_d = hi_d = None
        for n in numbers:
            e = p[n] - P0
            lf = (C[n] * a_par - p[n] * base_pair) if have_pair else 0.0
            dv = (counts.get(n, 0) - P0 * Deff) * inv_z if inv_z > 0.0 else 0.0
            sc = gate * (W_EW * e + W_PL * lf) + (1.0 - gate) * W_DV * dv
            scores[n] = sc
            if lo_e is None:
                lo_e = hi_e = e
                lo_l = hi_l = lf
                lo_d = hi_d = dv
            else:
                if e < lo_e:
                    lo_e = e
                elif e > hi_e:
                    hi_e = e
                if lf < lo_l:
                    lo_l = lf
                elif lf > hi_l:
                    hi_l = lf
                if dv < lo_d:
                    lo_d = dv
                elif dv > hi_d:
                    hi_d = dv

        state['gate'] = gate
        state['span_ew'] = hi_e - lo_e
        state['span_lift'] = hi_l - lo_l
        state['span_dev'] = hi_d - lo_d
        state['draw_index'] = draw_index

        return sorted(numbers, key=lambda nn: (-scores[nn], nn))

    rank._diag = lambda: dict(state)
    return rank