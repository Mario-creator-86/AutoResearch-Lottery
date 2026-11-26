import math

NAME = "zphaseec"
DESCRIPTION = 'Oscilator de faza pe ciclurile individuale de aparitie, combinat cu ecou de perechi co-aparute cu ultima extragere, EWMA exponential pe frecventa recenta, uniformitate entropica pe blocuri si deviatie seculare fata de frecventa asteptata.'
SOURCE = 'qresearch phase-oscillator + pair-echo model, 2026-09-08'

DIAG = {'orient_flips': 0, 'orient_newest_last': 0, 'calls': 0}


def create(config):
    P = int(config.number_range)
    D = int(config.numbers_drawn)
    p = float(D) / float(P)

    GAMMA = 0.97      # decay EWMA
    KPAIR = 60        # fereastra pt perechi (ecou)
    BLOCKS = 4        # blocuri pt entropia de uniformitate
    W_EWMA = 1.00
    W_OSC = 0.35
    W_ECHO = 0.50
    W_PERS = 0.50
    W_LONG = 0.15

    _cache = {}
    lnB = math.log(BLOCKS)

    def _newest_is_last(seq, last_seen, draw_index):
        # Infera capatul cel mai nou din deque: extragerea noua are recenta medie ~1
        if len(seq) < 2:
            return True
        def mean_rec(s):
            tot = 0.0
            cnt = 0
            for x in s:
                ls = last_seen.get(x)
                if ls is not None:
                    tot += (draw_index - ls)
                    cnt += 1
            return (tot / cnt) if cnt else None
        a = mean_rec(seq[0])
        b = mean_rec(seq[-1])
        if a is None or b is None:
            return True
        if b < a - 1e-9:
            return True
        if a < b - 1e-9:
            return False
        return True

    def _compute(counts, chrono, last_seen, draw_index):
        L = len(chrono)
        N = list(range(1, P + 1))
        ewma = {n: 0.0 for n in N}
        pos = {n: [] for n in N}
        blk = {n: [0] * BLOCKS for n in N}
        for i in range(L):
            s = chrono[i]
            wt = GAMMA ** (L - 1 - i)
            b = (i * BLOCKS) // L
            for x in s:
                ewma[x] += wt
                pos[x].append(i)
                blk[x][b] += 1

        vals = [ewma[n] for n in N]
        m = sum(vals) / float(P)
        var = sum((v - m) * (v - m) for v in vals) / float(P)
        sd = math.sqrt(var) if var > 0 else 1.0
        ewma_z = {n: (ewma[n] - m) / sd for n in N}

        last = chrono[-1] if L else frozenset()
        sub = chrono[max(0, L - KPAIR):]
        Lsub = len(sub)
        pc = {}
        for s in sub:
            arr = sorted(s)
            for ii in range(len(arr)):
                for jj in range(ii + 1, len(arr)):
                    k = (arr[ii], arr[jj])
                    pc[k] = pc.get(k, 0) + 1
        E_sum = float(D - 1) * Lsub * (D * (D - 1)) / float(P * (P - 1)) if (Lsub > 0 and P > 1) else 0.0
        sd_e = math.sqrt(E_sum) if E_sum > 0 else 1.0
        echo_z = {}
        for n in N:
            tot = 0
            for mm in last:
                if mm == n:
                    continue
                k = (n, mm) if n < mm else (mm, n)
                tot += pc.get(k, 0)
            echo_z[n] = (tot - E_sum) / sd_e

        total_draws = draw_index + 1
        mu = total_draws * p
        var_l = mu * (1.0 - p) * (P - D) / float(P - 1) if P > 1 else 1.0
        sd_l = math.sqrt(var_l) if var_l > 0 else 1.0
        two_pi = 2.0 * math.pi
        osc = {}
        for n in N:
            ps = pos[n]
            mg = 0.0
            reg = 0.0
            if len(ps) >= 3:
                gaps = [ps[j + 1] - ps[j] for j in range(len(ps) - 1)]
                mg = sum(gaps) / float(len(gaps))
                if mg > 0:
                    vv = sum((g - mg) * (g - mg) for g in gaps) / float(len(gaps))
                    reg = 1.0 / (1.0 + math.sqrt(vv) / mg)
            if mg <= 0.0:
                ct = counts.get(n, 0)
                if ct >= 2:
                    mg = total_draws / float(ct)
                    reg = 0.25
            if mg > 0.0:
                ls = last_seen.get(n)
                gap_since = float(draw_index - ls) if ls is not None else float(total_draws)
                osc[n] = reg * math.cos(two_pi * gap_since / mg)
            else:
                osc[n] = 0.0

        pers = {}
        for n in N:
            C = sum(blk[n])
            if C > 0:
                H = 0.0
                for bi in range(BLOCKS):
                    cb = blk[n][bi]
                    if cb > 0:
                        fr = cb / float(C)
                        H -= fr * math.log(fr)
                pers[n] = (H / lnB) * math.log(1.0 + C)
            else:
                pers[n] = 0.0

        score = {}
        for n in N:
            zl = (counts.get(n, 0) - mu) / sd_l
            score[n] = (W_EWMA * ewma_z[n] + W_OSC * osc[n]
                        + W_ECHO * echo_z[n] + W_PERS * pers[n]
                        + W_LONG * zl)
        return score

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        DIAG['calls'] += 1
        seq = list(recent_deque)
        newest_last = _newest_is_last(seq, last_seen, draw_index)
        if newest_last:
            DIAG['orient_newest_last'] += 1
            chrono = seq
        else:
            DIAG['orient_flips'] += 1
            chrono = seq[::-1]
        L = len(chrono)
        if L:
            key = (draw_index, L, recent_window, tuple(sorted(chrono[0])), tuple(sorted(chrono[-1])))
        else:
            key = (draw_index, 0, recent_window, (), ())
        sc = _cache.get(key)
        if sc is None:
            sc = _compute(counts, chrono, last_seen, draw_index)
            _cache[key] = sc
        return sorted(range(1, P + 1), key=lambda n: (-sc[n], n))

    return rank