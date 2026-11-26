import math

NAME = "zpairbor"
DESCRIPTION = 'Fuziune Borda ponderata pe 4 semnale noi: difuziune Markov pe graful perechilor co-aparute din fereastra recenta, deficit Poisson fata de frecventa asteptata (reversie), template pozitional quantil cu kernel Gauss pe medii EWMA ale valorilor sortate si ecou de cadenta (due-ness la multiplii intervalului personal al fiecarui numar).'
SOURCE = 'model cantitativ pair-graph diffusion + Borda rank fusion, 2026-09-09'


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    if N < 1:
        N = 1
    if K < 1:
        K = 1
    if K > N:
        K = N

    # parametri (grid-search ready)
    W_PAIR = 1.20      # difuziune pe perechi co-aparute
    W_DEV = 0.60       # deficit Poisson (reversie)
    W_POS = 0.45       # template pozitional quantil
    W_CAD = 0.45       # ecou de cadenta
    GAMMA_P = 0.97     # decay pe varsta draw-urilor (newest_last)
    K_TAIL = 3         # anchor Markov: ultimele k extrageri
    SIGMA_POS = 7.0    # latimea kernelului pozitional
    BASE_GAP = float(N) / float(K)
    INV2S2 = 1.0 / (2.0 * SIGMA_POS * SIGMA_POS)
    HALF = 0.5 * (N + 1)

    cache = {}

    def _borda(vals, hot):
        mx = vals[0]
        mn = vals[0]
        for v in vals:
            if v > mx:
                mx = v
            elif v < mn:
                mn = v
        pts = [HALF] * N
        if mx - mn < 1e-12:
            return pts
        idx = sorted(range(N), key=lambda i: (-vals[i], -hot[i], i))
        r = 0
        for i in idx:
            pts[i] = float(N - r)
            r += 1
        return pts

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        L = len(recent_deque)
        if L > 0:
            sig = (draw_index, L, recent_window,
                   tuple(sorted(recent_deque[0])),
                   tuple(sorted(recent_deque[-1])))
        else:
            sig = (draw_index, L, recent_window)
        hit = cache.get(sig)
        if hit is not None:
            return list(hit)

        hot = []
        tot = 0.0
        for n in range(1, N + 1):
            v = float(counts.get(n, 0))
            hot.append(v)
            tot += v
        if tot <= 0.0:
            tot = 1.0

        R = [0] * N
        pcw = {}
        pos_s = [0.0] * K
        pos_w = [0.0] * K
        posmap = [None] * N
        decay = [0.0] * L
        if L > 0:
            # deque newest_last (DIAG['orient_flips']==0): varsta = L-1-p
            decay[L - 1] = 1.0
            for p in range(L - 2, -1, -1):
                decay[p] = decay[p + 1] * GAMMA_P
            for p in range(L):
                w = decay[p]
                s = sorted(recent_deque[p])
                ls = len(s)
                for i in range(ls):
                    a = s[i]
                    ai = a - 1
                    R[ai] += 1
                    pm = posmap[ai]
                    if pm is None:
                        pm = []
                        posmap[ai] = pm
                    pm.append(p)
                    if i < K:
                        pos_s[i] += w * a
                        pos_w[i] += w
                    for j in range(i + 1, ls):
                        b = s[j]
                        key = a * 1000 + b
                        pcw[key] = pcw.get(key, 0.0) + w

        # (1) difuziune Markov pe perechi, ancorata in ultimele K_TAIL extrageri
        mw = [0.0] * N
        tail_idx = []
        if L > 0:
            tail = K_TAIL if K_TAIL < L else L
            base = L - tail
            for t in range(tail):
                p = base + t
                w = (t + 1.0) / float(tail)
                for m in recent_deque[p]:
                    mw[m - 1] += w
            for j in range(N):
                if mw[j] > 0.0:
                    tail_idx.append(j)
        pair_v = [0.0] * N
        for i in range(N):
            a = i + 1
            acc = 0.0
            for j in tail_idx:
                if j == i:
                    continue
                b = j + 1
                if a < b:
                    key = a * 1000 + b
                else:
                    key = b * 1000 + a
                v = pcw.get(key)
                if v is not None:
                    acc += v * mw[j]
            pair_v[i] = acc

        # (2) deficit Poisson fata de frecventa asteptata in fereastra
        dev_v = [0.0] * N
        if L > 0:
            slots = float(K * L)
            for i in range(N):
                e = slots * hot[i] / tot
                dev_v[i] = (e - float(R[i])) / math.sqrt(e + 1.0)

        # (3) template pozitional quantil (kernel Gauss in jurul cuantilelor EWMA)
        mu = [0.0] * K
        for i in range(K):
            if pos_w[i] > 0.0:
                mu[i] = pos_s[i] / pos_w[i]
            else:
                mu[i] = (i + 1) * (N + 1) / float(K + 1)
        pos_v = [0.0] * N
        for i in range(N):
            x = float(i + 1)
            acc = 0.0
            for m in mu:
                d = x - m
                acc += math.exp(-d * d * INV2S2)
            pos_v[i] = acc

        # (4) ecou de cadenta: due-ness la multiplii intervalului personal
        cad_v = [0.0] * N
        di = float(draw_index)
        for i in range(N):
            pm = posmap[i]
            if pm:
                gsum = 0.0
                gn = 0
                prev = pm[0]
                for t in range(1, len(pm)):
                    dt = pm[t] - prev
                    gsum += dt
                    gn += 1
                    prev = pm[t]
                tmean = gsum / gn if gn > 0 else BASE_GAP
            else:
                tmean = BASE_GAP
            if tmean <= 0.0:
                tmean = BASE_GAP
            sg = 0.45 * tmean
            if sg < 1.0:
                sg = 1.0
            s2 = sg * sg
            lsn = last_seen.get(i + 1) if last_seen is not None else None
            if lsn is not None:
                g = di - float(lsn)
            elif pm:
                g = float(L - pm[-1])
            else:
                g = di + 1.0
            if g < 0.0:
                g = 0.0
            acc = 0.0
            for kk in (1, 2, 3):
                d = g - kk * tmean
                acc += math.exp(-(d * d) / (2.0 * s2))
            cad_v[i] = acc

        # fuziune Borda ponderata -> permutare completa determinista 1..N
        b_pair = _borda(pair_v, hot)
        b_dev = _borda(dev_v, hot)
        b_pos = _borda(pos_v, hot)
        b_cad = _borda(cad_v, hot)
        final = [0.0] * N
        for i in range(N):
            final[i] = (W_PAIR * b_pair[i] + W_DEV * b_dev[i]
                        + W_POS * b_pos[i] + W_CAD * b_cad[i])
        order = sorted(range(N), key=lambda i: (-final[i], -hot[i], i))
        out = [i + 1 for i in order]
        if len(cache) < 4096:
            cache[sig] = out
        return list(out)

    return rank