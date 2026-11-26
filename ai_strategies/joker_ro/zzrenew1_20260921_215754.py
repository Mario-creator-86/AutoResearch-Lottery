NAME = "zzrenew1"
DESCRIPTION = 'Model de reinnoire per-numar: hazard = raportul varsta-curenta / interval-mediu propriu estimat din fereastra recenta (mean-reversion pe ritmul propriu, saturat tanh in log-space), combinat cu deficitele de echilibru ale sectoarelor consecutive de 5 si ale claselor mod-5, EWMA cu decay, ancora fina de frecventa lifetime si o penalizare anti-companion fata de top seeds (raspandirea acoperirii tichetului). Orientarea deque-ului este verificata fata de last_seen si corectata automat.'
SOURCE = 'model + 2026-09-21'

import math

# ---- constante tunabile (grile de scan pentru ablatie) ----
DECAY = 0.96
W_RENEW = 0.60
W_SECT = 0.45
W_MOD = 0.30
W_EW = 0.55
W_LIFE = 0.30
W_X = 0.20
RCAP = 1.6
SEEDS = 12

DIAG = {'contrib': {}, 'newest_first': True, 'orient_checks': 0, 'orient_flips': 0}


def create(config):
    N = int(config.number_range)
    KD = int(config.numbers_drawn)
    p = KD / float(N) if N > 0 else 0.0
    q = 1.0 - p
    m_glob = (1.0 / p) if p > 0 else float(N)
    sec_size = 5 if N >= 45 else max(1, N // 9)
    n_sec = (N + sec_size - 1) // sec_size
    n_mod = 5

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        D = int(draw_index)
        numbers = list(range(1, N + 1))
        seq = list(recent_deque)
        W = len(seq)

        # --- orientare: seq[-1] trebuie sa fie cel mai nou draw (index D-1) ---
        newest_first = True
        if W >= 2 and D >= 1:
            DIAG['orient_checks'] += 1
            hn = 0
            ho = 0
            for n in seq[-1]:
                if last_seen.get(n, -10 ** 9) == D - 1:
                    hn += 1
            for n in seq[0]:
                if last_seen.get(n, -10 ** 9) == D - 1:
                    ho += 1
            flip = ho > hn
            if ho == hn:
                vn = [last_seen.get(n, -10 ** 9) for n in seq[-1]]
                vo = [last_seen.get(n, -10 ** 9) for n in seq[0]]
                vn = [v for v in vn if v > -10 ** 8]
                vo = [v for v in vo if v > -10 ** 8]
                if vn and vo:
                    flip = (sum(vo) / float(len(vo))) > (sum(vn) / float(len(vn)))
                elif vo and not vn:
                    flip = True
            if flip:
                seq.reverse()
                newest_first = False
                DIAG['orient_flips'] += 1
        DIAG['newest_first'] = newest_first

        base = D - W
        ew = [0.0] * (N + 1)
        sec_cnt = [0] * n_sec
        mod_cnt = [0] * n_mod
        appear = {}
        pairc = {}

        for i, draw in enumerate(seq):
            wgt = DECAY ** (W - 1 - i)
            idx = base + i
            dl = sorted(draw)
            for n in dl:
                ew[n] += wgt
                sec_cnt[(n - 1) // sec_size] += 1
                mod_cnt[n % n_mod] += 1
                if n in appear:
                    appear[n].append(idx)
                else:
                    appear[n] = [idx]
            L = len(dl)
            for ai in range(L - 1):
                a = dl[ai]
                for bi in range(ai + 1, L):
                    key = (a, dl[bi])
                    if key in pairc:
                        pairc[key] += 1
                    else:
                        pairc[key] = 1

        def zstd(vals):
            mu = 0.0
            for v in vals:
                mu += v
            mu /= float(len(vals))
            var = 0.0
            for v in vals:
                d = v - mu
                var += d * d
            var /= float(len(vals))
            if var < 1e-12:
                return [0.0] * len(vals)
            sd = math.sqrt(var)
            return [(v - mu) / sd for v in vals]

        # (1) renewal: log(age / own-mean-interval), saturat
        renew = []
        for n in numbers:
            ap = appear.get(n)
            a = D - last_seen[n] if n in last_seen else D
            if a < 1:
                a = 1
            if ap is not None and len(ap) >= 2:
                si = 0
                for j in range(1, len(ap)):
                    si += ap[j] - ap[j - 1]
                m_hat = (si + m_glob) / float(len(ap))
            else:
                m_hat = m_glob
            if m_hat < 0.5:
                m_hat = 0.5
            lr = math.log(a / m_hat)
            if lr > RCAP:
                lr = RCAP
            elif lr < -RCAP:
                lr = -RCAP
            renew.append(math.tanh(lr))
        zr = zstd(renew)

        # (2) deficit sectoare consecutive de 5
        exp_s = W * KD * (sec_size / float(N))
        zs = zstd([(exp_s - sec_cnt[(n - 1) // sec_size]) / math.sqrt(exp_s + 1.0) for n in numbers])

        # (3) deficit clase mod-5
        exp_m = W * KD / float(n_mod)
        zm = zstd([(exp_m - mod_cnt[n % n_mod]) / math.sqrt(exp_m + 1.0) for n in numbers])

        # (4) EWMA recent
        zew = zstd([ew[n] for n in numbers])

        # (5) ancora lifetime z
        lm = D * p
        lsd = math.sqrt(D * p * q) if D > 0 else 0.0
        zl = [0.0] * N
        if lsd > 1e-12:
            for i2, n in enumerate(numbers):
                zl[i2] = (counts.get(n, 0) - lm) / lsd

        base_score = [0.0] * (N + 1)
        for i2, n in enumerate(numbers):
            base_score[n] = (W_RENEW * zr[i2] + W_SECT * zs[i2] + W_MOD * zm[i2]
                             + W_EW * zew[i2] + W_LIFE * zl[i2])

        # (6) non-liniar: penalizare companion fata de top seeds
        order0 = sorted(numbers, key=lambda n: (-base_score[n], n))
        seeds = order0[:SEEDS]
        comp = [0.0] * (N + 1)
        for n in numbers:
            c = 0
            for s in seeds:
                if s == n:
                    continue
                key = (n, s) if n < s else (s, n)
                c += pairc.get(key, 0)
            comp[n] = c / float(len(seeds))
        zc = zstd([comp[n] for n in numbers])

        final = [0.0] * (N + 1)
        for i2, n in enumerate(numbers):
            final[n] = base_score[n] - W_X * zc[i2]

        order = sorted(numbers, key=lambda n: (-final[n], n))

        contrib = {'renew': 0.0, 'sect': 0.0, 'mod': 0.0, 'ew': 0.0, 'life': 0.0, 'x': 0.0}
        for n in order[:5]:
            i2 = n - 1
            contrib['renew'] += W_RENEW * zr[i2]
            contrib['sect'] += W_SECT * zs[i2]
            contrib['mod'] += W_MOD * zm[i2]
            contrib['ew'] += W_EW * zew[i2]
            contrib['life'] += W_LIFE * zl[i2]
            contrib['x'] -= W_X * zc[i2]
        DIAG['contrib'] = contrib

        return order

    rank._diag = DIAG
    return rank