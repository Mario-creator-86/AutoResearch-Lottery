import math

NAME = "znxdecay"
DESCRIPTION = 'Scor hibrid determinist: EWMA exponentiala pe aparitiile recente, afinitate de perechi cu clicica fierbinte (rezidualizata fata de recenta proprie), semnal de intarziere saturat tanh, reversie la medie pe sectoare si ancora de frecventa pe termen lung; combinare liniara + interactiune recen*perechi; detecteaza orientarea deque-ului.'
SOURCE = 'model cantitativ + statistici kaskada_pl, 2026-09-21'

DECAY = 0.96      # decay EWMA pe recenta
EW_DRAWS = 60     # adancime maxima EWMA
PAIR_DRAWS = 30   # fereastra pt afinitatea de perechi
TOPK = 6          # marimea clicicii fierbinti
SEC_DRAWS = 20    # fereastra pt reversia pe sectoare
W_REC = 1.00
W_PAIR = 0.55
W_GAP = 0.35
W_SECT = 0.30
W_FREQ = 0.25
W_X = 0.15
NEG = -1 << 30


def create(config):
    nr = config.number_range
    nd = config.numbers_drawn
    p = float(nd) / float(nr)

    def zscore(vals):
        m = sum(vals) / len(vals)
        var = sum((v - m) * (v - m) for v in vals) / len(vals)
        s = var ** 0.5
        if s < 1e-9:
            return [0.0] * len(vals)
        return [(v - m) / s for v in vals]

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, nr + 1))
        L = len(recent_deque)

        # orientarea deque-ului: capatul cu cele mai multe numere din ultima extragere e cel mai nou
        newest_first = True
        if L >= 2:
            tgt = draw_index - 1
            f0 = recent_deque[0]
            f1 = recent_deque[L - 1]
            h0 = sum(1 for n in f0 if last_seen.get(n, NEG) == tgt)
            h1 = sum(1 for n in f1 if last_seen.get(n, NEG) == tgt)
            if h0 != h1:
                newest_first = h0 > h1
            else:
                s0 = sum(last_seen.get(n, NEG) for n in f0)
                s1 = sum(last_seen.get(n, NEG) for n in f1)
                newest_first = s0 >= s1

        def draw_at(d):  # d = distanta de la cel mai nou
            return recent_deque[d] if newest_first else recent_deque[L - 1 - d]

        # 1) EWMA recență
        Le = min(L, EW_DRAWS)
        rec = {n: 0.0 for n in numbers}
        for d in range(Le):
            w = DECAY ** d
            for n in draw_at(d):
                rec[n] += w
        topk = set(sorted(numbers, key=lambda n: (-rec[n], n))[:TOPK]) if Le > 0 else set()
        zr = zscore([rec[n] for n in numbers])

        # 2) afinitate de perechi cu clicica fierbinte, rezidualizata fata de recenta din fereastra
        pair = {n: 0.0 for n in numbers}
        rpw = {n: 0.0 for n in numbers}
        Lp = min(L, PAIR_DRAWS)
        for d in range(Lp):
            s = draw_at(d)
            w = DECAY ** d
            inter = s & topk
            h = len(inter)
            for n in s:
                pair[n] += w * h
                rpw[n] += w
            for m in inter:
                pair[m] -= w  # fara auto-perechi
        mp = sum(pair.values()) / nr
        mr = sum(rpw.values()) / nr
        cov = sum((pair[n] - mp) * (rpw[n] - mr) for n in numbers) / nr
        var = sum((rpw[n] - mr) ** 2 for n in numbers) / nr
        beta = cov / var if var > 1e-9 else 0.0
        zp = zscore([pair[n] - beta * rpw[n] for n in numbers])

        # 3) intarziere (ciclu overdue), saturata
        ov_raw = []
        for n in numbers:
            c = counts.get(n, 0)
            gap = (draw_index + 1) - last_seen[n] if n in last_seen else draw_index + 1
            mu = max(draw_index / c, 1.0) if c > 0 else 2.0
            ov_raw.append(math.tanh((gap - mu) / (math.sqrt(mu) + 1.0)))
        ov = zscore(ov_raw)

        # 4) reversie la medie pe sectoare
        sec_len = max(1, nr // 4)
        Ls = min(L, SEC_DRAWS)
        acc = {}
        for d in range(Ls):
            for n in draw_at(d):
                k = (n - 1) // sec_len
                acc[k] = acc.get(k, 0) + 1
        exp_sec = sec_len * p * Ls
        secdef = [(exp_sec - acc.get((n - 1) // sec_len, 0)) / (math.sqrt(exp_sec) + 1.0)
                  for n in numbers]

        # 5) ancora frecventa pe termen lung
        zfreq = zscore([counts.get(n, 0) for n in numbers])

        scores = {}
        for i, n in enumerate(numbers):
            scores[n] = (W_REC * zr[i] + W_PAIR * zp[i] + W_GAP * ov[i]
                         + W_SECT * secdef[i] + W_FREQ * zfreq[i]
                         + W_X * zr[i] * zp[i])
        out = sorted(numbers, key=lambda n: (-scores[n], n))
        rank._diag = {
            'newest_first': newest_first,
            'contrib': {'rec': W_REC, 'pair': W_PAIR, 'gap': W_GAP,
                        'sect': W_SECT, 'freq': W_FREQ, 'x': W_X},
            'L': L, 'ew_used': Le, 'pair_used': Lp,
        }
        return out

    return rank