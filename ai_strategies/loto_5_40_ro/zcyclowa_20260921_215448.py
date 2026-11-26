import math

NAME = "zcyclowa"
DESCRIPTION = 'Scor ciclic personal per numar: distanta fata de multiplii perioadei proprii de reaparitie (cicluri de intarziere), combinata cu EWMA exponential pe fereastra recenta, reversie la medie fata de frecventa asteptata si entropie de distributie pe 3 segmente ale ferestrei; componente z-standardizate, sortare deterministica.'
SOURCE = 'glm-5.3-flash + 2026-09-21'


def create(config):
    N = config.number_range
    K = config.numbers_drawn
    DECAY = 0.94
    W_REC = 1.0
    W_CYC = 0.85
    W_DEV = 0.55
    W_ENT = 0.35

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        valid = set(numbers)
        total = draw_index if (draw_index and draw_index > 0) else 1
        p = (K / float(N)) if N and N > 0 else 0.0
        rate_exp = p if p > 0.0 else 0.15

        try:
            hist = list(recent_deque)
        except Exception:
            hist = []
        if recent_window and len(hist) > recent_window:
            hist = hist[-recent_window:]
        L = len(hist)
        seg = max(1, L // 3)

        rec_w = dict((n, 0.0) for n in numbers)
        seg_hits = dict((n, [0, 0, 0]) for n in numbers)
        wsum = 0.0
        for i in range(L):
            w = DECAY ** (L - 1 - i)
            wsum += w
            si = 0 if i < seg else (1 if i < 2 * seg else 2)
            for n in hist[i]:
                if n in valid:
                    rec_w[n] += w
                    seg_hits[n][si] += 1
        if wsum <= 1e-12:
            wsum = 1.0

        cyc_raw = {}
        dev_raw = {}
        ent_raw = {}
        rec_rate = {}
        for n in numbers:
            c = counts.get(n, 0)
            rec_rate[n] = (rec_w[n] / wsum) / rate_exp
            if c > 0:
                period = total / float(c)
            else:
                period = 1.0 / rate_exp
            if period < 0.5:
                period = 0.5
            gap = None
            if last_seen:
                try:
                    if n in last_seen:
                        gap = draw_index - int(last_seen[n])
                except Exception:
                    gap = None
            if gap is None:
                cyc_raw[n] = -2.0
            else:
                best = abs(gap - period)
                for k in range(2, 9):
                    d = abs(gap - k * period)
                    if d < best:
                        best = d
                cyc_raw[n] = -best / (period + 1.0)
            exp_n = total * rate_exp
            sd_n = math.sqrt(max(total * rate_exp * (1.0 - rate_exp), 1e-9))
            dev_raw[n] = -(c - exp_n) / sd_n
            hits = seg_hits[n]
            tot = hits[0] + hits[1] + hits[2]
            if tot > 0:
                h = 0.0
                for q in hits:
                    if q > 0:
                        fr = q / float(tot)
                        h -= fr * math.log(fr)
                ent_raw[n] = h / math.log(3.0)
            else:
                ent_raw[n] = 0.0

        def zscore(d):
            m = 0.0
            for n in numbers:
                m += d[n]
            m /= float(len(numbers))
            v = 0.0
            for n in numbers:
                dv = d[n] - m
                v += dv * dv
            v /= float(len(numbers))
            sd = math.sqrt(v)
            if sd <= 1e-12:
                return dict((n, 0.0) for n in numbers)
            return dict((n, (d[n] - m) / sd) for n in numbers)

        z_rec = zscore(rec_rate)
        z_cyc = zscore(cyc_raw)
        z_dev = zscore(dev_raw)
        z_ent = zscore(ent_raw)

        score = {}
        for n in numbers:
            score[n] = W_REC * z_rec[n] + W_CYC * z_cyc[n] + W_DEV * z_dev[n] + W_ENT * z_ent[n]

        try:
            cnt = float(len(numbers))
            rank._diag = {
                'newest_first': False,
                'contrib': {
                    'rec': sum(abs(W_REC * z_rec[n]) for n in numbers) / cnt,
                    'cyc': sum(abs(W_CYC * z_cyc[n]) for n in numbers) / cnt,
                    'gap': sum(abs(W_DEV * z_dev[n]) for n in numbers) / cnt,
                    'ent': sum(abs(W_ENT * z_ent[n]) for n in numbers) / cnt,
                    'x': sum(abs(score[n]) for n in numbers) / cnt,
                },
            }
        except Exception:
            pass

        return sorted(numbers, key=lambda n: (-score[n], n))

    return rank