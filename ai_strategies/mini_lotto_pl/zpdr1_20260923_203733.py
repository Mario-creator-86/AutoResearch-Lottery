import math

NAME = "zpdr1"
DESCRIPTION = 'Difuzie de perechi de la un nucleu EWMA-hot (top RCUT) + deviatie EWMA fata de frecventa asteptata + intarziere normalizata (overdue); componentele sunt z-standardizate si combinate liniar, fara a folosi frecventele globale.'
SOURCE = 'glm-5.3-flash + 2026-09-23'

DECAY = 0.94
W_PAIR = 1.0
W_DIFF = 0.6
W_REV = 0.4
RCUT = 8
GAP_NORM = 8.4
REV_CAP = 4.0


def create(config):
    P = config.number_range
    rate = config.numbers_drawn / float(P)

    def zscore(vals):
        n = len(vals)
        mean = 0.0
        for v in vals:
            mean += v
        mean /= n
        var = 0.0
        for v in vals:
            d = v - mean
            var += d * d
        var /= n
        std = math.sqrt(var)
        if std < 1e-12:
            return [0.0] * n
        inv = 1.0 / std
        return [(v - mean) * inv for v in vals]

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, P + 1))
        seq = list(recent_deque)
        k = len(seq)

        # detectarea orientarii deque-ului: setul numerelor cu last_seen maxim
        # este extragerea cea mai noua; pozitia lui in deque arata capatul proaspat
        newest_first = False
        if k and last_seen:
            max_seen = max(last_seen.values())
            target = set(n for n, v in last_seen.items() if v == max_seen)
            pos = None
            for i in range(k):
                if set(seq[i]) == target:
                    pos = i
                    break
            if pos is not None:
                newest_first = (pos <= (k - 1) // 2)
        chrono = seq[::-1] if newest_first else seq

        # ponderi exponential-decay: cele mai noi extrageri au greutate 1
        weights = [0.0] * k
        if k:
            weights[k - 1] = 1.0
            for j in range(k - 2, -1, -1):
                weights[j] = weights[j + 1] * DECAY

        # 1) EWMA pe frecventele recente -> deviatia fata de rata asteptata
        ew = [0.0] * (P + 1)
        wsum = 0.0
        for j in range(k):
            w = weights[j]
            wsum += w
            for m in chrono[j]:
                if 1 <= m <= P:
                    ew[m] += w
        expected = rate * wsum
        diff = [ew[n] - expected for n in numbers]

        # 2) nucleu top-RCUT dupa EWMA; scor de difuzie = co-aparitii decay-uite cu nucleul
        nucleus = sorted(numbers, key=lambda n: (-ew[n], n))[:RCUT]
        nuc_set = set(nucleus)
        cow = {}
        for m in nucleus:
            cow[m] = {}
        for j in range(k):
            w = weights[j]
            s = chrono[j]
            inter = nuc_set.intersection(s)
            if not inter:
                continue
            for m in inter:
                row = cow[m]
                for x in s:
                    if x != m:
                        row[x] = row.get(x, 0.0) + w
        pair_raw = [0.0] * (P + 1)
        for n in numbers:
            tot = 0.0
            for m in nucleus:
                if m != n:
                    v = cow[m].get(n)
                    if v is not None:
                        tot += v
            pair_raw[n] = tot

        # 3) intarziere normalizata: gap / gap_asteptat (42/5), cu plafon
        cap = REV_CAP * GAP_NORM
        rev = [0.0] * (P + 1)
        for n in numbers:
            ls = last_seen.get(n)
            if ls is None:
                gap = draw_index if draw_index > cap else cap
            else:
                gap = draw_index - ls
            g = gap / GAP_NORM
            if g > REV_CAP:
                g = REV_CAP
            rev[n] = g

        zp = zscore(pair_raw[1:])
        zd = zscore(diff)
        zr = zscore(rev[1:])

        keyed = []
        for i in range(P):
            n = numbers[i]
            sc = W_PAIR * zp[i] + W_DIFF * zd[i] + W_REV * zr[i]
            keyed.append((-sc, n))
        keyed.sort()
        order = [t[1] for t in keyed]

        try:
            rank._diag = {
                'newest_first': newest_first,
                'contrib': {
                    'pair': sum(pair_raw[1:]) / P,
                    'diff': sum(diff) / P,
                    'rev': sum(rev[1:]) / P
                },
                'nucleus': list(nucleus),
                'params': {'DECAY': DECAY, 'W_PAIR': W_PAIR, 'W_DIFF': W_DIFF, 'W_REV': W_REV, 'RCUT': RCUT},
                'draw_index': draw_index
            }
        except Exception:
            pass

        return order

    return rank