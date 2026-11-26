NAME = "zzpairfl"
DESCRIPTION = 'Flux de perechi pe graful de co-aparitie decayed (DECAY) ancorat pe ultima extragere, combinat cu z-score-ul pe fereastra recenta (B), hazardul de gap tanh pe raportul gap/gap-mediu-empiric (CG, A, semn GZ), invechirea sectoarelor de 10 si divergenta lifetime-vs-recent (lz); toate componentele standardizate zero-medie/deviatie-1 inainte de ponderare, plus un termen de interactiune perechi x gap.'
SOURCE = 'model cantitativ + 2026-09-20'

import math

# hiperparametri expusi pentru grid-search / ablatie (nume stabile)
DECAY = 0.93       # decay pe varsta extragerilor in graful de perechi
PAIR_SCALE = 0.30  # ponderea fluxului de perechi (pz)
B = 0.60           # ponderea bazei hot-z pe fereastra recenta (hz)
CG = 0.70          # castigul de urmarire a gap-ului / ponderea gz
A = 1.00           # scara tanh pe raportul de intarziere
GZ = 1             # +1 = urmareste overdue, -1 = anti-overdue
GAP_CAP = 6.0      # plafon pe raportul gap/gap-mediu
W_SZ = 0.35        # ponderea invechirii sectorului (sz)
W_LZ = 0.30        # ponderea divergentei lifetime-vs-recent (lz)
W_INT = 0.25       # ponderea interactiunii perechi x gap

DIAG = {'contrib': {}, 'calls': 0, 'orient': None, 'orient_flips': 0, 'last_top': ()}


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    p = K / float(N)
    S = (N + 9) // 10

    # precompute: sectorul fiecarui numar (sectoare de 10)
    sec_of = [0] * (N + 1)
    for s in range(S):
        lo = 10 * s + 1
        hi = min(10 * s + 10, N)
        for n in range(lo, hi + 1):
            sec_of[n] = s

    def _zstd(vals):
        m = sum(vals) / float(len(vals))
        var = 0.0
        for v in vals:
            d = v - m
            var += d * d
        sd = math.sqrt(var / float(len(vals)))
        if sd < 1e-12:
            return [0.0] * len(vals)
        return [(v - m) / sd for v in vals]

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        DIAG['calls'] += 1
        numbers = list(range(1, N + 1))
        T = float(max(draw_index, 1))
        L = len(recent_deque)

        # auto-detectarea orientarii deque-ului: setul cel mai nou are
        # recenza medie minima fata de last_seen
        def mean_recency(st):
            if not st:
                return float('inf')
            tot = 0.0
            for n in st:
                ls = last_seen.get(n)
                tot += (draw_index - ls) if ls is not None else float(draw_index)
            return tot / float(len(st))

        newest_last = True
        if L >= 2:
            newest_last = mean_recency(recent_deque[-1]) <= mean_recency(recent_deque[0])
        orient = 'newest_last' if newest_last else 'newest_first'
        if DIAG['orient'] is not None and DIAG['orient'] != orient:
            DIAG['orient_flips'] += 1
        DIAG['orient'] = orient

        seq = list(recent_deque)
        if newest_last and L >= 2:
            seq.reverse()  # seq[0] = cea mai noua extragere

        # (1) count-uri pe fereastra + invechirea primei aparitii per sector
        wcnt = dict.fromkeys(numbers, 0)
        sec_stale = [float(L)] * S
        sec_seen = [False] * S
        for age in range(L):
            hit = set()
            for n in seq[age]:
                if 1 <= n <= N:
                    wcnt[n] += 1
                    hit.add(sec_of[n])
            for s in hit:
                if not sec_seen[s]:
                    sec_stale[s] = float(age)
                    sec_seen[s] = True

        # (2) graful de perechi decayed pe fereastra recenta
        P = {}
        for age in range(L):
            w = DECAY ** age
            st = sorted(n for n in seq[age] if 1 <= n <= N)
            for i in range(len(st)):
                a = st[i]
                for j in range(i + 1, len(st)):
                    key = (a, st[j])
                    P[key] = P.get(key, 0.0) + w

        # (3) componente brute
        mu_w = L * p
        sd_w = math.sqrt(L * p * (1.0 - p)) if L > 0 else 0.0
        mu_l = T * p
        sd_l = math.sqrt(T * p * (1.0 - p) + 1e-9)

        hz_raw = [0.0] * N
        lz_raw = [0.0] * N
        gz_raw = [0.0] * N
        sz_raw = [0.0] * N
        for i in range(N):
            n = i + 1
            c = float(counts.get(n, 0))
            if sd_w > 1e-12:
                hz_raw[i] = (wcnt[n] - mu_w) / sd_w
            lz_raw[i] = (c - mu_l) / sd_l
            ls = last_seen.get(n)
            gap = float(draw_index - ls) if ls is not None else float(draw_index + 1)
            r = gap / ((T + 1.0) / (c + 1.0))
            if r > GAP_CAP:
                r = GAP_CAP
            gz_raw[i] = GZ * math.tanh(A * (r - 1.0))
            sz_raw[i] = sec_stale[sec_of[n]]

        # flux de perechi: afinitatea decaya a fiecarui numar cu numerele din ultima extragere
        pz_raw = [0.0] * N
        if L > 0:
            last = [n for n in seq[0] if 1 <= n <= N]
            for i in range(N):
                n = i + 1
                acc = 0.0
                for m in last:
                    if m != n:
                        key = (n, m) if n < m else (m, n)
                        acc += P.get(key, 0.0)
                pz_raw[i] = acc

        # (4) standardizare zero-medie / deviatie-1 pentru comparabilitatea ponderilor
        hz_s = _zstd(hz_raw)
        pz_s = _zstd(pz_raw)
        gz_s = _zstd(gz_raw)
        sz_s = _zstd(sz_raw)
        lz_s = _zstd(lz_raw)
        int_s = _zstd([pz_s[i] * gz_s[i] for i in range(N)])

        score = [0.0] * N
        for i in range(N):
            score[i] = (B * hz_s[i] + PAIR_SCALE * pz_s[i] + CG * gz_s[i]
                        + W_SZ * sz_s[i] + W_LZ * lz_s[i] + W_INT * int_s[i])

        order = sorted(range(N), key=lambda i: (-score[i], i + 1))
        DIAG['last_top'] = tuple(i + 1 for i in order[:K])

        # diagnostic: contributia medie absoluta a fiecarei componente pe top-K
        comp = {'hz': (B, hz_s), 'pz': (PAIR_SCALE, pz_s), 'gz': (CG, gz_s),
                'sz': (W_SZ, sz_s), 'lz': (W_LZ, lz_s), 'int': (W_INT, int_s)}
        top = order[:K]
        contrib = {}
        for name, (w, arr) in comp.items():
            tot = 0.0
            for i in top:
                tot += abs(w * arr[i])
            contrib[name] = round(tot / float(max(len(top), 1)), 6)
        DIAG['contrib'] = contrib

        return sorted(numbers, key=lambda n: (-score[n - 1], n))

    rank._diag = DIAG
    return rank