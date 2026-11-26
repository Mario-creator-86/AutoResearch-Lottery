NAME = "zpairflo"
DESCRIPTION = 'Flux de afinitate Markov pe perechi: co-occurrentele din fereastra recenta, ponderate exponential, sunt normalizate la numarul asteptat sub independenta (lift) si propagate catre parteneri; combinate cu un nucleu de difuzie spatiala exp(-d/sigma) peste numerele extrase recent si o mica revertie contrarian pe frecventa locala din fereastra.'
SOURCE = 'model cantitativ + 2026-09-21'

import math
from collections import defaultdict

DIAG = {'orient': None, 'newest_first': None, 'orient_flips': 0, 'contrib': {}}

# hiperparametri expusi pentru grid-search
DECAY = 0.94    # decay exponential pe extrageri (half-life ~11)
W_PAIR = 1.0    # greutate flux lift pe perechi
W_DIFF = 0.6    # greutate difuzie spatiala
W_REV = 0.4     # greutate contrarian local
W_CNT = 0.0     # ancora frecventa pe termen lung (0 = dezactivata)
SIGMA = 4.0     # banda nucleului de difuzie
RCUT = 5        # raza taiata a nucleului
PAIR_WIN = 160  # fereastra pt. perechi (decay ~9e-5 la margine)
DIFF_WIN = 60   # fereastra pt. difuzie


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    p = K / float(N)
    pe = K * (K - 1.0) / (N * (N - 1.0))  # P(numar si partener in aceeasi extragere)
    kernel = [math.exp(-d / float(SIGMA)) for d in range(RCUT + 1)]

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        L = len(recent_deque)

        # auto-detectarea orientarii deque-ului: capatul cu recenta medie
        # mai mica este cel mai nou
        def mean_rec(st):
            if not st:
                return float('inf')
            tot = 0.0
            for x in st:
                ls = last_seen.get(x)
                tot += (draw_index - ls) if ls is not None else float(draw_index)
            return tot / float(len(st))

        newest_last = True
        if L >= 2:
            newest_last = mean_rec(recent_deque[-1]) <= mean_rec(recent_deque[0])
        orient = 'newest_last' if newest_last else 'newest_first'
        if DIAG['orient'] is not None and DIAG['orient'] != orient:
            DIAG['orient_flips'] += 1
        DIAG['orient'] = orient
        DIAG['newest_first'] = (orient == 'newest_first')

        seq = list(recent_deque)
        if newest_last:
            seq.reverse()  # seq[i] are varsta i (0 = cel mai nou)

        # 1) perechi decay-uite -> lift fata de independenta
        pairc = defaultdict(float)
        norm_decay = 0.0
        for i in range(min(L, PAIR_WIN)):
            draw = seq[i]
            w = DECAY ** i
            norm_decay += w
            ds = [x for x in draw if 1 <= x <= N]
            dl = len(ds)
            for ai in range(dl):
                na = ds[ai]
                for bi in range(ai + 1, dl):
                    nb = ds[bi]
                    key = (na, nb) if na < nb else (nb, na)
                    pairc[key] += w
        exp_pairs = pe * norm_decay

        s_pair = dict.fromkeys(numbers, 0.0)
        if exp_pairs > 1e-9:
            inv = 1.0 / exp_pairs
            for (a, b), c in pairc.items():
                lift = c * inv
                s_pair[a] += lift
                s_pair[b] += lift

        # 2) difuzie spatiala pe axa numerelor
        s_diff = dict.fromkeys(numbers, 0.0)
        for i in range(min(L, DIFF_WIN)):
            draw = seq[i]
            w = DECAY ** i
            for m in draw:
                if 1 <= m <= N:
                    lo = m - RCUT
                    if lo < 1:
                        lo = 1
                    hi = m + RCUT
                    if hi > N:
                        hi = N
                    for n in range(lo, hi + 1):
                        s_diff[n] += w * kernel[abs(n - m)]

        # 3) frecventa locala (pentru contrarian)
        loc = dict.fromkeys(numbers, 0)
        for i in range(min(L, PAIR_WIN)):
            for x in seq[i]:
                if 1 <= x <= N:
                    loc[x] += 1

        def zstd(d):
            vals = list(d.values())
            m = len(vals)
            mu = sum(vals) / float(m)
            var = 0.0
            for v in vals:
                var += (v - mu) * (v - mu)
            sd = math.sqrt(var / m)
            if sd < 1e-9:
                return {k: 0.0 for k in d}
            return {k: (v - mu) / sd for k, v in d.items()}

        zp = zstd(s_pair)
        zd = zstd(s_diff)

        Lp = L * p
        sdp = math.sqrt(L * p * (1.0 - p)) if L > 0 else 1.0
        if sdp < 1e-9:
            sdp = 1.0

        cnts = [counts.get(n, 0) for n in numbers]
        cmax = max(cnts) if cnts else 0
        if cmax <= 0:
            cmax = 1

        scores = {}
        a_pair = 0.0
        a_diff = 0.0
        a_rev = 0.0
        for j, n in enumerate(numbers):
            zl = (loc[n] - Lp) / sdp
            cp = W_PAIR * zp[n]
            cd = W_DIFF * zd[n]
            cr = -W_REV * zl
            scores[n] = cp + cd + cr + W_CNT * (cnts[j] / float(cmax))
            a_pair += abs(cp)
            a_diff += abs(cd)
            a_rev += abs(cr)
        DIAG['contrib'] = {'pair': round(a_pair / N, 4),
                           'diff': round(a_diff / N, 4),
                           'rev': round(a_rev / N, 4)}

        return sorted(numbers, key=lambda n: (-scores[n], -cnts[n - 1], n))

    rank._diag = DIAG
    return rank