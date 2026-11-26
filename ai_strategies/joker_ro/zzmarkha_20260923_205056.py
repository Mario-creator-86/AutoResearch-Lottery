NAME = "zzmarkha"
DESCRIPTION = 'Model Markov pe seturi cu hazard empiric: (1) matrice de tranzitii intre extrageri consecutive din fereastra recenta, ponderata exponential cu varsta perechii, agregata peste membrii ultimei extrageri; (2) raport de hazard empiric per numar = (gap+1)/(own-rate+1), cu own-rate=(D+1)/(counts+1); (3) reversion pe sectoare de 5 numere dupa deviatia z a acoperirii deculate (EWMA) fata de asteptarea binomiala; (4) penalizare a vecinilor numerici (n-1, n+1) des intalniti recent. Toate semnalele z-standardizate; orientarea deque-ului verificata fata de last_seen si corectata automat (diag newest_first).'
SOURCE = 'model + 2026-09-23'

import math

W_MARKOV = 1.00
W_HAZ = 0.85
W_SECTOR = 0.60
W_SPACE = 0.45
DECAY_MK = 0.94
DECAY_SEC = 0.94
DECAY_SP = 0.97
SEC_SIZE = 5
EPS = 1e-12

DIAG = {'newest_first': False, 'orient_checks': 0, 'orient_flips': 0, 'contrib': {}}


def create(config):
    N = int(config.number_range)
    KD = int(config.numbers_drawn)
    numbers = list(range(1, N + 1))
    p_sec = float(SEC_SIZE) / float(N) if N > 0 else 0.0
    var_sec = float(KD) * p_sec * (1.0 - p_sec)
    S = (N + SEC_SIZE - 1) // SEC_SIZE
    sec_of = {}
    for n in numbers:
        sec_of[n] = (n - 1) // SEC_SIZE

    def zstd(vals):
        m = sum(vals) / float(len(vals))
        v = sum((x - m) * (x - m) for x in vals) / float(len(vals))
        sd = math.sqrt(v)
        if sd < EPS:
            return [0.0] * len(vals)
        return [(x - m) / sd for x in vals]

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        D = int(draw_index)
        seq = [set(s) for s in recent_deque]
        L = len(seq)

        # orientare: seq[-1] ar trebui sa fie cel mai nou draw (toate membres au last_seen == D-1)
        if L >= 2 and D >= 1:
            DIAG['orient_checks'] += 1
            hn = 0
            ho = 0
            for x in seq[-1]:
                if last_seen.get(x, -1000000000) == D - 1:
                    hn += 1
            for x in seq[0]:
                if last_seen.get(x, -1000000000) == D - 1:
                    ho += 1
            if ho > hn:
                seq.reverse()
                DIAG['orient_flips'] += 1
                DIAG['newest_first'] = True
            else:
                DIAG['newest_first'] = False
        newest = seq[-1] if L > 0 else set()

        # (1) matrice Markov de tranzitii intre extrageri consecutive, decay exponential pe varsta
        T = [[0.0] * (N + 1) for _ in range(N + 1)]
        for i in range(L - 1, 0, -1):
            age = L - 1 - i
            w = DECAY_MK ** age
            a = seq[i - 1]
            b = seq[i]
            for m in a:
                row = T[m]
                for n2 in b:
                    row[n2] += w
        mk = [0.0] * N
        if newest:
            kn = float(len(newest))
            for n in numbers:
                s = 0.0
                for m in newest:
                    s += T[m][n]
                mk[n - 1] = s / kn
        mz = zstd(mk)

        # (2) hazard empiric per numar: gap curent raportat la own-rate
        hz = [0.0] * N
        for n in numbers:
            c = counts.get(n, 0)
            g = D - last_seen.get(n, -1)
            mean_gap = (D + 1) / float(c + 1)
            hz[n - 1] = (g + 1.0) / (mean_gap + 1.0)
        hzz = zstd(hz)

        # (3) reversion pe sectoare: faverez sectoarele subacoperite in fereastra deculata
        cs = [0.0] * S
        wsum = 0.0
        for i in range(L - 1, -1, -1):
            age = L - 1 - i
            w = DECAY_SEC ** age
            wsum += w
            for x in seq[i]:
                cs[sec_of[x]] += w
        exp_s = wsum * float(KD) * p_sec
        sd_s = math.sqrt(max(wsum * var_sec, EPS))
        sc = [0.0] * N
        for n in numbers:
            sc[n - 1] = -(cs[sec_of[n]] - exp_s) / sd_s

        # (4) densitate de vecini numerici din fereastra, penalizata
        dens = [0.0] * N
        for i in range(L - 1, -1, -1):
            age = L - 1 - i
            w = DECAY_SP ** age
            for x in seq[i]:
                if x > 1:
                    dens[x - 2] += w
                if x < N:
                    dens[x] += w
        dzz = zstd(dens)

        contrib = {'markov': 0.0, 'haz': 0.0, 'sector': 0.0, 'space': 0.0, 'total': 0.0}
        scores = {}
        for i in range(N):
            n = numbers[i]
            s = W_MARKOV * mz[i] + W_HAZ * hzz[i] + W_SECTOR * sc[i] + W_SPACE * (-dzz[i])
            s += 1e-09 * float(counts.get(n, 0))
            scores[n] = s
            contrib['markov'] += abs(W_MARKOV * mz[i]) / float(N)
            contrib['haz'] += abs(W_HAZ * hzz[i]) / float(N)
            contrib['sector'] += abs(W_SECTOR * sc[i]) / float(N)
            contrib['space'] += abs(W_SPACE * dzz[i]) / float(N)
            contrib['total'] += abs(s) / float(N)
        DIAG['contrib'] = contrib
        return sorted(numbers, key=lambda v: (-scores[v], v))

    rank._diag = DIAG
    return rank