NAME = "zzsecres"
DESCRIPTION = 'Combinatie de 3 mecanisme ortogonale: (1) sector-cycle — intarziere geometrica standardizata la nivel de sectoare de marime SECTOR_SIZE (z al gapului de la ultima aparitie a sectorului, p-sector hypergeometric cu FPC); (2) deviatia macro — z-score lifetime per numar sub binomial, semn tunabil; (3) rezonanta de gap — faza raportului gap curent / ciclu personal (media gapurilor dintre aparitii in fereastra) recompensata gaussian pe multiplii intregi 1..3. DIAG expune sector_z si rezonanta per draw, flip-urile de semn sectoriale si span-ul rezonantei pentru poarta de pre-ablatie.'
SOURCE = 'model + 2026-09-17'

import math

W_SEC = 0.7
SECTOR_SIZE = 7
W_DEV = 0.4
DEV_SIGN = 1.0
W_RES = 0.5
RES_SIGMA = 0.55
RES_KMAX = 3
RES_BASE = 0.30
ZCAP = 3.0
RES_GMAX = 400

DIAG = {
    'sector_z': [],
    'resonance': [],
    'resonance_min': None,
    'resonance_max': None,
    'last_res_span': 0.0,
    'sector_flips': 0,
    'orient_checks': 0,
    'orient_flips': 0,
    'prev_sector_z': [],
}


def create(config):
    N = int(config.number_range)
    KD = int(config.numbers_drawn)
    S = SECTOR_SIZE if SECTOR_SIZE >= 1 else 5
    NS = (N + S - 1) // S
    sector_of = [0] * (N + 1)
    members = [[] for _ in range(NS)]
    for n in range(1, N + 1):
        s = (n - 1) // S
        sector_of[n] = s
        members[s].append(n)
    p = KD / float(N) if N > 0 else 0.0
    q = 1.0 - p

    mu_sec = [1.0] * NS
    sd_sec = [1.0] * NS
    for s in range(NS):
        sz = len(members[s])
        if sz <= 0 or N - sz < KD:
            ps = 1.0
        else:
            nohit = 1.0
            for i in range(KD):
                nohit *= (N - sz - i) / float(N - i)
            ps = 1.0 - nohit
        if 0.0 < ps < 1.0:
            qs = 1.0 - ps
            mu_sec[s] = qs / ps
            sd_sec[s] = math.sqrt(qs) / ps

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        D = int(draw_index)
        numbers = list(range(1, N + 1))
        seq = [set(x) for x in recent_deque]
        Wd = len(seq)

        # orientare: seq[-1] ar trebui sa fie cel mai nou draw; validare fata de last_seen == D-1
        flipped = False
        if Wd >= 2 and D >= 1:
            DIAG['orient_checks'] += 1
            so = 0
            sf = 0
            for n in seq[Wd - 1]:
                if last_seen.get(n, -10 ** 9) == D - 1:
                    so += 1
            for n in seq[0]:
                if last_seen.get(n, -10 ** 9) == D - 1:
                    sf += 1
            if sf > so:
                flipped = True
                DIAG['orient_flips'] += 1

        # varste hit-urilor per numar din fereastra (ascending)
        ages_by_n = {}
        for i in range(Wd):
            age = (i + 1) if flipped else (Wd - i)
            sset = seq[i]
            for n in sset:
                if 1 <= n <= N:
                    lst = ages_by_n.get(n)
                    if lst is None:
                        ages_by_n[n] = [age]
                    else:
                        lst.append(age)
        for n in ages_by_n:
            ages_by_n[n].sort()

        # (1) sector-cycle: z geometric al gapului sectorial
        z_sec = [0.0] * NS
        for s in range(NS):
            lh = -1
            for m in members[s]:
                v = last_seen.get(m, -1)
                if v > lh:
                    lh = v
            gap = (D - lh) if lh >= 0 else (D + 1)
            z = (gap - mu_sec[s]) / sd_sec[s]
            if z > ZCAP:
                z = ZCAP
            elif z < -ZCAP:
                z = -ZCAP
            z_sec[s] = z
        prev = DIAG['prev_sector_z']
        if len(prev) == NS:
            for s in range(NS):
                a = prev[s]
                b = z_sec[s]
                if (a > 0.05 and b < -0.05) or (a < -0.05 and b > 0.05):
                    DIAG['sector_flips'] += 1
        DIAG['prev_sector_z'] = list(z_sec)
        DIAG['sector_z'] = [round(v, 6) for v in z_sec]

        # (2) deviatia macro: parametri binomiali lifetime
        muD = D * p
        sdD = math.sqrt(max(D * p * q, 1e-12))

        # (3) rezonanta de gap per numar
        res = [0.0] * (N + 1)
        inv_sig = 1.0 / RES_SIGMA
        for n in numbers:
            la = last_seen.get(n, -1)
            if la < 0:
                continue
            g = D - la
            if g <= 0 or g > RES_GMAX:
                continue
            ages = ages_by_n.get(n)
            if ages is None or len(ages) < 3:
                continue
            csum = 0.0
            for k in range(1, len(ages)):
                csum += ages[k] - ages[k - 1]
            c = csum / (len(ages) - 1)
            if c < 0.5:
                continue
            r = g / c
            best = 0.0
            for k in range(1, RES_KMAX + 1):
                d = (r - k) * inv_sig
                v = math.exp(-0.5 * d * d)
                if v > best:
                    best = v
            res[n] = best
        lmin = 0.0
        lmax = 0.0
        for n in numbers:
            v = res[n]
            if v > lmax:
                lmax = v
        if DIAG['resonance_min'] is None or lmin < DIAG['resonance_min']:
            DIAG['resonance_min'] = lmin
        if DIAG['resonance_max'] is None or lmax > DIAG['resonance_max']:
            DIAG['resonance_max'] = lmax
        DIAG['last_res_span'] = round(DIAG['resonance_max'] - DIAG['resonance_min'], 6)
        DIAG['resonance'] = [round(res[n], 6) for n in numbers]

        # scor compozit
        sc = {}
        for n in numbers:
            z_n = (counts.get(n, 0) - muD) / sdD
            if z_n > ZCAP:
                z_n = ZCAP
            elif z_n < -ZCAP:
                z_n = -ZCAP
            v = W_SEC * z_sec[sector_of[n]] + DEV_SIGN * W_DEV * z_n + W_RES * (res[n] - RES_BASE)
            sc[n] = round(v, 9)
        return sorted(numbers, key=lambda x: (-sc[x], x))

    return rank