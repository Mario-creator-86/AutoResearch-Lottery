import math

NAME = "zmkvflux"
DESCRIPTION = ('Predictor Markov lag-1: estimeaza din fereastra recenta matricea de tranzitii '
               'si o conditioneaza pe numerele ultimei extrageri (ce tinde sa urmeze dupa ele), '
               'amestecat cu z-scori de frecventa (recenta + globala) si un tilt de mean-reversion '
               'pe sumele ultimelor extrageri, reponderat de o poarta de entropie.')
SOURCE = 'glm-5.3-flash + 2026-09-08'


def create(config):
    N = config.number_range          # ex. 40
    W = config.numbers_drawn         # ex. 6

    # --- hiperparametri deterministi ---
    ALPHA = 6.0        # pseudo-count Laplace pe randurile de tranzitie
    GLOB_W = 0.35      # pondere frecventa globala
    REC_W = 0.85       # pondere frecventa recenta
    MKV_W = 1.00       # pondere componenta Markov (semnalul principal)
    SUM_W = 0.50       # pondere tilt suma
    ENT_LO = 0.90      # poarta entropie: sub => structura locala
    ENT_HI = 0.985     # peste => zgomot uniform
    K_SUM = 30         # cate draw-uri pentru media sumelor
    P0 = float(W) / N
    MEAN_SUM = W * (N + 1) / 2.0
    MEAN_N = (N + 1) / 2.0
    STD_N = math.sqrt((N * N - 1) / 12.0)
    if N > 1 and N > W:
        SUM_STD = math.sqrt(W * ((N * N - 1) / 12.0) * ((N - W) / float(N - 1)))
    else:
        SUM_STD = 1.0

    def _z(vals):
        m = sum(vals) / len(vals)
        var = sum((v - m) * (v - m) for v in vals) / len(vals)
        sd = math.sqrt(var)
        if sd <= 1e-12:
            return [0.0] * len(vals)
        return [(v - m) / sd for v in vals]

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        D = list(recent_deque)   # conventie: append la dreapta => D[-1] = ultima extragere
        L = len(D)

        # 1) z pe frecventa globala
        z_glob = _z([float(counts.get(n, 0)) for n in numbers])

        # 2) z pe frecventa recenta + 3) entropia distributiei recente
        rec = dict.fromkeys(numbers, 0)
        for s in D:
            for i in s:
                if 1 <= i <= N:
                    rec[i] += 1
        z_rec = _z([float(rec[n]) for n in numbers])

        tot = sum(rec.values())
        if tot > 0:
            H = 0.0
            for n in numbers:
                p = rec[n] / float(tot)
                if p > 0.0:
                    H -= p * math.log(p)
            H /= math.log(N)
        else:
            H = 1.0
        ec = (H - ENT_LO) / (ENT_HI - ENT_LO)
        if ec < 0.0:
            ec = 0.0
        elif ec > 1.0:
            ec = 1.0

        # 4) Markov lag-1 conditionat pe ultima extragere
        z_mkv = [0.0] * N
        if L >= 2:
            last = [i for i in D[-1] if 1 <= i <= N]
            last_set = set(last)
            Trow = {}
            for sp, sn in zip(D, D[1:]):
                for i in sp:
                    if i in last_set:
                        row = Trow.get(i)
                        if row is None:
                            row = {}
                            Trow[i] = row
                        for j in sn:
                            if 1 <= j <= N:
                                row[j] = row.get(j, 0) + 1
            if last:
                cond = []
                for n in numbers:
                    acc = 0.0
                    for i in last:
                        row = Trow.get(i)
                        r = sum(row.values()) if row else 0
                        t = row.get(n, 0) if row else 0
                        acc += (t + ALPHA * P0) / (r + ALPHA)
                    cond.append(acc / len(last))
                z_mkv = _z(cond)

        # 5) tilt mean-reversion pe suma extragerilor
        if L >= 3:
            kk = K_SUM if K_SUM < L else L
            sbar = 0.0
            for s in D[L - kk:]:
                sbar += sum(s)
            sbar /= float(kk)
            dz = (MEAN_SUM - sbar) / SUM_STD
            if dz > 2.0:
                dz = 2.0
            elif dz < -2.0:
                dz = -2.0
            z_tilt = [((n - MEAN_N) / STD_N) * dz for n in numbers]
        else:
            z_tilt = [0.0] * N

        # 6) combinare: entropie mare => mai mult global, mai putin recent/Markov
        g_e = 0.5 + 0.5 * ec
        r_e = 1.0 - 0.40 * ec
        m_e = 1.0 - 0.55 * ec
        out = []
        for k in range(N):
            sc = (GLOB_W * g_e * z_glob[k]
                  + REC_W * r_e * z_rec[k]
                  + MKV_W * m_e * z_mkv[k]
                  + SUM_W * z_tilt[k])
            out.append((numbers[k], sc))
        out.sort(key=lambda t: (-t[1], t[0]))
        return [t[0] for t in out]

    return rank