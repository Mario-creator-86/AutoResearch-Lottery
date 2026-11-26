import math

NAME = "zzdiffga"
DESCRIPTION = 'Difuzaie pe graf de co-occerenta recenta (vecinii numerelor calde primesc lift), plus EWMA half-life 12, hazard de gap raportat la 9, cicluri de deficit pe 9 sectoare x 5 numere, ancora in frecventa de lunga durata, toate modulate adaptiv de o poarta de dispersie a ferestrei.'
SOURCE = 'glm-5.3-flash quant research | 2026-09-08'


def create(config):
    N = config.number_range
    KDRAW = config.numbers_drawn
    P_KD = float(KDRAW) / float(N)
    GAP_EXPECT = 1.0 / P_KD
    GAP_CAP = 6.0 * GAP_EXPECT
    HALF_LIFE = 12.0
    LAM = 0.5 ** (1.0 / HALF_LIFE)
    PAIR_WIN = 40
    EW_WIN = 120
    SEC_SIZE = 5
    N_SEC = N // SEC_SIZE
    SEC_WIN = 15
    W_TF = 1.0
    W_EW = 1.0
    W_PAIR = 1.15
    W_GAP = 0.7
    W_SEC = 0.6
    W_INT = 0.3

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        draws = list(recent_deque)
        Wd = len(draws)
        cget = counts.get
        if Wd == 0:
            return sorted(numbers, key=lambda n: -cget(n, 0))

        # frecventa de lunga durata, standardizata
        tot = 0
        for n in numbers:
            tot += cget(n, 0)
        mean_c = tot / float(N)
        var_c = 0.0
        for n in numbers:
            d = cget(n, 0) - mean_c
            var_c += d * d
        std_c = math.sqrt(var_c / N)
        if std_c < 1e-12:
            std_c = 1.0

        # numarari pe fereastra + poarta de dispersie (std real / std binomial)
        wcount = [0] * (N + 1)
        for s in draws:
            for n in s:
                wcount[n] += 1
        mean_w = KDRAW * Wd / float(N)
        std_bin = math.sqrt(Wd * P_KD * (1.0 - P_KD))
        var_w = 0.0
        for n in numbers:
            d = wcount[n] - mean_w
            var_w += d * d
        disp = math.sqrt(var_w / N) / std_bin
        gate = disp
        if gate < 0.6:
            gate = 0.6
        elif gate > 1.6:
            gate = 1.6
        gate = 0.5 + 0.5 * gate

        # EWMA pe ultimele EW_WIN extrageri (ultimul element = cel mai recent)
        ew = [0.0] * (N + 1)
        eww = EW_WIN if Wd > EW_WIN else Wd
        w = 1.0
        wsum = 0.0
        for t in range(Wd - 1, Wd - 1 - eww, -1):
            for n in draws[t]:
                ew[n] += w
            wsum += w
            w *= LAM
        mean_e = KDRAW * wsum / float(N)
        var_e = 0.0
        for n in numbers:
            d = ew[n] - mean_e
            var_e += d * d
        std_e = math.sqrt(var_e / N)
        if std_e < 1e-12:
            std_e = 1.0

        # graful de co-occerenta pe ultimele PAIR_WIN extrageri
        pw = PAIR_WIN if Wd > PAIR_WIN else Wd
        nbr = [None] * (N + 1)
        for t in range(Wd - pw, Wd):
            ls = list(draws[t])
            L = len(ls)
            for i in range(L):
                a = ls[i]
                ra = nbr[a]
                if ra is None:
                    ra = {}
                    nbr[a] = ra
                for j in range(i + 1, L):
                    b = ls[j]
                    rb = nbr[b]
                    if rb is None:
                        rb = {}
                        nbr[b] = rb
                    ra[b] = ra.get(b, 0) + 1
                    rb[a] = rb.get(a, 0) + 1

        # difuzie: pd_n = suma A[n][m] * deviatia EWMA a vecinului m
        pd = [0.0] * (N + 1)
        for n in numbers:
            ra = nbr[n]
            if ra:
                acc = 0.0
                for m, c in ra.items():
                    acc += c * (ew[m] - mean_e)
                pd[n] = acc
        mean_p = 0.0
        for n in numbers:
            mean_p += pd[n]
        mean_p /= N
        var_p = 0.0
        for n in numbers:
            d = pd[n] - mean_p
            var_p += d * d
        std_p = math.sqrt(var_p / N)
        if std_p < 1e-12:
            std_p = 1.0

        # hazard de gap (numerele niciodata vazute primesc gap = draw_index+1, caplit)
        gh = [0.0] * (N + 1)
        lget = last_seen.get
        for n in numbers:
            li = lget(n, -1)
            gap = draw_index - li
            if gap < 0:
                gap = 0
            elif gap > GAP_CAP:
                gap = GAP_CAP
            gh[n] = gap
        mean_g = 0.0
        for n in numbers:
            mean_g += gh[n]
        mean_g /= N
        var_g = 0.0
        for n in numbers:
            d = gh[n] - mean_g
            var_g += d * d
        std_g = math.sqrt(var_g / N)
        if std_g < 1e-12:
            std_g = 1.0

        # cicluri de deficit pe sectoare (9 sectoare x 5 numere)
        sw = SEC_WIN if Wd > SEC_WIN else Wd
        sec_hits = [0] * N_SEC
        for t in range(Wd - sw, Wd):
            for n in draws[t]:
                sec_hits[(n - 1) // SEC_SIZE] += 1
        exp_sec = KDRAW * sw / float(N_SEC)
        sden = math.sqrt(exp_sec + 1.0)
        sec_sc = [0.0] * N_SEC
        for k in range(N_SEC):
            d = (exp_sec - sec_hits[k]) / sden
            if d > 2.0:
                d = 2.0
            elif d < -2.0:
                d = -2.0
            sec_sc[k] = d

        out = []
        for n in numbers:
            zc = (cget(n, 0) - mean_c) / std_c
            ze = (ew[n] - mean_e) / std_e
            zp = (pd[n] - mean_p) / std_p
            zg = (gh[n] - mean_g) / std_g
            sec = sec_sc[(n - 1) // SEC_SIZE]
            sc = W_TF * zc + gate * (W_EW * ze + W_PAIR * zp + W_GAP * zg + W_SEC * sec + W_INT * ze * zg)
            out.append((sc, n))
        out.sort(key=lambda t: (-t[0], t[1]))
        return [t[1] for t in out]

    return rank