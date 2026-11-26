NAME = "zzsumres"
DESCRIPTION = 'Scor cross-scale pe trei scari: deviatia z a frecventei lifetime, z pe fereastra recenta cu decay exponential (DEC) si un termen de rezonanta neliniar (produsul zL*zW) care amplifica numerele confirmate pe ambele scari si depreciaza cele aflate in conflict (cald-life/recent-cold sau invers). Adauga steer al sumei tichetului catre media istorica (blend SUM_MIX intre media istorica si media ferestrei) si tilt de paritate spre compozitia odd/even tipica. Toate mecanismele sunt termeni aditivi cu ponderi expuse (WL/WR/WG_RES/MU_SUM/KPAR) si diag per-componenta (contributii la varianta scorului) pentru ablatie.'
SOURCE = 'model cantitativ + 2026-09-18'

import math

DIAG = {
    'orient': None,
    'orient_flips': 0,
    'sum_steer_margin': 0.0,
    'rec_mean_sum': 0.0,
    'd_par': 0.0,
    'contrib': {'zL': 0.0, 'zW': 0.0, 'res': 0.0, 'sum': 0.0, 'par': 0.0},
}

# hiperparametri expusi pentru grid-search / ablatie
WL = 0.6       # z pe frecventa lifetime
WR = 0.5       # z pe fereastra decay
DEC = 0.95     # decay fereastra (jumatate de viata ~13 extrageri)
WG_RES = 0.25  # rezonanta cross-scale zL*zW
MU_SUM = 0.02  # steer suma tichet
SUM_MIX = 0.7  # tinta suma = SUM_MIX*medie_istorica + (1-SUM_MIX)*medie_recenta
KPAR = 0.25    # tilt paritate odd/even
ZCLIP = 3.0    # clipping z-score


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    p = K / float(N)
    mu_sum = K * (N + 1) / 2.0
    var_n = (N * N - 1.0) / 12.0
    sd_n = math.sqrt(var_n)
    center_n = (N + 1) / 2.0
    var_sum = K * var_n * (N - K) / float(max(N - 1, 1))
    sd_sum = math.sqrt(var_sum) if var_sum > 0 else 1.0

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        T = float(max(draw_index, 1))
        mu_T = T * p
        sd_T = math.sqrt(mu_T * (1.0 - p) + 1e-12)
        L = len(recent_deque)

        # auto-detectia orientarii deque-ului: setul cel mai nou are recenzy medie minima
        def mean_rec(st):
            if not st:
                return float('inf')
            tot = 0.0
            for n in st:
                ls = last_seen.get(n)
                tot += (draw_index - ls) if ls is not None else float(draw_index)
            return tot / float(len(st))

        newest_last = True
        if L >= 2:
            newest_last = mean_rec(recent_deque[-1]) <= mean_rec(recent_deque[0])
        orient = 'newest_last' if newest_last else 'newest_first'
        if DIAG['orient'] is not None and DIAG['orient'] != orient:
            DIAG['orient_flips'] += 1
        DIAG['orient'] = orient

        seq = list(recent_deque)
        if newest_last and L >= 2:
            seq.reverse()  # seq[0] cel mai vechi .. seq[-1] cel mai nou

        w = dict.fromkeys(numbers, 0.0)
        Sw = 0.0
        win_sums = []
        odd_hits = 0
        n_trials = 0
        for i in range(L):
            aw = DEC ** (L - 1 - i)
            Sw += aw
            s = 0
            for n in seq[i]:
                if 1 <= n <= N:
                    w[n] += aw
                    s += n
                    if n & 1:
                        odd_hits += 1
                    n_trials += 1
            win_sums.append(s)

        sd_w = math.sqrt(Sw * p * (1.0 - p) + 1e-12)
        mu_w = Sw * p

        rec_mean_sum = (sum(win_sums) / float(L)) if L else mu_sum
        margin = SUM_MIX * (mu_sum - rec_mean_sum) / sd_sum
        DIAG['rec_mean_sum'] = rec_mean_sum
        DIAG['sum_steer_margin'] = margin

        if n_trials > 0:
            sd_o = math.sqrt(0.25 / n_trials)
            d_par = (0.5 - odd_hits / float(n_trials)) / sd_o
        else:
            d_par = 0.0
        DIAG['d_par'] = d_par

        csum = [0.0] * 5
        cross = [[0.0] * 5 for _ in range(5)]
        M = len(numbers)
        scores = {}
        for n in numbers:
            c = counts.get(n, 0)
            zl = (c - mu_T) / sd_T
            if zl > ZCLIP:
                zl = ZCLIP
            elif zl < -ZCLIP:
                zl = -ZCLIP
            zw = (w[n] - mu_w) / sd_w
            if zw > ZCLIP:
                zw = ZCLIP
            elif zw < -ZCLIP:
                zw = -ZCLIP
            v0 = WL * zl
            v1 = WR * zw
            v2 = WG_RES * (zl * zw)
            v3 = MU_SUM * margin * ((n - center_n) / sd_n)
            v4 = KPAR * d_par * (1.0 if n & 1 else -1.0)
            scores[n] = v0 + v1 + v2 + v3 + v4
            vals = (v0, v1, v2, v3, v4)
            for a in range(5):
                va = vals[a]
                csum[a] += va
                ra = cross[a]
                for b in range(a, 5):
                    ra[b] += va * vals[b]

        # diag: contributia fiecarei componente la varianta scorului final
        try:
            inv_m = 1.0 / M
            cov = [[0.0] * 5 for _ in range(5)]
            for a in range(5):
                ma = csum[a] * inv_m
                for b in range(a, 5):
                    mb = csum[b] * inv_m
                    cv = cross[a][b] * inv_m - ma * mb
                    cov[a][b] = cv
                    cov[b][a] = cv
            tot = 0.0
            for a in range(5):
                cab = cov[a]
                for b in range(5):
                    tot += cab[b]
            if tot > 1e-12:
                names = ('zL', 'zW', 'res', 'sum', 'par')
                out = {}
                for a in range(5):
                    row = 0.0
                    cab = cov[a]
                    for b in range(5):
                        row += cab[b]
                    out[names[a]] = row / tot
                DIAG['contrib'] = out
        except Exception:
            pass

        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank