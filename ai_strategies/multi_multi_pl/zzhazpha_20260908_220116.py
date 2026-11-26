NAME = "zzhazpha"
DESCRIPTION = 'Scor hibrid din 4 componente per numar: z-score al frecventei recente (momentum comprimat tanh, amestecat cu z-score pe istoricul total), EWMA decazuta a aparitiilor, hazard de intarziere overdue normalizat cu capul RCAP si deficit de sector pe 8 sectoare a cate 10; plus faza ciclica (perioda 7) si interactiune neliniara intarziere x sector. Garda de orientare newest_last este raportata in DIAG orient_flips; contributiile per componenta pentru ablatie sunt in DIAG last_contrib.'
SOURCE = 'model cantitativ propriu + 2026-09-08'

import math

# --- hiperparametri (intervalele pentru grid-search in PARAM_RANGES) ---
W_Z = 0.50    # z-score frecventa recenta + istoric, grid [0, 1]
W_EW = 0.40   # EWMA aparitii, grid [0, 0.8]
W_HAZ = 1.20  # hazard/overdue, grid [0, 1.5]
W_SEC = 0.60  # deficit de sector, grid [0, 1]
W_CYC = 0.0   # faza ciclica, grid [0, 3]
KAPPA = 0.20  # interactiune neliniara haz x sec, grid [0, 0.5]
GAMMA = 0.95  # decay EWMA, grid [0.90, 0.99]
RCAP = 6      # cap intarziere (in extrageri), grid [3, 8]
Z_BLEND = 0.35  # cat z din istoricul total intra in componenta z
PERIOD = 7
SECTOR_SIZE = 10

PARAM_RANGES = {
    'W_Z': (0.0, 1.0),
    'W_EW': (0.0, 0.8),
    'W_HAZ': (0.0, 1.5),
    'W_SEC': (0.0, 1.0),
    'W_CYC': (0.0, 3.0),
    'KAPPA': (0.0, 0.5),
    'GAMMA': (0.90, 0.99),
    'RCAP': (3, 8),
    'Z_BLEND': (0.0, 1.0),
}

DIAG = {'orient_flips': 0, 'orientation': 'newest_last', 'calls': 0, 'last_contrib': {}}


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    p = K / float(N)
    n_sec = (N + SECTOR_SIZE - 1) // SECTOR_SIZE

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        DIAG['calls'] += 1
        ids = list(range(1, N + 1))
        draws = list(recent_deque)
        Wd = len(draws)

        # --- garda orientare: presupunem newest_last (draws[-1] = cea mai noua) ---
        if Wd >= 4:
            def mgap(s):
                gs = []
                for n in s:
                    if n in last_seen:
                        g = draw_index - last_seen[n]
                        if g > 0:
                            gs.append(g)
                if len(gs) < 8:
                    return None
                return sum(gs) / float(len(gs))
            g_new = mgap(draws[-1])
            g_old = mgap(draws[0])
            if g_new is not None and g_old is not None:
                if g_new > g_old * 1.25 + 1.0:
                    draws.reverse()
                    DIAG['orient_flips'] += 1
            Wd = len(draws)

        # --- talie unica peste fereastra recenta ---
        win = [0] * (N + 1)
        ewa = [0.0] * (N + 1)
        same = [0] * (N + 1)
        sec_cnt = [0] * n_sec
        n_same = 0
        wgt = 1.0
        for a in range(1, Wd + 1):
            d = draws[Wd - a]
            ph_hit = (a % PERIOD == 0)
            if ph_hit:
                n_same += 1
            for n in d:
                if 1 <= n <= N:
                    win[n] += 1
                    ewa[n] += wgt
                    if ph_hit:
                        same[n] += 1
                    sec_cnt[(n - 1) // SECTOR_SIZE] += 1
            wgt *= GAMMA

        # --- denominators si asteptari ---
        exp_w = Wd * p
        v = math.sqrt(Wd * p * (1.0 - p))
        sd_w = v if v > 0 else 1.0
        if Wd > 0 and 0.0 < GAMMA < 1.0:
            sum_w = (1.0 - GAMMA ** Wd) / (1.0 - GAMMA)
        else:
            sum_w = float(Wd)
        exp_same = n_same * p
        v = math.sqrt(n_same * p * (1.0 - p))
        sd_same = v if v > 0 else 1.0
        exp_sec = []
        sd_sec = []
        for s in range(n_sec):
            sz = min(N, (s + 1) * SECTOR_SIZE) - s * SECTOR_SIZE
            q = sz / float(N)
            exp_sec.append(Wd * q)
            vv = math.sqrt(Wd * q * (1.0 - q))
            sd_sec.append(vv if vv > 0 else 1.0)

        # --- scoruri per componenta ---
        tz = [0.0] * (N + 1)
        ewn = [0.0] * (N + 1)
        hzv = [0.0] * (N + 1)
        sdv = [0.0] * (N + 1)
        phv = [0.0] * (N + 1)
        max_g = 2 * RCAP
        dt = draw_index * p * (1.0 - p)
        sdt = math.sqrt(dt) if dt > 0 else 1.0
        for n in ids:
            zr = (win[n] - exp_w) / sd_w
            zt = (counts.get(n, 0) - draw_index * p) / sdt
            tz[n] = math.tanh(zr + Z_BLEND * zt)
            ewn[n] = (ewa[n] / sum_w) if sum_w > 0 else 0.0
            if n in last_seen:
                g = draw_index - last_seen[n]
                if g < 1:
                    g = 1
            else:
                g = max_g + 1
            gc = g if g < max_g else max_g
            hzv[n] = (gc - RCAP) / float(RCAP)
            si = (n - 1) // SECTOR_SIZE
            sdv[n] = (exp_sec[si] - sec_cnt[si]) / sd_sec[si]
            phv[n] = (same[n] - exp_same) / sd_same

        # --- centrare (pt combinare echilibrata si termenul bilinear) ---
        def centered(arr):
            m = sum(arr[1:]) / float(N)
            return [0.0] + [x - m for x in arr[1:]]

        ewc = centered(ewn)
        hzc = centered(hzv)
        sdc = centered(sdv)
        phc = centered(phv)

        score = [0.0] * (N + 1)
        for n in ids:
            score[n] = (W_Z * tz[n] + W_EW * ewc[n] + W_HAZ * hzc[n]
                        + W_SEC * sdc[n] + W_CYC * phc[n]
                        + KAPPA * (hzc[n] * sdc[n]))

        # --- contributii per componenta (proxy de ablatie: pondere de varianta) ---
        def variance(vals):
            m = sum(vals) / float(len(vals))
            acc = 0.0
            for x in vals:
                acc += (x - m) * (x - m)
            return acc / float(len(vals))

        cvz = variance([W_Z * tz[n] for n in ids])
        cvm = variance([W_EW * ewc[n] for n in ids])
        cvh = variance([W_HAZ * hzc[n] for n in ids])
        cvs = variance([W_SEC * sdc[n] for n in ids])
        cvc = variance([W_CYC * phc[n] for n in ids])
        cvk = variance([KAPPA * hzc[n] * sdc[n] for n in ids])
        tot = cvz + cvm + cvh + cvs + cvc + cvk
        if tot > 0:
            DIAG['last_contrib'] = {
                'z': round(cvz / tot, 4),
                'mom': round(cvm / tot, 4),
                'haz': round(cvh / tot, 4),
                'sec': round(cvs / tot, 4),
                'cyc': round(cvc / tot, 4),
                'kappa': round(cvk / tot, 4),
            }

        return sorted(ids, key=lambda nn: (-score[nn], nn))

    return rank