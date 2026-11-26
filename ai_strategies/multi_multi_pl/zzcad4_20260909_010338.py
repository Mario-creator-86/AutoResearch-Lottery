import math

NAME = "zzcad4"
DESCRIPTION = ('Combinatie liniara de 4 semnale z-normalizate: (1) afinitate de perechi co-aparute '
               'ponderata EWMA (GAMMA_P) fata de ancorele din ultimele K_TAIL extrageri, '
               '(2) deviatia binomiala a frecventei recente fata de asteptat 20/80, '
               '(3) kernel gaussian de recency pe ultima aparitie (SIGMA_POS), '
               '(4) cadenta due-now din intervalele medii de reaparitie din fereastra recenta. '
               'Ponderile W_PAIR/W_DEV/W_POS/W_CAD controleaza contributia fiecarui semnal, '
               'iar DIAG[orient_flips] verifica orientarea newest_last a deque-ului la fiecare apel.')
SOURCE = 'model + 2026-09-08'

# Grid-search ranges (pastrate explicit pentru tuner):
#   W_PAIR    in [0, 1.5]
#   W_DEV     in [-0.8, 1.0]
#   W_POS     in [0, 1]
#   W_CAD     in [0, 1]
#   GAMMA_P   in [0.90, 0.99]
#   K_TAIL    in {1, 2, 3, 5}
#   SIGMA_POS in [4, 12]
PARAMS = {
    'W_PAIR': 0.60,
    'W_DEV': 0.30,
    'W_POS': 0.45,
    'W_CAD': 0.30,
    'GAMMA_P': 0.97,
    'K_TAIL': 3,
    'SIGMA_POS': 8.0,
}

DIAG = {
    'orient_flips': 0,
    'orient_checks': 0,
    'empty_windows': 0,
}

CAD_WIDTH = 0.75


def _zscore(d, numbers):
    m = 0.0
    for n in numbers:
        m += d[n]
    m /= len(numbers)
    v = 0.0
    for n in numbers:
        x = d[n] - m
        v += x * x
    sd = (v / len(numbers)) ** 0.5
    if sd < 1e-12:
        return dict.fromkeys(numbers, 0.0)
    inv = 1.0 / sd
    return {n: (d[n] - m) * inv for n in numbers}


def create(config):
    total_numbers = config.number_range
    drawn = config.numbers_drawn
    p_hit = float(drawn) / float(total_numbers)
    exp_fn = math.exp

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, total_numbers + 1))
        L = len(recent_deque)
        if L == 0:
            DIAG['empty_windows'] += 1
            return sorted(numbers, key=lambda n: (-counts.get(n, 0), n))

        # --- verificare orientare deque (newest_last): deque[-1] = extragerea cea mai noua,
        # deci mediile last_seen ale numerelor sale trebuie sa fie maxime fata de deque[0].
        if L >= 2:
            DIAG['orient_checks'] += 1
            s_new = recent_deque[-1]
            s_old = recent_deque[0]
            if s_new and s_old:
                m_new = 0.0
                for n in s_new:
                    m_new += last_seen.get(n, -10**9)
                m_new /= len(s_new)
                m_old = 0.0
                for n in s_old:
                    m_old += last_seen.get(n, -10**9)
                m_old /= len(s_old)
                if m_new < m_old:
                    DIAG['orient_flips'] += 1

        # --- DEV: deviatia binomiala a count-urilor din fereastra recenta ---
        k = dict.fromkeys(numbers, 0)
        for s in recent_deque:
            for n in s:
                k[n] += 1
        exp_c = L * p_hit
        sd_c = (L * p_hit * (1.0 - p_hit)) ** 0.5
        inv_sd = 1.0 / sd_c if sd_c > 1e-12 else 0.0
        dev = {n: (k[n] - exp_c) * inv_sd for n in numbers}

        # --- PAIR: co-aparitie EWMA cu ancore din ultimele K_TAIL extrageri ---
        kp = min(PARAMS['K_TAIL'], L)
        anchor = set()
        for i in range(L - kp, L):
            anchor |= recent_deque[i]
        gamma = PARAMS['GAMMA_P']
        weights = [0.0] * L
        w = 1.0
        for i in range(L - 1, -1, -1):
            weights[i] = w
            w *= gamma
        inter = [len(recent_deque[i] & anchor) for i in range(L)]
        pair = dict.fromkeys(numbers, 0.0)
        for i in range(L):
            c = inter[i]
            if c == 0:
                continue
            wi = weights[i]
            si = recent_deque[i]
            for n in si:
                if n in anchor:
                    pair[n] += wi * (c - 1)
                else:
                    pair[n] += wi * c

        # --- POS: kernel de recency pe ultima aparitie ---
        sig = PARAMS['SIGMA_POS']
        pos = {}
        for n in numbers:
            t = last_seen.get(n)
            if t is None:
                pos[n] = 0.0
            else:
                gap = draw_index - t
                if gap < 0:
                    gap = 0
                pos[n] = exp_fn(-gap / sig)

        # --- CAD: cadenta due-now din intervalele de reaparitie din fereastra ---
        base = draw_index - L
        posin = {}
        for i in range(L):
            t_abs = base + i
            for n in recent_deque[i]:
                lst = posin.get(n)
                if lst is None:
                    posin[n] = [t_abs]
                else:
                    lst.append(t_abs)
        cad = dict.fromkeys(numbers, 0.0)
        for n, ts in posin.items():
            if len(ts) >= 2:
                gsum = 0.0
                for j in range(len(ts) - 1):
                    gsum += ts[j + 1] - ts[j]
                mu = gsum / (len(ts) - 1)
                if mu > 0.5:
                    due = (draw_index - ts[-1]) / mu
                    d = (due - 1.0) / CAD_WIDTH
                    if d < -6.0:
                        d = -6.0
                    elif d > 6.0:
                        d = 6.0
                    cad[n] = exp_fn(-0.5 * d * d)

        # --- combinare liniara pe componente z-normalizate ---
        zp = _zscore(pair, numbers)
        zd = _zscore(dev, numbers)
        zo = _zscore(pos, numbers)
        zc = _zscore(cad, numbers)
        w_pair = PARAMS['W_PAIR']
        w_dev = PARAMS['W_DEV']
        w_pos = PARAMS['W_POS']
        w_cad = PARAMS['W_CAD']
        tot = {}
        for n in numbers:
            tot[n] = (w_pair * zp[n] + w_dev * zd[n]
                      + w_pos * zo[n] + w_cad * zc[n])
        return sorted(numbers, key=lambda n: (-tot[n], -counts.get(n, 0), n))

    return rank