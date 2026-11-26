import math

NAME = "zsecdevg"
DESCRIPTION = 'Reversie multi-nivel: ciclu de sector pe deficit EWMA (sectoare de SECTOR_SIZE numere, z-score hipergeometric), reversie macro EWMA per numar fata de rata asteptata numbers_drawn/number_range, ciclu de gap capped, plus ecou de perechi co-aparute (centralitate z-normalizata) si un mic termen de reversie lifetime. Scor liniar ponderat, sortare determinista completa 1..24.'
SOURCE = 'model 2026-09-17'

SEC_SIZE_DEFAULT = 6
SEC_DECAY = 0.94
W_SEC = 0.70
DEV_SIGN = -1
NUM_DECAY = 0.94
W_DEV = 0.40
W_GAP = 0.50
GAP_CAP = 8
W_ECHO = 0.30
COV_WIN = 36
W_LIFE = 0.15


def create(config):
    N = int(config.number_range)
    n_drawn = int(config.numbers_drawn)
    p = n_drawn / N
    sec_size = int(getattr(config, 'sector_size', 0) or SEC_SIZE_DEFAULT)
    if sec_size < 1:
        sec_size = SEC_SIZE_DEFAULT
    if sec_size > N:
        sec_size = N
    n_sec = (N + sec_size - 1) // sec_size
    sec_of = [0] * (N + 1)
    for n in range(1, N + 1):
        sec_of[n] = (n - 1) // sec_size
    p_sec = sec_size / N
    var_sec_draw = n_drawn * p_sec * (1.0 - p_sec) * (N - n_drawn) / max(1.0, float(N - 1))
    var_num_draw = p * (1.0 - p)
    cap = float(GAP_CAP) if GAP_CAP > 0 else 1.0
    d2s = SEC_DECAY * SEC_DECAY
    d2n = NUM_DECAY * NUM_DECAY

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        hist = list(recent_deque)
        L = len(hist)
        if L > 0:
            g1s = (1.0 - SEC_DECAY ** L) / (1.0 - SEC_DECAY)
            g2s = (1.0 - d2s ** L) / (1.0 - d2s)
            g1n = (1.0 - NUM_DECAY ** L) / (1.0 - NUM_DECAY)
            g2n = (1.0 - d2n ** L) / (1.0 - d2n)
        else:
            g1s = g2s = g1n = g2n = 0.0
        sec_ew = [0.0] * n_sec
        num_ew = [0.0] * (N + 1)
        w_s = 1.0
        w_n = 1.0
        for draw in reversed(hist):
            for x in draw:
                if 1 <= x <= N:
                    s = sec_of[x]
                    sec_ew[s] += w_s
                    num_ew[x] += w_n
            w_s *= SEC_DECAY
            w_n *= NUM_DECAY
        exp_s = n_drawn * p_sec * g1s
        sd_s = math.sqrt(var_sec_draw * g2s)
        exp_n = p * g1n
        sd_n = math.sqrt(var_num_draw * g2n)
        pair = {}
        if L > 1 and COV_WIN > 0:
            pw = hist[-COV_WIN:]
            for draw in pw:
                dl = [x for x in draw if 1 <= x <= N]
                dl.sort()
                lp = len(dl)
                for i in range(lp):
                    a = dl[i]
                    ai = a * 32
                    for j in range(i + 1, lp):
                        k = ai + dl[j]
                        pair[k] = pair.get(k, 0.0) + 1.0
        echo = [0.0] * (N + 1)
        for k, c in pair.items():
            a = k >> 5
            b = k & 31
            echo[a] += c
            echo[b] += c
        vals = [echo[n] for n in numbers]
        m_e = sum(vals) / N
        var_e = 0.0
        for v in vals:
            dv = v - m_e
            var_e += dv * dv
        sd_e = math.sqrt(var_e / N)
        di = float(draw_index)
        sd_life = math.sqrt(max(1.0, di * p * (1.0 - p)))
        exp_life = di * p
        out = []
        for n in numbers:
            s_idx = sec_of[n]
            sz = (sec_ew[s_idx] - exp_s) / sd_s if sd_s > 1e-9 else 0.0
            nz = (num_ew[n] - exp_n) / sd_n if sd_n > 1e-9 else 0.0
            ls = last_seen.get(n)
            g = di - ls if ls is not None else cap
            if g < 0.0:
                g = 0.0
            elif g > cap:
                g = cap
            ez = (echo[n] - m_e) / sd_e if sd_e > 1e-9 else 0.0
            lz = (counts.get(n, 0) - exp_life) / sd_life
            sc = (W_SEC * (-sz)
                  + W_DEV * DEV_SIGN * nz
                  + W_GAP * (g / cap)
                  + W_ECHO * ez
                  + W_LIFE * (-lz))
            out.append((-sc, n))
        out.sort()
        return [n for _, n in out]

    return rank