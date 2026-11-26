import math

NAME = "zzsectro"
DESCRIPTION = (
    'Reversion cu shrinkage neliniar pentru 6/49: scorul combina un z-score hipergeometric '
    'al frecventei totale (tanh-cap; numerele sub-reprezentate fata de medie urca in clasament), '
    'un contrast de rotatie pe fereastra recenta si un ciclu de intarziere pe 7 sectoare a cate '
    '7 numere; amplificarea sectoriala e modulata de deficitul de entropie Shannon al distributiei '
    'sectoriale recente (cluster recent pe putine sectoare => sector-overdue cantareste mai mult).'
)
SOURCE = 'model quant-strategist + 2026-09-18'


def create(config):
    P = config.number_range
    W = config.numbers_drawn

    # sectoare egale: 49 -> 7 sectoare x 7 numere
    sec_size = 7 if P % 7 == 0 else max(1, int(round(P / 7.0)))
    n_sec = max(1, int(math.ceil(P / float(sec_size))))
    sector_of = [0] * (P + 1)
    for n in range(1, P + 1):
        sector_of[n] = min((n - 1) // sec_size, n_sec - 1)

    p_num = W / float(P)
    var_num_per_draw = p_num * (1.0 - p_num)

    # momente hipergeometrice per sector (per extragere)
    sec_mean_draw = [0.0] * n_sec
    sec_var_draw = [0.0] * n_sec
    for k in range(n_sec):
        s_k = sum(1 for n in range(1, P + 1) if sector_of[n] == k) / float(P)
        sec_mean_draw[k] = W * s_k
        sec_var_draw[k] = (W * s_k * (1.0 - s_k) * (P - W) / (P - 1.0)) if P > 1 else 1e-9

    ZS_LONG = 1.6    # scala tanh pentru z pe termen lung
    ZS_REC = 1.3     # scala tanh pentru contrastul recent
    ZS_SEC = 2.0     # scala tanh pentru z sectorial
    W_LONG = 1.0     # pondere reversion long-run
    W_REC = 0.85     # pondere rotatie fereastra recenta
    W_SEC = 0.8      # pondere de baza sector-overdue
    GATE_GAIN = 5.0  # amplificarea deficitului de entropie
    GATE_SEC_MULT = 1.5

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, P + 1))

        # statistici pe fereastra recenta: un singur pas
        rec_cnt = {}
        sec_rec = [0] * n_sec
        rec_hits = 0
        for window_set in recent_deque:
            for n in window_set:
                rec_cnt[n] = rec_cnt.get(n, 0) + 1
                if 1 <= n <= P:
                    sec_rec[sector_of[n]] += 1
                    rec_hits += 1

        total_hits = 0
        sec_tot = [0] * n_sec
        for n in numbers:
            c = counts.get(n, 0)
            total_hits += c
            sec_tot[sector_of[n]] += c
        T = total_hits / float(W) if W > 0 else 0.0
        if T <= 0.0:
            T = 1.0

        mu_long = T * p_num
        sd_long = math.sqrt(max(T * var_num_per_draw, 1e-12))

        L = len(recent_deque)
        mu_rec = L * p_num
        sd_rec = math.sqrt(max(L * var_num_per_draw, 1e-12))

        # gate de entropie Shannon a distributiei sectoriale recente
        gate = 0.0
        if L >= 3 and rec_hits > 0 and n_sec > 1:
            H = 0.0
            for k in range(n_sec):
                pk = sec_rec[k] / float(rec_hits)
                if pk > 0.0:
                    H -= pk * math.log(pk)
            Hmax = math.log(n_sec)
            gate = (Hmax - H) / Hmax * GATE_GAIN
            if gate < 0.0:
                gate = 0.0
            elif gate > 1.0:
                gate = 1.0

        # z-score sectorial pe termen lung (ciclu de intarziere pe sectoare)
        z_sec = [0.0] * n_sec
        for k in range(n_sec):
            mu_k = T * sec_mean_draw[k]
            sd_k = math.sqrt(max(T * sec_var_draw[k], 1e-12))
            z_sec[k] = (sec_tot[k] - mu_k) / sd_k

        w_sec_eff = W_SEC * (1.0 + GATE_SEC_MULT * gate)
        scores = {}
        for n in numbers:
            c = counts.get(n, 0)
            s_long = -math.tanh((c - mu_long) / sd_long / ZS_LONG)
            if L > 0:
                rc = rec_cnt.get(n, 0)
                s_rec = -math.tanh((rc - mu_rec) / sd_rec / ZS_REC)
            else:
                s_rec = 0.0
            s_sec = -math.tanh(z_sec[sector_of[n]] / ZS_SEC)
            scores[n] = W_LONG * s_long + W_REC * s_rec + w_sec_eff * s_sec

        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank