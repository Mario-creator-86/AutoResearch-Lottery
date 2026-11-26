import math

NAME = "zseccyc3"
DESCRIPTION = ('Reversion ciclic pe 3 mecanisme: (1) SEC - cicluri de intarziere pe sectoare '
               'balansate (gap de sector vs gap-ul asteptat hipergeometric, compressie log), '
               '(2) GAP - intarziere individuala normalizata la 49/6 cu cap, (3) DEV - deviatia '
               'macro in z-score fata de frecventa medie istorica. Combinare liniara amestecata '
               'cu poarta tanh (GATE_MIX); SEC_SIGN/DEV_SIGN comuta reversion/momentum.')
SOURCE = 'model + 2026-09-17'

# Punctul implicit din grid; constantele se editeaza la nivel de plugin pentru grid-run:
SEC_SIZE = 6     # dimensiunea tinta a sectorului -> n_sec = round(49/SEC_SIZE)
SEC_SIGN = +1    # +1 = reversion (sectorul intarziat primeste bonus), -1 = momentum
DEV_SIGN = +1    # +1 = reversion (numerele sub-medie primesc bonus), -1 = momentum
W_SEC = 0.7
W_GAP = 0.5
W_DEV = 0.7
GAP_CAP = 2.5    # cap pe gap normalizat (multipli ai gap-ului asteptat)
GATE_MIX = 0.5   # 1.0 = doar poarta tanh, 0.0 = doar liniar


def create(config):
    total = config.number_range
    drawn = config.numbers_drawn

    # sectori balansati: (n-1)*n_sec // total
    n_sec = max(1, int(round(total / float(SEC_SIZE))))
    sector = [((n - 1) * n_sec) // total for n in range(1, total + 1)]
    sec_size = [0] * n_sec
    for s in sector:
        sec_size[s] += 1

    # gap-ul asteptat intre hit-uri de sector (geometric, P(miss)=C(49-k,6)/C(49,6))
    log_exp_sec = []
    for k in sec_size:
        p_miss = 1.0
        for i in range(drawn):
            p_miss *= (total - k - i) / float(total - i)
        p_hit = 1.0 - min(max(p_miss, 0.0), 1.0)
        e_gap = (1.0 / p_hit) if p_hit > 1e-12 else 1e12
        log_exp_sec.append(math.log1p(e_gap))

    exp_num_gap = total / float(drawn)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, total + 1))

        # orizontul ciclului = fereastra recenta (fallback: lungimea deque-ului)
        horizon = 0
        if isinstance(recent_window, int) and recent_window > 0:
            horizon = recent_window
        else:
            try:
                horizon = len(recent_deque)
            except Exception:
                horizon = 0

        # (1) SEC: ultimul hit per sector din last_seen (independent de ordinea deque-ului)
        sec_last = [-1] * n_sec
        for n in numbers:
            ls = last_seen.get(n)
            if ls is not None:
                s0 = sector[n - 1]
                if ls > sec_last[s0]:
                    sec_last[s0] = ls
        s_sec = {}
        for n in numbers:
            s0 = sector[n - 1]
            gap_s = draw_index - 1 - sec_last[s0]
            if gap_s < 0:
                gap_s = 0
            if horizon > 0 and gap_s > horizon:
                gap_s = horizon
            le = log_exp_sec[s0]
            ratio = (math.log1p(gap_s) / le) if le > 1e-12 else 0.0
            s_sec[n] = SEC_SIGN * (ratio - 1.0)

        # (2) GAP: intarziere individuala normalizata, cu cap
        s_gap = {}
        for n in numbers:
            ls = last_seen.get(n)
            gap = (draw_index - 1 - ls) if ls is not None else draw_index
            if gap < 0:
                gap = 0
            norm = gap / exp_num_gap
            if norm > GAP_CAP:
                norm = GAP_CAP
            s_gap[n] = norm - 1.0

        # (3) DEV: deviatia macro fata de frecventa medie (z-score istoric)
        vals = [counts.get(n, 0) for n in numbers]
        mean_c = sum(vals) / float(total)
        var_c = sum((v - mean_c) * (v - mean_c) for v in vals) / float(total)
        sd = math.sqrt(var_c)
        s_dev = {}
        for n, v in zip(numbers, vals):
            s_dev[n] = 0.0 if sd <= 1e-12 else DEV_SIGN * ((mean_c - v) / sd)

        # combinare: amestec liniar / poarta tanh
        score = {}
        for n in numbers:
            a = s_sec[n]
            b = s_gap[n]
            c = s_dev[n]
            lin = W_SEC * a + W_GAP * b + W_DEV * c
            gate = W_SEC * math.tanh(a) + W_GAP * math.tanh(b) + W_DEV * math.tanh(c)
            score[n] = GATE_MIX * gate + (1.0 - GATE_MIX) * lin

        # diagnostic pentru ablatie (se cere la next run: hits per-draw + rank._diag)
        rank._diag = {'sec': dict(s_sec), 'gap': dict(s_gap), 'dev': dict(s_dev),
                      'horizon': horizon, 'mean_c': mean_c, 'sd_c': sd}

        return sorted(numbers, key=lambda x: (-score[x], x))

    return rank