import math

NAME = "zsecgapd"
DESCRIPTION = 'Cicluri de intarziere pe sectoare (datorie de sector in fereastra recenta) + reversion capulat pe intarzierea individuala fata de gap-ul asteptat N/D + deviatia z a frecventei istorice, combinate liniar; un gate de diversitate penalizeaza aglomerarea pe sectoare in top. Expune rank._diag (span per componente + top per draw) pentru ablate.'
SOURCE = 'glm-5.3-flash:cloud 2026-09-17'

# Parametri (tinti de grid): SEC_SIZE{4,6,8} x SEC_SIGN x DEV_SIGN x
# W_SEC{0.35,0.7,1.05} x W_GAP{0.3,0.5,0.8} x W_DEV{0,0.4,0.8} x
# GAP_CAP{1.5,2.5,4.0} x GATE_MIX{0.3,0.5,0.7}
SEC_SIZE = 6
SEC_SIGN = 1      # +1 = reversion (sector datornic boostat); -1 = momentum
DEV_SIGN = 1      # +1 = reversion (numar sub-frecventa boostat); -1 = momentum
W_SEC = 0.7
W_GAP = 0.5
W_DEV = 0.4
GAP_CAP = 2.5
GATE_MIX = 0.5


def create(config):
    # config: LotteryConfig — folosim doar .number_range si .numbers_drawn
    N = config.number_range
    D = config.numbers_drawn
    numbers = list(range(1, N + 1))

    sec_len = -(-N // SEC_SIZE)
    S = (N + sec_len - 1) // sec_len
    sector_idx = [0] * (N + 1)
    sec_members = [[] for _ in range(S)]
    for n in numbers:
        s = (n - 1) // sec_len
        sector_idx[n] = s
        sec_members[s].append(n)
    sec_size = [len(m) for m in sec_members]

    p_hit = float(D) / float(N)

    cum = {'calls': 0,
           'sec_lo': math.inf, 'sec_hi': -math.inf,
           'gap_lo': math.inf, 'gap_hi': -math.inf,
           'dev_lo': math.inf, 'dev_hi': -math.inf}

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # counts: dict {nr: aparitii totale}; recent_deque: deque de seturi;
        # last_seen: dict {nr: index ultima aparitie}; draw_index: int;
        # recent_window: int. Returneaza permutare completa 1..N.
        L = len(recent_deque)

        # (1) SEC: hits pe sector in fereastra recenta, z-like
        sec_hits = [0] * S
        for draw in recent_deque:
            for n in draw:
                sec_hits[sector_idx[n]] += 1

        w_sec = W_SEC * (1.0 if SEC_SIGN >= 0 else -1.0)
        w_dev = W_DEV * (1.0 if DEV_SIGN >= 0 else -1.0)

        sec_term = [0.0] * (N + 1)
        if L > 0:
            for s in range(S):
                exp_s = L * sec_size[s] * p_hit
                debt = (exp_s - sec_hits[s]) / math.sqrt(exp_s + 1.0)
                base = w_sec * debt
                for n in sec_members[s]:
                    sec_term[n] = base

        # (2) GAP: intarziere individuala / gap asteptat, capulata, centrata la 1
        gap_term = [0.0] * (N + 1)
        for n in numbers:
            ls = last_seen.get(n)
            gap = (draw_index - ls) if ls is not None else (draw_index + 1)
            if gap < 0:
                gap = 0
            r = gap * p_hit
            if r > GAP_CAP:
                r = GAP_CAP
            gap_term[n] = W_GAP * (r - 1.0)

        # (3) DEV: deviatia z fata de frecventa asteptata pe istoric
        dev_term = [0.0] * (N + 1)
        T = float(draw_index)
        if T >= 1.0:
            exp_tot = T * p_hit
            std_tot = math.sqrt(T * p_hit * (1.0 - p_hit)) + 1e-9
            for n in numbers:
                dev_term[n] = w_dev * (exp_tot - counts.get(n, 0)) / std_tot

        raw = [0.0] * (N + 1)
        ssum = 0.0
        for n in numbers:
            raw[n] = sec_term[n] + gap_term[n] + dev_term[n]
            ssum += raw[n]
        mean_r = ssum / N
        vsum = 0.0
        for n in numbers:
            d = raw[n] - mean_r
            vsum += d * d
        std_r = math.sqrt(vsum / N) + 1e-12

        # GATE: diversitate — al k-lea numar din acelasi sector (in ordinea raw)
        # primeste penalizare GATE_MIX * std(raw) * k; reordoneaza real
        order_raw = sorted(numbers, key=lambda n: -raw[n])
        load = [0] * S
        final = [0.0] * (N + 1)
        for n in order_raw:
            s = sector_idx[n]
            final[n] = raw[n] - GATE_MIX * std_r * load[s]
            load[s] += 1

        result = sorted(numbers, key=lambda n: (-final[n], n))

        # diag pentru ablate: span max-min per componenta + top per draw
        sp_sec = max(sec_term[1:]) - min(sec_term[1:])
        sp_gap = max(gap_term[1:]) - min(gap_term[1:])
        sp_dev = max(dev_term[1:]) - min(dev_term[1:])
        cum['calls'] += 1
        if sp_sec < cum['sec_lo']:
            cum['sec_lo'] = sp_sec
        if sp_sec > cum['sec_hi']:
            cum['sec_hi'] = sp_sec
        if sp_gap < cum['gap_lo']:
            cum['gap_lo'] = sp_gap
        if sp_gap > cum['gap_hi']:
            cum['gap_hi'] = sp_gap
        if sp_dev < cum['dev_lo']:
            cum['dev_lo'] = sp_dev
        if sp_dev > cum['dev_hi']:
            cum['dev_hi'] = sp_dev

        top = tuple(result[:D])
        rank._diag = {
            'strategy': NAME,
            'call': cum['calls'],
            'draw_index': draw_index,
            'window_len': L,
            'recent_window': recent_window,
            'span_call': {'sec': sp_sec, 'gap': sp_gap, 'dev': sp_dev},
            'span_cum': {'sec': cum['sec_hi'] - cum['sec_lo'],
                         'gap': cum['gap_hi'] - cum['gap_lo'],
                         'dev': cum['dev_hi'] - cum['dev_lo']},
            'params': {'SEC_SIZE': SEC_SIZE, 'SEC_SIGN': SEC_SIGN,
                       'DEV_SIGN': DEV_SIGN, 'W_SEC': W_SEC, 'W_GAP': W_GAP,
                       'W_DEV': W_DEV, 'GAP_CAP': GAP_CAP, 'GATE_MIX': GATE_MIX},
            'sector_load_top': tuple(sector_idx[n] for n in top),
            'top': top,
        }
        return result

    return rank