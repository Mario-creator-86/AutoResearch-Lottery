import math

NAME = "zsecycre"
DESCRIPTION = ('Strategie contrarian pe termen scurt cu 3 mecanisme: (1) sector-cycle - z-score pe '
               'acoperirea fiecarui sector in fereastra recenta, semn reversion; (2) deviatie macro '
               'per numar - z pe aparitiile recente fata de asteptat, semn reversion; (3) gap/overdue '
               'normalizat (raport la gap-ul asteptat 1/p), ponderat de entropia distributiei pe '
               'sectoare (entropy-gating aplicat doar pe gap).')
SOURCE = 'model + 2026-09-17'

PARAMS = {
    'sector_size': 8,     # SEC_SIZE, expus in config
    'w_sector': 0.50,
    'sector_sign': -1.0,  # -1 = reversion: sectoarele sub-hit urca in rank
    'w_dev': 0.40,
    'dev_sign': -1.0,     # -1 = reversion: numerele sub-hit recent urca in rank
    'w_gap': 0.60,
    'gap_cap': 2.5,       # plafon pentru gap / gap_asteptat
    'gate_mix': 0.50,     # ponderea minima a gap-ului la entropie maxima
    'z_clip': 3.0,
}


def _resolve(config):
    p = dict(PARAMS)
    if config is not None:
        for k in p:
            v = getattr(config, k, None)
            if v is not None:
                p[k] = v
    return p


def _clip(x, lim):
    return -lim if x < -lim else (lim if x > lim else x)


def _core(recent_deque, last_seen, draw_index, number_range, numbers_drawn, p):
    n_win = len(recent_deque)
    p_num = numbers_drawn / float(number_range)
    exp_gap = 1.0 / p_num
    sec = max(int(p['sector_size']), 1)
    n_sec = (number_range + sec - 1) // sec

    rc = dict.fromkeys(range(1, number_range + 1), 0)
    sec_tot = [0] * n_sec
    for s in recent_deque:
        for x in s:
            if 1 <= x <= number_range:
                rc[x] += 1
                sec_tot[min((x - 1) // sec, n_sec - 1)] += 1
    tot_hits = sum(sec_tot)

    if tot_hits > 0 and n_sec > 1:
        h = 0.0
        for c in sec_tot:
            if c > 0:
                pr = c / float(tot_hits)
                h -= pr * math.log(pr)
        h_norm = h / math.log(n_sec)
    else:
        h_norm = 1.0
    gate = p['gate_mix'] + (1.0 - p['gate_mix']) * (1.0 - h_norm)

    z_sec = [0.0] * n_sec
    for i in range(n_sec):
        sec_len = sec if i < n_sec - 1 else max(number_range - sec * (n_sec - 1), 1)
        e = n_win * sec_len * p_num
        sd = math.sqrt(max(e * (1.0 - p_num), 1e-9))
        z_sec[i] = _clip((sec_tot[i] - e) / sd, p['z_clip'])

    exp_n = n_win * p_num
    sd_n = math.sqrt(max(exp_n * (1.0 - p_num), 1e-9))

    scores = {}
    devs = []
    gaps = []
    for n in range(1, number_range + 1):
        z_dev = _clip((rc[n] - exp_n) / sd_n, p['z_clip']) if n_win > 0 else 0.0
        last = last_seen.get(n)
        gap = (draw_index - last) if last is not None else (draw_index + 1)
        ratio = min(gap / exp_gap, p['gap_cap'])
        si = min((n - 1) // sec, n_sec - 1)
        scores[n] = (p['w_sector'] * p['sector_sign'] * z_sec[si]
                     + p['w_dev'] * p['dev_sign'] * z_dev
                     + p['w_gap'] * gate * ratio)
        devs.append(z_dev)
        gaps.append(ratio)

    diag = {
        'entropy': round(h_norm, 4),
        'gate': round(gate, 4),
        'sector_z': [round(z, 3) for z in z_sec],
        'sector_z_span': round(max(z_sec) - min(z_sec), 4),
        'dev_z_span': round(max(devs) - min(devs), 4),
        'gap_span': round(max(gaps) - min(gaps), 4),
    }
    return scores, diag


def create(config):
    number_range = config.number_range
    numbers_drawn = config.numbers_drawn
    p = _resolve(config)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        scores, _ = _core(recent_deque, last_seen, draw_index,
                          number_range, numbers_drawn, p)
        numbers = list(range(1, number_range + 1))
        return sorted(numbers, key=lambda n: (-scores[n], -counts.get(n, 0), n))

    return rank


def diagnose(counts, recent_deque, last_seen, draw_index, recent_window, config=None):
    # DIAG pentru ablatie: verifica span>0.01 per componenta inainte de eliminare.
    nr = config.number_range if config is not None else 40
    nd = config.numbers_drawn if config is not None else 6
    _, diag = _core(recent_deque, last_seen, draw_index, nr, nd, _resolve(config))
    return diag