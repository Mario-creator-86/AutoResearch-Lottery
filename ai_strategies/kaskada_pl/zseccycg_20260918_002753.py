import math
from itertools import islice

NAME = "zseccycg"
DESCRIPTION = 'Scor aditiv pe 3 componente: (1) cicluri de sector - deficit z al sectorului de SEC_SIZE numere fata de aparitiile asteptate, mix GATE_MIX intre orizont lung si fereastra scurta; (2) gap per numar - intarziere fata de ultima aparitie normalizata la 1/p si capata la GAP_CAP; (3) deviatia macro de frecventa - z-score mix GATE_MIX lung/scurt. Semnele SEC_SIGN/DEV_SIGN comuta reversion/momentum; W_GAP=0 daca GAP iese redundant cu SEC+DEV.'
SOURCE = 'model + 2026-09-17'

# Grid planificat: SEC_SIZE{4,6,8} x SEC_SIGN{+-1} x DEV_SIGN{+-1} x W_SEC{0.35,0.7,1.05} x W_GAP{0.3,0.5,0.8} x W_DEV{0,0.4,0.8} x GAP_CAP{1.5,2.5,4.0} x GATE_MIX{0.3,0.5,0.7}
SEC_SIZE = 6
SEC_SIGN = 1
DEV_SIGN = 1
W_SEC = 0.7
W_GAP = 0.5
W_DEV = 0.4
GAP_CAP = 2.5
GATE_MIX = 0.5
SHORT_WIN = 24


def create(config):
    n_max = config.number_range
    k = config.numbers_drawn
    p = (k / float(n_max)) if n_max > 0 else 0.5
    exp_gap = (1.0 / p) if p > 0 else 2.0

    sec_of = {}
    sec_size = {}
    for n in range(1, n_max + 1):
        s = (n - 1) // SEC_SIZE
        sec_of[n] = s
        sec_size[s] = sec_size.get(s, 0) + 1

    comps = ('sec', 'gap', 'dev', 'total')
    diag = {
        'calls': 0,
        'span': dict((c, [float('inf'), float('-inf')]) for c in comps),
        'span_sum': dict((c, 0.0) for c in comps),
        'last': {},
        'params': {'SEC_SIZE': SEC_SIZE, 'SEC_SIGN': SEC_SIGN, 'DEV_SIGN': DEV_SIGN,
                   'W_SEC': W_SEC, 'W_GAP': W_GAP, 'W_DEV': W_DEV,
                   'GAP_CAP': GAP_CAP, 'GATE_MIX': GATE_MIX, 'SHORT_WIN': SHORT_WIN},
    }

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, n_max + 1))
        try:
            T = int(draw_index)
        except Exception:
            T = 0
        if T < 0:
            T = 0

        L = len(recent_deque)
        try:
            w = int(recent_window)
        except Exception:
            w = L
        if w <= 0:
            w = L
        m = min(L, SHORT_WIN, w)

        short_cnt = dict.fromkeys(numbers, 0)
        sec_hits = {}
        if m > 0:
            for sset in islice(recent_deque, L - m, None):
                for n in sset:
                    short_cnt[n] += 1
                    s = sec_of.get(n)
                    if s is not None:
                        sec_hits[s] = sec_hits.get(s, 0) + 1

        csum_sec = dict((s, 0) for s in sec_size)
        if T > 0:
            for n in numbers:
                csum_sec[sec_of[n]] += counts.get(n, 0)

        sec_term = {}
        for s, sz in sec_size.items():
            if m > 0:
                z_s = (sec_hits.get(s, 0) - m * sz * p) / math.sqrt(m * sz * p * (1.0 - p))
            else:
                z_s = 0.0
            if T > 0:
                z_l = (csum_sec[s] - T * sz * p) / math.sqrt(T * sz * p * (1.0 - p))
            else:
                z_l = 0.0
            sec_term[s] = SEC_SIGN * (-(GATE_MIX * z_l + (1.0 - GATE_MIX) * z_s))

        sd_long = math.sqrt(T * p * (1.0 - p)) if T > 0 else 0.0
        exp_long = T * p
        sd_short = math.sqrt(m * p * (1.0 - p)) if m > 0 else 0.0
        exp_short = m * p

        gap_vec = {}
        dev_vec = {}
        tot_vec = {}
        for n in numbers:
            z_l = ((counts.get(n, 0) - exp_long) / sd_long) if sd_long > 0 else 0.0
            z_s = ((short_cnt[n] - exp_short) / sd_short) if sd_short > 0 else 0.0
            dev_vec[n] = DEV_SIGN * (-(GATE_MIX * z_l + (1.0 - GATE_MIX) * z_s))

            ls = last_seen.get(n)
            if ls is None:
                gap = T
            else:
                gap = T - int(ls) - 1
                if gap < 0:
                    gap = 0
            ratio = gap / exp_gap
            if ratio > GAP_CAP:
                ratio = GAP_CAP
            gap_vec[n] = ratio - 1.0

            tot_vec[n] = W_SEC * sec_term[sec_of[n]] + W_GAP * gap_vec[n] + W_DEV * dev_vec[n]

        for c, vec in (('sec', sec_term), ('gap', gap_vec), ('dev', dev_vec), ('total', tot_vec)):
            vals = list(vec.values())
            span = max(vals) - min(vals)
            a = diag['span'][c]
            if span < a[0]:
                a[0] = span
            if span > a[1]:
                a[1] = span
            diag['span_sum'][c] += span
        diag['last'] = {'draw_index': draw_index,
                        'sec': dict((n, sec_term[sec_of[n]]) for n in numbers),
                        'gap': dict(gap_vec), 'dev': dict(dev_vec), 'total': dict(tot_vec)}
        diag['calls'] += 1

        return sorted(numbers, key=lambda n: (-tot_vec[n], n))

    rank._diag = diag
    return rank