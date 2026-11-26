NAME = "zzsecdev"
DESCRIPTION = 'Ciclu de intarziere pe sectoare + gap individual capacit + deviatie macro, toate standardizate; GATE_MIX amesteca deviatia de count a sectorului cu gap-ul mediu al sectorului, iar SEC_SIGN/DEV_SIGN comuta intre reversion si momentum.'
SOURCE = 'model + 2026-09-17'

import math

SEC_SIZE = 6
W_SEC = 0.70
W_GAP = 0.50
W_DEV = 0.40
GAP_CAP = 2.5
GATE_MIX = 0.50
SEC_SIGN = -1.0
DEV_SIGN = -1.0
ZCLIP = 4.0
MIN_WINDOW = 10

DIAG = {'calls': 0, 'span_sec': 0.0, 'span_gap': 0.0, 'span_dev': 0.0, 'window_used': 0, 'sectors': 0, 'gate': 0.0}


def _sector_hit_prob(N, KD, size):
    if size <= 0 or N <= 0:
        return 0.0
    if size >= N:
        return 1.0
    if KD <= 0:
        return 0.0
    if KD > N - size:
        return 1.0
    prod = 1.0
    for i in range(KD):
        prod *= (N - size - i) / float(N - i)
        if prod <= 0.0:
            return 1.0
    return 1.0 - prod


def _standardize(vals):
    m = sum(vals) / float(len(vals))
    var = 0.0
    for v in vals:
        var += (v - m) * (v - m)
    sd = math.sqrt(var / float(len(vals)))
    if sd < 1e-12:
        return [0.0] * len(vals)
    return [(v - m) / sd for v in vals]


def _clip_list(vals):
    out = []
    for v in vals:
        if v > ZCLIP:
            v = ZCLIP
        elif v < -ZCLIP:
            v = -ZCLIP
        out.append(v)
    return out


def create(config):
    N = int(config.number_range)
    KD = int(config.numbers_drawn)
    p = KD / float(N) if N > 0 else 0.0
    q = 1.0 - p
    exp_gap = (N / float(KD)) if KD > 0 else float(N)
    cap = GAP_CAP * exp_gap

    sectors = []
    n0 = 1
    while n0 <= N:
        size = min(SEC_SIZE, N - n0 + 1)
        sectors.append((n0, n0 + size - 1, size))
        n0 += SEC_SIZE
    S = len(sectors)
    ps_list = [_sector_hit_prob(N, KD, s[2]) for s in sectors]
    DIAG['sectors'] = S
    DIAG['gate'] = GATE_MIX

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        seq = list(recent_deque)
        rw = int(recent_window) if recent_window else 0
        if rw > 0 and len(seq) > rw:
            seq = seq[-rw:]
        Weff = len(seq)
        if Weff < MIN_WINDOW:
            Weff = int(draw_index)
            cw = {n: counts.get(n, 0) for n in numbers}
        else:
            cw = dict.fromkeys(numbers, 0)
            for st in seq:
                for n in st:
                    if 1 <= n <= N:
                        cw[n] += 1

        gaps = []
        for n in numbers:
            ls = last_seen.get(n)
            if ls is None:
                g = float(draw_index)
            else:
                g = float(draw_index - 1 - int(ls))
            if g < 0.0:
                g = 0.0
            gaps.append(min(g, cap))

        dev_raw = [0.0] * N
        if Weff > 0 and 0.0 < p < 1.0:
            var_n = Weff * p * q
            if var_n > 1e-12:
                mu_n = Weff * p
                sdn = math.sqrt(var_n)
                for i in range(N):
                    dev_raw[i] = (cw[numbers[i]] - mu_n) / sdn
        zdev = _clip_list(_standardize(dev_raw))
        zgap = _clip_list(_standardize(gaps))

        mgap = [0.0] * S
        zsec = [0.0] * S
        if Weff > 0:
            for si in range(S):
                lo, hi, size = sectors[si]
                so = 0
                sg = 0.0
                for n in range(lo, hi + 1):
                    so += cw[n]
                    sg += gaps[n - 1]
                mgap[si] = sg / float(size)
                ps = ps_list[si]
                var_s = Weff * ps * (1.0 - ps)
                if var_s > 1e-12:
                    zsec[si] = (so - Weff * ps) / math.sqrt(var_s)
        zsec = _clip_list(zsec)
        zgsec = _clip_list(_standardize(mgap))

        sec_comp = [0.0] * N
        for si in range(S):
            lo, hi, size = sectors[si]
            sv = GATE_MIX * zsec[si] + (1.0 - GATE_MIX) * zgsec[si]
            for n in range(lo, hi + 1):
                sec_comp[n - 1] = sv

        score = [0.0] * N
        for i in range(N):
            score[i] = W_SEC * SEC_SIGN * sec_comp[i] + W_GAP * zgap[i] + W_DEV * DEV_SIGN * zdev[i]

        DIAG['calls'] += 1
        DIAG['window_used'] = Weff
        DIAG['span_sec'] = max(sec_comp) - min(sec_comp)
        DIAG['span_gap'] = max(zgap) - min(zgap)
        DIAG['span_dev'] = max(zdev) - min(zdev)

        return sorted(numbers, key=lambda n: (-score[n - 1], n))

    return rank