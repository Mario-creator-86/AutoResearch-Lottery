import math

NAME = "zhalogap"
DESCRIPTION = 'Multi-horizon scorer: recency-decayed co-occurrence halo ratio (quality of draw-mates), geometric gap z (overdue), sector mean-reversion and long-run frequency reversion; z-standardized blocks, linear mix.'
SOURCE = 'quant-research 2026-09-26'

W_GAP = 0.60
W_HALO = 0.75
W_SECT = 0.30
W_REV = 0.20
PAIR_WIN = 30
SECT_WIN = 40
HALO_TAU = 12.0
N_SECT = 7
Z_CLIP = 3.5
Z_FLOOR = -1.5


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    p = K / float(N)
    mean_gap = (1.0 - p) / p
    std_gap = math.sqrt(1.0 - p) / p
    sector_of = [0] * (N + 1)
    sector_size = [0] * N_SECT
    for n in range(1, N + 1):
        s = min((n - 1) * N_SECT // N, N_SECT - 1)
        sector_of[n] = s
        sector_size[s] += 1
    diag = {'newest_first': None, 'contrib': {}}

    def _z(vals):
        m = sum(vals) / float(len(vals))
        var = sum((v - m) * (v - m) for v in vals) / float(len(vals))
        sd = math.sqrt(var)
        if sd <= 1e-12:
            return [0.0] * len(vals)
        return [(v - m) / sd for v in vals]

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        NEG = -(1 << 60)

        nf = True
        if recent_deque:
            thr = draw_index - 3
            hl = 0
            for x in recent_deque[-1]:
                if last_seen.get(x, NEG) >= thr:
                    hl += 1
            hf = 0
            for x in recent_deque[0]:
                if last_seen.get(x, NEG) >= thr:
                    hf += 1
            nf = hl >= hf
        diag['newest_first'] = nf
        seq = recent_deque if nf else reversed(recent_deque)

        need = PAIR_WIN if PAIR_WIN >= SECT_WIN else SECT_WIN
        pw = []
        sw = []
        for i, d in enumerate(seq):
            if i < PAIR_WIN:
                pw.append(d)
            if i < SECT_WIN:
                sw.append(d)
            if i >= need:
                break

        gap_sig = []
        for n in numbers:
            ls = last_seen.get(n)
            g = (draw_index - ls) if ls is not None else (draw_index + 1)
            z = (g - mean_gap) / std_gap
            if z > Z_CLIP:
                z = Z_CLIP
            elif z < Z_FLOOR:
                z = Z_FLOOR
            gap_sig.append(z)
        s_gap = _z(gap_sig)

        hot = [0.0] * (N + 1)
        for i, d in enumerate(pw):
            w = math.exp(-i / HALO_TAU)
            for m in d:
                hot[m] += w
        aff = [0.0] * (N + 1)
        for d in pw:
            s_d = 0.0
            for m in d:
                s_d += hot[m]
            for m in d:
                aff[m] += s_d - hot[m]
        denom = [aff[n] / (hot[n] + 2.0) for n in numbers]
        s_halo = _z(denom)

        s_sect = [0.0] * len(numbers)
        if sw:
            D = len(sw)
            act = [0] * N_SECT
            for d in sw:
                for m in d:
                    act[sector_of[m]] += 1
            devs = []
            for si in range(N_SECT):
                e = D * K * sector_size[si] / float(N)
                devs.append(-((act[si] - e) / e) if e > 1e-9 else 0.0)
            zs = _z(devs)
            for idx, n in enumerate(numbers):
                s_sect[idx] = zs[sector_of[n]]

        tot = 0.0
        for n in numbers:
            tot += counts.get(n, 0)
        exp_f = tot / float(N)
        sd_f = math.sqrt(exp_f * (1.0 - p)) if exp_f > 0.0 else 1.0
        s_rev = _z([-(counts.get(n, 0) - exp_f) / sd_f for n in numbers])

        final = [0.0] * (N + 1)
        cg = ch = cs = cr = 0.0
        for idx, n in enumerate(numbers):
            vg = W_GAP * s_gap[idx]
            vh = W_HALO * s_halo[idx]
            vs = W_SECT * s_sect[idx]
            vr = W_REV * s_rev[idx]
            final[n] = vg + vh + vs + vr
            cg += vg if vg >= 0.0 else -vg
            ch += vh if vh >= 0.0 else -vh
            cs += vs if vs >= 0.0 else -vs
            cr += vr if vr >= 0.0 else -vr
        M = float(len(numbers))
        diag['contrib'] = {'gap': cg / M, 'halo': ch / M, 'sector': cs / M, 'freq_rev': cr / M}

        return sorted(numbers, key=lambda n: (-final[n], n))

    rank._diag = diag
    return rank