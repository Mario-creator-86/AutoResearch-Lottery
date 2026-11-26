NAME = "zmetrono"
DESCRIPTION = "Scor deterministic agnostic la orientarea deque-ului: hazard de intarziere normalizat la intervalul asteptat 49/6, fit ciclic 'metronom' pe intervalul personal mediu al fiecarui numar, deficit sectorial pe 5 sectoare in fereastra recenta si contrarian z-score pe fereastra recenta; componentele sunt standardizate si combinate liniar."
SOURCE = "model glm + 2026-09-23"

import math

WG = 0.55   # hazard intarziere
WC = 1.00   # fit ciclic metronom
WS = 0.35   # deficit sectorial
WD = -0.25  # contrarian fereastra scurta


def create(config):
    P = config.number_range
    W = config.numbers_drawn
    p = W / float(P)
    mu = 1.0 / p
    n_sect = 5
    sect_of = [0] * (P + 1)
    sect_size = [0] * n_sect
    for n in range(1, P + 1):
        s = min((n - 1) * n_sect // P, n_sect - 1)
        sect_of[n] = s
        sect_size[s] += 1

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, P + 1))
        rc = dict.fromkeys(numbers, 0)
        rsec = [0] * n_sect
        rw = recent_window if recent_window and recent_window > 0 else 1
        if recent_deque:
            for st in recent_deque:
                if not st:
                    continue
                for x in st:
                    if 1 <= x <= P:
                        rc[x] += 1
                        rsec[sect_of[x]] += 1
        e_w = rw * p
        sd_w = math.sqrt(max(e_w * (1.0 - p), 1e-12))
        sec_z = []
        for i in range(n_sect):
            es = max(rw * sect_size[i] * p, 1e-12)
            sds = math.sqrt(max(es * (1.0 - p), 1e-12))
            sec_z.append((es - rsec[i]) / sds)
        raw = []
        for n in numbers:
            g = float(draw_index - last_seen.get(n, -1))
            if g < 0.0:
                g = 0.0
            haz = min(g / mu, 3.0)
            c = counts.get(n, 0)
            if c > 0:
                m = (draw_index + 1.0) / (c + 1.0)
                sig = max(0.5 * m, 2.5)
                cyc = math.exp(-0.5 * ((g - m) / sig) ** 2)
            else:
                cyc = 0.0
            dev = (rc[n] - e_w) / sd_w
            raw.append((haz, cyc, sec_z[sect_of[n]], dev))
        wts = (WG, WC, WS, WD)
        zs = []
        for j in range(4):
            col = [r[j] for r in raw]
            mean = sum(col) / float(P)
            var = sum((v - mean) ** 2 for v in col) / float(P)
            std = math.sqrt(var)
            if std > 1e-12:
                zs.append([(v - mean) / std for v in col])
            else:
                zs.append([0.0] * P)
        scores = {}
        for i, n in enumerate(numbers):
            s = 0.0
            for j in range(4):
                s += wts[j] * zs[j][i]
            scores[n] = s
        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank