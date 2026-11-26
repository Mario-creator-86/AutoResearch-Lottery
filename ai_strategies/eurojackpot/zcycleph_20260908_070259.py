NAME = "zcycleph"
DESCRIPTION = 'Faza de reinnoare per numar: varf Gaussian cand gap-ul curent coincide cu ciclul propriu mediu (phi=gap/ciclu~1, ecou la ~2), amplificata neliniar de gate-ul de deficit al sectorului (banda de 10), combinata cu deviatia z long-run si cu rata din fereastra recenta.'
SOURCE = 'glm-5.3-flash + 2026-09-08'

import math

SIGMA = 0.30
W_PH1 = 1.00
W_PH2 = 0.40
W_DEV = 0.55
W_REC = 0.35
W_BAND = 0.45
BAND_SCALE = 3.0
DEV_CLIP = 3.0
REC_CLIP = 2.0


def create(config):
    nr = int(config.number_range)
    nd = int(config.numbers_drawn)
    p = nd / float(nr)
    nb = max(1, nr // 10)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, nr + 1))
        D = float(draw_index) if draw_index > 0 else 1.0
        exp_c = p * D
        sd_c = math.sqrt(D * p * (1.0 - p))
        if sd_c <= 0.0:
            sd_c = 1.0

        rcount = {}
        win = len(recent_deque)
        sd_r = 1.0
        if win:
            for s in recent_deque:
                for m in s:
                    rcount[m] = rcount.get(m, 0) + 1
            sd_r = math.sqrt(win * p * (1.0 - p))
            if sd_r <= 0.0:
                sd_r = 1.0

        band_of = {}
        gap = {}
        for n in numbers:
            ls = last_seen.get(n)
            g = draw_index - ls if ls is not None else int(D) + 1
            if g < 1:
                g = 1
            gap[n] = g
            band_of[n] = ((n - 1) * nb) // nr

        band_score = {}
        for b in range(nb):
            gmin = None
            for n in numbers:
                if band_of[n] == b:
                    g = gap[n]
                    if gmin is None or g < gmin:
                        gmin = g
            if gmin is None:
                band_score[b] = 0.0
            else:
                excess = gmin - 1.0
                if excess < 0.0:
                    excess = 0.0
                band_score[b] = min(excess / BAND_SCALE, 1.2)

        inv = 1.0 / (2.0 * SIGMA * SIGMA)
        scores = {}
        for n in numbers:
            c = counts.get(n, 0)
            dev = (c - exp_c) / sd_c
            if dev > DEV_CLIP:
                dev = DEV_CLIP
            elif dev < -DEV_CLIP:
                dev = -DEV_CLIP

            if win:
                rec = (rcount.get(n, 0) - win * p) / sd_r
                if rec > REC_CLIP:
                    rec = REC_CLIP
                elif rec < -REC_CLIP:
                    rec = -REC_CLIP
            else:
                rec = 0.0

            if c > 0:
                lam = D / c
                if lam < 1.0:
                    lam = 1.0
            else:
                lam = 1.0 / p
            phi = gap[n] / lam
            gate = 1.0 + W_BAND * band_score[band_of[n]]
            harm = (W_PH1 * math.exp(-((phi - 1.0) ** 2) * inv)
                    + W_PH2 * math.exp(-((phi - 2.0) ** 2) * inv))
            scores[n] = gate * harm + W_DEV * dev + W_REC * rec

        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank