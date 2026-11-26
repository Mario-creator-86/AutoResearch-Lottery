NAME = "zgaphaze"
DESCRIPTION = 'Hazard de intarziere (raport gap curent / gap mediu empiric) combinat neliniar (tanh) cu deviatia z pe termen lung, moment EWMA (gamma=0.95) pe fereastra recenta si presiune de revazere pe sectoare de 10, modulata de entropia distributiei extragerilor recente pe sectoare.'
SOURCE = 'model cantitativ + 2026-09-08'

import math

DIAG = {'orient': None, 'orient_flips': 0}

# hiperparametri expusi pentru grid-search ulterior
W_Z = 0.6      # deviatie z pe termen lung
W_OV = 0.9     # hazard de gap (mean-reversion pe timpul de asteptare)
W_SEC = 0.55   # presiune sector (inainte de modularea entropiei)
W_EW = 0.35    # moment EWMA
GAMMA = 0.95   # decay EWMA (jumatate de viata ~13 extrageri)
TK = 0.5       # scalarea tanh
CYC = 1.5      # ponderea ciclului de gap pe sector
RCAP = 6.0     # plafon pe raportul de intarziere


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    p = K / float(N)
    S = (N + 9) // 10
    sec_size = []
    for s in range(S):
        lo = 10 * s + 1
        hi = min(10 * s + 10, N)
        sec_size.append(hi - lo + 1)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        T = float(max(draw_index, 1))
        mu = T * p
        sd = math.sqrt(T * p * (1.0 - p) + 1e-9)
        L = len(recent_deque)

        # auto-detectarea orientarii deque-ului: setul cel mai nou are
        # recenta medie last_seen ~ 1, cel mai vechi mult mai mare
        def mean_recency(st):
            if not st:
                return float('inf')
            tot = 0.0
            for n in st:
                ls = last_seen.get(n)
                tot += (draw_index - ls) if ls is not None else float(draw_index)
            return tot / float(len(st))

        newest_last = True
        if L >= 2:
            newest_last = mean_recency(recent_deque[-1]) <= mean_recency(recent_deque[0])
        orient = 'newest_last' if newest_last else 'newest_first'
        if DIAG['orient'] is not None and DIAG['orient'] != orient:
            DIAG['orient_flips'] += 1
        DIAG['orient'] = orient

        seq = list(recent_deque)
        if newest_last and L >= 2:
            seq.reverse()

        ew = dict.fromkeys(numbers, 0.0)
        sec_hits = [0.0] * S
        sec_gap = [L] * S
        for age in range(len(seq)):
            w = GAMMA ** age
            for n in seq[age]:
                if 1 <= n <= N:
                    ew[n] += w
                    si = (n - 1) // 10
                    sec_hits[si] += 1.0
                    if sec_gap[si] > age:
                        sec_gap[si] = age

        S_w = (1.0 - GAMMA ** L) / (1.0 - GAMMA) if L > 0 else 0.0

        sec_score = [0.0] * S
        w_sec_eff = 0.0
        if L > 0:
            tot_hits = sum(sec_hits)
            ent = 0.0
            if tot_hits > 0:
                for h in sec_hits:
                    if h > 0:
                        q = h / tot_hits
                        ent -= q * math.log(q)
            max_ent = math.log(S) if S > 1 else 1.0
            ent_def = max(0.0, (max_ent - ent) / max_ent)
            w_sec_eff = W_SEC * (0.5 + ent_def)
            for s in range(S):
                Es = L * K * (sec_size[s] / float(N))
                deficit = (Es - sec_hits[s]) / math.sqrt(Es + 1.0)
                cyc = sec_gap[s] / float(L)
                sec_score[s] = math.tanh(TK * (deficit + CYC * cyc))

        out = []
        for n in numbers:
            c = float(counts.get(n, 0))
            zs = math.tanh(TK * ((c - mu) / sd))
            ls = last_seen.get(n)
            if ls is None or c <= 0.0:
                r = 3.0
            else:
                gap = max(draw_index - ls, 0)
                mg = T / c
                r = min(gap / max(mg, 1e-9), RCAP)
            haz = math.tanh(TK * (r - 1.0))
            ewn = (ew[n] / S_w) if S_w > 0 else p
            mom = math.tanh(TK * (ewn - p) / p)
            si = (n - 1) // 10
            score = W_Z * zs + W_OV * haz + w_sec_eff * sec_score[si] + W_EW * mom
            out.append((n, round(score, 9)))

        out.sort(key=lambda t: (-t[1], t[0]))
        return [t[0] for t in out]

    return rank