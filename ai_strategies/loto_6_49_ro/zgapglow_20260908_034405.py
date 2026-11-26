import math

NAME = "zgapglow"
DESCRIPTION = (
    'Hibrid recenta-gap: EWMA cu half-life fix pe fereastra recenta, modulata de un gate '
    'de dispersie si shrinkage cu pseudo-count a0, plus scor pe bucket-uri de gap '
    '(1-2, 3-5, 6-10, 11-20, >20), glow pentru numerele aparute in ultimele 1-2 extrageri '
    'si o ancora slaba de deviatie a frecventei pe tot istoricul.'
)
SOURCE = 'gap-bucket EWMA + dispersion gate, model cantitativ 2026-09-07'

# Hiperparametri si domenii de recalibrare:
#   HALF_LIFE in [8, 16], A0 in [6, 20], GATE_SLOPE in [0.5, 1.5], GLOW_AMP in [0.2, 0.6]
HALF_LIFE = 12.0
A0 = 12.0
GATE_SLOPE = 1.0
GLOW_AMP = 0.40
W_COUNT = 0.45
W_EWMA = 1.00
W_GAP = 0.60
GAP_BUCKET_W = (0.50, 0.25, 0.15, 0.12, 0.18)  # 1-2, 3-5, 6-10, 11-20, >20
GATE_MIN = 0.50
GATE_MAX = 1.80


def create(config):
    n_max = int(config.number_range)
    k_drawn = int(config.numbers_drawn)
    decay = math.log(2.0) / HALF_LIFE
    null_disp = math.sqrt(max(1e-9, 1.0 - k_drawn / float(n_max)))

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, n_max + 1))

        # 1) ancora de deviatie fata de frecventa asteptata (tot istoricul)
        vals = [counts.get(n, 0) for n in numbers]
        cnt = len(vals)
        cmean = sum(vals) / float(cnt)
        cvar = sum((v - cmean) ** 2 for v in vals) / float(cnt)
        cstd = math.sqrt(cvar) + 1e-9
        cz = {n: (counts.get(n, 0) - cmean) / cstd for n in numbers}

        # 2) fereastra ordonata vechi->nou, robusta la ordinea deque-ului
        dq = list(recent_deque)
        if len(dq) > 1 and draw_index > 0:
            prev_idx = draw_index - 1

            def fresh_ratio(s):
                if not s:
                    return 0.0
                h = 0
                for x in s:
                    ls = last_seen.get(x)
                    if ls is not None and ls == prev_idx:
                        h += 1
                return h / float(len(s))

            if fresh_ratio(dq[0]) > fresh_ratio(dq[-1]):
                dq.reverse()

        # 3) EWMA cu half-life
        ewma = dict.fromkeys(numbers, 0.0)
        eff_draws = 0.0
        m = len(dq)
        for i in range(m):
            w = math.exp(-decay * (m - 1 - i))
            eff_draws += w
            for x in dq[i]:
                if 1 <= x <= n_max:
                    ewma[x] += w
        total_w = eff_draws * k_drawn
        mu = total_w / float(n_max)
        inv_sqrt_mu = (1.0 / math.sqrt(mu)) if mu > 1e-9 else 0.0

        # 4) gate de dispersie: std(ewma)/sqrt(mu) raportat la asteptarea sub null
        if mu > 1e-9:
            evar = sum((ewma[n] - mu) ** 2 for n in numbers) / float(cnt)
            disp_norm = (math.sqrt(evar) * inv_sqrt_mu) / null_disp
        else:
            disp_norm = 1.0
        gate = 1.0 + GATE_SLOPE * (disp_norm - 1.0)
        if gate < GATE_MIN:
            gate = GATE_MIN
        elif gate > GATE_MAX:
            gate = GATE_MAX

        # 5) shrinkage cu pseudo-count a0 (in extrageri echivalente)
        shrink = eff_draws / (eff_draws + A0)

        scores = {}
        for n in numbers:
            ls = last_seen.get(n)
            gap = (draw_index - ls) if ls is not None else 1000000
            if gap < 1:
                gap = 1
            if gap <= 2:
                b = 0
            elif gap <= 5:
                b = 1
            elif gap <= 10:
                b = 2
            elif gap <= 20:
                b = 3
            else:
                b = 4
            glow = GLOW_AMP if gap == 1 else (0.5 * GLOW_AMP if gap == 2 else 0.0)
            ez = (ewma[n] - mu) * inv_sqrt_mu
            scores[n] = (W_COUNT * cz[n]
                         + gate * W_EWMA * shrink * ez
                         + W_GAP * GAP_BUCKET_W[b]
                         + glow)

        return sorted(numbers, key=lambda x: (-scores[x], x))

    return rank