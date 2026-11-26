import math

NAME = "zzsecyc3"
DESCRIPTION = (
    'Fuziune neliniara pe 3 orizonturi de persistenta: (1) ecou short-lag - boost pentru '
    'numerele reaparinse in ultimele 1-3 extrageri (tendinta de repetitie); (2) ciclu de '
    'intarziere pe sectoare de 7 numere - sectorul a carui ultima aparitie e mai veche decat '
    'media geometrica (~1.62 extrageri intre lovituri) primeste deficit-score comprimat; '
    '(3) deviatie pe termen lung fata de frecventa asteptata (z-score pe counts, momentum slab). '
    'Componentele sunt z-standardizate, trecute prin tanh si combinate cu ponderi fixe; '
    'componentele constante (std~0) se dezactiveaza automat.'
)
SOURCE = 'model + 2026-09-16'


def create(config):
    N = config.number_range            # 42
    W = config.numbers_drawn           # 5
    SECT = 7 if N >= 14 else max(1, N // 6)
    NSEC = (N + SECT - 1) // SECT

    # probabilitatea ca un sector sa fie lovit intr-o extragere: 1 - C(N-SECT, W)/C(N, W)
    p_miss = math.comb(N - SECT, W) / math.comb(N, W)
    p_sect = 1.0 - p_miss              # ~0.618 pentru 5/42 cu sector de 7
    exp_age_sec = 1.0 / p_sect         # ~1.617 extrageri intre lovituri pe sector

    W_LAG, W_SEC, W_DEV = 0.90, 0.70, 0.40
    CAP_SEC = 8.0

    def _zs(vals):
        m = sum(vals) / len(vals)
        var = sum((v - m) * (v - m) for v in vals) / len(vals)
        sd = math.sqrt(var)
        if sd < 1e-12:
            return [0.0] * len(vals)
        inv = 1.0 / sd
        return [(v - m) * inv for v in vals]

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))

        # gap-uri exacte din last_seen (independent de ordinea din recent_deque)
        gaps = []
        for n in numbers:
            ls = last_seen.get(n)
            gaps.append((draw_index - ls) if ls is not None else (draw_index + 1))

        # (1) ecou short-lag: persistenta pe 1-3 extrageri
        lag_raw = []
        for g in gaps:
            if g == 1:
                lag_raw.append(1.0)
            elif g == 2:
                lag_raw.append(0.45)
            elif g == 3:
                lag_raw.append(0.18)
            else:
                lag_raw.append(0.0)

        # (2) ciclu de intarziere pe sectoare: varsta = gap-ul minim din sector
        min_gap = [None] * NSEC
        for i in range(N):
            s = i // SECT
            g = gaps[i]
            if min_gap[s] is None or g < min_gap[s]:
                min_gap[s] = g
        sec_raw = []
        for i in range(N):
            s = i // SECT
            over = min_gap[s] - exp_age_sec
            if over < 0.0:
                over = 0.0
            elif over > CAP_SEC:
                over = CAP_SEC
            sec_raw.append(over ** 0.75)

        # (3) deviatie pe termen lung fata de frecventa asteptata
        dev_raw = [float(counts.get(n, 0)) for n in numbers]

        z_lag = _zs(lag_raw)
        z_sec = _zs(sec_raw)
        z_dev = _zs(dev_raw)

        scores = {}
        for i, n in enumerate(numbers):
            scores[n] = (W_LAG * math.tanh(z_lag[i])
                         + W_SEC * math.tanh(z_sec[i])
                         + W_DEV * math.tanh(z_dev[i]))

        return sorted(numbers, key=lambda x: (-scores[x], x))

    return rank