import math

NAME = "zstateco"
DESCRIPTION = "Lant Markov conditionat pe starea compozitiei ultimei extrageri (bucket paritate, low/high, banda de suma): frecventa conditionata a fiecarui numar la draw-ul urmator, estimata cu kernel gaussian pe distanta de stare si shrinkage catre frecventa marginala globala."
SOURCE = "draw-state Markov kernel + marginal anchor, 2026-09-08"

def create(config):
    N = config.number_range
    K = config.numbers_drawn

    # muchii fixe de banda de suma (deterministe, din formula analitica)
    mean_sum = K * (N + 1) / 2.0
    sd_sum = (K * (N * N - 1) / 12.0) ** 0.5
    E1 = mean_sum - 1.2 * sd_sum
    E2 = mean_sum - 0.4 * sd_sum
    E3 = mean_sum + 0.4 * sd_sum
    E4 = mean_sum + 1.2 * sd_sum

    SIGMA2 = 0.14   # latimea nucleului gaussian (sigma^2), distante normalizate
    ALPHA = 30.0    # pseudo-numarator de shrinkage catre marginala

    def _state(draw):
        odds = 0
        lows = 0
        s = 0
        for n in draw:
            if n % 2 == 1:
                odds += 1
            if n <= N // 2:
                lows += 1
            s += n
        b1 = 0 if odds <= 1 else (1 if odds == 2 else (2 if odds == 3 else 3))
        b2 = 0 if lows <= 1 else (1 if lows == 2 else (2 if lows == 3 else 3))
        b3 = 0 if s < E1 else (1 if s < E2 else (2 if s < E3 else (3 if s < E4 else 4)))
        return (b1, b2, b3)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        H = draw_index if draw_index and draw_index > 0 else 1
        p = {n: counts.get(n, 0) / float(H) for n in numbers}

        draws = list(recent_deque)
        m = len(draws)
        if m < 10:
            return sorted(numbers, key=lambda n: (-p[n], n))

        cur = _state(draws[-1])
        c = {n: 0.0 for n in numbers}
        W = 0.0
        for t in range(m - 1):
            st = _state(draws[t])
            d1 = (st[0] - cur[0]) / 3.0
            d2 = (st[1] - cur[1]) / 3.0
            d3 = (st[2] - cur[2]) / 4.0
            w = math.exp(-(d1 * d1 + d2 * d2 + d3 * d3) / (2.0 * SIGMA2))
            if w < 1e-6:
                continue
            W += w
            nxt = draws[t + 1]
            for n in nxt:
                c[n] += w

        if W <= 1e-9:
            return sorted(numbers, key=lambda n: (-p[n], n))

        denom = W + ALPHA
        scores = {n: (c[n] + ALPHA * p[n]) / denom for n in numbers}
        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank