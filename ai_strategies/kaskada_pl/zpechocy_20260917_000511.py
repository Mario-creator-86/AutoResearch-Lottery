import math

NAME = "zpechocy"
DESCRIPTION = 'Ecou de perechi ancorat pe ultima extragere (afinitati conditionale recen-ponderate) + ciclu de reechilibrare pe sectoare + gap saturat per numar; fara counts globale (fara hot/cold clasic).'
SOURCE = 'model cantitativ + 2026-09-16'


def create(config):
    W_PAIR = 0.8
    PAIR_DECAY = 0.92
    ANCHOR_DRAWS = 1
    W_SEC = 0.35
    SECTOR_SIZE = 6
    W_GAP = 0.5
    GAP_CAP = 5.0
    EPS = 1e-12
    cache = {}

    def build(draws):
        N = config.number_range
        L = len(draws)
        anchor = set()
        for d in draws[L - ANCHOR_DRAWS:]:
            anchor |= d
        hist = draws[:L - ANCHOR_DRAWS] if L > ANCHOR_DRAWS else []
        H = len(hist)
        cnt = {m: 0.0 for m in anchor}
        co = {m: dict.fromkeys(range(1, N + 1), 0.0) for m in anchor}
        for i, d in enumerate(hist):
            w = PAIR_DECAY ** (H - 1 - i)
            if w < 1e-4:
                break
            for m in anchor:
                if m in d:
                    cnt[m] += w
                    row = co[m]
                    for x in d:
                        if x != m:
                            row[x] += w
        valid = [m for m in anchor if cnt[m] > EPS]
        echo = {}
        if valid:
            inv = 1.0 / len(valid)
            for n in range(1, N + 1):
                s = 0.0
                for m in valid:
                    s += co[m][n] / cnt[m]
                echo[n] = s * inv
        else:
            for n in range(1, N + 1):
                echo[n] = 0.0
        vals = list(echo.values())
        mu = sum(vals) / N
        var = sum((v - mu) ** 2 for v in vals) / N
        if var > EPS:
            sd = math.sqrt(var)
            ez = {n: (echo[n] - mu) / sd for n in echo}
        else:
            ez = {n: 0.0 for n in echo}
        ns = max(1, N // SECTOR_SIZE)
        sec_hits = [0.0] * ns
        for d in draws:
            for x in d:
                sec_hits[(x - 1) // SECTOR_SIZE] += 1.0
        denom = float(L * SECTOR_SIZE)
        rates = [h / denom for h in sec_hits]
        mu_s = sum(rates) / ns
        var_s = sum((r - mu_s) ** 2 for r in rates) / ns
        if var_s > EPS:
            sd_s = math.sqrt(var_s)
            sz = {n: (mu_s - rates[(n - 1) // SECTOR_SIZE]) / sd_s for n in range(1, N + 1)}
        else:
            sz = {n: 0.0 for n in range(1, N + 1)}
        return ez, sz

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        N = config.number_range
        numbers = list(range(1, N + 1))
        draws = list(recent_deque)
        L = len(draws)
        if L == 0:
            return sorted(numbers, key=lambda n: (-counts.get(n, 0), n))
        key = (draw_index, L)
        st = cache.get(key)
        if st is None:
            st = build(draws)
            if len(cache) > 16:
                cache.clear()
            cache[key] = st
        ez, sz = st

        gap = {}
        for n in numbers:
            ls = last_seen.get(n)
            g = (draw_index - ls) if ls is not None else (draw_index + 1)
            if g > GAP_CAP:
                g = GAP_CAP
            gap[n] = g
        vals = [gap[n] for n in numbers]
        mu_g = sum(vals) / N
        var_g = sum((v - mu_g) ** 2 for v in vals) / N
        if var_g > EPS:
            sd_g = math.sqrt(var_g)
            gz = {n: (gap[n] - mu_g) / sd_g for n in numbers}
        else:
            gz = {n: 0.0 for n in numbers}

        scores = {n: W_PAIR * ez[n] + W_SEC * sz[n] + W_GAP * gz[n] for n in numbers}
        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank