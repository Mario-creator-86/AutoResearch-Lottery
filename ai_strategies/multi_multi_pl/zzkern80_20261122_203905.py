NAME = "zzkern80"
DESCRIPTION = ('Convolutie spatiala pe axa numerelor: densitate side-lobe (+-1/+-2) a aparitiilor '
               'din fereastra recenta (co-localizare locala), normalizata pe sloturi de vecini validi, '
               'combinata ne-liniar (squash tanh + termen de interactiune) cu momentum-ul ferestrei '
               'si z-score-ul frecventei globale; due doar ca tie-breaker determinist.')
SOURCE = 'model + 2026-11-12'

import math


def create(config):
    P = int(config.number_range)
    D = int(config.numbers_drawn)

    # Knobi expusi pentru sweep (citire defensiva; fallback = default determinist)
    W_GLOB = float(getattr(config, 'w_glob', 0.55))
    W_WIN = float(getattr(config, 'w_win', 0.85))
    W_NB = float(getattr(config, 'w_nb', 0.70))
    W_INT = float(getattr(config, 'w_int', 0.35))
    SZ_WIN = float(getattr(config, 'sz_win', 1.2))
    SZ_GLOB = float(getattr(config, 'sz_glob', 2.0))
    SZ_NB = float(getattr(config, 'sz_nb', 1.4))
    K1 = float(getattr(config, 'k1', 0.60))
    K2 = float(getattr(config, 'k2', 0.30))

    p = (D / float(P)) if P > 0 else 0.25

    def _tanh(z, s):
        return math.tanh(z / s) if s > 1e-12 else math.tanh(z)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, P + 1))
        if not numbers:
            return numbers

        # 1) Apartenenta in fereastra: doar membership -> invariant la ordinea/flip-ul deque-ului
        w = [0] * (P + 2)
        w_used = 0
        if recent_deque is not None:
            for s in recent_deque:
                if not s:
                    continue
                w_used += 1
                for n in s:
                    if 1 <= n <= P:
                        w[n] += 1
        W_eff = w_used if w_used > 0 else max(1, int(recent_window or 1))

        lam = W_eff * p
        wsd = math.sqrt(max(W_eff * p * (1.0 - p), 1e-12))

        # 2) z global empiric (auto-calibrat pe 80 contoare)
        cs = [counts.get(n, 0) for n in numbers]
        cm = sum(cs) / float(P)
        csd = math.sqrt(sum((c - cm) * (c - cm) for c in cs) / float(P))
        if csd < 1e-12:
            csd = 1.0

        # 3) Convolutie side-lobe: d=1 -> K1, d=2 -> K2; normalizare pe nr. de vecini validi
        #    (interior=4, aproape-margini=3, capete=2) -> fara bias structural la margini
        kslot = [0] * (P + 2)
        for n in range(1, P + 1):
            c = (1 if n - 1 >= 1 else 0) + (1 if n + 1 <= P else 0) + (1 if n - 2 >= 1 else 0) + (1 if n + 2 <= P else 0)
            kslot[n] = c if c > 0 else 1
        rate = [0.0] * (P + 2)
        for n in range(1, P + 1):
            kn = 0.0
            if n - 1 >= 1:
                kn += K1 * w[n - 1]
            if n + 1 <= P:
                kn += K1 * w[n + 1]
            if n - 2 >= 1:
                kn += K2 * w[n - 2]
            if n + 2 <= P:
                kn += K2 * w[n + 2]
            rate[n] = kn / kslot[n]

        rm = sum(rate[1:P + 1]) / float(P)
        rsd = math.sqrt(sum((rate[n] - rm) * (rate[n] - rm) for n in range(1, P + 1)) / float(P))
        if rsd < 1e-12:
            rsd = 1.0

        di = draw_index if isinstance(draw_index, int) else (draw_index or 0)
        keys = [None] * (P + 1)
        for n in numbers:
            z_w = (w[n] - lam) / wsd
            z_g = (cs[n - 1] - cm) / csd
            z_k = (rate[n] - rm) / rsd
            tw = _tanh(z_w, SZ_WIN)
            tg = _tanh(z_g, SZ_GLOB)
            tk = _tanh(z_k, SZ_NB)
            sc = W_WIN * tw + W_GLOB * tg + W_NB * tk + W_INT * (tw * tk)
            ls = last_seen.get(n, None) if last_seen else None
            gap = (di - ls) if ls is not None else (di + 1)
            keys[n] = (-sc, -gap, n)

        return sorted(numbers, key=lambda n: keys[n])

    return rank