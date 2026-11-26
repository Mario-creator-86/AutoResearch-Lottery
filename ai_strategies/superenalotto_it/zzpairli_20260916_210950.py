NAME = "zzpairli"
DESCRIPTION = 'Echo de afinitate pe perechi: lift Bayesian al co-ocurentelor (observat cu decay exponential / asteptat sub independenta) intre numerele din ultimele 2 extrageri si restul ferestrei, combinat liniar cu momentum de frecventa pe fereastra HL, deficit z pe sectoare de 10 si reversiune tanh fata de z-ul pe termen lung.'
SOURCE = 'model cantitativ + 2026-09-17'

import math

DIAG = {'orient': None, 'orient_flips': 0, 'calls': 0, 'echo_min': 0.0, 'echo_max': 0.0}

# hiperparametri expusi pentru grid-search / ablatie
W_AFF = 0.8        # ponderea echo-ului de perechi (mecanism principal)
AFF_DECAY = 0.85   # decay exponential al co-ocurentei pe vechime
PAIR_WINDOW = 120  # fereastra maxima pentru co-ocurente
ANCHOR_DRAWS = 2   # extragerile recente folosite ca ancore
ANCHOR_DECAY = 0.5 # decay pe vechimea ancorei
AFF_SMOOTH = 1.0   # netezire Bayesiana a lift-ului
W_FREQ = 0.30      # momentum de frecventa
HL = 20            # fereastra de frecventa
W_SEC = 0.15       # deficit pe sectoare (semn: -z => subreprezentate favorizate)
SEC_WINDOW = 60
W_Z = 0.15         # reversiune fata de z pe termen lung (tanh, clampata)


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    p = K / float(N)
    p_pair = (K * (K - 1)) / float(N * (N - 1))
    S = (N + 9) // 10
    sec_lo = [10 * s + 1 for s in range(S)]
    sec_hi = [min(10 * s + 10, N) for s in range(S)]

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        DIAG['calls'] += 1
        L = len(recent_deque)
        if L == 0:
            return sorted(numbers, key=lambda n: (-counts.get(n, 0), n))

        T = float(max(draw_index, 1))

        # detectare determinista a orientarii deque-ului (nu trebuie sa basculeze)
        def mean_recency(st):
            if not st:
                return float('inf')
            tot = 0.0
            for x in st:
                ls = last_seen.get(x)
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
            seq.reverse()  # seq[0] = extragerea cea mai recenta

        # 1) echo de perechi (mecanism principal)
        M = min(ANCHOR_DRAWS, L)
        anchors = set()
        for m in range(M):
            for x in seq[m]:
                if 1 <= x <= N:
                    anchors.add(x)

        lp = min(PAIR_WINDOW, L)
        pair_w = {}
        for age in range(lp):
            dr = seq[age]
            inter = sorted(x for x in dr if x in anchors)
            if not inter:
                continue
            w = AFF_DECAY ** age
            seen = set()
            for a in inter:
                for v in dr:
                    if v == a or v < 1 or v > N:
                        continue
                    key = (a, v) if a < v else (v, a)
                    if key in seen:
                        continue
                    seen.add(key)
                    pair_w[key] = pair_w.get(key, 0.0) + w

        s_w = (1.0 - AFF_DECAY ** lp) / (1.0 - AFF_DECAY) if lp > 0 else 0.0
        e_exp = p_pair * s_w
        denom = 2.0 * K * sum(ANCHOR_DECAY ** m for m in range(M))
        if denom <= 0.0:
            denom = 1.0

        echo = {}
        for n in numbers:
            s = 0.0
            for m in range(M):
                aw = ANCHOR_DECAY ** m
                for a in sorted(seq[m]):
                    if a == n or a < 1 or a > N:
                        continue
                    key = (a, n) if a < n else (n, a)
                    cw = pair_w.get(key, 0.0)
                    s += aw * (cw + AFF_SMOOTH) / (e_exp + AFF_SMOOTH)
            echo[n] = s / denom

        # 2) momentum de frecventa pe fereastra HL
        hl = min(HL, L)
        freq = dict.fromkeys(numbers, 0)
        for age in range(hl):
            for x in seq[age]:
                if 1 <= x <= N:
                    freq[x] += 1
        mu_f = hl * p
        sd_f = math.sqrt(hl * p * (1.0 - p))
        if sd_f <= 0.0:
            sd_f = 1.0

        # 3) deficit z pe sectoare de 10
        sw = min(SEC_WINDOW, L)
        s_hits = [0] * S
        for age in range(sw):
            for x in seq[age]:
                if 1 <= x <= N:
                    s_hits[(x - 1) // 10] += 1
        sec_term = {}
        for si in range(S):
            q = (sec_hi[si] - sec_lo[si] + 1) / float(N)
            mu_s = sw * q
            sd_s = math.sqrt(sw * q * (1.0 - q))
            z_s = (s_hits[si] - mu_s) / sd_s if sd_s > 0 else 0.0
            for n in range(sec_lo[si], sec_hi[si] + 1):
                sec_term[n] = -z_s

        # 4) reversiune usoara fata de z pe termen lung
        mu = T * p
        sd = math.sqrt(T * p * (1.0 - p) + 1e-9)

        DIAG['echo_min'] = min(echo.values())
        DIAG['echo_max'] = max(echo.values())

        scores = {}
        for n in numbers:
            t_z = -math.tanh(((counts.get(n, 0) - mu) / sd) / 2.0)
            t_f = (freq[n] - mu_f) / sd_f
            scores[n] = W_AFF * echo[n] + W_FREQ * t_f + W_SEC * sec_term[n] + W_Z * t_z

        return sorted(numbers, key=lambda n: (-scores[n], -counts.get(n, 0), n))

    return rank