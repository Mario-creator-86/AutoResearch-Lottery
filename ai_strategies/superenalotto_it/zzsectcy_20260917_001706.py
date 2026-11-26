NAME = "zzsectcy"
DESCRIPTION = 'Ciclu de intarziere pe sectoare de SECTOR_SIZE (raport gap curent / gap asteptat, tanh), rezonanta multiplicativa intre gap-ul numeric si cel sectorial, echo exponential pe ultimele ANCHOR_DRAWS extrageri si deviatie z macro cu semn configurabil; DIAG expune sector_z, resonance, echo, sector_flips si last_res_span pentru gate-urile de dinaintea ablaticiei.'
SOURCE = 'model cantitativ + 2026-09-17'

import math

# ---- hiperparametri expusi pentru grid-search (Decision Ledger) ----
W_SEC = 0.7          # ciclu sectorial {0.35, 0.7, 1.05}
SECTOR_SIZE = 9      # latime sector {5, 7, 9}
W_DEV = 0.4          # deviatie z macro {0, 0.4, 0.8}
DEV_SIGN = -1.0      # semn z: {+1 momentum, -1 reversie}
W_GAP = 0.5          # rezonanta gap numeric x sectorial {0.3, 0.5, 0.8}
W_LAG = 0.9          # echo short-lag {0.4, 0.9, 1.4}
LAG_DEEP = 0.45      # canal deep-lag {0, 0.45}
ANCHOR_DRAWS = 2     # fereastra de ancore {1, 2, 3}
LAG_TAU = 2.0        # tau de decay pentru echo
TK = 1.0             # scala tanh pe raportul de gap
RCAP = 8.0           # plafon pe raportul de gap

DIAG = {
    'calls': 0,
    'sector_flips': 0,
    'last_res_span': 0.0,
    'last_sector_z': None,
    'last_resonance': None,
    'last_echo': None,
    'echo_min': None,
    'echo_max': None,
    'res_min': None,
    'res_max': None,
    'last_sector_argmax': None,
}


def _p_sector(N, K, sz):
    # P(cel putin un numar din sector intr-o extragere) = 1 - C(N-sz,K)/C(N,K)
    prod = 1.0
    for i in range(K):
        prod *= (N - sz - i) / float(N - i)
    return 1.0 - prod


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    SS = max(2, int(SECTOR_SIZE))
    S = (N + SS - 1) // SS
    sec_of = [0] * (N + 1)
    sec_bounds = []
    for s in range(S):
        lo = s * SS + 1
        hi = min((s + 1) * SS, N)
        sec_bounds.append((lo, hi))
        for n in range(lo, hi + 1):
            sec_of[n] = s
    p_sec = [_p_sector(N, K, hi - lo + 1) for (lo, hi) in sec_bounds]
    E_sec_gap = [1.0 / p for p in p_sec]
    p_num = K / float(N)
    E_num_gap = 1.0 / p_num

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        T = float(draw_index) if draw_index > 0 else 1.0
        mu = T * p_num
        sd = math.sqrt(T * p_num * (1.0 - p_num))
        if sd <= 0.0:
            sd = 1.0

        # 1) ciclu de gap pe sectoare
        sector_z = [0.0] * S
        sec_term = [0.0] * (N + 1)
        for s in range(S):
            lo, hi = sec_bounds[s]
            g = draw_index
            for n in range(lo, hi + 1):
                ls = last_seen.get(n)
                if ls is not None:
                    a = draw_index - ls
                    if a < g:
                        g = a
            r = g / E_sec_gap[s]
            if r > RCAP:
                r = RCAP
            z = math.tanh((r - 1.0) / TK)
            sector_z[s] = z
            for n in range(lo, hi + 1):
                sec_term[n] = z
        DIAG['last_sector_z'] = list(sector_z)
        argmax_s = max(range(S), key=lambda i: sector_z[i])
        prev_am = DIAG['last_sector_argmax']
        if prev_am is not None and prev_am != argmax_s:
            DIAG['sector_flips'] += 1
        DIAG['last_sector_argmax'] = argmax_s

        # 2) rezonanta gap numeric x sectorial
        resonance = [0.0] * (N + 1)
        for n in numbers:
            ls = last_seen.get(n)
            g = (draw_index - ls) if ls is not None else draw_index
            r = g / E_num_gap
            if r > RCAP:
                r = RCAP
            resonance[n] = math.tanh((r - 1.0) / TK) * sector_z[sec_of[n]]
        DIAG['last_resonance'] = list(resonance[1:])
        rmin = min(resonance[1:])
        rmax = max(resonance[1:])
        DIAG['res_min'] = rmin
        DIAG['res_max'] = rmax
        DIAG['last_res_span'] = rmax - rmin

        # 3) echo exponential pe ancorele din ultimele ANCHOR_DRAWS extrageri
        A = max(1, int(ANCHOR_DRAWS))
        echo = [0.0] * (N + 1)
        L = len(recent_deque)
        if L > 0:
            def mean_age(st):
                tot = 0.0
                cnt = 0
                for n in st:
                    ls = last_seen.get(n)
                    if ls is not None:
                        tot += draw_index - ls
                        cnt += 1
                return (tot / cnt) if cnt > 0 else float('inf')

            newest_last = mean_age(recent_deque[-1]) <= mean_age(recent_deque[0])
            seq = list(recent_deque)
            if newest_last:
                seq.reverse()  # seq[0] = extragerea cea mai recenta
            for age in range(len(seq)):
                w = math.exp(-age / LAG_TAU)
                bw = W_LAG * w if age < A else LAG_DEEP * W_LAG * w
                if bw <= 0.0:
                    continue
                for n in seq[age]:
                    if 1 <= n <= N and bw > echo[n]:
                        echo[n] = bw
        DIAG['last_echo'] = list(echo[1:])
        DIAG['echo_min'] = min(echo[1:])
        DIAG['echo_max'] = max(echo[1:])

        # 4) deviatia z macro (semn configurabil) + combinare liniara
        scores = [0.0] * (N + 1)
        for n in numbers:
            z_n = (counts.get(n, 0) - mu) / sd
            scores[n] = (W_SEC * sec_term[n]
                         + W_GAP * resonance[n]
                         + echo[n]
                         + W_DEV * (DEV_SIGN * z_n))
        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank