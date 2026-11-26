import math

NAME = "zaffinne"
DESCRIPTION = 'Propagare pe graful de co-aparitii din fereastra recenta: scor = co-aparitii ponderate cu caldura EWMA (z) a vecinilor, centrat la asteptarea nula a perechilor; poarta de entropie moduleaza greutatea afinitatii in functie de structura ferestrei. Fara frecvente pe termen lung, fara gap/overdue, fara balance.'
SOURCE = 'glm strategist + 2026-09-16'

def create(config):
    N = config.number_range
    K = config.numbers_drawn

    LN_N = math.log(N) if N > 1 else 1.0
    H_MIN = (math.log(K) / LN_N) if K > 1 else 0.0
    LAMBDA_PAIR = float(K * (K - 1)) / float(N * (N - 1)) if N > 1 else 0.0
    GAMMA = math.exp(math.log(0.5) / 10.0)  # half-life 10 draws
    W_AFF = 1.0
    W_EW = 0.5

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        win = list(recent_deque)
        wlen = len(win)
        if wlen == 0:
            sc0 = {n: counts.get(n, 0) for n in numbers}
            return sorted(numbers, key=lambda n: (-sc0[n], n))

        # frecvente brute + EWMA pe prezenta (ultimul element = cel mai recent)
        rf = [0] * N
        ew = [0.0] * N
        w = 1.0
        wsum = 0.0
        for i in range(wlen - 1, -1, -1):
            for x in win[i]:
                j = x - 1
                rf[j] += 1
                ew[j] += w
            wsum += w
            w *= GAMMA

        p = float(K) / float(N)
        ew_mean = wsum * p
        ew_std = math.sqrt(max(wsum * p * (1.0 - p), 1e-12))
        ez = [(ew[i] - ew_mean) / ew_std for i in range(N)]

        # poarta de entropie: 1 = fereastra concentrata (structura), 0 = uniforma
        inv_T = 1.0 / (wlen * K)
        H = 0.0
        for i in range(N):
            c = rf[i]
            if c > 0:
                q = c * inv_T
                H -= q * math.log(q)
        Hn = H / LN_N
        rng = 1.0 - H_MIN
        struct = 1.0 - (Hn - H_MIN) / rng if rng > 1e-12 else 0.0
        if struct < 0.0:
            struct = 0.0
        elif struct > 1.0:
            struct = 1.0
        w_aff = W_AFF * (0.5 + 0.5 * struct)

        # contoare de co-aparitii (cheie 0-based unica: i*N + j, i < j)
        pair = {}
        for s in win:
            lst = sorted(s)
            L = len(lst)
            for ai in range(L - 1):
                base = (lst[ai] - 1) * N
                for bi in range(ai + 1, L):
                    key = base + (lst[bi] - 1)
                    pair[key] = pair.get(key, 0) + 1

        Wtot = 0.0
        W2 = 0.0
        for i in range(N):
            zi = ez[i]
            Wtot += zi
            W2 += zi * zi

        # afinitate propagata: pa(n) = sum_m c(n,m) * ez(m)
        pa = [0.0] * N
        for key, c in pair.items():
            j = key % N
            i = key // N
            zi = ez[i]
            zj = ez[j]
            if zi:
                pa[j] += c * zi
            if zj:
                pa[i] += c * zj

        lam = wlen * LAMBDA_PAIR
        s_aff = math.sqrt(max(lam * W2, 1e-6))

        scores = {}
        for i in range(N):
            aff = (pa[i] - lam * (Wtot - ez[i])) / s_aff
            scores[i + 1] = w_aff * aff + W_EW * ez[i]
        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank