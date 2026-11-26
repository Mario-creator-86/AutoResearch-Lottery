import math

NAME = "zewmaflu"
DESCRIPTION = 'Compun z-score EWMA rapid, flux de co-aparitie pe graf (message-passing cantarit cu puterea vecinilor) si sticky/overdue pe gap; sortare descrescatoare dupa scorul compus.'
SOURCE = 'q-engineer + 2026-09-08'


def create(config):
    N = config.number_range
    K = config.numbers_drawn
    H_FAST = 35.0      # half-life EWMA frecventa
    H_FLOW = 45.0      # half-life co-aparitii
    TAU_STICKY = 5.0   # scara sticky
    GAP0 = 12.0        # prag overdue
    W_FAST = 1.0
    W_LT = 0.40
    W_FLOW = 0.55
    W_STICKY = 0.45
    W_OVER = 0.12
    CUT_FAST = 240
    CUT_FLOW = 120

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        draws = list(recent_deque)
        L = len(draws)

        # --- EWMA rapida pe aparitii ---
        ef = [0.0] * (N + 1)
        sw = 0.0
        lo = L - min(L, CUT_FAST)
        for i in range(lo, L):
            w = 0.5 ** ((L - 1 - i) / H_FAST)
            sw += w
            d = draws[i]
            for n in d:
                if 1 <= n <= N:
                    ef[n] += w
        exp_fast = sw * K / N
        sig_fast = math.sqrt(exp_fast) if exp_fast > 1e-9 else 1.0

        # --- frecventa pe termen lung (counts) ca z-score ---
        vals = [counts.get(n, 0) for n in range(1, N + 1)]
        mean_c = sum(vals) / N
        var_c = sum((v - mean_c) * (v - mean_c) for v in vals) / N
        std_c = math.sqrt(var_c)
        if std_c < 1e-9:
            std_c = 1.0

        # --- graf de co-aparitie ponderat (ferestra scurta) ---
        F = [None] * (N + 1)
        for a in range(1, N + 1):
            F[a] = [0.0] * (N + 1)
        lof = L - min(L, CUT_FLOW)
        for i in range(lof, L):
            w = 0.5 ** ((L - 1 - i) / H_FLOW)
            d = [n for n in draws[i] if 1 <= n <= N]
            m = len(d)
            for x in range(m):
                a = d[x]
                Fa = F[a]
                for y in range(m):
                    b = d[y]
                    if a != b:
                        Fa[b] += w

        # puterea EWMA a fiecarui nod (pentru message-passing)
        strength = [0.0] * (N + 1)
        for n in range(1, N + 1):
            zf = (ef[n] - exp_fast) / sig_fast
            strength[n] = 0.5 + 0.5 * max(0.0, zf)

        # flux de ordin 1: cat de mult curge spre n din vecini puternici
        flow = [0.0] * (N + 1)
        for n in range(1, N + 1):
            Fn = F[n]
            s = 0.0
            for m2 in range(1, N + 1):
                wv = Fn[m2]
                if wv > 0.0:
                    s += wv * strength[m2]
            flow[n] = s
        mf = sum(flow[1:]) / N
        sf = math.sqrt(sum((flow[n] - mf) * (flow[n] - mf) for n in range(1, N + 1)) / N)
        if sf < 1e-9:
            sf = 1.0

        # --- scor compus ---
        out = []
        for n in range(1, N + 1):
            zf = (ef[n] - exp_fast) / sig_fast
            zc = (vals[n - 1] - mean_c) / std_c
            zfl = (flow[n] - mf) / sf
            ls = last_seen.get(n)
            if ls is None:
                gap = 500.0
            else:
                gap = float(draw_index - ls)
            sticky = math.exp(-gap / TAU_STICKY)
            over = max(0.0, gap - GAP0) / GAP0
            score = (W_FAST * zf
                     + W_LT * zc
                     + W_FLOW * math.tanh(zfl)
                     + W_STICKY * sticky
                     + W_OVER * math.tanh(over))
            out.append((n, score))

        out.sort(key=lambda t: (-t[1], t[0]))
        return [n for n, _ in out]

    return rank