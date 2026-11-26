import math

NAME = "zzglrmar"
DESCRIPTION = 'Dovada binomiala GLR (log-odds deviatie in fereastra recenta vs rata corecta 6/40, cu shrinkage Beta) + faza de renoire per-numar (gap normalizat la propriul interval mediu istoric, kernel Gaussian pe ln(phi)) + lift Markov de ordinul 1 (auto-tranzitii per-numar), combinate aditiv in log-space.'
SOURCE = 'model + 2026-09-26'


def create(config):
    N = config.number_range          # ex. 40
    WD = config.numbers_drawn        # ex. 6
    p_ref = WD / float(N)            # rata corecta per numar per extragere

    # greutati / hiperparametri (deterministi, fixati)
    W_GLR = 0.6
    W_PHASE = 0.5
    W_MK = 0.3
    W_LR = 0.15
    SHRINK = 60.0                    # pseudo-numarare Beta pentru GLR
    SIGMA = 0.9                      # latimea kernelului de faza (in log phi)
    S_MK = 8.0                       # smoothing pentru ratele Markov
    RATE_LO = 0.5 * p_ref
    RATE_HI = 2.0 * p_ref
    G_CLIP = 2.5

    def _logit(x):
        return math.log(x / (1.0 - x))

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        tot = max(draw_index, 1)     # extrageri completate pana acum
        dq = list(recent_deque)
        T = len(dq)
        if T == 0:
            scores = {n: counts.get(n, 0) for n in numbers}
            return sorted(numbers, key=lambda n: (-scores[n], n))

        # orientare determinista: capatul care coicide cu ultima extragere reala
        # (numerele cu last_seen == draw_index-1 formeaza exact ultima extragere)
        target = set(n for n in numbers if last_seen.get(n) == draw_index - 1)
        if target:
            ov_first = len(dq[0] & target)
            ov_last = len(dq[-1] & target)
            chrono = dq[::-1] if ov_first > ov_last else dq
        else:
            chrono = dq              # default: capatul din dreapta e cel mai recent

        # aparitii in fereastra (cronologic)
        c_win = dict.fromkeys(numbers, 0)
        for s in chrono:
            for n in s:
                c_win[n] += 1

        # tranzitii de ordinul 1: prev=chrono[k-1], succ=chrono[k]
        Tm = T - 1
        A1 = dict.fromkeys(numbers, 0)   # nr. de prev-draws care contin n
        B1 = dict.fromkeys(numbers, 0)   # dintre acestea, cate au succesor cu n
        if Tm >= 1:
            for k in range(1, T):
                prev = chrono[k - 1]
                succ = chrono[k]
                for n in prev:
                    A1[n] += 1
                    if n in succ:
                        B1[n] += 1

        csucc = {n: (c_win[n] - 1) if n in chrono[0] else c_win[n] for n in numbers}

        lp_ref = _logit(p_ref)
        base_lr = tot * p_ref + 1.0
        last_set = chrono[-1]

        scores = {}
        for n in numbers:
            cn = counts.get(n, 0)

            # (1) GLR: log-odds deviatie cu shrinkage Beta(p_ref*k, (1-p_ref)*k)
            r = (c_win[n] + SHRINK * p_ref) / (T + SHRINK)
            glr = _logit(r) - lp_ref
            if glr > G_CLIP:
                glr = G_CLIP
            elif glr < -G_CLIP:
                glr = -G_CLIP

            # (2) faza de renoire: gap curent / propriul interval mediu istoric
            if cn >= 1:
                mu = tot / float(cn)
            else:
                mu = float(tot)
            g = tot - last_seen[n] if n in last_seen else tot
            if g < 1:
                g = 1
            phi = g / mu
            if phi > 3.0:
                phi = 3.0
            z = math.log(phi) / SIGMA
            ph = math.exp(-0.5 * z * z)

            # (3) Markov: p1 = P(apare | a aparut la pasul anterior), p0 = P(apare | a lipsit)
            if Tm >= 1:
                p1 = (B1[n] + S_MK * p_ref) / (A1[n] + S_MK)
                p0 = ((csucc[n] - B1[n]) + S_MK * p_ref) / ((Tm - A1[n]) + S_MK)
                if p1 < RATE_LO:
                    p1 = RATE_LO
                elif p1 > RATE_HI:
                    p1 = RATE_HI
                if p0 < RATE_LO:
                    p0 = RATE_LO
                elif p0 > RATE_HI:
                    p0 = RATE_HI
                mk = math.log(p1) if n in last_set else math.log(p0)
            else:
                mk = 0.0

            # stabilizator minimal: deviatie log pe frecventa pe termen lung
            lr = math.log((cn + 1.0) / base_lr)

            scores[n] = W_GLR * glr + W_PHASE * ph + W_MK * mk + W_LR * lr

        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank