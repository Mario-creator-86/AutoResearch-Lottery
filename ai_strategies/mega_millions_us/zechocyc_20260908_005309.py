import math

NAME = "zechocyc"
DESCRIPTION = ('Compozit z-standardizat: deviatia fata de frecventa asteptata pe termen lung, '
               'forma recenta pe fereastra scurta, ecou exponential al gap-ului individual, '
               'ciclul de intarziere pe sectoare si co-occurinta cu ultima extragere.')
SOURCE = 'echocycle log-linear hazard mix, 2026-09-08'


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        p = K / float(N) if N > 0 else 0.0

        # numarul efectiv de extrageri observate, robust: sum(counts)/K
        total = 0
        for n in numbers:
            total += counts.get(n, 0)
        T = total / float(K) if K > 0 else 0.0

        mu = p * T
        sd = math.sqrt(max(T * p * (1.0 - p), 1e-12))

        # histograma recenta: cap la 40 extrageri (cost constant per apel)
        H = 0
        rc = {}
        pe = {}
        last_set = set()
        if recent_deque is not None and len(recent_deque) > 0:
            prev_idx = draw_index - 1
            for n in numbers:
                if last_seen.get(n, -1) == prev_idx:
                    last_set.add(n)
            head = 40 if len(recent_deque) > 40 else len(recent_deque)
            i = 0
            for d in recent_deque:
                if i >= head:
                    break
                i += 1
                if not d:
                    continue
                H += 1
                shared = 0
                for m in d:
                    if m in last_set:
                        shared += 1
                if d == last_set:
                    shared = 0
                if shared:
                    for x in d:
                        pe[x] = pe.get(x, 0) + shared
                for x in d:
                    rc[x] = rc.get(x, 0) + 1

        mu_r = p * H
        sd_r = math.sqrt(max(H * p * (1.0 - p), 1e-12))

        # prob. ca o extragere aleatoare sa share-uiasca >= 1 numar cu ultima extragere
        q_no = 1.0
        for i in range(K):
            q_no *= (N - K - i) / float(N - i)
        q_ov = 1.0 - q_no
        mu_pe = H * p * q_ov
        sd_pe = math.sqrt(max(mu_pe, 1e-12))

        # vid de sector: 10 sectoare a cate ceil(N/10) numere
        S = 10
        sect = (N + S - 1) // S
        NEG = -1 << 30
        sec_last = {}
        for n in numbers:
            s = (n - 1) // sect
            ls = last_seen.get(n, NEG)
            if ls > sec_last.get(s, NEG):
                sec_last[s] = ls

        scores = {}
        for n in numbers:
            zl = (counts.get(n, 0) - mu) / sd
            if zl > 3.0:
                zl = 3.0
            elif zl < -3.0:
                zl = -3.0

            if H > 0:
                zr = (rc.get(n, 0) - mu_r) / sd_r
            else:
                zr = 0.0
            if zr > 3.0:
                zr = 3.0
            elif zr < -3.0:
                zr = -3.0

            ls = last_seen.get(n, None)
            if ls is None or ls < 0:
                echo = 0.0
            else:
                g = draw_index - ls
                if g < 1:
                    g = 1
                echo = 1.3 * math.exp(-(g - 1) / 3.0)

            s = (n - 1) // sect
            v = draw_index - sec_last.get(s, NEG)
            if v < 0:
                v = 0
            elif v > 30:
                v = 30
            dv = (v - 2.5) / 2.0
            cyc = math.exp(-0.5 * dv * dv)

            if H > 0:
                zp = (pe.get(n, 0) - mu_pe) / sd_pe
            else:
                zp = 0.0
            if zp > 2.0:
                zp = 2.0
            elif zp < -2.0:
                zp = -2.0

            scores[n] = zl + 0.55 * zr + 0.45 * echo + 0.35 * cyc + 0.25 * zp

        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank