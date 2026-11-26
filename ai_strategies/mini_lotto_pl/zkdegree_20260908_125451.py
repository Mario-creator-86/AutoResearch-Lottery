NAME = "zkdegree"
DESCRIPTION = ('Estimare neparametrica de densitate in spatiul combinatiilor: fiecare extragere recenta '
               'este un punct cantarit exponential dupa recenta, iar similaritatea cu un bilet candidat '
               'se masoara printr-un kernel Jaccard pe submultimi. Biletul se construieste greedy, '
               'adaugand la fiecare pas numarul cu cel mai mare castig marginal de densitate kernel '
               '(forma inchisa 1/(s+D-c)), plus un mic prior z pe frecventa pe termen lung.')
SOURCE = 'glm-5.3-flash:cloud + 2026-09-08'

import math


def create(config):
    N = config.number_range        # ex. 42
    D = config.numbers_drawn      # ex. 5
    T = min(getattr(config, 'ticket_size', D) or D, N)

    LAM = 0.01        # rata de decay a ponderilor de recenta (half-life ~69 extrageri)
    WZ = 0.30         # ponderea priorului z pe termen lung
    MIN_DRAWS = 10    # prag minim de extrageri in fereastra pentru kernel

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))

        # --- z-score pe termen lung fata de frecventa asteptata ---
        p = D / float(N)
        nd = float(draw_index) if draw_index and draw_index > 0 else 1.0
        sd = math.sqrt(nd * p * (1.0 - p))
        if sd <= 0.0:
            sd = 1e-09
        z = {}
        for n in numbers:
            z[n] = (counts.get(n, 0) - nd * p) / sd

        # --- mascari bit pe fereastra recenta (deque[-1] = cea mai noua extragere) ---
        draws = []
        for sset in recent_deque:
            m = 0
            for x in sset:
                if 1 <= x <= N:
                    m |= 1 << (x - 1)
            draws.append(m)
        L = len(draws)

        picked = []
        wf = dict((n, 0.0) for n in numbers)

        if L >= MIN_DRAWS:
            weights = [math.exp(-LAM * (L - 1 - i)) for i in range(L)]
            contain = dict((n, []) for n in numbers)
            for i in range(L):
                wgt = weights[i]
                mm = draws[i]
                b = 0
                while mm:
                    if mm & 1:
                        contain[b + 1].append(i)
                        wf[b + 1] += wgt
                    mm >>= 1
                    b += 1

            # --- constructie greedy: castig marginal de densitate Jaccard ---
            # dK = w_i * [K(c+1,s+1) - K(c,s)] = w_i / (s + D - c_i)
            c_arr = [0] * L          # |S curent intersect draw_i|
            s = 0                    # dimensiunea setului curent
            in_pick = set()
            for _step in range(T):
                base = s + D
                best_n = None
                best_sc = None
                for n in numbers:
                    if n in in_pick:
                        continue
                    sc = WZ * z[n]
                    for i in contain[n]:
                        sc += weights[i] / (base - c_arr[i])
                    if best_sc is None or sc > best_sc:
                        best_sc = sc
                        best_n = n
                if best_n is None:
                    break
                picked.append(best_n)
                in_pick.add(best_n)
                for i in contain[best_n]:
                    c_arr[i] += 1
                s += 1

        # --- completarea permutarii: restul dupa frecventa recenta cantarita + z ---
        chosen = set(picked)
        rest = [n for n in numbers if n not in chosen]
        rest.sort(key=lambda n: (-(wf[n] + WZ * z[n]), n))
        return picked + rest

    return rank