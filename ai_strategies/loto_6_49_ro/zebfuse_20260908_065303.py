import math

NAME = "zebfuse"
DESCRIPTION = ('Fuziune Borda pe ranguri a cinci semnale empirice-Bayes: lift de frecventa '
               'globala, momentum recent shrink-uit, tilt de gap pe model de renewall '
               'personalizat, deficit de sector in fereastra recenta si gravitatie de '
               'perechi catre extragerile cele mai recente, plus bonus de concordanta.')
SOURCE = 'model: empirical-Bayes rank-fusion; 2026-09-08'

def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    q = K / float(N)

    def _pct(numbers, vals):
        xs = [vals[n] for n in numbers]
        if max(xs) - min(xs) < 1e-12:
            return dict((n, 0.5) for n in numbers)
        order = sorted(numbers, key=lambda n: vals[n])
        m = float(len(order))
        return dict((n, (i + 0.5) / m) for i, n in enumerate(order))

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        D = float(draw_index)
        if D < 2.0:
            D = 2.0
        R = len(recent_deque)
        dec = 0.965
        RR = R if R < 300 else 300
        nb_sec = 7
        size = N // nb_sec if N >= nb_sec else 1
        szs = [0] * nb_sec
        for n in numbers:
            szs[min((n - 1) // size, nb_sec - 1)] += 1

        # (a) lift binomial global
        mu_g = D * q
        sd_g = math.sqrt(max(mu_g * (1.0 - q), 1e-9))
        z_freq = {}
        for n in numbers:
            z_freq[n] = (counts.get(n, 0) - mu_g) / sd_g

        # (b) momentum recent cu decay + shrinkage James-Stein; (d) deficit de sector
        z_rec = dict((n, 0.0) for n in numbers)
        sec = dict((n, 0.0) for n in numbers)
        eff = 0.0
        sums = [0.0] * nb_sec
        if RR > 0:
            wr = dict((n, 0.0) for n in numbers)
            for i in range(R - RR, R):
                s = recent_deque[i]
                w = dec ** (R - 1 - i)
                eff += w
                for m in s:
                    if 1 <= m <= N:
                        wr[m] += w
                        sums[min((m - 1) // size, nb_sec - 1)] += w
            mu_r = q * eff
            sd_r = math.sqrt(max(mu_r * (1.0 - q), 1e-9))
            shrink = eff / (eff + 18.0)
            for n in numbers:
                z_rec[n] = shrink * (wr[n] - mu_r) / sd_r
                k = min((n - 1) // size, nb_sec - 1)
                mu_s = eff * szs[k] * q
                sd_s = math.sqrt(max(mu_s * (1.0 - szs[k] * q), 1e-9))
                sec[n] = (mu_s - sums[k]) / sd_s

        # (c) tilt de gap personalizat (renewall): mu_n din rata proprie
        tilt = {}
        for n in numbers:
            c = float(counts.get(n, 0))
            mu_n = (D + 1.0) / (c + 1.0)
            g = float(draw_index - last_seen.get(n, -1))
            tilt[n] = math.tanh(0.42 * (g - mu_n) / math.sqrt(max(mu_n, 1e-9)))

        # (e) gravitatie de perechi catre ancorele ultimelor 3 extrageri
        grav = dict((n, 0.0) for n in numbers)
        if RR > 0:
            cooc = {}
            for i in range(R - RR, R):
                arr = [m for m in recent_deque[i] if 1 <= m <= N]
                w = dec ** (R - 1 - i)
                for ai in range(len(arr)):
                    for bi in range(ai + 1, len(arr)):
                        a = arr[ai]
                        b = arr[bi]
                        key = (a, b) if a < b else (b, a)
                        cooc[key] = cooc.get(key, 0.0) + w
            for j in range(min(R, 3)):
                idx = R - 1 - j
                wa = 0.85 ** j
                for m in recent_deque[idx]:
                    if not (1 <= m <= N):
                        continue
                    for n in numbers:
                        if n == m:
                            continue
                        key = (n, m) if n < m else (m, n)
                        grav[n] += wa * cooc.get(key, 0.0)

        # fuziune Borda: percentile centrate + concordanta
        pF = _pct(numbers, z_freq)
        pR = _pct(numbers, z_rec)
        pT = _pct(numbers, tilt)
        pS = _pct(numbers, sec)
        pG = _pct(numbers, grav)
        W = (0.95, 1.45, 0.50, 0.65, 0.75)
        final = {}
        for n in numbers:
            comps = (pF[n], pR[n], pT[n], pS[n], pG[n])
            sc = 0.0
            for w, x in zip(W, comps):
                sc += w * (2.0 * x - 1.0)
            conc = 0
            for x in comps:
                if x >= 0.70:
                    conc += 1
            if conc > 2:
                sc += 0.22 * (conc - 2)
            final[n] = sc
        return sorted(numbers, key=lambda n: (-final[n], n))

    return rank