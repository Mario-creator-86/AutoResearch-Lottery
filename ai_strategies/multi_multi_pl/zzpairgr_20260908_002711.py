import numpy as np

NAME = "zzpairgr"
DESCRIPTION = 'Graf de co-aparitie conditionat: z-scoruri de perechi cu ancorele din ultimele 5 extrageri (soft-threshold la 1 sigma), fuzionate neliniar cu deviatia de frecventa pe tot istoricul si cu ciclul de intarziere geometric.'
SOURCE = 'model + 2026-09-08'


def create(config):
    # config: LotteryConfig - folosim doar atributele .number_range si .numbers_drawn
    N = int(config.number_range)
    D = int(config.numbers_drawn)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        try:
            cap = max(1, min(int(recent_window), 400))
        except Exception:
            cap = 200
        draws = list(recent_deque)[-cap:]
        W = len(draws)
        if W < 8:
            base = [counts.get(n, 0) for n in numbers]
            order = sorted(range(N), key=lambda i: (-base[i], numbers[i]))
            return [numbers[i] for i in order]

        # matrice binara fereastra x numere (cronologica, oldest_last)
        A = np.zeros((W, N), dtype=np.float64)
        for i, s in enumerate(draws):
            for n in s:
                if 1 <= n <= N:
                    A[i, n - 1] = 1.0

        r = A.sum(axis=0)  # aparitii in fereastra
        M = A.T @ A        # numar de co-aparitii pe perechi
        np.fill_diagonal(M, 0.0)

        # asteptare sub independenta E=r_n*r_m/W; z Poisson; soft-threshold la 1 sigma
        E = np.outer(r, r) / float(W)
        np.fill_diagonal(E, 0.0)
        Z = (M - E) / np.sqrt(E + 1.0)
        Zt = np.sign(Z) * np.maximum(np.abs(Z) - 1.0, 0.0)

        # ancore: numerele din ultimele K extrageri, pondere exponentiala 0.5^t
        K = min(5, W)
        anchor = np.zeros(N, dtype=np.float64)
        for t in range(K):
            w = 0.5 ** t
            for n in draws[-1 - t]:
                if 1 <= n <= N:
                    anchor[n - 1] += w
        pair = Zt @ anchor

        # deviatie de frecventa pe tot istoricul (regresie spre medie de lunga durata)
        cvec = np.array([counts.get(n, 0) for n in numbers], dtype=np.float64)
        life = -(cvec - cvec.mean()) / (cvec.std() + 1e-9)

        # ciclu de intarziere: gap curent vs geometric(p): medie (1-p)/p, std sqrt(1-p)/p
        p = float(D) / float(N)
        g_mean = (1.0 - p) / p
        g_std = float(np.sqrt(1.0 - p) / p)
        gap = np.array([draw_index - last_seen.get(n, 0) for n in numbers], dtype=np.float64)
        gapz = np.clip((gap - g_mean) / g_std, -4.0, 4.0)

        def zstd(x):
            return (x - x.mean()) / (x.std() + 1e-9)

        score = (1.00 * np.tanh(zstd(pair))
                 + 0.40 * np.tanh(zstd(life))
                 + 0.20 * np.tanh(gapz / 2.0))

        order = sorted(range(N), key=lambda i: (-float(score[i]), numbers[i]))
        return [numbers[i] for i in order]

    return rank