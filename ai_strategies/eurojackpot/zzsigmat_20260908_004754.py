import numpy as np

NAME = "zzsigmat"
DESCRIPTION = ('Potrivire de semnatura pe lag-uri: pentru fiecare numar, ultimii K=4 biti binari de aparitie '
               'formeaza un pattern; probabilitatea de aparitie la extragerea urmatoare este estimata printr-o '
               'regresie kernel pe distanta Hamming fata de toate pozitiile istorice din fereastra recenta, '
               'cu shrinkage Beta catre rata de baza a numarului.')
SOURCE = 'model self + 2026-09-07'


def create(config):
    N = int(getattr(config, 'number_range', 50))
    K = 4        # adancimea semnaturii (lag-uri 1..K)
    DECAY = 0.55 # decay kernelului pe numarul de potriviri Hamming
    ALPHA = 16.0 # pseudo-numarari Beta pentru shrinkage
    PS = 30.0    # pseudo-observatii pentru stabilizarea ratei de baza
    MAXT = 150   # fereastra maxima folosita
    MINT = 48    # minim de extrageri pentru activarea modelului
    wpow = np.array([DECAY ** (K - m) for m in range(K + 1)], dtype=np.float64)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        Tfull = len(recent_deque)
        if Tfull < MINT:
            base = {n: counts.get(n, 0) for n in numbers}
            return sorted(numbers, key=lambda n: (-base[n], n))
        window = list(recent_deque)[-MAXT:]
        T = len(window)
        X = np.zeros((T, N), dtype=np.float64)
        for t, s in enumerate(window):
            for n in s:
                ni = int(n)
                if 1 <= ni <= N:
                    X[t, ni - 1] = 1.0
        pbar = float(X.mean())
        p = X.mean(axis=0)
        p = np.clip((p * T + pbar * PS) / (T + PS), 0.03, 0.6)
        nt = T - K
        Mm = np.full((nt, N), K, dtype=np.int16)
        for k in range(1, K + 1):
            bits = X[K - k: T - k, :]   # x_{t-k} pentru t = K..T-1
            edge = X[T - k, :]          # conditia de margine pentru predictia pasului urmator
            Mm -= np.abs(bits - edge).astype(np.int16)  # scade nepotrivirile -> Mm = nr. potriviri
        W = wpow[Mm]                    # pondere geometrica pe potrivirile Hamming
        Y = X[K:, :]                    # tinte binare aliniate cu randurile de pattern
        num = (W * Y).sum(axis=0)
        den = W.sum(axis=0)
        shat = (num + ALPHA * p) / (den + ALPHA)  # probabilitatea prezisa, shrinkuita Beta
        return sorted(numbers, key=lambda n: (-shat[n - 1], n))
    return rank