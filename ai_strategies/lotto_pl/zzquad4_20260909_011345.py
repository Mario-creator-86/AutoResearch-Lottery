NAME = "zzquad4"
DESCRIPTION = 'Scor liniar pe 4 componente z-normalizate: (1) afinitate de perechi co-aparute, ponderata EWMA, intre fiecare numar si ancora formata din ultimele K_TAIL extrageri; (2) deviatia frecventei EWMA fata de asteptat (6/49) in fereastra recenta; (3) aliniere pozitionala gaussiana fata de pozitia mecanica asteptata 5*(n-1)/(N-1); (4) cadenta de intarziere normalizata fata de rata istorica per numar.'
SOURCE = 'quant-4component-model + 2026-09-08'

import numpy as np

# Hiperparametri patch-abili pentru grid-search / ablatie (cititi la fiecare apel):
W_PAIR = 0.55    # [0, 1.5]
W_DEV = 0.40     # [-0.8, 1.0]
W_POS = 0.20     # [0, 1]
W_CAD = 0.30     # [0, 1]
GAMMA_P = 0.98   # [0.90, 0.99] decay EWMA pe varsta extragerii (newest_last)
K_TAIL = 2       # {1, 2, 3, 5} marimea ancorei de perechi
SIGMA_POS = 8.0  # [4, 12] latimea kernelului gaussian positional


def create(config):
    N = int(config.number_range)
    ND = int(config.numbers_drawn)
    numbers = list(range(1, N + 1))
    pos_axis = np.arange(ND, dtype=np.float64)
    mu_pos = (ND - 1.0) * (np.arange(1, N + 1, dtype=np.float64) - 1.0) / float(N - 1)
    gcache = {'sig': None, 'G': None}

    def _gmat():
        if gcache['sig'] != SIGMA_POS:
            gcache['G'] = np.exp(-((pos_axis[None, :] - mu_pos[:, None]) ** 2) / (2.0 * float(SIGMA_POS) ** 2))
            gcache['sig'] = SIGMA_POS
        return gcache['G']

    def _z(x):
        x = np.asarray(x, dtype=np.float64)
        sd = x.std()
        if not np.isfinite(sd) or sd < 1e-12:
            return np.zeros(N, dtype=np.float64)
        return (x - x.mean()) / sd

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # (4) cadenta adaptiva: (gap - gap_asteptat)/sqrt(gap_asteptat), rata din counts
        gaps = np.empty(N, dtype=np.float64)
        cnts = np.empty(N, dtype=np.float64)
        for j, n in enumerate(numbers):
            ls = last_seen.get(n)
            gaps[j] = draw_index - (ls if ls is not None else -1)
            cnts[j] = counts.get(n, 0)
        m_exp = (draw_index + 1.0) / (cnts + 0.5)
        cad_raw = (gaps - m_exp) / np.sqrt(m_exp + 1.0)
        zcad = _z(cad_raw)

        # fereastra recenta; deque newest_last => elementul -1 este cea mai recenta extragere
        draws = list(recent_deque)
        if recent_window:
            rw = int(recent_window)
            if rw > 0 and len(draws) > rw:
                draws = draws[-rw:]
        L = len(draws)
        if L == 0:
            order = np.argsort(-cad_raw, kind='stable')
            return [int(t) + 1 for t in order]

        ages = (L - 1) - np.arange(L, dtype=np.float64)
        w = np.power(float(GAMMA_P), ages)

        col_list = []
        pos_list = []
        sizes = np.empty(L, dtype=np.int64)
        for i, s in enumerate(draws):
            sizes[i] = len(s)
            for p, n in enumerate(sorted(s)):
                col_list.append(n - 1)
                pos_list.append(p)
        cols = np.fromiter(col_list, dtype=np.int64, count=len(col_list))
        posv = np.minimum(np.fromiter(pos_list, dtype=np.int64, count=len(pos_list)), ND - 1)
        w_rep = np.repeat(w, sizes)
        rows = np.repeat(np.arange(L, dtype=np.int64), sizes)

        M = np.zeros((L, N), dtype=np.float64)
        M[rows, cols] = 1.0

        # (2) deviatia EWMA a frecventei in fereastra recenta
        ec = np.bincount(cols, weights=w_rep, minlength=N)
        zdev = _z(ec)

        # (1) afinitate de perechi: co-aparitii ponderate cu ancora ultimelor K extrageri
        zpair = np.zeros(N, dtype=np.float64)
        K = int(K_TAIL)
        if 0 < K < L:
            anchor = set()
            for s in draws[L - K:]:
                anchor.update(s)
            L2 = L - K
            C = (M[:L2] * w[:L2, None]).T @ M[:L2]
            ai = np.fromiter((x - 1 for x in anchor), dtype=np.int64, count=len(anchor))
            zpair = _z(C[:, ai].sum(axis=1))

        # (3) aliniere pozitionala: media ponderata a kernelului pe pozitia sortata
        G = _gmat()
        gvals = G[cols, posv]
        possum = np.bincount(cols, weights=w_rep * gvals, minlength=N)
        zpos = _z(possum / (ec + 0.5))

        score = W_PAIR * zpair + W_DEV * zdev + W_POS * zpos + W_CAD * zcad
        order = np.argsort(-score, kind='stable')
        return [int(t) + 1 for t in order]

    return rank