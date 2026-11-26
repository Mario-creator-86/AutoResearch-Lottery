import math
import numpy as np

NAME = "zzaffine"
DESCRIPTION = 'Afinitate Markov de co-occurinta ponderata recen (decay 0.9, diagonala zero) + propagare de ordin 2 pe graf, combinate cu echo exponential pe gap (din last_seen), due liniar, momentum pe fereastra recenta si z-score de termen lung cu shrinkage; orientarea deque-ului este detectata la runtime prin cross-check cu last_seen.'
SOURCE = 'glm strategist + 2026-09-08'

# ponderi fixe, plasate la mijlocul grilelor de scan planificate
W_Z = 0.5      # z pe termen lung
W_MOM = 0.35   # momentum fereastra recenta
W_ECHO = 0.6   # echo exp(-gap/TAU), scan 0.2-1.0
W_DUE = 0.15   # due liniar, scan semn +-0.25
W_AFF = 0.4    # afinitate Markov ordin 1, scan 0-0.8
W_MSG2 = 0.3   # pas ordin 2, scan 0-1.5
K_SHRINK = 3.0 # shrinkage z, scan {1,3,5,8}
TAU_ECHO = 2.0 # constanta de timp echo (gap mediu asteptat ~ 1/p)
GAMMA_AFF = 0.9   # decay recen pe graful de co-occurinta
MAX_CO_DRAWS = 60 # fereastra eficienta pentru gram (w < 0.001 dincolo)


def create(config):
    N = int(config.number_range)
    KD = int(config.numbers_drawn)
    p = KD / float(N) if N > 0 else 0.5

    def _std(v):
        s = float(np.std(v))
        if s < 1e-12:
            return np.zeros_like(v)
        return (v - np.mean(v)) / s

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        di = int(draw_index)
        n_draws = max(di, 1)

        # 1) z pe termen lung cu shrinkage c/(c+k)
        c = np.array([float(counts.get(n, 0)) for n in numbers])
        mu = n_draws * p
        sig = math.sqrt(max(n_draws * p * (1.0 - p), 1e-9))
        z = ((c - mu) / sig) * (c / (c + K_SHRINK))

        # 2) gap din last_seen (sursa de adevar pentru recenta)
        big = float(n_draws + 1)
        gap = np.array([
            float(di - last_seen[n]) if (n in last_seen and last_seen[n] >= 0) else big
            for n in numbers
        ])
        gap = np.clip(gap, 0.0, 50.0)
        echo = np.exp(-gap / TAU_ECHO)
        due = gap

        seq = list(recent_deque) if recent_deque is not None else []
        Wn = len(seq)
        mom = np.zeros(N)
        aff1 = np.zeros(N)
        aff2 = np.zeros(N)
        if Wn > 0:
            # 3) orientare deque: ipoteza seq[-1]=cel mai nou, verificata prin last_seen
            def _fresh(s):
                t = 0
                for n in s:
                    if n in last_seen and last_seen[n] >= 0:
                        t += di - last_seen[n]
                    else:
                        t += di + 1
                return t
            if Wn >= 2 and _fresh(seq[0]) < _fresh(seq[-1]):
                newest_first = seq
            else:
                newest_first = seq[::-1]

            A = np.array([[1.0 if n in s else 0.0 for n in numbers] for s in newest_first])

            # 4) momentum pe fereastra completa
            cnt = A.sum(axis=0)
            sd_w = math.sqrt(max(Wn * p * (1.0 - p), 1e-9))
            mom = (cnt - Wn * p) / sd_w

            # 5) afinitate Markov: co-occurinta ponderata recen, fara diagonala
            m = min(Wn, MAX_CO_DRAWS)
            Asub = A[:m]
            w = np.array([GAMMA_AFF ** i for i in range(m)])
            M = (Asub * w[:, None]).T @ Asub
            np.fill_diagonal(M, 0.0)
            a1 = M @ Asub[0]
            aff1 = a1
            # 6) pas de ordin 2: flux pe graf
            aff2 = M @ _std(a1)

        score = (
            W_Z * _std(z)
            + W_MOM * _std(mom)
            + W_ECHO * _std(echo)
            + W_DUE * _std(due)
            + W_AFF * _std(aff1)
            + W_MSG2 * _std(aff2)
        )
        smap = {n: float(score[i]) for i, n in enumerate(numbers)}
        return sorted(numbers, key=lambda n: (-smap[n], n))

    return rank