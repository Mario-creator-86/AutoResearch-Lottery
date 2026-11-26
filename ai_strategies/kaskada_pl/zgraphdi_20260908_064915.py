import numpy as np
from itertools import chain
from math import sqrt

NAME = "zgraphdi"
DESCRIPTION = ("Hibrid graf-difuziune: ranguri din z-score-uri tanh ale frecventei pe tot istoricul, "
               "amestecate cu un scor de recență ponderat exponential, difuzat un hop peste graful "
               "de co-aparitie al ferestrei recente; poarta de dispersie amplifica momentum-ul doar "
               "la clustering real, plus un mic hedge overdue.")
SOURCE = "cercetare cantitativa proprie (diffuziune pe graf de co-occurrence) + 2026-09-08"

def create(config):
    n = int(config.number_range)      # 24 numere in pool
    nd = int(config.numbers_drawn)    # 12 extrase per tragere

    HALF_LIFE = 18.0   # half-life EWMA pe extrageri
    BETA = 0.40        # ponderea hop-ului de difuziune vs recența marginala
    BASE_W = 1.00      # ponderea tilt-ului long-run
    TANH_SHARP = 0.9   # comprimarea z-urilor long-run
    DUE_W = 0.10       # hedge overdue
    ZCLIP = 3.0

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, n + 1))

        # --- componenta long-run: z-score-uri ale numararilor totale ---
        cs = np.fromiter((counts.get(k, 0) for k in numbers), dtype=np.float64, count=n)
        mu = cs.mean()
        sd = cs.std()
        if sd > 1e-9:
            base = np.tanh(TANH_SHARP * (cs - mu) / sd)
        else:
            base = np.zeros(n)

        # --- componenta recenta: co-occurrence ponderat + difuziune pe graf ---
        rz = np.zeros(n)
        gate = 0.0
        draws = list(recent_deque)
        W = len(draws)
        if W >= 2:
            lens = [len(s) for s in draws]
            total = sum(lens)
            if total > 0:
                cols = np.fromiter(chain.from_iterable(draws), dtype=np.int64, count=total)
                rows = np.repeat(np.arange(W, dtype=np.int64), lens)
                Dfull = np.zeros((W, n + 1), dtype=np.float64)
                ok = (cols >= 1) & (cols <= n)
                Dfull[rows[ok], cols[ok]] = 1.0
                D = Dfull[:, 1:]  # matrice binara W x n

                w = np.power(0.5, (W - 1 - np.arange(W, dtype=np.float64)) / HALF_LIFE)
                Dw = D * w[:, None]
                Mw = Dw.T @ D              # co-occurrence ponderat (diagonala = recența)
                rec = Mw.diagonal().copy()
                tot = rec.sum()
                if tot > 1e-9:
                    q = rec / tot          # frecvente relative recente
                    wsum = float(w.sum())
                    wss = float(w @ w)
                    w_eff = (wsum * wsum / wss) if wss > 1e-12 else float(W)
                    se = 0.5 / (nd * sqrt(max(w_eff, 2.0)))  # esec corectat pt nd extrase/draw
                    zq = np.clip((q - 0.5) / se, -ZCLIP, ZCLIP)

                    # poarta: dispersie observata vs asteptarea uniforma (disp~1 sub H0)
                    disp = float(np.sqrt(np.mean((q - 0.5) ** 2))) / se
                    gate = min(max(disp - 1.0, 0.0) / 0.35, 1.0)

                    # un hop de difuziune pe graful normalizat simetric
                    off = Mw - np.diag(rec)
                    deg = off.sum(axis=1)
                    dinv = 1.0 / np.sqrt(deg + 1.0)
                    S = off * dinv[:, None] * dinv[None, :]
                    hop = S @ zq
                    rz = (1.0 - BETA) * zq + BETA * np.clip(hop, -ZCLIP, ZCLIP)

        # --- hedge overdue: gap centrat pe media geometrica (~2 la p=0.5) ---
        di = float(draw_index)
        gaps = np.fromiter((di - last_seen.get(k, -1) for k in numbers),
                           dtype=np.float64, count=n)
        due = np.clip((gaps - 2.0) / 5.0, -1.0, 1.0)

        wrec = 0.45 + 0.45 * gate
        score = BASE_W * base + wrec * rz + DUE_W * due

        sdict = {num: float(score[i]) for i, num in enumerate(numbers)}
        return sorted(numbers, key=lambda t: (-sdict[t], t))

    return rank