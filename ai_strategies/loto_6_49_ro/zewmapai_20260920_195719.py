import math
import numpy as np

NAME = "zewmapai"
DESCRIPTION = (
    'Strategie hibrida non-liniara: frecventa EWMA cu decay adaptat din entropia ferestrei, '
    'lift de co-aparitie pe perechi (ponderat exponential fata de asteptarea sub independenta), '
    'presiune de intarziere tip hazard geometric (tanh bornat), deficit pe 7 sectoare si '
    'abatere de la frecventa lifetime; regim de dispersie chi2 moduleaza momentum/intarziere.'
)
SOURCE = 'glm-5.3-flash:cloud | 2026-09-20'


def create(config):
    P = int(config.number_range)
    WD = int(config.numbers_drawn)

    p_num = WD / P
    p_pair = (WD * (WD - 1)) / (P * (P - 1))
    mu_gap = (1.0 - p_num) / p_num
    sd_gap = math.sqrt(1.0 - p_num) / p_num
    K = P // 7 if P % 7 == 0 else 0
    var_sec_draw = 0.0
    if K:
        var_sec_draw = WD * (K / P) * ((P - K) / P) * ((P - WD) / (P - 1))

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        N = P
        nums = np.arange(1, N + 1)
        L = len(recent_deque)
        draws = list(recent_deque)

        # orientare: ultimul element = cea mai recenta extragere;
        # verificare determinista prin last_seen vs draw_index-1
        if L >= 2:
            t_new = draw_index - 1

            def match_frac(s):
                if not s:
                    return 0.0
                m = 0
                for n in s:
                    ls = last_seen.get(n)
                    if ls is not None and ls >= t_new:
                        m += 1
                return m / len(s)

            f_end = match_frac(draws[-1])
            f_start = match_frac(draws[0])
            if f_start > f_end + 0.34 and f_start >= 0.5:
                draws.reverse()

        # entropia distributiei din fereastra -> decay adaptiv
        cw = np.zeros(N)
        for s in draws:
            for n in s:
                cw[n - 1] += 1.0
        tot_w = float(cw.sum())
        if tot_w > 0.0:
            q = cw / tot_w
            qp = q[q > 0]
            Hn = float(-(qp * np.log(qp)).sum() / math.log(N))
        else:
            Hn = 1.0
        x = min(1.0, max(0.0, (Hn - 0.88) / 0.12))
        decay = 0.90 + 0.07 * x

        # regim de dispersie chi2 pe fereastra
        if L > 0 and tot_w > 0.0:
            e_num = tot_w / N
            x2 = float(((cw - e_num) ** 2 / e_num).sum())
            t_disp = (x2 - (N - 1)) / math.sqrt(2.0 * (N - 1))
        else:
            t_disp = 0.0
        td = math.tanh(t_disp / 3.0)

        # hazard de intarziere: gap-uri normalizate la distributia geometrica sub null
        gaps = np.empty(N)
        for j in range(N):
            ls = last_seen.get(j + 1)
            if ls is not None and ls >= 0:
                gaps[j] = float(max(draw_index - 1 - ls, 0))
            else:
                gaps[j] = float(max(draw_index, 1))
        gz = np.tanh((gaps - mu_gap) / sd_gap / 2.0)

        # abatere de la frecventa lifetime
        tot_counts = 0
        for v in counts.values():
            tot_counts += v
        if tot_counts > 0:
            D = tot_counts / float(WD)
            mu_c = tot_counts / float(N)
            sd_c = math.sqrt(max(D * p_num * (1.0 - p_num), 1e-9))
            lc = np.fromiter((counts.get(j + 1, 0) for j in range(N)), dtype=float, count=N)
            lz = np.tanh((lc - mu_c) / sd_c / 3.0)
        else:
            lz = np.zeros(N)

        if L == 0:
            score = 0.7 * gz + 0.2 * lz
            order = np.argsort(-score, kind='stable')
            return [int(nums[i]) for i in order]

        ages = np.arange(L - 1, -1, -1, dtype=float)
        w = decay ** ages
        sw = float(w.sum())
        sw2 = float((w * w).sum())

        Ix = np.zeros((L, N))
        for i, s in enumerate(draws):
            if s:
                Ix[i, np.fromiter((n - 1 for n in s), dtype=int, count=len(s))] = 1.0

        # momentum EWMA standardizat (abatere de la frecventa asteptata)
        ewma = w @ Ix
        var_hot = p_num * (1.0 - p_num) * sw2
        z_hot = (ewma - p_num * sw) / math.sqrt(max(var_hot, 1e-12))
        hz = np.tanh(z_hot / 2.0)

        # lift de co-aparitie pe perechi, ponderat exponential
        Cm = (Ix * w[:, None]).T @ Ix
        Eexp = p_pair * sw
        R = np.maximum(Cm - Eexp, 0.0)
        np.fill_diagonal(R, 0.0)
        aff = R.sum(axis=1) / max(sw, 1e-9)
        pz = np.tanh(aff / 0.30)

        # deficit de acoperire pe sectoare (7 grupuri de cate 7)
        if K:
            Sw = ewma.reshape(7, K).sum(axis=1)
            mu_s = (K / float(P)) * sw
            sd_s = math.sqrt(max(var_sec_draw * sw2, 1e-12))
            zs = (Sw - mu_s) / sd_s
            sz = np.repeat(np.tanh(-zs / 2.0), K)
        else:
            sz = np.zeros(N)

        # ponderi modulate de regim
        A = 1.00 * (1.0 + 0.20 * td)
        B = 0.60 * (1.0 + 0.15 * (1.0 - Hn))
        Cg = 0.70 * (1.0 - 0.20 * td)

        score = A * hz + B * pz + Cg * gz + 0.25 * sz + 0.15 * lz + 0.30 * hz * pz

        order = np.argsort(-score, kind='stable')
        return [int(nums[i]) for i in order]

    return rank