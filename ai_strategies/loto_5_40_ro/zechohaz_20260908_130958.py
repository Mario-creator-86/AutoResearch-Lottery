import math

NAME = "zechohaz"
DESCRIPTION = 'Model aditiv pe 7 componente: z-score istoric, momentum EWMA, eco decay pe ultimele K extrageri, mesaj de ordin 2 (penultima extragere), hazard de intarziere cu exponent configurabil si afinitate de co-aparitie ponderata cu ultima/penultima extragere. Combinarea foloseste o poarta de entropie (clamp neliniar r in [0.7, 1.4]) si un termen convex pe dinamica standardizata. Orientarea deque-ului (deque[-1] vs deque[0] = cel mai nou) este detectata explicit din last_seen la fiecare apel.'
SOURCE = 'glm-5.3-flash:cloud + 2026-09-08'


def create(config):
    N = config.number_range
    D = config.numbers_drawn

    def _get(attr, default):
        v = getattr(config, attr, default)
        return default if v is None else v

    w_z = float(_get('w_z', 0.12))
    w_mom = float(_get('w_mom', 0.60))
    w_echo = float(_get('w_echo', 0.30))
    w_msg2 = float(_get('w_msg2', 0.12))
    w_due = float(_get('w_due', 0.15))
    w_aff1 = float(_get('w_aff1', _get('w_aff', 0.35)))
    w_aff2 = float(_get('w_aff2', 0.20))
    w_sq = float(_get('w_sq', 0.04))
    h_mom = max(1.0, float(_get('h_mom', 25.0)))
    k_echo = max(1, int(_get('k_echo', 3)))
    gamma_echo = float(_get('gamma_echo', 0.55))
    gamma_aff = float(_get('gamma_aff', 0.90))
    haz_exp = float(_get('haz_exp', 1.0))
    haz_slope = float(_get('haz_slope', 1.0))
    r_lo = float(_get('r_lo', 0.7))
    r_hi = float(_get('r_hi', 1.4))
    cap_win = max(2, int(_get('cap_win', 150)))

    inv_gap = N / float(D) if D else 6.7
    numbers_all = list(range(1, N + 1))

    def _standardize(d):
        m = sum(d[n] for n in numbers_all) / N
        s = math.sqrt(sum((d[n] - m) ** 2 for n in numbers_all) / N)
        if s < 1e-12:
            return {n: 0.0 for n in numbers_all}
        return {n: (d[n] - m) / s for n in numbers_all}

    def _orient_key(s, last_seen, prev):
        eq = 0
        tot = 0
        for n in s:
            ls = last_seen.get(n, -1)
            if ls == prev:
                eq += 1
            tot += ls
        return (eq, tot / max(1, len(s)))

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = numbers_all
        try:
            seq = [set(s) for s in recent_deque]
        except TypeError:
            seq = []

        # verificare explicita a ipotezei: deque[-1] este cea mai noua extragere
        newest_is_last = True
        if len(seq) >= 2:
            prev = draw_index - 1
            if _orient_key(seq[0], last_seen, prev) > _orient_key(seq[-1], last_seen, prev):
                newest_is_last = False
        newest_first = list(reversed(seq)) if newest_is_last else list(seq)
        newest_first = newest_first[:cap_win]
        m_win = len(newest_first)

        cs = [counts.get(n, 0) for n in numbers]
        mean_c = sum(cs) / N
        std_c = math.sqrt(sum((c - mean_c) ** 2 for c in cs) / N)
        zc = {}
        for n, c in zip(numbers, cs):
            zv = (c - mean_c) / std_c if std_c > 1e-12 else 0.0
            zc[n] = -2.5 if zv < -2.5 else (2.5 if zv > 2.5 else zv)

        # hazard de intarziere: ((1 + s*gap) / E[gap]) ** exp - 1, clamp [-1, 2]
        due = {}
        for n in numbers:
            ls = last_seen.get(n, -1)
            gap = draw_index if ls < 0 else (draw_index - 1 - ls)
            if gap < 0:
                gap = 0
            val = ((1.0 + haz_slope * gap) / inv_gap) ** haz_exp - 1.0
            due[n] = -1.0 if val < -1.0 else (2.0 if val > 2.0 else val)

        L1 = newest_first[0] if m_win >= 1 else set()
        L2 = newest_first[1] if m_win >= 2 else set()
        msg2 = {n: 0.0 for n in numbers}
        for n in L2:
            msg2[n] = 1.0

        ew = {n: 0.0 for n in numbers}
        cnt_w = {n: 0 for n in numbers}
        aff1 = {n: 0.0 for n in numbers}
        aff2 = {n: 0.0 for n in numbers}
        wsum = 0.0
        for j in range(m_win):
            d = newest_first[j]
            w = 0.5 ** ((j + 1) / h_mom)
            wsum += w
            o1 = len(d & L1) if L1 else 0
            o2 = len(d & L2) if L2 else 0
            g1 = gamma_aff ** (j - 1) if j >= 1 else 0.0
            g2 = gamma_aff ** (j - 2) if j >= 2 else 0.0
            for n in d:
                ew[n] += w
                cnt_w[n] += 1
                if j >= 1 and o1:
                    aff1[n] += g1 * o1
                if j >= 2 and o2:
                    aff2[n] += g2 * o2

        if wsum > 1e-9:
            mom = {n: (ew[n] / wsum) * (N / float(D)) - 1.0 for n in numbers}
        else:
            mom = {n: 0.0 for n in numbers}
        mom = _standardize(mom)
        aff1 = _standardize(aff1)
        aff2 = _standardize(aff2)

        kk = min(k_echo, m_win)
        wnorm = sum(gamma_echo ** i for i in range(kk)) if kk > 0 else 0.0
        echo = {n: 0.0 for n in numbers}
        if wnorm > 1e-9:
            for i in range(kk):
                w = gamma_echo ** i
                for n in newest_first[i]:
                    echo[n] += w
            for n in numbers:
                echo[n] /= wnorm

        # poarta de entropie -> clamp neliniar r in [r_lo, r_hi]
        gate = r_lo
        tot_balls = D * m_win
        if tot_balls > 0:
            H = 0.0
            for n in numbers:
                p = cnt_w[n] / tot_balls
                if p > 0.0:
                    H -= p * math.log(p)
            denom = math.log(min(N, tot_balls)) if tot_balls > 1 else 1.0
            if denom > 0:
                hhat = H / denom
                hhat = 0.0 if hhat < 0.0 else (1.0 if hhat > 1.0 else hhat)
                gate = r_lo + (r_hi - r_lo) * hhat

        dyn = {}
        for n in numbers:
            dyn[n] = (w_mom * mom[n] + w_echo * echo[n] + w_msg2 * msg2[n]
                      + w_due * due[n] + w_aff1 * aff1[n] + w_aff2 * aff2[n])
        dm = sum(dyn.values()) / N
        ds = math.sqrt(sum((dyn[n] - dm) ** 2 for n in numbers) / N)
        if ds < 1e-12:
            ds = 1.0
        scores = {}
        for n in numbers:
            dz = (dyn[n] - dm) / ds
            scores[n] = w_z * zc[n] + gate * dyn[n] + w_sq * (dz * abs(dz))
        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank