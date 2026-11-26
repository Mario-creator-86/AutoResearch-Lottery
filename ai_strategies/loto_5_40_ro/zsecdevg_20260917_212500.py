import math

NAME = "zsecdevg"
DESCRIPTION = 'Sector-cycle pe intarzierea mediata a sectoarelor + deviatie EWMA fata de frecventa asteptata, amestecate printr-un entropy-gate; gap-uri individuale normalizate si limitate de GAP_CAP; SEC_SIGN/DEV_SIGN orienteaza reversion vs momentum.'
SOURCE = 'quant-research secdev-gapx v1 (2026-09-17)'

DEFAULTS = {
    'SEC_SIZE': 6,
    'SEC_SIGN': 1,
    'DEV_SIGN': 1,
    'W_SEC': 0.7,
    'W_GAP': 0.5,
    'W_DEV': 0.4,
    'GAP_CAP': 2.5,
    'GATE_MIX': 0.5,
    'ORIENT_AUTO': 0,
    'W_BASE': 0.02,
    'EWMA_HALFLIFE': 10,
}


def create(config):
    N = int(config.number_range)
    WD = int(config.numbers_drawn)
    _TS = int(config.ticket_size)

    cfg = {}
    for k, v in DEFAULTS.items():
        val = getattr(config, k, None)
        if val is None:
            val = getattr(config, k.lower(), v)
        cfg[k] = val

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        L = len(recent_deque)
        p = WD / float(N) if N > 0 else 0.0
        expected_gap = (1.0 / p) if p > 0 else float(N)
        cap = max(1.0, float(cfg['GAP_CAP'])) * expected_gap

        # 1) EWMA pe numere in fereastra recenta (cel mai nou = ultimul element din deque)
        halflife = max(1.0, float(cfg['EWMA_HALFLIFE']))
        decay = 0.5 ** (1.0 / halflife)
        ewma_new = dict.fromkeys(numbers, 0.0)
        ewma_old = dict.fromkeys(numbers, 0.0)
        w_new = 0.0
        w_old = 0.0
        Lh = L // 2
        age = 0
        for draw_set in reversed(recent_deque):
            w = decay ** age
            tgt = ewma_new if age < Lh else ewma_old
            for n in draw_set:
                if 1 <= n <= N:
                    tgt[n] += w
            if age < Lh:
                w_new += w
            else:
                w_old += w
            age += 1
        wsum = w_new + w_old
        dev_z = dict.fromkeys(numbers, 0.0)
        if wsum > 1e-12:
            sd = math.sqrt(wsum * p * (1.0 - p))
            if sd < 1e-12:
                sd = 1.0
            for n in numbers:
                dev_z[n] = ((ewma_new[n] + ewma_old[n]) - wsum * p) / sd

        # pers_z: persistenta momentum = corelatie intre jumatatea recenta si jumatatea veche a ferestrei
        pers_z = 0.0
        if Lh >= 4 and w_new > 1e-12 and w_old > 1e-12:
            sa = math.sqrt(sum(v * v for v in ewma_new.values()))
            sb = math.sqrt(sum(v * v for v in ewma_old.values()))
            if sa > 1e-12 and sb > 1e-12:
                cov = sum(ewma_new[n] * ewma_old[n] for n in numbers)
                r = cov / (sa * sb)
                if r > 0.999:
                    r = 0.999
                elif r < -0.999:
                    r = -0.999
                pers_z = math.atanh(r) * math.sqrt(Lh - 3)

        # 2) orientare semne: flip automat optional cand regimul momentum/reversion e clar
        sec_sign = 1 if int(cfg['SEC_SIGN']) >= 0 else -1
        dev_sign = 1 if int(cfg['DEV_SIGN']) >= 0 else -1
        flips = 0
        if int(cfg['ORIENT_AUTO']) and L >= 20:
            want = -1 if pers_z > 1.0 else (1 if pers_z < -1.0 else 0)
            if want != 0:
                if sec_sign != want:
                    sec_sign = want
                    flips += 1
                if dev_sign != want:
                    dev_sign = want
                    flips += 1

        # 3) sector-cycle: intarziere mediata pe sector, normalizata la gap asteptat si centrata
        K = max(2, int(cfg['SEC_SIZE']))
        sector_of = {n: ((n - 1) * K) // N for n in numbers}
        members = {s: [] for s in range(K)}
        for n in numbers:
            members[sector_of[n]].append(n)
        sec_x = {}
        for s in range(K):
            mem = members[s]
            if not mem:
                sec_x[s] = 0.0
                continue
            g = 0.0
            for n in mem:
                ls = last_seen.get(n)
                gn = draw_index - ls if ls is not None else draw_index + 1
                if gn < 0:
                    gn = 0
                g += gn
            g /= len(mem)
            sec_x[s] = (g if g < cap else cap) / expected_gap
        mean_sx = sum(sec_x.values()) / K
        sec_centered = {s: sec_x[s] - mean_sx for s in range(K)}

        # 4) gap individual: aceeasi normalizare + cap + centrare (evita explozia intarziatilor)
        gap_x = {}
        for n in numbers:
            ls = last_seen.get(n)
            gn = draw_index - ls if ls is not None else draw_index + 1
            if gn < 0:
                gn = 0
            gap_x[n] = (gn if gn < cap else cap) / expected_gap
        mean_gx = sum(gap_x.values()) / N
        gap_centered = {n: gap_x[n] - mean_gx for n in numbers}

        # 5) entropia distributiei sectoriale pe fereastra recenta -> entropy-gate adaptiv
        sec_counts = [0] * K
        tot = 0
        for draw_set in recent_deque:
            for n in draw_set:
                if 1 <= n <= N:
                    sec_counts[sector_of[n]] += 1
                    tot += 1
        entropy_norm = 0.0
        if tot > 0 and K > 1:
            H = 0.0
            for c in sec_counts:
                if c > 0:
                    pr = c / float(tot)
                    H -= pr * math.log(pr)
            entropy_norm = H / math.log(K)
            if entropy_norm > 1.0:
                entropy_norm = 1.0
        gm = float(cfg['GATE_MIX'])
        gate = gm + (1.0 - gm) * (1.0 - entropy_norm)
        w_sec = float(cfg['W_SEC']) * gate
        w_dev = float(cfg['W_DEV']) * (1.0 + (1.0 - gate))

        # 6) prior mic de frecventa pe tot istoricul (stabilizare determinista a ordonarii)
        cvals = [counts.get(n, 0) for n in numbers]
        cm = sum(cvals) / float(N)
        cs = math.sqrt(sum((c - cm) ** 2 for c in cvals) / float(N))
        if cs < 1e-12:
            cs = 1.0
        w_base = float(cfg['W_BASE'])
        w_gap = float(cfg['W_GAP'])

        scores = {}
        sec_terms = []
        gap_terms = []
        dev_terms = []
        for n in numbers:
            bz = (counts.get(n, 0) - cm) / cs
            t_sec = w_sec * sec_sign * sec_centered[sector_of[n]]
            t_gap = w_gap * sec_sign * gap_centered[n]
            t_dev = w_dev * dev_sign * (-dev_z[n])
            scores[n] = t_sec + t_gap + t_dev + w_base * bz
            sec_terms.append(t_sec)
            gap_terms.append(t_gap)
            dev_terms.append(t_dev)

        order = sorted(numbers, key=lambda n: (-scores[n], n))

        rank._diag = {
            'sector_span': max(sec_terms) - min(sec_terms),
            'gap_span': max(gap_terms) - min(gap_terms),
            'dev_span': max(dev_terms) - min(dev_terms),
            'entropy_norm': entropy_norm,
            'gate_factor': gate,
            'orient_flips': flips,
            'pers_z': pers_z,
            'L': L,
            'sec_sign_eff': sec_sign,
            'dev_sign_eff': dev_sign,
        }
        return order

    return rank