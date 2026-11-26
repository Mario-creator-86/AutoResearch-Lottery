NAME = "zzgluese"
DESCRIPTION = 'Scorer multi-scala pentru 12/24: deviatie-z a frecventei globale, momentum pe apartenenta la fereastra recenta modulat de entropia distributiei de apartenenta, faza saturanta de intarziere peste gap-ul asteptat (2 extrageri la 12/24), interactie ne-liniara z x recenta si pair-glue din co-aparitiile perelor esantionate simetric de la ambele capete ale deque-ului (invariant la inversarea ordinei). Determinist, cu cache pe amprenta intrarii.'
SOURCE = 'model+2026-11-24'

import math

W_GLOB = 0.50
W_REC = 0.45
W_GAP = 0.50
W_PAIR = 0.30
W_X = 0.08
GAP_K = 2.5
EXP_GAP = 2.0
Z_CLAMP = 3.0
PAIR_K = 8
ENT_FLOOR = 0.35

_CACHE = {}
_CACHE_MAX = 300000


def create(config):
    # config: LotteryConfig — folosim doar atributele numerice; NU iteram config-ul.
    n_max = config.number_range
    n_draw = config.numbers_drawn
    nums = tuple(range(1, n_max + 1))
    p_draw = n_draw / float(n_max)
    ln_n = math.log(n_max) if n_max > 1 else 1.0

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # --- amprenta pentru cache (O(24)); deque-ul NU este parcurs aici ---
        cvals = []
        csum = 0
        wsum = 0
        for i in range(n_max):
            v = counts.get(i + 1, 0)
            cvals.append(v)
            csum += v
            wsum += (i + 1) * v
        ls_sum = 0
        for n in nums:
            ls_sum += last_seen.get(n, -1)
        try:
            rec_len = len(recent_deque)
            seq = recent_deque
        except TypeError:
            seq = list(recent_deque)
            rec_len = len(seq)

        key = (n_max, n_draw, draw_index, rec_len, csum, wsum, ls_sum,
               W_GLOB, W_REC, W_GAP, W_PAIR, W_X, GAP_K, EXP_GAP, Z_CLAMP,
               PAIR_K, ENT_FLOOR)
        got = _CACHE.get(key)
        if got is not None:
            return list(got)

        # --- deviatia globala fata de frecventa asteptata (binomial) ---
        total_draws = csum / float(n_draw)
        if total_draws < 1.0:
            total_draws = 1.0
        exp_c = total_draws * p_draw
        sd_c = math.sqrt(total_draws * p_draw * (1.0 - p_draw))
        if sd_c < 1e-9:
            sd_c = 1.0

        # --- apartenenta la fereastra recenta (independenta de ordine) ---
        rec = [0] * (n_max + 1)
        for s in seq:
            for x in s:
                if 1 <= x <= n_max:
                    rec[x] += 1
        wl = rec_len if rec_len > 0 else 1
        exp_r = wl * 0.5
        sd_r = math.sqrt(wl * 0.25)
        if sd_r < 1e-9:
            sd_r = 1.0

        # --- entropia distributiei de apartenenta -> modularea momentum-ului ---
        scale = 1.0
        if rec_len > 0:
            tot = float(n_draw) * rec_len
            H = 0.0
            for n in nums:
                q = rec[n] / tot
                if q > 0.0:
                    H -= q * math.log(q)
            frac = H / ln_n
            if frac > 1.0:
                frac = 1.0
            elif frac < 0.0:
                frac = 0.0
            scale = ENT_FLOOR + (1.0 - ENT_FLOOR) * frac

        # --- co-aparitii: primele PAIR_K si ultimele PAIR_K seturi (uniune
        #     simetrica, identica la inversarea deque-ului); fereastra mica -> toate ---
        sel = []
        if rec_len > 0:
            k = PAIR_K
            if 2 * k >= rec_len:
                sel = list(seq)
            else:
                tail_start = rec_len - k
                i2 = 0
                for s in seq:
                    if i2 < k or i2 >= tail_start:
                        sel.append(s)
                    i2 += 1
        co = {}
        for s in sel:
            memb = []
            for x in s:
                if 1 <= x <= n_max:
                    memb.append(x)
            Lm = len(memb)
            for ai in range(Lm):
                a = memb[ai]
                for bi in range(ai + 1, Lm):
                    b = memb[bi]
                    kk = (a, b) if a < b else (b, a)
                    co[kk] = co.get(kk, 0) + 1

        # --- pair-glue: co-aparitii cantarite cu forta recenta a partenerului ---
        glue = [0.0] * (n_max + 1)
        for kk, c in co.items():
            a, b = kk
            glue[a] += c * rec[b]
            glue[b] += c * rec[a]
        gmean = 0.0
        for n in nums:
            gmean += glue[n]
        gmean /= n_max
        gvar = 0.0
        for n in nums:
            d = glue[n] - gmean
            gvar += d * d
        gstd = math.sqrt(gvar / n_max)
        if gstd < 1e-12:
            gstd = 1.0

        # --- combinarea finala, tie-break determinist pe numar ---
        eff_rec = W_REC * scale
        eff_pair = W_PAIR * scale
        di = float(draw_index)
        out = []
        for idx in range(n_max):
            n = idx + 1
            z = (cvals[idx] - exp_c) / sd_c
            if z > Z_CLAMP:
                z = Z_CLAMP
            elif z < -Z_CLAMP:
                z = -Z_CLAMP
            r = (rec[n] - exp_r) / sd_r
            if r > Z_CLAMP:
                r = Z_CLAMP
            elif r < -Z_CLAMP:
                r = -Z_CLAMP
            ls = last_seen.get(n)
            if ls is None:
                gap = di + 1.0
            else:
                gap = di - float(ls)
            if gap < 0.0:
                gap = 0.0
            og = gap - EXP_GAP
            if og < 0.0:
                og = 0.0
            phase = og / (og + GAP_K)
            gz = (glue[n] - gmean) / gstd
            if gz > Z_CLAMP:
                gz = Z_CLAMP
            elif gz < -Z_CLAMP:
                gz = -Z_CLAMP
            sc = (W_GLOB * z + eff_rec * r + W_GAP * phase +
                  eff_pair * gz + W_X * (z * r))
            out.append((-sc, n))
        out.sort()
        perm = tuple(t[1] for t in out)
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.clear()
        _CACHE[key] = perm
        return list(perm)

    return rank