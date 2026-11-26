import math

NAME = "zpairdev"
DESCRIPTION = (
    'Scorer liniar pe 4 componente standardizate: PAIR - notorietate conditionata de '
    'parteneri, din co-aparitii ponderate exponential (GAMMA_P) in fereastra recenta; '
    'DEV - z-score binomial pe termen lung fata de frecventa asteptata N*K/R, cu semn '
    'liber (flip cald/rece); POS - afinitate gaussiana fata de mediile pe slot-uri ale '
    'extragerilor recente (SIGMA_POS); CAD - rezonanta de ciclu personala: nuclee '
    'gaussiene la multiplii gap-ului mediu al fiecarui numar (K_TAIL armonici). '
    'Combinare liniara + sortare descrescatora, tie-break determinist.'
)
SOURCE = 'model + 2026-09-08'

DIAG = {'orient_flips': 0, 'orient_check': 'none', 'calls': 0, 'fallbacks': 0, 'pair_edges': 0}

PARAM_GRID = {
    'w_pair': [0.0, 0.3, 0.6, 0.9, 1.2, 1.5],
    'w_dev': [-0.8, -0.4, 0.0, 0.4, 0.8, 1.0],
    'w_pos': [0.0, 0.25, 0.5, 0.75, 1.0],
    'w_cad': [0.0, 0.25, 0.5, 0.75, 1.0],
    'gamma_p': [0.90, 0.92, 0.94, 0.96, 0.98, 0.99],
    'k_tail': [1, 2, 3, 5],
    'sigma_pos': [4.0, 6.0, 8.0, 10.0, 12.0],
}

_PAIR_TAIL = 120  # coada maxima folosita pentru tabelul de perechi (cost limitat)


def create(config):
    R = int(config.number_range)
    K = int(config.numbers_drawn)

    defaults = {
        'w_pair': 0.6,
        'w_dev': 0.2,
        'w_pos': 0.3,
        'w_cad': 0.3,
        'gamma_p': 0.96,
        'k_tail': 2,
        'sigma_pos': 7.0,
    }
    for key, val in defaults.items():
        if not hasattr(config, key):
            setattr(config, key, val)

    W_PAIR = float(getattr(config, 'w_pair'))
    W_DEV = float(getattr(config, 'w_dev'))
    W_POS = float(getattr(config, 'w_pos'))
    W_CAD = float(getattr(config, 'w_cad'))
    GAMMA_P = float(getattr(config, 'gamma_p'))
    K_TAIL = max(1, int(getattr(config, 'k_tail')))
    SIGMA_POS = float(getattr(config, 'sigma_pos'))

    DIAG['orient_flips'] = 0
    DIAG['orient_check'] = 'none'
    DIAG['calls'] = 0
    DIAG['fallbacks'] = 0
    DIAG['pair_edges'] = 0

    gpow = [1.0]  # gpow[a] = GAMMA_P ** a, extins lazy

    def _z(src):
        # standardizare cross-sectional pe 1..R
        m = 0.0
        for n in range(1, R + 1):
            m += src[n]
        m /= R
        v = 0.0
        for n in range(1, R + 1):
            d = src[n] - m
            v += d * d
        s = math.sqrt(v / R)
        out = [0.0] * (R + 1)
        if s > 1e-12:
            inv = 1.0 / s
            for n in range(1, R + 1):
                out[n] = (src[n] - m) * inv
        return out

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        DIAG['calls'] += 1
        numbers = list(range(1, R + 1))
        N = int(draw_index)
        dlist = list(recent_deque)
        L = len(dlist)

        if N <= 0 or L < 2:
            DIAG['fallbacks'] += 1
            return sorted(numbers, key=lambda n: (-counts.get(n, 0), n))

        # --- orientare deque: presupunem newest_last; detectie pe copie locala ---
        ml = 0.0
        for x in dlist[-1]:
            ml += last_seen.get(x, -1)
        ml /= float(len(dlist[-1]))
        mf = 0.0
        for x in dlist[0]:
            mf += last_seen.get(x, -1)
        mf /= float(len(dlist[0]))
        if mf > ml + 0.5:
            dlist.reverse()  # flip doar pe copie, niciodata pe deque-ul original
            DIAG['orient_flips'] += 1
            DIAG['orient_check'] = 'flipped_to_newest_last'
        elif ml > mf + 0.5:
            DIAG['orient_check'] = 'confirmed_newest_last'
        else:
            DIAG['orient_check'] = 'ambiguous_kept_newest_last'

        while len(gpow) < L:
            gpow.append(gpow[-1] * GAMMA_P)

        slists = [sorted(s) for s in dlist]

        # --- aparitii decay-uite in fereastra (h) ---
        h = [0.0] * (R + 1)
        for i in range(L):
            w = gpow[L - 1 - i]
            for x in slists[i]:
                h[x] += w

        # --- PAIR: co-aparitii decay-uite (chei intregi a*64+b) pe coada ferestrei ---
        start = L - _PAIR_TAIL if L > _PAIR_TAIL else 0
        pw = {}
        pw_get = pw.get
        for i in range(start, L):
            w = gpow[L - 1 - i]
            t = slists[i]
            lt = len(t)
            for ai in range(lt - 1):
                a = t[ai]
                base = a << 6
                for bi in range(ai + 1, lt):
                    k2 = base | t[bi]
                    pw[k2] = pw_get(k2, 0.0) + w
        DIAG['pair_edges'] = len(pw)

        p_raw = [0.0] * (R + 1)
        for k2, w in pw.items():
            a = k2 >> 6
            b = k2 & 63
            p_raw[a] += h[b] * w
            p_raw[b] += h[a] * w
        p_cond = [0.0] * (R + 1)
        for n in numbers:
            hn = h[n]
            if hn > 1e-12:
                p_cond[n] = p_raw[n] / (hn + 0.25)  # hotness medie a partenerilor

        # --- DEV: deviatie z pe termen lung ---
        pexp = float(K) / float(R)
        exp_ct = N * pexp
        sd_bin = math.sqrt(N * pexp * (1.0 - pexp))
        if sd_bin <= 1e-12:
            sd_bin = 1.0
        dev = [0.0] * (R + 1)
        for n in numbers:
            dev[n] = (counts.get(n, 0) - exp_ct) / sd_bin

        # --- POS: medii pe slot-uri ponderate decay + afinitate gaussiana ---
        slot_w = [0.0] * K
        slot_sum = [0.0] * K
        for i in range(L):
            w = gpow[L - 1 - i]
            t = slists[i]
            for si in range(K):
                slot_sum[si] += w * t[si]
                slot_w[si] += w
        inv2s2 = 0.5 / (SIGMA_POS * SIGMA_POS)
        pos = [0.0] * (R + 1)
        for n in numbers:
            acc = 0.0
            for si in range(K):
                sw = slot_w[si]
                if sw > 1e-12:
                    d = n - slot_sum[si] / sw
                    acc += math.exp(-d * d * inv2s2)
            pos[n] = acc

        # --- CAD: rezonanta la multiplii gap-ului mediu personal ---
        cad = [0.0] * (R + 1)
        for n in numbers:
            c = counts.get(n, 0)
            gap = N - 1 - last_seen.get(n, -1)
            if gap < 0:
                gap = 0
            gm = (N + 1.0) / (c + 1.0)
            if gm < 1.5:
                gm = 1.5
            elif gm > 60.0:
                gm = 60.0
            acc = 0.0
            target = gm
            for _j in range(K_TAIL):
                width = 0.35 * target + 1.0
                dd = gap - target
                acc += math.exp(-0.5 * (dd / width) * (dd / width))
                target += gm
            cad[n] = acc

        # --- combinare standardizata ---
        pz = _z(p_cond)
        dz = _z(dev)
        qz = _z(pos)
        cz = _z(cad)

        score = [0.0] * (R + 1)
        for n in numbers:
            score[n] = W_PAIR * pz[n] + W_DEV * dz[n] + W_POS * qz[n] + W_CAD * cz[n]

        cg = counts.get
        return sorted(numbers, key=lambda n: (-score[n], -cg(n, 0), n))

    return rank