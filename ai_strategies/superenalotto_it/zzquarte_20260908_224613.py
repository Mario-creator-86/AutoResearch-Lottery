NAME = "zzquarte"
DESCRIPTION = 'Patru semnale standardizate z combinate liniar: (1) lift de perechi - afinitatea de co-aparitie in fereastra recenta cu numerele din ultimele K_TAIL extrageri; (2) z-deviatia frecventei pe fereastra recenta cu semn comutabil (momentum vs reverzie); (3) drift pozitional - centroizi EWMA ai pozitiilor sortate fata de asteptarea combinatorica, cu attractie gaussiana SIGMA_POS; (4) cadenta ritmica - faza gaussiana cu varf la gap-ul mediu de revenire al fiecarui numar. Ponderile W_PAIR/W_DEV/W_POS/W_CAD pot fi puse pe zero individual pentru ablatie.'
SOURCE = 'model cantitativ + 2026-09-08'

import math

DIAG = {'orient': None, 'orient_flips': 0}

# hiperparametri expusi pentru grid-search:
# W_PAIR [0,1.5]; W_DEV [-0.8,1.0] (negativ = flip anti-momentum/cold);
# W_POS [0,1]; W_CAD [0,1]; GAMMA_P [0.90,0.99]; K_TAIL in {1,2,3,5}; SIGMA_POS [4,12]
W_PAIR = 0.85
W_DEV = 0.35
W_POS = 0.55
W_CAD = 0.40
GAMMA_P = 0.94
K_TAIL = 2
SIGMA_POS = 7.0

_LIFT_CAP = 4.0      # plafon pe lift-ul individual de perechi
_CAD_G_MIN = 1.5     # cadenta minima credibila (extrageri intre aparitii)
_CAD_G_MAX = 40.0    # cadenta maxima credibila


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    p = K / float(N)
    # asteptarea combinatorica pentru valoarea de pe pozitia sortata j (1-indexat)
    E = [(j + 1) * (N + 1) / float(K + 1) for j in range(K)]

    # stare incrementala: matrice de perechi + oglinda ferestrei (idx, tuple sortat)
    st = {'idx': None, 'newest': None,
          'pair': [0] * (N * N), 'mirror': [], 'appear': {}}

    def add_pairs(tup, sign):
        pair = st['pair']
        m = len(tup)
        for i in range(m):
            row = (tup[i] - 1) * N
            for j in range(i + 1, m):
                pair[row + tup[j] - 1] += sign

    def rebuild(seq, draw_index):
        st['pair'] = [0] * (N * N)
        mirror = []
        appear = {}
        base = draw_index - 1
        for i in range(len(seq) - 1, -1, -1):
            tup = tuple(sorted(seq[i]))
            idx = base - i
            add_pairs(tup, 1)
            mirror.append((idx, tup))
            for x in tup:
                if x in appear:
                    appear[x].append(idx)
                else:
                    appear[x] = [idx]
        st['mirror'] = mirror
        st['appear'] = appear
        st['newest'] = mirror[-1][1] if mirror else None
        st['idx'] = draw_index

    def incorporate(seq, draw_index):
        # seq[0] = cea mai noua extragere, neincorporata inca
        tup = tuple(sorted(seq[0]))
        add_pairs(tup, 1)
        idx = draw_index - 1
        st['mirror'].append((idx, tup))
        appear = st['appear']
        for x in tup:
            if x in appear:
                appear[x].append(idx)
            else:
                appear[x] = [idx]
        while len(st['mirror']) > len(seq):
            _oi, old = st['mirror'].pop(0)
            add_pairs(old, -1)
        st['newest'] = tup
        st['idx'] = draw_index

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # --- auto-detectarea orientarii deque-ului (newest_last) ---
        def mean_recency(stset):
            if not stset:
                return float('inf')
            tot = 0.0
            for n in stset:
                ls = last_seen.get(n)
                tot += (draw_index - ls) if ls is not None else float(draw_index)
            return tot / float(len(stset))

        Ld = len(recent_deque)
        newest_last = True
        if Ld >= 2:
            newest_last = mean_recency(recent_deque[-1]) <= mean_recency(recent_deque[0])
        orient = 'newest_last' if newest_last else 'newest_first'
        if DIAG['orient'] is not None and DIAG['orient'] != orient:
            DIAG['orient_flips'] += 1
        DIAG['orient'] = orient

        seq = list(recent_deque)
        if newest_last and Ld >= 2:
            seq.reverse()  # seq[0] = cea mai noua extragere

        # --- sincronizare incrementala a statisticilor de fereastra ---
        if not seq:
            if st['mirror']:
                rebuild(seq, draw_index)
        elif st['idx'] is None or draw_index < st['idx'] or draw_index > st['idx'] + 1:
            rebuild(seq, draw_index)
        elif draw_index == st['idx'] + 1:
            tup0 = tuple(sorted(seq[0]))
            if tup0 != st['newest']:
                incorporate(seq, draw_index)

        mirror = st['mirror']
        Leff = len(mirror)
        numbers = list(range(1, N + 1))

        # numarari in fereastra recenta
        cnt = [0] * (N + 1)
        for _oi, tup in mirror:
            for x in tup:
                cnt[x] += 1

        # --- componenta 1: PAIR (lift de perechi vs ultimele K_TAIL extrageri) ---
        pair_raw = [0.0] * (N + 1)
        pair_arr = st['pair']
        anchor_list = []
        for _oi, tup in mirror[-K_TAIL:]:
            anchor_list.extend(tup)
        if anchor_list:
            da = float(len(anchor_list))
            for n in numbers:
                cn = cnt[n]
                s = 0.0
                for a in anchor_list:
                    if a == n:
                        continue
                    if a < n:
                        v = pair_arr[(a - 1) * N + (n - 1)]
                    else:
                        v = pair_arr[(n - 1) * N + (a - 1)]
                    if v:
                        lift = v * Leff / (cn * cnt[a] + 1e-9)
                        if lift > _LIFT_CAP:
                            lift = _LIFT_CAP
                        s += lift
                pair_raw[n] = s / da

        # --- componenta 2: DEV (z-deviatia frecventei pe fereastra) ---
        Lp = Leff * p
        sdw = math.sqrt(Lp * (1.0 - p) + 1e-12)
        dev_raw = [0.0] * (N + 1)
        for n in numbers:
            dev_raw[n] = (cnt[n] - Lp) / sdw

        # --- componenta 3: POS (drift pozitional, centroizi EWMA + kernel gauss) ---
        pos_raw = [0.0] * (N + 1)
        if Leff:
            msum = [0.0] * K
            wsum = 0.0
            w = 1.0
            for i in range(Leff):
                tup = mirror[Leff - 1 - i][1]
                for j in range(K):
                    msum[j] += w * tup[j]
                wsum += w
                w *= GAMMA_P
            if wsum > 0.0:
                inv = 0.5 / (SIGMA_POS * SIGMA_POS)
                for j in range(K):
                    dj = msum[j] / wsum - E[j]
                    if -1e-9 < dj < 1e-9:
                        continue
                    ej = E[j]
                    for n in numbers:
                        d = n - ej
                        pos_raw[n] += dj * math.exp(-d * d * inv)
                kn = 1.0 / K
                for n in numbers:
                    pos_raw[n] *= kn

        # --- componenta 4: CAD (faza de cadenta ritmica) ---
        cad_raw = [0.0] * (N + 1)
        if Leff:
            oldest_idx = mirror[0][0]
            appear = st['appear']
            di = draw_index
            for n in numbers:
                lst = appear.get(n)
                if not lst:
                    continue
                while lst and lst[0] < oldest_idx:
                    lst.pop(0)
                if len(lst) < 2:
                    continue
                gsum = 0.0
                for i in range(1, len(lst)):
                    gsum += lst[i] - lst[i - 1]
                g = gsum / (len(lst) - 1)
                if g < _CAD_G_MIN or g > _CAD_G_MAX:
                    continue
                d = (di - lst[-1]) - g
                sig = 0.6 * g + 1.0
                r = d / sig
                cad_raw[n] = math.exp(-0.5 * r * r)

        # --- standardizare z per componenta + combinare liniara ---
        score = [0.0] * (N + 1)
        for vals, wtx in ((pair_raw, W_PAIR), (dev_raw, W_DEV),
                          (pos_raw, W_POS), (cad_raw, W_CAD)):
            if wtx == 0.0:
                continue
            comp = [vals[n] for n in numbers]
            m = 0.0
            for x in comp:
                m += x
            m /= len(comp)
            var = 0.0
            for x in comp:
                dv = x - m
                var += dv * dv
            sdv = math.sqrt(var / len(comp) + 1e-12)
            if sdv <= 1e-12:
                continue
            iw = wtx / sdv
            for n in numbers:
                score[n] += iw * (vals[n] - m)

        return sorted(numbers, key=lambda n: (-score[n], n))

    return rank