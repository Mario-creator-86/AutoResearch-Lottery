# -*- coding: utf-8 -*-
'''paircad4 - Kaskada 12/24: 4 componente standardizate cross-sectional, combinate liniar.
PAIR: tractiune de perechi co-aparute (seed din ultimele K_TAIL extrageri, propagare gamma pe fereastra, bitmask+popcount)
DEV: deviatia frecventei din fereastra fata de asteptarea binomiala K/P (semn configurabil prin W_DEV)
POS: consistenta positionala (kernel gaussian in jurul pozitiei teoretice E[pos|aprut])
CAD: cadenta personala (apropierea gapului curent de propriul gap mediu estimat din fereastra)
DIAG['orient_flips'] trebuie sa fie 0 inainte de rulare (deque asteptat oldest-first/newest-last).'''

import math

NAME = "zpaircad"
DESCRIPTION = ('Combină liniar 4 componente standardizate cross-sectional: tracțiunea perechilor '
               'co-apărute (seed din ultimele K_TAIL extrageri, propagare gamma-ponderată pe '
               'fereastra recentă prin bitmask+popcount), deviația frecvenței din fereastra față de '
               'așteptarea binomială K/P (semn configurabil), un prior de consistență pozițională '
               '(kernel gaussian în jurul poziției teoretice E[pos|apărut]) și cadența personală a '
               'fiecărui număr (apropierea gapului curent de propriul gap mediu). Determinist, '
               'stdlib-only, O(W*(K+N)) per apel, tie-break pe (score, count, n).')
SOURCE = 'model + 2026-09-08'

PARAMS = {
    'W_PAIR': 0.8,     # [0, 1.5]
    'W_DEV': 0.2,      # [-0.8, 1.0] (negativ = fade spre momentum/contrarian)
    'W_POS': 0.2,      # [0, 1]
    'W_CAD': 0.2,      # [0, 1]
    'GAMMA_P': 0.94,   # [0.90, 0.99]
    'K_TAIL': 2,       # {1, 2, 3, 5}
    'SIGMA_POS': 6.0,  # [4, 12]
}

DIAG = {'orient_flips': 0, 'calls': 0, 'last_orient': 'unknown', 'last_window': 0}

if hasattr(int, 'bit_count'):
    def _pc(x):
        return x.bit_count()
else:
    def _pc(x):
        return bin(x).count('1')


def create(config):
    P = int(config.number_range)
    K = int(config.numbers_drawn)
    G0 = float(P) / float(K) if K > 0 else 2.0
    Q = float(K) / float(P) if P > 0 else 0.5

    def _zs(xs):
        n = len(xs)
        if n == 0:
            return []
        m = sum(xs) / n
        var = sum((x - m) * (x - m) for x in xs) / n
        s = var ** 0.5
        if s < 1e-12:
            return [0.0] * n
        return [(x - m) / s for x in xs]

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        DIAG['calls'] += 1
        numbers = list(range(1, P + 1))

        # ---- orientarea deque-ului (tinta: orient_flips == 0, asteptat newest_last)
        lst = list(recent_deque)
        orient = 'newest_last'
        if lst:
            mx = -1
            for n in numbers:
                ls = last_seen.get(n, -1)
                if ls > mx:
                    mx = ls
            if mx >= 0:
                newest_set = set(n for n in numbers if last_seen.get(n, -1) == mx)
                if newest_set:
                    if newest_set.issubset(lst[-1]):
                        orient = 'newest_last'
                    elif newest_set.issubset(lst[0]):
                        orient = 'newest_first'
                        DIAG['orient_flips'] += 1
                    else:
                        orient = 'ambiguous'
        DIAG['last_orient'] = orient
        if orient != 'newest_first':
            lst.reverse()  # lst[0] = cea mai noua extragere
        if recent_window is not None and recent_window > 0 and len(lst) > recent_window:
            lst = lst[:recent_window]
        Wd = len(lst)
        DIAG['last_window'] = Wd

        gamma = float(PARAMS['GAMMA_P'])
        w_pow = []
        xw = 1.0
        for _ in range(Wd):
            w_pow.append(xw)
            xw *= gamma

        bit = {n: 1 << (n - 1) for n in numbers}

        # ---- o singura trecere: masca pe biti, numaratori, prima/ultima adancime, pozitii
        cntw = [0] * (P + 1)
        fd = [-1] * (P + 1)
        ld = [0] * (P + 1)
        sw = [0.0] * (P + 1)
        cw = [0.0] * (P + 1)
        masks = []
        for d in range(Wd):
            s = lst[d]
            w = w_pow[d]
            mk = 0
            for n in s:
                b = bit.get(n)
                if b is None:
                    continue
                mk |= b
                cntw[n] += 1
                if fd[n] < 0:
                    fd[n] = d
                ld[n] = d
            masks.append(mk)
            si = 0
            for n in sorted(s):
                if n in bit:
                    si += 1
                    sw[n] += w * si
                    cw[n] += w

        # ---- PAIR: parteneri co-aparuti cu j in ultimele K_TAIL extrageri
        kt = int(PARAMS['K_TAIL'])
        if kt < 0:
            kt = 0
        if kt > Wd:
            kt = Wd
        pmask = [0] * (P + 1)
        for d in range(kt):
            s = lst[d]
            for a in s:
                if a not in bit:
                    continue
                for b in s:
                    if b == a or b not in bit:
                        continue
                    pmask[a] |= bit[b]
        pair_raw = [0.0] * (P + 1)
        for n in numbers:
            pm = pmask[n]
            if pm:
                acc = 0.0
                for d in range(Wd):
                    w = w_pow[d]
                    if w < 1e-6:
                        continue
                    c = _pc(masks[d] & pm)
                    if c:
                        acc += w * c
                pair_raw[n] = acc

        # ---- DEV: z binomial pe fereastra
        exp_cnt = Wd * Q
        sd_w = (Wd * Q * (1.0 - Q)) ** 0.5 if Wd > 0 else 0.0
        dev_raw = [0.0] * (P + 1)
        if sd_w > 1e-12:
            for n in numbers:
                dev_raw[n] = (cntw[n] - exp_cnt) / sd_w

        # ---- POS: consistenta positionala (penalizare gaussiana a anomaliei de pozitie)
        sig = float(PARAMS['SIGMA_POS'])
        if sig < 1e-6:
            sig = 1e-6
        denom = float(P - 1) if P > 1 else 1.0
        pos_raw = [0.0] * (P + 1)
        if K > 0:
            two_sig2 = 2.0 * sig * sig
            for n in numbers:
                if cntw[n] >= 2 and cw[n] > 0.0:
                    mu = sw[n] / cw[n]
                    epos = 1.0 + (K - 1) * (n - 1) / denom
                    dpos = mu - epos
                    pos_raw[n] = math.exp(-(dpos * dpos) / two_sig2) - 1.0

        # ---- CAD: cadenta personala (gap propriu din fereastra, prior P/K)
        cad_raw = [0.0] * (P + 1)
        for n in numbers:
            c = cntw[n]
            g = (ld[n] - fd[n]) / float(c - 1) if c >= 2 else G0
            if g < 0.5:
                g = 0.5
            elif g > 12.0:
                g = 12.0
            ls = last_seen.get(n, None)
            if ls is None:
                continue
            delta = draw_index - ls
            if delta < 1:
                continue
            v = 1.0 - abs(delta - g) / g
            if v > 0.0:
                cad_raw[n] = v

        wp = float(PARAMS['W_PAIR'])
        wdv = float(PARAMS['W_DEV'])
        wpo = float(PARAMS['W_POS'])
        wcd = float(PARAMS['W_CAD'])

        zp = _zs([pair_raw[n] for n in numbers])
        zd = _zs([dev_raw[n] for n in numbers])
        zq = _zs([pos_raw[n] for n in numbers])
        zc = _zs([cad_raw[n] for n in numbers])

        scores = {}
        for i in range(len(numbers)):
            n = numbers[i]
            scores[n] = wp * zp[i] + wdv * zd[i] + wpo * zq[i] + wcd * zc[i]

        return sorted(numbers, key=lambda nn: (-scores[nn], -counts.get(nn, 0), nn))

    return rank