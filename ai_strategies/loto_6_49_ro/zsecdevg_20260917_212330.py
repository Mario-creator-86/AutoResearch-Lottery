import math

NAME = "zsecdevg"
DESCRIPTION = ('Strategie contrarian pe trei mecanisme ne-liniare: (1) cicluri de intarziere pe sectoare - z-score al aparitiilor recente per sector, favorizeaza sectoarele sub-visitate, modulat de entropia ferestrei; (2) gap/overdue normalizat si capat la GAP_CAP; (3) deviatie macro fata de frecventa asteptata pe tot istoricul. Termenii se combina liniar (W_SEC/W_GAP/W_DEV) si returneaza permutarea completa 1..49.')
SOURCE = 'model + 2026-09-17 secdev_gate v1 (contrarian, ORIENT_AUTO=0)'

# --- tunables pentru grid: SEC_SIZE{4,6,8} x SEC_SIGN{+-1} x DEV_SIGN{+-1} x W_SEC{0.35,0.7,1.05} x W_GAP{0.3,0.5,0.8} x W_DEV{0,0.4,0.8} x GAP_CAP{1.5,2.5,4.0} x GATE_MIX{0.3,0.5,0.7} ---
SEC_SIZE = 6
SEC_SIGN = -1.0   # -1 = reversion (favor sectoare sub-visitate), +1 = momentum
DEV_SIGN = -1.0   # -1 = favor numere sub-frecvente (macro reversion), +1 = momentum
W_SEC = 0.7
W_GAP = 0.5
W_DEV = 0.4
GAP_CAP = 2.5
GATE_MIX = 0.5
ORIENT_AUTO = 0


def create(config):
    # config: LotteryConfig - folosit doar prin atributele number_range / numbers_drawn / ticket_size
    N = config.number_range
    K = config.numbers_drawn
    TS = config.ticket_size
    EXP_GAP = N / float(K)

    cache = {'k': None, 'out': None}

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        W = len(recent_deque)
        S = SEC_SIZE
        tot = 0
        for n in range(1, N + 1):
            tot += counts.get(n, 0)
        key = (draw_index, W, recent_window, tot, S, SEC_SIGN, DEV_SIGN, W_SEC, W_GAP, W_DEV, GAP_CAP, GATE_MIX)
        if cache['k'] == key and cache['out'] is not None:
            return list(cache['out'])

        # sectoare cat mai egale pe 1..N
        sec_of = [0] * (N + 1)
        sizes = [0] * S
        for n in range(1, N + 1):
            s = (n - 1) * S // N
            sec_of[n] = s
            sizes[s] += 1

        # activitate recenta per sector (flat, order-free) + jumatati pentru diag de persistenta
        h = W // 2
        act = [0.0] * S
        a1 = [0.0] * S
        a2 = [0.0] * S
        total_hits = 0.0
        for i, draw in enumerate(recent_deque):
            tgt = a1 if i < h else a2
            for m in draw:
                if 1 <= m <= N:
                    s = (m - 1) * S // N
                    act[s] += 1.0
                    tgt[s] += 1.0
                    total_hits += 1.0

        # z-score per sector fata de asteptare
        sec_z = [0.0] * S
        for s in range(S):
            e = W * sizes[s] * K / float(N)
            sec_z[s] = (act[s] - e) / math.sqrt(e + 1.0)

        # entropie normalizata a distributiei hit-urilor pe sectoare -> gate
        if total_hits > 0 and S > 1:
            Hh = 0.0
            for s in range(S):
                if act[s] > 0.0:
                    p = act[s] / total_hits
                    Hh -= p * math.log(p)
            Hn = Hh / math.log(S)
            if Hn > 1.0:
                Hn = 1.0
            elif Hn < 0.0:
                Hn = 0.0
        else:
            Hn = 1.0
        gate = (1.0 - GATE_MIX) + GATE_MIX * (1.0 - Hn)

        # persistenta: corelatie intre jumatatea veche si cea noua a ferestrei (diag)
        m1 = sum(a1) / float(S)
        m2 = sum(a2) / float(S)
        v1 = sum((x - m1) * (x - m1) for x in a1)
        v2 = sum((x - m2) * (x - m2) for x in a2)
        cov = 0.0
        for s in range(S):
            cov += (a1[s] - m1) * (a2[s] - m2)
        pz = cov / math.sqrt(v1 * v2) if (v1 > 0.0 and v2 > 0.0) else 0.0

        # deviatie macro: z pe counts totale fata de medie
        mean_c = tot / float(N)
        var_c = 0.0
        for n in range(1, N + 1):
            d = counts.get(n, 0) - mean_c
            var_c += d * d
        std_c = math.sqrt(var_c / float(N)) + 1e-9

        scored = []
        gmin = 1e18
        gmax = -1e18
        dmin = 1e18
        dmax = -1e18
        for n in range(1, N + 1):
            ls = last_seen.get(n)
            gap = draw_index - ls if ls is not None else draw_index + 1
            if gap < 0:
                gap = 0
            g = gap / EXP_GAP
            if g > GAP_CAP:
                g = GAP_CAP
            gt = g - 1.0
            if gt < gmin:
                gmin = gt
            if gt > gmax:
                gmax = gt
            dz = (counts.get(n, 0) - mean_c) / std_c
            if dz < dmin:
                dmin = dz
            if dz > dmax:
                dmax = dz
            sc = W_SEC * SEC_SIGN * sec_z[sec_of[n]] * gate + W_GAP * gt + W_DEV * DEV_SIGN * dz
            scored.append((sc, n))
        scored.sort(key=lambda t: (-t[0], t[1]))
        out = [n for _, n in scored]

        rank._diag = {
            'sector_span': max(sec_z) - min(sec_z),
            'dev_span': dmax - dmin,
            'gap_span': gmax - gmin,
            'entropy_norm': Hn,
            'gate_factor': gate,
            'orient_flips': 0,
            'pers_z': pz,
            'L': W,
            'ORIENT_AUTO': ORIENT_AUTO,
            'SEC_SIZE': S,
            'SEC_SIGN': SEC_SIGN,
            'DEV_SIGN': DEV_SIGN,
            'W_SEC': W_SEC,
            'W_GAP': W_GAP,
            'W_DEV': W_DEV,
            'GAP_CAP': GAP_CAP,
            'GATE_MIX': GATE_MIX,
        }
        cache['k'] = key
        cache['out'] = out
        return out

    return rank