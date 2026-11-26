NAME = "zzpairli"
DESCRIPTION = 'Memorie scurta Markov pe relatii: lift de exces standardizat (c-e)/sqrt(e+0.25) pentru perechi co-aparute de cel putin 2 ori (L1) si pentru ecouri la distanta exact lag-2 (L2) in fereastra RG, plus lift de tranzitie lag-1 (REP) conditionat de ultimul draw; blocul de lift, modulat de poarta de entropie GATE_K a distributiei EWMA, se combina liniar cu momentum EWMA (semi-viata KD).'
SOURCE = 'model cantitativ + 2026-09-17'

import math

# --- knoburi expuse pentru grid-search (plan de ablatie din Decision Ledger) ---
W_L1 = 2.2       # greutate pair-lift (co-aparitii repetate in fereastra)
W_L2 = 0.6       # greutate echo-pair lag-2 (prima candidata la ablatie)
W_REP = 1.5      # greutate tranzitie lag-1 conditionata de ultimul draw
W_EW = 0.35      # momentum EWMA
KD = 20          # semi-viata EWMA (in extrageri)
RG = 16          # fereastra pt statistici de perechi/tranzitii
GATE_K = 0.75    # intensitatea portii de entropie
LIFT_SIGN = 1.0  # +1 momentum, -1 reversion (doar blocul pair-lift L1/L2)
VCAP = 4.0       # clip pe excesul standardizat


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)  # 6 pentru 6/90; nu e folosit direct (marjinale empirice)

    # diagnostics per instanta (agregate pe toate apelurile rank)
    DIAG = {
        'calls': 0, 'orient': None, 'orient_flips': 0,
        'span_sum': {'L1': 0.0, 'L2': 0.0, 'REP': 0.0, 'EW': 0.0},
        'span_min': {'L1': None, 'L2': None, 'REP': None, 'EW': None},
        'span_max': {'L1': None, 'L2': None, 'REP': None, 'EW': None},
        'gate_sum': 0.0, 'H_sum': 0.0,
        'avg_span': None, 'avg_gate': None, 'avg_H': None,
        'note': 'hits per-draw se solicita din harness; aici doar spanuri de contributie',
    }

    def zvec(vals):
        m = 0.0
        for v in vals:
            m += v
        m /= float(len(vals))
        var = 0.0
        for v in vals:
            d = v - m
            var += d * d
        sd = math.sqrt(var / float(len(vals)))
        if sd < 1e-12:
            return [0.0] * len(vals)
        return [(v - m) / sd for v in vals]

    def excess(c, e):
        v = (c - e) / math.sqrt(e + 0.25)
        if v > VCAP:
            v = VCAP
        elif v < -VCAP:
            v = -VCAP
        return v

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # counts (istoric complet) nu e folosit: strategia este strict de memorie scurta
        numbers = list(range(1, N + 1))
        L = len(recent_deque)

        # --- detectarea orientarii deque-ului: setul cel mai nou are recenta medie minima ---
        def mean_recency(st):
            if not st:
                return draw_index + 1.0
            tot = 0.0
            for n in st:
                ls = last_seen.get(n)
                tot += (draw_index - ls) if ls is not None else float(draw_index + 1)
            return tot / float(len(st))

        newest_last = True
        if L >= 2:
            newest_last = mean_recency(recent_deque[-1]) <= mean_recency(recent_deque[0])
        orient = 'newest_last' if newest_last else 'newest_first'
        if DIAG['orient'] is not None and DIAG['orient'] != orient:
            DIAG['orient_flips'] += 1
        DIAG['orient'] = orient

        seq = list(recent_deque)
        if newest_last and L >= 2:
            seq.reverse()  # seq[0] = cel mai nou draw

        # --- EWMA pe frecvente (semi-viata KD) ---
        decay = 0.5 ** (1.0 / float(KD))
        ew = [0.0] * (N + 1)
        w = 1.0
        for st in seq:
            for n in st:
                if 1 <= n <= N:
                    ew[n] += w
            w *= decay
        ew_z = zvec(ew[1:])

        # --- entropia normalizata a distributiei EWMA = sursa portii ---
        tot_ew = 0.0
        for v in ew[1:]:
            tot_ew += v
        H = 0.0
        if tot_ew > 0.0:
            for v in ew[1:]:
                if v > 0.0:
                    q = v / tot_ew
                    H -= q * math.log(q)
            H /= math.log(float(N))
            if H > 1.0:
                H = 1.0
        gate = 1.0 - GATE_K * (1.0 - H)
        if gate < 0.0:
            gate = 0.0

        R = RG if RG < L else L

        # --- L1: pair-lift; doar perechi co-aparute de >= 2 ori in cele mai noi R drawuri ---
        l1 = [0.0] * (N + 1)
        if R >= 2:
            mf = [0] * (N + 1)
            pc = {}
            for t in range(R):
                arr = [n for n in seq[t] if 1 <= n <= N]
                for n in arr:
                    mf[n] += 1
                na = len(arr)
                for i in range(na):
                    a = arr[i]
                    for j in range(i + 1, na):
                        b = arr[j]
                        key = (a, b) if a < b else (b, a)
                        pc[key] = pc.get(key, 0) + 1
            for (a, b), c in pc.items():
                if c >= 2:
                    e = (mf[a] * mf[b]) / float(R)
                    v = excess(float(c), e)
                    l1[a] += v
                    l1[b] += v
        l1_z = zvec(l1[1:])

        # --- L2: echo lag-2; perechi la distanta exact 2 drawuri, de >= 2 ori ---
        l2 = [0.0] * (N + 1)
        if R >= 4:
            ma = [0] * (N + 1)
            mb = [0] * (N + 1)
            dc = {}
            for t in range(R - 2):
                sa = [n for n in seq[t] if 1 <= n <= N]
                sb = [n for n in seq[t + 2] if 1 <= n <= N]
                for n in sa:
                    ma[n] += 1
                for n in sb:
                    mb[n] += 1
                for a in sa:
                    for b in sb:
                        key = (a, b)
                        dc[key] = dc.get(key, 0) + 1
            for (a, b), c in dc.items():
                if c >= 2:
                    e = (ma[a] * mb[b]) / float(R - 2)
                    v = excess(float(c), e)
                    l2[a] += v
                    l2[b] += v
        l2_z = zvec(l2[1:])

        # --- REP: tranzitie lag-1 (mai-vechi -> mai-nou) repetata de >= 2 ori,
        #     conditionata de ultimul draw (membrii sai joaca rolul elementului mai vechi) ---
        rep = [0.0] * (N + 1)
        if R >= 3:
            mo = [0] * (N + 1)  # marjinala pe partea mai veche (pozitiile 1..R-1)
            mi = [0] * (N + 1)  # marjinala pe partea mai noua (pozitiile 0..R-2)
            tc = {}
            for t in range(R - 1):
                so = [n for n in seq[t + 1] if 1 <= n <= N]
                sw = [n for n in seq[t] if 1 <= n <= N]
                for n in so:
                    mo[n] += 1
                for n in sw:
                    mi[n] += 1
                for o in so:
                    for v2 in sw:
                        key = (o, v2)
                        tc[key] = tc.get(key, 0) + 1
            last = seq[0]
            for o in last:
                if o < 1 or o > N:
                    continue
                moo = mo[o]
                for n in numbers:
                    c = tc.get((o, n), 0)
                    if c >= 2:
                        e = (moo * mi[n]) / float(R - 1)
                        rep[n] += excess(float(c), e)
        rep_z = zvec(rep[1:])

        # --- combinare liniara; poarta de entropie moduleaza doar blocul de lift ---
        score = [0.0] * N
        for i in range(N):
            liftc = LIFT_SIGN * W_L1 * l1_z[i] + W_L2 * l2_z[i] + W_REP * rep_z[i]
            score[i] = W_EW * ew_z[i] + gate * liftc

        # --- diag: span max-min per componenta, agregat pe extrageri ---
        for name, vec in (('L1', l1_z), ('L2', l2_z), ('REP', rep_z), ('EW', ew_z)):
            sp = max(vec) - min(vec)
            DIAG['span_sum'][name] += sp
            smin = DIAG['span_min'][name]
            smax = DIAG['span_max'][name]
            if smin is None or sp < smin:
                DIAG['span_min'][name] = sp
            if smax is None or sp > smax:
                DIAG['span_max'][name] = sp
        DIAG['gate_sum'] += gate
        DIAG['H_sum'] += H
        DIAG['calls'] += 1
        cc = float(DIAG['calls'])
        DIAG['avg_span'] = {k2: DIAG['span_sum'][k2] / cc for k2 in DIAG['span_sum']}
        DIAG['avg_gate'] = DIAG['gate_sum'] / cc
        DIAG['avg_H'] = DIAG['H_sum'] / cc

        order = sorted(range(N), key=lambda i: (-score[i], i))
        return [numbers[i] for i in order]

    rank._diag = DIAG
    return rank