import math

NAME = "zovregpa"
DESCRIPTION = 'Regim de persistenta intre extrageri consecutive (overlap ultimelor 2 extrageri vs asteptare hipergeometrica) moduleaza ponderile de momentum; scor = z frecventa in fereastra + afinitate de co-aparitie cu ultima extragere (matrice de perechi incremental) + hazard de intarziere taiat + termen non-liniar, cu modulare adaptiva dupa entropia ferestrei.'
SOURCE = 'glm-5.3-flash:cloud + 2026-09-20'


def create(config):
    # config: LotteryConfig (.number_range, .numbers_drawn, .ticket_size)
    NR = int(config.number_range)
    ND = int(config.numbers_drawn)
    KW_MAX = 90
    G_CAP = 7.0
    WL0, WL, WR, WG = 0.15, 0.60, 0.40, 0.10
    AINT, ENT, REGW, REP = 0.08, 0.50, 0.15, 0.35
    P_DRAW = ND / NR if NR > 0 else 0.5
    Q_PAIR = (ND / NR) * ((ND - 1) / (NR - 1)) if NR > 1 else 0.25
    OVL_MEAN = ND * ND / NR if NR > 0 else 0.0
    OVL_SD = math.sqrt(max(ND * (ND / NR) * (1.0 - ND / NR) * ((NR - ND) / max(NR - 1, 1)), 1e-9))

    st = {'idx': None, 'buf': [], 'C': None, 'r': None}

    def _add(s, sign):
        C = st['C']
        r = st['r']
        sn = sorted(s)
        for i in range(len(sn)):
            a = sn[i]
            r[a] += sign
            Ca = C[a]
            for j in range(i + 1, len(sn)):
                b = sn[j]
                Ca[b] += sign
                C[b][a] += sign

    def _fresh(buf):
        st['buf'] = list(buf)
        st['C'] = [[0] * (NR + 2) for _ in range(NR + 2)]
        st['r'] = [0] * (NR + 2)
        for s in st['buf']:
            _add(s, 1)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, NR + 1))
        vals = [counts.get(n, 0) for n in numbers]
        ml = sum(vals) / NR
        va = sum((v - ml) ** 2 for v in vals) / NR
        sl = math.sqrt(va) or 1.0

        dq = list(recent_deque)
        Wd = len(dq)
        if Wd < 2:
            zl = {n: (counts.get(n, 0) - ml) / sl for n in numbers}
            return sorted(numbers, key=lambda n: (-zl[n], n))

        ms_last = max(last_seen.get(n, -1) for n in dq[-1])
        ms_first = max(last_seen.get(n, -1) for n in dq[0])
        if ms_first > ms_last:
            dq.reverse()

        target = min(Wd, KW_MAX)
        if st['C'] is None:
            _fresh(dq[-target:])
        elif draw_index == st['idx']:
            pass
        elif draw_index == st['idx'] + 1:
            if dq[-1] is not st['buf'][-1]:
                _add(dq[-1], 1)
                st['buf'].append(dq[-1])
            while len(st['buf']) > target:
                _add(st['buf'].pop(0), -1)
            if len(st['buf']) < target:
                _fresh(dq[-target:])
        else:
            _fresh(dq[-target:])
        st['idx'] = draw_index

        buf = st['buf']
        W = len(buf)
        C = st['C']
        r = st['r']
        newest = buf[-1]

        mu_w = W * P_DRAW
        sd_w = math.sqrt(max(W * P_DRAW * (1.0 - P_DRAW), 1e-9))
        z_win = {n: (r[n] - mu_w) / sd_w for n in numbers}

        L = sorted(newest)
        nL = len(L)
        z_aff = {}
        for n in numbers:
            k = nL - 1 if n in newest else nL
            mu_n = k * W * Q_PAIR
            sd_n = math.sqrt(max(mu_n * (1.0 - Q_PAIR), 1e-9))
            s = 0
            for m in L:
                if m != n:
                    s += C[n][m]
            z_aff[n] = (s - mu_n) / sd_n

        if W >= 2:
            ovl = len(newest & buf[-2])
            reg = (ovl - OVL_MEAN) / OVL_SD
        else:
            ovl = 0
            reg = 0.0
        mreg = 1.0 + REGW * reg
        if mreg < 0.55:
            mreg = 0.55
        elif mreg > 1.45:
            mreg = 1.45

        T = W * ND
        H = 0.0
        for n in numbers:
            rn = r[n]
            if rn > 0:
                p = rn / T
                H -= p * math.log(p)
        conc = 1.0 - H / math.log(NR)
        if conc < 0.0:
            conc = 0.0
        elif conc > 1.0:
            conc = 1.0

        hz = {}
        for n in numbers:
            ls = last_seen.get(n, -1)
            g = (draw_index - 1) - ls if ls >= 0 else draw_index
            if g < 0:
                g = 0
            hz[n] = (g if g < G_CAP else G_CAP) / G_CAP

        WL_eff = WL * mreg
        WR_eff = WR * mreg * (1.0 + ENT * conc)
        t1 = {n: math.tanh(z_win[n] * 0.5) for n in numbers}
        t2 = {n: math.tanh(z_aff[n] * 0.5) for n in numbers}

        def score(n):
            rep = REP * (mreg - 1.0) if n in newest else 0.0
            return (WL0 * ((counts.get(n, 0) - ml) / sl)
                    + WL_eff * z_win[n]
                    + WR_eff * z_aff[n]
                    + WG * hz[n]
                    + AINT * t1[n] * t2[n]
                    + rep)

        out = sorted(numbers, key=lambda n: (-score(n), n))
        d = rank._diag
        d['ovl'] = float(ovl)
        d['reg'] = float(reg)
        d['mreg'] = float(mreg)
        d['conc'] = float(conc)
        d['z_win_mean'] = sum(z_win.values()) / NR
        d['z_aff_mean'] = sum(z_aff.values()) / NR
        return out

    rank._diag = {'ovl': 0.0, 'reg': 0.0, 'mreg': 1.0, 'conc': 0.0, 'z_win_mean': 0.0, 'z_aff_mean': 0.0}
    return rank