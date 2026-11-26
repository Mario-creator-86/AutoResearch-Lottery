import math

NAME = "zzpairaf"
DESCRIPTION = 'Familie noua de ordinul doi: pair-lift conditionat (excesul de caldura al co-frecventistilor unui numar, EWMA-ponderat), lift de tranzitie lag-1 (probabilitate conditionata data ultima extragere, corectata cu marginea weighted), memorie scurta pe ultimele RG extrageri si EWMA pe frecvente, amestecate printr-un gate de entropie al ferestrei recente.'
SOURCE = 'model + 2026-09-18'

# Parametri la nivel de modul: grid-ul ii poate suprascrie (atribute config sau config.params).
KD = 20        # jumatate de viata EWMA (extrageri)
WIN_CAP = 160  # limita ferestrei efective pentru ew/perechi/tranzitii
RG = 8         # fereastra memoriei scurte (repeat)
W_L1 = 2.2     # pondere pair-lift conditionat
W_L2 = 0.8     # pondere tranzitie lag-1 corectata marginal
W_REP = 1.5    # pondere repeat scurt
W_EW = 0.35    # pondere EWMA
W_CNT = 0.35   # pondere counts istorice (z)
GATE_K = 0.75  # intensitatea gate-ului de entropie


def create(config):
    P = int(config.number_range)
    WD = int(config.numbers_drawn)

    def _ovr(name, default):
        v = getattr(config, name, None)
        if v is None:
            pr = getattr(config, 'params', None)
            if isinstance(pr, dict):
                v = pr.get(name)
        return default if v is None else v

    kd = max(1.0, float(_ovr('KD', KD)))
    win_cap = max(2, int(_ovr('WIN_CAP', WIN_CAP)))
    rg = max(1, int(_ovr('RG', RG)))
    w_l1 = float(_ovr('W_L1', W_L1))
    w_l2 = float(_ovr('W_L2', W_L2))
    w_rep = float(_ovr('W_REP', W_REP))
    w_ew = float(_ovr('W_EW', W_EW))
    w_cnt = float(_ovr('W_CNT', W_CNT))
    gate_k = float(_ovr('GATE_K', GATE_K))

    lam = 0.5 ** (1.0 / kd)
    p_uni = float(WD) / float(P)

    def _z(numbers, vals):
        s = 0.0
        for n in numbers:
            s += vals[n]
        mu = s / P
        v = 0.0
        for n in numbers:
            d = vals[n] - mu
            v += d * d
        sd = math.sqrt(v / P)
        if sd < 1e-12:
            return dict.fromkeys(numbers, 0.0)
        inv = 1.0 / sd
        return {n: (vals[n] - mu) * inv for n in numbers}

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, P + 1))
        cntv = {n: float(counts.get(n, 0)) for n in numbers}
        z_cnt = _z(numbers, cntv)

        draws = list(recent_deque)
        if len(draws) > win_cap:
            draws = draws[len(draws) - win_cap:]
        # sortare determinista a fiecarei extrageri (ordine de adunare fixa)
        draws = [sorted(d) for d in draws]
        D = len(draws)

        spans = {'cnt': max(cntv.values()) - min(cntv.values())}

        if D == 0:
            scores = {n: w_cnt * z_cnt[n] for n in numbers}
            rank._diag = {'draw_index': draw_index, 'D': 0, 'conc': 1.0, 'gate': 0.0, 'span': spans}
            return sorted(numbers, key=lambda n: (-scores[n], n))

        # ponderi recenza: cea mai noua extragere (ultima din deque) are ponderea 1
        ws = [1.0] * D
        acc = 1.0
        for i in range(D - 2, -1, -1):
            acc *= lam
            ws[i] = acc

        # EWMA frecvente + contoare brute (pentru entropie)
        ew = dict.fromkeys(numbers, 0.0)
        raw = dict.fromkeys(numbers, 0)
        wsum = 0.0
        for i in range(D):
            w = ws[i]
            wsum += w
            for m in draws[i]:
                ew[m] += w
                raw[m] += 1

        # memorie scurta: aparitii in ultimele RG extrageri
        rep = dict.fromkeys(numbers, 0.0)
        for i in range(max(0, D - rg), D):
            for m in draws[i]:
                rep[m] += 1.0

        # exces weighted fata de asteptarea uniforma
        ex = {n: ew[n] - wsum * p_uni for n in numbers}

        # lift1: media ponderata a excesului partenerilor la aparitiile lui n
        # (normalizarea eliminate redundanta de ordinul unu cu EWMA)
        l1 = dict.fromkeys(numbers, 0.0)
        wown = dict.fromkeys(numbers, 0.0)
        for i in range(D):
            w = ws[i]
            dr = draws[i]
            ti = 0.0
            for m in dr:
                ti += ex[m]
            for n in dr:
                l1[n] += w * (ti - ex[n])
                wown[n] += w
        for n in numbers:
            wo = wown[n]
            l1[n] = l1[n] / wo if wo > 1e-12 else 0.0

        # lift2: tranzitii lag-1, ancorate pe ultima extragere, corectate cu marginea
        l2 = dict.fromkeys(numbers, 0.0)
        if D >= 2:
            trans = [[0.0] * (P + 1) for _ in range(P + 1)]
            for j in range(D - 1):
                w = ws[j + 1]
                src = draws[j]
                dst = draws[j + 1]
                for a in src:
                    ra = trans[a]
                    for b in dst:
                        ra[b] += w
            last = draws[-1]
            s_out = 0.0
            for m in last:
                row = trans[m]
                for b in range(1, P + 1):
                    s_out += row[b]
            if s_out > 1e-12:
                inv_out = 1.0 / s_out
                for n in numbers:
                    num = 0.0
                    for m in last:
                        num += trans[m][n]
                    l2[n] = num * inv_out - ew[n] / wsum

        z1 = _z(numbers, l1)
        z2 = _z(numbers, l2)
        zr = _z(numbers, rep)
        zew = _z(numbers, ew)

        # entropia distributiei brute din fereastra; scara null analitica
        nballs = 0
        for d in draws:
            nballs += len(d)
        H = 0.0
        if nballs > 0:
            for n in numbers:
                c = raw[n]
                if c > 0:
                    pm = c / nballs
                    H -= pm * math.log(pm)
        hmax_eff = math.log(min(P, nballs)) if nballs > 0 else 0.0
        hmax = math.log(P) if P > 1 else 0.0
        hmax_use = min(hmax, hmax_eff) if nballs < P else hmax
        s0 = (P - WD) / (2.0 * WD * D) if WD > 0 else 1.0
        conc = (hmax_use - H) / s0 if s0 > 1e-12 else 1.0
        if conc < 0.3:
            conc = 0.3
        elif conc > 2.5:
            conc = 2.5
        gate = gate_k * conc

        blk = {n: w_l1 * z1[n] + w_l2 * z2[n] + w_rep * zr[n] + w_ew * zew[n] for n in numbers}
        scores = {n: w_cnt * z_cnt[n] + gate * blk[n] for n in numbers}

        spans['lift1'] = max(l1.values()) - min(l1.values())
        spans['lift2'] = max(l2.values()) - min(l2.values())
        spans['rep'] = max(rep.values()) - min(rep.values())
        spans['ew'] = max(ew.values()) - min(ew.values())
        rank._diag = {'draw_index': draw_index, 'D': D, 'conc': conc, 'gate': gate, 'span': spans}
        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank