import math

NAME = "zliftple"
DESCRIPTION = 'Combinatie neliniara de patru mecanisme z-scorate: pair-lift pe co-ocurrente ponderate exponential (L1), interactie de ordin 2 lift-partener-x-numar-fierbinte (L2), prezenta recenta pe orizont scurt (REP) si EWMA de frecventa (EW). Un gate de regim (rata de repetitii fata de asteptat, fereastra RG, forta GATE_K) moduleaza mecanismele de momentum. Returneaza permutarea completa 1..49.'
SOURCE = 'glm-5.3-flash:cloud + 2026-09-18'

# Parametri pentru grid/ablatie (conventia run-urilor viitoare):
# ablatie: W_L2=0 intai, apoi W_EW=0 daca |delta avg_hits|<0.005
# grid: W_L1{1.1,2.2,3.3} x W_REP{0,1.5,3.0} x W_EW{0,0.35,0.7} x KD{12,20,30} x RG{4,8,16} x GATE_K{0.4,0.75,1.2}
# LIFT_SIGN: +1 momentum, -1 reversion (flip doar daca momentum pierde clar; max 3 prunings)
W_L1 = 2.2
W_L2 = 0.6
W_REP = 1.5
W_EW = 0.35
KD = 20.0
RG = 8
GATE_K = 0.75
LIFT_SIGN = 1.0

_EPS = 1e-12


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    p = K / float(N)
    exp_rep_draw = K * p  # ~0.735 numere repetate fata de extragerea anterioara
    numbers = list(range(1, N + 1))
    state = {'key': None, 'feat': None}

    def _build(recent_deque, recent_window):
        draws = list(recent_deque)
        L = len(draws)
        if L == 0:
            z = {n: 0.0 for n in numbers}
            return {'l1': z, 'l2': dict(z), 'rep': dict(z), 'ew': dict(z), 'gate': 0.5, 'H': 0.0, 'L': 0}
        w_list = [0.0] * L
        sum_w = 0.0
        sum_w2 = 0.0
        for i in range(L):
            w = 0.5 ** ((L - 1 - i) / KD)
            w_list[i] = w
            sum_w += w
            sum_w2 += w * w
        ew = {n: 0.0 for n in numbers}
        pres = {n: 0 for n in numbers}
        for i in range(L):
            w = w_list[i]
            for a in draws[i]:
                if 1 <= a <= N:
                    ew[a] += w
                    pres[a] += 1
        exp_ew = p * sum_w
        sd_ew = math.sqrt(max(p * (1.0 - p) * sum_w2, _EPS))
        ew_z = {n: (ew[n] - exp_ew) / sd_ew for n in numbers}
        s1 = {n: 0.0 for n in numbers}
        s2 = {n: 0.0 for n in numbers}
        for i in range(L):
            w = w_list[i]
            sl = [a for a in draws[i] if 1 <= a <= N]
            m = len(sl)
            if m < 2:
                continue
            hot = 0.0
            for a in sl:
                hot += ew_z[a]
            for a in sl:
                s2[a] += w * (hot - ew_z[a])
            for ii in range(m):
                a = sl[ii]
                for jj in range(ii + 1, m):
                    b = sl[jj]
                    s1[a] += w
                    s1[b] += w
        exp_s1 = p * (K - 1) * sum_w
        sd_s1 = math.sqrt(max(exp_s1, _EPS))
        l1_z = {n: (s1[n] - exp_s1) / sd_s1 for n in numbers}
        v2 = [s2[n] for n in numbers]
        mean2 = sum(v2) / float(N)
        var2 = sum((v - mean2) * (v - mean2) for v in v2) / float(N)
        sd2 = math.sqrt(var2)
        l2_z = {n: (s2[n] - mean2) / (sd2 if sd2 > _EPS else 1.0) for n in numbers}
        hr = 4.0
        srw = 0.0
        srw2 = 0.0
        rep = {n: 0.0 for n in numbers}
        for i in range(L):
            w = 0.5 ** ((L - 1 - i) / hr)
            srw += w
            srw2 += w * w
            for a in draws[i]:
                if 1 <= a <= N:
                    rep[a] += w
        exp_rep = p * srw
        sd_rep = math.sqrt(max(p * (1.0 - p) * srw2, _EPS))
        rep_z = {n: (rep[n] - exp_rep) / sd_rep for n in numbers}
        reps = []
        for i in range(1, L):
            reps.append(len(draws[i - 1] & draws[i]))
        rg_list = reps[-RG:] if (RG > 0 and reps) else []
        if rg_list:
            rate = sum(rg_list) / float(len(rg_list))
            sd_mean = math.sqrt(max(K * p * (1.0 - p) / len(rg_list), _EPS))
            gate = 0.5 * (1.0 + math.tanh(GATE_K * (rate - exp_rep_draw) / sd_mean))
        else:
            gate = 0.5
        H = 0.0
        for n in numbers:
            q = pres[n] / float(L)
            if q > 0.0:
                H -= q * math.log(q)
        H = H / math.log(N) if N > 1 else 0.0
        return {'l1': l1_z, 'l2': l2_z, 'rep': rep_z, 'ew': ew_z, 'gate': gate, 'H': H, 'L': L}

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        key = (draw_index, recent_window, len(recent_deque))
        if state['feat'] is None or state['key'] != key:
            state['key'] = key
            state['feat'] = _build(recent_deque, recent_window)
        feat = state['feat']
        gm = 2.0 * feat['gate']
        scores = {}
        for n in numbers:
            scores[n] = gm * (W_L1 * LIFT_SIGN * feat['l1'][n] + W_REP * feat['rep'][n] + W_EW * feat['ew'][n]) + W_L2 * feat['l2'][n]
        cL1 = [gm * W_L1 * LIFT_SIGN * feat['l1'][n] for n in numbers]
        cL2 = [W_L2 * feat['l2'][n] for n in numbers]
        cRE = [gm * W_REP * feat['rep'][n] for n in numbers]
        cEW = [gm * W_EW * feat['ew'][n] for n in numbers]

        def agg(arr):
            m = sum(arr) / float(len(arr))
            v = sum((x - m) * (x - m) for x in arr) / float(len(arr))
            return {'mean': m, 'std': math.sqrt(v), 'min': min(arr), 'max': max(arr), 'span': max(arr) - min(arr)}

        rank._diag = {'draw_index': draw_index, 'gate': feat['gate'], 'gm': gm, 'H': feat['H'], 'L': feat['L'],
                      'L1': agg(cL1), 'L2': agg(cL2), 'REP': agg(cRE), 'EW': agg(cEW)}
        cum = rank._diag_cum
        if cum is None:
            rank._diag_cum = {'n': 1, 'gate': feat['gate'], 'H': feat['H'],
                              'L1': dict(rank._diag['L1']), 'L2': dict(rank._diag['L2']),
                              'REP': dict(rank._diag['REP']), 'EW': dict(rank._diag['EW'])}
        else:
            k = cum['n'] + 1
            cum['n'] = k
            cum['gate'] += (feat['gate'] - cum['gate']) / k
            cum['H'] += (feat['H'] - cum['H']) / k
            for mech in ('L1', 'L2', 'REP', 'EW'):
                for st in ('mean', 'std', 'min', 'max', 'span'):
                    cum[mech][st] += (rank._diag[mech][st] - cum[mech][st]) / k
        return sorted(numbers, key=lambda n: (-scores[n], n))

    rank._diag = None
    rank._diag_cum = None
    rank._params = {'W_L1': W_L1, 'W_L2': W_L2, 'W_REP': W_REP, 'W_EW': W_EW,
                    'KD': KD, 'RG': RG, 'GATE_K': GATE_K, 'LIFT_SIGN': LIFT_SIGN}
    return rank