NAME = "zgapharm"
DESCRIPTION = 'Faza gap-ului personal (g vs media istorica) cu rezonanta armonica la 1x/2x/3x mu, plus rezonanta de perechi ancorata in ultimele extrageri, ecou de ordinul 2 fata de ultima extragere si corectii de deficit pe 7 sectoare si 10 decenii, conditionate de un band-gate pe entropia unei sub-ferestre scurte.'
SOURCE = 'quant-model gaphase v1 + 2026-09-08'

import math

B0 = 0.55          # amplitudine baza overdue (pe faza phi=g/mu)
W_H = 0.90         # pondere armonice
HARM_W = (1.0, 0.55, 0.30)  # ponderi armonici h=1,2,3
SHARP = 2.4        # ascutime armonici (x=(g-h*mu)*SHARP/mu)
S0 = 1.20          # pondere rezonanta perechi
PAIR_WINDOW = 16   # extrageri scanate pentru co-ocurenta perechilor
ANCHORS = 2        # ultimele extrageri folosite ca ancore
E0 = 1.60          # pondere ecou de ordinul 2
ECHO_WINDOW = 10   # extrageri scanate pentru ecou
L1 = 1.00          # blocaj sector (7x7)
L2 = 0.60          # blocaj decenii (10x~5)
SECT_WINDOW = 120  # fereastra pentru deficite sector/deceniu
GAIN = 0.50        # castig tanh pe z-urile de deficit
ENT_WINDOW = 12    # fereastra scurta pentru entropia gate-ului
E_LO = 0.80        # prag inferior gate (hn in [E_LO, E_HI] -> boost)
E_HI = 0.93        # prag superior gate
GATE_BOOST = 0.8   # amplificare cand gate e ON

def create(config):
    N = config.number_range
    K = config.numbers_drawn
    PPAIR = (K * (K - 1)) / float(N * (N - 1))
    lnN = math.log(N)
    n_sec = (N + 6) // 7
    n_dec = (N + 4) // 5
    sec_size = [7] * n_sec
    if N % 7:
        sec_size[-1] = N - (n_sec - 1) * 7
    dec_size = [5] * n_dec
    if N % 5:
        dec_size[-1] = N - (n_dec - 1) * 5

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        D = draw_index
        draws = list(recent_deque)
        nd = len(draws)

        # o singura trecere: aparitii per numar + per sector + per deceniu
        rec = [0] * N
        sec = [0] * n_sec
        dec = [0] * n_dec
        T = SECT_WINDOW if nd > SECT_WINDOW else nd
        if T > 0:
            for s in draws[-T:]:
                for m in s:
                    j = m - 1
                    rec[j] += 1
                    sec[j // 7] += 1
                    dec[j // 5] += 1

        t_sec = [0.0] * n_sec
        t_dec = [0.0] * n_dec
        if T > 0:
            for i in range(n_sec):
                e = T * K * (sec_size[i] / float(N))
                z = (e - sec[i]) / math.sqrt(e if e > 1.0 else 1.0)
                t_sec[i] = L1 * math.tanh(GAIN * z)
            for i in range(n_dec):
                e = T * K * (dec_size[i] / float(N))
                z = (e - dec[i]) / math.sqrt(e if e > 1.0 else 1.0)
                t_dec[i] = L2 * math.tanh(GAIN * z)

        # gate de entropie pe sub-fereastra scurta
        gate_on = False
        TE = ENT_WINDOW if nd > ENT_WINDOW else nd
        if TE >= 3:
            cnt = [0] * N
            for s in draws[-TE:]:
                for m in s:
                    cnt[m - 1] += 1
            tot = float(TE * K)
            H = 0.0
            for i in range(N):
                p = cnt[i] / tot
                if p > 0.0:
                    H -= p * math.log(p)
            hn = H / lnN
            gate_on = (E_LO <= hn <= E_HI)
        boost = (1.0 + GATE_BOOST) if gate_on else 1.0
        s_eff = S0 * boost
        e_eff = E0 * boost

        # perechi co-aparute in fereastra scurta
        pc = {}
        PW = PAIR_WINDOW if nd > PAIR_WINDOW else nd
        if PW > 0:
            for s in draws[-PW:]:
                lst = sorted(s)
                for i in range(K):
                    ai = lst[i]
                    for j in range(i + 1, K):
                        key = (ai, lst[j])
                        pc[key] = pc.get(key, 0) + 1

        # ancore = uniunea ultimelor ANCHORS extrageri
        A = set()
        na = ANCHORS if nd > ANCHORS else nd
        if na > 0:
            for s in draws[-na:]:
                A |= set(s)
        alist = sorted(A)
        nA = len(alist)
        denom = (PPAIR * PW * nA) if (PW > 0 and nA > 0) else 0.0

        # ecou de ordinul 2: overlap cu ultima extragere
        echo = [0.0] * N
        inv_EW = 0.0
        if nd >= 2:
            EW = ECHO_WINDOW if (nd - 1) > ECHO_WINDOW else (nd - 1)
            if EW > 0:
                anchor = draws[-1]
                for s in draws[-(EW + 1):-1]:
                    ov = 0
                    for m in s:
                        if m in anchor:
                            ov += 1
                    if ov:
                        for m in s:
                            echo[m - 1] += ov
                inv_EW = 1.0 / EW

        scores = [0.0] * N
        cget = counts.get
        lget = last_seen.get
        for idx in range(N):
            n = idx + 1
            c = cget(n, 0)
            gn = lget(n)
            g = D - gn if gn is not None else D + 1
            if g < 1:
                g = 1
            mu = (D + 1.0) / (c + 1.0)
            if mu < 1.0:
                mu = 1.0
            sc = B0 * math.tanh((g / mu) * 0.7142857142857143)
            inv = SHARP / mu
            x = (g - mu) * inv
            if -6.0 < x < 6.0:
                sc += W_H * HARM_W[0] * math.exp(-0.5 * x * x)
            x = (g - 2.0 * mu) * inv
            if -6.0 < x < 6.0:
                sc += W_H * HARM_W[1] * math.exp(-0.5 * x * x)
            x = (g - 3.0 * mu) * inv
            if -6.0 < x < 6.0:
                sc += W_H * HARM_W[2] * math.exp(-0.5 * x * x)
            if denom > 0.0:
                inA = n in A
                raw = 0
                for m in alist:
                    if m == n or (inA and m in A):
                        continue
                    key = (n, m) if n < m else (m, n)
                    raw += pc.get(key, 0)
                sc += s_eff * (raw / denom - 1.0)
            if inv_EW > 0.0:
                sc += e_eff * echo[idx] * inv_EW
            sc += t_sec[idx // 7] + t_dec[idx // 5]
            scores[idx] = sc

        return sorted(numbers, key=lambda nn: (-scores[nn - 1], nn))

    return rank