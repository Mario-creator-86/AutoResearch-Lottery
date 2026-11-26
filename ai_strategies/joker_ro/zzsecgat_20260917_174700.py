NAME = "zzsecgat"
DESCRIPTION = 'Model cu 3 mecanisme standardizate: (1) cicluri de intarziere pe sectoare — 1..N partitionat in sectoare de SEC_SIZE, fiecare sector are un interval de returnare asteptat derivat din acoperirea combinatorica P(sector gol intr-un draw)=C(N-s,K)/C(N,K), semnalul este varsta de la ultima aparitie a sectorului standardizata sub geometrica cu p=acoperire, cu semn SEC_SIGN; (2) deviatia macro — z-score autocalibrat al contarilor lifetime fata de media binomiala D_hist*p, semn DEV_SIGN; (3) gap individual — varsta de la ultima aparitie a numarului standardizata sub geometrica(p=K/N) cu cap GAP_CAP, modulata de un entropy-gate: entropia Shannon normalizata a distributiei numarului in fereastra recenta atenueaza gap-ul cand distributia e uniforma si il lasa full-weight cand aparitiile sunt clusterite. Orientarea deque-ului recent este verificata fata de last_seen si corectata automat; DIAG expune span-urile componentelor pentru ablatie.'
SOURCE = 'model + 2026-09-16'

import math

# ---- constante tunabile (grile de scan pentru ablatie) ----
W_SEC = 0.70
SEC_SIZE = 6
SEC_SIGN = 1.0
W_DEV = 0.40
DEV_SIGN = 1.0
W_GAP = 0.50
GAP_CAP = 2.5
GATE_MIX = 0.5
SEC_CLAMP = 3.0

DIAG = {'orient_checks': 0, 'orient_flips': 0, 'sector_span': 0.0, 'dev_span': 0.0, 'gap_span': 0.0, 'entropy_norm': 1.0, 'gate_factor': 1.0}


def create(config):
    N = int(config.number_range)
    KD = int(config.numbers_drawn)
    p = KD / float(N) if N > 0 else 0.0
    q = 1.0 - p

    n_sec = (N + SEC_SIZE - 1) // SEC_SIZE
    c_all = math.comb(N, KD) if KD >= 0 and KD <= N else 0

    # parametri geometrici per sector: mu=q_cov/p_cov, sd=sqrt(q_cov)/p_cov
    sec_mu = [0.0] * n_sec
    sec_sd = [1.0] * n_sec
    for s in range(n_sec):
        size = N - s * SEC_SIZE
        if size > SEC_SIZE:
            size = SEC_SIZE
        if size <= 0:
            size = SEC_SIZE
        miss = 0.0
        if c_all > 0 and (N - size) >= KD and KD >= 0:
            miss = math.comb(N - size, KD) / float(c_all)
        pc = 1.0 - miss
        qc = miss
        if 0.0 < pc < 1.0:
            sec_mu[s] = qc / pc
            sd = math.sqrt(qc) / pc
            sec_sd[s] = sd if sd > 1e-9 else 1.0
        else:
            sec_mu[s] = 0.0
            sec_sd[s] = 1.0

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        D = int(draw_index)
        numbers = list(range(1, N + 1))

        # --- orientare deque: seq[-1] ar trebui sa fie cel mai nou draw ---
        seq = [set(s) for s in recent_deque]
        Wn = len(seq)
        if Wn >= 2 and D >= 1:
            DIAG['orient_checks'] += 1
            so = 0
            sf = 0
            for n in seq[-1]:
                if last_seen.get(n, -10 ** 9) == D - 1:
                    so += 1
            for n in seq[0]:
                if last_seen.get(n, -10 ** 9) == D - 1:
                    sf += 1
            if sf > so:
                seq.reverse()
                DIAG['orient_flips'] += 1

        # --- (2) deviatie macro autocalibrata ---
        total_slots = 0
        for n in numbers:
            total_slots += counts.get(n, 0)
        z_dev = [0.0] * N
        if KD > 0 and total_slots > 0:
            D_hist = total_slots / float(KD)
            if D_hist >= 1.0 and 0.0 < p < 1.0:
                mu = D_hist * p
                sd = math.sqrt(D_hist * p * q)
                if sd > 1e-9:
                    for i in range(N):
                        z_dev[i] = (counts.get(numbers[i], 0) - mu) / sd

        # --- (3) gap individual sub geometrica(p), cu cap ---
        z_gap = [0.0] * N
        if 0.0 < p < 1.0:
            mu_g = q / p
            sd_g = math.sqrt(q) / p
            for i in range(N):
                ls = last_seen.get(numbers[i], None)
                g = D - ls if ls is not None else D + 1
                if g < 0:
                    g = 0
                zv = (g - mu_g) / sd_g
                if zv > GAP_CAP:
                    zv = GAP_CAP
                elif zv < -GAP_CAP:
                    zv = -GAP_CAP
                z_gap[i] = zv

        # --- entropy gate pe fereastra recenta ---
        toks = {}
        for sset in seq:
            for n in sset:
                toks[n] = toks.get(n, 0) + 1
        tot = 0
        for n in toks:
            tot += toks[n]
        Hn = 1.0
        if tot > 0 and len(toks) >= 2:
            H = 0.0
            for n in toks:
                fr = toks[n] / float(tot)
                H -= fr * math.log(fr)
            Hn = H / math.log(len(toks))
            if Hn > 1.0:
                Hn = 1.0
            elif Hn < 0.0:
                Hn = 0.0
        gate = 1.0 - GATE_MIX * Hn

        # --- (1) varsta sectorului standardizata ---
        sec_ls = [-1] * n_sec
        for n in numbers:
            ls = last_seen.get(n, None)
            if ls is not None:
                s = (n - 1) // SEC_SIZE
                if s < n_sec and ls > sec_ls[s]:
                    sec_ls[s] = ls
        z_sec = [0.0] * N
        for i in range(N):
            n = numbers[i]
            s = (n - 1) // SEC_SIZE
            if s < n_sec:
                age = D - sec_ls[s]
                if age < 0:
                    age = 0
                zv = (age - sec_mu[s]) / sec_sd[s]
                if zv > SEC_CLAMP:
                    zv = SEC_CLAMP
                elif zv < -SEC_CLAMP:
                    zv = -SEC_CLAMP
                z_sec[i] = zv

        # --- DIAG: span-uri pentru ablatie ---
        DIAG['sector_span'] = max(z_sec) - min(z_sec)
        DIAG['dev_span'] = max(z_dev) - min(z_dev)
        DIAG['gap_span'] = max(z_gap) - min(z_gap)
        DIAG['entropy_norm'] = Hn
        DIAG['gate_factor'] = gate

        # --- scor compozit ---
        scores = [0.0] * N
        gw = W_GAP * gate
        for i in range(N):
            scores[i] = W_SEC * SEC_SIGN * z_sec[i] + W_DEV * DEV_SIGN * z_dev[i] + gw * z_gap[i]

        order = sorted(range(N), key=lambda i: (-scores[i], numbers[i]))
        return [numbers[i] for i in order]

    return rank