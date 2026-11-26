NAME = "zzsecwav"
DESCRIPTION = "Strategie pe cicluri de intarziere pe sectoare: 1..N este partitionat in blocuri consecutive de SEC_SIZE numere; pentru fiecare sector se masoara seceta curenta (draws de la ultimul hit al sectorului) si se standardizeaza sub distributia geometrica cu probabilitatea de hit a sectorului calculata exact hipergeometric (1 - C(N-m,K)/C(N,K) prin produs, nu m/N). La nivel de numar se folosesc gap-ul individual de la ultima aparitie (capuit la GAP_CAP x media geometrica q/p) si deviatia frecventei in fereastra recenta fata de asteptat; GATE_MIX amesteca semnalul gros de sector cu cel fin per-numar. Orientarea deque-ului este verificata fata de last_seen si corectata automat."
SOURCE = "model + 2026-09-17"

import math

# ---- constante tunabile (grile de scan pentru ablatie) ----
SEC_SIZE = 6
SEC_SIGN = 1.0
DEV_SIGN = 1.0
W_SEC = 0.70
W_GAP = 0.50
W_DEV = 0.40
GAP_CAP = 2.5
GATE_MIX = 0.5
DEV_WIN = 50

DIAG = {'calls': 0, 'orient_checks': 0, 'orient_flips': 0, 'orient_rule': 'lastseen-hit'}


def create(config):
    N = int(config.number_range)
    KD = int(config.numbers_drawn)
    p = KD / float(N) if N > 0 else 0.0
    q = 1.0 - p

    # momente geometrice per numar: gap G cu P(g=k) = p*q^k
    if 0.0 < p < 1.0:
        mu_gap = q / p
        sd_gap = math.sqrt(q) / p
    else:
        mu_gap = 0.0
        sd_gap = 1.0
    if sd_gap < 1e-9:
        sd_gap = 1e-9
    cap_gap = GAP_CAP * mu_gap

    # sectoare de SEC_SIZE numere consecutive; probabilitatea de hit a sectorului exacta:
    # ps = 1 - prod_{i=0..KD-1} (N-m-i)/(N-i)  [hipergeometric]
    sec_ids = [0] * (N + 1)
    sec_members = []
    sec_mu = []
    sec_sd = []
    start = 1
    while start <= N:
        end = min(start + SEC_SIZE - 1, N)
        members = list(range(start, end + 1))
        m = len(members)
        sid = len(sec_members)
        for n in members:
            sec_ids[n] = sid
        sec_members.append(members)
        if KD >= N - m + 1:
            ps = 1.0
        elif 0.0 < p < 1.0:
            ratio = 1.0
            for i in range(KD):
                ratio *= (float(N - m - i) / float(N - i))
            ps = 1.0 - ratio
            if ps <= 0.0:
                ps = 1e-9
        else:
            ps = 1e-9
        qs = 1.0 - ps
        if 0.0 < ps < 1.0:
            mu_s = qs / ps
            sd_s = math.sqrt(qs) / ps
        else:
            mu_s = 0.0
            sd_s = 1.0
        sec_mu.append(mu_s)
        sec_sd.append(sd_s if sd_s > 1e-9 else 1e-9)
        start = end + 1
    S = len(sec_members)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        DIAG['calls'] += 1
        D = int(draw_index)
        numbers = list(range(1, N + 1))

        # gap per numar: draws de la ultima aparitie; niciodata vazut = D
        gaps = [0] * (N + 1)
        for n in numbers:
            ls = last_seen.get(n, -1)
            g = (D - 1 - ls) if ls >= 0 else D
            gaps[n] = g if g > 0 else 0

        # seceta sectorului = minimul gap-urilor membrilor
        sec_gap = [1 << 30] * S
        for n in numbers:
            sid = sec_ids[n]
            if gaps[n] < sec_gap[sid]:
                sec_gap[sid] = gaps[n]

        # deviatia pe fereastra recenta; orientarea deque-ului verificata fata de last_seen
        seq = list(recent_deque)
        newest_last = True
        if len(seq) >= 2 and D >= 1:
            DIAG['orient_checks'] += 1
            so = 0
            sf = 0
            for n in seq[-1]:
                if last_seen.get(n, -(1 << 30)) == D - 1:
                    so += 1
            for n in seq[0]:
                if last_seen.get(n, -(1 << 30)) == D - 1:
                    sf += 1
            if sf > so:
                newest_last = False
                DIAG['orient_flips'] += 1
        if newest_last:
            newest_first = list(reversed(seq))
        else:
            newest_first = seq

        occ = {}
        W_eff = 0
        for s in newest_first[:DEV_WIN]:
            W_eff += 1
            for n in s:
                occ[n] = occ.get(n, 0) + 1

        z_dev = [0.0] * (N + 1)
        if W_eff > 0 and 0.0 < p < 1.0:
            mu_w = W_eff * p
            sd_w = math.sqrt(W_eff * p * q)
            if sd_w > 1e-9:
                inv = 1.0 / sd_w
                for n in numbers:
                    z_dev[n] = (occ.get(n, 0) - mu_w) * inv

        gm = GATE_MIX
        scores = [0.0] * (N + 1)
        for n in numbers:
            g = gaps[n]
            if g > cap_gap:
                g = cap_gap
            z_gap = (g - mu_gap) / sd_gap
            sid = sec_ids[n]
            z_sec = (sec_gap[sid] - sec_mu[sid]) / sec_sd[sid]
            coarse = W_SEC * (SEC_SIGN * z_sec)
            fine = W_GAP * z_gap + W_DEV * (DEV_SIGN * z_dev[n])
            scores[n] = gm * coarse + (1.0 - gm) * fine

        return sorted(numbers, key=lambda n: (-scores[n], n))

    rank._diag = DIAG
    return rank