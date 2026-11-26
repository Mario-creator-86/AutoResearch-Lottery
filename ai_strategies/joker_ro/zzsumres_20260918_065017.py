import math

NAME = "zzsumres"
DESCRIPTION = 'Construcție non-liniară greedy a permutării: bază dual-rate (z frecvență lifetime autocalibrat + z EWMA pe fereastra recentă), ghidare greedy spre suma-țintă a extragerii (EWMA pe sumele din fereastră, penalizare pătratică pe bugetul marginal rămas), caps moi de paritate 3 impar/2 par și un termen de rezonanță clockwork care recompensează gap-ul curent aproape de gap-ul mediu istoric al fiecărui număr. Orientarea deque-ului este validată și corectată automat față de last_seen; primele ticket_size poziții formează un subset cu sumă și paritate aproape de profilul istoric.'
SOURCE = 'greedy subset construction + 2026-09-18'

# ---- constante tunabile (grile de scan pentru ablatie) ----
WL = 0.60      # greutate z lifetime
WR = 0.50      # greutate z EWMA recent
DEC = 0.95     # decay EWMA pe fereastra recenta
SUM_MIX = 0.70 # mix suma tintă: recent vs media lifetime
MU_SUM = 0.020 # penalizare deviatie de la bugetul marginal de suma
WG_RES = 0.25  # greutate rezonanta clockwork
RES_SIG = 0.60 # latimea rezonantei (fractie din gap-ul mediu)
KPAR = 0.50    # penalizare caps paritate
ODD_CAP = 3
EVEN_CAP = 2

DIAG = {'orient_checks': 0, 'orient_flips': 0, 'empty_window': 0, 'greedy_fallbacks': 0}


def create(config):
    N = int(config.number_range)
    KD = int(config.numbers_drawn)
    TS = int(getattr(config, 'ticket_size', KD) or KD)
    if TS <= 0:
        TS = KD
    p = KD / float(N) if N > 0 else 0.0
    q = 1.0 - p
    base_sum = KD * (N + 1) / 2.0  # suma asteptata a unui extras
    REM = max(TS, KD)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        D = int(draw_index)

        seq = [set(s) for s in recent_deque]
        W = len(seq)
        if recent_window is not None and recent_window > 0 and W > int(recent_window):
            seq = seq[-int(recent_window):]
            W = len(seq)

        # ---- orientare deque: seq[-1] ar trebui sa fie cel mai nou extras ----
        if W >= 2:
            DIAG['orient_checks'] += 1

            def _ms(idx):
                tot = 0.0
                cnt = 0
                for n in seq[idx]:
                    lv = last_seen.get(n)
                    if lv is not None:
                        tot += lv
                        cnt += 1
                return (tot / cnt) if cnt > 0 else -1e18

            if _ms(0) > _ms(W - 1) + 1e-9:
                seq.reverse()
                DIAG['orient_flips'] += 1

        # ---- (1) z-score lifetime autocalibrat ----
        total_slots = 0
        for n in numbers:
            total_slots += counts.get(n, 0)
        D_hist = (total_slots / float(KD)) if KD > 0 else float(D)

        zL = [0.0] * (N + 1)
        if D_hist >= 1.0 and 0.0 < p < 1.0:
            mu = D_hist * p
            inv = 1.0 / math.sqrt(D_hist * p * q)
            for n in numbers:
                zL[n] = (counts.get(n, 0) - mu) * inv

        # ---- (2) z-score EWMA pe fereastra recenta + sumele per extras ----
        zR = [0.0] * (N + 1)
        win_sums = []
        if W > 0:
            acc = [0.0] * (N + 1)
            wsum = 0.0
            w2sum = 0.0
            decay = 1.0
            for a in range(W):
                s = seq[W - 1 - a]  # a=0 -> cel mai nou
                sw = 0
                for n in s:
                    acc[n] += decay
                    sw += n
                win_sums.append((decay, sw))
                wsum += decay
                w2sum += decay * decay
                decay *= DEC
            mu_w = p * wsum
            inv = 1.0 / math.sqrt(max(p * q * w2sum, 1e-12))
            for n in numbers:
                zR[n] = (acc[n] - mu_w) * inv
        else:
            DIAG['empty_window'] += 1

        # ---- (3) suma tintă: EWMA pe sumele ferestrei, mix cu media lifetime ----
        target_sum = base_sum
        if win_sums:
            wnum = 0.0
            wden = 0.0
            for wgt, sw in win_sums:
                wnum += wgt * sw
                wden += wgt
            target_sum = SUM_MIX * (wnum / wden) + (1.0 - SUM_MIX) * base_sum

        # ---- (4) rezonanta clockwork: gap curent vs gap mediu istoric ----
        res = [0.0] * (N + 1)
        if WG_RES != 0.0 and D_hist >= 1.0:
            for n in numbers:
                c = counts.get(n, 0)
                if c <= 0:
                    continue
                mg = D_hist / float(c)
                lv = last_seen.get(n)
                g = float(D - 1 - lv) if lv is not None else float(D)
                if g < 0.0:
                    g = 0.0
                dev = (g - mg) / max(RES_SIG * mg, 1.0)
                res[n] = -WG_RES * dev * dev

        # ---- (5) bază compusă ----
        base = [0.0] * (N + 1)
        for n in numbers:
            base[n] = WL * zL[n] + WR * zR[n] + res[n]

        # ---- (6) constructie greedy: sum- steer + caps paritate ----
        order = []
        chosen = [False] * (N + 1)
        odds = 0
        evens = 0
        budget = target_sum
        for step in range(REM):
            rleft = REM - step
            per = budget / float(rleft)
            best_n = -1
            best_v = -1e300
            for n in numbers:
                if chosen[n]:
                    continue
                eff = base[n] - MU_SUM * (n - per) * (n - per)
                if (n & 1) == 1:
                    over = odds + 1 - ODD_CAP
                    if over > 0:
                        eff -= KPAR * over * over
                else:
                    over = evens + 1 - EVEN_CAP
                    if over > 0:
                        eff -= KPAR * over * over
                if eff > best_v:
                    best_v = eff
                    best_n = n
            if best_n < 0:
                DIAG['greedy_fallbacks'] += 1
                break
            chosen[best_n] = True
            order.append(best_n)
            budget -= best_n
            if (best_n & 1) == 1:
                odds += 1
            else:
                evens += 1

        rest = [n for n in numbers if not chosen[n]]
        rest.sort(key=lambda n: (-base[n], n))
        order.extend(rest)
        return order

    return rank