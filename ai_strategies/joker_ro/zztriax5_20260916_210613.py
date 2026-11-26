NAME = "zztriax5"
DESCRIPTION = 'Trei axe de echilibru combinatorial: (1) reversie z-standardizata pe sectoare decadal (n//10) cu poarta de entropie a distributiei sectoriale din fereastra recenta; (2) reechilibrarea compozitiei ultimelor ~20 extrageri (suma, paritate, split low/high fata de expected 5/45); (3) cicluri armonice per numar (gap curent vs media gapurilor, la multiplii 1..3). Ancora slaba pe z-score lifetime; determinist, O(N + Wd*KD) per apel.'
SOURCE = 'model + 2026-09-16'

import math

# ---- constante tunabile (grile de ablatie) ----
W_BASE = 0.25      # grid {0, 0.25, 0.5}
W_SECT = 0.35      # grid {0, 0.2, 0.35}
SECT_DIR = 1       # +1 = reversie (boost deficit), -1 = momentum pe sectoare
GATE_GAIN = 1.0    # amplificare poarta entropie: w_eff = W_SECT*(1+GAIN*(1-struct))
W_COMP = 0.30      # grid {0, 0.15, 0.3}
COMPO_K = 20       # grid {10, 20, 40}
W_CYC = 0.30       # grid {0, 0.15, 0.3}
CYC_TOL = 0.30
CYC_KMAX = 3
CYC_MIN_C = 2.0
CYC_MAX_C = 40.0

DIAG = {'calls': 0, 'orient_checks': 0, 'orient_flips': 0,
        'struct_last': 0.0, 'struct_min': 9.9, 'struct_max': -9.9}


def create(config):
    N = int(config.number_range)
    KD = int(config.numbers_drawn)
    if KD < 1:
        KD = 1
    if KD > N:
        KD = N
    p = KD / float(N)
    q = 1.0 - p
    ffd = (N - KD) / float(N - 1) if N > 1 else 1.0  # finite population correction

    # axa sectoriala: decada n//10
    DEC = 10 if N >= 20 else max(1, N // 5)
    S = N // DEC + 1
    sec_of = [0] * (N + 1)
    sec_size = [0] * S
    for n in range(1, N + 1):
        s = n // DEC
        sec_of[n] = s
        sec_size[s] += 1

    # constante pentru compozitia draw-level
    center = (N + 1) / 2.0
    halfspan = (N - 1) / 2.0
    E_sum = KD * center
    var_pop = (N * N - 1) / 12.0
    sd_sum = math.sqrt(max(KD * (N - KD) / float(N - 1) * var_pop, 1e-12))
    n_odd = (N + 1) // 2
    p_odd = n_odd / float(N)
    sd_odd = math.sqrt(max(KD * p_odd * (1.0 - p_odd) * ffd, 1e-12))
    n_low = N // 2
    p_low = n_low / float(N)
    sd_low = math.sqrt(max(KD * p_low * (1.0 - p_low) * ffd, 1e-12))
    lnS = math.log(float(S)) if S > 1 else 1.0

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        DIAG['calls'] += 1
        D = int(draw_index)
        numbers = list(range(1, N + 1))

        # ---- (0) z-score lifetime, autocalibrat (ancora slaba) ----
        total_c = 0
        for n in numbers:
            total_c += counts.get(n, 0)
        base = [0.0] * (N + 1)
        if total_c > 0 and 0.0 < p < 1.0:
            D_hist = total_c / float(KD)
            mu = D_hist * p
            inv = 1.0 / math.sqrt(max(D_hist * p * q, 1e-12))
            for n in numbers:
                base[n] = (counts.get(n, 0) - mu) * inv

        # ---- fereastra cu verificare de orientare (seq[-1] = cel mai nou draw) ----
        seq = [set(s) for s in recent_deque]
        Wd = len(seq)
        if Wd >= 2 and D >= 1:
            DIAG['orient_checks'] += 1
            hit_new = 0
            hit_old = 0
            for n in seq[-1]:
                if last_seen.get(n, -10 ** 9) == D - 1:
                    hit_new += 1
            for n in seq[0]:
                if last_seen.get(n, -10 ** 9) == D - 1:
                    hit_old += 1
            if hit_old > hit_new:
                seq.reverse()
                DIAG['orient_flips'] += 1
        newest_first = list(reversed(seq))  # newest_first[0] = cel mai nou draw

        # ---- (1) reversie pe sectoare, z-standardizata, gated de entropie ----
        z_sec = [0.0] * S
        struct = 1.0
        w_eff = W_SECT
        if Wd > 0:
            cs = [0] * S
            for s in newest_first:
                for n in s:
                    cs[sec_of[n]] += 1
            tot_hits = Wd * KD
            for si in range(S):
                size_s = sec_size[si]
                if size_s == 0:
                    continue
                ps = size_s / float(N)
                E_s = tot_hits * ps
                var_s = tot_hits * ps * (1.0 - ps) * ffd
                z_sec[si] = (cs[si] - E_s) / math.sqrt(max(var_s, 1e-12))
            if tot_hits > 0 and lnS > 0.0:
                Hs = 0.0
                for si in range(S):
                    qq = cs[si] / float(tot_hits)
                    if qq > 1e-12:
                        Hs -= qq * math.log(qq)
                struct = Hs / lnS
                if struct < 0.0:
                    struct = 0.0
                elif struct > 1.0:
                    struct = 1.0
                DIAG['struct_last'] = struct
                if struct < DIAG['struct_min']:
                    DIAG['struct_min'] = struct
                if struct > DIAG['struct_max']:
                    DIAG['struct_max'] = struct
            w_eff = W_SECT * (1.0 + GATE_GAIN * (1.0 - struct))

        # ---- (2) reechilibrarea compozitiei (suma, paritate, low/high) ----
        comp = [0.0] * (N + 1)
        if Wd > 0:
            K2 = COMPO_K if COMPO_K > 0 else Wd
            if K2 > Wd:
                K2 = Wd
            if K2 < 1:
                K2 = 1
            ssum = 0.0
            sodd = 0.0
            slow = 0.0
            for s in newest_first[:K2]:
                for n in s:
                    ssum += n
                    if n & 1:
                        sodd += 1.0
                    if n <= n_low:
                        slow += 1.0
            se_sum = sd_sum / math.sqrt(K2)
            se_odd = sd_odd / math.sqrt(K2)
            se_low = sd_low / math.sqrt(K2)
            z_sum = (E_sum - ssum / K2) / max(se_sum, 1e-12)
            z_par = (KD * p_odd - sodd / K2) / max(se_odd, 1e-12)
            z_low = (KD * p_low - slow / K2) / max(se_low, 1e-12)
            a2 = 0.5 * W_COMP * z_sum / halfspan
            b2 = 0.25 * W_COMP * z_par
            c2 = 0.25 * W_COMP * z_low
            for n in numbers:
                v = a2 * (n - center)
                if n & 1:
                    v += b2
                else:
                    v -= b2
                if n <= n_low:
                    v += c2
                else:
                    v -= c2
                comp[n] = v

        # ---- (3) cicluri armonice per numar ----
        cyc = [0.0] * (N + 1)
        if Wd >= 3 and W_CYC != 0.0:
            posmap = {}
            for j in range(Wd):
                for n in newest_first[j]:
                    if n in posmap:
                        posmap[n].append(j)
                    else:
                        posmap[n] = [j]
            for n in numbers:
                lst = posmap.get(n)
                if lst is None or len(lst) < 3:
                    continue
                cgap = 0.0
                for i in range(len(lst) - 1):
                    cgap += lst[i] - lst[i + 1]
                cgap /= (len(lst) - 1)
                if cgap < CYC_MIN_C or cgap > CYC_MAX_C:
                    continue
                age = lst[0]
                best = 0.0
                tol = CYC_TOL * cgap
                for k in range(1, CYC_KMAX + 1):
                    dd = (age - k * cgap) / tol
                    g = math.exp(-0.5 * dd * dd)
                    if g > best:
                        best = g
                cyc[n] = W_CYC * best

        # ---- combinare liniara + sortare determinista ----
        scores = [0.0] * (N + 1)
        for n in numbers:
            scores[n] = (W_BASE * base[n]
                         - SECT_DIR * w_eff * z_sec[sec_of[n]]
                         + comp[n]
                         + cyc[n])
        return sorted(numbers, key=lambda n: (-round(scores[n], 9), n))

    return rank