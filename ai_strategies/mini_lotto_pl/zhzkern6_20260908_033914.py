import math

NAME = "zhzkern6"
DESCRIPTION = "Hibrid pe 6 canale: z de frecventa globala (ancora) + kernel gaussian pe gap normalizat (centru 0.35, latime 0.742) + EWMA half-life 8 + datorie z pe sectoare de 7 + perechi cu decay recen + echo al ultimelor extrageri modulat de gate-ul de re-overlap empiric."
SOURCE = "model-quant + auto-learned calibration 2026-09-08"


def create(config):
    N = config.number_range
    K = config.numbers_drawn
    p = K / float(N)
    exp_gap = 1.0 / p
    KC = 0.35
    KW = 0.742
    DEC = 0.5 ** (1.0 / 8.0)
    SEC = 7
    PAIR_CAP = 30
    PO = p
    WK = 0.30
    WE = 0.48
    WD = 0.35
    WP = 0.60
    WG = 0.20

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        T = max(1, int(draw_index))
        E = T * p
        sd = max(0.5, math.sqrt(T * p * (1.0 - p)))

        win = list(recent_deque)
        if win:
            target = set(n for n in numbers if last_seen.get(n, -10 ** 9) == draw_index - 1)
            if target and (len(win[0] & target) < len(win[-1] & target)):
                win.reverse()
        L = len(win)

        ew = [0.0] * (N + 1)
        for i in range(min(L, 64)):
            w = DEC ** i
            for n in win[i]:
                ew[n] += w
        maxew = max(ew) or 1.0

        wcnt = [0] * (N + 1)
        for s in win:
            for n in s:
                wcnt[n] += 1
        debt = [0.0] * (N + 1)
        if L >= 10:
            exp_sec = L * K * SEC / float(N)
            for s0 in range(0, N, SEC):
                act = 0
                for n in range(s0 + 1, s0 + SEC + 1):
                    act += wcnt[n]
                zb = math.tanh(((exp_sec - act) / max(1.0, math.sqrt(exp_sec))) / 2.0)
                for n in range(s0 + 1, s0 + SEC + 1):
                    debt[n] = zb

        pair_w = {}
        for i in range(min(L, PAIR_CAP)):
            s = sorted(win[i])
            w = DEC ** i
            for ai in range(len(s)):
                a = s[ai]
                for bi in range(ai + 1, len(s)):
                    key = (a, s[bi])
                    if key in pair_w:
                        pair_w[key] += w
                    else:
                        pair_w[key] = w
        paf = [0.0] * (N + 1)
        for (a, b), w in pair_w.items():
            paf[a] += w
            paf[b] += w
        maxpaf = max(paf) or 1.0

        inters = []
        for i in range(1, min(L, 25)):
            inters.append(len(win[i] & win[i - 1]))
        if inters:
            r_emp = (sum(inters) / float(len(inters))) / float(K)
        else:
            r_emp = PO
        gate = (r_emp - PO) / (2.0 * PO)
        if gate < 0.0:
            gate = 0.0
        elif gate > 1.0:
            gate = 1.0

        out = []
        for n in numbers:
            c = counts.get(n, 0)
            fz = math.tanh(((c - E) / sd) / 2.0)
            ls = last_seen.get(n, None)
            if ls is None:
                gap = 4 * exp_gap
            else:
                gap = draw_index - ls
                if gap < 0:
                    gap = 0
            g = gap / exp_gap
            d = (g - KC) / KW
            kern = math.exp(-0.5 * d * d)
            sc = fz + WK * kern + WE * (ew[n] / maxew) + WD * debt[n] + WP * (paf[n] / maxpaf)
            echo = 0.0
            if gap == 1:
                echo = 1.0
            elif gap == 2:
                echo = 0.5
            sc += WG * gate * echo
            out.append((sc, n))
        out.sort(key=lambda t: (-t[0], t[1]))
        return [n for _, n in out]

    return rank