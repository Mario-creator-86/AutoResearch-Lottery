import math

NAME = "zgapcycl"
DESCRIPTION = 'Trei mecanisme de scor: (1) nucleu pe gap-ul personal fata de gap-ul asteptat P/W, amestec GATE_MIX intre momentum exp(-g/TAU) si reversie saturata la GAP_CAP; (2) reversie pe sectoare de SEC_SIZE numere, boost pentru deficitul relativ din ultimele 30 extrageri; (3) reversie long-run fata de frecventa medie, -tanh(z/2). Sortare descrescatoare, tie-break pe numar.'
SOURCE = 'model + 2026-09-17'


def create(config):
    P = int(config.number_range)
    WD = int(config.numbers_drawn)
    exp_gap = float(P) / float(WD) if WD > 0 else 4.0

    # knob-uri de grid (max 3 mecanisme de scor)
    SEC_SIZE = 8      # {4, 6, 8}
    SEC_SIGN = 1      # {+1, -1}: +1 = reversion (deficit => boost)
    DEV_SIGN = 1      # {+1, -1}: +1 = reversion fata de frecventa medie
    W_GAP = 0.8       # {0.3, 0.5, 0.8}
    W_SEC = 0.7       # {0.35, 0.7, 1.05}
    W_DEV = 0.4       # {0, 0.4, 0.8}
    GAP_CAP = 2.5     # {1.5, 2.5, 4.0} in multipli de exp_gap
    GATE_MIX = 0.3    # {0.3, 0.5, 0.7}: pondere momentum vs reversie pe gap
    TAU = 12.0
    SEC_WIN = 30
    DEV_SCALE = 2.0

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, P + 1))

        # deviatia long-run fata de frecventa medie (z, limitata cu tanh)
        vals = [counts.get(n, 0) for n in numbers]
        mean_c = float(sum(vals)) / P
        var_c = sum((v - mean_c) * (v - mean_c) for v in vals) / float(P)
        std_c = math.sqrt(var_c)
        if std_c <= 0.0:
            std_c = 1.0

        # contoare de sector pe fereastra recenta; independent de ordinea deque-ului
        k = len(recent_deque)
        if k > SEC_WIN:
            k = SEC_WIN
        n_sect = (P + SEC_SIZE - 1) // SEC_SIZE
        sec_cnt = [0] * n_sect
        tot_seen = 0
        for i in range(k):
            s = recent_deque[i]
            tot_seen += len(s)
            for n in s:
                si = (n - 1) // SEC_SIZE
                if si >= n_sect:
                    si = n_sect - 1
                sec_cnt[si] += 1
        exp_sec = float(tot_seen) / n_sect if tot_seen > 0 else 0.0

        cap = GAP_CAP * exp_gap
        scores = {}
        for n in numbers:
            ls = last_seen.get(n)
            if ls is None:
                g = float(draw_index + 1)
            else:
                g = float(draw_index - ls)
            if g < 0.0:
                g = 0.0
            mom = math.exp(-g / TAU)
            rev = g / cap
            if rev > 1.0:
                rev = 1.0
            gap_score = GATE_MIX * mom + (1.0 - GATE_MIX) * rev
            si = (n - 1) // SEC_SIZE
            if si >= n_sect:
                si = n_sect - 1
            if exp_sec > 0.0:
                sec = SEC_SIGN * (exp_sec - sec_cnt[si]) / exp_sec
            else:
                sec = 0.0
            z = (counts.get(n, 0) - mean_c) / std_c
            dev = DEV_SIGN * (-math.tanh(z / DEV_SCALE))
            scores[n] = W_GAP * gap_score + W_SEC * sec + W_DEV * dev

        return sorted(numbers, key=lambda nn: (-scores[nn], nn))

    return rank