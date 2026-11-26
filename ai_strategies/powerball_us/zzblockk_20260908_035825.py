import math

NAME = "zzblockk"
DESCRIPTION = 'Multi-scale analog (kNN block) matcher: compares the current block of the last s draws (s in 3,6,12) with every same-length historical block inside the recent window and scores each number by the similarity-weighted frequency with which it appeared in the draw that immediately followed the best-matching analog blocks; the conditional next-draw profile is blended with a long-run frequency z-anchor, a recent-window frequency anchor and a top-set consensus bonus.'
SOURCE = 'model + 2026-09-08'

GAMMA = 0.7
KTOP = 28
QTHRESH = 0.5
SCALES = (3, 6, 12)
SCALE_W = (0.40, 0.35, 0.30)
W_ANALOG = 0.50
W_LONG = 0.30
W_RECENT = 0.20
CONSENSUS = 0.30
TOP_C = 24
MIN_L = 10
MAX_L = 160


def _popcnt(x):
    return bin(x).count('1')


def _zscore(v, n):
    m = sum(v) / n
    var = sum((x - m) * (x - m) for x in v) / n
    sd = math.sqrt(var)
    if sd < 1e-12:
        return None
    return [(x - m) / sd for x in v]


def create(config):
    N = config.number_range

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))

        lst = [set(s) for s in recent_deque]
        L = len(lst)

        if L >= 2:
            def newness(st):
                vs = [last_seen.get(n) for n in st]
                vs = [v for v in vs if v is not None]
                if not vs:
                    return (0, -(1 << 60), -1.0e18)
                eq = 1 if all(v == vs[0] for v in vs) else 0
                return (eq, max(vs), sum(vs) / float(len(vs)))

            if newness(lst[0]) > newness(lst[-1]):
                lst.reverse()

        if L > MAX_L:
            lst = lst[L - MAX_L:]
            L = MAX_L

        masks = []
        for st in lst:
            m = 0
            for n in st:
                if 1 <= n <= N:
                    m |= 1 << (n - 1)
            masks.append(m)

        cs = [float(counts.get(n, 0)) for n in numbers]
        zL = _zscore(cs, N)

        fc = [0.0] * N
        for st in lst:
            for n in st:
                if 1 <= n <= N:
                    fc[n - 1] += 1.0
        zR = _zscore(fc, N) if L > 0 else None

        analog = None
        if L >= MIN_L:
            acc = [0.0] * N
            wsum_tot = 0.0
            for si, s in enumerate(SCALES):
                if L < 2 * s + 2:
                    continue
                S = 0
                for m in masks[L - s:]:
                    S |= m
                sims = []
                for t in range(s, L):
                    A = 0
                    for m in masks[t - s:t]:
                        A |= m
                    inter = _popcnt(A & S)
                    if inter:
                        uni = _popcnt(A | S)
                        sims.append((inter / uni) if uni else 0.0)
                    else:
                        sims.append(0.0)
                order = sorted(range(len(sims)), key=lambda i: (-sims[i], -i))
                thr = sims[order[min(int(QTHRESH * len(order)), len(order) - 1)]]
                kept = 0
                sw = 0.0
                acc_s = [0.0] * N
                for i in order:
                    if kept >= KTOP:
                        break
                    sim = sims[i]
                    if sim < thr or sim <= 0.0:
                        break
                    w = sim ** GAMMA
                    T = masks[i + s]
                    sw += w
                    kept += 1
                    mm = T
                    while mm:
                        low = mm & -mm
                        acc_s[low.bit_length() - 1] += w
                        mm ^= low
                if sw > 0.0:
                    alpha = SCALE_W[si]
                    wsum_tot += alpha
                    inv = alpha / sw
                    for j in range(N):
                        acc[j] += inv * acc_s[j]
            if wsum_tot > 0.0:
                analog = [a / wsum_tot for a in acc]

        zA = _zscore(analog, N) if analog is not None else None

        if zA is not None:
            parts = [(W_ANALOG, zA), (W_LONG, zL), (W_RECENT, zR)]
        elif zR is not None:
            ws = W_LONG + W_RECENT
            parts = [(W_LONG / ws, zL), (W_RECENT / ws, zR)]
        else:
            parts = [(1.0, zL)]

        score = [0.0] * N
        for w, z in parts:
            if z is None:
                continue
            for j in range(N):
                score[j] += w * z[j]

        if zA is not None and CONSENSUS > 0.0:
            anchor = [0.0] * N
            if zL is not None:
                for j in range(N):
                    anchor[j] += W_LONG * zL[j]
            if zR is not None:
                for j in range(N):
                    anchor[j] += W_RECENT * zR[j]
            topA = set(sorted(numbers, key=lambda n: (-zA[n - 1], n))[:TOP_C])
            topH = set(sorted(numbers, key=lambda n: (-anchor[n - 1], n))[:TOP_C])
            for n in topA & topH:
                score[n - 1] += CONSENSUS

        return sorted(numbers, key=lambda n: (-score[n - 1], n))

    return rank