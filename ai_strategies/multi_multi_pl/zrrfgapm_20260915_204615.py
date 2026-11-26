import math

NAME = "zrrfgapm"
DESCRIPTION = 'Fuziune non-lineara Reciprocal Rank Fusion (RRF) a 4 semnale: propensity EWMA cu shrinkage Bayesian spre rata globala 0.25, hazard de gap adaptiv per numar, afinitate de co-occuranta cu ancorele din ultimele K extrageri, si deviatia z a frecventei pe istoricul complet; reordonare finala greedy cu penalizare de diversitate pe 10 sectoare.'
SOURCE = 'model + 2026-09-15'

PARAMS = {
    'W_HOT': 1.0,
    'W_GAP': 1.0,
    'W_PAIR': 1.0,
    'W_FREQ': 0.35,
    'W_DIV': 0.05,
    'GAMMA': 0.96,
    'ANCHOR_K': 3,
    'RRF_K': 60.0,
    'SHRINK': 8.0,
    'ORIENT_EPS': 1e-9,
}

DIAG = {
    'orient_flips': 0,
    'calls': 0,
    'newest_mean': -1.0,
    'oldest_mean': -1.0,
    'window': 0,
}


def create(config):
    total = config.number_range
    drawn = config.numbers_drawn
    ticket = config.ticket_size
    p0 = drawn / float(total) if total > 0 else 0.25
    sector_size = max(1, total // 10)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        P = PARAMS
        DIAG['calls'] += 1
        numbers = list(range(1, total + 1))
        window = list(recent_deque)
        L = len(window)

        # poarta de orientare newest_last: deque[-1] = cea mai noua extragere,
        # membrii ei au last_seen == draw_index - 1 (maxim), deci media lor
        # trebuie sa depaseasca media capetului vechi; altfel numaram un flip.
        if L >= 2:
            newest = window[-1]
            oldest = window[0]
            if newest and oldest:
                nm = 0.0
                for x in newest:
                    nm += last_seen.get(x, -1)
                om = 0.0
                for x in oldest:
                    om += last_seen.get(x, -1)
                nm /= len(newest)
                om /= len(oldest)
                DIAG['newest_mean'] = nm
                DIAG['oldest_mean'] = om
                DIAG['window'] = L
                if nm < om - P.get('ORIENT_EPS', 1e-9):
                    DIAG['orient_flips'] += 1

        # semnal 1: HOT = propensity EWMA cu shrinkage spre p0, in log-odds
        gamma = P['GAMMA']
        hot = dict.fromkeys(numbers, 0.0)
        wtotal = 0.0
        w = 1.0
        for s in reversed(window):
            for x in s:
                hot[x] += w
            wtotal += w
            w *= gamma
        alpha = P['SHRINK']
        sig_hot = {}
        for x in numbers:
            q = (hot[x] + alpha * p0) / (wtotal + alpha) if wtotal > 0 else p0
            q = min(max(q, 1e-4), 1.0 - 1e-4)
            sig_hot[x] = math.log(q / (1.0 - q))

        # semnal 2: GAP = (gap curent - gap tipic per numar) / (tipic + 1)
        expected_gap = 1.0 / p0 if p0 > 0 else 4.0
        pos = {x: [] for x in numbers}
        for j in range(L):
            d = draw_index - L + j
            for x in window[j]:
                pos[x].append(d)
        sig_gap = {}
        for x in numbers:
            ls = last_seen.get(x)
            g = (draw_index - ls) if ls is not None else (draw_index + 1)
            gp = pos[x]
            if len(gp) >= 3:
                mg = 0.0
                for k in range(1, len(gp)):
                    mg += gp[k] - gp[k - 1]
                mg /= (len(gp) - 1)
                mg = max(mg, 0.5)
            else:
                mg = expected_gap
            sig_gap[x] = (g - mg) / (mg + 1.0)

        # semnal 3: PAIR = afinitate de co-occuranta cu ancorele din ultimele K extrageri
        K = max(1, int(P.get('ANCHOR_K', 3)))
        anchors = set()
        for s in window[-K:]:
            anchors.update(s)
        aff = dict.fromkeys(numbers, 0)
        if anchors:
            for s in window:
                inter = anchors.intersection(s)
                ai = len(inter)
                if ai:
                    for x in s:
                        aff[x] += ai - (1 if x in inter else 0)
        sig_pair = {x: float(aff[x]) for x in numbers}

        # semnal 4: FREQ = z binomial pe istoricul complet
        mu = draw_index * p0
        sd = math.sqrt(max(draw_index * p0 * (1.0 - p0), 1e-12))
        sig_freq = {}
        for x in numbers:
            sig_freq[x] = (counts.get(x, 0) - mu) / sd

        # fuziune RRF: pondere / (RRF_K + rank_in_signal), semnale constante sarite
        rrk = P.get('RRF_K', 60.0)
        score = dict.fromkeys(numbers, 0.0)
        tables = (('hot', sig_hot), ('gap', sig_gap), ('pair', sig_pair), ('freq', sig_freq))
        for name, vals in tables:
            wgt = P.get('W_' + name.upper(), 0.0)
            if abs(wgt) < 1e-12:
                continue
            hi = -1e300
            lo = 1e300
            for x in numbers:
                v = vals[x]
                if v > hi:
                    hi = v
                if v < lo:
                    lo = v
            if hi - lo < 1e-12:
                continue
            order = sorted(numbers, key=lambda v: (-vals[v], v))
            for r, x in enumerate(order, 1):
                score[x] += wgt / (rrk + r)

        # pas non-liniar: selectie greedy cu penalizare de diversitate pe sectoare
        wdiv = P.get('W_DIV', 0.0)
        if wdiv > 1e-12:
            sector_of = {}
            sector_members = {}
            for x in numbers:
                sec = (x - 1) // sector_size
                sector_of[x] = sec
                sector_members.setdefault(sec, []).append(x)
            sec_count = {}
            mult = dict.fromkeys(numbers, 1.0)
            remaining = list(numbers)
            picked = []
            while remaining:
                best = None
                best_val = -1e300
                for x in remaining:
                    v = score[x] * mult[x]
                    if v > best_val:
                        best_val = v
                        best = x
                picked.append(best)
                remaining.remove(best)
                sec = sector_of[best]
                c = sec_count.get(sec, 0) + 1
                sec_count[sec] = c
                nm2 = math.exp(-wdiv * c)
                for x in sector_members[sec]:
                    mult[x] = nm2
            return picked
        return sorted(numbers, key=lambda x: (-score[x], x))

    return rank