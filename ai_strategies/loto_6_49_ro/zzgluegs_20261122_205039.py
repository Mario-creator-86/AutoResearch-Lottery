# zglue_gsat_efloor: z-marginal + entropy-gated local (momentum + pair glue) + gap saturation.
# Invarianta de orientare: foloseste doar apartenenta (set-uri), counts, last_seen si perechi
# pe continut (uniune simetrica) => deque inversat produce exact aceeasi permutare.
# Determinism: acumulari pe intregi (exacte), iteratie fixa pe 1..49 pentru sumele float,
# fara RNG, fara termeni de ecou/adjacenta (ECHO_W=0 implicit).
import math
import itertools

NAME = "zzgluegs"
DESCRIPTION = ('Scor hibrid 6/49: z-score marginal (binomial pe counts) + termen local conditionat '
               'de entropia ferestrei recente (momentum z pe frecvente + glue z din perechi '
               'co-aparute, numarate pe continut) + saturatie de intarziere gap/(gap+HALF) din '
               'last_seen. Poarta de entropie (linie de baza Miller-Madow) ridica ponderea '
               'componentei locale doar cand fereastra este supra-concentrata fata de uniform.')
SOURCE = 'glm-quant + 2026-11-23'

# Grid sweep recomandat: W_PAIR{0,0.15,0.3,0.6} x W_GAP{0.2,0.5,0.9} x ENT_FLOOR{0.0,0.35,0.7}
W_PAIR = 0.3      # 0 = ablatie nativa (glue omis complet din calcul)
W_GAP = 0.5
ENT_FLOOR = 0.35  # pondere minima a termenului local
GAP_HALF = 12.0   # jumatate de saturatie pentru gap (desene)
GATE_HALF = 0.015 # jumatate de saturatie pentru poarta de entropie
DIAG_TOPK = 12


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = 0.0
    my = 0.0
    for i in range(n):
        mx += xs[i]
        my += ys[i]
    mx /= n
    my /= n
    sxy = 0.0
    sx = 0.0
    sy = 0.0
    for i in range(n):
        dx = xs[i] - mx
        dy = ys[i] - my
        sxy += dx * dy
        sx += dx * dx
        sy += dy * dy
    if sx <= 0.0 or sy <= 0.0:
        return 0.0
    return sxy / math.sqrt(sx * sy)


def _quantiles(vals, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    if not vals:
        return dict((q, 0.0) for q in qs)
    s = sorted(vals)
    m = len(s)
    out = {}
    for q in qs:
        i = int(q * (m - 1) + 0.5)
        if i < 0:
            i = 0
        elif i > m - 1:
            i = m - 1
        out[q] = s[i]
    return out


def _score_all(counts, recent_deque, last_seen, draw_index, K, D):
    nums = range(1, K + 1)
    lnK = math.log(K) if K > 1 else 1.0
    inv_lnK = 1.0 / lnK
    p = (D / float(K)) if K > 0 else 0.0

    total = 0
    for n in nums:
        total += counts.get(n, 0)
    N = total / float(D) if D > 0 else 0.0

    # 1) z marginal: c_n ~ Binomial(N, p) in aproximare marginala
    if N > 30.0 and 0.0 < p < 1.0:
        mu = N * p
        sg = math.sqrt(N * p * (1.0 - p))
        z_marg = dict((n, (counts.get(n, 0) - mu) / sg) for n in nums)
    else:
        z_marg = dict((n, 0.0) for n in nums)

    L = len(recent_deque)
    freq_w = {}
    pair = {}
    if L > 0:
        use_pairs = (W_PAIR != 0.0) and (D > 1)
        for s in recent_deque:
            mem = tuple(sorted(s))
            for n in mem:
                freq_w[n] = freq_w.get(n, 0) + 1
            if use_pairs:
                for ab in itertools.combinations(mem, 2):
                    pair[ab] = pair.get(ab, 0) + 1

    # masa perechilor per numar: adunari pe intregi, exacte => invarianta la ordine
    glue = {}
    if pair:
        for ab, c in pair.items():
            a = ab[0]
            b = ab[1]
            glue[a] = glue.get(a, 0) + c
            glue[b] = glue.get(b, 0) + c

    # 2) momentum: z pe frecventa din fereastra recenta
    if L > 0 and 0.0 < p < 1.0:
        mu_w = L * p
        sg_w = math.sqrt(L * p * (1.0 - p))
        hw_z = dict((n, (freq_w.get(n, 0) - mu_w) / sg_w) for n in nums)
    else:
        hw_z = dict((n, 0.0) for n in nums)

    # 3) glue: z-score fata de masa de perechi asteptata sub uniform
    p2 = p * ((D - 1) / float(K - 1)) if (K > 1 and D > 1) else 0.0
    if L > 0 and p2 > 0.0 and W_PAIR != 0.0:
        mu_g = (K - 1) * L * p2
        sg_g = math.sqrt(mu_g) if mu_g > 0.0 else 1.0
        glue_z = dict((n, (glue.get(n, 0) - mu_g) / sg_g) for n in nums)
    else:
        glue_z = dict((n, 0.0) for n in nums)

    # 4) poarta de entropie: exces de concentrare fata de asteptarea sub-uniform (Miller-Madow)
    if L > 0 and D > 0 and ENT_FLOOR < 1.0:
        tot_w = float(L * D)
        Hw = 0.0
        for n in nums:
            c = freq_w.get(n, 0)
            if c > 0:
                q = c / tot_w
                Hw -= q * math.log(q)
        hn = Hw * inv_lnK
        hn_ref = (lnK - (K - 1) / (2.0 * tot_w)) * inv_lnK
        if hn_ref < 0.0:
            hn_ref = 0.0
        d = hn_ref - hn
        if d < 0.0:
            d = 0.0
        gate = d / (d + GATE_HALF)
        w_local = ENT_FLOOR + (1.0 - ENT_FLOOR) * gate
    elif ENT_FLOOR >= 1.0:
        hn = 0.0
        hn_ref = 0.0
        gate = 0.0
        w_local = 1.0
    else:
        hn = 0.0
        hn_ref = 0.0
        gate = 0.0
        w_local = 0.0

    # 5) saturatie gap + scor final; never-seen => gap = draw_index+1 (saturatie plafonata)
    gap_sat = {}
    score = {}
    for n in nums:
        ls = last_seen.get(n, -1)
        if ls is None or ls < 0:
            gap = float(draw_index + 1)
        else:
            gap = float(draw_index - ls)
        if gap < 0.0:
            gap = 0.0
        gs = gap / (gap + GAP_HALF)
        gap_sat[n] = gs
        score[n] = z_marg[n] + w_local * (hw_z[n] + W_PAIR * glue_z[n]) + W_GAP * gs
    return {
        'score': score,
        'z_marg': z_marg,
        'glue_z': glue_z,
        'hw_z': hw_z,
        'gap_sat': gap_sat,
        'w_local': w_local,
        'gate': gate,
        'hn': hn,
        'hn_ref': hn_ref,
        'L': L,
    }


def create(config):
    # config: LotteryConfig (.number_range, .numbers_drawn, .ticket_size); nu se itera peste el
    K = config.number_range
    D = config.numbers_drawn
    nums = tuple(range(1, K + 1))

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # recent_window nu intra in nicio cheie de cache: modulul nu tine cache,
        # scorul depinde doar de counts + multisetul set-urilor din deque + last_seen + draw_index.
        st = _score_all(counts, recent_deque, last_seen, draw_index, K, D)
        sc = st['score']
        return sorted(nums, key=lambda n: (-sc[n], n))

    return rank


def diagnose(counts, recent_deque, last_seen, draw_index, recent_window,
             number_range=49, numbers_drawn=6):
    # DIAG: corr(gz,z) si corr(hw,z) pentru calibrare; shift mediu de rang fata de ordinea
    # pur-z; ponderea top-K/top-6 diferita de argmax(z); distributia boost-ului (score-z_marg);
    # quantile glue/gap; modul gate-ului (hn, hn_ref, w_local).
    K = number_range
    D = numbers_drawn
    st = _score_all(counts, recent_deque, last_seen, draw_index, K, D)
    nums = list(range(1, K + 1))
    order_z = sorted(nums, key=lambda n: (-st['z_marg'][n], n))
    order_f = sorted(nums, key=lambda n: (-st['score'][n], n))
    rank_z = {}
    rank_f = {}
    for i, n in enumerate(order_z):
        rank_z[n] = i + 1
    for i, n in enumerate(order_f):
        rank_f[n] = i + 1
    t = DIAG_TOPK if DIAG_TOPK <= K else K
    top_f = order_f[:t]
    top_z = set(order_z[:t])
    shifts = [rank_z[n] - rank_f[n] for n in top_f]
    boost = [st['score'][n] - st['z_marg'][n] for n in nums]
    top_f6 = order_f[:6]
    top_z6 = set(order_z[:6])
    return {
        'L': st['L'],
        'w_local': st['w_local'],
        'gate': st['gate'],
        'hn': st['hn'],
        'hn_ref': st['hn_ref'],
        'corr_glue_z': _pearson([st['glue_z'][n] for n in nums],
                                [st['z_marg'][n] for n in nums]),
        'corr_hw_z': _pearson([st['hw_z'][n] for n in nums],
                              [st['z_marg'][n] for n in nums]),
        'frac_top_diff_z': sum(1 for n in top_f if n not in top_z) / float(t),
        'frac_top6_diff_z': sum(1 for n in top_f6 if n not in top_z6) / 6.0,
        'mean_rank_shift_top': (sum(shifts) / float(t)) if shifts else 0.0,
        'boost_q': _quantiles(boost),
        'glue_q': _quantiles([st['glue_z'][n] for n in nums]),
        'gap_q': _quantiles([st['gap_sat'][n] for n in nums]),
        'top_final': list(top_f),
        'top_z': list(order_z[:t]),
    }