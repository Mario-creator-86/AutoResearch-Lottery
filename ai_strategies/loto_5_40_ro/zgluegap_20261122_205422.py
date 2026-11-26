import math

NAME = "zgluegap"
DESCRIPTION = (
    'Pair-glue gate-uit de entropie + hazard de intarziere: scor per numar = z-marginal '
    'global (deviatie fata de frecventa asteptata d*p), + saturarea geometrica a gap-ului '
    '1-exp(-g/E[g]), + vecinatatea de co-aparitie din fereastra recenta (parteneri ponderati '
    'dupa momentum-z-ul lor; agregare exact-integer => invarianta la ordinea/orientarea '
    'deque-ului), gate-uita de entropia normalizata a ferestrei, + interactiune ne-liniara '
    'momentum x centralitate. Determinist; tie-break crescator pe numar. Exposeaza '
    'rank.last_diag cu cen_z, rz, gz, gap_sat, hw, h_norm, gate, mean_rank_shift_top, '
    'frac_top_diff_z pentru sweep si calibrare.'
)
SOURCE = 'glm-5.3-flash + 2026-11-22'

# Greutati calibrabile; grila prescrisa:
#   W_PAIR in {0, 0.15, 0.3, 0.6} | W_GAP in {0.2, 0.5, 0.9} | ENT_FLOOR in {0.0, 0.35, 0.7}
# Ablatia W_PAIR=0 decupleaza glue-ul de marginals;
# |Δavg_hits|<0.005 => KAPPA_W -> inf (interactiunea se stinge).
W_PAIR = 0.30
W_GAP = 0.50
ENT_FLOOR = 0.35
KAPPA_W = 6.0
_GATE_LO = 0.80
_GATE_HI = 0.92
_DIAG_TOP = 12


def create(config):
    NR = int(config.number_range)
    ND = int(config.numbers_drawn)
    PD = ND / float(NR)
    EG = 1.0 / PD
    LN_NR = math.log(NR)
    INF = float('inf')
    _CACHE = {}

    def _agg(recent_deque):
        fr = [0] * (NR + 1)
        for s in recent_deque:
            for x in s:
                if 1 <= x <= NR:
                    fr[x] += 1
        Wd = len(recent_deque)
        mu_w = Wd * PD
        var_w = mu_w * (1.0 - PD)
        sd_w = math.sqrt(var_w) if var_w > 0.0 else 1e-09
        # greutati integer exacte: FB[m] = NR*freq_r[m] - Wd*ND  (proportional cu rz[m])
        FB = [NR * fr[n] - Wd * ND for n in range(NR + 1)]
        acc = [0] * (NR + 1)
        for s in recent_deque:
            Ws = 0
            for m in s:
                if 1 <= m <= NR:
                    Ws += FB[m]
            for x in s:
                if 1 <= x <= NR:
                    acc[x] += Ws
        tot = 0
        for n in range(1, NR + 1):
            tot += fr[n]
        Hn = 0.0
        if tot > 0:
            H = 0.0
            for n in range(1, NR + 1):
                q = fr[n] / tot
                if q > 0.0:
                    H -= q * math.log(q)
            Hn = H / LN_NR
        # glue in spatiul wz: (acc[n] - freq_r[n]*FB[n]) / (NR*sd_w)
        den = NR * sd_w
        wz = [0.0] * (NR + 1)
        gl = [0.0] * (NR + 1)
        for n in range(1, NR + 1):
            wz[n] = (fr[n] - mu_w) / sd_w
            gl[n] = (acc[n] - fr[n] * FB[n]) / den
        gm = 0.0
        gv = 0.0
        for n in range(1, NR + 1):
            gm += gl[n]
        gm /= NR
        for n in range(1, NR + 1):
            df = gl[n] - gm
            gv += df * df
        gv /= NR
        gsd = math.sqrt(gv) if gv > 0.0 else 1e-09
        gz = [(gl[n] - gm) / gsd for n in range(NR + 1)]
        return (wz, gz, Hn, tot)

    def _ramp(x, a, b):
        if b <= a:
            return 1.0 if x >= b else 0.0
        if x <= a:
            return 0.0
        if x >= b:
            return 1.0
        return (x - a) / (b - a)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # cheie FARA recent_window; suma continutului (intregi, invarianta la ordine) aparare contra stale-cache
        key = (draw_index, len(recent_deque), sum(map(sum, recent_deque)))
        ent = _CACHE.get(key)
        if ent is None:
            ent = _agg(recent_deque)
            if len(_CACHE) > 30000:
                _CACHE.clear()
            _CACHE[key] = ent
        wz, gz, Hn, tot_w = ent

        gate = min(_ramp(Hn, ENT_FLOOR, 1.0), _ramp(Hn, _GATE_LO, _GATE_HI))
        wp = W_PAIR * gate
        kv = KAPPA_W
        use_inter = isinstance(kv, (int, float)) and 0 < kv < INF

        d = draw_index if draw_index > 0 else 1
        mu = d * PD
        sdz = math.sqrt(mu * (1.0 - PD))
        if sdz <= 0.0:
            sdz = 1e-09

        z_d = {}
        gs_d = {}
        it_d = {}
        sc = []
        for n in range(1, NR + 1):
            z = (counts.get(n, 0) - mu) / sdz
            lz = last_seen.get(n)
            g = draw_index - lz if lz is not None else draw_index + 1
            if g < 0:
                g = 0
            gsv = 1.0 - math.exp(-g / EG)
            it = (wz[n] * z) / kv if use_inter else 0.0
            s = z + W_GAP * gsv + wp * gz[n] + it
            sc.append((-s, n))
            z_d[n] = z
            gs_d[n] = gsv
            it_d[n] = it
        sc.sort()
        order = [n for _, n in sc]

        z_order = sorted(z_d, key=lambda n: (-z_d[n], n))
        zpos = {}
        for i, n in enumerate(z_order):
            zpos[n] = i
        z_top = set(z_order[:_DIAG_TOP])
        fdiff = 0
        sh = 0
        for i, n in enumerate(order[:_DIAG_TOP]):
            if n not in z_top:
                fdiff += 1
            sh += abs(i - zpos[n])
        rank.last_diag = {
            'cen_z': z_d,
            'rz': {n: wz[n] for n in range(1, NR + 1)},
            'gz': {n: gz[n] for n in range(1, NR + 1)},
            'gap_sat': gs_d,
            'hw': it_d,
            'h_norm': Hn,
            'gate': gate,
            'window_total_hits': tot_w,
            'mu': mu,
            'mean_rank_shift_top': sh / float(_DIAG_TOP),
            'frac_top_diff_z': fdiff / float(_DIAG_TOP)
        }
        return order

    return rank