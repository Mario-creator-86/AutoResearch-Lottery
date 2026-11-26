NAME = "zzsteerr"
DESCRIPTION = ('Amestec cantitativ instrumentat: z-score de deviatie fata de frecventa asteptata pe istoricul complet (WL) '
               'combinat cu z-score EWMA pe fereastra recenta (WR, DEC), termen de rezonanta a intarzierii (WG_RES: bonus '
               'cand gap-ul curent al unui numar cade pe 1x/2x din gap-ul mediu de lifetime), tilt de steer al sumei catre '
               'o tinta mean-reversion fata de sumele recente (MU_SUM, SUM_MIX, FLIP) si capuri de paritate pe pick-urile '
               'de top (KPAR). Expune rank._diag si rank._diag_cum cu contributii per-componenta (zL, zS, res, sum, par).')
SOURCE = 'model + 2026-09-20'

import math

# ---- hiperparametri (griddabili; se pot suprascrie la nivel de modul) ----
WL = 0.6         # pondere z long (deviatie lifetime)
WR = 0.5         # pondere z short (EWMA fereastra recenta)
DEC = 0.95       # decay EWMA pe extragere
MU_SUM = 0.02    # pondere steer suma
WG_RES = 0.25    # pondere rezonanta gap
SUM_MIX = 0.7    # mix tinta suma: centru fix vs medie inversata a sumelor recente
KPAR = 1.0       # pondere override cap paritate (0 = off)
FLIP = 1.0       # +1 = reversion (tinta = medie inversata); -1 = contrarian (chase)
RES_SIG = 0.30   # latime lob rezonanta (unitati de gap/mgap)
CAP_ODD = 4      # max numere impare in top-T (cand KPAR>0)
CAP_EVEN = 4     # max numere pare in top-T (cand KPAR>0)
SUM_WINDOW = 30  # extrageri recente pentru suma de referinta
W_EPS = 1e-3     # prag taiere EWMA (coada geometrica adaugata analitic)


def create(config):
    P = config.number_range
    W = config.numbers_drawn
    T = max(1, int(config.ticket_size))
    p = float(W) / float(P)
    center_sum = W * (P + 1) / 2.0
    half_span = (P - 1) / 2.0
    comps = ('zL', 'zS', 'res', 'sum', 'par')

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, P + 1))
        n_obs = (sum(counts.values()) / float(W)) if counts else 0.0
        N = n_obs if n_obs > 0 else float(max(draw_index, 1))

        # z long: deviatia fata de frecventa asteptata pe tot istoricul
        var_l = N * p * (1.0 - p)
        sd_l = math.sqrt(var_l) if var_l > 0 else 1.0
        mu_l = N * p

        # EWMA pe fereastra recenta: cele mai noi au pondere 1, decay DEC inapoi
        acc = dict.fromkeys(numbers, 0.0)
        wsum = 0.0
        w = 1.0
        for s in reversed(recent_deque):
            if w < W_EPS:
                wsum += (w / (1.0 - DEC)) if DEC < 1.0 else w
                break
            for n in s:
                if 1 <= n <= P:
                    acc[n] += w
            wsum += w
            w *= DEC
        if wsum <= 0.0:
            wsum = 1.0
        var_s = wsum * p * (1.0 - p)
        sd_s = math.sqrt(var_s) if var_s > 0 else 1.0
        mu_s = p * wsum

        # tinta sumei: medie inversata a sumelor recente, amestecata cu centrul teoretic
        cnt = 0
        tot = 0
        for s in reversed(recent_deque):
            if cnt >= SUM_WINDOW:
                break
            tot += sum(s)
            cnt += 1
        s_rec = (tot / float(cnt)) if cnt > 0 else center_sum
        t_inv = 2.0 * center_sum - s_rec
        target_sum = center_sum + FLIP * SUM_MIX * (t_inv - center_sum)
        m_star = target_sum / float(W)

        zL_c = {}
        zS_c = {}
        res_c = {}
        sum_c = {}
        par_c = {}
        base = {}
        for n in numbers:
            zl = (counts.get(n, 0) - mu_l) / sd_l
            zs = (acc[n] - mu_s) / sd_s
            c = counts.get(n, 0)
            mgap = (N / c) if c > 0 else N
            g = float(draw_index - last_seen[n]) if n in last_seen else N
            r = (g / mgap) if mgap > 0 else 1.0
            res = (math.exp(-0.5 * ((r - 1.0) / RES_SIG) ** 2)
                   + 0.5 * math.exp(-0.5 * ((r - 2.0) / RES_SIG) ** 2))
            steer = MU_SUM * max(0.0, 1.0 - ((n - m_star) / half_span) ** 2)
            zL_c[n] = WL * zl
            zS_c[n] = WR * zs
            res_c[n] = WG_RES * res
            sum_c[n] = steer
            base[n] = zL_c[n] + zS_c[n] + res_c[n] + steer

        # capuri de paritate: selectie greedy top-T; depasirile de cap sunt penalizate
        capped = set()
        order0 = sorted(numbers, key=lambda n: (-base[n], n))
        if KPAR > 0.0:
            odd = 0
            even = 0
            picked = 0
            for n in order0:
                if picked >= T:
                    break
                if n & 1:
                    if odd >= CAP_ODD:
                        capped.add(n)
                        continue
                    odd += 1
                else:
                    if even >= CAP_EVEN:
                        capped.add(n)
                        continue
                    even += 1
                picked += 1
        pen = 1000.0 * KPAR
        for n in capped:
            par_c[n] = -pen
            base[n] -= pen

        final = sorted(numbers, key=lambda n: (-base[n], n))
        top = final[:T]

        contrib_top = {
            'zL': sum(zL_c[n] for n in top),
            'zS': sum(zS_c[n] for n in top),
            'res': sum(res_c[n] for n in top),
            'sum': sum(sum_c[n] for n in top),
            'par': sum(par_c.values()),
        }
        contrib_mean = {
            'zL': sum(abs(v) for v in zL_c.values()) / float(P),
            'zS': sum(abs(v) for v in zS_c.values()) / float(P),
            'res': sum(abs(v) for v in res_c.values()) / float(P),
            'sum': sum(abs(v) for v in sum_c.values()) / float(P),
            'par': (pen * len(capped)) / float(P),
        }
        rank._diag = {
            'contrib': contrib_top,
            'contrib_mean': contrib_mean,
            'top': list(top),
            'n_capped': len(capped),
            'target_sum': target_sum,
            'recent_sum': s_rec,
            'flip': FLIP,
        }
        cum = getattr(rank, '_diag_cum', None)
        if cum is None:
            cum = {'calls': 0, 'zL': 0.0, 'zS': 0.0, 'res': 0.0, 'sum': 0.0, 'par': 0.0}
        cum['calls'] += 1
        for k in comps:
            cum[k] += contrib_top[k]
        rank._diag_cum = cum

        return final

    return rank