import math
from itertools import islice

NAME = "zgapflow"
DESCRIPTION = 'Flux anti-momentum pe termen scurt: deviatie long-run soft-praguita (|z|<DEV_K ignorat ca zgomot), hazard de gap per-numar cu rata personala (Laplace), lift de co-aparitie conditionat de ultima extragere si rezidualizat fata de caldura ferestrei, fade anti-momentum modulat de entropia ferestrei; cele patru canale z-score combinate liniar.'
SOURCE = 'glm-5.3-flash:cloud 2026-09-21'

PARAMS = {
    'W_DEV': 1.0,
    'DEV_K': 1.0,
    'W_GAP': 0.55,
    'W_PAIR': 0.70,
    'PAIR_WIN': 10,
    'W_FADE': 0.40,
    'FADE_WIN': 10,
}


def create(config):
    N = int(config.number_range)
    D = int(config.numbers_drawn)
    q = D / float(N)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        p = dict(PARAMS)
        over = getattr(config, 'strategy_params', None)
        if isinstance(over, dict):
            for k in p:
                if k in over:
                    p[k] = over[k]

        numbers = list(range(1, N + 1))
        T = float(draw_index) if draw_index and draw_index > 0 else 1.0

        # orientarea deque-ului: inferata din last_seen (setul ultimei extrageri)
        newest_first = True
        target = draw_index - 1
        last_set = set(n for n in numbers if last_seen.get(n, -10 ** 9) == target)
        if last_set and len(recent_deque) > 0:
            try:
                if recent_deque[0] == last_set:
                    newest_first = True
                elif recent_deque[-1] == last_set:
                    newest_first = False
            except (IndexError, TypeError):
                pass

        rw = recent_window if isinstance(recent_window, int) and recent_window > 0 else len(recent_deque)
        if len(recent_deque) > 0:
            src_a = recent_deque if newest_first else reversed(recent_deque)
            src_b = recent_deque if newest_first else reversed(recent_deque)
            pair_sets = list(islice(src_a, min(int(p['PAIR_WIN']), rw)))
            fade_sets = list(islice(src_b, min(int(p['FADE_WIN']), rw)))
        else:
            pair_sets = []
            fade_sets = []
        newest = pair_sets[0] if pair_sets else set()

        # (1) deviatie long-run soft-praguita
        mean_k = T * q
        sd_k = math.sqrt(T * q * (1.0 - q))
        if sd_k <= 0.0:
            sd_k = 1.0
        w_dev = float(p['W_DEV'])
        k_thr = float(p['DEV_K'])

        # (4) fade: counts pe fereastra scurta + entropie
        f_win = {}
        for s in fade_sets:
            for n in s:
                f_win[n] = f_win.get(n, 0) + 1
        Ef = float(len(fade_sets))
        Kf = len(f_win)
        Hn = 1.0
        if Kf > 1:
            tot = float(sum(f_win.values()))
            H = 0.0
            for c in f_win.values():
                pr = c / tot
                H -= pr * math.log(pr)
            Hn = H / math.log(Kf)
        fade_mult = 1.5 - Hn
        if fade_mult < 0.5:
            fade_mult = 0.5
        elif fade_mult > 1.5:
            fade_mult = 1.5
        sd_f = math.sqrt(Ef * q * (1.0 - q))
        if sd_f <= 0.0:
            sd_f = 1.0
        w_fade = float(p['W_FADE']) * fade_mult

        # (3) lift de co-aparitie conditionat de ultima extragere (fara auto-numarare)
        ov_sum = 0.0
        a = {}
        fp = {}
        for idx in range(1, len(pair_sets)):
            s = pair_sets[idx]
            ov = len(newest & s)
            if ov:
                ov_sum += ov
            for n in s:
                fp[n] = fp.get(n, 0) + 1
                if ov:
                    a[n] = a.get(n, 0) + ov
        Ep = float(len(pair_sets) - 1) if len(pair_sets) > 1 else 0.0
        w_pair = float(p['W_PAIR'])

        scores = {}
        cdev = 0.0
        cgap = 0.0
        cpair = 0.0
        cfade = 0.0
        w_gap = float(p['W_GAP'])
        for n in numbers:
            k = float(counts.get(n, 0))
            z = (k - mean_k) / sd_k
            az = abs(z) - k_thr
            if az <= 0.0:
                s1 = 0.0
            elif z > 0.0:
                s1 = az
            else:
                s1 = -az
            v1 = w_dev * s1

            gap = float(draw_index - last_seen.get(n, -1))
            qh = (k + 1.0) / (T + 2.0)
            mg = (1.0 - qh) / qh
            sg = math.sqrt(1.0 - qh) / qh
            zg = (gap - mg) / sg
            if zg > 3.0:
                zg = 3.0
            elif zg < -3.0:
                zg = -3.0
            v2 = w_gap * zg

            if Ep > 0.0 and ov_sum > 0.0 and n in fp:
                exp_n = ov_sum * fp[n] / Ep
                lift = (a[n] - exp_n) / (1.0 + math.sqrt(exp_n))
            else:
                lift = 0.0
            v3 = w_pair * lift

            zf = (f_win.get(n, 0) - Ef * q) / sd_f
            if zf > 2.5:
                zf = 2.5
            elif zf < -2.5:
                zf = -2.5
            v4 = -w_fade * zf

            scores[n] = v1 + v2 + v3 + v4
            cdev += v1
            cgap += v2
            cpair += v3
            cfade += v4

        _diag['newest_first'] = newest_first
        _diag['entropy'] = Hn
        _diag['overlap_sum'] = ov_sum
        _diag['contrib'] = {'dev': cdev / N, 'gap': cgap / N, 'pair': cpair / N, 'fade': cfade / N}

        return sorted(numbers, key=lambda n: (-scores[n], n))

    _diag = {}
    rank._diag = _diag
    return rank