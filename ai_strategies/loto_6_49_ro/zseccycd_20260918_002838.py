import math

NAME = "zseccycd"
DESCRIPTION = 'Ciclu de sectoare cu reversion: imparte 1..49 in sectoare si boostuieste sectoarele sub-reprezentate in fereastra recenta; amesteca (GATE_MIX) deficitul de sector cu un termen de gap overdue normalizat/capped si cu deviatia macro a numarului fata de frecventa asteptata (tanh). Ataseaza rank._diag cu span-urile per componenta pentru ablatare.'
SOURCE = 'glm-5.3-flash:cloud + 2026-09-17'


def create(config):
    # config: LotteryConfig — foloseste doar atributele .number_range / .numbers_drawn / .ticket_size
    N = config.number_range
    K = config.numbers_drawn

    # Hyperparams mecanism (override optional prin config pentru grid; fallback la default)
    SEC_SIZE = max(1, int(getattr(config, 'sec_size', 7)))
    SEC_SIGN = float(getattr(config, 'sec_sign', 1.0))
    DEV_SIGN = float(getattr(config, 'dev_sign', -1.0))
    W_SEC = float(getattr(config, 'w_sec', 0.7))
    W_GAP = float(getattr(config, 'w_gap', 0.5))
    W_DEV = float(getattr(config, 'w_dev', 0.4))
    GAP_CAP = float(getattr(config, 'gap_cap', 2.5))
    GATE_MIX = float(getattr(config, 'gate_mix', 0.5))

    n_sec = (N + SEC_SIZE - 1) // SEC_SIZE
    sec_share = SEC_SIZE / float(N)
    expected_gap = N / float(K)
    p = K / float(N)

    def _sector(x):
        i = (x - 1) // SEC_SIZE
        return i if i < n_sec else n_sec - 1

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # counts: {numar: aparitii totale}; recent_deque: deque de set();
        # last_seen: {numar: index ultima aparitie}; draw_index: indexul curent
        numbers = list(range(1, N + 1))
        total_draws = draw_index + 1
        expected_total = total_draws * p
        std = math.sqrt(total_draws * p * (1.0 - p))
        if std <= 0.0:
            std = 1e-9

        # ciclul de sector pe fereastra recenta
        recent_len = len(recent_deque)
        sec_hits = [0] * n_sec
        for s in recent_deque:
            for x in s:
                sec_hits[_sector(x)] += 1
        exp_sec = recent_len * K * sec_share

        sec_vals = []
        gap_vals = []
        dev_vals = []
        scored = []
        for n in numbers:
            idx = _sector(n)
            deficit = (exp_sec - sec_hits[idx]) / exp_sec if exp_sec > 0.0 else 0.0
            sec_c = GATE_MIX * W_SEC * SEC_SIGN * deficit

            ls = last_seen.get(n)
            gap = draw_index - ls if ls is not None else draw_index + 1
            if gap < 0:
                gap = 0
            gap_term = min(gap / expected_gap, GAP_CAP) / GAP_CAP
            gap_c = (1.0 - GATE_MIX) * W_GAP * gap_term

            dev = (counts.get(n, 0) - expected_total) / std
            dev_c = W_DEV * DEV_SIGN * math.tanh(dev / 3.0)

            scored.append((sec_c + gap_c + dev_c, n))
            sec_vals.append(sec_c)
            gap_vals.append(gap_c)
            dev_vals.append(dev_c)

        scored.sort(key=lambda t: (-t[0], t[1]))
        try:
            rank._diag = {
                'sec': (min(sec_vals), max(sec_vals)),
                'gap': (min(gap_vals), max(gap_vals)),
                'dev': (min(dev_vals), max(dev_vals)),
                'sec_size': SEC_SIZE,
                'exp_sec': exp_sec,
            }
        except Exception:
            pass
        return [n for _, n in scored]

    return rank