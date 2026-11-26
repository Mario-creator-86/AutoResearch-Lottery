NAME = "zsecgapd"
DESCRIPTION = 'Sector-cycle + capped-gap + macro z-deviation scorer: splits 1..N into fixed sectors and scores each number by (a) overdue pressure of its sector vs its expected touch cycle, (b) gap since last appearance capped in mean-gap units, (c) z-deviation of its recent-window frequency from expectation; z-normalized components are gated against a global-frequency anchor.'
SOURCE = 'quant model + 2026-09-18'

import math

# Parametri patchabili la nivel de modul (grid-ul de auto-improvement ii poate suprascrie)
SEC_SIZE = 6  # {4, 6, 8}
SEC_SIGN = 1  # +1 = reversion pe sectoare (favorizeaza sectoarele intariate), -1 = momentum
DEV_SIGN = 1  # +1 = favorizeaza numerele recent calde, -1 = pe cele reci
W_SEC = 0.7  # {0.35, 0.7, 1.05}
W_GAP = 0.5  # {0.3, 0.5, 0.8}; 0 = ablatez gap
W_DEV = 0.4  # {0.0, 0.4, 0.8}
GAP_CAP = 2.5  # {1.5, 2.5, 4.0} x gap mediu (P/W)
GATE_MIX = 0.5  # {0.3, 0.5, 0.7}


def create(config):
    # config: LotteryConfig - folosesc doar .number_range, .numbers_drawn, .ticket_size
    P = config.number_range
    Wd = config.numbers_drawn
    mean_gap = P / float(Wd)
    p_single = Wd / float(P)

    def _z(d):
        vals = list(d.values())
        m = sum(vals) / len(vals)
        var = sum((v - m) * (v - m) for v in vals) / len(vals)
        sd = math.sqrt(var)
        if sd < 1e-12:
            return dict.fromkeys(d, 0.0)
        return {k: (v - m) / sd for k, v in d.items()}

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # counts: {numar: aparitii totale}; recent_deque: deque de seturi
        # last_seen: {numar: index ultima aparitie}; draw_index: index extragere; recent_window: marime fereastra
        numbers = list(range(1, P + 1))

        rc = dict.fromkeys(numbers, 0)
        for dset in recent_deque:
            for n in dset:
                if n in rc:
                    rc[n] += 1

        nsec = (P + SEC_SIZE - 1) // SEC_SIZE
        sec_max = [-1] * nsec
        sec_k = [0] * nsec
        for n in numbers:
            s = (n - 1) // SEC_SIZE
            sec_k[s] += 1
            ls = last_seen.get(n, -1)
            if ls > sec_max[s]:
                sec_max[s] = ls
        cyc = []
        for k in sec_k:
            q = 1.0
            for i in range(Wd):
                q *= (P - k - i) / float(P - i)
            pt = 1.0 - q
            cyc.append(1.0 / pt if pt > 1e-9 else 1e9)

        sec_comp = {}
        gap_comp = {}
        for n in numbers:
            s = (n - 1) // SEC_SIZE
            sec_comp[n] = SEC_SIGN * ((draw_index - sec_max[s]) / cyc[s])
            gap_comp[n] = min((draw_index - last_seen.get(n, -1)) / mean_gap, GAP_CAP)

        exp_rc = recent_window * p_single
        if exp_rc > 0.0:
            sd_rc = math.sqrt(exp_rc * (1.0 - p_single))
            if sd_rc < 1e-9:
                sd_rc = 1e-9
            dev_comp = {n: DEV_SIGN * (rc[n] - exp_rc) / sd_rc for n in numbers}
        else:
            dev_comp = dict.fromkeys(numbers, 0.0)

        base_z = _z({n: float(counts.get(n, 0)) for n in numbers})
        sec_z = _z(sec_comp)
        gap_z = _z(gap_comp)
        dev_z = _z(dev_comp)
        mech = {n: W_SEC * sec_z[n] + W_GAP * gap_z[n] + W_DEV * dev_z[n] for n in numbers}
        final = {n: (1.0 - GATE_MIX) * base_z[n] + GATE_MIX * mech[n] for n in numbers}

        def _span(d):
            vs = list(d.values())
            return max(vs) - min(vs)

        rank._diag = {
            'draw_index': draw_index,
            'recent_window': recent_window,
            'sec_size': SEC_SIZE,
            'sec_sign': SEC_SIGN,
            'dev_sign': DEV_SIGN,
            'w_sec': W_SEC,
            'w_gap': W_GAP,
            'w_dev': W_DEV,
            'gap_cap': GAP_CAP,
            'gate_mix': GATE_MIX,
            'span_sec': _span(sec_z),
            'span_gap': _span(gap_z),
            'span_dev': _span(dev_z),
            'span_mech': _span(mech),
        }

        return sorted(numbers, key=lambda n: (-final[n], n))

    return rank