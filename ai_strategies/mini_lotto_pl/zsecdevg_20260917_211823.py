import math

NAME = "zsecdevg"
DESCRIPTION = 'Sector-cycle mean reversion + per-number deviation + capped overdue gap, gated by recent-window entropy and auto-oriented (momentum vs reversion) via persistence z between window halves.'
SOURCE = 'glm-5.3-flash quant plugin, 2026-09-17'

_DEFAULTS = {
    'SEC_SIZE': 6.0, 'SEC_SIGN': -1.0, 'DEV_SIGN': -1.0,
    'W_SEC': 0.7, 'W_GAP': 0.5, 'W_DEV': 0.4,
    'GAP_CAP': 2.5, 'GATE_MIX': 0.5,
    'ORIENT_AUTO': 1.0, 'PERS_DEADBAND': 0.3, 'L': 40.0,
}
_INT_KEYS = ('SEC_SIZE', 'SEC_SIGN', 'DEV_SIGN', 'ORIENT_AUTO', 'L')


def _read_params(config):
    p = dict(_DEFAULTS)
    extra = getattr(config, 'params', None)
    for k in _DEFAULTS:
        v = getattr(config, k, None)
        if v is None and isinstance(extra, dict):
            v = extra.get(k)
        if v is not None:
            try:
                p[k] = int(round(float(v))) if k in _INT_KEYS else float(v)
            except (TypeError, ValueError):
                pass
    return p


def create(config):
    state = {'flips': 0, 'prev': None}
    last_diag = {}

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        params = _read_params(config)
        P = int(config.number_range)
        Wd = int(config.numbers_drawn)
        numbers = list(range(1, P + 1))
        S = max(1, min(int(params['SEC_SIZE']), P))
        Lw = int(params['L'])
        if Lw <= 0:
            Lw = int(recent_window)
        draws = list(recent_deque)
        L_eff = min(Lw, len(draws))
        if L_eff > 0:
            draws = draws[-L_eff:]

        if L_eff < 3 or Wd <= 0 or P <= 1:
            sc = {n: float(counts.get(n, 0)) for n in numbers}
            last_diag.clear()
            last_diag.update({
                'sector_span': 0.0, 'dev_span': 0.0, 'gap_span': 0.0,
                'entropy_norm': 1.0,
                'gate_factor': 1.0 - float(params['GATE_MIX']),
                'orient_flips': float(state['flips']), 'pers_z': 0.0,
                'L': 0.0, 'eff_sec': float(params['SEC_SIGN']),
                'eff_dev': float(params['DEV_SIGN']), 'warmup': 1.0})
            return sorted(numbers, key=lambda n: (-sc[n], n))

        phw = Wd / float(P)
        egap = 1.0 / phw
        nsec = (P + S - 1) // S
        obs_sec = [0.0] * nsec
        cnt = [0] * P
        for dset in draws:
            for n in dset:
                j = n - 1
                if 0 <= j < P:
                    obs_sec[j // S] += 1.0
                    cnt[j] += 1

        corr = (P - Wd) / float(P - 1)
        sec_z = []
        for k in range(nsec):
            size = min(S, P - k * S)
            if size <= 0:
                sec_z.append(0.0)
                continue
            pk = size / float(P)
            e = L_eff * pk * Wd
            var = L_eff * Wd * pk * (1.0 - pk) * corr
            sd = math.sqrt(var) if var > 1e-12 else 1.0
            sec_z.append(max(-3.0, min(3.0, (obs_sec[k] - e) / sd)))

        eN = L_eff * phw
        sdN = math.sqrt(L_eff * phw * (1.0 - phw))
        if sdN <= 1e-12:
            sdN = 1.0
        dev_z = [max(-3.0, min(3.0, (cnt[j] - eN) / sdN)) for j in range(P)]

        cap = max(1.0, float(params['GAP_CAP']))
        gap_raw = [0.0] * P
        for j in range(P):
            ls = last_seen.get(j + 1)
            g = draw_index - ls if ls is not None else draw_index + 1
            if g < 0:
                g = 0
            gap_raw[j] = (min(float(g), egap * cap) - egap) / egap

        total = float(L_eff * Wd)
        Hs = 0.0
        for j in range(P):
            c = cnt[j]
            if c > 0:
                q = c / total
                Hs -= q * math.log(q)
        ent = max(0.0, min(1.0, Hs / math.log(P)))

        half = L_eff // 2
        pers_z = 0.0
        if half >= 2 and (L_eff - half) >= 2:
            A, B = draws[:half], draws[half:]
            la = float(len(A))
            lb = float(len(B))
            ea = la * phw
            eb = lb * phw
            sa = math.sqrt(la * phw * (1.0 - phw)) or 1.0
            sb = math.sqrt(lb * phw * (1.0 - phw)) or 1.0
            cA = [0] * P
            cB = [0] * P
            for dset in A:
                for n in dset:
                    j = n - 1
                    if 0 <= j < P:
                        cA[j] += 1
            for dset in B:
                for n in dset:
                    j = n - 1
                    if 0 <= j < P:
                        cB[j] += 1
            acc = 0.0
            for j in range(P):
                acc += ((cA[j] - ea) / sa) * ((cB[j] - eb) / sb)
            pers_z = acc / float(P)

        dead = float(params['PERS_DEADBAND'])
        if int(params['ORIENT_AUTO']) and abs(pers_z) > dead:
            o = 1.0 if pers_z > 0 else -1.0
            eff_sec = o
            eff_dev = o
        else:
            eff_sec = float(params['SEC_SIGN'])
            eff_dev = float(params['DEV_SIGN'])
        cur = (eff_sec, eff_dev)
        if state['prev'] is not None and cur != state['prev']:
            state['flips'] += 1
        state['prev'] = cur

        gate = 1.0 - float(params['GATE_MIX']) * ent
        wS = float(params['W_SEC'])
        wG = float(params['W_GAP'])
        wD = float(params['W_DEV'])
        sc = {}
        for j in range(P):
            sc[j + 1] = gate * (wS * eff_sec * sec_z[j // S]
                                + wG * gap_raw[j]
                                + wD * eff_dev * dev_z[j])

        last_diag.clear()
        last_diag.update({
            'sector_span': max(sec_z) - min(sec_z),
            'dev_span': max(dev_z) - min(dev_z),
            'gap_span': max(gap_raw) - min(gap_raw),
            'entropy_norm': ent,
            'gate_factor': gate,
            'orient_flips': float(state['flips']),
            'pers_z': pers_z,
            'L': float(L_eff),
            'eff_sec': eff_sec, 'eff_dev': eff_dev, 'warmup': 0.0})

        return sorted(numbers, key=lambda n: (-sc[n], n))

    rank._diag = last_diag
    rank.diag = lambda: dict(last_diag)
    return rank