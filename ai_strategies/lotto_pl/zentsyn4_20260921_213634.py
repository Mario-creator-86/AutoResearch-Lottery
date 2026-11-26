import math

NAME = "zentsyn4"
DESCRIPTION = (
    "Model condițional pe graf de co-ocurență: sinergie ponderată recențial (convoluție "
    "pe graf), corectată cu z-score global/recent, hazard logaritmic de întârziere și "
    "presiune pe sectoare; ponderarea semnalelor recente este modulată de entropia ferestrei."
)
SOURCE = "glm-5.3-flash:cloud + 2026-09-21"

_W_GLOB = 0.55
_W_REC = 1.00
_W_HZ = 0.70
_W_SYN = 0.45
_W_SECT = 0.50
_TAU = 12.0
_D_SYN = 12
_ENT_LO = 0.60
_ENT_HI = 1.35
_SECTORS = 7
_TOP_PULL = 15


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)

    def _std(vals):
        m = sum(vals) / float(len(vals))
        v = sum((x - m) ** 2 for x in vals) / float(len(vals))
        return m, math.sqrt(v)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        dl = list(recent_deque)
        if not dl:
            base = {n: float(counts.get(n, 0)) for n in numbers}
            return sorted(numbers, key=lambda n: (-base[n], n))

        # orientare robustă: capătul cu media last_seen mai mare e cel mai recent
        m0 = sum(last_seen.get(x, -1) for x in dl[0]) / float(max(1, len(dl[0])))
        ml = sum(last_seen.get(x, -1) for x in dl[-1]) / float(max(1, len(dl[-1])))
        newest_first = m0 >= ml
        rw = recent_window if isinstance(recent_window, int) and recent_window > 0 else len(dl)
        ordered = (dl if newest_first else dl[::-1])[:rw]
        W = len(ordered)
        q = K / float(N)

        # z global (frecvențe pe tot istoricul)
        cvals = [float(counts.get(n, 0)) for n in numbers]
        mu_g, sd_g = _std(cvals)
        if sd_g <= 1e-12:
            sd_g = 1.0
        z_glob = {n: (float(counts.get(n, 0)) - mu_g) / sd_g for n in numbers}

        # z recent + entropia distribuției recente
        rc = dict.fromkeys(numbers, 0)
        for s in ordered:
            for x in s:
                if 1 <= x <= N:
                    rc[x] += 1
        mu_r = W * q
        sd_r = math.sqrt(max(W * q * (1.0 - q), 1e-9))
        z_rec = {n: (rc[n] - mu_r) / sd_r for n in numbers}
        tot = float(W * K)
        ent = 0.0
        if tot > 0:
            for n in numbers:
                p = rc[n] / tot
                if p > 0.0:
                    ent -= p * math.log(p)
        conc = 1.0 - ent / math.log(N)
        if conc < 0.0:
            conc = 0.0
        elif conc > 1.0:
            conc = 1.0
        mult = _ENT_LO + (_ENT_HI - _ENT_LO) * conc

        # graf de perechi pe ultimele D extrageri + pondere recențială per număr
        D = min(W, _D_SYN)
        sub = ordered[:D]
        pc = {}
        for s in sub:
            lst = sorted(x for x in s if 1 <= x <= N)
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    key = (lst[i], lst[j])
                    pc[key] = pc.get(key, 0) + 1
        wapp = dict.fromkeys(numbers, 0.0)
        for i, s in enumerate(sub):
            wi = math.exp(-float(i) / _TAU)
            for x in s:
                if 1 <= x <= N:
                    wapp[x] += wi
        syn_raw = dict.fromkeys(numbers, 0.0)
        for (a, b), c in pc.items():
            wa = wapp[a]
            wb = wapp[b]
            if wb > 0.0:
                syn_raw[a] += c * wb
            if wa > 0.0:
                syn_raw[b] += c * wa
        _, sd_syn = _std(list(syn_raw.values()))
        if sd_syn > 1e-12:
            z_syn = {n: syn_raw[n] / sd_syn for n in numbers}
        else:
            z_syn = dict.fromkeys(numbers, 0.0)

        # hazard logaritmic de întârziere (concav)
        mg = N / float(K)
        hz_raw = {}
        for n in numbers:
            g = draw_index - last_seen.get(n, -1)
            ex = g - mg
            if ex < 0.0:
                ex = 0.0
            hz_raw[n] = math.log1p(ex / mg)
        _, sd_h = _std(list(hz_raw.values()))
        if sd_h > 1e-12:
            z_hz = {n: hz_raw[n] / sd_h for n in numbers}
        else:
            z_hz = dict.fromkeys(numbers, 0.0)

        # presiune pe sectoare: deficit recent standardizat
        S = _SECTORS
        ssz = [0] * S
        ssc = [0] * S
        for n in numbers:
            ssz[min(S - 1, (n - 1) * S // N)] += 1
        for n, c in rc.items():
            ssc[min(S - 1, (n - 1) * S // N)] += c
        corr = (N - K) / float(N - 1) if N > 1 else 1.0
        press = [0.0] * S
        for si in range(S):
            share = ssz[si] / float(N)
            exp_s = W * K * share
            sd_s = math.sqrt(max(W * K * share * (1.0 - share) * corr, 1e-9))
            press[si] = (exp_s - ssc[si]) / sd_s
        z_sect = {n: press[min(S - 1, (n - 1) * S // N)] for n in numbers}

        score = {
            n: (
                _W_GLOB * z_glob[n]
                + mult * _W_REC * z_rec[n]
                + _W_HZ * z_hz[n]
                + mult * _W_SYN * z_syn[n]
                + _W_SECT * z_sect[n]
            )
            for n in numbers
        }
        order = sorted(numbers, key=lambda n: (-score[n], n))

        try:
            top = order[:_TOP_PULL]
            rank._diag = {
                'contrib': {
                    'glob': _W_GLOB * sum(z_glob[n] for n in top) / float(_TOP_PULL),
                    'rec': mult * _W_REC * sum(z_rec[n] for n in top) / float(_TOP_PULL),
                    'hz': _W_HZ * sum(z_hz[n] for n in top) / float(_TOP_PULL),
                    'syn': mult * _W_SYN * sum(z_syn[n] for n in top) / float(_TOP_PULL),
                    'sect': _W_SECT * sum(z_sect[n] for n in top) / float(_TOP_PULL),
                },
                'conc': conc,
                'mult': mult,
                'newest_first': bool(newest_first),
            }
        except Exception:
            pass

        return order

    return rank