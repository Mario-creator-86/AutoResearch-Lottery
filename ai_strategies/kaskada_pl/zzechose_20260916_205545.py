import math

NAME = "zzechose"
DESCRIPTION = 'EWMA a frecventelor recente (half-life reglabil) + eco al ultimei extrageri + deviatie pe axa decadala (n-1)//10 cu comutator momentum/reversie; ponderi W_*/gamma expuse pe config pentru ablatie, fiecare componenta z-scored.'
SOURCE = 'quant-research + 2026-09-16'

# Ponderi implicit: 3 mecanisme active (regula max-3). w_base/w_x/w_gap/w_jitter = 0
# pentru ablatie; se pot suprascrie ca atribute pe config (ex. config.w_x = 0.25).
PARAMS = {
    'gamma_a': 0.50,      # pondere EWMA recenta
    'ewma_hl': 10.0,      # half-life EWMA (sensibilitate: 5/10/20)
    'w_base': 0.0,        # frecventa globala z-scored (ablare: 0/0.25/0.5)
    'w_echo': 0.30,       # apartenenta la ultima extragere (ablare: 0/0.15/0.3)
    'w_x': 0.0,           # afinitate perechi cu ultima extragere (exclus ultima extragere)
    'w_sect': 0.20,       # deviatie pe decada (ablare: 0/0.1/0.2)
    'digit_dir': -1.0,    # +1 momentum / -1 reversie pe axa n//10
    'w_gap': 0.0,         # overdue/last_seen (oprit implicit)
    'w_jitter': 0.0,      # tie-break determinist; >0 singura sursa de orient_flips
    'sect_window': 30,
    'pair_window': 60,
}

# Gate pentru harness: cu w_jitter==0 (implicit) DIAG['orient_flips'] ramane 0.
DIAG = {'orient_flips': 0, 'calls': 0, 'active': None}


def _z(vals):
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / len(vals)
    s = math.sqrt(var)
    if s < 1e-12:
        return [0.0] * len(vals)
    inv = 1.0 / s
    return [(x - m) * inv for x in vals]


def _h(n, d):
    # hash determinist mic in [-0.5, 0.5], doar pentru w_jitter (off implicit)
    x = (n * 2654435761 + d * 40503) & 0xFFFFFFFF
    x = ((x ^ (x >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((x >> 8) & 0xFFFF) / 65535.0 - 0.5


def create(config):
    P = {k: getattr(config, k, v) for k, v in PARAMS.items()}
    nr = getattr(config, 'number_range', 24)
    nd = getattr(config, 'numbers_drawn', 12)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        DIAG['calls'] += 1
        numbers = list(range(1, nr + 1))
        hist = list(recent_deque)   # conventional: hist[-1] = cea mai recenta extragere
        K = len(hist)
        last = hist[-1] if K else set()

        # --- componenta 1: EWMA frecvente recente (recen continua, nu top-N binar)
        cap = 400
        histE = hist[-cap:] if K > cap else hist
        KE = len(histE)
        hl = max(1.0, float(P['ewma_hl']))
        ew = [0.0] * nr
        for idx, s in enumerate(histE):
            w = 0.5 ** ((KE - 1 - idx) / hl)
            for n in s:
                ew[n - 1] += w
        z_ew = _z(ew)

        # --- componenta 2 (optional): frecventa globala z-scored
        if P['w_base']:
            z_b = _z([float(counts.get(n, 0)) for n in numbers])
        else:
            z_b = [0.0] * nr

        # --- componenta 3: eco al ultimei extrageri
        echo = [1.0 if n in last else 0.0 for n in numbers]

        # --- componenta 4 (optional): afinitate perechi cu ultima extragere,
        # calculata pe fereastra de perechi EXCLUSIV ultima extragere (nu dubleaza eco-ul)
        z_x = [0.0] * nr
        if P['w_x']:
            xsc = [0.0] * nr
            Pw = min(K - 1, int(P['pair_window']))
            if Pw > 0 and last:
                for idx in range(K - 1 - Pw, K - 1):
                    s = hist[idx]
                    for a in last:
                        for n in s:
                            if n != a:
                                xsc[n - 1] += 1.0
            z_x = _z(xsc)

        # --- componenta 5: deviatie pe decada (n-1)//10, momentum(+1)/reversie(-1)
        z_s = [0.0] * nr
        if P['w_sect']:
            S = min(K, int(P['sect_window']))
            if S > 0:
                sizes = {}
                for n in numbers:
                    sizes[(n - 1) // 10] = sizes.get((n - 1) // 10, 0) + 1
                obs = dict.fromkeys(sizes, 0)
                for s in hist[K - S:]:
                    for n in s:
                        obs[(n - 1) // 10] += 1
                d = 1.0 if P['digit_dir'] > 0 else -1.0
                for n in numbers:
                    k = (n - 1) // 10
                    exp = S * nd * sizes[k] / float(nr)
                    dev = (obs[k] - exp) / exp if exp > 0 else 0.0
                    z_s[n - 1] = dev * d

        # --- componenta 6 (optional): gap/overdue capped
        z_g = [0.0] * nr
        if P['w_gap']:
            g = []
            for n in numbers:
                ls = last_seen.get(n)
                gg = (draw_index - ls) if ls is not None else (draw_index + 1)
                g.append(min(float(gg), 400.0))
            z_g = _z(g)

        # --- combinare liniara pe componente z-scored
        combd = {}
        for i, n in enumerate(numbers):
            combd[n] = (P['gamma_a'] * z_ew[i] + P['w_base'] * z_b[i]
                        + P['w_echo'] * echo[i] + P['w_x'] * z_x[i]
                        + P['w_sect'] * z_s[i] + P['w_gap'] * z_g[i])

        DIAG['active'] = {k: P[k] for k in ('gamma_a', 'w_base', 'w_echo', 'w_x', 'w_sect', 'w_gap')}
        order = sorted(numbers, key=lambda n: (-combd[n], n))

        # --- jitter determinist (off implicit => orient_flips == 0, gate respectat)
        if P['w_jitter']:
            jcomb = {n: combd[n] + P['w_jitter'] * _h(n, draw_index) for n in numbers}
            jorder = sorted(numbers, key=lambda n: (-jcomb[n], n))
            DIAG['orient_flips'] += sum(1 for a, b in zip(order, jorder) if a != b)
            order = jorder
        return order

    return rank