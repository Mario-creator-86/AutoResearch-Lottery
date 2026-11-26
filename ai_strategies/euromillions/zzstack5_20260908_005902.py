import math

NAME = "zzstack5"
DESCRIPTION = 'Ensambu liniar de 5 semnale z-standardizate: momentum pe fereastra recenta, concentratie Herfindahl a co-aparitiilor pe perechi, deviatie de frecventa pe tot istoricul, hazard de intarziere saturat si deficit de echilibru pe sectoare, plus interactie nestationara tanh(z_recent)*tanh(z_long).'
SOURCE = 'quant-research zstack v1, 2026-09-08'


def _z(vals):
    m = sum(vals) / len(vals)
    var = sum((v - m) * (v - m) for v in vals) / len(vals)
    if var < 1e-12:
        return [0.0] * len(vals)
    sd = math.sqrt(var)
    return [(v - m) / sd for v in vals]


def create(config):
    N = int(getattr(config, 'number_range', 50))
    K = int(getattr(config, 'numbers_drawn', 5))
    nsec = max(2, N // 10)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        T = int(draw_index) if draw_index is not None else 0
        draws = [s for s in recent_deque if s]
        rw = len(draws)

        # 1) deviatia de frecventa pe tot istoricul
        z_long = _z([float(counts.get(n, 0)) for n in numbers])

        # 2) momentum pe fereastra recenta (order-invariant fata de deque)
        r_list = [0.0] * N
        for s in draws:
            for n in s:
                if 1 <= n <= N:
                    r_list[n - 1] += 1.0
        z_rec = _z(r_list) if rw >= 3 else [0.0] * N

        # 3) concentratie de perechi: sum patrat a co-aparitiilor din fereastra
        conc = [0.0] * N
        if rw >= 5:
            pc = {}
            for s in draws:
                lst = sorted(x for x in s if 1 <= x <= N)
                if len(lst) < 2:
                    continue
                for i in range(len(lst) - 1):
                    base = lst[i] * 100
                    for j in range(i + 1, len(lst)):
                        key = base + lst[j]
                        pc[key] = pc.get(key, 0) + 1
            for key, cnt in pc.items():
                c2 = float(cnt) * float(cnt)
                conc[key // 100] += c2
                conc[key % 100] += c2
        z_pair = _z(conc)

        # 4) hazard de intarziere: log1p(gap/gmed), saturat la 2.2
        gmed = float(N) / float(K)
        due_raw = []
        for n in numbers:
            ls = last_seen.get(n)
            g = T - ls if ls is not None else T + 1
            if g < 0:
                g = 0
            due_raw.append(min(math.log1p(g / gmed), 2.2))
        z_due = _z(due_raw)

        # 5) deficit de echilibru pe sectoare de cate ~10 numere
        sec_raw = [0.0] * N
        if rw >= 5:
            obs = [0.0] * nsec
            for s in draws:
                for n in s:
                    if 1 <= n <= N:
                        obs[(n - 1) * nsec // N] += 1.0
            e_sec = rw * K / float(nsec)
            den = math.sqrt(e_sec) if e_sec > 1e-9 else 1.0
            deficit = [max(-2.0, min((e_sec - o) / den, 2.0)) for o in obs]
            sec_raw = [deficit[(n - 1) * nsec // N] for n in numbers]
        z_sec = _z(sec_raw)

        # combinare: ponderi 1.5 / 0.9 / 0.7 / 0.45 / 0.5 + interactie 0.6
        ranked = []
        for i, n in enumerate(numbers):
            inter = math.tanh(z_rec[i]) * math.tanh(z_long[i])
            score = (1.5 * z_rec[i] + 0.9 * z_pair[i] + 0.7 * z_long[i]
                     + 0.45 * z_due[i] + 0.5 * z_sec[i] + 0.6 * inter)
            ranked.append((score, n))
        ranked.sort(key=lambda t: (-t[0], t[1]))
        return [n for _, n in ranked]

    return rank