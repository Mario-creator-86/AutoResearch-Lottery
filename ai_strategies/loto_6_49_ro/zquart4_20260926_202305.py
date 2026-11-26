NAME = "zquart4"
DESCRIPTION = ('Semnal compus z-standardizat din 5 componente invariante la orientarea deque-ului: '
               'afinitate de co-aparitii in fereastra recenta (boost pentru perechi cu numerele ultimei '
               'extrageri, derivata doar din last_seen), exces de intarziere raportat la gap-ul asteptat, '
               'momentum EWMA prin kernel exponential pe gap, deviatia globala de frecventa si deficit pe '
               '7 sectoare; ponderile pair/fade sunt modulate de entropia ferestrei, plus termen neliniar '
               'z_pair*max(z_mom,0) si auto-verificare flip in rank.last_diag.')
SOURCE = 'glm-5.3-flash + 2026-09-26'

import math


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    EG = N / float(K)          # gap mediu asteptat (~8.167)
    TAU = 12.0                 # jumatate-viata momentum (extrageri)
    CAP = 3.0                  # cap exces de intarziere
    W_PAIR, W_DUE, W_FADE, W_GLOB, W_SECT, W_NL = 1.0, 0.6, 0.8, 0.4, 0.4, 0.3
    PAIR_BOOST = 0.5
    NSEC = 7
    _calls = [0]

    def _z(vals):
        m = sum(vals) / float(len(vals))
        var = sum((v - m) * (v - m) for v in vals) / float(len(vals))
        sd = math.sqrt(var)
        if sd < 1e-12:
            return [0.0] * len(vals)
        return [(v - m) / sd for v in vals]

    def _accumulate(seq):
        # adunari exacte de intregi -> rezultat independent de ordinea iterarii
        mem = [0] * N
        pair = [0.0] * (N * N)
        for s in seq:
            if not s:
                continue
            ids = [x - 1 for x in s if 1 <= x <= N]
            for a in ids:
                mem[a] += 1
            L = len(ids)
            for i in range(L):
                ai = ids[i]
                base = ai * N
                for j in range(i + 1, L):
                    b = ids[j]
                    pair[base + b] += 1.0
                    pair[b * N + ai] += 1.0
        return mem, pair

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        _calls[0] += 1
        numbers = list(range(1, N + 1))
        prev_idx = draw_index - 1

        # (4) deviatie globala fata de frecventa medie
        cnt = [float(counts.get(n, 0)) for n in numbers]
        z_glob = _z(cnt)

        # gap-uri: DERIVATE DOAR DIN last_seen -> invariante la flip
        gaps = []
        for n in numbers:
            ls = last_seen.get(n)
            g = float(prev_idx - ls) if ls is not None else (draw_index + EG)
            if g < 0.0:
                g = 0.0
            gaps.append(g)

        # (2) exces de intarziere
        z_due = _z([min(g / EG - 1.0, CAP) for g in gaps])
        # (3) momentum EWMA (kernel jumatate-viata TAU)
        z_mom = _z([0.5 ** (g / TAU) for g in gaps])

        seq = list(recent_deque) if recent_deque else []
        mem, pair = _accumulate(seq)
        total_hits = float(sum(mem))

        # setul ultimei extrageri, derivat exclusiv din last_seen (flip-invariant)
        latest_ids = [n - 1 for n in numbers if last_seen.get(n) == prev_idx]

        # (1) afinitate de perechi in fereastra + boost cu ultimul set
        z_pair = [0.0] * N
        if seq:
            ps = []
            for a in range(N):
                base = a * N
                tot = sum(pair[base:base + N])
                boost = 0.0
                for b in latest_ids:
                    boost += pair[base + b]
                ps.append(tot + PAIR_BOOST * boost)
            z_pair = _z(ps)

        # (5) deficit de aparitii pe sectoare (cicluri de intarziere pe sectoare)
        z_sect = [0.0] * N
        if total_hits > 0.0:
            sect_hit = [0] * NSEC
            sect_size = [0] * NSEC
            for a in range(N):
                si = (a * NSEC) // N
                sect_hit[si] += mem[a]
                sect_size[si] += 1
            raw = []
            for a in range(N):
                si = (a * NSEC) // N
                exp_i = total_hits * (sect_size[si] / float(N))
                d_i = exp_i - sect_hit[si]
                raw.append(d_i / math.sqrt(max(1.0, exp_i * (1.0 - K / float(N)))))
            z_sect = _z(raw)

        # entropia ferestrei -> modulatie de ponderi (structura amplifica pair/fade)
        structure = 0.0
        if total_hits > 0.0:
            H = 0.0
            for m in mem:
                if m > 0:
                    p = m / total_hits
                    H -= p * math.log(p)
            structure = max(0.0, min(1.0, 1.0 - H / math.log(N)))
        mult = 0.5 + structure
        w_pair = W_PAIR * mult
        w_fade = W_FADE * mult

        # termen neliniar: perechi AND momentum pozitiv (evit double-negative)
        z_nl = []
        for i in range(N):
            zm = z_mom[i] if z_mom[i] > 0.0 else 0.0
            z_nl.append(z_pair[i] * zm)

        scores = {}
        for i, n in enumerate(numbers):
            scores[n] = (w_pair * z_pair[i] + W_DUE * z_due[i] + w_fade * z_mom[i]
                         + W_GLOB * z_glob[i] + W_SECT * z_sect[i] + W_NL * z_nl[i])
        order = sorted(numbers, key=lambda n: (-scores[n], n))

        # DIAG: contributii per componenta pe top-6 + consistenta last_seen + flip check
        top6 = order[:K]
        contrib = {}
        for cname, comp in (('pair', z_pair), ('due', z_due), ('fade', z_mom),
                            ('glob', z_glob), ('sect', z_sect), ('nl', z_nl)):
            contrib[cname] = round(sum(comp[n - 1] for n in top6) / float(K), 4)
        diag = {'draw_index': draw_index, 'window_len': len(seq),
                'latest_from_last_seen': len(latest_ids),
                'latest_ok': len(latest_ids) == K,
                'structure': round(structure, 4),
                'w_pair_eff': round(w_pair, 4), 'w_fade_eff': round(w_fade, 4),
                'contrib': contrib, 'top6': top6}
        flip_ok = True
        if len(seq) > 1 and (_calls[0] % 97 == 1):
            mem2, pair2 = _accumulate(reversed(seq))
            d = 0.0
            for i in range(N * N):
                dv = pair[i] - pair2[i]
                if dv < 0.0:
                    dv = -dv
                if dv > d:
                    d = dv
            flip_ok = (mem2 == mem) and (d == 0.0)
            diag['flip_maxdiff'] = d
        diag['flip_ok'] = flip_ok
        rank.last_diag = diag

        return order

    return rank