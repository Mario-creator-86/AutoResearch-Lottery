import math

NAME = "zzdigitr"
DESCRIPTION = 'Reversie coarse pe cifre finale: z-score de deficit per cifra finala 0-9 cu ponderi exponentiale (half-life 40) pe fereastra recenta, plus mic momentum all-time si penalizare usoara de aglomerare care imparte topul intre mai multe cifre.'
SOURCE = 'model + 2026-09-16'


def create(config):
    nr = config.number_range
    nd = config.numbers_drawn
    hl = float(getattr(config, 'halflife', 40.0))
    w_dig = float(getattr(config, 'w_digit', 1.0))
    w_base = float(getattr(config, 'w_base', 0.25))
    w_crowd = float(getattr(config, 'w_crowd', 0.20))
    dig_dir = float(getattr(config, 'digit_dir', -1.0))
    cap = int(getattr(config, 'window_cap', 160))

    decay = 0.5 ** (1.0 / hl) if hl > 0 else 0.0
    groups = {}
    for n in range(1, nr + 1):
        groups.setdefault(n % 10, []).append(n)
    k_of = {d: len(g) for d, g in groups.items()}
    p = nd / float(nr)
    var_per = {}
    for d, g in groups.items():
        kd = len(g)
        var_per[d] = nd * (kd / float(nr)) * (1.0 - kd / float(nr)) * ((nr - nd) / float(max(nr - 1, 1)))

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, nr + 1))
        all_draws = list(recent_deque) if recent_deque else []
        newest_right = True
        if all_draws and last_seen:
            n_rec = max(last_seen.items(), key=lambda kv: kv[1])[0]
            if last_seen[n_rec] >= draw_index - 1:
                if (n_rec in all_draws[0]) and (n_rec not in all_draws[-1]):
                    newest_right = False
        take = cap
        if recent_window is not None:
            try:
                rw = int(recent_window)
                if rw > 0:
                    take = min(take, rw)
            except (TypeError, ValueError):
                pass
        take = min(take, len(all_draws))
        draws = all_draws[-take:] if newest_right else all_draws[:take]
        w = 1.0
        wsum = 0.0
        w2sum = 0.0
        obs = dict.fromkeys(groups, 0.0)
        seq = reversed(draws) if newest_right else draws
        for draw in seq:
            for x in draw:
                d = x % 10
                if d in obs:
                    obs[d] += w
            wsum += w
            w2sum += w * w
            w *= decay
        zdig = {}
        for d in groups:
            exp = wsum * k_of[d] * p
            var = w2sum * var_per[d]
            sd = math.sqrt(var) if var > 1e-12 else 0.0
            zdig[d] = (obs[d] - exp) / sd if sd > 1e-12 else 0.0
        t = float(draw_index)
        tp = t * p
        sd_all = math.sqrt(max(t * p * (1.0 - p), 1e-12))
        score = {}
        for n in numbers:
            zall = (counts.get(n, 0) - tp) / sd_all if t > 0 else 0.0
            score[n] = dig_dir * w_dig * zdig[n % 10] + w_base * zall
        for d, g in groups.items():
            g_sorted = sorted(g, key=lambda m: (-score[m], m))
            for i, m in enumerate(g_sorted):
                score[m] -= w_crowd * i
        return sorted(numbers, key=lambda m: (-score[m], m))

    return rank