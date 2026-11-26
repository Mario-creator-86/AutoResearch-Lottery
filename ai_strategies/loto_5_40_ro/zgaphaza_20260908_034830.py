import math

NAME = "zgaphaza"
DESCRIPTION = 'Empirical gap-hazard (life-table) ranking: h(g)=P(hit|current gap g) estimated from completed vs censored gaps in the recent window, plus per-number cycle-phase resonance, chi2-gated momentum/reversion and a mild long-run z anchor.'
SOURCE = 'quant research model, 2026-09-08'

def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    BINS = ((1, 1), (2, 2), (3, 3), (4, 4), (5, 6), (7, 9), (10, 14), (15, 24), (25, 10 ** 9))
    NB = len(BINS)

    def bin_of(g):
        for bi in range(NB):
            lo, hi = BINS[bi]
            if lo <= g <= hi:
                return bi
        return NB - 1

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        draws = list(recent_deque)
        if len(draws) >= 2:
            a = max((last_seen.get(n, -1) for n in draws[0]), default=-1)
            b = max((last_seen.get(n, -1) for n in draws[-1]), default=-1)
            if a > b:
                draws.reverse()
        if recent_window and len(draws) > recent_window:
            draws = draws[-recent_window:]
        wlen = len(draws)
        pos = {n: [] for n in numbers}
        wf = dict.fromkeys(numbers, 0)
        for i in range(wlen):
            for n in draws[i]:
                if 1 <= n <= N:
                    pos[n].append(i)
                    wf[n] += 1
        tokens = wlen * K
        if tokens > 0:
            E = tokens / N
            sd = math.sqrt(max(E * (1.0 - K / N), 1e-9))
            chi2n = sum(((wf[n] - E) / sd) ** 2 for n in numbers) / N
        else:
            E = 0.0
            sd = 1.0
            chi2n = 1.0
        conc = max(-1.0, min(1.5, (chi2n - 1.0) / 0.5))
        gate_persist = max(0.0, conc)
        gate_flat = max(0.0, -conc)

        comp = [0] * NB
        open_len = [0] * (wlen + 2)
        for n in numbers:
            p = pos[n]
            if not p:
                for g in range(1, wlen + 1):
                    open_len[g] += 1
                continue
            prev = -1
            for i in p:
                gap = i + 1 if prev < 0 else i - prev
                comp[bin_of(gap)] += 1
                prev = i
            tail = wlen - 1 - p[-1]
            for g in range(1, tail + 1):
                open_len[g] += 1
        comp_ge = [0] * (NB + 1)
        for bi in range(NB - 1, -1, -1):
            comp_ge[bi] = comp_ge[bi + 1] + comp[bi]
        suf = [0] * (wlen + 2)
        for g in range(wlen, 0, -1):
            suf[g] = suf[g + 1] + open_len[g]
        base = (K / N) if K > 0 else 0.0
        haz = [0.0] * NB
        for bi in range(NB):
            lo = BINS[bi][0]
            risk = comp_ge[bi] + (suf[lo] if lo <= wlen else 0)
            haz[bi] = (comp[bi] + 1.0) / (risk + 5.0) - base

        tot = 0
        for n in numbers:
            tot += counts.get(n, 0)
        mu = tot / N
        var = 0.0
        for n in numbers:
            d = counts.get(n, 0) - mu
            var += d * d
        sds = math.sqrt(var / N) or 1.0
        mean_gap = (N / K) if K > 0 else float(N)

        scores = {}
        for n in numbers:
            g = draw_index - last_seen[n] if n in last_seen else draw_index + 1
            if g < 1:
                g = 1
            hz = 4.0 * haz[bin_of(g)]
            p = pos[n]
            bonus = 0.0
            if len(p) >= 4:
                gaps = [p[j] - p[j - 1] for j in range(1, len(p))]
                mg = sum(gaps) / len(gaps)
                if mg > 0.0:
                    vv = sum((x - mg) ** 2 for x in gaps) / len(gaps)
                    reg = 1.0 / (1.0 + math.sqrt(vv) / mg)
                    bonus = 0.35 * reg * math.exp(-0.5 * ((g / mg - 1.0) / 0.6) ** 2)
            mom = 0.45 * gate_persist * math.tanh((wf[n] - E) / (2.0 * sd))
            dz = (g - mean_gap) / math.sqrt(max(mean_gap, 1.0))
            due = 0.35 * gate_flat * math.tanh(dz / 3.0)
            anc = 0.6 * math.tanh((counts.get(n, 0) - mu) / (4.0 * sds))
            scores[n] = hz + bonus + mom + due + anc
        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank