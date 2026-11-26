NAME = "zspectra"
DESCRIPTION = ('Regime-gated spectral co-occurrence strategy: builds the pair co-appearance graph '
               'over the recent window (membership-only aggregates), extracts a regularized '
               'eigenvector centrality via fixed power iterations, and mixes a momentum side '
               '(centrality, recent-hotness tanh, centrality x overdue-phase interaction) against '
               'a mean-reversion side (sector deficit z, per-number overdue gap phase, long-run '
               'cold z). A gate on the normalized sector entropy of the window sets the blend; all '
               'z-terms are clipped to +/-2.5 and the ranking is invariant to deque orientation.')
SOURCE = 'quant plugin, 2026-11-22'

import math
import numpy as np

_W_CEN = 0.60      # greutate centralitate eigenvector (momentum)
_W_RH = 0.30       # recent-hotness tanh(z/2)
_W_INT = 0.40      # interactiune cen_z x gap_sat
_W_SECT = 0.50     # deficit de sector (reversie)
_W_GAP = 0.45      # faza de gap per numar (reversie)
_W_DEV = 0.25      # deviatie contrariana pe termen lung
_POWIT = 4         # iteratii power-iteration
_SECTORS = 6
_GATE_LO = 0.9980  # praguri gate entropie sector
_GATE_HI = 0.9995
_ZCLIP = 2.5


def create(config):
    N = int(config.number_range)
    K = int(config.numbers_drawn)
    S = _SECTORS if N % _SECTORS == 0 else max(2, int(round(N / 7.0)))
    sec_size = -(-N // S)

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))

        # agregate fara ordine: counts pe numar + matrice de perechi
        rc = dict.fromkeys(numbers, 0)
        pair = np.zeros((N, N), dtype=np.float64)
        Wn = 0
        for draw in recent_deque:
            d = set(draw)
            idx = [n - 1 for n in d if 1 <= n <= N]
            if not idx:
                continue
            Wn += 1
            for n in d:
                if 1 <= n <= N:
                    rc[n] += 1
            m = len(idx)
            for i in range(m):
                a = idx[i]
                for j in range(i + 1, m):
                    b = idx[j]
                    pair[a, b] += 1.0
                    pair[b, a] += 1.0

        q = K / float(N)
        total_tokens = 0
        for n in numbers:
            total_tokens += counts.get(n, 0)
        draws_seen = total_tokens / float(K)

        if Wn == 0 or draws_seen <= 0:
            base = {n: counts.get(n, 0) for n in numbers}
            return sorted(numbers, key=lambda x: (-base[x], x))

        # z pe fereastra recenta
        mu_r = Wn * q
        sd_r = math.sqrt(Wn * q * (1.0 - q)) + 1e-12
        rz = {}
        for n in numbers:
            z = (rc[n] - mu_r) / sd_r
            rz[n] = _ZCLIP if z > _ZCLIP else (-_ZCLIP if z < -_ZCLIP else z)

        # centralitate eigenvector regularizata (matrice ne-negativa, start uniform)
        exp_pair = Wn * (K * (K - 1.0)) / (N * (N - 1.0))
        B = pair + (0.25 * exp_pair + 1e-9)
        np.fill_diagonal(B, 0.0)
        v = np.full(N, 1.0 / N)
        for _ in range(_POWIT):
            v = B @ v
            s = float(v.sum())
            if s > 1e-300:
                v = v / s
            else:
                v = np.full(N, 1.0 / N)
        cm = float(v.mean())
        cs = float(v.std())
        if cs < 1e-6:
            cen_z = dict.fromkeys(numbers, 0.0)
        else:
            cen_z = {}
            k2 = 0
            for n in numbers:
                z = (float(v[k2]) - cm) / cs
                cen_z[n] = _ZCLIP if z > _ZCLIP else (-_ZCLIP if z < -_ZCLIP else z)
                k2 += 1

        # faza de intarziere, saturata
        egap = max(N / float(K) - 1.0, 1e-9)
        gap_sat = {}
        for n in numbers:
            ls = last_seen.get(n)
            g = draw_index - ls if ls is not None else draw_index + 1
            if g < 0:
                g = 0
            phi = g / egap
            gap_sat[n] = phi / (1.0 + phi)

        # deficit de sector + entropie gate
        exp_s = Wn * K / float(S)
        sd_s = math.sqrt(Wn * (1.0 / S) * (1.0 - 1.0 / S)) + 1e-12
        obs_s = [0] * S
        for n in numbers:
            obs_s[(n - 1) // sec_size] += rc[n]
        tot_s = float(sum(obs_s)) + 1e-12
        p_s = [0.0] * S
        sz = [0.0] * S
        for s2 in range(S):
            p_s[s2] = obs_s[s2] / tot_s
            z = (obs_s[s2] - exp_s) / sd_s
            sz[s2] = _ZCLIP if z > _ZCLIP else (-_ZCLIP if z < -_ZCLIP else z)
        Hs = 0.0
        for s2 in range(S):
            if p_s[s2] > 1e-12:
                Hs -= p_s[s2] * math.log(p_s[s2])
        Hs /= math.log(S)
        G = (Hs - _GATE_LO) / (_GATE_HI - _GATE_LO)
        if G < 0.0:
            G = 0.0
        elif G > 1.0:
            G = 1.0

        # deviatie contrariana pe tot istoricul (favorizeaza recele)
        mu_l = draws_seen * q
        sd_l = math.sqrt(draws_seen * q * (1.0 - q)) + 1e-12
        dev_cold = {}
        for n in numbers:
            z = (counts.get(n, 0) - mu_l) / sd_l
            if z > _ZCLIP:
                z = _ZCLIP
            elif z < -_ZCLIP:
                z = -_ZCLIP
            dev_cold[n] = -z

        # compunere cu gate de regim
        inv_g = 1.0 - G
        scores = {}
        for n in numbers:
            inter = cen_z[n] * gap_sat[n]
            mom = G * (_W_CEN * cen_z[n] + _W_RH * math.tanh(0.5 * rz[n]) + _W_INT * inter)
            sec = (n - 1) // sec_size
            rev = inv_g * (_W_SECT * (-sz[sec]) + _W_GAP * gap_sat[n] + _W_DEV * dev_cold[n])
            scores[n] = mom + rev

        return sorted(numbers, key=lambda x: (-scores[x], x))

    return rank