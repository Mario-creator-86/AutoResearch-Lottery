NAME = "zzentros"
DESCRIPTION = 'Regime-switching condționat de entropia ferestrei recente: chi2 Pearson al profilului de aparitii (proxy pentru deficitul de entropie KL(uniform||fereastra)) comuta liniar, prin poarta logistica, intre momentum bland (fereastra uniforma) si recompensare anti-clustering (fereastra concentrata: fade pe supra-reprezentati, boost pe sub-reprezentati). Adauga hazard de gap geometric cu saturatie tanh si un micro-tilt de frecventa globala auto-calibrat. Tr asaturile folosesc doar numaratori de apartenenta, last_seen si draw_index — strategia este invarianta la orientarea deque-ului (flip-test prin constructie).'
SOURCE = 'quant-model + 2026-10-21'

import math


def create(config):
    P = int(config.number_range)
    D = int(config.numbers_drawn)
    if P < 2:
        P = 2
    p = float(D) / float(P)
    if p <= 0.0:
        p = 1e-6
    if p >= 1.0:
        p = 1.0 - 1e-6
    # parametrii procesului de gap sub ipoteza geometrica (numar de esecuri inainte de succes)
    mu_g = (1.0 - p) / p
    sd_g = math.sqrt(1.0 - p) / p

    # ponderi expuse ca override-uri optionale pe config pentru sweep-uri
    W_MOM = float(getattr(config, 'zs_w_mom', 0.70))
    W_REV = float(getattr(config, 'zs_w_rev', 1.10))
    W_DUE = float(getattr(config, 'zs_w_due', 0.35))
    W_ALL = float(getattr(config, 'zs_w_all', 0.12))
    GATE_TH = float(getattr(config, 'zs_gate_th', 0.5))
    GATE_K = float(getattr(config, 'zs_gate_k', 0.9))

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, P + 1))

        # profil de apartenenta in fereastra: w[n] = cate extractii din deque contin n
        # INVARIANT la orientarea deque-ului (doar apartenenta, nu ordinea)
        sets = list(recent_deque) if recent_deque is not None else []
        W_eff = len(sets)
        w = dict.fromkeys(numbers, 0)
        for S in sets:
            for n in S:
                if 1 <= n <= P:
                    w[n] += 1

        slots = W_eff * D
        gate_rev = 0.0
        gate_mom = 1.0
        lam = 0.0
        sd_w = 1.0
        if slots > 0 and W_eff >= 2:
            lam = p * W_eff
            sd_w = math.sqrt(W_eff * p * (1.0 - p))
            # chi2 Pearson pe cele P numere = proxy pentru deficitul de entropie:
            # KL(uniform||q) aprox chi2 / (2*slots); z-score cu df = P-1 calibreaza pragul
            chi2 = 0.0
            for n in numbers:
                d = w[n] - lam
                chi2 += (d * d) / lam
            df = float(P - 1)
            z_chi = (chi2 - df) / math.sqrt(2.0 * df)
            x = GATE_K * (z_chi - GATE_TH)
            if x > 60.0:
                x = 60.0
            elif x < -60.0:
                x = -60.0
            gate_rev = 1.0 / (1.0 + math.exp(-x))
            gate_mom = 1.0 - gate_rev

        # micro-tilt static din counts: z empiric auto-calibrat pe cele 80 de contoare
        vals = [counts.get(n, 0) for n in numbers]
        m_all = sum(vals) / float(P)
        var_all = sum((v - m_all) * (v - m_all) for v in vals) / float(P)
        sd_all = math.sqrt(var_all)

        # coeficientul de regim: semnul tilt-ului pe fereastra e condionat de entropie
        zw_coef = W_MOM * gate_mom - W_REV * gate_rev
        w_due_eff = W_DUE * (0.5 + gate_rev)
        inv_sd_w = 1.0 / sd_w

        scores = {}
        for n in numbers:
            ls = last_seen.get(n)
            g = (draw_index - ls) if ls is not None else (draw_index + 1)
            if g < 0:
                g = 0
            zg = (g - mu_g) / sd_g
            zwa = (w[n] - lam) * inv_sd_w
            za = ((counts.get(n, 0) - m_all) / sd_all) if sd_all > 1e-9 else 0.0
            scores[n] = zw_coef * zwa + w_due_eff * math.tanh(zg / 1.5) + W_ALL * za

        # permutare completa determinista 1..P
        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank