import math
import itertools
import numpy as np

NAME = "zzentrog"
DESCRIPTION = ('Regime-switching pe entropie: entropia Shannon a distributiei din fereastra recenta, '
               'comparata cu valoarea asteptata analitic sub uniformitate, comuta semnul momentum-ului '
               '(chase la regim concentrat, fade la regim dispersat). Se combina cu z-scorul de frecventa '
               'pe termen lung, cu hazardul geometric de intarziere si cu un bonus nelinear saturant de convergenta.')
SOURCE = 'quant research + lottery mathematics, 2026-09-08'


def create(config):
    # config nu este iterat niciodata; acces exclusiv prin getattr pe atributele documentate
    K = int(getattr(config, 'number_range', 49) or 49)
    D = int(getattr(config, 'numbers_drawn', 6) or 6)
    _ticket = int(getattr(config, 'ticket_size', D) or D)

    K = max(K, 2)
    D = min(max(D, 1), K)
    p = D / float(K)
    g_mean = 1.0 / p
    g_std = math.sqrt(1.0 - p) / p
    ln_k = math.log(K)

    W_REC = 0.60
    W_OD = 0.15
    W_CF = 0.20
    K_GATE = 8.0
    ZCAP = 4.0

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, K + 1))
        try:
            di = int(draw_index)
        except (TypeError, ValueError):
            di = 0
        n_draws = max(di + 1, 1)
        mu = n_draws * p
        sd_all = max(math.sqrt(max(mu * (1.0 - p), 0.0)), 1.0)

        cvec = np.zeros(K, dtype=np.float64)
        if isinstance(counts, dict):
            for n in numbers:
                cvec[n - 1] = float(counts.get(n, 0))

        dq = []
        if recent_deque is not None:
            try:
                dq = list(recent_deque)
            except TypeError:
                dq = []
        try:
            rw = int(recent_window) if recent_window is not None else 0
        except (TypeError, ValueError):
            rw = 0
        if 0 < rw < len(dq):
            dq = dq[-rw:]
        w = len(dq)

        wcnt = np.zeros(K, dtype=np.float64)
        if w > 0:
            try:
                flat = np.fromiter(itertools.chain.from_iterable(dq), dtype=np.int64)
                flat = flat[(flat >= 1) & (flat <= K)]
                if flat.size > 0:
                    wcnt = np.bincount(flat, minlength=K + 1)[1:K + 1].astype(np.float64)
            except (TypeError, ValueError):
                wcnt = np.zeros(K, dtype=np.float64)

        z_all = np.clip((cvec - mu) / sd_all, -ZCAP, ZCAP)

        w_mom = 0.0
        z_rec = np.zeros(K, dtype=np.float64)
        if w > 0:
            m_tot = float(w * D)
            wp = w * p
            wsd = max(math.sqrt(max(wp * (1.0 - p), 0.0)), 1.0)
            z_rec = np.clip((wcnt - wp) / wsd, -ZCAP, ZCAP)
            nz = wcnt[wcnt > 0]
            h = 0.0
            if nz.size > 0:
                q = nz / m_tot
                h = float(-(q * np.log(q)).sum())
            h_exp = ln_k - (K - 1) / (2.0 * m_tot)
            x = K_GATE * (h - h_exp)
            if x >= 30.0:
                m_gate = 1.0
            elif x <= -30.0:
                m_gate = 0.0
            else:
                m_gate = 1.0 / (1.0 + math.exp(-x))
            w_mom = W_REC * (1.0 - 2.0 * m_gate)

        gvec = np.full(K, float(n_draws), dtype=np.float64)
        if isinstance(last_seen, dict):
            for n, idx in last_seen.items():
                try:
                    ni = int(n)
                except (TypeError, ValueError):
                    continue
                if 1 <= ni <= K and idx is not None:
                    gvec[ni - 1] = max(float(di - int(idx)), 0.0)
        f_od = np.tanh((gvec - g_mean) / (2.0 * g_std))

        cf = np.maximum(z_all, 0.0) * np.maximum(z_rec, 0.0)
        scores = z_all + w_mom * z_rec + W_OD * f_od + W_CF * (cf / (1.0 + cf))

        order = np.argsort(-scores, kind='stable')
        return [int(numbers[i]) for i in order]

    return rank