import math
from collections import Counter

NAME = "zzsecdev"
DESCRIPTION = ('Sector-cycle deficit + short-lag echo + macro deviation: boosts numbers in '
               'sectors whose hit-count over a short window is below the z-normalized '
               'expectation, numbers drawn at exact lags K,2K,3K (comb echo, geometric '
               'decay), and numbers whose recent-window frequency runs above their own '
               'long-run rate. All weights exposed for ablation.')
SOURCE = 'model + 2026-09-16'

CFG = {
    'W_SEC': 0.7,        # pondere deficit sector (z)
    'SECTOR_SIZE': 7,    # numere pe sector (49 -> 7x7)
    'SEC_WINDOW': 12,    # extrageri pentru deficitul de sector
    'W_LAG': 0.9,        # pondere ecou la lag LAG_K
    'LAG_K': 3,          # lag-ul de bază al pieptenelui
    'LAG_DEEP': 0.45,    # decay geometric per dinte suplimentar
    'ECHO_LAGS': 3,      # nr. de dinți (lag-uri K, 2K, 3K)
    'W_DEV': 0.4,        # pondere deviație față de rata proprie
    'DEV_WINDOW': 40,    # extrageri pentru statistica de deviație
}


def create(config):
    N = config.number_range
    K = config.numbers_drawn

    def _p(key):
        v = getattr(config, key, None)
        return CFG[key] if v is None else v

    def _newest_first(draws, last_seen, draw_index):
        # Invariant: toate numerele din cea mai nouă extragere completă au
        # last_seen == draw_index-1 (sau == draw_index dacă extragerea curentă
        # e deja în deque). Detectez orientarea; fallback: append-order.
        draws = list(draws)
        if len(draws) < 2:
            return draws

        def frac(dset):
            if not dset:
                return 0.0
            hits = 0
            for n in dset:
                ls = last_seen.get(n, -10)
                if ls == draw_index - 1 or ls == draw_index:
                    hits += 1
            return hits / float(len(dset))

        if frac(draws[-1]) >= 0.999:
            return draws[::-1]
        if frac(draws[0]) >= 0.999:
            return draws
        return draws[::-1]

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        w_sec = float(_p('W_SEC'))
        sector_size = max(1, int(_p('SECTOR_SIZE')))
        sec_win = int(_p('SEC_WINDOW'))
        w_lag = float(_p('W_LAG'))
        lag_k = max(1, int(_p('LAG_K')))
        lag_deep = float(_p('LAG_DEEP'))
        echo_lags = int(_p('ECHO_LAGS'))
        w_dev = float(_p('W_DEV'))
        dev_win = int(_p('DEV_WINDOW'))

        numbers = list(range(1, N + 1))
        n_sec = (N + sector_size - 1) // sector_size
        order = _newest_first(recent_deque, last_seen, draw_index)
        ndraws = len(order)

        # --- deficit pe sectoare, z-normalizat (pozitiv = sub-reprezentat) ---
        sector_z = [0.0] * n_sec
        sec_len = min(sec_win, ndraws) if sec_win > 0 else 0
        if sec_len > 0:
            sc = [0] * n_sec
            for d in order[:sec_len]:
                for n in d:
                    sc[(n - 1) // sector_size] += 1
            corr = (N - K) / float(N - 1) if N > 1 else 1.0
            for s in range(n_sec):
                lo = s * sector_size + 1
                hi = min((s + 1) * sector_size, N)
                size = hi - lo + 1
                if size <= 0:
                    continue
                ps = size / float(N)
                var = sec_len * K * ps * (1.0 - ps) * corr
                exp = sec_len * K * ps
                if var > 1e-12:
                    sector_z[s] = (exp - sc[s]) / math.sqrt(var)

        # --- ecou comb la lag-uri fixe K, 2K, 3K cu decay geometric ---
        echo = {}
        if lag_k > 0 and echo_lags > 0 and w_lag != 0.0:
            for j in range(1, echo_lags + 1):
                idx = j * lag_k - 1
                if idx < ndraws:
                    wgt = lag_deep ** (j - 1)
                    for n in order[idx]:
                        echo[n] = echo.get(n, 0.0) + wgt

        # --- deviația frecvenței recente față de rata istorică proprie ---
        dev = {}
        dev_len = min(dev_win, ndraws) if dev_win > 0 else 0
        if dev_len > 0 and draw_index > 0:
            wc = Counter()
            for d in order[:dev_len]:
                for n in d:
                    wc[n] += 1
            total = float(draw_index)
            for n in numbers:
                p = counts.get(n, 0) / total
                if 1e-12 < p < 1.0 - 1e-12:
                    e = dev_len * p
                    sd = math.sqrt(dev_len * p * (1.0 - p))
                    dev[n] = (wc.get(n, 0) - e) / sd

        scores = {}
        for n in numbers:
            s = w_sec * sector_z[(n - 1) // sector_size]
            s += w_lag * echo.get(n, 0.0)
            s += w_dev * dev.get(n, 0.0)
            s += 1e-9 * counts.get(n, 0)  # tie-break determinist, neglijabil
            scores[n] = s

        return sorted(numbers, key=lambda m: (-scores[m], m))

    return rank