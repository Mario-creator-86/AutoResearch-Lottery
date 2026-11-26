import math
from collections import defaultdict

NAME = "zewmgape"
DESCRIPTION = 'Scor hibrid determinist: EWMA pe frecventele recente, momentum pe ultimele 5 extrageri, deviatie z fata de frecventa asteptata pe termen lung, cicluri de intarziere pe numar si pe sector, plus bonus pentru perechi co-aparute recent; ponderile momentum/perechi vs intarziere sunt modulate neliniar de entropia ferestrei recente.'
SOURCE = 'cercetare proprie: EWMA + entropie + gap-cycle, 2026-09-08'


def create(config):
    N = int(getattr(config, 'number_range', 50) or 50)
    K = int(getattr(config, 'numbers_drawn', 5) or 5)
    if K < 1:
        K = 1
    if K > N:
        K = N
    eps = 1e-12

    def _clip3(x):
        if x > 3.0:
            return 3.0
        if x < -3.0:
            return -3.0
        return x

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))
        if counts is None:
            counts = {}
        if last_seen is None:
            last_seen = {}
        hist = []
        if recent_deque is not None:
            try:
                hist = list(recent_deque)
            except Exception:
                hist = []
        rw = recent_window
        if not isinstance(rw, int) or rw <= 0:
            rw = len(hist)
        if rw > 0 and len(hist) > rw:
            hist = hist[-rw:]
        L = len(hist)
        di = draw_index
        if not isinstance(di, (int, float)) or isinstance(di, bool):
            di = L
        di = float(di)
        if di < 0.0:
            di = 0.0
        p = K / float(N)

        # 1) EWMA pe fereastra recenta (cel mai nou set = greutate maxima)
        hl = 0.15 * rw if rw > 26 else 12.0
        if hl < 4.0:
            hl = 4.0
        if hl > 24.0:
            hl = 24.0
        lam = 0.5 ** (1.0 / hl)
        ew = defaultdict(float)
        wsum = 0.0
        w2sum = 0.0
        wgt = 1.0
        for i in range(L - 1, -1, -1):
            wsum += wgt
            w2sum += wgt * wgt
            st = hist[i]
            if st:
                for n in st:
                    if 1 <= n <= N:
                        ew[n] += wgt
            wgt *= lam
        mu_e = wsum * p
        sg_e = math.sqrt(max(w2sum * p * (1.0 - p), eps))
        z_ew = {}
        for n in numbers:
            z_ew[n] = _clip3((ew.get(n, 0.0) - mu_e) / sg_e)

        # 2) momentum pe ultimele M extrageri
        M = 5 if L >= 5 else L
        mom = defaultdict(float)
        if M > 0:
            for st in hist[L - M:]:
                for n in st:
                    if 1 <= n <= N:
                        mom[n] += 1.0
        mu_m = M * p
        sg_m = math.sqrt(max(M * p * (1.0 - p), eps))
        z_mom = {}
        for n in numbers:
            z_mom[n] = _clip3((mom.get(n, 0.0) - mu_m) / sg_m)

        # 3) deviatie fata de frecventa asteptata pe termen lung
        tot = 0
        for n in numbers:
            tot += counts.get(n, 0)
        mu_c = tot / float(N)
        sg_c = math.sqrt(max(mu_c * (1.0 - p), eps))
        z_cnt = {}
        for n in numbers:
            z_cnt[n] = _clip3((counts.get(n, 0) - mu_c) / sg_c)

        # 4) cicluri de intarziere pe numar (gap vs gap asteptat N/K)
        eg = N / float(K)
        z_gap = {}
        for n in numbers:
            ls = last_seen.get(n, None)
            if ls is None:
                gap = di + 1.0
            else:
                gap = di - float(ls)
            if gap < 0.0:
                gap = 0.0
            ratio = min(gap, 4.0 * eg) / eg
            z_gap[n] = ratio - 1.0

        # 5) cicluri de intarziere pe sectoare de ~10 numere
        nsec = (N + 9) // 10
        if nsec < 1:
            nsec = 1
        ssize = (N + nsec - 1) // nsec
        sec_cnt = [0] * nsec
        sec_ls = [None] * nsec
        for n in numbers:
            sid = (n - 1) // ssize
            if sid < 0 or sid >= nsec:
                continue
            c = counts.get(n, 0)
            if c:
                sec_cnt[sid] += c
                ls = last_seen.get(n, None)
                if ls is not None:
                    if sec_ls[sid] is None or float(ls) > float(sec_ls[sid]):
                        sec_ls[sid] = ls
        z_sec = {}
        mu_sc = float(ssize) * mu_c
        sg_sc = math.sqrt(max(mu_sc * (1.0 - p), eps))
        for n in numbers:
            sid = (n - 1) // ssize
            if sid < 0 or sid >= nsec:
                sid = 0
            zc = (sec_cnt[sid] - mu_sc) / sg_sc
            ls = sec_ls[sid]
            if ls is None:
                sgap = di + 1.0
            else:
                sgap = di - float(ls)
            if sgap < 0.0:
                sgap = 0.0
            sratio = min(sgap, 4.0 * eg) / eg - 1.0
            z_sec[n] = 0.5 * _clip3(zc) + 0.5 * sratio

        # 6) perechi co-aparute in ultimele P extrageri
        P = 25 if L >= 25 else L
        pw = defaultdict(float)
        if P > 0:
            for st in hist[L - P:]:
                arr = []
                for x in st:
                    if isinstance(x, (int, float)) and not isinstance(x, bool):
                        xi = int(x)
                        if 1 <= xi <= N:
                            arr.append(xi)
                arr.sort()
                m = len(arr)
                for i2 in range(m):
                    a = arr[i2]
                    for j2 in range(i2 + 1, m):
                        b = arr[j2]
                        pw[(a, b)] += 1.0
                        pw[(b, a)] += 1.0
        ps = defaultdict(float)
        for kk, cc in pw.items():
            ps[kk[0]] += cc
            ps[kk[1]] += cc
        mu_ps = 0.0
        for n in numbers:
            mu_ps += ps.get(n, 0.0)
        mu_ps /= float(N)
        var_ps = 0.0
        for n in numbers:
            dd = ps.get(n, 0.0) - mu_ps
            var_ps += dd * dd
        sg_ps = math.sqrt(var_ps / float(N)) if N > 0 else 0.0
        if sg_ps < eps:
            sg_ps = eps
        z_pair = {}
        for n in numbers:
            z_pair[n] = _clip3((ps.get(n, 0.0) - mu_ps) / sg_ps)

        # 7) entropia distributiei recente -> adaptare neliniara a ponderilor
        Hn = 1.0
        if L > 0:
            freq = defaultdict(float)
            totf = 0.0
            for st in hist:
                for n in st:
                    if 1 <= n <= N:
                        freq[n] += 1.0
                        totf += 1.0
            if totf > 0:
                H = 0.0
                for n in numbers:
                    f = freq.get(n, 0.0) / totf
                    if f > 0.0:
                        H -= f * math.log(f)
                lg = math.log(float(N))
                if lg > 0:
                    Hn = H / lg
        conc = 1.0 - Hn
        if conc < 0.0:
            conc = 0.0
        if conc > 1.0:
            conc = 1.0

        w_ew = 0.60
        w_mom = 0.55 + 2.0 * conc
        w_pair = 0.35 + 1.5 * conc
        w_gap = 0.45 - 0.6 * conc
        w_sec = 0.20
        w_cnt = 0.25

        score = {}
        for n in numbers:
            score[n] = (w_ew * z_ew[n] + w_mom * z_mom[n] + w_cnt * z_cnt[n]
                        + w_gap * z_gap[n] + w_sec * z_sec[n] + w_pair * z_pair[n])
        return sorted(numbers, key=lambda nn: (-score[nn], nn))

    return rank