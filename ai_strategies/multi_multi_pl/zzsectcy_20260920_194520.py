NAME = "zzsectcy"
DESCRIPTION = 'Z-score deviation momentum pe doua ferestre + cicluri de intarziere pe sectoare: numarul e scorat cu z-score-uri winsorized ale conteoagelor pe fereastra medie (recent_window) si pe fereastra scurta (30 extrageri), plus ciclul de intarziere al sectorului sau (media intarzierilor capped fata de asteptarea geometrica 1/p). Un termen de interactiune z_short*z_sector confirma momentum-ul doar cand sectorul numarului e overdue. Mentenanta incrementala O(1) amortizat per extragere cu rebuild la desincronizare; determinist, fara random.'
SOURCE = 'model + 2026-09-18'

import math
from collections import deque

ZS_WIN = 30      # lungimea ferestrei scurte (extrageri)
A_MED = 0.9      # greutate momentum fereastra medie
A_SHO = 0.5      # greutate deviatie fereastra scurta
A_CYC = 0.35     # greutate ciclu intarziere sector
A_INT = 0.15     # greutate interactiune z_short * z_sector
CAP_MULT = 10.0  # cap intarziere = CAP_MULT / p extrageri
ZCAP = 3.5       # winsorizare z-score
N_SECT = 8       # numar de sectoare


def create(config):
    N = int(config.number_range)
    D = int(config.numbers_drawn)
    p = (float(D) / N) if (N > 0 and 0 < D < N) else 0.25
    mu_gap = 1.0 / p
    cap_delay = max(5.0, CAP_MULT / p)
    numbers = list(range(1, N + 1))
    sect_of = {}
    sect_size = {}
    for n in numbers:
        s = ((n - 1) * N_SECT) // N
        sect_of[n] = s
        sect_size[s] = sect_size.get(s, 0) + 1
    sd_mean = {}
    for s in sect_size:
        sz = sect_size[s]
        sd_mean[s] = math.sqrt((1.0 - p) / (p * p * sz)) if sz > 0 else 0.0

    st = {
        'ready': False, 'dq': None, 'rw': None, 'rev': False, 'last': None,
        'med': dict.fromkeys(numbers, 0),
        'shc': dict.fromkeys(numbers, 0),
        'mq': deque(), 'sq': deque(), 'checks': 0,
    }

    def _push(s):
        med = st['med']
        st['mq'].append(s)
        for n in s:
            med[n] += 1
        while len(st['mq']) > st['rw']:
            for n in st['mq'].popleft():
                med[n] -= 1
        shc = st['shc']
        st['sq'].append(s)
        for n in s:
            shc[n] += 1
        while len(st['sq']) > ZS_WIN:
            for n in st['sq'].popleft():
                shc[n] -= 1

    def _argmax_last(last_seen):
        best = None
        bv = -1
        for n in numbers:
            v = last_seen.get(n, -1)
            if v > bv:
                bv = v
                best = n
        return best if bv >= 0 else None

    def _rebuild(recent_deque, recent_window, last_seen, draw_index):
        st['med'] = dict.fromkeys(numbers, 0)
        st['shc'] = dict.fromkeys(numbers, 0)
        st['mq'] = deque()
        st['sq'] = deque()
        st['rw'] = recent_window
        st['dq'] = recent_deque
        rev = st['rev']
        m = _argmax_last(last_seen)
        if m is not None and len(recent_deque) > 0:
            if m in recent_deque[-1]:
                rev = False
            elif m in recent_deque[0]:
                rev = True
            else:
                st['checks'] += 1
        seq = reversed(recent_deque) if rev else recent_deque
        for s in seq:
            _push(s)
        st['rev'] = rev
        st['last'] = draw_index
        st['ready'] = True

    def _ingest(recent_deque, d, last_seen):
        L = len(recent_deque)
        if d > L:
            return False
        if st['rev']:
            items = [recent_deque[i] for i in range(d - 1, -1, -1)]
            newest = recent_deque[0]
        else:
            base = L - d
            items = [recent_deque[base + i] for i in range(d)]
            newest = recent_deque[-1]
        if st['checks'] <= 3:
            m = _argmax_last(last_seen)
            if m is not None and m not in newest:
                st['checks'] += 1
                return False
        for s in items:
            _push(s)
        return True

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        if recent_deque is None:
            recent_deque = ()
        d = None
        if st['ready'] and st['last'] is not None:
            d = draw_index - st['last']
        dq_ok = recent_deque is st['dq']
        inc_ok = False
        if dq_ok and st['ready'] and recent_window == st['rw'] and d is not None:
            inc_ok = (0 <= d) and (d <= recent_window + 2) and (d <= len(recent_deque))
        adopt_ok = False
        if (not dq_ok) and st['ready'] and d == 0 and recent_window == st['rw'] and len(st['mq']) > 0:
            if len(recent_deque) == len(st['mq']) and recent_deque[-1] is st['mq'][-1] and recent_deque[0] is st['mq'][0]:
                adopt_ok = True
        if inc_ok:
            if d > 0:
                if not _ingest(recent_deque, d, last_seen):
                    _rebuild(recent_deque, recent_window, last_seen, draw_index)
            st['last'] = draw_index
        elif adopt_ok:
            st['dq'] = recent_deque
        else:
            _rebuild(recent_deque, recent_window, last_seen, draw_index)

        med = st['med']
        shc = st['shc']
        L = len(st['mq'])
        S = len(st['sq'])
        EL = L * p
        ES = S * p
        sdL = math.sqrt(L * p * (1.0 - p)) if L > 0 else 0.0
        sdS = math.sqrt(S * p * (1.0 - p)) if S > 0 else 0.0

        ssum = {}
        scnt = {}
        for n in numbers:
            ls = last_seen.get(n)
            if ls is None or ls > draw_index:
                dly = draw_index + 1.0
            else:
                dly = float(draw_index - ls)
            if dly < 1.0:
                dly = 1.0
            elif dly > cap_delay:
                dly = cap_delay
            s = sect_of[n]
            ssum[s] = ssum.get(s, 0.0) + dly
            scnt[s] = scnt.get(s, 0) + 1

        zcyc = {}
        for s in scnt:
            sz = scnt[s]
            sd = sd_mean.get(s, 0.0)
            if sz > 0 and sd > 0.0:
                z = (ssum[s] / sz - mu_gap) / sd
                if z > ZCAP:
                    z = ZCAP
                elif z < -ZCAP:
                    z = -ZCAP
                zcyc[s] = z
            else:
                zcyc[s] = 0.0

        zm_sum = 0.0
        zs_sum = 0.0
        zc_sum = 0.0
        out = []
        for n in numbers:
            zm = (med[n] - EL) / sdL if sdL > 0.0 else 0.0
            zs = (shc[n] - ES) / sdS if sdS > 0.0 else 0.0
            if zm > ZCAP:
                zm = ZCAP
            elif zm < -ZCAP:
                zm = -ZCAP
            if zs > ZCAP:
                zs = ZCAP
            elif zs < -ZCAP:
                zs = -ZCAP
            zc = zcyc[sect_of[n]]
            sc = A_MED * zm + A_SHO * zs + A_CYC * zc + A_INT * (zs * zc)
            out.append((sc, n))
            zm_sum += zm
            zs_sum += zs
            zc_sum += zc
        out.sort(key=lambda t: (-t[0], t[1]))

        rank._diag = {
            'L': L, 'S': S,
            'z_med': zm_sum / N, 'z_short': zs_sum / N, 'z_cyc': zc_sum / N,
            'span': out[0][0] - out[-1][0],
        }
        c = getattr(rank, '_diag_cum', None)
        if c is None:
            c = {'k': 0, 'z_med': 0.0, 'z_short': 0.0, 'z_cyc': 0.0}
        k = c['k']
        nk = k + 1
        c['z_med'] = (c['z_med'] * k + zm_sum / N) / nk
        c['z_short'] = (c['z_short'] * k + zs_sum / N) / nk
        c['z_cyc'] = (c['z_cyc'] * k + zc_sum / N) / nk
        c['k'] = nk
        rank._diag_cum = c

        return [n for _, n in out]

    return rank