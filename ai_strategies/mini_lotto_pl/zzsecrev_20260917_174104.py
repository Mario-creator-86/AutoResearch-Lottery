import math

NAME = "zzsecrev"
DESCRIPTION = 'Ensemble contrarian de reversie: cicluri de deficit pe sectoare de 7 numere (saturate tanh), reversie fata de frecventa istorica macro si ancora log-saturata de intarziere pe last_seen. Fara momentum, fara perechi, fara EWMA.'
SOURCE = 'qmodel 2026-09-16'

def create(config):
    # config: LotteryConfig — folosit doar prin atributele number_range / numbers_drawn / ticket_size
    N = config.number_range            # ex. 42
    K = config.numbers_drawn           # ex. 5

    SEC_SIZE = 7                       # 6 sectoare x 7 numere
    W_SEC = 1.05                       # pondere deficit sector (fereastra recenta)
    W_MAC = 0.40                       # pondere reversie macro (istoric total), semn contrarian
    W_GAP = 0.80                       # pondere ancora intarziere
    GAP_EXP = N / float(K)             # gap asteptat ~ N/K extrageri

    nsec = (N + SEC_SIZE - 1) // SEC_SIZE

    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        numbers = list(range(1, N + 1))

        cs = [counts.get(n, 0) for n in numbers]
        mean_c = sum(cs) / float(N)
        var_c = sum((c - mean_c) ** 2 for c in cs) / float(N)
        std_c = math.sqrt(var_c) or 1.0

        scores = {}
        Wd = len(recent_deque)
        if Wd == 0:
            # fallback determinist: doar gap + reversie macro
            for n in numbers:
                zc = (counts.get(n, 0) - mean_c) / std_c
                g = max(0, draw_index - last_seen.get(n, -1))
                scores[n] = W_GAP * math.log1p(g / GAP_EXP) - W_MAC * zc
            return sorted(numbers, key=lambda n: (-scores[n], n))

        exp_n = Wd * K / float(N)          # asteptat per numar in fereastra
        exp_s = exp_n * SEC_SIZE           # asteptat per sector
        sd_s = math.sqrt(exp_s) or 1.0

        freq = dict.fromkeys(numbers, 0)
        for dset in recent_deque:
            for x in dset:
                if 1 <= x <= N:
                    freq[x] += 1

        sec_act = [0.0] * nsec
        for n in numbers:
            sec_act[(n - 1) // SEC_SIZE] += freq[n]

        # deficit de sector saturat neliniar (tanh): sector rece -> scor pozitiv
        sec_score = [math.tanh(0.5 * ((exp_s - a) / sd_s)) for a in sec_act]

        for n in numbers:
            zc = (counts.get(n, 0) - mean_c) / std_c
            g = max(0, draw_index - last_seen.get(n, -1))
            scores[n] = (W_SEC * sec_score[(n - 1) // SEC_SIZE]
                         + W_GAP * math.log1p(g / GAP_EXP)
                         - W_MAC * zc)

        return sorted(numbers, key=lambda n: (-scores[n], n))

    return rank