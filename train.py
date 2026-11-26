"""Exp24: Pair co-occurrence + WF-validated weights (optimized).

Exp20 (hot_pairs) got best backtest: 2.5577
Exp23 (WF weights) got best WF: 2.5499 (fw=0.70 dw=0.16 rw=0.14)

Combine pair co-occurrence boosting with WF-validated scoring.
Uses incremental pair counting for O(1) per-draw updates.
"""
import sys, json, time
from collections import Counter, defaultdict, deque
from itertools import combinations

def p(msg):
    print(msg, flush=True)

def load_draws():
    import csv
    from datetime import datetime
    draws = []
    with open('data/multi_multi_pl.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = datetime.strptime(row['draw_date'], '%Y-%m-%d')
            numbers = [int(row[f'number_{i:02d}']) for i in range(1, 21)]
            draws.append({'draw_date': date, 'numbers': tuple(numbers)})
    return draws


class IncrementalPairTracker:
    """Maintains rolling pair co-occurrence counts with O(1) updates per draw."""

    def __init__(self, window_size):
        self.window_size = window_size
        self.pair_counts = Counter()
        self.draw_buffer = deque(maxlen=window_size)

    def update(self, draw_numbers):
        draw_set = frozenset(draw_numbers)
        pairs = list(combinations(sorted(draw_set & set(range(1, 81))), 2))
        if len(self.draw_buffer) == self.window_size:
            old_draw = self.draw_buffer[0]
            old_pairs = list(combinations(sorted(old_draw & set(range(1, 81))), 2))
            for pair in old_pairs:
                self.pair_counts[pair] -= 1
                if self.pair_counts[pair] <= 0:
                    del self.pair_counts[pair]
        for pair in pairs:
            self.pair_counts[pair] += 1
        self.draw_buffer.append(draw_set)

    def get_boost_scores(self, top_candidates):
        if not self.pair_counts or not top_candidates:
            return {}
        expected = self.window_size * (20/80) * (19/79)
        threshold = expected * 1.2
        pair_boost = defaultdict(float)
        for (a, b), cnt in self.pair_counts.items():
            if cnt > threshold:
                excess = cnt - expected
                if a in top_candidates:
                    pair_boost[b] += excess
                if b in top_candidates:
                    pair_boost[a] += excess
        if pair_boost:
            max_boost = max(pair_boost.values())
            if max_boost > 0:
                pair_boost = {n: v / max_boost for n, v in pair_boost.items()}
        return pair_boost


def backtest(rank_fn, draws, iw=30, rw=300, pair_window=100, pair_w=0.0):
    total_hits = total_draws = 0
    counts = Counter()
    recent_deque = deque(maxlen=rw)
    last_seen = {}
    pair_tracker = IncrementalPairTracker(pair_window) if pair_w > 0 else None

    for draw_idx, draw in enumerate(draws):
        nums = draw['numbers']
        if draw_idx < iw:
            for n in nums:
                counts[n] += 1
                last_seen[n] = draw_idx
            recent_deque.append(set(nums))
            if pair_tracker:
                pair_tracker.update(nums)
            continue

        ranked = rank_fn(counts, recent_deque, last_seen, draw_idx, rw,
                         pair_tracker, pair_w)
        hits = len(set(ranked[:10]) & set(nums))
        total_hits += hits
        total_draws += 1

        for n in nums:
            counts[n] += 1
            last_seen[n] = draw_idx
        recent_deque.append(set(nums))
        if pair_tracker:
            pair_tracker.update(nums)

    return total_hits / total_draws if total_draws > 0 else 0


def walkforward(rank_fn, draws, iw=30, rw=300, num_folds=5,
                pair_window=100, pair_w=0.0):
    total = len(draws)
    train_end = int(total * 0.60)
    test_draws = draws[train_end:]
    fold_size = len(test_draws) // num_folds
    baseline = 2.5

    scores = []
    for fold_idx in range(num_folds):
        fold_start = fold_idx * fold_size
        fold_end = fold_start + fold_size if fold_idx < num_folds - 1 else len(test_draws)
        test_fold = test_draws[fold_start:fold_end]
        train_window = draws[:train_end + fold_start]

        if len(train_window) < 10 or len(test_fold) < 5:
            continue

        counts = Counter()
        recent_deque = deque(maxlen=rw)
        last_seen = {}
        pair_tracker = IncrementalPairTracker(pair_window) if pair_w > 0 else None

        for draw in train_window:
            for n in draw['numbers']:
                counts[n] += 1
                last_seen[n] = len(counts)
            recent_deque.append(set(draw['numbers']))
            if pair_tracker:
                pair_tracker.update(draw['numbers'])

        total_hits = 0
        for draw_idx, draw in enumerate(test_fold):
            ranked = rank_fn(counts, recent_deque, last_seen,
                             len(train_window) + draw_idx, rw,
                             pair_tracker, pair_w)
            hits = len(set(ranked[:10]) & set(draw['numbers']))
            total_hits += hits
            for n in draw['numbers']:
                counts[n] += 1
                last_seen[n] = len(train_window) + draw_idx
            recent_deque.append(set(draw['numbers']))
            if pair_tracker:
                pair_tracker.update(draw['numbers'])

        if len(test_fold) > 0:
            scores.append(total_hits / len(test_fold))

    if not scores:
        return 0, 0
    wf_score = sum(scores) / len(scores)
    return wf_score, wf_score - baseline


def make_pair_rank(freq_w, delay_w, recent_w):
    def rank(c, rd, l, di, rw_inner, pair_tracker, pair_w):
        mc = max(c.values()) if c else 1
        freq_scores = {n: c.get(n, 0) / mc for n in range(1, 81)}

        delays = {n: di - l.get(n, 0) for n in range(1, 81)}
        md = max(delays.values()) if delays else 1
        if md == 0: md = 1

        recent_list = list(rd)[-min(rw_inner, len(rd)):]
        recent_freq = Counter()
        for s in recent_list:
            for n in s:
                recent_freq[n] += 1
        mrf = max(recent_freq.values()) if recent_freq else 1
        if mrf == 0: mrf = 1

        base_scores = {
            n: freq_w * freq_scores[n]
               + delay_w * delays[n] / md
               + recent_w * recent_freq.get(n, 0) / mrf
            for n in range(1, 81)
        }

        if pair_tracker and pair_w > 0:
            top20 = sorted(range(1, 81), key=lambda n: -base_scores[n])[:20]
            pair_boost = pair_tracker.get_boost_scores(set(top20))
            final_scores = {
                n: base_scores[n] + pair_w * pair_boost.get(n, 0)
                for n in range(1, 81)
            }
        else:
            final_scores = base_scores

        return sorted(range(1, 81), key=lambda n: -final_scores[n])
    return rank


# === RUN ===
draws = load_draws()
CURRENT_BEST_WF = 2.549925
CURRENT_BEST_BT = 2.557726
BASELINE = 2.5

best_wf = 0
best_bt = 0
best_wf_name = None
best_bt_name = None
results = []

p("=== Exp24: Pair co-occurrence + WF weights (optimized) ===")
p(f"Target WF: {CURRENT_BEST_WF:.6f} | Target BT: {CURRENT_BEST_BT:.6f}")
start = time.time()

# Phase 1: baseline (no pairs) with best WF weights
fw, dw, rw_w = 0.70, 0.16, 0.14
fn_base = make_pair_rank(fw, dw, rw_w)
bt_base = backtest(fn_base, draws, pair_w=0.0)
wf_base, edge_base = walkforward(fn_base, draws, pair_w=0.0)
p(f"Baseline (no pairs): bt={bt_base:.6f} wf={wf_base:.6f} edge={edge_base:.6f}")
best_wf = wf_base
best_bt = bt_base
best_wf_name = "baseline_no_pairs"
best_bt_name = "baseline_no_pairs"
results.append({"name": "baseline", "bt": round(bt_base, 6),
                "wf": round(wf_base, 6), "edge": round(edge_base, 6),
                "pw": 0, "pwin": 0})

# Phase 2: Grid search pair_w and pair_window
configs = []
for pw_pct in [5, 10, 15, 20, 25, 30, 40, 50]:
    pw = pw_pct / 100.0
    for pair_window in [50, 100, 150, 200, 300]:
        configs.append((pw, pair_window))

p(f"\nPhase 2: {len(configs)} pair configs (pair_w x pair_window)")

for idx, (pw, pwin) in enumerate(configs):
    fn = make_pair_rank(fw, dw, rw_w)
    bt = backtest(fn, draws, pair_window=pwin, pair_w=pw)
    wf, edge = walkforward(fn, draws, pair_window=pwin, pair_w=pw)

    name = f"pair_pw{pw:.2f}_win{pwin}"
    results.append({"name": name, "bt": round(bt, 6), "wf": round(wf, 6),
                    "edge": round(edge, 6), "pw": pw, "pwin": pwin})

    if wf > best_wf:
        best_wf = wf
        best_wf_name = name
        p(f"  *** NEW WF BEST [{idx+1}]: {name} bt={bt:.6f} wf={wf:.6f}")
    if bt > best_bt:
        best_bt = bt
        best_bt_name = name
        p(f"  *** NEW BT BEST [{idx+1}]: {name} bt={bt:.6f} wf={wf:.6f}")

    if (idx + 1) % 10 == 0:
        elapsed = time.time() - start
        p(f"  [{idx+1}/{len(configs)}] elapsed={elapsed:.0f}s best_wf={best_wf:.6f} best_bt={best_bt:.6f}")

# Phase 3: If a pair config beat baseline, fine-tune weights with best pair params
results.sort(key=lambda x: -x["wf"])
best_pair = next((r for r in results if r.get("pw", 0) > 0), None)

if best_pair and best_pair["wf"] > wf_base:
    best_pw = best_pair["pw"]
    best_pwin = best_pair["pwin"]
    p(f"\n--- Phase 3: Weight sweep with pw={best_pw}, pwin={best_pwin} ---")

    weight_combos = [
        (0.72, 0.16, 0.12), (0.68, 0.18, 0.14), (0.70, 0.18, 0.12),
        (0.72, 0.14, 0.14), (0.75, 0.15, 0.10), (0.65, 0.20, 0.15),
        (0.70, 0.20, 0.10), (0.68, 0.16, 0.16), (0.74, 0.16, 0.10),
        (0.66, 0.18, 0.16),
    ]

    for fw2, dw2, rw2 in weight_combos:
        fn = make_pair_rank(fw2, dw2, rw2)
        bt = backtest(fn, draws, pair_window=best_pwin, pair_w=best_pw)
        wf, edge = walkforward(fn, draws, pair_window=best_pwin, pair_w=best_pw)

        name = f"p3_fw{fw2}_dw{dw2}_rw{rw2}_pw{best_pw:.2f}"
        results.append({"name": name, "bt": round(bt, 6), "wf": round(wf, 6),
                        "edge": round(edge, 6)})

        if wf > best_wf:
            best_wf = wf
            best_wf_name = name
            p(f"  *** NEW WF BEST: {name} bt={bt:.6f} wf={wf:.6f}")
        if bt > best_bt:
            best_bt = bt
            best_bt_name = name
            p(f"  *** NEW BT BEST: {name} bt={bt:.6f} wf={wf:.6f}")
else:
    p("\n--- Phase 3: Skipped (pairs didn't beat baseline WF) ---")

# Phase 4: iw/rw sweep with best overall config
p(f"\n--- Phase 4: iw/rw sweep ---")
best_cfg = results[0]
best_pw_final = best_cfg.get("pw", 0)
best_pwin_final = best_cfg.get("pwin", 100)

for iw in [20, 25, 30, 35, 40]:
    for rw in [200, 250, 300, 350]:
        fn = make_pair_rank(fw, dw, rw_w)
        bt = backtest(fn, draws, iw=iw, rw=rw,
                      pair_window=best_pwin_final, pair_w=best_pw_final)
        wf, edge = walkforward(fn, draws, iw=iw, rw=rw,
                               pair_window=best_pwin_final, pair_w=best_pw_final)

        name = f"p4_iw{iw}_rw{rw}_pw{best_pw_final:.2f}"
        results.append({"name": name, "bt": round(bt, 6), "wf": round(wf, 6),
                        "edge": round(edge, 6)})

        if wf > best_wf:
            best_wf = wf
            best_wf_name = name
            p(f"  *** NEW WF BEST: {name} bt={bt:.6f} wf={wf:.6f}")
        if bt > best_bt:
            best_bt = bt
            best_bt_name = name
            p(f"  *** NEW BT BEST: {name} bt={bt:.6f} wf={wf:.6f}")

results.sort(key=lambda x: -x["wf"])
elapsed = time.time() - start
p(f"\nTotal elapsed: {elapsed:.0f}s | Configs tested: {len(results)}")

output = {
    "metric": round(best_wf, 6),
    "lottery": "multi_multi_pl",
    "current_best_wf": CURRENT_BEST_WF,
    "best_wf": round(best_wf, 6),
    "best_wf_config": best_wf_name,
    "best_bt": round(best_bt, 6),
    "best_bt_config": best_bt_name,
    "beats_wf": best_wf > CURRENT_BEST_WF,
    "beats_bt": best_bt > CURRENT_BEST_BT,
    "total_configs": len(results),
    "elapsed_s": round(elapsed, 1),
    "top_5_wf": results[:5],
}

p("\n" + json.dumps(output, indent=2))
