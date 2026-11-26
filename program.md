# Autoresearch Program: Multi Multi 20/80 Lottery Optimization

## Problem

Poland's Multi Multi 20/80 lottery: 20 numbers drawn from 1-80 each round.
A random ticket of size K has expected hits = K * 20/80 = K/4.
For ticket_size=10, random baseline = **2.500 average_hits**.

Current best strategies achieve approximately **2.50-2.55**. Any improvement, even to 2.56+, is statistically significant over 16,665 evaluated draws.

## Objective

Maximize `average_hits` on rolling backtest over the full 16,675-draw history (1996-2026). Secondary metrics: `hit_rate_2_plus`, `hit_rate_3_plus`, `portfolio_average_best_hits`.

## Infrastructure

### Data
- `load_data.py`: `load_draws()` returns `list[DrawRecord]` (16,675 draws)
- Each draw has `.numbers` (tuple of 20 ints, 1-80) and `.draw_date`

### Backtest Engine (import from loto_polonia)
```python
from loto_polonia.analysis.backtest import backtest_strategy, backtest_all_strategies, RankFn

# RankFn signature:
# (counts: dict[int,int], recent_deque: deque[set], last_seen: dict[int,int], draw_index: int, recent_window: int) -> list[int]
# Must return all 80 numbers ranked best-to-worst. Top ticket_size numbers become ticket 1, next become ticket 2, etc.

result = backtest_strategy(
    strategy_key="my_strategy",
    rank_fn=my_rank_function,      # implements RankFn
    draws=draws,
    ticket_size=10,
    recent_window=25,
    initial_window=10,
)
# result.average_hits -> primary metric
# result.hit_rate_2_plus, result.hit_rate_3_plus -> secondary
# result.portfolio_average_best_hits -> best-of-3-tickets metric
```

### Scoring (import from loto_polonia)
```python
from loto_polonia.generators.incremental_stats import compute_simplified_score
# Current weights: frequency=0.45, recent=0.35, delay=0.20
# Returns dict[int, float] mapping each number 1-80 to a score
```

### Existing 5 Strategies
Available via `backtest_all_strategies()`:
1. **top_frequency** - ranks by total historical count
2. **hot_cold_mix** - alternates recent-hot and long-delayed numbers
3. **recent_trend** - ranks by recent window frequency, then absolute
4. **weighted_random** - weighted sampling by combined score (seed-based)
5. **balanced_distribution** - combined score + odd/even + low/high constraints

## Experiment Ideas (Priority Order)

### Phase 1: Parameter Tuning
1. **Scoring weight grid search** - Try all combinations of (frequency, recent, delay) weights that sum to 1.0, in steps of 0.05. Find the optimal weights that maximize average_hits.
2. **Recent window sweep** - Test recent_window values: 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200. Each value changes how "recent" is defined.
3. **Ticket size exploration** - Test sizes 6, 8, 10, 12, 15, 20. Larger tickets have more hits but less precision.
4. **Initial window tuning** - Test initial_window: 5, 10, 15, 20, 30, 50.

### Phase 2: New Strategies
5. **Exponential decay scoring** - Weight recent draws exponentially (lambda decay) rather than uniform window.
6. **Moving average crossover** - Like trading: short MA vs long MA of number frequencies. Pick numbers where short > long (momentum).
7. **Delay-based reversion** - Numbers with the longest delay have higher probability of appearing (mean reversion hypothesis).
8. **Pair-aware ranking** - Incorporate pair co-occurrence: pick numbers that historically appear together.
9. **Sector rotation** - Divide 1-80 into sectors (decades). Rotate emphasis based on which sector is "due".

### Phase 3: Machine Learning
10. **Logistic regression per number** - For each number 1-80, train logistic regression on features (frequency, recency, delay, gap, day_of_week) to predict P(appears in next draw).
11. **Random Forest classifier** - Binary classification per number using rolling features.
12. **XGBoost/LightGBM** - Gradient boosted trees on engineered features.
13. **Feature engineering** - Moving averages at multiple windows (5,10,25,50,100), acceleration (change in frequency), pair scores, sector balance.

### Phase 4: Ensembles & Portfolios
14. **Rank aggregation** - Average ranks across multiple strategies. Ensemble should be more robust.
15. **Weighted ensemble** - Weight each strategy by its recent backtest performance.
16. **Portfolio optimization** - Instead of optimizing ticket 1 alone, optimize the 3-ticket portfolio to maximize `portfolio_average_best_hits`.
17. **Diversity-aware portfolio** - Ensure the 3 tickets cover different "zones" of the number space.

### Phase 5: Advanced
18. **Adaptive window** - Dynamically select recent_window per draw based on recent prediction accuracy.
19. **Regime detection** - Identify regimes (streaky vs mean-reverting periods) and switch strategies.
20. **Genetic algorithm** - Evolve weight vectors and strategy parameters.

### Phase 6: Deep Feature Engineering & Non-Linear ML (HIGH PRIORITY)

These ideas address fundamental limitations of the current approach — all existing strategies are linear combinations of frequency/recency/delay. Non-linear models can capture interactions that linear scoring misses.

21. **Rich feature vector per number** - For each number 1-80 at each draw, compute a feature vector:
    - Moving average frequency at windows 5, 10, 25, 50, 100, 250, 500
    - Frequency acceleration (delta between short MA and long MA)
    - Delay since last appearance (raw + normalized)
    - Streak length (consecutive draws present/absent)
    - Pair co-occurrence score (top-K most correlated partner numbers' recent frequencies)
    - Day-of-week encoding (cyclical: sin/cos of day index)
    - Draw index position (cyclical patterns over months/years)
    - Sector balance (how many of 1-10, 11-20, ..., 71-80 appeared recently)

22. **Rolling Random Forest classifier** - For each draw in the backtest:
    - Build feature matrix from the last N draws
    - Target: binary (did number appear in next draw?)
    - Train RF on rolling window, predict P(appear) for each number
    - Rank by predicted probability, use as RankFn
    - Key: retrain every K draws (not every draw — too slow)

23. **XGBoost/LightGBM rolling predictor** - Same as #22 but with gradient boosted trees. LightGBM is faster for rolling retraining. Try different retrain intervals (every 50, 100, 200 draws).

24. **Logistic regression with interactions** - Create polynomial features (degree 2) from the feature vector. Logistic regression on interactions can capture non-linear effects cheaply.

25. **Pair/triplet exploitation** - Analyze co-occurrence of number pairs and triplets. If pair (a,b) appears together significantly more than expected by chance (chi-squared test), build a strategy that when picking number a, also favors b. Create a graph of "affinity" between numbers and use it to influence ranking.

26. **Portfolio-first optimization** - Instead of optimizing ticket 1's average_hits, directly optimize `portfolio_average_best_hits` (the best-of-3-tickets metric). This changes the ranking: ticket 1 should be "high conviction", ticket 2 should be "hedged/diversified", ticket 3 should be "contrarian".

27. **Meta-strategy with regime detection** - Track rolling accuracy of each strategy over the last 50 draws. At each draw, pick the strategy that performed best recently. This is "strategy momentum" — exploit the fact that different regimes favor different approaches.

28. **Stacking meta-model** - Use the output ranks from all 5 base strategies as features. Train a meta-model (logistic regression or small NN) to predict which numbers will appear, using strategy ranks as inputs. The meta-model learns which strategy to trust in which conditions.

### Phase 7: Insights din Experimentele Anterioare

Observatii din Exp1-5 care ar trebui exploatate:

29. **Frecventa istorica totala domina** - Weight optim pe frequency = 0.75. Asta inseamna ca unele numere au un bias real (sau cel putin persistent). Investigheaza: care numere au deviatii statistice semnificative de la 25% (expected P per number)?

30. **Recency e zgomot** - Weight optim pe recent = 0.05. Strategiile "hot number" sunt practic anti-patterns. Testeaza: o strategie care evita activ numerele recente ("contrarian anti-hot").

31. **Ensemble diversity matters** - top_freq + hot_cold + balanced functioneaza pentru ca sunt diverse. hot_cold aduce diversitate (numere rare). Testeaza: creaza deliberat strategii maximally diverse si ensemble-aza.

## Rules

1. **Primary metric**: `average_hits` (higher is better). Report to 6 decimal places.
2. **Output format**: `train.py` must print a single JSON object with at minimum `{"metric": <average_hits>}`.
3. **Reproducibility**: Use fixed seeds for any random operations. Default seed: 42.
4. **Runtime**: Each experiment must complete in under 5 minutes on CPU.
5. **Only modify train.py**: All other files are read-only infrastructure.
6. **Imports**: You may import from `loto_polonia.*` (via sys.path) and any standard library. For ML, use scikit-learn, xgboost, lightgbm (installed in venv).
7. **New rank functions**: Must match `RankFn` signature to use `backtest_strategy()`.
8. **Track progress**: After each experiment, note what was tried and the result.

## Baseline to Beat

Run `backtest_all_strategies(draws, ticket_size=10, recent_window=25, initial_window=10)` first. The best strategy's `average_hits` is your baseline. Every experiment should aim to exceed it.

## NEVER STOP

Keep running experiments. After each result, immediately propose and try the next idea. Iterate through the night if possible. Each 2-5 minute experiment adds data. 12+ experiments per hour.
