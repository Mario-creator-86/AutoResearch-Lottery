# AGENTS.md - Autoresearch Agent Loop

## Project Overview

Autonomous lottery strategy optimization using the autoresearch pattern. The agent iteratively modifies `train.py`, runs experiments, evaluates results, and tracks progress — discovering better strategies for Poland's Multi Multi 20/80 lottery.

## How to Run the Agent Loop

### Setup
```bash
cd /path/to/AutoResearch-Lottery
.venv\Scripts\activate  # or: source .venv/Scripts/activate (bash)
```

### The Loop (repeat until stopped)

1. **Read** `program.md` for the research agenda and experiment ideas
2. **Read** `results.tsv` to see all past experiments and the current best `average_hits`
3. **Pick** the next experiment idea from `program.md` (prioritize untried ideas)
4. **Modify** `train.py` — only the EXPERIMENT ZONE section between the markers
5. **Run** `python train.py` and capture the JSON output
6. **Evaluate**:
   - If `metric` (average_hits) > current best in results.tsv → status = **KEEP**
   - Otherwise → status = **DISCARD**
7. **Log** — append a row to `results.tsv`:
   ```
   <commit_hash>\t<average_hits>\t<KEEP|DISCARD>\t<strategy>\t<hit_rate_2_plus>\t<hit_rate_3_plus>\t<portfolio_avg>\t<elapsed_s>\t<description>
   ```
8. **Commit** — `git add train.py results.tsv && git commit -m "<description>"`
   - If DISCARD: revert train.py to the last KEEP version before the next experiment
   - If KEEP: train.py stays as-is for the next iteration
9. **Repeat** from step 2

### Important Rules

- **Only modify `train.py`** — all other files are infrastructure
- **Primary metric**: `average_hits` (higher = better). Random baseline for ticket_size=10 is 2.500
- **Runtime budget**: each experiment must finish in under 5 minutes
- **Reproducibility**: use fixed seeds (default 42) for any random operations
- **JSON output**: `train.py` must print valid JSON with at least `{"metric": <float>}`
- **Never stop**: after each experiment, immediately start the next one

## Commands

```bash
# Run experiment
python train.py

# Check data loading
python load_data.py

# View experiment history
cat results.tsv

# View git log of experiments
git log --oneline
```

## Architecture

- `train.py` — experiment script (agent modifies the EXPERIMENT ZONE)
- `load_data.py` — CSV data loader (read-only)
- `program.md` — research agenda with experiment ideas (read-only)
- `results.tsv` — experiment tracking log (append-only)
- `multi_multi_official_history_full.csv` — 16,675 draws 1996-2026 (read-only)

## Autonomous System (server + agents)

`python server.py` → FastAPI dashboard on http://localhost:8020 (override with
`--port N` or `AUTORESEARCH_PORT`; the server exports the port to the agents it
spawns, `agents/utils.py` uses it for the health check) that runs the
system autonomously (no agent-in-the-loop required):

- **auto_loop.py** — per-lottery infinite optimizer (grid search around best
  params, KEEP/DISCARD into `data/{id}/results.tsv`). `lottery_manager` agent
  keeps one loop running per lottery with enough data (max 2 concurrent,
  RO/IT priority, 6h rotation via `loop_rotation_hours` in agents/config.json).
  A `.autoloop.paused` marker in `data/{id}/` (written by the UI Stop button)
  prevents auto-restart of that lottery.
- **History auto-download** — server background task refreshes every lottery's
  CSV at boot + every 6h (`data/.auto_update.json` holds state; endpoints
  `/api/auto-update/status`, `/api/auto-update/run`). Sources in
  `scrape_lotteries.py` (Romania: official loto.ro archive via romania_lotto.py,
  Italy SuperEnalotto: national-lottery.com yearly archives, PL: wynikilotto
  CSVs, US: data.ny.gov, UK: national-lottery API).
- **Ollama AI** — `ollama_ai.py` + `agents/ollama_strategist.py` (runs every
  `interval_minutes`, default 360): computes real stats per lottery, asks an
  Ollama model (default `glm-5.3-flash:cloud` via localhost:11434, proxying
  Ollama Cloud) for analysis + ranked numbers + parameter suggestions. Output:
  `data/{id}/ai_predictions.json`, suggestions exported to
  `agents/strategies/pending_review.json`. Endpoints: `/api/ai/status`,
  `/api/{id}/ai-predictions`, `POST /api/{id}/ai-analyze`. The AI is advisory —
  the deterministic backtest still decides KEEP/DISCARD.
- **agent gotchas** — `agents/utils.py update_agent_status` must MERGE into the
  status file (a full overwrite drops `last_activity` and makes system_monitor
  see every agent as stale). `system_monitor` restarts an agent only when its
  PID is dead — a stale-but-alive agent is only logged, never duplicated.

## Dependencies

The project imports from the Loto Polonia codebase via sys.path:
- `loto_polonia.analysis.backtest` — rolling backtest engine
- `loto_polonia.generators.incremental_stats` — simplified scoring
- `loto_polonia.models` — DrawRecord dataclass

ML libraries available: scikit-learn, xgboost, lightgbm, numpy, pandas, scipy.

## Disclaimer

Lottery draws are random and independent. This is exploratory analysis — no strategy guarantees predictive advantage over future draws.

## AI Self-Improvement Stack (septembrie 2026)

- **Decision Ledger** — `decision_ledger.py`: jurnal append-only JSONL
  (`data/{id}/decision_ledger.jsonl` + flux global) cu CE a propus / CE a
  gândit / DE CE / rezultat. Endpoint: `/api/ledger`, `/api/{id}/ledger`.
- **Bilete AI zilnice** — `agents/ai_ticket_agent.py`: bilete propuse per
  extragere (`data/{id}/ai_tickets.json`) cu raționament per bilet, verificate
  automat după extragere (hits vs numerele câștigătoare reale), reflecție a
  modelului → ledger + actualizarea propriilor prompt-uri. Endpoint:
  `/api/{id}/ai-tickets`.
- **Strategii scrise de AI** — `agents/ai_strategy_coder.py` +
  `ai_strategy_lab.py`: modelul citește statistici + cercetare web (read-only)
  și scrie plugin-uri Python în `ai_strategies/{lottery}/` (interfață fixă:
  `create(config) -> rank(counts, recent_deque, last_seen, draw_index,
  recent_window)`, NAME obligatoriu prefix „z"). Validare → backtest →
  promovare în registry (doar dacă bate best-ul) → auto_loop le include în
  căutare. Plugin-urile eșuate rămân arhivate cu eroarea lor (material de
  învățare în ledger).
- **Auto-îmbunătățire prompt-uri** — `data/ai_self/prompt_overrides.json`
  (analysis_extra / coder_extra / ticket_extra) editat de chiar AI-ul.
- **Stare agent (UI)** — `/api/agent-activity`: state
  coding/researching/thinking/working/idle + de când; widget animat în header.

## Reality Check & Portfolio Optimizer (15 septembrie 2026)

Diagnostic pe datele reale: numerele ies uniform la toate loteriile (chi² p 0.2–0.9),
backtestul NU are look-ahead (strategiile dau exact baseline-ul pe extrageri sintetice),
iar edge-urile găsite de auto_loop sunt compatibile cu zgomotul după corecția pentru
numărul de configurații testate (mii–sute de mii). Deci nu se poate crește șansa unui
bilet; se poate îmbunătăți portofoliul și onestitatea deciziilor:

- **portfolio_optimizer.py** — căutare locală (numpy, extrageri simulate) care maximizează
  P(≥1 premiu pe extragere) pentru biletele AI la același cost + penalizare de popularitate
  (1–31, multipli de 7, secvențe, ultima extragere) la loteriile pari-mutuel. Biletul 1
  (favoritul) rămâne fix; metricele se raportează pe o simulare independentă și nu se
  livrează niciodată un portofoliu mai slab. Integrat în `ai_ticket_agent` (câmp
  `portfolio` în ai_tickets.json + linie în Telegram).
- **significance.py** — varianță hipergeometrică exactă, z, p ajustat Šidák pentru numărul
  de experimente (suma „(N configs" din results.tsv), verdict significant/suggestive/noise,
  recordul prospectiv al biletelor AI (SE pe extrageri, nu pe bilete). Endpoint-uri
  `/api/reality-check` și `/api/{id}/reality-check`; panou „Reality Check" în tabul Strategie.
- Folosit ca poartă: `ai_strategy_coder` promovează doar dacă plugin-ul bate best-ul ȘI
  baseline-ul pe holdout (ultimele 25%, `ai_strategy_lab.backtest_plugin_holdout`) ȘI nu e
  zgomot (`promotion_requires_significance` în config, implicit true); notificarea KEEP din
  auto_loop include verdictul; analiza Ollama primește `reality_check` în statistici;
  reflecția agentului de bilete folosește recordul cumulat și păstrează doar 3 lecții.

## Telegram & AI Full Control (8 septembrie 2026)

- **telegram_bot.py** — client Telegram (TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID din
  .env sau .telegram_chat_id). Rate-limited (min 20s între mesaje), nichilist pe
  erori — poate fi apelat din orice agent fără risc.
- **ai_actions.py** — magistrala de acțiuni prin care AI-ul controlează proiectul:
  restart_loop / stop_loop / start_loop / apply_strategy_params / run_walkforward /
  update_history / run_ai_analysis / set_agent_config (whitelist de căi).
  Flux: propunere → ledger (ai_action, „pending_approval") → Telegram cu butoane
  ✅/❌ → execuție → rezultat în ledger + chat. Acțiunile cu risc scăzut
  (walkforward, update_history, run_ai_analysis, start_loop) se execută direct.
- **agents/telegram_agent.py** — long-polling: procesează butoanele de aprobare
  și comenzile /status, /activity, /best, /pending, /help.
- Notificări automate: strategie promovată de AI-coder, strategie îmbunătățită
  (auto-loop KEEP), bilete cu ≥3 hits, propunerile de acțiuni din analizele AI
  (câmpul „proposed_actions" din JSON-ul de analiză), erorile agenților.
- Testele izolează ACTIONS_DIR cu monkeypatch — căile se calculează dinamic
  (_pending_file()/_history_file()).

## Imported Claude Cowork project instructions
