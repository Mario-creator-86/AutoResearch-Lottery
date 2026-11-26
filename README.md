# AutoResearch Lottery

An autonomous, agent-driven research loop for lottery number-selection strategies,
inspired by Andrej Karpathy's "autoresearch" pattern: an agent edits an experiment
file, runs it, keeps improvements, discards regressions and logs every result.
This is an independent hobby/engineering project. It is not affiliated with or
endorsed by Andrej Karpathy.

## Honest framing

Lotteries are negative-expected-value games of chance, and nothing in this
repository predicts draws. The project is a research and engineering exercise:
backtesting and walk-forward validation of ranking strategies, statistical
significance testing that corrects for the number of configurations tried, and
an autonomous experiment loop with a live dashboard. Its own reality-check tools
exist precisely to show when an apparent "edge" is selection noise.

## What it does

- **Supported lotteries** (`lottery_config.py`): Multi Multi, Lotto, Mini Lotto and
  Kaskada (Poland); Loto 6/49, Loto 5/40 and Joker (Romania); SuperEnalotto (Italy);
  EuroJackpot; EuroMillions; UK Lotto; Powerball and Mega Millions (US). Each has a
  CSV history under `data/` and its own state directory `data/<lottery_id>/`.
- **Experiment loop** (`program.md`, `AGENTS.md`): the classic autoresearch cycle.
  `train.py` is the single editable experiment file; it prints JSON with a `metric`
  (average hits per ticket on a rolling backtest). A result that beats the current
  best is KEEP, otherwise DISCARD, and every run is appended to `results.tsv`.
- **Autonomous optimizer** (`auto_loop.py`): per-lottery grid search around the best
  known parameters, narrowing and widening automatically, writing KEEP/DISCARD rows
  to `data/<lottery_id>/results.tsv`.
- **Backtesting and validation**: `generic_backtest.py` (rolling backtest for any
  lottery format), `walkforward.py` (anchored 5-fold walk-forward),
  `forward_test.py` (parameters frozen at a date, evaluated only on later draws),
  `significance.py` (exact hypergeometric variance, z-score, multiple-comparison
  corrected p-values, verdict significant / suggestive / noise) and `bias_scan.py`
  (chi-square style tests for real anomalies in the draws themselves).
- **Agents system** (`agents/`): file-based multi-agent runtime (strategy researcher,
  lottery manager, ticket generator, system monitor, experiment monitor, scheduler,
  Ollama strategist, AI strategy coder, AI ticket agent, Telegram agent). Agents
  write status, logs and messages under `agents/`; see `agents/README.md`.
- **FastAPI dashboard** (`server.py`, `static/`): starts and supervises the agents,
  refreshes draw histories, streams progress and exposes a JSON API
  (`/api/health`, `/api/lotteries`, `/api/<lottery_id>/experiments`,
  `/api/<lottery_id>/walkforward`, `/api/<lottery_id>/reality-check`, `/api/ledger`,
  and more).
- **Optional Ollama AI strategist** (`ollama_ai.py`, `agents/ollama_strategist.py`,
  `ai_strategy_lab.py`): asks a local Ollama model for analysis, ranked numbers and
  parameter suggestions, and lets it write strategy plugins into `ai_strategies/`.
  Advisory only: the deterministic backtest still decides KEEP/DISCARD.
- **Optional Telegram notifications** (`telegram_bot.py`, `agents/telegram_agent.py`).
- **Data scraping** (`scrape_lotteries.py`, `update_lottery_data.py`,
  `romania_lotto.py`, `scrape_jackpot_*.py`): download and append draw histories
  and jackpot data from public sources; no API keys needed.
- **Decision ledger and play advisor** (`decision_ledger.py`, `play_advisor.py`,
  `portfolio_optimizer.py`, `prize_estimates.py`): append-only JSONL log of what
  was proposed and why, plus expected-value and ticket-overlap tooling.

## Repository layout

| Path | Purpose |
| --- | --- |
| `server.py`, `static/` | FastAPI backend and single-page dashboard |
| `auto_loop.py` | Per-lottery autonomous optimizer |
| `train.py`, `results.tsv`, `program.md` | Original single-file experiment loop and its log/agenda |
| `generic_backtest.py`, `walkforward.py`, `forward_test.py`, `significance.py`, `bias_scan.py` | Backtest and validation engines |
| `lottery_config.py`, `load_data.py`, `draw_matching.py` | Lottery definitions and CSV loading |
| `agents/` | Multi-agent runtime, `config.json`, strategy registry |
| `ai_strategies/<lottery_id>/` | AI-generated strategy plugins per lottery |
| `ollama_ai.py`, `ai_strategy_lab.py`, `ai_actions.py` | Ollama integration and AI action approval queue |
| `telegram_bot.py` | Telegram notification helper |
| `scrape_lotteries.py`, `update_lottery_data.py`, `romania_lotto.py`, `scrape_jackpot_*.py`, `jackpot_*.py` | Data download and jackpot tracking |
| `data/` | Draw histories (`<lottery_id>.csv`) and per-lottery state |
| `tests/` | pytest suite |
| `watchdog.py`, `install_autostart.ps1`, `start_autoresearch.bat`, `restart_server.py` | Windows process supervision and launchers |
| `AGENTS.md`, `CLAUDE.md`, `AUDIT_LOGIC_*.md` | Agent instructions and logic audits |

## Requirements

- Python 3.11 or newer (the project uses modern typing and `slots=True` dataclasses;
  it was developed on Python 3.14).
- Packages in `requirements.txt`: numpy, pandas, scipy, scikit-learn, xgboost,
  lightgbm, fastapi, uvicorn, httpx, requests, beautifulsoup4, pytest, psutil.
- Optional: [Ollama](https://ollama.com) running locally (default
  `http://localhost:11434`) for the AI strategist. Without it the AI features
  report "not reachable" and everything else works.
- Optional: a Telegram bot for notifications.
- The dashboard, launchers and watchdog are written for Windows (process handling
  via `ctypes`/`CREATE_NEW_PROCESS_GROUP`); the backtest, loop and tests are
  plain Python.

## Quick start

```bash
git clone https://github.com/<your-user>/AutoResearch-Lottery.git
cd AutoResearch-Lottery
python -m venv .venv
.venv\Scripts\activate          # Windows; on bash: source .venv/Scripts/activate
pip install -r requirements.txt
copy .env.example .env           # then edit .env (optional, Telegram only)
```

`.env` keys (all optional; leave empty to disable Telegram):

- `TELEGRAM_BOT_TOKEN` - token of the Telegram bot that sends notifications.
- `TELEGRAM_CHAT_ID` - numeric chat ID that receives them.

Ollama settings are plain environment variables: `OLLAMA_HOST` (default
`http://localhost:11434`), `OLLAMA_MODEL` / `OLLAMA_MODELS` (comma-separated
candidate models), `OLLAMA_TIMEOUT`.

**Start the dashboard**

```bash
python server.py --port 8020     # or double-click start_autoresearch.bat on Windows
```

Open <http://localhost:8020>. The port can also be set with `AUTORESEARCH_PORT`.
The server starts the agents, refreshes draw histories and lets you start/stop
optimizer loops per lottery from the UI.

**Run one experiment (classic autoresearch loop)**

```bash
python train.py                  # prints JSON with "metric" = average hits
```

Edit only the experiment zone in `train.py`, run it, and append the result to
`results.tsv` as described in `AGENTS.md`.

**Run the autonomous optimizer directly**

```bash
python auto_loop.py --lottery lotto_pl          # infinite loop for one lottery
python auto_loop.py --lottery lotto_pl --once   # a single iteration
python run_walkforward_all.py                   # walk-forward for every lottery's best
python bias_scan.py                             # anomaly scan of the draw data
```

`autoloop_manager.py` is the legacy driver of the original single-lottery loop
and depends on an external package that is not part of this repository; use
`auto_loop.py` instead.

## Testing

```bash
pytest
```

`pytest.ini` points at `tests/`. The fixtures in `tests/conftest.py` redirect all
runtime state to a temporary directory, so the suite never touches real
ledgers or forward-test snapshots.

## Windows autostart

`install_autostart.ps1` registers a scheduled task that runs `watchdog.py` at
logon; the watchdog polls `/api/health` and (re)starts `server.py` when it is
missing or unresponsive. Run `powershell -ExecutionPolicy Bypass -File
install_autostart.ps1` to install and add `-Remove` to uninstall.
`python watchdog.py --once` performs a single health check.

## Data

Draw histories live in `data/<lottery_id>.csv` in a unified format
(`draw_date,draw_time,draw_id,number_01,...,bonus_01,...`); the original Multi
Multi archive is `multi_multi_official_history_full.csv`. The server refreshes
them on boot and every 6 hours, or run `python scrape_lotteries.py` (all) /
`python scrape_lotteries.py lotto_pl` (one) / `python scrape_lotteries.py --list`.
Per-lottery results, walk-forward and forward-test snapshots, ledgers and
jackpot histories are committed under `data/<lottery_id>/`. Runtime state is
gitignored: progress and loop-state JSON, lock files, logs, AI predictions,
agent status/messages/incidents and `.env`.

## License

MIT License, see LICENSE.
