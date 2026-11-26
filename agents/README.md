# Multi-Agent Lottery System

Sistem multi-agent pentru optimizarea strategiilor de loterie, care rulează continuu și autonom.

## Arhitectură

```
┌─────────────────────────────────────────────────────────┐
│                   AGENT SYSTEM                            │
├─────────────────────────────────────────────────────────┤
│  Agent 1: Strategy Researcher  │   On-demand (6h)         │
│  Agent 2: Lottery Manager      │   Continuous             │
│  Agent 3: Ticket Generator     │   On-demand (daily)      │
│  Agent 4: System Monitor       │   Continuous             │
└─────────────────────────────────────────────────────────┘
                            │
               ┌────────────┴────────────┐
               │   File-based Messages   │
               └────────────┬────────────┘
                            │
┌───────────────────────────┴─────────────────────────────┐
│              EXISTING LOTTERY SYSTEM                      │
│   auto_loop.py │ server.py │ telegram_bot.py            │
└─────────────────────────────────────────────────────────┘
```

## Cei 4 Agenți

### 1. Strategy Researcher (`strategy_researcher.py`)
- **Schedule**: La fiecare 6 ore (00:00, 06:00, 12:00, 18:00)
- **Funcție**: Caută strategii noi online (forumuri, papers, site-uri statistice)
- **Output**: Salvează în `agents/strategies/pending_review.json`
- **Acțiuni**:
  - Extrage strategii din text (Gap Analysis, Hot/Cold, Wheeling, etc.)
  - Evaluează încrederea (low/medium/high)
  - Notifică pe Telegram când găsește strategii noi

### 2. Lottery Manager (`lottery_manager.py`)
- **Schedule**: Continuu (verifică la fiecare 60 secunde)
- **Funcție**: Gestionează loteriile și balansează resursele
- **Restricții**:
  - Max 2 loterii simultan în experimente
  - Max 80% CPU, 75% Memory
- **Acțiuni**:
  - Priority queue pentru task-uri
  - Auto-scheduling (zilnic bilete, săptămânal walkforward)
  - Load balancing automat

### 3. Ticket Generator (`ticket_generator.py`)
- **Schedule**: Zilnic la 8:00 AM (generare) + după extrageri (verificare)
- **Funcție**: Generează bilete și verifică câștigurile
- **Acțiuni**:
  - Generează 5 bilete per loterie folosind cea mai bună strategie
  - Salvează în `agents/tickets/today/`
  - Verifică rezultatele după fiecare extragere
  - Propune îmbunătățiri dacă >7 pierderi consecutive
  - Trimite raport zilnic pe Telegram la 22:00

### 4. System Monitor (`system_monitor.py`)
- **Schedule**: Continuu (verifică la fiecare 30 secunde)
- **Funcție**: Monitorizează și repară sistemul
- **Acțiuni**:
  - Verifică heartbeat pentru toți agenții
  - Auto-restart dacă un agent crash-uiește
  - Omoară procese zombie
  - Repară fișiere JSON corupte (din recovery)
  - Eliberează lock files stale
  - Monitorizează logurile pentru erori cunoscute
  - Trimite alerte pe Telegram

## Quick Start

### Pornire rapidă

```bash
# Pornește toți agenții
python start_agents.py

# Sau folosește CLI
python agents/agent_cli.py start all
```

### Management

```bash
# Status
python agents/agent_cli.py status

# Loguri
python agents/agent_cli.py logs system_monitor
python agents/agent_cli.py logs lottery_manager

# Oprire
python agents/agent_cli.py stop all

# Sau force stop
python agents/agent_cli.py stop all --force

# Incidente
python agents/agent_cli.py incidents

# Declanșare manuală
python agents/agent_cli.py trigger research
python agents/agent_cli.py trigger generate_tickets --lottery multi_multi_pl
```

### Restart

```bash
python start_agents.py --restart
```

## Structura fișierelor

```
agents/
├── agent_base.py              # Clasa de bază pentru agenți
├── agent_cli.py               # CLI pentru management
├── agent_scheduler.py         # Scheduler pentru on-demand agents
├── config.json                # Configurație globală
│
├── strategy_researcher.py     # Agent 1
├── lottery_manager.py         # Agent 2
├── ticket_generator.py        # Agent 3
└── system_monitor.py          # Agent 4
│
├── status/                    # Heartbeat files
│   ├── strategy_researcher.json
│   ├── lottery_manager.json
│   ├── ticket_generator.json
│   └── system_monitor.json
│
├── messages/
│   ├── broadcast.json         # Mesaje pentru toți agenții
│   └── commands/              # Mesaje către agenți specifici
│       ├── to_strategy_researcher.json
│       ├── to_lottery_manager.json
│       └── to_ticket_generator.json
│
├── tasks/                     # Task queue
│   ├── pending.json
│   ├── in_progress.json
│   └── completed.json
│
├── strategies/                # Strategii descoperite
│   ├── pending_review.json
│   ├── approved.json
│   └── rejected.json
│
├── tickets/                   # Bilete generate
│   ├── today/                 # Bilete azi (per loterie)
│   └── results/               # Rezultate verificări
│
├── incidents/                 # Incidente raportate
└── logs/                      # Loguri per agent
```

## Configurare

Editează `agents/config.json`:

```json
{
  "system": {
    "max_simultaneous_lotteries": 2,
    "heartbeat_interval_seconds": 30
  },
  "agents": {
    "strategy_researcher": {
      "enabled": true,
      "schedule": "0 */6 * * *"
    },
    "lottery_manager": {
      "enabled": true,
      "max_cpu_percent": 80,
      "max_memory_percent": 75
    },
    "ticket_generator": {
      "enabled": true,
      "tickets_per_lottery": 5,
      "consecutive_losses_threshold": 7
    }
  }
}
```

## Comunicare între agenți

Agenții comunică prin fișiere JSON atomice:

1. **Heartbeat**: Fiecare agent scrie în `agents/status/{agent_id}.json` la fiecare 30 secunde
2. **Mesaje**: Agenții citesc/scriu în `agents/messages/commands/to_{agent}.json`
3. **Task-uri**: Lottery Manager gestionează task-urile în `agents/tasks/`

### Exemplu mesaj:

```json
{
  "id": "uuid-1234",
  "from": "system_monitor",
  "to": "lottery_manager",
  "type": "reduce_load",
  "payload": {"reason": "high_memory", "severity": "high"},
  "timestamp": "2026-04-09T10:30:00Z"
}
```

## Troubleshooting

### Agent nu pornește

```bash
# Verifică logurile
type agents/logs/system_monitor.log

# Pornește manual pentru debug
python agents/system_monitor.py
```

### Lock files stale

```bash
# System Monitor le curăță automat, sau manual:
python agents/agent_cli.py stop all --force
```

### Fișiere JSON corupte

System Monitor le detectează și le repară automat din `.recovery` files.

## Integrare cu sistemul existent

Agenții folosesc infrastructura existentă:
- `atomic_io.py` - Operațiuni atomice cu fișiere
- `telegram_bot.py` - Notificări
- `lottery_config.py` - Configurări loterii
- `generic_backtest.py` - Testare strategii
- `auto_loop.py` - Rulare experimente

## Monitorizare

System Monitor raportează automat pe Telegram:
- Agenți care nu mai răspund (auto-restart)
- Spațiu disc insuficient
- Resurse sistem critice
- Incidente auto-rezolvate

## Dezvoltare

### Adăugare agent nou

1. Creează `agents/my_agent.py` care moștenește `BaseAgent`
2. Implementează metoda `process()`
3. Adaugă în `AGENT_SCHEDULE` dacă e on-demand
4. Adaugă în `AGENT_SCRIPTS` în `agent_cli.py`

### Exemplu agent minimal:

```python
from agent_base import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__("my_agent", poll_interval=60)

    def process(self):
        # Logica ta aici
        self.log_info("Working...")
        self.update_heartbeat("running", progress_percent=50)

if __name__ == "__main__":
    agent = MyAgent()
    agent.run()
```

## Suport

Pentru probleme sau întrebări, verifică:
1. Logurile în `agents/logs/`
2. Incidentele în `agents/incidents/`
3. Statusul cu `python agents/agent_cli.py status`
