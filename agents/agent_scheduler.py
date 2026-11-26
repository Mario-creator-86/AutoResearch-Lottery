"""Agent Scheduler

Scheduler pentru agenții care rulează on-demand.
Gestionează pornirea agenților la orele programate.

Schedule:
    - strategy_researcher: La fiecare 6 ore (00:00, 06:00, 12:00, 18:00)
    - ticket_generator: Zilnic la 8:00 AM
    - ticket_checker: Zilnic după extrageri (21:00)

Agenți continui (nu sunt gestionați de scheduler):
    - lottery_manager
    - system_monitor
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "agents"))

AGENTS_DIR = PROJECT_DIR / "agents"


# Configurație schedule (format cron-like)
AGENT_SCHEDULE = [
    {
        "agent_id": "strategy_researcher",
        "cron": "0 */6 * * *",  # La fiecare 6 ore
        "enabled": True,
        "description": "Research new strategies online"
    },
    {
        "agent_id": "ticket_generator",
        "cron": "0 8 * * *",  # Zilnic la 8:00 AM
        "enabled": True,
        "description": "Generate daily tickets"
    },
    {
        "agent_id": "ticket_checker",
        "cron": "0 21 * * *",  # Zilnic la 21:00
        "enabled": True,
        "description": "Check yesterday's ticket results"
    }
]

AGENT_SCRIPTS = {
    "strategy_researcher": "strategy_researcher.py",
    "ticket_generator": "ticket_generator.py",
    "ticket_checker": "ticket_generator.py"  # Ticket generator poate verifica și rezultate
}


class CronParser:
    """Parser simplu pentru expresii cron."""

    @staticmethod
    def parse(cron_expr: str) -> dict:
        """Parsează o expresie cron și returnează componentele."""
        parts = cron_expr.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {cron_expr}")

        return {
            "minute": CronParser._parse_field(parts[0], 0, 59),
            "hour": CronParser._parse_field(parts[1], 0, 23),
            "day": CronParser._parse_field(parts[2], 1, 31),
            "month": CronParser._parse_field(parts[3], 1, 12),
            "weekday": CronParser._parse_field(parts[4], 0, 6)
        }

    @staticmethod
    def _parse_field(field: str, min_val: int, max_val: int) -> list:
        """Parsează un câmp cron."""
        values = []

        if field == "*":
            return list(range(min_val, max_val + 1))

        if "/" in field:
            # Step values: */6 sau 0-23/6
            base, step = field.split("/")
            step = int(step)
            if base == "*":
                values = list(range(min_val, max_val + 1, step))
            else:
                start, end = map(int, base.split("-"))
                values = list(range(start, end + 1, step))
        elif "," in field:
            # List values: 1,2,3
            values = [int(x) for x in field.split(",")]
        elif "-" in field:
            # Range: 1-5
            start, end = map(int, field.split("-"))
            values = list(range(start, end + 1))
        else:
            # Single value
            values = [int(field)]

        return [v for v in values if min_val <= v <= max_val]

    @staticmethod
    def should_run(cron_expr: str, dt: datetime | None = None) -> bool:
        """Verifică dacă ar trebui să ruleze la momentul dat."""
        if dt is None:
            dt = datetime.now()

        try:
            parsed = CronParser.parse(cron_expr)

            # Verifică fiecare componentă
            if dt.minute not in parsed["minute"]:
                return False
            if dt.hour not in parsed["hour"]:
                return False
            if dt.day not in parsed["day"]:
                return False
            if dt.month not in parsed["month"]:
                return False
            if dt.weekday() not in parsed["weekday"]:
                return False

            return True
        except Exception:
            return False


class AgentScheduler:
    """Scheduler pentru agenți on-demand."""

    def __init__(self):
        self.schedule = AGENT_SCHEDULE
        self.running_agents: dict[str, subprocess.Popen] = {}
        self.last_run: dict[str, datetime] = {}
        self.check_interval = 60  # Verifică la fiecare minut

    def _is_agent_running(self, agent_id: str) -> bool:
        """Verifică dacă un agent rulează deja."""
        if agent_id in self.running_agents:
            proc = self.running_agents[agent_id]
            if proc.poll() is None:
                return True
            else:
                # Procesul s-a terminat, curăță
                del self.running_agents[agent_id]

        # Verifică și în status file
        try:
            status_file = AGENTS_DIR / "status" / f"{agent_id}.json"
            if status_file.exists():
                with open(status_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                last_activity = data.get("last_activity")
                if last_activity:
                    last_time = datetime.fromisoformat(last_activity)
                    if (datetime.utcnow() - last_time).total_seconds() < 300:  # 5 minute
                        return True
        except Exception:
            pass

        return False

    def _start_agent(self, agent_id: str) -> bool:
        """Pornește un agent."""
        script_name = AGENT_SCRIPTS.get(agent_id)
        if not script_name:
            print(f"[scheduler] Unknown agent: {agent_id}")
            return False

        script_path = AGENTS_DIR / script_name

        try:
            print(f"[scheduler] Starting {agent_id}...")

            # Pornește agentul
            if sys.platform == "win32":
                proc = subprocess.Popen(
                    [sys.executable, str(script_path)],
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    cwd=str(PROJECT_DIR)
                )
            else:
                proc = subprocess.Popen(
                    [sys.executable, str(script_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    cwd=str(PROJECT_DIR)
                )

            self.running_agents[agent_id] = proc
            self.last_run[agent_id] = datetime.now()

            print(f"[scheduler] {agent_id} started (PID: {proc.pid})")
            return True

        except Exception as e:
            print(f"[scheduler] Failed to start {agent_id}: {e}")
            return False

    def _cleanup_finished(self) -> None:
        """Curăță agenții care s-au terminat."""
        finished = []
        for agent_id, proc in self.running_agents.items():
            if proc.poll() is not None:
                finished.append(agent_id)
                print(f"[scheduler] {agent_id} finished (return code: {proc.returncode})")

        for agent_id in finished:
            del self.running_agents[agent_id]

    def _should_run_now(self, agent_id: str, cron_expr: str) -> bool:
        """Verifică dacă agentul ar trebui să ruleze acum."""
        now = datetime.now()

        # Verifică expresia cron
        if not CronParser.should_run(cron_expr, now):
            return False

        # Verifică dacă nu a rulat deja în ultimul minut
        last_run = self.last_run.get(agent_id)
        if last_run:
            time_since_last = (now - last_run).total_seconds()
            if time_since_last < 60:  # Nu rula de mai multe ori pe minut
                return False

        return True

    def run(self) -> None:
        """Loop principal al scheduler-ului."""
        print("[scheduler] Agent Scheduler started")
        print(f"[scheduler] Managing {len(self.schedule)} scheduled agents")
        print("[scheduler] Press Ctrl+C to stop\n")

        running = True
        while running:
            try:
                # Curăță agenți terminați
                self._cleanup_finished()

                # Verifică fiecare agent programat
                for schedule_item in self.schedule:
                    agent_id = schedule_item["agent_id"]
                    cron_expr = schedule_item["cron"]
                    enabled = schedule_item.get("enabled", True)

                    if not enabled:
                        continue

                    # Verifică dacă ar trebui să ruleze
                    if self._should_run_now(agent_id, cron_expr):
                        if not self._is_agent_running(agent_id):
                            self._start_agent(agent_id)
                        else:
                            print(f"[scheduler] {agent_id} already running, skipping")

                # Așteaptă următoarea verificare
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                print("\n[scheduler] Stopping...")
                running = False
            except Exception as e:
                print(f"[scheduler] Error: {e}")
                time.sleep(self.check_interval)

        print("[scheduler] Scheduler stopped")


def main():
    """Entry point."""
    scheduler = AgentScheduler()
    scheduler.run()


if __name__ == "__main__":
    main()
