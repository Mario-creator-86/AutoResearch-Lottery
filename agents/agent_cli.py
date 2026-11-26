#!/usr/bin/env python3
"""CLI pentru managementul Multi-Agent Lottery System.

Comenzi disponibile:
    python agent_cli.py start <agent_name>    - Pornește un agent
    python agent_cli.py stop <agent_name>     - Oprește un agent
    python agent_cli.py status                 - Status toți agenții
    python agent_cli.py logs <agent_name>     - Afișează logurile
    python agent_cli.py trigger <action>      - Declanșează acțiune

Agenți disponibili:
    - strategy_researcher   (on-demand, la fiecare 6 ore)
    - lottery_manager       (continuu)
    - ticket_generator      (on-demand, zilnic la 8:00)
    - system_monitor        (continuu)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Import agent_base pentru constante
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "agents"))

from agent_base import AGENTS_DIR, STATUS_DIR, LOGS_DIR


AGENT_SCRIPTS = {
    "strategy_researcher": "strategy_researcher.py",
    "lottery_manager": "lottery_manager.py",
    "ticket_generator": "ticket_generator.py",
    "system_monitor": "system_monitor.py"
}

CONTINUOUS_AGENTS = ["lottery_manager", "system_monitor"]


def _get_agent_pid(agent_id: str) -> int | None:
    """Obține PID-ul unui agent dacă rulează."""
    status_file = STATUS_DIR / f"{agent_id}.json"

    if not status_file.exists():
        return None

    try:
        with open(status_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        pid = data.get("pid")
        if pid:
            # Verifică dacă procesul există
            try:
                if sys.platform == "win32":
                    import ctypes
                    process = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
                    if process:
                        ctypes.windll.kernel32.CloseHandle(process)
                        return pid
                else:
                    os.kill(pid, 0)
                    return pid
            except (OSError, ProcessLookupError):
                return None
    except Exception:
        pass

    return None


def _is_agent_running(agent_id: str) -> bool:
    """Verifică dacă un agent rulează."""
    return _get_agent_pid(agent_id) is not None


def _send_stop_signal(agent_id: str) -> bool:
    """Trimite semnal de oprire către un agent."""
    from agent_base import COMMANDS_DIR

    command_file = COMMANDS_DIR / f"to_{agent_id}.json"

    try:
        # Citește mesajele existente
        messages = []
        if command_file.exists():
            with open(command_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    messages = data

        # Adaugă mesajul de stop
        messages.append({
            "id": f"stop_{time.time()}",
            "from": "cli",
            "to": agent_id,
            "type": "stop_request",
            "payload": {},
            "timestamp": datetime.utcnow().isoformat()
        })

        # Salvează
        with open(command_file, "w", encoding="utf-8") as f:
            json.dump(messages, f)

        return True
    except Exception as e:
        print(f"Error sending stop signal: {e}", file=sys.stderr)
        return False


def _kill_agent(agent_id: str) -> bool:
    """Oprește forțat un agent."""
    pid = _get_agent_pid(agent_id)
    if not pid:
        return False

    try:
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(1, False, int(pid))  # 1 = PROCESS_TERMINATE
            if handle:
                kernel32.TerminateProcess(handle, 0)
                kernel32.CloseHandle(handle)
        else:
            import signal
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
            try:
                os.kill(pid, 0)
                os.kill(pid, signal.SIGKILL)  # Force kill
            except ProcessLookupError:
                pass

        return True
    except Exception as e:
        print(f"Error killing agent: {e}", file=sys.stderr)
        return False


def cmd_start(args) -> int:
    """Comanda start."""
    agent_id = args.agent

    if agent_id == "all":
        # Pornește toți agenții continuu + scheduler
        print("Starting all agents...")
        success_count = 0

        for aid in CONTINUOUS_AGENTS:
            if _is_agent_running(aid):
                print(f"  {aid}: already running")
            else:
                if _start_single_agent(aid):
                    print(f"  {aid}: started")
                    success_count += 1
                else:
                    print(f"  {aid}: FAILED to start")

        # Pornește scheduler-ul
        if _start_scheduler():
            print(f"  scheduler: started")
            success_count += 1

        print(f"\nStarted {success_count}/{len(CONTINUOUS_AGENTS) + 1} agents")
        return 0 if success_count > 0 else 1

    elif agent_id == "scheduler":
        return 0 if _start_scheduler() else 1

    elif agent_id in AGENT_SCRIPTS:
        if _is_agent_running(agent_id):
            print(f"Agent {agent_id} is already running (PID: {_get_agent_pid(agent_id)})")
            return 0

        if _start_single_agent(agent_id):
            print(f"Agent {agent_id} started successfully")
            return 0
        else:
            print(f"Failed to start agent {agent_id}", file=sys.stderr)
            return 1
    else:
        print(f"Unknown agent: {agent_id}", file=sys.stderr)
        print(f"Available agents: {', '.join(AGENT_SCRIPTS.keys())}, scheduler, all", file=sys.stderr)
        return 1


def _start_single_agent(agent_id: str) -> bool:
    """Pornește un singur agent."""
    script_name = AGENT_SCRIPTS.get(agent_id)
    if not script_name:
        return False

    script_path = AGENTS_DIR / script_name

    try:
        # Pornește în background
        if sys.platform == "win32":
            subprocess.Popen(
                [sys.executable, str(script_path)],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(PROJECT_DIR)
            )
        else:
            subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                cwd=str(PROJECT_DIR)
            )

        time.sleep(1)  # Așteaptă să pornească
        return _is_agent_running(agent_id)

    except Exception as e:
        print(f"Error starting agent: {e}", file=sys.stderr)
        return False


def _start_scheduler() -> bool:
    """Pornește scheduler-ul pentru agenții on-demand."""
    from agent_scheduler import main as scheduler_main
    import threading

    # Pornește scheduler-ul într-un thread separat
    def run_scheduler():
        try:
            scheduler_main()
        except Exception:
            pass

    thread = threading.Thread(target=run_scheduler, daemon=True)
    thread.start()
    return True


def cmd_stop(args) -> int:
    """Comanda stop."""
    agent_id = args.agent

    if agent_id == "all":
        print("Stopping all agents...")
        success_count = 0

        for aid in AGENT_SCRIPTS:
            if cmd_stop(argparse.Namespace(agent=aid, force=args.force)) == 0:
                success_count += 1

        print(f"\nStopped {success_count}/{len(AGENT_SCRIPTS)} agents")
        return 0

    if agent_id not in AGENT_SCRIPTS:
        print(f"Unknown agent: {agent_id}", file=sys.stderr)
        return 1

    pid = _get_agent_pid(agent_id)
    if not pid:
        print(f"Agent {agent_id} is not running")
        return 0

    print(f"Stopping {agent_id} (PID: {pid})...")

    # Încearcă graceful stop mai întâi
    if not args.force:
        _send_stop_signal(agent_id)
        print("  Sent stop signal, waiting...")
        time.sleep(3)

        if not _is_agent_running(agent_id):
            print(f"  {agent_id}: stopped gracefully")
            return 0

        print(f"  {agent_id}: graceful stop failed, forcing...")

    # Force kill
    if _kill_agent(agent_id):
        print(f"  {agent_id}: killed")
        return 0
    else:
        print(f"  {agent_id}: FAILED to stop", file=sys.stderr)
        return 1


def cmd_status(args) -> int:
    """Comanda status."""
    print("\n┌─────────────────────────────────────────────────────┐")
    print("│           AGENT STATUS                              │")
    print("├─────────────────────────────────────────────────────┤")

    for agent_id in AGENT_SCRIPTS:
        pid = _get_agent_pid(agent_id)

        if pid:
            # Citește status
            status_file = STATUS_DIR / f"{agent_id}.json"
            try:
                with open(status_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                status = data.get("status", "unknown")
                task = data.get("current_task", "N/A")
                progress = data.get("progress_percent", 0)

                print(f"│  ✓ {agent_id:<20} [{status:<10}] PID:{pid:<6} │")
                print(f"│     Task: {task[:40]:<40} │")
                print(f"│     Progress: {progress}%")
            except Exception:
                print(f"│  ✓ {agent_id:<20} [running]   PID:{pid:<6} │")
        else:
            print(f"│  ✗ {agent_id:<20} [stopped]             │")

    print("└─────────────────────────────────────────────────────┘")
    print()
    return 0


def cmd_logs(args) -> int:
    """Comanda logs."""
    agent_id = args.agent
    lines = args.lines if hasattr(args, "lines") else 50

    if agent_id not in AGENT_SCRIPTS:
        print(f"Unknown agent: {agent_id}", file=sys.stderr)
        return 1

    log_file = LOGS_DIR / f"{agent_id}.log"

    if not log_file.exists():
        print(f"No log file found for {agent_id}")
        return 0

    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
            # Afișează ultimele N linii
            for line in all_lines[-lines:]:
                print(line.rstrip())
    except Exception as e:
        print(f"Error reading logs: {e}", file=sys.stderr)
        return 1

    return 0


def cmd_incidents(args) -> int:
    """Comanda incidents."""
    from agent_base import INCIDENTS_DIR

    incidents = sorted(INCIDENTS_DIR.glob("*.json"), reverse=True)

    if not incidents:
        print("No incidents reported")
        return 0

    print(f"\nRecent incidents (showing last {args.limit}):\n")

    for incident_file in incidents[:args.limit]:
        try:
            with open(incident_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            timestamp = data.get("timestamp", "unknown")
            agent = data.get("agent_id", "unknown")
            incident_type = data.get("type", "unknown")
            severity = data.get("severity", "unknown")
            description = data.get("description", "No description")
            auto_resolved = data.get("auto_resolved", False)

            severity_icon = "🔴" if severity == "critical" else "🟠" if severity == "high" else "🟡"
            resolved_icon = "✓" if auto_resolved else "⚠"

            print(f"{severity_icon} [{timestamp}] {agent} - {incident_type}")
            print(f"   {description[:60]}")
            print(f"   Severity: {severity} | Auto-resolved: {resolved_icon}")
            print()
        except Exception:
            continue

    return 0


def cmd_trigger(args) -> int:
    """Comanda trigger - declanșează acțiuni specifice."""
    from agent_base import COMMANDS_DIR

    action = args.action

    if action == "research":
        # Declanșează research imediat
        command_file = COMMANDS_DIR / "to_strategy_researcher.json"
        message = {
            "id": f"trigger_{time.time()}",
            "from": "cli",
            "to": "strategy_researcher",
            "type": "manual_trigger",
            "payload": {"action": "research_now"},
            "timestamp": datetime.utcnow().isoformat()
        }

        try:
            messages = []
            if command_file.exists():
                with open(command_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        messages = data
            messages.append(message)

            with open(command_file, "w", encoding="utf-8") as f:
                json.dump(messages, f)

            print("Triggered strategy research")
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    elif action == "generate_tickets":
        # Declanșează generarea biletelor
        lottery = args.lottery if hasattr(args, "lottery") else None

        command_file = COMMANDS_DIR / "to_ticket_generator.json"
        message = {
            "id": f"trigger_{time.time()}",
            "from": "cli",
            "to": "ticket_generator",
            "type": "manual_trigger",
            "payload": {"action": "generate_now", "lottery": lottery},
            "timestamp": datetime.utcnow().isoformat()
        }

        try:
            messages = []
            if command_file.exists():
                with open(command_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        messages = data
            messages.append(message)

            with open(command_file, "w", encoding="utf-8") as f:
                json.dump(messages, f)

            print(f"Triggered ticket generation{f' for {lottery}' if lottery else ''}")
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    else:
        print(f"Unknown action: {action}", file=sys.stderr)
        print("Available actions: research, generate_tickets", file=sys.stderr)
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CLI for Multi-Agent Lottery System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python agent_cli.py start monitor              # Start system monitor
    python agent_cli.py start all                  # Start all agents
    python agent_cli.py stop strategy_researcher   # Stop research agent
    python agent_cli.py status                     # Show all agents status
    python agent_cli.py logs system_monitor         # Show monitor logs
    python agent_cli.py incidents                   # Show recent incidents
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start an agent")
    start_parser.add_argument("agent", help="Agent name, 'all', or 'scheduler'")

    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop an agent")
    stop_parser.add_argument("agent", help="Agent name or 'all'")
    stop_parser.add_argument("--force", "-f", action="store_true",
                             help="Force kill the agent")

    # Status command
    subparsers.add_parser("status", help="Show agent status")

    # Logs command
    logs_parser = subparsers.add_parser("logs", help="Show agent logs")
    logs_parser.add_argument("agent", help="Agent name")
    logs_parser.add_argument("--lines", "-n", type=int, default=50,
                             help="Number of lines to show")

    # Incidents command
    incidents_parser = subparsers.add_parser("incidents", help="Show recent incidents")
    incidents_parser.add_argument("--limit", "-l", type=int, default=10,
                                  help="Number of incidents to show")

    # Trigger command
    trigger_parser = subparsers.add_parser("trigger", help="Trigger specific actions")
    trigger_parser.add_argument("action", help="Action name (research, generate_tickets)")
    trigger_parser.add_argument("--lottery", help="Lottery ID (for generate_tickets)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Dispatch command
    commands = {
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
        "logs": cmd_logs,
        "incidents": cmd_incidents,
        "trigger": cmd_trigger
    }

    if args.command in commands:
        return commands[args.command](args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
