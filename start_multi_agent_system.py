#!/usr/bin/env python3
"""
Multi-Agent System Launcher

Script pentru pornirea automată a întregului sistem multi-agent.
Pornește în ordine:
1. System Monitor (continuu)
2. Lottery Manager (continuu)
3. Scheduler (pentru agenții on-demand)

Usage:
    python start_multi_agent_system.py        # Pornește tot sistemul
    python start_multi_agent_system.py --bg   # Pornește în background
    python start_multi_agent_system.py --dry-run # Doar verifică, nu pornește
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
AGENTS_DIR = PROJECT_DIR / "agents"

def check_python_version():
    """Verifică versiunea Python."""
    if sys.version_info < (3, 10):
        print("Error: Python 3.10+ required", file=sys.stderr)
        return False
    return True

def check_dependencies():
    """Verifică dependențele necesare."""
    required = ["atomic_io.py", "lottery_config.py", "telegram_bot.py"]
    missing = []

    for dep in required:
        if not (PROJECT_DIR / dep).exists():
            missing.append(dep)

    if missing:
        print(f"Error: Missing dependencies: {', '.join(missing)}", file=sys.stderr)
        return False
    return True

def start_agent(agent_id: str, background: bool = True) -> subprocess.Popen | None:
    """Pornește un agent."""
    script_map = {
        "system_monitor": "system_monitor.py",
        "lottery_manager": "lottery_manager.py",
        "scheduler": "agent_scheduler.py"
    }

    script_name = script_map.get(agent_id)
    if not script_name:
        print(f"Error: Unknown agent {agent_id}", file=sys.stderr)
        return None

    script_path = AGENTS_DIR / script_name
    if not script_path.exists():
        print(f"Error: Script not found: {script_path}", file=sys.stderr)
        return None

    try:
        if sys.platform == "win32":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if background else 0
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                creationflags=flags,
                stdout=subprocess.DEVNULL if background else None,
                stderr=subprocess.DEVNULL if background else None,
                cwd=str(PROJECT_DIR)
            )
        else:
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.DEVNULL if background else None,
                stderr=subprocess.DEVNULL if background else None,
                start_new_session=background,
                cwd=str(PROJECT_DIR)
            )

        return proc
    except Exception as e:
        print(f"Error starting {agent_id}: {e}", file=sys.stderr)
        return None

def check_agent_running(agent_id: str) -> bool:
    """Verifică dacă un agent rulează deja."""
    status_file = AGENTS_DIR / "status" / f"{agent_id}.json"

    if not status_file.exists():
        return False

    try:
        import json
        with open(status_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        pid = data.get("pid")
        if not pid:
            return False

        # Verifică dacă procesul există
        if sys.platform == "win32":
            import ctypes
            process = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
            if process:
                ctypes.windll.kernel32.CloseHandle(process)
                return True
            return False
        else:
            try:
                os.kill(pid, 0)
                return True
            except (OSError, ProcessLookupError):
                return False
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(description="Start Multi-Agent Lottery System")
    parser.add_argument("--bg", action="store_true", help="Run agents in background")
    parser.add_argument("--dry-run", action="store_true", help="Check only, don't start")
    args = parser.parse_args()

    print("=" * 60)
    print("Multi-Agent Lottery System Launcher")
    print("=" * 60)
    print()

    # Verificări preliminare
    if not check_python_version():
        return 1

    if not check_dependencies():
        return 1

    print("✓ Pre-flight checks passed")
    print()

    if args.dry_run:
        print("Dry run mode - not starting any agents")
        return 0

    # Agenții de pornit (în ordine)
    agents_to_start = [
        ("system_monitor", "System Monitor (24/7 debugging)"),
        ("lottery_manager", "Lottery Manager (load balancing)"),
        ("scheduler", "Scheduler (cron jobs)"),
    ]

    started = []
    already_running = []
    failed = []

    for agent_id, description in agents_to_start:
        print(f"Starting {agent_id}...")
        print(f"  Description: {description}")

        if check_agent_running(agent_id):
            print(f"  Status: Already running ✓")
            already_running.append(agent_id)
        else:
            proc = start_agent(agent_id, background=args.bg)
            if proc:
                print(f"  Status: Started (PID: {proc.pid}) ✓")
                started.append(agent_id)
                time.sleep(1)  # Așteaptă să se inițializeze
            else:
                print(f"  Status: Failed ✗")
                failed.append(agent_id)
        print()

    # Raport final
    print("=" * 60)
    print("STARTUP REPORT")
    print("=" * 60)
    print(f"Started: {len(started)}/{len(agents_to_start)}")
    print(f"Already running: {len(already_running)}/{len(agents_to_start)}")
    print(f"Failed: {len(failed)}/{len(agents_to_start)}")
    print()

    if started:
        print("Started agents:", ", ".join(started))
    if already_running:
        print("Already running:", ", ".join(already_running))
    if failed:
        print("Failed to start:", ", ".join(failed), file=sys.stderr)

    print()
    print("Commands:")
    print(f"  python {AGENTS_DIR / 'agent_cli.py'} status     # Check status")
    print(f"  python {AGENTS_DIR / 'agent_cli.py'} logs monitor # View monitor logs")
    print(f"  python {AGENTS_DIR / 'agent_cli.py'} stop all    # Stop all agents")
    print()

    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(main())
