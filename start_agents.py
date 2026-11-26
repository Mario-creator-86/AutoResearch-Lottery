#!/usr/bin/env python3
"""
Start Multi-Agent Lottery System

This script starts all agents in the correct order:
1. System Monitor (continuous)
2. Lottery Manager (continuous)
3. Experiment Monitor (continuous) - NEW
4. Scheduler (for on-demand agents)

Usage:
    python start_agents.py        # Start all agents
    python start_agents.py --stop # Stop all agents
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
AGENTS_DIR = PROJECT_DIR / "agents"


def start_agents():
    """Start all continuous agents."""
    print(">> Starting Multi-Agent Lottery System...\n")

    # Start System Monitor first
    print("1. Starting System Monitor (continuous)...")
    monitor_proc = subprocess.Popen(
        [sys.executable, str(AGENTS_DIR / "system_monitor.py")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        cwd=str(PROJECT_DIR)
    )
    time.sleep(2)
    print(f"   [OK] System Monitor started (PID: {monitor_proc.pid})\n")

    # Start Lottery Manager
    print("2. Starting Lottery Manager (continuous)...")
    manager_proc = subprocess.Popen(
        [sys.executable, str(AGENTS_DIR / "lottery_manager.py")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        cwd=str(PROJECT_DIR)
    )
    time.sleep(2)
    print(f"   [OK] Lottery Manager started (PID: {manager_proc.pid})\n")

    # Start Experiment Monitor (NEW)
    print("3. Starting Experiment Monitor (continuous)...")
    exp_proc = subprocess.Popen(
        [sys.executable, str(AGENTS_DIR / "experiment_monitor.py")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        cwd=str(PROJECT_DIR)
    )
    time.sleep(1)
    print(f"   [OK] Experiment Monitor started (PID: {exp_proc.pid})\n")

    # Start Scheduler
    print("4. Starting Scheduler (for on-demand agents)...")
    scheduler_proc = subprocess.Popen(
        [sys.executable, str(AGENTS_DIR / "agent_scheduler.py")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        cwd=str(PROJECT_DIR)
    )
    time.sleep(1)
    print(f"   [OK] Scheduler started (PID: {scheduler_proc.pid})\n")

    print("=" * 50)
    print("All agents started successfully!")
    print()
    print("Agents:")
    print("  - System Monitor    (continuous)")
    print("  - Lottery Manager   (continuous)")
    print("  - Experiment Monitor (continuous)")
    print("  - Scheduler         (on-demand tasks)")
    print()
    print("Commands:")
    print("  python agents/agent_cli.py status      # Check status")
    print("  python agents/agent_cli.py logs monitor  # View monitor logs")
    print("  python start_agents.py --stop          # Stop all agents")
    print("=" * 50)


def stop_agents():
    """Stop all agents gracefully."""
    print(">> Stopping Multi-Agent Lottery System...\n")

    # Stop all agents via CLI
    agents = ["system_monitor", "lottery_manager", "experiment_monitor",
              "strategy_researcher", "ticket_generator", "scheduler"]

    for agent in agents:
        try:
            print(f"Stopping {agent}...")
            subprocess.run(
                [sys.executable, str(AGENTS_DIR / "agent_cli.py"), "stop", agent],
                capture_output=True,
                timeout=5
            )
        except Exception:
            pass

    print("\nAll agents stopped.")


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Lottery System Control")
    parser.add_argument("--stop", action="store_true", help="Stop all agents")
    parser.add_argument("--restart", action="store_true", help="Restart all agents")
    args = parser.parse_args()

    if args.stop:
        stop_agents()
    elif args.restart:
        stop_agents()
        time.sleep(2)
        start_agents()
    else:
        start_agents()


if __name__ == "__main__":
    main()
