"""Experiment Monitor Agent

Acest agent monitorizează progresul experimentelor auto_loop în timp real.
Citește fișierele progress.json ale fiecărei loterii și raportează statusul.

Schedule: Continuous (verifică la fiecare 30 secunde)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from agent_base import BaseAgent, AGENTS_DIR

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from lottery_config import list_lottery_configs, get_lottery_config
from atomic_io import read_json_with_recovery

# Import our new utility functions
try:
    from agents.utils import is_server_alive, send_heartbeat, should_shutdown
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    print("Warning: agents.utils not available, using fallback methods")


class ExperimentMonitorAgent(BaseAgent):
    """Agent pentru monitorizarea experimentelor în timp real."""

    def __init__(self):
        super().__init__("experiment_monitor", poll_interval=30)
        self.experiments_cache: dict[str, dict] = {}
        self.last_check = 0
        self.server_check_counter = 0

    def _check_experiment_progress(self, lottery_id: str) -> dict | None:
        """Verifică progresul unui experiment."""
        config = get_lottery_config(lottery_id)
        progress_file = config.progress_file

        data, _ = read_json_with_recovery(progress_file)
        if not isinstance(data, dict):
            return None

        # Extrage informațiile relevante
        status = data.get("status", "idle")
        if status == "idle":
            return None

        # auto_loop scrie best-ul în best_so_far {config, average_hits} și faza în
        # loop_info; cheile plate rămân doar ca fallback pentru formate vechi.
        best = data.get("best_so_far") if isinstance(data.get("best_so_far"), dict) else {}
        loop_info = data.get("loop_info") if isinstance(data.get("loop_info"), dict) else {}
        best_metric = best.get("average_hits", data.get("best_metric"))
        if isinstance(best_metric, bool) or not isinstance(best_metric, (int, float)):
            best_metric = None

        return {
            "lottery": lottery_id,
            "status": status,
            "experiment_id": data.get("experiment_id"),
            "tested": data.get("tested", 0),
            "total": data.get("total", 0),
            "best_metric": best_metric,
            "best_config": best.get("config", data.get("best_config")),
            "eta_seconds": data.get("eta_seconds"),
            "elapsed_seconds": data.get("elapsed_seconds"),
            "search_phase": loop_info.get("search_phase", data.get("search_phase")),
            "timestamp": data.get("updated_at", data.get("timestamp"))
        }

    def _check_running_processes(self, lottery_id: str) -> bool:
        """Verifică dacă există un proces auto_loop activ pentru loterie."""
        config = get_lottery_config(lottery_id)

        # Verifică lock file
        lock_file = getattr(config, "lock_file", None)
        if lock_file and lock_file.exists():
            try:
                from lock_utils import lock_matches_process, read_lock_owner
                owner = read_lock_owner(lock_file)
                if owner.pid:
                    # Verifică dacă procesul există ȘI e auto_loop-ul nostru —
                    # Windows reciclează PID-urile, un PID viu nu ajunge.
                    try:
                        if sys.platform == "win32":
                            import ctypes
                            process = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(owner.pid))
                            if not process:
                                return False
                            ctypes.windll.kernel32.CloseHandle(process)
                        else:
                            os.kill(int(owner.pid), 0)
                        return lock_matches_process(
                            owner,
                            script_path=PROJECT_DIR / "auto_loop.py",
                            project_dir=PROJECT_DIR,
                        )
                    except (OSError, ProcessLookupError):
                        pass
            except Exception:
                pass
        return False

    def _get_all_experiments(self) -> list[dict]:
        """Obține toate experimentele active."""
        experiments = []

        for config in list_lottery_configs():
            lottery_id = config.id

            # Verifică dacă există un experiment activ
            progress = self._check_experiment_progress(lottery_id)
            is_running = self._check_running_processes(lottery_id)

            # Doar procesele vii contează ca active — progresul învechit
            # (procese ucise / mașină adormită) nu mai umflă contorul.
            if is_running:
                exp_data = progress or {
                    "lottery": lottery_id,
                    "status": "running",
                    "tested": 0,
                    "total": 0,
                    "best_metric": None
                }
                exp_data["process_running"] = True
                experiments.append(exp_data)

        # Sortează descrescător după best_metric (cele mai bune primele)
        experiments.sort(key=lambda x: x.get("best_metric") or 0, reverse=True)

        return experiments

    def _save_experiments_summary(self, experiments: list[dict]) -> None:
        """Salvează rezumatul experimentelor."""
        summary_file = AGENTS_DIR / "experiments_summary.json"

        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "active_count": len(experiments),
            "experiments": experiments
        }

        try:
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            self.log_error(f"Failed to save experiments summary: {e}")

    def _send_notifications(self, experiments: list[dict]) -> None:
        """Trimite notificări pentru evenimente importante."""
        for exp in experiments:
            lottery_id = exp.get("lottery")
            best_metric = exp.get("best_metric")

            # best_so_far e maximul iterației curente (nu best-ul global) — doar log,
            # fără Telegram: auto_loop trimite deja notificarea KEEP cu verdictul statistic.
            # Prima observare după pornire nu e „nou”, ci doar starea curentă.
            previous = self.experiments_cache.get(lottery_id)
            if (previous and best_metric
                    and previous.get("experiment_id") == exp.get("experiment_id")
                    and best_metric > (previous.get("best_metric") or 0)):
                self.log_success(f"New iteration best for {lottery_id}: {best_metric:.6f}")

        # Update cache
        self.experiments_cache = {e.get("lottery"): e for e in experiments if e.get("lottery")}

    def process(self) -> None:
        """Procesare principală."""
        current_time = time.time()

        # Verifică la fiecare 30 secunde
        if current_time - self.last_check < 30:
            return

        self.last_check = current_time

        # Periodic server health check (every 5 minutes)
        self.server_check_counter += 1
        if self.server_check_counter >= 10:  # 10 * 30 seconds = 5 minutes
            self.server_check_counter = 0

            # Check if server is still alive using our utility function
            if UTILS_AVAILABLE:
                if should_shutdown():
                    self.log_warning("Server appears to be down, shutting down agent...")
                    self.running = False
                    return
            else:
                # Fallback method
                try:
                    server_pid_file = AGENTS_DIR / ".server_pid"
                    if server_pid_file.exists():
                        with open(server_pid_file, "r") as f:
                            pid = int(f.read().strip())

                        # Check if process exists
                        try:
                            if sys.platform == "win32":
                                import subprocess
                                result = subprocess.run(
                                    ["tasklist", "/fi", f"PID eq {pid}"],
                                    capture_output=True,
                                    text=True,
                                    shell=True
                                )
                                if f" {pid} " not in result.stdout:
                                    self.log_warning("Server appears to be down, shutting down agent...")
                                    self.running = False
                                    return
                            else:
                                os.kill(pid, 0)
                        except (OSError, ProcessLookupError):
                            self.log_warning("Server appears to be down, shutting down agent...")
                            self.running = False
                            return
                except Exception:
                    # If we can't check, continue running
                    pass

        # Obține toate experimentele
        experiments = self._get_all_experiments()

        # Salvează rezumatul
        self._save_experiments_summary(experiments)

        # Trimite notificări
        self._send_notifications(experiments)

        # Update heartbeat
        active_count = len(experiments)
        best_overall = max([e.get("best_metric") or 0 for e in experiments]) if experiments else 0

        if best_overall is not None and isinstance(best_overall, (int, float)) and not isinstance(best_overall, bool):
            best_display = f"{float(best_overall):.4f}"
        else:
            best_display = "0.0000"
        self.update_heartbeat(
            "running",
            current_task=f"monitoring {active_count} experiments, best: {best_display}",
            progress_percent=100
        )

        # Log periodic
        if int(current_time) % 300 < 30:  # La fiecare 5 minute
            self.log_info(f"Monitoring {active_count} active experiments")
            for exp in experiments[:3]:  # Primele 3
                lottery = exp.get("lottery")
                tested = exp.get("tested", 0)
                best = exp.get("best_metric")
                if best is not None and isinstance(best, (int, float)) and not isinstance(best, bool):
                    self.log_info(f"  {lottery}: tested={tested}, best={float(best):.4f}")
                else:
                    self.log_info(f"  {lottery}: tested={tested}, best=N/A")

        # Send heartbeat using our utility function
        if UTILS_AVAILABLE:
            try:
                send_heartbeat("experiment_monitor", {
                    "active_experiments": active_count,
                    "best_metric": float(best_overall) if best_overall else None
                })
            except Exception as e:
                self.log_error(f"Failed to send heartbeat: {e}")


def main():
    """Pornește Experiment Monitor Agent."""
    agent = ExperimentMonitorAgent()
    agent.run()


if __name__ == "__main__":
    main()
