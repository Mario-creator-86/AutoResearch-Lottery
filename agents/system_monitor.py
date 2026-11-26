"""System Monitor & Debugger Agent

Acest agent rulează continuu și:
- Monitorizează toți ceilalți agenți prin fișierele heartbeat
- Detectează crash-uri și îi restartează automat
- Omoară procese zombie
- Repară fișiere JSON corupte
- Trimite alerte pe Telegram
- Urmărește logurile pentru erori și aplică fixuri cunoscute
"""

from __future__ import annotations

import ctypes
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent_base import BaseAgent, AGENTS_DIR, STATUS_DIR, INCIDENTS_DIR, LOGS_DIR, parse_utc_timestamp

# Import existing infrastructure
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from atomic_io import read_json_with_recovery, write_json_atomic

try:
    from telegram_bot import send_message as send_telegram_message
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False


class SystemMonitorAgent(BaseAgent):
    """Agent pentru monitorizare și debugging continuu."""

    # Toți ceilalți agenți: blocajele se pot întâmpla oricui (apeluri Ollama/rețea).
    MONITORED_AGENTS = [
        "strategy_researcher",
        "lottery_manager",
        "ticket_generator",
        "experiment_monitor",
        "strategy_optimizer",
        "strategy_learner",
        "ollama_strategist",
        "ai_strategy_coder",
        "ai_ticket_agent",
        "telegram_agent",
    ]

    STARTUP_GRACE_SECONDS = 180

    KNOWN_ERRORS = {
        "MemoryError": {
            "action": "reduce_load",
            "message": "Memorie insuficientă, reduc experimentele"
        },
        "JSONDecodeError": {
            "action": "restore_backup",
            "message": "Fișier JSON corupt, restaurez din backup"
        },
        "PermissionError": {
            "action": "fix_permissions",
            "message": "Eroare permisiuni, încerc să remediez"
        },
        "LockTimeout": {
            "action": "force_unlock",
            "message": "Lock blocat, încerc eliberarea forțată"
        }
    }

    def __init__(self):
        super().__init__("system_monitor", poll_interval=30)
        self.agent_processes: dict[str, subprocess.Popen] = {}
        self.stale_threshold = 120  # seconds
        self.hang_threshold = 1800  # seconds; minim înainte de a considera un agent viu ca blocat
        self.stale_warned: set[str] = set()
        self.down_warned: set[str] = set()
        self.last_incident_prune = 0.0
        self.started_at = time.time()
        self.last_log_check = 0
        self.log_check_interval = 30
        self.known_log_entries: set[str] = set()

    def _is_windows(self) -> bool:
        return sys.platform == "win32"

    def _pid_alive(self, pid: int) -> bool:
        """Verifică dacă un proces este activ."""
        if self._is_windows():
            try:
                process = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
                if process:
                    ctypes.windll.kernel32.CloseHandle(process)
                    return True
                return False
            except Exception:
                return False
        else:
            try:
                os.kill(pid, 0)
                return True
            except (OSError, ProcessLookupError):
                return False

    def _get_process_cpu_memory(self, pid: int) -> tuple[float, float]:
        """Obține CPU și memorie pentru un proces."""
        try:
            if self._is_windows():
                import psutil
                proc = psutil.Process(pid)
                return proc.cpu_percent(interval=0.1), proc.memory_percent()
            else:
                import psutil
                proc = psutil.Process(pid)
                return proc.cpu_percent(interval=0.1), proc.memory_percent()
        except Exception:
            return 0.0, 0.0

    def _restart_agent(self, agent_id: str) -> bool:
        """Restartează un agent."""
        self.log_info(f"Restarting agent: {agent_id}")

        # Oprește procesul vechi dacă există
        if agent_id in self.agent_processes:
            proc = self.agent_processes[agent_id]
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

        # Pornește proces nou
        try:
            agent_file = AGENTS_DIR / f"{agent_id}.py"
            if not agent_file.exists():
                self.log_error(f"Agent file not found: {agent_file}")
                return False

            proc = subprocess.Popen(
                [sys.executable, str(agent_file)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if self._is_windows() else 0
            )
            self.agent_processes[agent_id] = proc

            self.log_success(f"Agent {agent_id} restarted (PID: {proc.pid})")
            self.report_incident(f"{agent_id}_restarted",
                                  f"Agent {agent_id} restarted automatically",
                                  severity="medium", auto_resolved=True)

            if TELEGRAM_AVAILABLE:
                try:
                    send_telegram_message(f"🔄 <b>System Monitor</b>: Agent {agent_id} a fost restartat")
                except Exception:
                    pass

            return True
        except Exception as e:
            self.log_error(f"Failed to restart {agent_id}: {e}")
            return False

    def _kill_zombie_processes(self) -> None:
        """Omoară procesele zombie."""
        killed = []
        for agent_id, proc in list(self.agent_processes.items()):
            if proc.poll() is not None:
                # Proces mort dar încă în dicționar
                del self.agent_processes[agent_id]
                killed.append(agent_id)

        if killed:
            self.log_info(f"Cleaned up zombie processes: {killed}")

    @staticmethod
    def _poll_interval(data: dict) -> float:
        try:
            return float(data.get("poll_interval") or 0)
        except (TypeError, ValueError):
            return 0.0

    def _stale_threshold_for(self, data: dict) -> float:
        # Heartbeat-ul se scrie o dată pe ciclu, deci un agent care doarme
        # poll_interval secunde nu e „stale" înainte de ~2 cicluri.
        return max(self.stale_threshold, 2 * self._poll_interval(data) + 60)

    def _hang_threshold_for(self, data: dict) -> float:
        # Peste acest prag un proces viu dar tăcut e considerat blocat (apel de
        # rețea fără timeout, deadlock) și e repornit. Generos: analizele Ollama
        # și descărcările de istoric pot dura legitim minute întregi.
        return max(self.hang_threshold, 3 * self._poll_interval(data) + 600)

    @staticmethod
    def _heartbeat_age(data: dict) -> float | None:
        last_activity = data.get("last_activity")
        if not last_activity:
            return None
        try:
            return (datetime.now(timezone.utc) - parse_utc_timestamp(last_activity)).total_seconds()
        except (TypeError, ValueError):
            return None

    def _check_heartbeat_stale(self, agent_id: str) -> bool:
        """Verifică dacă heartbeat-ul unui agent este stale."""
        data, _ = read_json_with_recovery(STATUS_DIR / f"{agent_id}.json")
        if not isinstance(data, dict):
            return True
        age = self._heartbeat_age(data)
        return age is None or age > self._stale_threshold_for(data)

    def _agent_process(self, pid, agent_id: str):
        """psutil.Process pentru PID doar dacă rulează chiar scriptul agentului.

        Windows reciclează PID-urile: după moartea agentului, același PID poate
        aparține oricărui alt program, iar OpenProcess l-ar raporta „viu" la nesfârșit.
        """
        if not pid:
            return None
        try:
            import psutil
        except ImportError:
            return True if self._pid_alive(int(pid)) else None  # fără psutil: verificarea veche
        try:
            proc = psutil.Process(int(pid))
            cmdline = " ".join(proc.cmdline())
        except (psutil.Error, ValueError, TypeError, OSError):
            return None
        return proc if f"{agent_id}.py" in cmdline else None

    def _check_agent_status(self, agent_id: str) -> dict:
        """Verifică starea completă a unui agent."""
        status_file = STATUS_DIR / f"{agent_id}.json"
        data, _ = read_json_with_recovery(status_file)

        if not isinstance(data, dict):
            return {"status": "unknown", "pid": None, "alive": False, "stale": True, "hung": False}

        pid = data.get("pid")
        process = self._agent_process(pid, agent_id)
        age = self._heartbeat_age(data)
        is_stale = age is None or age > self._stale_threshold_for(data)

        return {
            "status": data.get("status", "unknown"),
            "pid": pid,
            "alive": process is not None,
            "process": process,
            "stale": is_stale,
            "hung": process is not None and age is not None and age > self._hang_threshold_for(data),
            "heartbeat_age": age,
            "current_task": data.get("current_task"),
            "progress": data.get("progress_percent", 0)
        }

    def _prune_incidents(self, max_age_days: int = 14, max_files: int = 500,
                         incidents_dir: Path | None = None) -> int:
        """Șterge incidentele vechi; fără limită directorul ajunsese la 21.000 de fișiere."""
        directory = incidents_dir or INCIDENTS_DIR
        try:
            files = sorted((f for f in directory.glob("*.json") if f.is_file()),
                           key=lambda f: f.stat().st_mtime, reverse=True)
        except OSError:
            return 0
        cutoff = time.time() - max_age_days * 86400
        removed = 0
        for index, file in enumerate(files):
            try:
                if index >= max_files or file.stat().st_mtime < cutoff:
                    file.unlink()
                    removed += 1
            except OSError:
                continue
        if removed:
            self.log_info(f"Pruned {removed} old incident files")
        return removed

    @staticmethod
    def _server_supervises() -> bool:
        try:
            from agents.utils import is_server_responsive
            return is_server_responsive(timeout=3)
        except Exception:
            return False

    def _terminate_hung(self, agent_id: str, process) -> bool:
        """Oprește un agent blocat (proces verificat ca fiind al agentului)."""
        if process is True or process is None:
            return False  # fără psutil nu putem confirma identitatea — nu omorâm nimic
        try:
            process.terminate()
            try:
                process.wait(timeout=10)
            except Exception:
                process.kill()
                process.wait(timeout=5)
            return True
        except Exception as e:
            self.log_error(f"Failed to terminate hung agent {agent_id} (pid={process.pid}): {e}")
            return False

    def _fix_corrupted_json(self, file_path: Path) -> bool:
        """Repară fișiere JSON corupte."""
        recovery_file = file_path.with_suffix(file_path.suffix + ".recovery")

        if recovery_file.exists():
            try:
                with open(recovery_file, "r", encoding="utf-8") as f:
                    data = f.read()
                    # Verifică dacă e valid JSON
                    import json
                    json.loads(data)

                # Copiază recovery peste fișierul corupt
                import shutil
                shutil.copy(str(recovery_file), str(file_path))
                self.log_success(f"Restored {file_path.name} from recovery")
                return True
            except Exception as e:
                self.log_error(f"Failed to restore {file_path}: {e}")

        return False

    def _check_disk_space(self) -> None:
        """Verifică spațiul pe disc."""
        try:
            import shutil
            total, used, free = shutil.disk_usage("/" if not self._is_windows() else "C:\\")
            free_gb = free / (1024**3)

            if free_gb < 1:
                self.log_error(f"CRITICAL: Low disk space: {free_gb:.2f} GB free")
                self.report_incident("low_disk_space",
                                      f"Only {free_gb:.2f} GB free on disk",
                                      severity="critical")
                if TELEGRAM_AVAILABLE:
                    try:
                        send_telegram_message(f"🚨 <b>CRITICAL</b>: Spațiu disc insuficient: {free_gb:.2f} GB liber")
                    except Exception:
                        pass
            elif free_gb < 5:
                self.log_warning(f"Low disk space: {free_gb:.2f} GB free")
        except Exception as e:
            self.log_error(f"Failed to check disk space: {e}")

    def _monitor_logs(self) -> None:
        """Monitorizează fișierele log pentru erori."""
        log_files = [
            PROJECT_DIR / "server.stderr.log",
            PROJECT_DIR / "nonstop.log",
            PROJECT_DIR / "error.txt"
        ]

        for log_file in log_files:
            if not log_file.exists():
                continue

            try:
                with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                    # Citește ultimele 100 de linii
                    lines = f.readlines()[-100:]

                    for line in lines:
                        line_hash = hash(line.strip())
                        if line_hash in self.known_log_entries:
                            continue
                        self.known_log_entries.add(line_hash)

                        # Verifică erori cunoscute
                        for error_pattern, action_info in self.KNOWN_ERRORS.items():
                            if error_pattern in line:
                                self._handle_known_error(error_pattern, action_info, line)

            except Exception as e:
                self.log_error(f"Failed to monitor {log_file}: {e}")

    def _handle_known_error(self, error_pattern: str, action_info: dict,
                            log_line: str) -> None:
        """Gestionează erori cunoscute."""
        action = action_info["action"]
        message = action_info["message"]

        self.log_warning(f"Known error detected: {error_pattern}")
        self.log_info(f"Action: {message}")

        if action == "reduce_load":
            # Trimite mesaj către lottery_manager să reducă load-ul
            self.send_message("lottery_manager", "reduce_load",
                              {"reason": "memory_error", "severity": "high"})

        elif action == "restore_backup":
            # Încearcă să repare fișierele corupte
            self._scan_and_fix_json_files()

        elif action == "force_unlock":
            # Eliberează lock files
            self._release_stale_locks()

        self.report_incident(f"known_error_{error_pattern}",
                              f"Detected {error_pattern}: {log_line[:100]}",
                              severity="medium", auto_resolved=True)

    def _scan_and_fix_json_files(self) -> None:
        """Scanează și repară fișiere JSON corupte."""
        json_dirs = [STATUS_DIR, AGENTS_DIR]

        for directory in json_dirs:
            for json_file in directory.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        import json
                        json.load(f)
                except json.JSONDecodeError:
                    self.log_warning(f"Corrupted JSON detected: {json_file}")
                    if self._fix_corrupted_json(json_file):
                        self.log_success(f"Fixed corrupted JSON: {json_file}")

    def _release_stale_locks(self) -> None:
        """Eliberează lock files stale."""
        lock_files = list((PROJECT_DIR / "data").rglob("*.lock"))
        lock_files.extend([PROJECT_DIR / ".server.lock"])

        for lock_file in lock_files:
            if not lock_file.exists():
                continue

            try:
                # Verifică dacă lock-ul e stale
                from lock_utils import read_lock_owner, lock_matches_process
                owner = read_lock_owner(lock_file)

                if owner.pid and not self._pid_alive(owner.pid):
                    lock_file.unlink(missing_ok=True)
                    self.log_success(f"Released stale lock: {lock_file}")
            except Exception as e:
                self.log_error(f"Failed to release lock {lock_file}: {e}")

    def _check_system_resources(self) -> None:
        """Verifică resursele sistemului."""
        try:
            import psutil

            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                self.log_warning(f"High CPU usage: {cpu_percent}%")
                self.send_message("lottery_manager", "reduce_load",
                                  {"reason": "high_cpu", "cpu_percent": cpu_percent})

            # Memory
            memory = psutil.virtual_memory()
            # procentul include toate programele de pe calculator; contează memoria
            # rămasă liberă. Avertizăm cel mult o dată la 30 de minute.
            if memory.percent > 90 and memory.available < 2**30                     and time.time() - getattr(self, "last_memory_warning", 0.0) > 1800:
                self.last_memory_warning = time.time()
                self.log_warning(f"High memory usage: {memory.percent}% "
                                 f"({memory.available / 2**30:.1f} GB free)")
                self.send_message("lottery_manager", "reduce_load",
                                  {"reason": "high_memory", "memory_percent": memory.percent})

        except ImportError:
            pass
        except Exception as e:
            self.log_error(f"Failed to check system resources: {e}")

    def process(self) -> None:
        """Procesare principală - rulează la fiecare 30 secunde."""

        # 1. Verifică heartbeat pentru fiecare agent monitorizat
        for agent_id in self.MONITORED_AGENTS:
            status = self._check_agent_status(agent_id)

            if status["alive"] and not status["stale"]:
                self.stale_warned.discard(agent_id)
                self.down_warned.discard(agent_id)
                continue
            if status["alive"] and not status["hung"]:
                # Procesul trăiește dar heartbeat-ul e întârziat (agent ocupat).
                # Un restart ar crea un DUPLICAT — avertizează o singură dată.
                if agent_id not in self.stale_warned:
                    self.stale_warned.add(agent_id)
                    self.log_warning(
                        f"Agent {agent_id} heartbeat stale but process alive (pid={status['pid']}); not restarting"
                    )
                continue
            if status["alive"]:
                # Blocat de mult: oprim procesul verificat, apoi repornim (fără duplicat).
                age_min = (status["heartbeat_age"] or 0) / 60
                self.log_warning(f"Agent {agent_id} hung (no heartbeat for {age_min:.0f} min, pid={status['pid']})")
                if not self.get_config().get("auto_restart_agents", True):
                    continue  # fără auto-restart nu-l oprim — ar rămâne mort
                if not self._terminate_hung(agent_id, status["process"]):
                    continue
                self.report_incident(f"{agent_id}_hung",
                                     f"Agent {agent_id} blocat {age_min:.0f} min; oprit pentru restart",
                                     severity="high", auto_resolved=True)
            self.stale_warned.discard(agent_id)

            if status["status"] == "stopped":
                continue  # oprit curat (utilizator / shutdown) — nu e o pană

            # Serverul e supraveghetorul: repornește el agenții morți (în ≤30 s) și
            # cei tocmai opriți mai sus. Repornim de aici doar dacă serverul nu
            # răspunde — altfel am porni instanțe duble.
            # La pornire serverul nu răspunde încă la /api/health (își lansează agenții):
            # în primele minute nu repornim nimic, altfel dublăm ce pornește el.
            starting_up = time.time() - getattr(self, "started_at", 0.0) < self.STARTUP_GRACE_SECONDS
            if starting_up or self._server_supervises():
                if agent_id not in self.down_warned:
                    self.down_warned.add(agent_id)
                    self.log_warning(f"Agent {agent_id} is down; server supervisor will restart it")
                continue

            self.log_warning(f"Agent {agent_id} appears down (stale={status['stale']}, alive={status['alive']})")

            # Încearcă restart automat doar pentru procese moarte
            if self.get_config().get("auto_restart_agents", True):
                self._restart_agent(agent_id)
            else:
                self.report_incident(f"{agent_id}_down",
                                      f"Agent {agent_id} is down but auto-restart disabled",
                                      severity="high")

        # 2. Omoară procese zombie
        self._kill_zombie_processes()

        # 3. Verifică spațiul pe disc
        self._check_disk_space()
        if time.time() - self.last_incident_prune > 3600:
            self.last_incident_prune = time.time()
            self._prune_incidents()

        # 4. Verifică resursele sistemului
        self._check_system_resources()

        # 5. Monitorizează logurile (la fiecare log_check_interval)
        current_time = time.time()
        if current_time - self.last_log_check > self.log_check_interval:
            self._monitor_logs()
            self.last_log_check = current_time

        # 6. Update heartbeat cu statusul sistemului
        self.update_heartbeat("running",
                              current_task="monitoring",
                              progress_percent=100)


def main():
    """Pornește System Monitor Agent."""
    agent = SystemMonitorAgent()

    # Creează marker file că agentul rulează
    marker_file = AGENTS_DIR / ".monitor_running"
    marker_file.write_text(str(os.getpid()))

    try:
        agent.run()
    finally:
        marker_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
