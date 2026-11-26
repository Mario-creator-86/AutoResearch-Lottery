"""Lottery Manager Agent

Acest agent rulează continuu și:
- Gestionează care loterii rulează
- Previne supraîncărcarea sistemului (max 2 loterii simultan)
- Prioritizează task-urile în queue
- Monitorizează resursele sistemului
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_base import BaseAgent, AGENTS_DIR, TASKS_DIR

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from lottery_config import list_lottery_configs, get_lottery_config
from atomic_io import read_json_with_recovery, write_json_atomic

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class LotteryManagerAgent(BaseAgent):
    """Agent pentru managementul loteriilor și load balancing."""

    def __init__(self):
        super().__init__("lottery_manager", poll_interval=60)

        self.config = self.get_config()
        self.max_simultaneous = self.config.get("max_simultaneous_lotteries", 2)
        self.max_cpu = self.config.get("max_cpu_percent", 80)
        self.max_memory = self.config.get("max_memory_percent", 75)
        # Procentul e al întregului sistem (alte programe); buclele folosesc ~50 MB
        # fiecare, deci blocăm doar dacă e și puțină memorie liberă în valoare absolută.
        self.min_free_memory_gb = float(self.config.get("min_free_memory_gb", 1.0))
        self.loop_rotation_hours = float(self.config.get("loop_rotation_hours", 6))

        self.running_experiments: dict[str, subprocess.Popen] = {}
        self.task_queue: list[dict] = []
        self.in_progress: list[dict] = []

        self.last_resource_check = 0
        self.resource_check_interval = 30

    def _get_active_experiments_count(self) -> int:
        """Verifică câte experimente sunt active citind din summary."""
        summary_file = AGENTS_DIR / "experiments_summary.json"
        try:
            if summary_file.exists():
                with open(summary_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("active_count", 0)
        except Exception:
            pass
        return 0

    def _get_system_resources(self) -> dict[str, float]:
        """Obține resursele sistemului."""
        if not PSUTIL_AVAILABLE:
            return {"cpu": 0.0, "memory": 0.0, "disk": 100.0}

        try:
            cpu = psutil.cpu_percent(interval=1)
            virtual = psutil.virtual_memory()
            memory = virtual.percent
            disk = psutil.disk_usage("/" if sys.platform != "win32" else "C:\\").percent

            return {"cpu": cpu, "memory": memory, "disk": disk,
                    "memory_available_gb": virtual.available / 2**30}
        except Exception as e:
            self.log_error(f"Failed to get system resources: {e}")
            return {"cpu": 0.0, "memory": 0.0, "disk": 100.0}

    def _can_start_new_experiment(self) -> bool:
        """Verifică dacă se poate porni un experiment nou."""
        # Verifică numărul de experimente active (proprii + externe)
        active_count = len(self.running_experiments) + self._external_loop_count()
        if active_count >= self.max_simultaneous:
            return False

        # Verifică resursele
        resources = self._get_system_resources()
        if resources["cpu"] > self.max_cpu:
            self.log_warning(f"CPU too high ({resources['cpu']}%), delaying new experiments")
            return False
        free_gb = resources.get("memory_available_gb")
        if resources["memory"] > self.max_memory and (free_gb is None or free_gb < self.min_free_memory_gb):
            self.log_warning(f"Memory too high ({resources['memory']}%, {free_gb or 0:.1f} GB free), "
                             "delaying new experiments")
            return False

        return True

    def _external_loop_count(self) -> int:
        return sum(1 for t in self.in_progress if t.get("note") == "externally_managed")

    def _reconcile_external_loops(self) -> None:
        """Buclele pornite de instanțe anterioare ale managerului (sau de utilizator)
        consumă capacitate: le detectăm după lock viu și le marcăm ca externe."""
        tracked = {t.get("lottery") for t in self.in_progress}
        for config in list_lottery_configs():
            if config.id in tracked:
                continue
            if self._is_auto_loop_running(config.id):
                self.in_progress.append({
                    "id": f"external_{config.id}_{int(time.time())}",
                    "lottery": config.id,
                    "type": "auto_loop",
                    "priority": 0,
                    "params": {},
                    "created_at": datetime.utcnow().isoformat(),
                    "status": "running",
                    "note": "externally_managed",
                    "started_at": datetime.utcnow().isoformat(),
                })
                self.log_info(
                    f"Detected live auto loop for {config.id} (started elsewhere); counting against capacity"
                )

    def _cleanup_finished_experiments(self) -> None:
        """Curăță experimentele terminate."""
        finished = []
        for lottery_id, proc in self.running_experiments.items():
            if proc.poll() is not None:
                finished.append(lottery_id)
                return_code = proc.returncode
                status = "success" if return_code == 0 else "failed"
                self.log_info(f"Experiment {lottery_id} finished with status: {status}")

        for lottery_id in finished:
            del self.running_experiments[lottery_id]
            # Mută din in_progress în completed
            self._complete_task(lottery_id)

        # Task-urile "externally_managed" (buclă pornită în afara managerului)
        # sunt eliminate din in_progress când bucla externă s-a terminat.
        stale = []
        for task in self.in_progress:
            lottery_id = task.get("lottery")
            if (
                task.get("note") == "externally_managed"
                and lottery_id
                and not self._is_auto_loop_running(lottery_id)
            ):
                stale.append(task)
        for task in stale:
            self.in_progress.remove(task)
            self.log_info(f"External auto loop for {task.get('lottery')} no longer running; task released")

    def _load_task_queue(self) -> None:
        """Încarcă task-urile din fișier."""
        queue_file = TASKS_DIR / "pending.json"
        data, _ = read_json_with_recovery(queue_file)

        if isinstance(data, list):
            # Filtrare duplicate
            existing_ids = {t.get("id") for t in self.task_queue}
            for task in data:
                if task.get("id") not in existing_ids:
                    self.task_queue.append(task)

        # Sortează după prioritate
        self.task_queue.sort(key=lambda t: t.get("priority", 5), reverse=True)

    def _save_task_queue(self) -> None:
        """Salvează task-urile în fișier."""
        queue_file = TASKS_DIR / "pending.json"
        try:
            write_json_atomic(queue_file, self.task_queue)
        except Exception as e:
            self.log_error(f"Failed to save task queue: {e}")

    def _complete_task(self, lottery_id: str) -> None:
        """Marchează un task ca finalizat."""
        # Găsește și mută în completed
        completed_file = TASKS_DIR / "completed.json"

        for i, task in enumerate(self.in_progress):
            if task.get("lottery") == lottery_id:
                task["completed_at"] = datetime.utcnow().isoformat()
                task["status"] = "completed"

                # Salvează în completed
                data, _ = read_json_with_recovery(completed_file)
                completed = data if isinstance(data, list) else []
                completed.append(task)

                try:
                    write_json_atomic(completed_file, completed[-100:])  # Păstrează ultimele 100
                except Exception as e:
                    self.log_error(f"Failed to save completed task: {e}")

                self.in_progress.pop(i)
                break

    def _add_task(self, lottery_id: str, task_type: str, priority: int = 5,
                  params: dict | None = None) -> str:
        """Adaugă un task nou în queue."""
        task = {
            "id": f"{lottery_id}_{datetime.utcnow().isoformat()}",
            "lottery": lottery_id,
            "type": task_type,
            "priority": priority,
            "params": params or {},
            "created_at": datetime.utcnow().isoformat(),
            "status": "pending"
        }

        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: t.get("priority", 5), reverse=True)
        self._save_task_queue()

        self.log_info(f"Added task: {task_type} for {lottery_id} (priority: {priority})")
        return task["id"]

    def _start_experiment(self, task: dict) -> bool:
        """Pornește un experiment."""
        lottery_id = task.get("lottery")
        task_type = task.get("type")

        if not lottery_id or not task_type:
            self.log_error(f"Invalid task: {task}")
            return False

        try:
            config = get_lottery_config(lottery_id)
        except Exception as e:
            self.log_error(f"Unknown lottery: {lottery_id}")
            return False

        # Verifică dacă există deja un auto_loop activ pentru această loterie
        if task_type == "auto_loop" and self._is_auto_loop_running(lottery_id):
            self.log_info(f"Auto loop already running for {lottery_id}, skipping new task")
            # Marchează task-ul ca running și adaugă în in_progress
            task["status"] = "running"
            task["started_at"] = datetime.utcnow().isoformat()
            task["note"] = "externally_managed"
            self.in_progress.append(task)
            return True

        # Construiește comanda în funcție de tipul task-ului
        if task_type == "auto_loop":
            cmd = [sys.executable, str(PROJECT_DIR / "auto_loop.py"), "--lottery", lottery_id]
        elif task_type == "walkforward":
            cmd = [sys.executable, str(PROJECT_DIR / "walkforward.py"), "--lottery", lottery_id]
        elif task_type == "experiment":
            # Rulează un experiment specific
            cmd = [sys.executable, str(PROJECT_DIR / "train.py"), "--lottery", lottery_id]
        else:
            self.log_error(f"Unknown task type: {task_type}")
            return False

        creation_flags = 0
        if sys.platform == "win32":
            creation_flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            creation_flags |= getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)

        try:
            # Loghează output-ul buclei în fișier per loterie (nu DEVNULL)
            stdout_path = config.state_dir / "loop.out.log"
            stderr_path = config.state_dir / "loop.err.log"
            with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open(
                "a", encoding="utf-8"
            ) as stderr_handle:
                proc = subprocess.Popen(
                    cmd,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    cwd=str(PROJECT_DIR),
                    creationflags=creation_flags,
                )

            self.running_experiments[lottery_id] = proc
            task["started_at"] = datetime.utcnow().isoformat()
            task["status"] = "running"
            task["pid"] = proc.pid
            self.in_progress.append(task)

            self.log_success(f"Started {task_type} for {lottery_id} (PID: {proc.pid})")
            return True

        except Exception as e:
            self.log_error(f"Failed to start experiment: {e}")
            return False

    def _is_auto_loop_running(self, lottery_id: str) -> bool:
        """Verifică dacă auto_loop rulează deja pentru o loterie.

        Nu e destul ca PID-ul din lock să fie viu — Windows reciclează PID-uri,
        deci verificăm și că procesul chiar e un auto_loop al acestui proiect.
        """
        from pathlib import Path
        import ctypes

        from lock_utils import lock_matches_process

        config = get_lottery_config(lottery_id)
        lock_file = config.lock_file

        if not lock_file.exists():
            return False

        try:
            from lock_utils import read_lock_owner

            owner = read_lock_owner(lock_file)
            if owner.pid:
                # Verifică dacă procesul mai e activ
                try:
                    if sys.platform == "win32":
                        process = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(owner.pid))
                        if process:
                            ctypes.windll.kernel32.CloseHandle(process)
                            # PID viu: e într-adevăr auto_loop-ul nostru?
                            return lock_matches_process(
                                owner,
                                script_path=PROJECT_DIR / "auto_loop.py",
                                project_dir=PROJECT_DIR,
                            )
                    else:
                        os.kill(int(owner.pid), 0)
                        return lock_matches_process(
                            owner,
                            script_path=PROJECT_DIR / "auto_loop.py",
                            project_dir=PROJECT_DIR,
                        )
                except Exception:
                    return False
        except Exception as e:
            self.log_error(f"Error checking lock: {e}")

        return False

    # Loterii prioritare (RO + IT) — primesc prioritate mai mare în coadă
    PRIORITY_LOTTERIES = ("loto_6_49_ro", "loto_5_40_ro", "joker_ro", "superenalotto_it")
    # Loterii cu prea puține date pentru un auto_loop util
    MIN_DRAWS_FOR_LOOP = 300

    def _check_scheduled_tasks(self) -> None:
        """Asigură acoperire non-stop: o buclă auto_loop pentru fiecare loterie
        cu date suficiente (care nu rulează deja), apoi walkforward săptămânal."""
        now = datetime.utcnow()

        # Non-stop: menține în coadă câte un task auto_loop pentru fiecare
        # loterie eligibilă care nu rulează deja și nu e deja în coadă.
        queued_lotteries = {t.get("lottery") for t in self.task_queue if t.get("type") == "auto_loop"}
        running_lotteries = {t.get("lottery") for t in self.in_progress if t.get("type") == "auto_loop"}
        for config in list_lottery_configs():
            if config.id in queued_lotteries or config.id in running_lotteries:
                continue
            # Sare loteriile puse pe pauză explicit din UI (loop/stop)
            if config.stop_request_file.parent.joinpath(".autoloop.paused").exists():
                continue
            if self._count_draws(config) < self.MIN_DRAWS_FOR_LOOP:
                continue
            priority = 9 if config.id in self.PRIORITY_LOTTERIES else 5
            self._add_task(config.id, "auto_loop", priority=priority)

        # Săptămânal: walkforward validation (duminica la 2 AM)
        if now.weekday() == 6 and now.hour == 2 and now.minute < 5:
            for config in list_lottery_configs():
                self._add_task(config.id, "walkforward", priority=6)

    def _rotate_long_running_loops(self) -> None:
        """Cere oprirea grațioasă a buclelor care rulează de prea mult timp,
        ca sloturile să se rotească către restul loteriilor din coadă.
        auto_loop salvează starea continuu — o rotire nu pierde progresul."""
        rotation_seconds = float(self.loop_rotation_hours) * 3600
        now = time.time()
        for task in self.in_progress:
            if task.get("type") != "auto_loop":
                continue
            started_at = task.get("started_at")
            if not started_at:
                continue
            try:
                age = now - datetime.fromisoformat(started_at).timestamp()
            except (ValueError, TypeError):
                continue
            if age < rotation_seconds:
                continue
            lottery_id = task.get("lottery")
            if not lottery_id:
                continue
            try:
                config = get_lottery_config(lottery_id)
                config.stop_request_file.parent.mkdir(parents=True, exist_ok=True)
                config.stop_request_file.write_text("stop\n", encoding="utf-8")
                task["rotate_requested_at"] = datetime.utcnow().isoformat()
                self.log_info(
                    f"Rotation: requested graceful stop of {lottery_id} loop "
                    f"(ran {age / 3600:.1f}h); next lottery from queue will start"
                )
            except Exception as exc:
                self.log_error(f"Rotation failed for {lottery_id}: {exc}")

    @staticmethod
    def _count_draws(config) -> int:
        """Numără rapid rândurile de date din CSV-ul loteriei."""
        csv_path = config.draws_file
        if not csv_path.exists():
            return 0
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                return max(0, sum(1 for _ in f) - 1)
        except OSError:
            return 0

    def _handle_resource_reduction(self, reason: str, severity: str) -> None:
        """Gestionează cererea de reducere a load-ului."""
        self.log_warning(f"Received reduce_load request: {reason} ({severity})")

        if severity == "high":
            # Oprește un experiment activ (cel cu prioritatea cea mai mică)
            if self.running_experiments:
                # Găsește experimentul cu prioritatea cea mai mică
                to_stop = None
                lowest_priority = 999

                for lottery_id in self.running_experiments:
                    for task in self.in_progress:
                        if task.get("lottery") == lottery_id:
                            if task.get("priority", 5) < lowest_priority:
                                lowest_priority = task.get("priority", 5)
                                to_stop = lottery_id

                if to_stop:
                    proc = self.running_experiments[to_stop]
                    try:
                        proc.terminate()
                        self.log_info(f"Terminated experiment {to_stop} due to resource constraints")
                    except Exception as e:
                        self.log_error(f"Failed to terminate {to_stop}: {e}")

    def handle_messages(self, messages: list[dict]) -> None:
        """Gestionează mesajele primite."""
        for msg in messages:
            msg_type = msg.get("type")

            if msg_type == "reduce_load":
                payload = msg.get("payload", {})
                self._handle_resource_reduction(
                    payload.get("reason", "unknown"),
                    payload.get("severity", "medium")
                )

            elif msg_type == "add_task":
                payload = msg.get("payload", {})
                self._add_task(
                    payload.get("lottery"),
                    payload.get("task_type", "auto_loop"),
                    payload.get("priority", 5),
                    payload.get("params", {})
                )

            elif msg_type == "status_request":
                self.send_message(msg.get("from", "system"), "status_response", {
                    "active_experiments": len(self.running_experiments),
                    "queued_tasks": len(self.task_queue),
                    "in_progress": len(self.in_progress),
                    "resources": self._get_system_resources()
                })

    def process(self) -> None:
        """Procesare principală - rulează la fiecare 60 secunde."""

        # 1. Curăță experimentele terminate
        self._cleanup_finished_experiments()

        # 1b. Detectează buclele externe încă vii (le consumă pe sloturi)
        self._reconcile_external_loops()

        # 2. Încarcă task-uri noi
        self._load_task_queue()

        # 3. Verifică task-uri programate + rotește buclele lungi
        self._check_scheduled_tasks()
        self._rotate_long_running_loops()

        # 4. Pornește task-uri din queue dacă sunt resurse disponibile
        started_count = 0
        while self.task_queue and self._can_start_new_experiment() and started_count < 2:
            task = self.task_queue.pop(0)

            if self._start_experiment(task):
                started_count += 1
            else:
                # Pune înapoi în queue dacă nu a putut porni
                self.task_queue.insert(0, task)
                break

        # 5. Salvează queue-ul actualizat
        self._save_task_queue()

        # 6. Update heartbeat cu status
        resources = self._get_system_resources()
        active_count = self._get_active_experiments_count()
        self.update_heartbeat(
            "running",
            current_task=f"managing {active_count} experiments, {len(self.task_queue)} queued",
            progress_percent=min(100, active_count * 25)  # 25% per experiment
        )

        # Log periodic
        if int(time.time()) % 300 < 60:  # La fiecare 5 minute
            self.log_info(f"Status: {active_count} active, {len(self.task_queue)} queued, "
                          f"CPU: {resources['cpu']:.1f}%, Memory: {resources['memory']:.1f}%")

def main():
    """Pornește Lottery Manager Agent."""
    agent = LotteryManagerAgent()
    agent.run()


if __name__ == "__main__":
    main()
