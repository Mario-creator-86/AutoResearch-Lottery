"""Base class for all agents in the Multi-Agent Lottery System."""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Agenții rulează cu stdout redirecționat (DEVNULL/fișier) → pe Windows Python
# folosește cp1252 și orice print cu diacritice („ș", „ă") arunca
# UnicodeEncodeError, oprind verificarea biletelor și restul ciclului.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

# Import existing infrastructure
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from atomic_io import read_json_with_recovery, write_json_atomic

MESSAGE_MAX_AGE_SECONDS = 300
LOG_MAX_BYTES = 2 * 1024 * 1024


def rotate_log_if_needed(log_file: Path, max_bytes: int = LOG_MAX_BYTES) -> None:
    """Păstrează logul curent sub max_bytes; versiunea anterioară rămâne în .log.1.

    Fără rotație logurile ajungeau la 8 MB, iar serverul le citește integral la
    fiecare cerere /api/agents/{id}/logs.
    """
    try:
        if log_file.stat().st_size < max_bytes:
            return
        os.replace(log_file, log_file.with_name(log_file.name + ".1"))
    except OSError:
        pass  # inexistent sau ținut deschis de alt proces (Windows) — reîncearcă la următorul log


def find_other_instance(agent_id: str) -> int | None:
    """PID-ul unei instanțe mai vechi a aceluiași agent din acest proiect, dacă există.

    Ignoră propria familie de procese: pe Windows `.venv\\Scripts\\python.exe` e un
    lansator care repornește interpretorul de bază, deci același agent apare ca
    părinte + copil cu aceeași linie de comandă. La pornire simultană câștigă
    procesul mai vechi (cel mai nou iese).
    """
    try:
        import psutil
    except ImportError:
        return None
    try:
        me = psutil.Process()
        family = {me.pid} | {p.pid for p in me.parents()} | {p.pid for p in me.children(recursive=True)}
        my_start = me.create_time()
    except psutil.Error:
        return None
    script = f"{agent_id}.py"
    project = str(PROJECT_DIR).replace("/", "\\").lower()
    for proc in psutil.process_iter(["pid", "cmdline", "create_time"]):
        try:
            if proc.info["pid"] in family:
                continue
            cmdline = proc.info["cmdline"] or []
            if not any(os.path.basename(arg) == script for arg in cmdline):
                continue
            if project not in " ".join(cmdline).replace("/", "\\").lower():
                continue
            if (proc.info["create_time"] or 0) <= my_start:
                return int(proc.info["pid"])
        except (psutil.Error, OSError, TypeError):
            continue
    return None


def parse_utc_timestamp(value: str) -> datetime:
    """ISO timestamp → datetime UTC aware.

    Agenții și CLI-ul scriu ``datetime.utcnow().isoformat()`` (naiv, dar UTC);
    API-ul scrie variante cu ``+00:00``. Un timestamp naiv trebuie tratat ca UTC —
    ``.timestamp()`` pe el l-ar interpreta ca oră locală (UTC+2/+3 în România).
    """
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def is_fresh_message(message: dict, max_age_seconds: float = MESSAGE_MAX_AGE_SECONDS) -> bool:
    try:
        sent = parse_utc_timestamp(message.get("timestamp", "1970-01-01"))
    except (TypeError, ValueError):
        return False
    return sent.timestamp() > time.time() - max_age_seconds

try:
    from telegram_bot import send_message as send_telegram_message
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

# Constants
AGENTS_DIR = PROJECT_DIR / "agents"
STATUS_DIR = AGENTS_DIR / "status"
MESSAGES_DIR = AGENTS_DIR / "messages"
COMMANDS_DIR = MESSAGES_DIR / "commands"
TASKS_DIR = AGENTS_DIR / "tasks"
INCIDENTS_DIR = AGENTS_DIR / "incidents"
LOGS_DIR = AGENTS_DIR / "logs"

# Ensure directories exist
for d in [STATUS_DIR, MESSAGES_DIR, COMMANDS_DIR, TASKS_DIR, INCIDENTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class AgentStatus:
    """Status data for an agent."""
    agent_id: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    status: str = "idle"
    current_task: str | None = None
    progress_percent: int = 0
    last_activity: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    pid: int = field(default_factory=os.getpid)

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "status": self.status,
            "current_task": self.current_task,
            "progress_percent": self.progress_percent,
            "last_activity": self.last_activity,
            "pid": self.pid
        }


class BaseAgent(ABC):
    """Base class for all agents in the system."""

    def __init__(self, agent_id: str, poll_interval: int = 30):
        self.agent_id = agent_id
        self.poll_interval = poll_interval
        self.running = False
        self.status_file = STATUS_DIR / f"{agent_id}.json"
        self.log_file = LOGS_DIR / f"{agent_id}.log"
        self.command_file = COMMANDS_DIR / f"to_{agent_id}.json"

    def _log(self, level: str, message: str) -> None:
        """Log message to file and optionally Telegram."""
        timestamp = datetime.utcnow().isoformat()
        log_entry = f"[{timestamp}] [{level}] {message}\n"

        # Write to log file
        rotate_log_if_needed(self.log_file)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)

        # Print to console
        print(f"[{self.agent_id}] {log_entry.strip()}")

    def log_info(self, message: str) -> None:
        self._log("INFO", message)

    def log_warning(self, message: str) -> None:
        self._log("WARNING", message)

    def log_error(self, message: str) -> None:
        self._log("ERROR", message)
        if TELEGRAM_AVAILABLE:
            try:
                send_telegram_message(f"⚠️ <b>{self.agent_id}</b>: {message}")
            except Exception:
                pass

    def log_success(self, message: str) -> None:
        self._log("SUCCESS", message)

    def update_heartbeat(self, status: str, current_task: str | None = None,
                          progress_percent: int = 0, additional_info: dict = None) -> None:
        """Update agent heartbeat status."""
        agent_status = AgentStatus(
            agent_id=self.agent_id,
            timestamp=datetime.utcnow().isoformat(),
            status=status,
            current_task=current_task,
            progress_percent=progress_percent,
            last_activity=datetime.utcnow().isoformat(),
            pid=os.getpid()
        )

        status_dict = agent_status.to_dict()
        # system_monitor își calibrează pragul de „stale" după cât doarme agentul.
        status_dict["poll_interval"] = self.poll_interval

        # Add additional info if provided
        if additional_info:
            status_dict.update(additional_info)

        try:
            write_json_atomic(self.status_file, status_dict)
        except Exception as e:
            self.log_error(f"Failed to update heartbeat: {e}")

        # Also update using the new utility functions for consistency
        try:
            from agents.utils import send_heartbeat
            heartbeat_info = {
                "status": status,
                "current_task": current_task,
                "progress_percent": progress_percent
            }
            if additional_info:
                heartbeat_info.update(additional_info)
            send_heartbeat(self.agent_id, heartbeat_info)
        except ImportError:
            # Utility functions not available, continue with file-based approach
            pass

    def read_status(self) -> dict | None:
        """Read current status from file."""
        data, _ = read_json_with_recovery(self.status_file)
        return data if isinstance(data, dict) else None

    def send_message(self, target: str, message_type: str, payload: dict) -> bool:
        """Send a message to another agent or broadcast."""
        message = {
            "id": str(uuid.uuid4()),
            "from": self.agent_id,
            "to": target,
            "type": message_type,
            "payload": payload,
            "timestamp": datetime.utcnow().isoformat()
        }

        if target == "broadcast":
            target_file = MESSAGES_DIR / "broadcast.json"
        else:
            target_file = COMMANDS_DIR / f"to_{target}.json"

        try:
            # Read existing messages
            data, _ = read_json_with_recovery(target_file)
            messages = data if isinstance(data, list) else []
            messages.append(message)

            # Keep only last 100 messages
            messages = messages[-100:]

            write_json_atomic(target_file, messages)
            return True
        except Exception as e:
            self.log_error(f"Failed to send message to {target}: {e}")
            return False

    def read_messages(self) -> list[dict]:
        """Read messages addressed to this agent."""
        try:
            data, _ = read_json_with_recovery(self.command_file)
            if isinstance(data, list):
                # Filter messages newer than 5 minutes
                return [m for m in data if isinstance(m, dict) and is_fresh_message(m)]
        except Exception:
            pass
        return []

    def clear_messages(self) -> None:
        """Clear all messages for this agent."""
        try:
            write_json_atomic(self.command_file, [])
        except Exception as e:
            self.log_error(f"Failed to clear messages: {e}")

    def report_incident(self, incident_type: str, description: str,
                        severity: str = "medium", auto_resolved: bool = False) -> None:
        """Report an incident."""
        timestamp = datetime.utcnow().isoformat()
        filename = f"{timestamp.replace(':', '-')}_{self.agent_id}_{incident_type}.json"
        incident_file = INCIDENTS_DIR / filename

        incident = {
            "timestamp": timestamp,
            "agent_id": self.agent_id,
            "type": incident_type,
            "description": description,
            "severity": severity,
            "auto_resolved": auto_resolved,
            "pid": os.getpid()
        }

        try:
            write_json_atomic(incident_file, incident)
            self.log_warning(f"Incident reported: {incident_type} - {description}")
        except Exception as e:
            self.log_error(f"Failed to report incident: {e}")

    def get_config(self) -> dict:
        """Load agent configuration."""
        config_file = AGENTS_DIR / "config.json"
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
                return config.get("agents", {}).get(self.agent_id, {})
        except Exception:
            return {}

    def check_stop_requested(self) -> bool:
        """Check if stop was requested for this agent."""
        messages = self.read_messages()
        for msg in messages:
            if msg.get("type") == "stop_request":
                return True
        return False

    def run(self) -> None:
        """Main agent loop."""
        other = find_other_instance(self.agent_id)
        if other is not None:
            # Serverul și system_monitor repornesc amândoi agenții morți; fără această
            # verificare rulau câte două instanțe (bilete și mesaje Telegram duble).
            self.log_warning(f"Agent {self.agent_id} already running (PID: {other}); this instance exits")
            return

        self.running = True
        self.log_info(f"Agent {self.agent_id} started (PID: {os.getpid()})")

        try:
            while self.running:
                # Update heartbeat
                self.update_heartbeat("running")

                # Check for stop request
                if self.check_stop_requested():
                    self.log_info("Stop requested, shutting down...")
                    # Consumă cererea: altfel un agent repornit în <5 min s-ar opri imediat.
                    self.clear_messages()
                    self.running = False
                    break

                # Check messages
                messages = self.read_messages()
                if messages:
                    self.handle_messages(messages)
                    self.clear_messages()

                # Run agent-specific processing
                try:
                    self.process()
                except Exception as e:
                    self.log_error(f"Error in process(): {e}")
                    self.report_incident("process_error", str(e), severity="high")

                # Sleep
                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            self.log_info("Interrupted, shutting down...")
        finally:
            self.update_heartbeat("stopped")
            self.log_info(f"Agent {self.agent_id} stopped")

    @abstractmethod
    def process(self) -> None:
        """Main processing logic - to be implemented by subclasses."""
        pass

    def handle_messages(self, messages: list[dict]) -> None:
        """Handle incoming messages - can be overridden by subclasses."""
        for msg in messages:
            msg_type = msg.get("type")
            if msg_type == "ping":
                self.send_message(msg.get("from", "system"), "pong",
                                  {"agent_id": self.agent_id})
            elif msg_type == "status_request":
                self.send_message(msg.get("from", "system"), "status_response",
                                  self.read_status())
