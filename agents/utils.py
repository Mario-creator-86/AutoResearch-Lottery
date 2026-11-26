"""
Utility functions for agents to interact with the server and monitor their own health.
"""

import os
import sys
import json
import time
import subprocess
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Assuming this file is in agents/utils.py, PROJECT_DIR would be the parent of the parent
PROJECT_DIR = Path(__file__).parent.parent

DEFAULT_SERVER_PORT = 8020


def server_base_url() -> str:
    """Dashboard URL. server.py exports AUTORESEARCH_PORT, inherited by the agents it spawns."""
    port = os.environ.get("AUTORESEARCH_PORT") or DEFAULT_SERVER_PORT
    return f"http://localhost:{port}"


def is_server_alive() -> bool:
    """
    Check if the server process is still alive by verifying the PID in the server PID file.

    Returns:
        bool: True if server is alive, False otherwise
    """
    AGENTS_DIR = PROJECT_DIR / "agents"
    server_pid_file = AGENTS_DIR / ".server_pid"

    try:
        if not server_pid_file.exists():
            return False

        with open(server_pid_file, "r") as f:
            pid = int(f.read().strip())

        # Check if process with that PID is still running
        if sys.platform == "win32":
            # On Windows, use tasklist to check if process exists
            result = subprocess.run(
                ["tasklist", "/fi", f"PID eq {pid}"],
                capture_output=True,
                text=True,
                shell=True
            )
            return f" {pid} " in result.stdout
        else:
            # On Unix-like systems, use os.kill with signal 0 to check if process exists
            os.kill(pid, 0)
            return True
    except (ValueError, FileNotFoundError, ProcessLookupError, PermissionError, subprocess.SubprocessError):
        return False


def is_server_responsive(timeout: int = 5) -> bool:
    """
    Check if the server is responsive by making a health check API call.

    Args:
        timeout (int): Request timeout in seconds

    Returns:
        bool: True if server responds successfully, False otherwise
    """
    try:
        response = requests.get(f"{server_base_url()}/api/health", timeout=timeout)
        return response.status_code == 200 and response.json().get("status") != "shutting_down"
    except requests.RequestException:
        return False


def get_server_pid() -> int | None:
    """
    Get the server PID from the PID file.

    Returns:
        int | None: Server PID if available, None otherwise
    """
    AGENTS_DIR = PROJECT_DIR / "agents"
    server_pid_file = AGENTS_DIR / ".server_pid"

    try:
        if not server_pid_file.exists():
            return None

        with open(server_pid_file, "r") as f:
            return int(f.read().strip())
    except (ValueError, FileNotFoundError):
        return None


def update_agent_status(agent_id: str, status: dict) -> None:
    """
    Update the agent's status file with current information.

    Merges into the existing status so fields written by BaseAgent
    (last_activity, timestamp, ...) are never lost — system_monitor treats a
    missing last_activity as a stale heartbeat and would otherwise restart
    healthy agents in a loop.
    """
    AGENTS_DIR = PROJECT_DIR / "agents"
    STATUS_DIR = AGENTS_DIR / "status"
    status_file = STATUS_DIR / f"{agent_id}.json"

    # Ensure status directory exists
    STATUS_DIR.mkdir(parents=True, exist_ok=True)

    merged: dict = {}
    try:
        if status_file.exists():
            with open(status_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if isinstance(existing, dict):
                    merged = existing
    except (OSError, json.JSONDecodeError):
        merged = {}

    merged.update(status)
    now_iso = datetime.utcnow().isoformat()
    merged.setdefault("last_activity", now_iso)
    merged.setdefault("timestamp", now_iso)
    merged["last_update"] = now_iso
    merged["pid"] = os.getpid()

    try:
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not update status file for {agent_id}: {e}")


def send_heartbeat(agent_id: str, additional_info: dict = None) -> None:
    """
    Send a heartbeat signal by updating the agent's status file.

    Args:
        agent_id (str): The agent identifier
        additional_info (dict, optional): Additional information to include in heartbeat
    """
    status = {
        "status": "running",
        "last_heartbeat": datetime.utcnow().isoformat()
    }

    if additional_info:
        status.update(additional_info)

    update_agent_status(agent_id, status)


def should_shutdown() -> bool:
    """
    Check if the agent should shut down (server is not alive or shutting down).

    Returns:
        bool: True if agent should shut down, False otherwise
    """
    # First check if server is responsive via API
    if is_server_responsive():
        return False

    # If API is not responsive, fall back to PID file check
    return not is_server_alive() and get_server_pid() is not None


def wait_for_server(timeout: int = 30) -> bool:
    """
    Wait for the server to become available.

    Args:
        timeout (int): Maximum time to wait in seconds

    Returns:
        bool: True if server became available, False on timeout
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_server_alive() and is_server_responsive():
            return True
        time.sleep(1)
    return False


def get_agent_config(agent_id: str) -> Dict[str, Any]:
    """
    Get agent configuration from its status file.

    Args:
        agent_id (str): The agent identifier

    Returns:
        Dict[str, Any]: Agent configuration or empty dict if not found
    """
    AGENTS_DIR = PROJECT_DIR / "agents"
    STATUS_DIR = AGENTS_DIR / "status"
    status_file = STATUS_DIR / f"{agent_id}.json"

    if status_file.exists():
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return {}


def register_shutdown_handler(handler_func) -> None:
    """
    Register a function to be called when the agent should shut down.

    Args:
        handler_func: Function to call on shutdown
    """
    import atexit
    import signal

    def _shutdown_handler(signum=None, frame=None):
        try:
            handler_func()
        except Exception as e:
            print(f"Error in shutdown handler: {e}")
        finally:
            sys.exit(0)

    # Register for SIGTERM and SIGINT
    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    # Register for normal exit
    atexit.register(_shutdown_handler)