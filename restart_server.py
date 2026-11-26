#!/usr/bin/env python3
"""Script to restart the AutoResearch server"""

import os
import signal
import sys
import time
from pathlib import Path

def stop_server():
    """Stop the server by removing PID file and sending termination signals"""
    project_dir = Path(__file__).parent
    agents_dir = project_dir / "agents"
    server_pid_file = agents_dir / ".server_pid"

    # Check if server PID file exists
    if server_pid_file.exists():
        try:
            with open(server_pid_file, "r") as f:
                pid = int(f.read().strip())
            print(f"Found server PID: {pid}")

            # Try to remove the PID file
            server_pid_file.unlink()
            print("Removed server PID file")
        except Exception as e:
            print(f"Could not read/remove PID file: {e}")

    print("Please manually stop any remaining Python processes in Task Manager")
    print("Then run 'python server.py' to start the server")

if __name__ == "__main__":
    stop_server()