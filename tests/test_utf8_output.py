"""Agenții pornesc cu stdout redirecționat; print-urile cu diacritice nu trebuie să crape."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run_redirected(code: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONIOENCODING", "PYTHONUTF8"}}
    env["PYTHONUTF8"] = "0"
    if sys.platform == "win32":
        env["PYTHONIOENCODING"] = "cp1252"  # ca un proces cu stdout către fișier/DEVNULL
    # calea proiectului conține apostrof („Karpathy's") → o trimitem prin argv, nu în cod
    return subprocess.run(
        [sys.executable, "-c", code, str(ROOT / "agents")], cwd=str(ROOT / "agents"), env=env,
        capture_output=True, timeout=120,
    )


def test_agent_base_makes_stdout_utf8_safe():
    proc = _run_redirected(
        "import sys; sys.path.insert(0, sys.argv[1]); import agent_base; "
        "print('verificare bilete: actualizarea istoricului a eșuat — ă î â ș ț')"
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert "eșuat".encode("utf-8") in proc.stdout
