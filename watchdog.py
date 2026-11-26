"""Watchdog — ține serverul AutoResearch pornit fără intervenție umană.

Serverul își supraveghează agenții, dar pe el nu îl supraveghea nimeni: după un
crash sau o repornire a calculatorului sistemul rămânea oprit până la pornirea
manuală. Watchdog-ul verifică /api/health și:
  - pornește serverul dacă nu rulează;
  - îl oprește și îl repornește dacă procesul există dar nu mai răspunde
    (FAILURES_BEFORE_RESTART verificări la rând).

Rulare:  python watchdog.py            (buclă infinită)
         python watchdog.py --once     (o singură verificare; cod 0 = server OK)
Pornire automată la logare: install_autostart.ps1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
SERVER_SCRIPT = PROJECT_DIR / "server.py"
BOOT_LOG = PROJECT_DIR / "server_boot.log"
WATCHDOG_LOG = PROJECT_DIR / "watchdog.log"
LOG_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_PORT = 8020
CHECK_INTERVAL_SECONDS = 60
FAILURES_BEFORE_RESTART = 3
STARTUP_GRACE_SECONDS = 180  # serverul pornește 11 agenți + reconciliază lock-uri


def server_port() -> int:
    try:
        return int(os.environ.get("AUTORESEARCH_PORT") or DEFAULT_PORT)
    except ValueError:
        return DEFAULT_PORT


def _rotate(path: Path) -> None:
    try:
        if path.stat().st_size >= LOG_MAX_BYTES:
            os.replace(path, path.with_name(path.name + ".1"))
    except OSError:
        pass


def log(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    try:
        _rotate(WATCHDOG_LOG)
        with open(WATCHDOG_LOG, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass
    try:
        print(line, flush=True)
    except (OSError, ValueError, AttributeError):
        pass  # pythonw.exe nu are consolă


def server_healthy(port: int, timeout: float = 10.0) -> bool:
    import json
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=timeout) as response:
            if response.status != 200:
                return False
            return json.loads(response.read().decode("utf-8")).get("status") == "ok"
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _matches(cmdline: list[str], script: Path) -> bool:
    project = str(PROJECT_DIR).replace("/", "\\").lower()
    joined = " ".join(cmdline).replace("/", "\\").lower()
    return project in joined and any(os.path.basename(arg) == script.name for arg in cmdline)


def find_processes(script: Path) -> list:
    """Procesele care rulează `script` din acest proiect (fără familia procesului curent)."""
    import psutil

    try:
        me = psutil.Process()
        family = {me.pid} | {p.pid for p in me.parents()} | {p.pid for p in me.children(recursive=True)}
    except psutil.Error:
        family = {os.getpid()}
    found = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            if proc.info["pid"] not in family and _matches(proc.info["cmdline"] or [], script):
                found.append(proc)
        except (psutil.Error, OSError, TypeError):
            continue
    return found


def stop_server() -> int:
    """Oprește procesele server.py ale proiectului (și copiii lor: agenți, bucle)."""
    import psutil

    targets = []
    for proc in find_processes(SERVER_SCRIPT):
        try:
            targets.extend(proc.children(recursive=True))
        except psutil.Error:
            pass
        targets.append(proc)
    for proc in targets:
        try:
            proc.terminate()
        except psutil.Error:
            pass
    _gone, alive = psutil.wait_procs(targets, timeout=15)
    for proc in alive:
        try:
            proc.kill()
        except psutil.Error:
            pass
    return len(targets)


def start_server(port: int) -> int:
    _rotate(BOOT_LOG)
    env = dict(os.environ, AUTORESEARCH_PORT=str(port), PYTHONIOENCODING="utf-8")
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    python = Path(sys.executable)
    if python.name.lower() == "pythonw.exe":  # serverul are nevoie de stdout real
        python = python.with_name("python.exe")
    with open(BOOT_LOG, "a", encoding="utf-8") as boot_log:
        proc = subprocess.Popen(
            [str(python), str(SERVER_SCRIPT), "--port", str(port)],
            cwd=str(PROJECT_DIR), env=env, stdout=boot_log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, creationflags=flags,
        )
    return proc.pid


def check_once(port: int, failures: int, last_start: float, now: float | None = None) -> tuple[int, float]:
    """O verificare. Returnează (eșecuri consecutive, momentul ultimei porniri)."""
    now = time.time() if now is None else now
    if server_healthy(port):
        if failures:
            log("server răspunde din nou")
        return 0, last_start
    if now - last_start < STARTUP_GRACE_SECONDS:
        return failures, last_start  # tocmai pornit — încă se inițializează

    running = find_processes(SERVER_SCRIPT)
    if not running:
        pid = start_server(port)
        log(f"serverul nu rula → pornit (PID {pid}, port {port})")
        return 0, now

    failures += 1
    log(f"serverul rulează dar nu răspunde ({failures}/{FAILURES_BEFORE_RESTART})")
    if failures < FAILURES_BEFORE_RESTART:
        return failures, last_start
    stopped = stop_server()
    pid = start_server(port)
    log(f"server blocat → oprite {stopped} procese, repornit (PID {pid})")
    return 0, now


def main() -> int:
    parser = argparse.ArgumentParser(description="AutoResearch server watchdog")
    parser.add_argument("--once", action="store_true", help="o singură verificare")
    parser.add_argument("--port", type=int, default=server_port())
    args = parser.parse_args()

    if args.once:
        check_once(args.port, 0, 0.0)
        return 0 if server_healthy(args.port) else 1

    if find_processes(Path(__file__).resolve()):
        log("alt watchdog rulează deja — ies")
        return 0
    log(f"watchdog pornit (port {args.port}, verificare la {CHECK_INTERVAL_SECONDS}s)")
    failures, last_start = 0, 0.0
    while True:
        try:
            failures, last_start = check_once(args.port, failures, last_start)
        except Exception as exc:  # watchdog-ul nu are voie să moară
            log(f"eroare în verificare: {exc!r}")
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
