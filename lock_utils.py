from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LockOwner:
    pid: int | None
    executable: str | None = None
    script: str | None = None
    project_dir: str | None = None
    kind: str | None = None
    lottery_id: str | None = None


def current_lock_owner(
    *,
    script_path: Path,
    project_dir: Path,
    kind: str,
    lottery_id: str | None = None,
) -> LockOwner:
    return LockOwner(
        pid=os.getpid(),
        executable=str(Path(sys.executable).resolve()),
        script=str(script_path.resolve()),
        project_dir=str(project_dir.resolve()),
        kind=kind,
        lottery_id=lottery_id,
    )


def read_lock_owner(path: Path) -> LockOwner:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return LockOwner(pid=None)

    if not raw:
        return LockOwner(pid=None)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        try:
            pid = int(payload["pid"]) if payload.get("pid") is not None else None
        except (TypeError, ValueError):
            pid = None
        return LockOwner(
            pid=pid,
            executable=_optional_str(payload.get("executable")),
            script=_optional_str(payload.get("script")),
            project_dir=_optional_str(payload.get("project_dir")),
            kind=_optional_str(payload.get("kind")),
            lottery_id=_optional_str(payload.get("lottery_id")),
        )

    try:
        return LockOwner(pid=int(raw))
    except ValueError:
        return LockOwner(pid=None)


def serialize_lock_owner(owner: LockOwner) -> str:
    return json.dumps(
        {
            "pid": owner.pid,
            "executable": owner.executable,
            "script": owner.script,
            "project_dir": owner.project_dir,
            "kind": owner.kind,
            "lottery_id": owner.lottery_id,
        },
        ensure_ascii=True,
    )


def lock_matches_process(
    owner: LockOwner,
    *,
    script_path: Path | None = None,
    project_dir: Path | None = None,
) -> bool:
    if owner.pid is None:
        return False

    executable, command_line = _process_identity(owner.pid)
    if executable is None and command_line is None:
        return True

    expected_executable = owner.executable
    if expected_executable and executable and _norm_path(executable) != _norm_path(expected_executable):
        # Windows venv: .venv\Scripts\python.exe is a launcher that re-executes the
        # base interpreter, so the process image (Python314\python.exe) differs from
        # sys.executable, but the command line keeps the venv path. Without this, every
        # live auto_loop looked "stale": its lock was deleted and duplicates were started.
        if not (command_line and _norm_text(expected_executable) in _norm_text(command_line)):
            return False

    expected_script = owner.script or (str(script_path.resolve()) if script_path else None)
    if expected_script and command_line:
        if _norm_text(expected_script) not in _norm_text(command_line):
            return False

    expected_project_dir = owner.project_dir or (str(project_dir.resolve()) if project_dir else None)
    if expected_project_dir and command_line:
        if _norm_text(expected_project_dir) not in _norm_text(command_line):
            return False

    return True


def process_runs_script(pid: int, script_path: Path) -> bool:
    """True dacă PID-ul rulează script_path (sau identitatea nu poate fi citită)."""
    _executable, command_line = _process_identity(pid)
    if command_line is None:
        return True
    return _norm_text(script_path.name) in _norm_text(command_line)


def _optional_str(value: object) -> str | None:
    if value in {None, ""}:
        return None
    return str(value)


def _norm_path(value: str) -> str:
    try:
        return str(Path(value).resolve()).replace("/", "\\").lower()
    except OSError:
        return value.replace("/", "\\").lower()


def _norm_text(value: str) -> str:
    return value.replace("/", "\\").lower()


def _process_identity(pid: int) -> tuple[str | None, str | None]:
    identity = _psutil_process_identity(pid)
    if identity is not None:
        return identity
    if sys.platform == "win32":
        return _windows_process_identity(pid)
    return _posix_process_identity(pid)


def _psutil_process_identity(pid: int) -> tuple[str | None, str | None] | None:
    """Rapid (fără un PowerShell nou per verificare); None → folosește metoda nativă."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        proc = psutil.Process(int(pid))
        command_line = subprocess.list2cmdline(proc.cmdline())
        try:
            executable = proc.exe() or None
        except psutil.AccessDenied:
            executable = None
        return executable, command_line or None
    except psutil.NoSuchProcess:
        return None, None
    except (psutil.Error, OSError, ValueError):
        return None


def _windows_process_identity(pid: int) -> tuple[str | None, str | None]:
    command = (
        f"$proc = Get-CimInstance Win32_Process -Filter \"ProcessId = {int(pid)}\"; "
        "if ($proc) { "
        "[pscustomobject]@{ExecutablePath=$proc.ExecutablePath; CommandLine=$proc.CommandLine} "
        "| ConvertTo-Json -Compress }"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None

    if completed.returncode != 0 or not completed.stdout.strip():
        return None, None

    try:
        payload = json.loads(completed.stdout.strip())
    except json.JSONDecodeError:
        return None, None

    if not isinstance(payload, dict):
        return None, None

    executable = _optional_str(payload.get("ExecutablePath"))
    command_line = _optional_str(payload.get("CommandLine"))
    return executable, command_line


def _posix_process_identity(pid: int) -> tuple[str | None, str | None]:
    executable = None
    command_line = None

    proc_root = Path("/proc") / str(pid)
    try:
        executable = str(proc_root.joinpath("exe").resolve())
    except OSError:
        executable = None

    try:
        raw = proc_root.joinpath("cmdline").read_bytes()
    except OSError:
        raw = b""
    if raw:
        command_line = raw.replace(b"\0", b" ").decode("utf-8", errors="ignore").strip() or None

    if command_line is None:
        try:
            completed = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return executable, None
        if completed.returncode == 0:
            command_line = completed.stdout.strip() or None

    return executable, command_line
