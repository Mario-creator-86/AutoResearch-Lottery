from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

DEFAULT_ATOMIC_WRITE_RETRIES = 60
DEFAULT_ATOMIC_WRITE_RETRY_DELAY = 0.1
TRANSIENT_REPLACE_WINERRORS = {5, 32}
RECOVERY_FILE_SUFFIX = ".recovery"


def _is_transient_replace_error(exc: OSError) -> bool:
    if isinstance(exc, PermissionError):
        return True
    return getattr(exc, "winerror", None) in TRANSIENT_REPLACE_WINERRORS


def write_text_atomic(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    retries: int = DEFAULT_ATOMIC_WRITE_RETRIES,
    retry_delay: float = DEFAULT_ATOMIC_WRITE_RETRY_DELAY,
) -> None:
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 1:
        raise ValueError("retries must be a positive integer")
    if not isinstance(retry_delay, (int, float)) or not 0 <= retry_delay < float("inf"):
        raise ValueError("retry_delay must be finite and non-negative")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")

    try:
        with tmp.open("w", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(retries):
            try:
                os.replace(str(tmp), str(path))
                return
            except OSError as exc:
                if not _is_transient_replace_error(exc) or attempt == retries - 1:
                    raise
                time.sleep(retry_delay)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def write_json_atomic(
    path: Path,
    data: Any,
    *,
    ensure_ascii: bool = True,
    encoding: str = "utf-8",
    retries: int = DEFAULT_ATOMIC_WRITE_RETRIES,
    retry_delay: float = DEFAULT_ATOMIC_WRITE_RETRY_DELAY,
) -> None:
    write_text_atomic(
        path,
        json.dumps(data, ensure_ascii=ensure_ascii),
        encoding=encoding,
        retries=retries,
        retry_delay=retry_delay,
    )


def recovery_path(path: Path) -> Path:
    return path.with_name(f"{path.name}{RECOVERY_FILE_SUFFIX}")


def write_json_resilient(
    path: Path,
    data: Any,
    *,
    ensure_ascii: bool = True,
    encoding: str = "utf-8",
    retries: int = DEFAULT_ATOMIC_WRITE_RETRIES,
    retry_delay: float = DEFAULT_ATOMIC_WRITE_RETRY_DELAY,
) -> Path:
    fallback_path = recovery_path(path)
    try:
        write_json_atomic(
            path,
            data,
            ensure_ascii=ensure_ascii,
            encoding=encoding,
            retries=retries,
            retry_delay=retry_delay,
        )
    except OSError:
        write_json_atomic(
            fallback_path,
            data,
            ensure_ascii=ensure_ascii,
            encoding=encoding,
            retries=retries,
            retry_delay=retry_delay,
        )
        return fallback_path
    try:
        fallback_path.unlink(missing_ok=True)
    except OSError:
        pass
    return path


def read_json_with_recovery(path: Path, *, encoding: str = "utf-8") -> tuple[Any | None, Path | None]:
    candidates = []
    for candidate in (path, recovery_path(path)):
        try:
            candidates.append((candidate.stat().st_mtime, candidate))
        except FileNotFoundError:
            continue
    candidates.sort(key=lambda item: item[0], reverse=True)
    for _mtime, candidate in candidates:
        try:
            return json.loads(candidate.read_text(encoding=encoding)), candidate
        except (OSError, json.JSONDecodeError):
            continue
    return None, None
