"""Decision Ledger — jurnal apend-only cu toate deciziile agentului AI.

Fiecare intrare răspunde la: CE a propus, CE a gândit, DE CE a făcut-o și
CARE a fost rezultatul (completat ulterior când e disponibil).

Stocare: data/{lottery}/decision_ledger.jsonl (per loterie) + o copie a
ultimelor intrări globale în data/decision_ledger_global.jsonl pentru vizualizare
agregată. Format JSONL = un obiect JSON pe linie (append sigur, crash-safe).

Tipuri de intrări:
  - ai_analysis      : analiza Ollama pe o loterie (numere propuse, insight-uri)
  - ai_tickets       : biletele propuse pentru o extragere + raționamentul
  - ticket_result    : rezultatul bileteilor după extragere + ce a învățat
  - strategy_code    : strategie nouă scrisă de AI (cod), cu rezultatul backtest
  - web_research     : ce a citit de pe web și ce idei a extras
  - self_improvement : modificări aduse propriilor prompt-uri / configurări
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"

LEDGER_VERSION = 1
MAX_GLOBAL_ENTRIES = 2000
MAX_READ_ENTRIES = 500

_WRITE_LOCK = threading.Lock()

VALID_TYPES = {
    "ai_analysis",
    "ai_tickets",
    "ticket_result",
    "strategy_code",
    "web_research",
    "self_improvement",
    "ai_action",
    "system",
}


def _known_lottery(lottery_id: str | None) -> bool:
    if not lottery_id:
        return False
    try:
        from lottery_config import LOTTERIES
    except Exception:
        return lottery_id.replace("_", "").isalnum()
    return lottery_id in LOTTERIES


def _ledger_path(lottery_id: str | None) -> Path:
    # ID-urile vin și din răspunsurile modelului AI („loto_pl" inventat) — doar
    # loteriile cunoscute primesc ledger propriu (și nicio cale în afara DATA_DIR).
    if _known_lottery(lottery_id):
        return DATA_DIR / lottery_id / "decision_ledger.jsonl"
    return DATA_DIR / "decision_ledger_global.jsonl"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_entry(
    entry_type: str,
    lottery_id: str | None = None,
    thought: str = "",
    action: str = "",
    why: str = "",
    payload: dict | None = None,
    status: str = "done",
) -> dict:
    """Adaugă o intrare în ledger. Returnează intrarea scrisă."""
    if entry_type not in VALID_TYPES:
        entry_type = "system"
    entry = {
        # id unic — două intrări în aceeași milisecundă nu trebuie să se dedupe
        "id": f"{int(time_now_ms())}-{entry_type}-{uuid.uuid4().hex[:8]}",
        "ts": _utc_now_iso(),
        "type": entry_type,
        "lottery": lottery_id,
        "thought": (thought or "")[:4000],
        "action": (action or "")[:2000],
        "why": (why or "")[:4000],
        "status": status,
        "payload": _jsonable(payload or {}),
    }
    path = _ledger_path(lottery_id)
    global_path = _ledger_path(None)
    line = json.dumps(entry, ensure_ascii=False)
    with _WRITE_LOCK:
        try:
            if path != global_path:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            with open(global_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            return entry
    _truncate_global()
    return entry


def time_now_ms() -> float:
    import time

    return time.time() * 1000


def _jsonable(obj):
    """Convertește recursiv obiecte non-JSON (date, set-uri etc.)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def _truncate_global() -> None:
    """Păstrează global-ul la MAX_GLOBAL_ENTRIES (per-lottery nu e trunchiat)."""
    global_path = _ledger_path(None)
    try:
        if not global_path.exists():
            return
        with open(global_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) <= MAX_GLOBAL_ENTRIES:
            return
        tmp = global_path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(lines[-MAX_GLOBAL_ENTRIES:])
        os.replace(tmp, global_path)
    except OSError:
        pass


def read_entries(
    lottery_id: str | None = None,
    limit: int = 100,
    entry_type: str | None = None,
) -> list[dict]:
    """Citește ultimele `limit` intrări (cele mai noi primele)."""
    if lottery_id and not _known_lottery(lottery_id):
        return []
    path = _ledger_path(lottery_id)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []
    entries: list[dict] = []
    for line in reversed(lines[-MAX_READ_ENTRIES:]):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry_type and item.get("type") != entry_type:
            continue
        entries.append(item)
        if len(entries) >= max(1, min(limit, MAX_READ_ENTRIES)):
            break
    return entries


def update_latest_result(
    match: dict,
    result: dict,
    lottery_id: str | None = None,
) -> bool:
    """Atașează rezultatul celei mai recente intrări care se potrivește cu `match`.

    Nu rescrie istoricul în mod agresiv: doar completează câmpul `result` al
    primei (cea mai noi) intrări care corespunde tipului/loteriei și nu are
    deja un rezultat.
    """
    path = _ledger_path(lottery_id)
    if not path.exists():
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return False

    changed = False
    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx].strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not _matches(item, match):
            continue
        if item.get("result"):
            continue
        item["result"] = _jsonable(result)
        item["status"] = "evaluated"
        lines[idx] = json.dumps(item, ensure_ascii=False) + "\n"
        changed = True
        break

    if changed:
        # scriere atomică sub lock: un append concurent nu poate fi pierdut
        # într-o rescriere pe jumătate
        with _WRITE_LOCK:
            try:
                tmp = path.with_suffix(path.suffix + ".tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                os.replace(tmp, path)
            except OSError:
                return False
    return changed


def _matches(item: dict, match: dict) -> bool:
    """Potrivire parțială: sub-dicționarele din `match` sunt submulțimi.

    Ex. {"payload": {"id": "act_1"}} se potrivește cu o intrare al cărei payload
    are și alte chei (risk, needs_approval...). Egalitatea strictă pe dict
    făcea ca rezultatele acțiunilor AI să nu fie atașate niciodată.
    """
    for key, expected in match.items():
        actual = item.get(key)
        if isinstance(expected, dict):
            if not isinstance(actual, dict) or not _matches(actual, expected):
                return False
        elif actual != expected:
            return False
    return True
