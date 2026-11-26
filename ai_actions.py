"""AI Action Bus — controlul complet al proiectului de către AI, cu ledgers.

AI-ul poate PROPUNE acțiuni asupra proiectului (repornirea buclelor, aplicarea
parametrilor, rulări walkforward, actualizarea istoricului, ajustarea propriei
configurări). Fiecare acțiune:
  1. se validează pe whitelist,
  2. se înscrie în Decision Ledger (status „proposed"),
  3. se trimite pe Telegram cu butoane ✅ Aprob / ❌ Refuz,
  4. la aprobare se execută și rezultatul se notează în ledger + Telegram.

Evenimentele informative (strategie promovată, rezultate bilete, erori) merg
pe notify_event() — doar informare, fără aprobare.

Stocare: data/ai_actions/pending.json (în așteptare) + data/ai_actions/history.json
(ultimele 200 executate/respinse).
"""

from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from decision_ledger import append_entry, update_latest_result

ACTIONS_DIR = PROJECT_DIR / "data" / "ai_actions"
MAX_HISTORY = 200

_lock = threading.Lock()


def _pending_file() -> Path:
    # dinamic — testează pot izola cu monkeypatch pe ACTIONS_DIR
    return ACTIONS_DIR / "pending.json"


def _autonomous_mode() -> bool:
    """True (implicit): AI-ul ia decizii singur — notifică, nu cere aprobare.

    Se poate dezactiva din agents/config.json: ai_actions.autonomous_mode=false
    — atunci acțiunile medium/high risk așteaptă butoanele ✅/❌ din Telegram.
    """
    try:
        cfg = json.loads((PROJECT_DIR / "agents" / "config.json").read_text(encoding="utf-8"))
        value = cfg.get("ai_actions", {}).get("autonomous_mode", True)
        return bool(value)
    except (OSError, json.JSONDecodeError):
        return True


def _history_file() -> Path:
    return ACTIONS_DIR / "history.json"


# ─── Whitelist & descrieri ───────────────────────────────────────────────────

ACTION_TYPES: dict[str, dict] = {
    "restart_loop": {
        "title": "Repornire buclă de optimizare",
        "params": {"lottery": str},
        "risk": "medium",
    },
    "stop_loop": {
        "title": "Oprire buclă de optimizare",
        "params": {"lottery": str},
        "risk": "medium",
    },
    "start_loop": {
        "title": "Pornire buclă de optimizare",
        "params": {"lottery": str},
        "risk": "low",
    },
    "apply_strategy_params": {
        "title": "Aplicare parametri de strategie propuși de AI",
        "params": {"lottery": str, "params": dict},
        "risk": "high",
    },
    "run_walkforward": {
        "title": "Rulare walkforward (validare out-of-sample)",
        "params": {"lottery": str},
        "risk": "low",
    },
    "update_history": {
        "title": "Actualizare istoric extrageri (scrape)",
        "params": {},
        "risk": "low",
    },
    "run_ai_analysis": {
        "title": "Analiză AI imediată pe o loterie",
        "params": {"lottery": str},
        "risk": "low",
    },
    "set_agent_config": {
        "title": "Ajustare configurare agenți (interval/capacitate)",
        "params": {"path": str, "value": (int, float, str, bool)},
        "risk": "high",
    },
}

ALLOWED_CONFIG_PATHS = {
    "agents.ai_strategy_coder.interval_minutes",
    "agents.ollama_strategist.interval_minutes",
    "agents.ai_ticket_agent.tickets_per_day",
    "agents.lottery_manager.loop_rotation_hours",
    "system.max_simultaneous_lotteries",
}


def _read_json(path: Path, default):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if data is not None else default
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def _validate_params(action_type: str, params: dict) -> str | None:
    spec = ACTION_TYPES.get(action_type)
    if spec is None:
        return f"tip de acțiune necunoscut: {action_type}"
    required = spec["params"]
    for key, expected in required.items():
        if key not in params:
            return f"parametrul lipsă: {key}"
        if not isinstance(params[key], expected):
            return f"parametrul {key} are tip greșit (așteptat {getattr(expected, '__name__', expected)})"
    if action_type == "set_agent_config":
        if params["path"] not in ALLOWED_CONFIG_PATHS:
            return f"cale de configurare nepermisă: {params['path']}"
    if "lottery" in params:
        from lottery_config import LOTTERIES

        if params["lottery"] not in LOTTERIES:
            return f"loterie necunoscută: {params['lottery']}"
    return None


# ─── Propunere / executare ───────────────────────────────────────────────────


def propose_action(
    action_type: str,
    params: dict | None = None,
    why: str = "",
    proposed_by: str = "ai",
    require_approval: bool | None = None,
) -> dict:
    """Propune o acțiune: ledger + Telegram (aprobare) sau execuție directă."""
    params = params or {}
    error = _validate_params(action_type, params)
    action = {
        "id": f"act_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}",
        "type": action_type,
        "params": params,
        "why": (why or "")[:1500],
        "proposed_by": proposed_by[:80],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if error:
        action.update({"status": "invalid", "error": error})
        append_entry(
            "ai_action", lottery_id=params.get("lottery"),
            thought=f"Propunere invalidă: {action_type}",
            action=f"{action_type} {json.dumps(params, ensure_ascii=False)[:200]}",
            why=why, status="invalid",
            payload={"error": error, "id": action["id"]},
        )
        return action

    risk = ACTION_TYPES[action_type]["risk"]
    needs_approval = require_approval if require_approval is not None else (
        risk in {"medium", "high"} and not _autonomous_mode()
    )

    with _lock:
        pending = _read_json(_pending_file(), [])
        pending.append(action)
        _write_json(_pending_file(), pending[-50:])

    append_entry(
        "ai_action",
        lottery_id=params.get("lottery"),
        thought=f"AI ({proposed_by}) propune: {ACTION_TYPES[action_type]['title']}",
        action=f"{action_type} {json.dumps(params, ensure_ascii=False)}",
        why=why,
        status="pending_approval" if needs_approval else "executing",
        payload={"id": action["id"], "risk": risk, "needs_approval": needs_approval},
    )

    import telegram_bot

    title = ACTION_TYPES[action_type]["title"]
    if needs_approval:
        telegram_bot.send_action_request(
            f"{title}\n<code>{action_type} {json.dumps(params, ensure_ascii=False)[:200]}</code>",
            why or "(fără motivație)",
            action["id"],
        )
    else:
        _execute_and_finalize(action, approved_by="auto (risc scăzut)")
        _pop_pending(action["id"])  # executat direct → nu rămâne în așteptare
    return action


def _pop_pending(action_id: str) -> dict | None:
    with _lock:
        pending = _read_json(_pending_file(), [])
        for i, a in enumerate(pending):
            if a.get("id") == action_id:
                action = pending.pop(i)
                _write_json(_pending_file(), pending)
                return action
    return None


def approve_action(action_id: str, approved_by: str = "user") -> dict:
    """Aprobă și execută o acțiune în așteptare."""
    action = _pop_pending(action_id)
    if not action:
        return {"ok": False, "error": "acțiunea nu mai există (deja procesată?)"}
    return _execute_and_finalize(action, approved_by=approved_by)


def reject_action(action_id: str, by: str = "user", reason: str = "") -> dict:
    """Respinge o acțiune în așteptare."""
    action = _pop_pending(action_id)
    if not action:
        return {"ok": False, "error": "acțiunea nu mai există (deja procesată?)"}
    action.update({"status": "rejected", "closed_at": datetime.now().isoformat(timespec="seconds"),
                   "closed_by": by, "close_reason": reason[:300]})
    _append_history(action)
    update_latest_result(
        {"payload": {"id": action_id}}, {"approved": False, "by": by, "reason": reason[:300]},
    )
    append_entry(
        "ai_action", lottery_id=action.get("params", {}).get("lottery"),
        thought=f"Utilizatorul a REFUZAT: {ACTION_TYPES.get(action['type'], {}).get('title', action['type'])}",
        action=f"{action['type']} {json.dumps(action.get('params', {}), ensure_ascii=False)[:200]}",
        why=reason or "refuzat de utilizator",
        status="rejected",
        payload={"id": action_id},
    )
    import telegram_bot

    telegram_bot.notify(f"❌ Acțiune refuzată: {action['type']}")
    return {"ok": True, "status": "rejected"}


def _append_history(action: dict) -> None:
    with _lock:
        history = _read_json(_history_file(), [])
        history.append(action)
        _write_json(_history_file(), history[-MAX_HISTORY:])


def _execute_and_finalize(action: dict, approved_by: str) -> dict:
    started = time.time()
    try:
        detail = _execute(action["type"], action.get("params", {}))
        ok = True
        status = "executed"
    except Exception as exc:
        detail = str(exc)[:500]
        ok = False
        status = "failed"
    action.update({
        "status": status,
        "approved_by": approved_by,
        "closed_at": datetime.now().isoformat(timespec="seconds"),
        "result": detail,
        "elapsed_s": round(time.time() - started, 1),
    })
    _append_history(action)
    update_latest_result(
        {"payload": {"id": action["id"]}},
        {"approved": True, "by": approved_by, "result": detail, "ok": ok},
    )
    append_entry(
        "ai_action", lottery_id=action.get("params", {}).get("lottery"),
        thought=f"Aprobat de {approved_by} și executat.",
        action=f"{action['type']} {json.dumps(action.get('params', {}), ensure_ascii=False)[:200]}",
        why=action.get("why", ""),
        status=status,
        payload={"id": action["id"], "result": detail, "elapsed_s": action["elapsed_s"]},
    )
    import telegram_bot

    telegram_bot.send_action_result(action["id"][:18], ok, detail)
    return {"ok": ok, "status": status, "result": detail}


# ─── Executori ───────────────────────────────────────────────────────────────


def _execute(action_type: str, params: dict) -> str:
    if action_type == "start_loop":
        return _exec_start_loop(params["lottery"])
    if action_type == "stop_loop":
        return _exec_stop_loop(params["lottery"])
    if action_type == "restart_loop":
        stop_msg = _exec_stop_loop(params["lottery"])
        time.sleep(2)
        start_msg = _exec_start_loop(params["lottery"])
        return f"stop: {stop_msg}; start: {start_msg}"
    if action_type == "apply_strategy_params":
        return _exec_apply_params(params["lottery"], params["params"])
    if action_type == "run_walkforward":
        return _exec_walkforward(params["lottery"])
    if action_type == "update_history":
        p = ACTIONS_DIR.parent / ".auto_update.json"
        state = _read_json(p, {})
        state["last_completed"] = 0
        _write_json(p, state)
        return "Actualizarea istoricului a fost programată (pornește în ~60s, fundal)."
    if action_type == "run_ai_analysis":
        import ollama_ai

        record = ollama_ai.analyze_lottery(params["lottery"])
        return (f"Analiză {params['lottery']} gata ({record['model']}): "
                f"top {record['recommended_numbers'][:6]}")
    if action_type == "set_agent_config":
        return _exec_set_config(params["path"], params["value"])
    raise ValueError(f"executor lipsă pentru {action_type}")


def _exec_start_loop(lottery_id: str) -> str:
    from lottery_config import get_lottery_config

    config = get_lottery_config(lottery_id)
    try:
        config.state_dir.joinpath(".autoloop.paused").unlink(missing_ok=True)
    except OSError:
        pass
    creation_flags = 0
    if sys.platform == "win32":
        import subprocess

        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.BELOW_NORMAL_PRIORITY_CLASS
    import subprocess

    out = config.state_dir / "loop.out.log"
    err = config.state_dir / "loop.err.log"
    with out.open("a", encoding="utf-8") as fo, err.open("a", encoding="utf-8") as fe:
        proc = subprocess.Popen(
            [sys.executable, str(PROJECT_DIR / "auto_loop.py"), "--lottery", lottery_id],
            cwd=str(PROJECT_DIR), stdout=fo, stderr=fe, creationflags=creation_flags,
        )
    return f"buclă {lottery_id} pornită (PID {proc.pid})"


def _exec_stop_loop(lottery_id: str) -> str:
    import os as _os
    import signal
    import subprocess

    from lock_utils import read_lock_owner
    from lottery_config import get_lottery_config

    config = get_lottery_config(lottery_id)
    config.state_dir.mkdir(parents=True, exist_ok=True)
    # marcaj de pauză (ca butonul Stop din UI) — altfel lottery_manager
    # repornește bucla imediat și oprirea nu are efect
    config.state_dir.joinpath(".autoloop.paused").write_text(
        datetime.now().isoformat(timespec="seconds"), encoding="utf-8"
    )
    config.stop_request_file.write_text("stop\n", encoding="utf-8")
    pid = read_lock_owner(config.lock_file).pid if config.lock_file.exists() else None
    if pid:
        try:
            _os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True, timeout=20, check=False)
        time.sleep(1)
        return f"buclă {lottery_id} oprită (PID {pid})"
    return f"buclă {lottery_id}: cerere de oprire scrisă (nu era pornită)"


def _exec_apply_params(lottery_id: str, new_params: dict) -> str:
    import auto_loop
    from atomic_io import write_json_atomic
    from lottery_config import get_lottery_config

    config = get_lottery_config(lottery_id)
    auto_loop.configure_runtime(lottery_id)
    state, _active = auto_loop.load_search_state_for_lottery(lottery_id)

    numeric_fields = {"tf", "hc", "bal", "sb", "delay", "momentum", "jc", "bf", "dd",
                      "prior_strength", "ewma_lambda"}
    int_fields = {"iw", "rw"}
    # rezumatul trimis modelului (_get_best_strategy_summary) numește „bal" → „b"
    aliases = {"b": "bal"}
    applied = {}
    for key, value in new_params.items():
        key = aliases.get(key, key)
        if key in numeric_fields:
            setattr(state, key, max(0.0, min(10.0, float(value))))
            applied[key] = getattr(state, key)
        elif key in int_fields:
            setattr(state, key, max(5, min(1000, int(value))))
            applied[key] = getattr(state, key)
    if not applied:
        return "niciun parametru valid de aplicat"
    payload = {"version": 1, "search_state": state.to_dict()}
    write_json_atomic(config.loop_state_file, payload, ensure_ascii=True)
    return (f"parametri aplicați pe {lottery_id}: "
            f"{json.dumps(applied)} — preluați de buclă la următoarea pornire")


def _exec_walkforward(lottery_id: str) -> str:
    import auto_loop
    from lottery_config import get_lottery_config
    from walkforward import run_and_save_walkforward

    config = get_lottery_config(lottery_id)
    auto_loop.configure_runtime(lottery_id)
    state, _active = auto_loop.load_search_state_for_lottery(lottery_id)
    rank_fn = auto_loop.build_rank_fn_from_state(state, config=config)
    wf = run_and_save_walkforward(
        lottery_id=lottery_id, rank_fn=rank_fn,
        recent_window=state.rw, backtest_score=state.best_metric,
        strategy_name=state.best_strategy or "auto",
    )
    if wf is None:
        return "walkforward n-a putut rula (date insuficiente?)"
    return (f"WF score={wf.wf_score:.6f}, gap={wf.gap:.6f}, stabilitate={wf.stability:.4f}")


def _exec_set_config(path: str, value) -> str:
    cfg_path = PROJECT_DIR / "agents" / "config.json"
    cfg = _read_json(cfg_path, {})
    node = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        node = node[int(part)] if part.isdigit() else node.setdefault(part, {})
    leaf = parts[-1]
    # modelele trimit uneori numerele ca string — convertește politicos
    if isinstance(value, str):
        try:
            value = float(value) if "." in value else int(value)
        except ValueError:
            pass
    old = node.get(leaf)
    node[leaf] = value
    _write_json(cfg_path, cfg)
    return f"config {path}: {old!r} → {value!r} (agenții îl citesc la următorul ciclu)"


# ─── Interogări ──────────────────────────────────────────────────────────────


def list_pending() -> list[dict]:
    return _read_json(_pending_file(), [])


def list_history(limit: int = 30) -> list[dict]:
    history = _read_json(_history_file(), [])
    return list(reversed(history[-limit:]))


def notify_event(text: str) -> bool:
    """Notificare informativă către utilizator (fără aprobare)."""
    import telegram_bot

    return telegram_bot.notify(text)
