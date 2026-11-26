"""Telegram bot — canalul de comunicare AI → utilizator.

Trimite notificări și cereri de aprobare (buton inline ✅/❌) către chat-ul
configurat. Funcționează cu TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID din
variabilele de mediu sau fișierul .env / .telegram_chat_id din rădăcina
proiectului. Toate funcțiile sunt fire-safe și NU aruncă excepții — o problema
de Telegram nu trebuie să blocheze vreodată agenții.

Limitare de rată integrată: minim NOTIFY_MIN_INTERVAL secunde între mesaje,
cu coadă mică și renunțare politicoasă la depășiri (fără spam).
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_DIR = Path(__file__).parent

API_BASE = "https://api.telegram.org/bot{token}/{method}"
NOTIFY_MIN_INTERVAL = 20.0  # secunde între mesaje (anti-spam)
MAX_QUEUE = 6

_lock = threading.Lock()
_last_sent = [0.0]
_queue_dropped = [0]


def _env_from_file(name: str) -> str | None:
    """Citește o variabilă din .env (fără dependințe externe)."""
    env_file = PROJECT_DIR / ".env"
    try:
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith(f"{name}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        return None
    return None


def get_bot_token() -> str | None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or _env_from_file("TELEGRAM_BOT_TOKEN")
    return token or None


def get_chat_id() -> str | None:
    chat = os.environ.get("TELEGRAM_CHAT_ID") or _env_from_file("TELEGRAM_CHAT_ID")
    if chat:
        return chat
    # fallback: fișierul istoric (conține doar cifre)
    chat_file = PROJECT_DIR / ".telegram_chat_id"
    try:
        if chat_file.exists():
            digits = "".join(ch for ch in chat_file.read_text(encoding="utf-8") if ch.isdigit())
            # fișierul istoric are id-ul dublat; ia prima jumătate
            if len(digits) % 2 == 0 and digits[: len(digits) // 2] == digits[len(digits) // 2:]:
                digits = digits[: len(digits) // 2]
            return digits or None
    except OSError:
        return None
    return None


def telegram_configured() -> bool:
    return bool(get_bot_token() and get_chat_id())


def _api_call(method: str, payload: dict, timeout: int = 15) -> dict | None:
    token = get_bot_token()
    if not token:
        return None
    url = API_BASE.format(token=token, method=method)
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def send_message(text: str, reply_markup: dict | None = None, chat_id: str | None = None) -> bool:
    """Trimite un mesaj (HTML). Nichilist pe eroare; limitează rata."""
    target = chat_id or get_chat_id()
    if not target or not get_bot_token():
        return False
    with _lock:
        now = time.time()
        wait = NOTIFY_MIN_INTERVAL - (now - _last_sent[0])
        if wait > 0:
            time.sleep(min(wait, NOTIFY_MIN_INTERVAL))
        ok = False
        try:
            payload: dict = {
                "chat_id": target,
                "text": text[:4000],
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }
            if reply_markup:
                payload["reply_markup"] = reply_markup
            result = _api_call("sendMessage", payload)
            ok = bool(result and result.get("ok"))
        except Exception:
            ok = False
        if ok:
            _last_sent[0] = time.time()
            if _queue_dropped[0]:
                _queue_dropped[0] = 0
        else:
            _queue_dropped[0] += 1
        return ok


def notify(text: str) -> bool:
    """Notificare informativă (fără butoane). Sigură de apelat din orice agent."""
    try:
        return send_message(text)
    except Exception:
        return False


def action_keyboard(action_id: str) -> dict:
    """Tastatură inline ✅ Aprob / ❌ Refuz pentru o acțiune propusă."""
    return {
        "inline_keyboard": [[
            {"text": "✅ Aprob", "callback_data": f"approve:{action_id}"},
            {"text": "❌ Refuz", "callback_data": f"reject:{action_id}"},
        ]]
    }


def send_action_request(title: str, why: str, action_id: str) -> bool:
    """Mesaj de propunere cu butoane de aprobare."""
    text = f"🤖 <b>AI propune:</b> {title}\n\n<i>{why[:800]}</i>\n\nAprobi execuția?"
    return send_message(text, reply_markup=action_keyboard(action_id))


def send_action_result(action_id: str, ok: bool, detail: str) -> bool:
    icon = "✅" if ok else "⚠️"
    return notify(f"{icon} <b>Acțiune {action_id}</b>: {('executată' if ok else 'eșuată')}\n{detail[:600]}")


def answer_callback(callback_query_id: str, text: str) -> None:
    _api_call("answerCallbackQuery", {"callback_query_id": callback_query_id, "text": text[:190]})


def get_updates(offset: int | None = None, timeout: int = 25) -> list[dict]:
    """Long-polling getUpdates. Returnează lista de update-uri (poate fi vidă)."""
    token = get_bot_token()
    if not token:
        return []
    params = {"timeout": max(1, min(50, timeout)), "allowed_updates": json.dumps(["message", "callback_query"])}
    if offset:
        params["offset"] = offset
    url = API_BASE.format(token=token, method="getUpdates") + "?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout + 10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        if result.get("ok"):
            return result.get("result", [])
    except Exception:
        pass
    return []


def set_webhook_delete() -> None:
    """Asigură că nu există webhook (long-polling are nevoie de asta)."""
    _api_call("deleteWebhook", {})
