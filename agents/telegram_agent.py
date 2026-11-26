"""Telegram Agent — puntea bidirecțională dintre AI și utilizator.

Long-polling pe botul Telegram:
  • butoanele ✅ Aprob / ❌ Refuz execută/resping acțiunile propuse de AI
    (prin ai_actions.py), cu ledger + confirmare în chat;
  • comenzi utile: /status, /activity, /best, /pending, /help;
  • trimite notificările generate de restul agenților (propuneri, promovări,
    rezultate bilete) — acelea merg direct prin ai_actions.notify_event.

Dacă token-ul/chat-id lipsesc, agentul intră în idle quiet și reîncearcă.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from agent_base import BaseAgent

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import ai_actions
import telegram_bot
from decision_ledger import read_entries


class TelegramAgent(BaseAgent):

    def __init__(self):
        super().__init__("telegram_agent", poll_interval=5)
        self._update_offset: int | None = None
        self._configured = False

    def process(self) -> None:
        if not telegram_bot.telegram_configured():
            if self._configured:
                self.log_warning("Telegram config pierdut — idle")
                self._configured = False
            self.update_heartbeat("degraded", current_task="telegram neconfigurat (TOKEN/CHAT_ID)")
            time.sleep(15)
            return
        if not self._configured:
            self._configured = True
            telegram_bot.set_webhook_delete()
            self.log_success("Telegram activ — long polling pornit")
            telegram_bot.notify(
                "🤖 <b>AutoResearch online.</b> Comenzi: /status, /activity, /best, /pending, /help. "
                "Când AI-ul vrea să facă o acțiune, primești aici butoane de ✅ Aprob / ❌ Refuz."
            )

        self.update_heartbeat("running", current_task="ascult comenzi și aprobări")

        allowed_chat = str(telegram_bot.get_chat_id() or "")
        updates = telegram_bot.get_updates(offset=self._update_offset, timeout=25)
        for update in updates:
            self._update_offset = max(self._update_offset or 0, int(update.get("update_id", 0))) + 1
            if not self._from_allowed_chat(update, allowed_chat):
                continue
            try:
                if "callback_query" in update:
                    self._handle_callback(update["callback_query"])
                elif "message" in update:
                    self._handle_message(update["message"])
            except Exception as exc:
                self.log_error(f"Eroare la procesarea update-ului: {exc}")

    @staticmethod
    def _from_allowed_chat(update: dict, allowed_chat: str) -> bool:
        """Doar chat-ul configurat poate da comenzi sau aproba acțiuni —
        botul e public: oricine îl găsește i-ar putea trimite /status sau callback-uri."""
        if "callback_query" in update:
            chat = (update["callback_query"].get("message") or {}).get("chat") or {}
        else:
            chat = (update.get("message") or {}).get("chat") or {}
        return bool(allowed_chat) and str(chat.get("id", "")) == allowed_chat

    # ─── aprobări ────────────────────────────────────────────────────────

    def _handle_callback(self, query: dict) -> None:
        data = str(query.get("data", ""))
        query_id = str(query.get("id", ""))
        # formatul callback_data: "<verdict>:<action_id>" (ex. approve:act_123_abc)
        verdict, _, action_id = data.partition(":")
        if verdict not in {"approve", "reject"} or not action_id:
            telegram_bot.answer_callback(query_id, "Comandă necunoscută")
            return
        if verdict == "approve":
            result = ai_actions.approve_action(action_id, approved_by="telegram")
            if result.get("ok"):
                telegram_bot.answer_callback(query_id, "Executat ✅")
            else:
                telegram_bot.answer_callback(query_id, result.get("error", "eroare"))
        else:
            result = ai_actions.reject_action(action_id, by="telegram")
            telegram_bot.answer_callback(query_id, "Refuzat" if result.get("ok") else "deja procesat")

    # ─── comenzi text ────────────────────────────────────────────────────

    def _handle_message(self, message: dict) -> None:
        text = str(message.get("text", "")).strip()
        chat_id = str(message.get("chat", {}).get("id", ""))
        if not text.startswith("/"):
            return
        command = text.split()[0].split("@")[0].lower()
        if command == "/status":
            telegram_bot.send_message(self._status_text(), chat_id=chat_id)
        elif command == "/activity":
            telegram_bot.send_message(self._activity_text(), chat_id=chat_id)
        elif command == "/best":
            telegram_bot.send_message(self._best_text(), chat_id=chat_id)
        elif command == "/pending":
            pending = ai_actions.list_pending()
            if not pending:
                telegram_bot.send_message("Nicio acțiune în așteptarea aprobării.", chat_id=chat_id)
            else:
                lines = [f"• <b>{a['type']}</b> {json.dumps(a.get('params', {}), ensure_ascii=False)[:120]}\n  <i>{a.get('why', '')[:140]}</i>"
                         for a in pending[:8]]
                telegram_bot.send_message("⏳ Acțiuni în așteptare:\n" + "\n".join(lines), chat_id=chat_id)
        elif command == "/help":
            telegram_bot.send_message(
                "<b>Comenzi:</b>\n/status — starea sistemului\n/activity — ce face agentul acum\n"
                "/best — cele mai bune strategii per loterie\n/pending — acțiuni ce așteaptă aprobarea\n"
                "/help — această listă", chat_id=chat_id)
        else:
            telegram_bot.send_message("Comandă necunoscută. Încearcă /help", chat_id=chat_id)

    # ─── texte de răspuns ────────────────────────────────────────────────

    @staticmethod
    def _status_text() -> str:
        import auto_loop
        from lottery_config import list_lottery_configs

        lines = ["<b>Stare sistem</b>"]
        loops = 0
        for config in list_lottery_configs():
            try:
                state, _active = auto_loop.load_search_state_for_lottery(config.id)
                if state.best_metric > 0:
                    lines.append(f"• {config.name}: best <b>{state.best_metric:.4f}</b> "
                                 f"(random {config.baseline:.3f})")
                if config.lock_file.exists():
                    from lock_utils import read_lock_owner

                    pid = read_lock_owner(config.lock_file).pid
                    if pid:
                        import ctypes

                        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid)) \
                            if sys.platform == "win32" else 1
                        if handle:
                            if sys.platform == "win32":
                                ctypes.windll.kernel32.CloseHandle(handle)
                            loops += 1
            except Exception:
                continue
        try:
            import ollama_ai as _oa

            llm = f"Ollama: ✅ {_oa.resolve_model()}" if _oa.ollama_available() else "Ollama: ❌ offline"
        except Exception:
            llm = "Ollama: ❓"
        lines.append(f"\n{llm}\nBucle active: {loops}")
        return "\n".join(lines)

    @staticmethod
    def _activity_text() -> str:
        entries = read_entries(None, limit=6)
        if not entries:
            return "Ledger gol — agentul nu a luat încă nicio decizie."
        icons = {"ai_analysis": "✦", "ai_tickets": "🎟", "ticket_result": "✅",
                 "strategy_code": "⌨", "web_research": "🌐", "self_improvement": "🧠",
                 "ai_action": "🎬", "system": "•"}
        lines = ["<b>Ultimele decizii AI:</b>"]
        for e in entries:
            icon = icons.get(e.get("type"), "•")
            lines.append(f"{icon} [{(e.get('ts') or '')[11:16]}] {e.get('lottery') or '—'}: "
                         f"{(e.get('action') or '')[:110]}")
        return "\n".join(lines)

    @staticmethod
    def _best_text() -> str:
        import auto_loop
        from lottery_config import list_lottery_configs

        lines = ["<b>Cele mai bune strategii</b>"]
        for config in list_lottery_configs():
            try:
                state, _active = auto_loop.load_search_state_for_lottery(config.id)
                if state.best_metric > 0:
                    edge = (state.best_metric / config.baseline - 1.0) * 100
                    lines.append(f"• {config.name}: <b>{state.best_metric:.4f}</b> "
                                 f"({edge:+.1f}% vs random {config.baseline:.3f})")
            except Exception:
                continue
        return "\n".join(lines)


def main():
    agent = TelegramAgent()
    agent.run()


if __name__ == "__main__":
    main()
