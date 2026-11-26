"""Ollama Strategist Agent

Folosește modelele Ollama (inclusiv modelele cloud din Ollama Cloud, accesate
prin serverul local Ollama) pentru a:
  - investiga periodic numerele și strategiile fiecărei loterii
  - salva analize + numere propuse pentru următoarea extragere
    (data/{lottery}/ai_predictions.json)
  - propune ajustări de parametri către auto-loop (agents/strategies/)

Frecvența e controlabilă prin agents/config.json:
    "agents": { "ollama_strategist": { "interval_minutes": 360, "lotteries": [...],
                "model": "glm-5.3-flash:cloud" } }
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

from agent_base import BaseAgent, AGENTS_DIR

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from lottery_config import list_lottery_configs

STRATEGIES_DIR = AGENTS_DIR / "strategies"


class OllamaStrategistAgent(BaseAgent):
    """Agent AI (Ollama) pentru analiza numerelor și strategiilor."""

    def __init__(self):
        super().__init__("ollama_strategist", poll_interval=60)
        config = self.get_config()
        self.interval_minutes = int(config.get("interval_minutes", 360))
        self.model = config.get("model") or None
        # Lotteries to prioritize first (default: Romania + Italy, apoi restul)
        default_ids = [c.id for c in list_lottery_configs()]
        configured = config.get("lotteries")
        self.lottery_order = configured if isinstance(configured, list) and configured else default_ids
        self._last_run: dict[str, float] = {}
        self._last_attempt: float = 0.0

    def _export_suggestions(self, lottery_id: str, record: dict) -> None:
        """Trimite sugestiile AI către auto-loop prin strategies/pending_review.json."""
        suggestions = record.get("strategy_suggestions") or []
        if not suggestions:
            return
        try:
            STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)
            path = STRATEGIES_DIR / "pending_review.json"
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            if not isinstance(data, list):
                data = []
            data.append({
                "id": f"ai_{lottery_id}_{record.get('generated_at', '')}",
                "source": "ollama_strategist",
                "lottery": lottery_id,
                "model": record.get("model"),
                "created_at": datetime.utcnow().isoformat(),
                "suggestions": suggestions,
                "status": "pending",
            })
            path.write_text(json.dumps(data[-100:], ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception as exc:
            self.log_error(f"Failed to export suggestions for {lottery_id}: {exc}")

    def _prediction_is_fresh(self, lottery_id: str) -> bool:
        """True dacă există deja o analiză mai nouă decât intervalul — evită
        re-analize inutile după un restart al agentului."""
        try:
            import ollama_ai

            record = ollama_ai.load_predictions(lottery_id)
            if not record:
                return False
            generated = record.get("generated_at")
            if not generated:
                return False
            age = time.time() - datetime.fromisoformat(generated).timestamp()
            return age < self.interval_minutes * 60
        except Exception:
            return False

    def process(self) -> None:
        # Evită reacumularea agresivă dacă Ollama pică: maxim 1 încercare / 5 min
        if time.time() - self._last_attempt < 300:
            return

        now = time.time()
        interval_s = self.interval_minutes * 60
        due_lotteries = [
            lottery_id for lottery_id in self.lottery_order
            if now - self._last_run.get(lottery_id, 0) >= interval_s
            and not self._prediction_is_fresh(lottery_id)
        ]
        if not due_lotteries:
            next_in = min(
                (interval_s - (now - self._last_run.get(l, 0)) for l in self.lottery_order if l in self._last_run),
                default=0,
            )
            self.update_heartbeat(
                "idle",
                current_task=f"next AI sweep in {max(0, int(next_in // 60))} min",
            )
            return

        self._last_attempt = time.time()

        try:
            import ollama_ai

            if not ollama_ai.ollama_available():
                self.log_warning(f"Ollama not reachable at {ollama_ai.OLLAMA_HOST}; retrying later")
                self.update_heartbeat("degraded", current_task="ollama unreachable")
                return

            from load_data import load_lottery_draws

            ran_any = False
            for lottery_id in due_lotteries:
                if stop_requested_for(self.agent_id):
                    self.log_info("Stop requested via message file")
                    break
                try:
                    draws = load_lottery_draws(lottery_id)
                except Exception:
                    draws = []
                if not draws:
                    continue  # fără istoric descărcat → sări până apare
                self.update_heartbeat("running", current_task=f"AI analyzing {lottery_id}")
                try:
                    # model=None → ollama_ai încearcă modelele preferate în ordine,
                    # cu fallback automat dacă un model cloud atârnă sau răspunde prost
                    record = ollama_ai.analyze_lottery(lottery_id, model=self.model)
                    self._export_suggestions(lottery_id, record)
                    self._last_run[lottery_id] = time.time()
                    ran_any = True
                    self.log_success(
                        f"AI analysis for {lottery_id}: model={record['model']} "
                        f"confidence={record['confidence']} "
                        f"top5={record['recommended_numbers'][:5]} "
                        f"suggestions={len(record['strategy_suggestions'])} "
                        f"({record['elapsed_seconds']}s)"
                    )
                except Exception as exc:
                    self.log_error(f"AI analysis failed for {lottery_id}: {exc}")

            if ran_any:
                self.update_heartbeat(
                    "running",
                    current_task=f"AI sweep done; next in {self.interval_minutes} min",
                )
        except ImportError as exc:
            self.log_error(f"ollama_ai module unavailable: {exc}")


def stop_requested_for(agent_id: str) -> bool:
    from agent_base import COMMANDS_DIR, is_fresh_message

    command_file = COMMANDS_DIR / f"to_{agent_id}.json"
    try:
        messages = json.loads(command_file.read_text(encoding="utf-8"))
        # Doar cereri recente — un stop vechi rămas în fișier ar întrerupe orice analiză viitoare.
        return any(m.get("type") == "stop_request" and is_fresh_message(m)
                   for m in messages if isinstance(m, dict))
    except Exception:
        return False


def main():
    agent = OllamaStrategistAgent()
    agent.run()


if __name__ == "__main__":
    main()
