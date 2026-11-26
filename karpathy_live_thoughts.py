"""
Karpathy Live Thoughts - trimite ganduri in timp real catre OpenClawThink dashboard
Follows the same pattern as pi_auto_thinker.py from OpenClawThink
"""
import requests
import time
from datetime import datetime
from enum import Enum

OPENCLAW_API = "http://localhost:8001/api/thought"
ENABLED = True  # Set to False to disable thought broadcasting


class ThoughtCategory(Enum):
    THINKING = "thinking"
    ANALYZE = "analyze"
    ACTION = "action"
    SOLUTION = "solution"
    ERROR = "error"
    SUCCESS = "success"
    PROGRESS = "progress"
    INFO = "info"


def _send_thought(message: str, category: str = "thinking", icon: str = "🤖") -> bool:
    """Trimite o singura gandire catre OpenClawThink dashboard"""
    if not ENABLED:
        return False
    try:
        payload = {
            "thought": f"[KAR] [Karpathy-Agent] [{category.upper()}] {message}",
            "category": category
        }
        resp = requests.post(OPENCLAW_API, json=payload, timeout=2)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def think(message: str) -> bool:
    """Gandire generala"""
    return _send_thought(message, "thinking")


def analyze(message: str) -> bool:
    """Analizare"""
    return _send_thought(message, "analyze")


def action(message: str) -> bool:
    """Actiune luata"""
    return _send_thought(message, "action")


def solution(message: str) -> bool:
    """Solutie descoperita"""
    return _send_thought(message, "solution")


def error(message: str) -> bool:
    """Eroare intalnita"""
    return _send_thought(message, "error")


def success(message: str) -> bool:
    """Succes"""
    return _send_thought(message, "success")


def progress(message: str) -> bool:
    """Progres"""
    return _send_thought(message, "progress")


def info(message: str) -> bool:
    """Info general"""
    return _send_thought(message, "info")


# ─── Karpathy-specific thought emitters ───

def thought_experiment_started(experiment_name: str, description: str = "") -> bool:
    """Un experiment nou a inceput"""
    msg = f"Started: {experiment_name}"
    if description:
        msg += f" - {description[:60]}"
    return _send_thought(msg, "action", "⚡")


def thought_experiment_complete(experiment_name: str, metric: float, status: str) -> bool:
    """Un experiment s-a incheiat"""
    icon = "✅" if status == "KEEP" else "❌"
    msg = f"{experiment_name}: metric={metric:.6f} → {status}"
    return _send_thought(msg, "progress", icon)


def thought_new_best(strategy: str, metric: float, improvement: float = None) -> bool:
    """NEW BEST gasit!"""
    msg = f"🎉 NEW BEST! {strategy} = {metric:.6f}"
    if improvement is not None:
        msg += f" (+{improvement:.4f}%)"
    return _send_thought(msg, "success", "🎉")


def thought_discard(reason: str, metric: float = None) -> bool:
    """Experiment descartat"""
    msg = f"DISCARD: {reason}"
    if metric is not None:
        msg += f" (metric={metric:.6f})"
    return _send_thought(msg, "info", "➖")


def thought_auto_loop_start(lottery_id: str) -> bool:
    """Auto-loop a pornit"""
    msg = f"AutoLoop STARTED for {lottery_id}"
    return _send_thought(msg, "action", "🚀")


def thought_auto_loop_stop(lottery_id: str, reason: str = "") -> bool:
    """Auto-loop s-a oprit"""
    msg = f"AutoLoop STOPPED for {lottery_id}"
    if reason:
        msg += f" - {reason}"
    return _send_thought(msg, "info", "⏹")


def thought_analyzing_results(lottery_id: str, experiment_count: int) -> bool:
    """Analizez rezultatele"""
    msg = f"Analyzing {experiment_count} experiments for {lottery_id}"
    return _send_thought(msg, "analyze", "🔍")


def thought_strategy_evolving(best_strategy: str, best_metric: float) -> bool:
    """Evolutie strategie"""
    msg = f"Best strategy: {best_strategy} (score={best_metric:.6f})"
    return _send_thought(msg, "solution", "🧬")


def thought_reading_program(agenda_item: str) -> bool:
    """Citesc program.md pentru urmatoarea idee"""
    msg = f"Reading agenda: {agenda_item[:50]}"
    return _send_thought(msg, "analyze", "📋")


def thought_commit(commit_hash: str, description: str) -> bool:
    """Git commit facut"""
    msg = f"Commit {commit_hash[:7]}: {description[:50]}"
    return _send_thought(msg, "action", "💾")


if __name__ == "__main__":
    # Test thought broadcasting
    print("Testing Karpathy Live Thoughts...")
    think("Test thought from Karpathy")
    success("NEW BEST! test_strategy = 2.555000")
    progress("AutoLoop running experiment 42...")
    info("Reading program.md for next idea...")
    print("Done! Check OpenClawThink dashboard.")
