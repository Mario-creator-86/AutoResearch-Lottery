"""Stocare partajată pentru biletele propuse de AI (data/{id}/ai_tickets.json).

Modul neutru (fără dependințe de agents/) ca să poată fi importat atât de
agenți, cât și de server.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"

TICKETS_VERSION = 1
MAX_STORED_DRAWS = 120


def tickets_file(lottery_id: str) -> Path:
    return DATA_DIR / lottery_id / "ai_tickets.json"


def load_ticket_store(lottery_id: str) -> dict:
    try:
        data = json.loads(tickets_file(lottery_id).read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("entries"), list):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"version": TICKETS_VERSION, "entries": []}


def save_ticket_store(lottery_id: str, store: dict) -> None:
    # atomic: serverul citește fișierul în paralel; o citire pe jumătate ar
    # returna un store gol, iar agentul l-ar rescrie → istoric pierdut
    from atomic_io import write_text_atomic

    store["entries"] = store["entries"][-MAX_STORED_DRAWS:]
    write_text_atomic(tickets_file(lottery_id), json.dumps(store, ensure_ascii=False, indent=1))


def summarize_tickets(lottery_id: str, store: dict) -> dict:
    """Progres cumulat al biletelor AI pentru o loterie.

    - din toate biletele jucate, câte numere am nimerit (distribuție pe lovituri)
    - cost aproximativ (toate biletele: jucate + în joc) × prețul variantei simple
    - câștig aproximativ (suma premiilor estimate pe categorii)
    Merge și pe intrări vechi fără „prize" per rezultat (le recalculează).
    """
    from prize_estimates import estimate_ticket_prize, ticket_price

    checked = [e for e in store.get("entries", []) if e.get("status") == "checked"]
    pending = [e for e in store.get("entries", []) if e.get("status") == "pending"]

    tickets_played = sum(len(e.get("tickets", [])) for e in checked)
    tickets_in_play = sum(len(e.get("tickets", [])) for e in pending)

    distribution: dict[int, int] = {}
    total_hits = 0
    best_hit = 0
    winnings = 0.0
    currency = ""
    for entry in checked:
        for r in entry.get("results") or []:
            hits = int(r.get("hit_count", 0) or 0)
            total_hits += hits
            best_hit = max(best_hit, hits)
            distribution[hits] = distribution.get(hits, 0) + 1
            # recalculat mereu din tabelul curent: premiile stocate la verificare pot
            # proveni dintr-un tabel mai vechi (ex. cel umflat de dinainte de 2026-09-21)
            bonus_hits = len(r.get("bonus_hits") or [])
            prize = estimate_ticket_prize(lottery_id, hits, bonus_hits)
            if isinstance(prize, dict) and prize.get("amount"):
                winnings += float(prize["amount"])
                currency = currency or str(prize.get("currency") or "")

    price_amount, price_currency = ticket_price(lottery_id)
    total_tickets = tickets_played + tickets_in_play
    return {
        "draws_played": len(checked),
        "draws_in_play": len(pending),
        "tickets_played": tickets_played,
        "tickets_in_play": tickets_in_play,
        "total_hits": total_hits,
        "best_hit": best_hit,
        "distribution": {str(k): distribution[k] for k in sorted(distribution)},
        "estimated_winnings": round(winnings, 2),
        "estimated_cost": round(total_tickets * price_amount, 2),
        "balance": round(winnings - total_tickets * price_amount, 2),
        "price_per_ticket": price_amount,
        "currency": currency or price_currency,
    }
