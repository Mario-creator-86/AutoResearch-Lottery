"""Teste pentru istoricul biletelor AI: sumarizare progres + endpoint History."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "agents")):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi.testclient import TestClient

import ai_tickets_store
from ai_tickets_store import summarize_tickets
import server


def test_summarize_tickets_progress_and_money():
    store = {"entries": [
        {
            "draw_date": "2026-09-03", "status": "checked",
            # intrare LEGACY: fără „prize" per rezultat și fără estimated_prize
            "results": [
                {"hit_count": 3, "bonus_hits": []},
                {"hit_count": 1, "bonus_hits": []},
            ],
            "tickets": [{}, {}],
        },
        {
            "draw_date": "2026-09-05", "status": "checked",
            "results": [
                # premiu stocat dintr-un tabel vechi/umflat → trebuie ignorat și recalculat
                {"hit_count": 0, "bonus_hits": [], "prize":
                    {"category": "vechi", "amount": 999, "currency": "RON"}},
                {"hit_count": 4, "bonus_hits": [], "prize":
                    {"category": "Categoria III", "amount": 1500, "currency": "RON"}},
            ],
            "tickets": [{}, {}],
        },
        {
            "draw_date": "2026-09-10", "status": "pending",
            "results": [],
            "tickets": [{} for _ in range(20)],
        },
    ]}

    s = summarize_tickets("loto_6_49_ro", store)

    assert s["draws_played"] == 2 and s["draws_in_play"] == 1
    assert s["tickets_played"] == 4 and s["tickets_in_play"] == 20
    assert s["total_hits"] == 8 and s["best_hit"] == 4
    # distribuție pe lovituri, din toate biletele jucate
    assert s["distribution"] == {"0": 1, "1": 1, "3": 1, "4": 1}
    # câștig recalculat din tabelul curent: 3 lovituri (cat. IV fix 50 RON) + 4 lovituri (1500);
    # premiul stocat de 999 RON pentru 0 lovituri este ignorat
    assert s["estimated_winnings"] == 1550.0
    # cost: 24 bilete × 8 RON (prețul oficial al variantei simple)
    assert s["price_per_ticket"] == 8.0
    assert s["estimated_cost"] == 192.0
    assert s["balance"] == 1550.0 - 192.0
    assert s["currency"] == "RON"


def test_summarize_empty_store():
    s = summarize_tickets("kaskada_pl", {"entries": []})
    assert s["draws_played"] == 0 and s["total_hits"] == 0
    assert s["distribution"] == {}
    assert s["estimated_winnings"] == 0.0 and s["estimated_cost"] == 0.0


def test_ai_ticket_history_endpoint():
    client = TestClient(server.app)
    resp = client.get("/api/loto_6_49_ro/ai-ticket-history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["lottery"] == "loto_6_49_ro"
    assert isinstance(data["entries"], list)
    assert "summary" in data
    for e in data["entries"]:
        assert {"draw_date", "status", "n_tickets", "total_hits"} <= set(e)
        if e["status"] == "checked":
            assert e["total_hits"] is not None
    # loterie inexistentă → 404
    assert client.get("/api/nu_exista/ai-ticket-history").status_code == 404


def test_ai_ticket_history_endpoint_uses_store(tmp_path, monkeypatch):
    store = {"entries": [{
        "draw_date": "2026-09-03", "status": "checked", "draw_index": 1,
        "draw_time": "14:00", "next_draw_times": ("14:00", "21:40"),
        "draw_numbers": [1, 2, 3], "draw_bonus": [],
        "tickets": [{}],
        "results": [{"hit_count": 2, "bonus_hits": []}],
    }]}
    monkeypatch.setattr(ai_tickets_store, "load_ticket_store", lambda lid: dict(store))
    monkeypatch.setattr(server, "load_ticket_store", lambda lid: dict(store), raising=False)

    client = TestClient(server.app)
    data = client.get("/api/kaskada_pl/ai-ticket-history").json()
    entry = data["entries"][0]
    assert entry["total_hits"] == 2 and entry["best_hits"] == 2
    assert entry["n_tickets"] == 1 and entry["status"] == "checked"
    s = data["summary"]
    assert s["draws_played"] == 1 and s["tickets_played"] == 1
