from datetime import date
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

import ai_tickets_store
import load_data
import server


@pytest.mark.parametrize("hits", range(13))
def test_kaskada_outcome_requires_eight_direct_hits(hits):
    outcome = server._evaluate_ticket_outcome("kaskada_pl", hits, 0, 12)
    assert outcome["won"] is (hits >= 8)
    assert "complement" not in outcome["reason"]


def test_history_counts_and_sorts_only_direct_winners(monkeypatch):
    draw = SimpleNamespace(draw_date=date(2026, 9, 23), draw_time="21:40",
                           numbers=tuple(range(1, 13)), bonuses=())
    monkeypatch.setattr(load_data, "load_specific_lottery_draws", lambda *a, **kw: [draw])
    entries = [{"id": str(hits), "main": list(range(1, hits + 1)) + list(range(13, 25 - hits)),
                "target_draw_date": "2026-09-23", "target_draw_time": "21:40"}
               for hits in (4, 7, 8, 10, 8)]
    server._history_payload_cached.cache_clear()
    try:
        payload = server._history_payload_cached("kaskada_pl", "regression", json.dumps(entries))
        group = payload["groups"][0]
        assert group["winning_ticket_count"] == 3
        assert [t["main_hits"] for t in group["tickets"]] == [10, 8, 8, 7, 4]
        assert [t["won"] for t in group["tickets"]] == [True, True, True, False, False]
    finally:
        server._history_payload_cached.cache_clear()


def test_ai_history_recalculates_old_complement_prizes(monkeypatch):
    store = {"entries": [{"draw_date": "2026-09-23", "status": "checked",
                          "tickets": [{}, {}, {}],
                          "results": [{"hit_count": h, "bonus_hits": []} for h in (4, 7, 8)],
                          "estimated_prize": {"amount": 4, "currency": "PLN"}}]}
    monkeypatch.setattr(ai_tickets_store, "load_ticket_store", lambda _: store)
    response = TestClient(server.app).get("/api/kaskada_pl/ai-ticket-history")
    assert response.status_code == 200
    payload = response.json()
    assert payload["entries"][0]["estimated_prize"]["amount"] == 2
    assert payload["summary"]["estimated_winnings"] == 2
