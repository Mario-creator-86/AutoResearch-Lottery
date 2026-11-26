"""Teste pentru ai_ticket_agent: ordonare bilete, poarta „doar în ziua
extragerii", estimarea premiilor și notificările Telegram (propunere + rezultate)."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "agents")):
    if p not in sys.path:
        sys.path.insert(0, p)

import agents.ai_ticket_agent as ata
import load_data
import telegram_bot
from agents.ai_ticket_agent import AITicketAgent
from prize_estimates import estimate_ticket_prize


@pytest.fixture(autouse=True)
def isolate_decision_ledger(tmp_path, monkeypatch):
    import decision_ledger
    import scrape_lotteries

    monkeypatch.setattr(decision_ledger, "DATA_DIR", tmp_path)
    monkeypatch.setattr(scrape_lotteries, "download_lottery", lambda _: None)


def make_agent(tickets_per_day: int = 20) -> AITicketAgent:
    agent = AITicketAgent.__new__(AITicketAgent)
    agent.tickets_per_day = tickets_per_day
    agent.lottery_order = ["loto_6_49_ro"]
    agent.log_warning = lambda *a, **k: None
    agent.log_error = lambda *a, **k: None
    agent.log_success = lambda *a, **k: None
    agent.update_heartbeat = lambda *a, **k: None
    return agent


# ─── ordonarea biletelor după probabilitate ──────────────────────────────

def test_sort_tickets_favorite_first():
    ranked = [7, 12, 19, 23, 41, 48, 3, 30]
    tickets = [
        {"main": [30, 3, 41, 23, 19, 12]},   # scor mai slab (include 30, 3)
        {"main": [7, 12, 19, 23, 41, 48]},   # top 6 → favorit
        {"main": [1, 2, 4, 5, 6, 8]},        # toate în afara clasamentului → cel mai slab
    ]
    ordered = AITicketAgent._sort_tickets_by_likelihood(tickets, ranked)
    assert ordered[0]["main"] == [7, 12, 19, 23, 41, 48]
    assert ordered[1]["main"] == [30, 3, 41, 23, 19, 12]
    assert ordered[2]["main"] == [1, 2, 4, 5, 6, 8]
    assert all("score" in t for t in ordered)
    assert ordered[0]["score"] < ordered[1]["score"] < ordered[2]["score"]


def test_sort_tickets_without_ranking_keeps_model_order():
    tickets = [{"main": [9, 8, 7]}, {"main": [1, 2, 3]}]
    assert AITicketAgent._sort_tickets_by_likelihood(tickets, []) == tickets


# ─── estimarea premiilor ─────────────────────────────────────────────────

def test_prize_estimates_categories():
    assert estimate_ticket_prize("loto_6_49_ro", 4) == {
        "category": "Categoria III", "amount": 1500, "currency": "RON"}
    assert estimate_ticket_prize("loto_6_49_ro", 2)["amount"] == 0
    assert estimate_ticket_prize("loto_6_49_ro", 0) is None
    # SuperEnalotto: 5 + Jolly trebuie să bată tier-ul de 5 puncte simple
    assert estimate_ticket_prize("superenalotto_it", 5, 1)["category"] == "5 punti + Jolly"
    assert estimate_ticket_prize("superenalotto_it", 5, 0)["category"] == "5 punti"
    assert estimate_ticket_prize("superenalotto_it", 2)["amount"] == 5
    # loterie fără tabel: raportăm doar loviturile
    unknown = estimate_ticket_prize("necunoscuta", 3)
    assert unknown["amount"] == 0 and "3" in unknown["category"]


# ─── poarta „doar în ziua extragerii" + retragerea propunerilor premature ─

def test_generate_only_on_draw_day_and_purges_early_proposals(monkeypatch):
    agent = make_agent()
    today = date.today()
    future = today.replace(day=min(today.day + 2, 28))
    planted = {"entries": [{
        "draw_date": future.isoformat(), "status": "pending", "tickets": [],
    }]}
    saved: dict = {}

    monkeypatch.setattr(ata, "load_ticket_store", lambda lid: planted)
    monkeypatch.setattr(ata, "save_ticket_store",
                        lambda lid, store: saved.update(deep=store))

    class Cfg:
        def next_draw_date(self):
            return future  # extragerea NU e azi

    called = {"analyzed": False}

    class FakeAI:
        def get_lottery_config(self, lid):
            return Cfg()

        def analyze_lottery(self, lid):
            called["analyzed"] = True
            return {}

    monkeypatch.setattr(load_data, "load_lottery_draws", lambda lid: [object()])
    agent.generate_tickets("loto_6_49_ro", FakeAI())

    assert not called["analyzed"]          # nu am consumat AI pentru o extragere viitoare
    assert saved["deep"]["entries"] == []  # propunerea prematură a fost retrasă


# ─── generare completă: ordonare + Telegram ──────────────────────────────

def test_generate_orders_tickets_and_notifies_telegram(monkeypatch):
    agent = make_agent()
    ranked = [5, 11, 23, 34, 42, 48, 7, 19, 27, 39]
    model_tickets = [
        {"main": [7, 19, 27, 39, 42, 48], "why": "explorație"},
        {"main": [5, 11, 23, 34, 42, 48], "why": "ancoră pe top strategie"},
    ]
    saved: dict = {}

    monkeypatch.setattr(ata, "load_ticket_store", lambda lid: {"entries": []})
    monkeypatch.setattr(ata, "save_ticket_store", lambda lid, store: saved.update(store=store))

    class Cfg:
        id = "loto_6_49_ro"
        name = "Loto 6/49 (Romania)"
        ticket_size = 6
        numbers_drawn = 6
        number_range = 49
        bonus_range = None
        min_bonus_numbers = 0
        draw_times_local = ("18:00",)

        def next_draw_date(self):
            return date.today()

    class FakeAI:
        def get_lottery_config(self, lid):
            return Cfg()

        def load_predictions(self, lid):
            return {"recommended_numbers": [5, 11, 23, 34, 42, 48, 7, 19]}

        def _get_best_strategy_summary(self, lid):
            return {"metric": 2.31}

        def model_candidates(self, _):
            return ["fake-model"]

        def chat(self, prompt, model, system=None):
            return json.dumps({"thoughts": "test", "tickets": model_tickets})

        def extract_json(self, text):
            return json.loads(text)

    import auto_loop

    monkeypatch.setattr(load_data, "load_lottery_draws", lambda lid: [object()])
    monkeypatch.setattr(auto_loop, "load_search_state_for_lottery",
                        lambda lid: (SimpleNamespace(best_metric=1.0), False))
    monkeypatch.setattr(auto_loop, "rank_numbers_from_draws", lambda *a, **k: ranked)

    sent: list[str] = []
    monkeypatch.setattr(telegram_bot, "notify", lambda text: sent.append(text) or True)

    # independent de ora la care rulează testul (slotul de 18:00 poate fi trecut)
    monkeypatch.setattr(agent, "_upcoming_slot", lambda config, now=None: (1, "18:00"))

    agent.generate_tickets("loto_6_49_ro", FakeAI())

    entry = saved["store"]["entries"][0]
    assert entry["status"] == "pending"
    assert entry["tickets"][0]["main"] == [5, 11, 23, 34, 42, 48]  # favoritul primul
    assert len(sent) == 1 and "Loto 6/49 (Romania)" in sent[0]
    assert "1." in sent[0] and "probabilitate" in sent[0]


# ─── verificarea post-extragere: raport Telegram mereu + premii estimate ──

def test_check_pending_notifies_results_with_prize(monkeypatch):
    agent = make_agent()
    draw_day = date(2026, 9, 3)
    draw = SimpleNamespace(draw_date=draw_day, numbers=[5, 11, 23, 34, 42, 48], bonuses=[9])
    entry = {
        "draw_date": draw_day.isoformat(),
        "status": "pending",
        "tickets": [
            {"main": [5, 11, 23, 34, 42, 48], "bonus": [9], "why": "x"},  # 6/6 + bonus
            {"main": [5, 11, 23, 1, 2, 3], "bonus": [], "why": "y"},      # 3 lovituri → 30 RON
            {"main": [1, 2, 4, 6, 7, 8], "bonus": [], "why": "z"},        # 0 lovituri
        ],
    }
    store = {"entries": [entry]}
    saved: dict = {}

    monkeypatch.setattr(ata, "load_ticket_store", lambda lid: store)
    monkeypatch.setattr(ata, "save_ticket_store", lambda lid, s: saved.update(store=s))
    monkeypatch.setattr(load_data, "load_lottery_draws", lambda lid: [draw])

    class FakeAI:
        def model_candidates(self, _):
            return ["fake-model"]

        def chat(self, *a, **k):
            return "lecție de învățat"

        def load_prompt_overrides(self):
            return {}

        def save_prompt_overrides(self, o):
            pass

    sent: list[str] = []
    monkeypatch.setattr(telegram_bot, "notify", lambda text: sent.append(text) or True)

    agent.check_pending_tickets("loto_6_49_ro", FakeAI())

    checked = saved["store"]["entries"][0]
    assert checked["status"] == "checked"
    assert checked["results"][0]["hit_count"] == 6
    assert checked["results"][0]["prize"]["amount"] > 0
    assert checked["estimated_prize"]["amount"] >= 30  # cel puțin biletul cu 3 lovituri
    # raportul se trimite întotdeauna, nu doar la rezultate notabile
    assert len(sent) == 1
    assert "rezultatele biletelor AI" in sent[0]
    assert "câștig estimat" in sent[0]
    assert "3." in sent[0]  # toate cele 3 bilete, enumerate


# ─── loterii cu mai multe extrageri pe zi (Kaskada/Multi Multi 14:00 + 21:40) ─

class MultiCfg:
    id = "kaskada_pl"
    name = "Kaskada 12/24 (Poland)"
    ticket_size = 12
    numbers_drawn = 12
    number_range = 24
    bonus_range = None
    min_bonus_numbers = 0
    draw_times_local = ("14:00", "21:40")

    def next_draw_date(self):
        return date.today()


def test_upcoming_slot_picks_correct_draw():
    slot_of = AITicketAgent._upcoming_slot
    day = date(2026, 9, 8)
    cfg = MultiCfg()
    assert slot_of(cfg, datetime(2026, 9, 8, 9, 0)) == (1, "14:00")   # dimineața → extragerea 1
    assert slot_of(cfg, datetime(2026, 9, 8, 15, 0)) == (2, "21:40")  # după 14:00 → extragerea 2
    assert slot_of(cfg, datetime(2026, 9, 8, 22, 0)) is None          # după 21:40 → nimic azi
    single = SimpleNamespace(draw_times_local=("18:00",))
    assert slot_of(single, datetime(2026, 9, 8, 12, 0)) == (1, "18:00")
    assert slot_of(single, datetime(2026, 9, 8, 19, 0)) is None
    assert slot_of(SimpleNamespace(draw_times_local=()), datetime(2026, 9, 8, 23, 0)) == (1, "")


def _stub_generate(monkeypatch, store_entries, slot):
    """Stuburi comune pentru generate_tickets pe Kaskada (fără AI real)."""
    agent = make_agent()
    tickets = [{"main": list(range(1, 13)), "why": "test"}]
    saved: dict = {}

    monkeypatch.setattr(ata, "load_ticket_store", lambda lid: {"entries": store_entries})
    monkeypatch.setattr(ata, "save_ticket_store", lambda lid, s: saved.update(store=s))
    monkeypatch.setattr(ata.AITicketAgent, "_upcoming_slot",
                        staticmethod(lambda config, now=None: slot))

    class FakeAI:
        def get_lottery_config(self, lid):
            return MultiCfg()

        def load_predictions(self, lid):
            return {"recommended_numbers": list(range(1, 13))}

        def _get_best_strategy_summary(self, lid):
            return {"metric": 1.0}

        def model_candidates(self, _):
            return ["fake-model"]

        def chat(self, prompt, model, system=None):
            return json.dumps({"thoughts": "t", "tickets": tickets})

        def extract_json(self, text):
            return json.loads(text)

    import auto_loop

    monkeypatch.setattr(load_data, "load_lottery_draws", lambda lid: [object()])
    monkeypatch.setattr(auto_loop, "load_search_state_for_lottery",
                        lambda lid: (SimpleNamespace(best_metric=1.0), False))
    monkeypatch.setattr(auto_loop, "rank_numbers_from_draws", lambda *a, **k: list(range(1, 25)))
    return agent, FakeAI(), saved


def test_generate_twice_per_day_separate_draws(monkeypatch):
    monkeypatch.setattr(telegram_bot, "notify", lambda text: True)
    # dimineața: se generează extragerea 1
    agent, fake_ai, saved = _stub_generate(monkeypatch, [], (1, "14:00"))
    agent.generate_tickets("kaskada_pl", fake_ai)
    entries = saved["store"]["entries"]
    assert len(entries) == 1
    assert entries[0]["draw_index"] == 1 and entries[0]["draw_time"] == "14:00"

    # aceeași extragere → fără duplicate
    agent, fake_ai, saved = _stub_generate(
        monkeypatch, [dict(entries[0], status="pending")], (1, "14:00"))
    agent.generate_tickets("kaskada_pl", fake_ai)
    assert "store" not in saved  # nimic nou salvat

    # după ora 14:00: se generează extragerea 2, PĂSTRÂND extragerea 1
    agent, fake_ai, saved = _stub_generate(
        monkeypatch, [dict(entries[0], status="pending")], (2, "21:40"))
    agent.generate_tickets("kaskada_pl", fake_ai)
    entries2 = saved["store"]["entries"]
    assert len(entries2) == 2
    assert entries2[1]["draw_index"] == 2 and entries2[1]["draw_time"] == "21:40"


def test_check_matches_correct_draw_of_the_day(monkeypatch):
    agent = make_agent()
    day = date(2026, 9, 8)
    # extragerea 1: conține 1..12 ; extragerea 2: altele
    d1 = SimpleNamespace(draw_date=day, numbers=tuple(range(1, 13)), bonuses=[])
    d2 = SimpleNamespace(draw_date=day, numbers=tuple(range(13, 25)), bonuses=[])
    entry = {
        "draw_date": day.isoformat(), "draw_index": 1,
        "draw_time": "14:00", "next_draw_times": ("14:00", "21:40"),
        "status": "pending",
        "tickets": [{"main": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "bonus": [], "why": "x"}],
    }
    saved: dict = {}
    monkeypatch.setattr(ata, "load_ticket_store", lambda lid: {"entries": [entry]})
    monkeypatch.setattr(ata, "save_ticket_store", lambda lid, s: saved.update(store=s))
    monkeypatch.setattr(load_data, "load_lottery_draws", lambda lid: [d1, d2])

    class FakeAI:
        def model_candidates(self, _):
            return ["fake-model"]

        def chat(self, *a, **k):
            raise RuntimeError("fără AI în test")

        def load_prompt_overrides(self):
            return {}

        def save_prompt_overrides(self, o):
            pass

    sent: list[str] = []
    monkeypatch.setattr(telegram_bot, "notify", lambda text: sent.append(text) or True)

    agent.check_pending_tickets("kaskada_pl", FakeAI())

    checked = saved["store"]["entries"][0]
    # dacă s-ar potrivi cu extragerea 2 (comportamentul vechi), hit_count ar fi 0
    assert checked["results"][0]["hit_count"] == 12
    assert "extragerea 1 din 2 (ora 14:00)" in sent[0]


def test_check_waits_when_slot2_draw_not_in_history_yet(monkeypatch):
    """Setul pentru extragerea 2 (21:40) NU trebuie verificat contra extragerii 1."""
    import scrape_lotteries
    monkeypatch.setattr(scrape_lotteries, "download_lottery", lambda lid: None)
    agent = make_agent()
    day = date(2026, 9, 8)
    d1 = SimpleNamespace(draw_date=day, numbers=tuple(range(1, 13)), bonuses=[])
    entry = {
        "draw_date": day.isoformat(), "draw_index": 2, "draw_time": "21:40",
        "next_draw_times": ("14:00", "21:40"),
        "status": "pending",
        "tickets": [{"main": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "bonus": [], "why": "x"}],
    }
    monkeypatch.setattr(ata, "load_ticket_store", lambda lid: {"entries": [entry]})
    monkeypatch.setattr(ata, "save_ticket_store", lambda lid, s: (_ for _ in ()).throw(
        AssertionError("nu trebuie salvat nimic")))
    monkeypatch.setattr(load_data, "load_lottery_draws", lambda lid: [d1])  # doar extragerea 1

    class FakeAI:
        pass

    sent: list[str] = []
    monkeypatch.setattr(telegram_bot, "notify", lambda text: sent.append(text) or True)

    agent.check_pending_tickets("kaskada_pl", FakeAI())

    assert entry["status"] == "pending"   # rămâne în joc, așteaptă extragerea 2
    assert sent == []


# ─── actualizarea istoricului după o extragere trecută ───────────────────

def test_maybe_refresh_history_triggers_when_draw_missing(monkeypatch):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls.combine(date.today(), cls.min.time()).replace(hour=15)

    monkeypatch.setattr(ata, "datetime", FixedDatetime)
    agent = make_agent()
    agent._history_refresh_at = {}
    today = date.today()
    pending = [{"draw_date": today.isoformat(), "draw_index": 1, "draw_time": "14:00"}]
    draws_by_date = {}  # extragerea de azi 14:00 lipsește din istoric

    calls = []
    import scrape_lotteries
    monkeypatch.setattr(scrape_lotteries, "SCRAPER_MAP", {"kaskada_pl": lambda: None})
    monkeypatch.setattr(scrape_lotteries, "download_lottery", lambda lid: calls.append(lid))
    agent.log_warning = lambda *a, **k: None
    agent.log_success = lambda *a, **k: None
    agent.update_heartbeat = lambda *a, **k: None

    assert agent._maybe_refresh_history("kaskada_pl", pending, draws_by_date) is True
    assert calls == ["kaskada_pl"]
    # throttle: a doua apelare imediată → fără download
    assert agent._maybe_refresh_history("kaskada_pl", pending, draws_by_date) is False
    assert calls == ["kaskada_pl"]


def test_check_pending_refreshes_even_when_history_is_empty(monkeypatch):
    agent = make_agent()
    entry = {"draw_date": "2026-09-18", "status": "pending", "draw_index": 2}
    monkeypatch.setattr(ata, "load_ticket_store", lambda _: {"entries": [entry]})
    monkeypatch.setattr(load_data, "load_lottery_draws", lambda _: [])
    refreshed = []
    monkeypatch.setattr(agent, "_maybe_refresh_history",
                        lambda lid, pending, draws: refreshed.append((lid, pending, draws)) or False)
    agent.check_pending_tickets("kaskada_pl", object())
    assert refreshed == [("kaskada_pl", [entry], {})]


def test_maybe_refresh_history_skips_future_and_upcoming_draws(monkeypatch):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls.combine(date.today(), cls.min.time()).replace(hour=12)

    monkeypatch.setattr(ata, "datetime", FixedDatetime)
    agent = make_agent()
    agent._history_refresh_at = {}
    now = datetime.now()
    # extragerea de azi nu a avut loc încă (ora în viitor)
    pending = [{"draw_date": now.date().isoformat(), "draw_time": "23:59"}]
    assert agent._maybe_refresh_history("kaskada_pl", pending, {}) is False
    # extragere în viitor (altă zi)
    future = (now.date() + timedelta(days=1)).isoformat()
    assert agent._maybe_refresh_history("kaskada_pl", [{"draw_date": future}], {}) is False
    # deja în istoric
    today_iso = now.date().isoformat()
    assert agent._maybe_refresh_history(
        "kaskada_pl", [{"draw_date": today_iso, "draw_time": "00:00"}],
        {today_iso: [SimpleNamespace(draw_time="00:00")]}) is False
