from datetime import date

import pytest

import load_data
import lottery_config
from lottery_config import LotteryConfig
from types import SimpleNamespace


@pytest.fixture
def csv_config(tmp_path, monkeypatch):
    config = LotteryConfig(id="integrity", name="Integrity", numbers_drawn=3,
                           number_range=10, ticket_size=3, bonus_count=1, bonus_range=5)
    monkeypatch.setattr(lottery_config, "DATA_DIR", tmp_path)
    monkeypatch.setitem(lottery_config.LOTTERIES, config.id, config)
    load_data._load_lottery_draws_cached.cache_clear()
    return config


@pytest.mark.parametrize("numbers,bonuses", [
    ([1, 1, 2], [1]), ([1, 2], [1]), ([1, 2, 3, 4], [1]),
    ([1, 2, 3], []), ([1, 2, 3], [1, 1]), ([1, 2, 11], [1]),
])
def test_invalid_draw_rejected(csv_config, numbers, bonuses):
    assert not load_data._draw_matches_current_rules(csv_config, numbers, bonuses)


def test_distinct_draws_with_same_numbers_are_preserved(csv_config):
    csv_config.draws_file.write_text(
        "draw_date,draw_time,draw_id,n1,n2,n3,b1\n"
        "2026-09-18,14:00,100,1,2,3,1\n"
        "2026-09-18,21:40,101,1,2,3,1\n"
        "2026-09-18,14:00,100,3,2,1,1\n", encoding="utf-8")
    full = load_data.load_lottery_draws(csv_config.id)
    filtered = load_data.load_specific_lottery_draws(csv_config.id, {"2026-09-18"})
    assert [d.draw_identifier for d in full] == ["100", "101"]
    assert filtered == full


def test_filtered_loader_uses_validated_history(csv_config):
    csv_config.draws_file.write_text(
        "draw_date,draw_time,draw_id,n1,n2,n3,b1\n"
        "2026-09-17,,100,1,2,3,1\n"
        "2026-09-18,,101,1,1,3,1\n"
        "2026-09-18,,102,1,2,3,\n"
        "2026-09-18,,103,2,3,4,2\n", encoding="utf-8")
    full = load_data.load_lottery_draws(csv_config.id)
    assert [d.draw_identifier for d in full] == ["100", "103"]
    assert load_data.load_specific_lottery_draws(csv_config.id, {"2026-09-18"}) == full[1:]
    assert load_data.load_specific_lottery_draws(csv_config.id, set()) == []


def test_live_prediction_matches_backtest_next_prediction(monkeypatch):
    import auto_loop
    from generic_backtest import LotteryDraw, backtest_strategy, hot_cold_rank

    config = lottery_config.get_lottery_config("lotto_pl")
    draws = [LotteryDraw(date(2026, 9, i + 1), None, str(i),
                         (1, 2, 3, 4, 5, 6) if i % 2 else (7, 8, 9, 10, 11, 12))
             for i in range(15)]
    observed = []

    def ranker(counts, recent, last_seen, index, window):
        observed.append(index)
        return hot_cold_rank(config, counts, recent, last_seen, index, window)

    monkeypatch.setattr(auto_loop, "build_rank_fn_from_state", lambda *a, **k: ranker)
    monkeypatch.setattr(auto_loop.ai_strategy_lab, "promoted_plugins", lambda _: [])
    result = backtest_strategy(config, "audit", ranker, draws, recent_window=5)
    live = auto_loop.rank_numbers_from_draws(draws, auto_loop.SearchState(rw=5, extra_strategies=[]), config.id)
    assert observed[-2:] == [len(draws), len(draws)]
    assert tuple(live[:config.ticket_size]) == result.proposed_numbers


@pytest.mark.parametrize("target", [
    {"target_draw_id": "101"},
    {"target_draw_time": "21:40"},
    {"target_draw_time": "21:40", "target_draw_index": 1},
    {"target_draw_index": 2},
])
def test_history_does_not_substitute_another_draw(tmp_path, monkeypatch, target):
    import server

    config = SimpleNamespace(id="kaskada_pl", history_file=tmp_path / "history.json")
    entries = [{"id": "audit", "saved_at": "2026-09-18T10:00:00",
                "saved_date": "2026-09-18", "target_draw_date": "2026-09-18",
                "main": [1, 2, 3], "bonus": [], **target}]
    draws = [SimpleNamespace(draw_date=date(2026, 9, 18), draw_time="14:00",
                             draw_identifier="100", numbers=(1, 2, 3), bonuses=())]
    monkeypatch.setattr(server, "_load_ticket_history", lambda _: entries)
    monkeypatch.setattr(server, "_ai_history_entries", lambda _: [])
    monkeypatch.setattr(load_data, "load_specific_lottery_draws", lambda *a, **k: draws)
    server._history_payload_cached.cache_clear()
    assert server._history_payload(config)["groups"][0]["draw_available"] is False
