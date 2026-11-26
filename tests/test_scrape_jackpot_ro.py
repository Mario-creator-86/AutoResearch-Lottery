from __future__ import annotations

from datetime import date

from romania_lotto import RomaniaDrawResult
import scrape_jackpot_ro


def _result(draw_date: str) -> RomaniaDrawResult:
    return RomaniaDrawResult(
        lottery_id="loto_6_49_ro",
        draw_date=date.fromisoformat(draw_date),
        numbers=(1, 2, 3, 4, 5, 6),
        bonuses=(),
        tier1_winners=0,
        tier1_amount=123.45,
        is_rollover=True,
        total_pool=456.78,
    )


def test_scrape_lottery_appends_only_new_rows(tmp_path, monkeypatch):
    csv_path = tmp_path / "jackpot_history.csv"
    csv_path.write_text(
        "draw_date,tier1_winners,tier1_amount,is_rollover,total_pool\n"
        "2026-03-19,0,100.0,true,1000.0\n"
        "2026-03-22,0,200.0,true,2000.0\n",
        encoding="utf-8",
    )

    config = type("Config", (), {"jackpot_csv": csv_path})()

    calls: list[date | None] = []

    def fake_fetch_results_since(lottery_id: str, start_date: date | None = None, *, delay_seconds: float = 0.0):
        calls.append(start_date)
        assert lottery_id == "loto_6_49_ro"
        return [
            _result("2026-03-22"),
            _result("2026-03-26"),
            _result("2026-03-26"),
        ]

    monkeypatch.setattr(scrape_jackpot_ro, "get_lottery_config", lambda lottery_id: config)
    monkeypatch.setattr(scrape_jackpot_ro, "ensure_lottery_storage", lambda _config: None)
    monkeypatch.setattr(scrape_jackpot_ro, "fetch_results_since", fake_fetch_results_since)

    written = scrape_jackpot_ro.scrape_lottery("loto_6_49_ro")

    assert written == 1
    assert calls == [date(2026, 3, 22)]
    assert csv_path.read_text(encoding="utf-8").splitlines() == [
        "draw_date,tier1_winners,tier1_amount,is_rollover,total_pool",
        "2026-03-19,0,100.0,true,1000.0",
        "2026-03-22,0,200.0,true,2000.0",
        "2026-03-26,0,123.45,true,456.78",
    ]


def test_scrape_lottery_is_idempotent_when_nothing_new_exists(tmp_path, monkeypatch):
    csv_path = tmp_path / "jackpot_history.csv"
    csv_path.write_text(
        "draw_date,tier1_winners,tier1_amount,is_rollover,total_pool\n"
        "2026-03-19,0,100.0,true,1000.0\n",
        encoding="utf-8",
    )

    config = type("Config", (), {"jackpot_csv": csv_path})()

    def fake_fetch_results_since(lottery_id: str, start_date: date | None = None, *, delay_seconds: float = 0.0):
        return [_result("2026-03-19")]

    monkeypatch.setattr(scrape_jackpot_ro, "get_lottery_config", lambda lottery_id: config)
    monkeypatch.setattr(scrape_jackpot_ro, "ensure_lottery_storage", lambda _config: None)
    monkeypatch.setattr(scrape_jackpot_ro, "fetch_results_since", fake_fetch_results_since)

    written = scrape_jackpot_ro.scrape_lottery("loto_6_49_ro")

    assert written == 0
    assert csv_path.read_text(encoding="utf-8").splitlines() == [
        "draw_date,tier1_winners,tier1_amount,is_rollover,total_pool",
        "2026-03-19,0,100.0,true,1000.0",
    ]
