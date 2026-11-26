from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from romania_lotto import RomaniaDrawResult
import scrape_lotteries


def _romanian_result(lottery_id: str, draw_date: str, numbers: tuple[int, ...], bonuses: tuple[int, ...] = ()) -> RomaniaDrawResult:
    return RomaniaDrawResult(
        lottery_id=lottery_id,
        draw_date=date.fromisoformat(draw_date),
        numbers=numbers,
        bonuses=bonuses,
        tier1_winners=0,
        tier1_amount=0.0,
        is_rollover=False,
        total_pool=0.0,
    )


def test_download_lottery_romania_uses_incremental_fetch_and_appends(tmp_path, monkeypatch):
    monkeypatch.setattr(scrape_lotteries, "DATA_DIR", tmp_path)
    csv_path = tmp_path / "joker_ro.csv"
    csv_path.write_text(
        "draw_date,draw_time,draw_id,number_01,number_02,number_03,number_04,number_05,bonus_01\n"
        "2026-03-20,,20260320,1,2,3,4,5,6\n",
        encoding="utf-8",
    )

    calls: list[date | None] = []

    def fake_fetch_results_since(lottery_id: str, start_date: date | None = None, end_date: date | None = None, *, delay_seconds: float = 0.0):
        calls.append(start_date)
        assert lottery_id == "joker_ro"
        return [_romanian_result("joker_ro", "2026-03-23", (7, 8, 9, 10, 11), (12,))]

    monkeypatch.setattr(scrape_lotteries, "fetch_results_since", fake_fetch_results_since)

    scrape_lotteries.download_lottery("joker_ro")

    assert calls == [date(2026, 3, 20)]
    assert csv_path.read_text(encoding="utf-8").splitlines() == [
        "draw_date,draw_time,draw_id,number_01,number_02,number_03,number_04,number_05,bonus_01",
        "2026-03-20,,20260320,1,2,3,4,5,6",
        "2026-03-23,,20260323,7,8,9,10,11,12",
    ]


def test_download_lottery_romania_bootstraps_full_history_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(scrape_lotteries, "DATA_DIR", tmp_path)

    calls: list[date | None] = []

    def fake_fetch_results_since(lottery_id: str, start_date: date | None = None, end_date: date | None = None, *, delay_seconds: float = 0.0):
        calls.append(start_date)
        assert lottery_id == "loto_6_49_ro"
        return [
            _romanian_result("loto_6_49_ro", "2026-03-20", (1, 2, 3, 4, 5, 6)),
            _romanian_result("loto_6_49_ro", "2026-03-23", (7, 8, 9, 10, 11, 12)),
        ]

    monkeypatch.setattr(scrape_lotteries, "fetch_results_since", fake_fetch_results_since)

    scrape_lotteries.download_lottery("loto_6_49_ro")

    assert calls == [None]
    assert (tmp_path / "loto_6_49_ro.csv").read_text(encoding="utf-8").splitlines() == [
        "draw_date,draw_time,draw_id,number_01,number_02,number_03,number_04,number_05,number_06",
        "2026-03-20,,20260320,1,2,3,4,5,6",
        "2026-03-23,,20260323,7,8,9,10,11,12",
    ]


def test_scrape_national_lottery_uk_parses_official_download_csv(monkeypatch):
    csv_text = (
        "DrawDate,Ball 1,Ball 2,Ball 3,Ball 4,Ball 5,Lucky Star 1,Lucky Star 2,UK Millionaire Maker,European Millionaire Maker,Ball Set,Machine,DrawNumber\n"
        "27-Mar-2026,4,10,43,44,48,2,4,JQCB13299,,19,15,1932\n"
        "24-Mar-2026,12,16,17,18,27,1,3,JPBM36493,,19,15,1931\n"
    )
    calls: list[str] = []

    def fake_get(url: str, headers=None, timeout: int = 0):
        calls.append(url)
        return SimpleNamespace(text=csv_text, raise_for_status=lambda: None)

    monkeypatch.setattr(scrape_lotteries.requests, "get", fake_get)

    draws = scrape_lotteries.scrape_national_lottery_uk("euromillions")

    assert calls == [scrape_lotteries.NATIONAL_LOTTERY_UK_DOWNLOADS["euromillions"]]
    assert draws == [
        {
            "draw_date": "2026-03-27",
            "draw_time": "",
            "draw_id": "1932",
            "numbers": [4, 10, 43, 44, 48],
            "bonuses": [2, 4],
        },
        {
            "draw_date": "2026-03-24",
            "draw_time": "",
            "draw_id": "1931",
            "numbers": [12, 16, 17, 18, 27],
            "bonuses": [1, 3],
        },
    ]


def test_download_lottery_uk_appends_only_new_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(scrape_lotteries, "DATA_DIR", tmp_path)
    csv_path = tmp_path / "uk_lotto.csv"
    csv_path.write_text(
        "draw_date,draw_time,draw_id,number_01,number_02,number_03,number_04,number_05,number_06,bonus_01\n"
        "2026-03-25,,3157,19,22,31,32,34,40,43\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        scrape_lotteries,
        "scrape_national_lottery_uk",
        lambda game: [
            {
                "draw_date": "2026-03-25",
                "draw_time": "",
                "draw_id": "3157",
                "numbers": [19, 22, 31, 32, 34, 40],
                "bonuses": [43],
            },
            {
                "draw_date": "2026-03-28",
                "draw_time": "",
                "draw_id": "3158",
                "numbers": [9, 24, 35, 40, 47, 52],
                "bonuses": [18],
            },
        ],
    )

    scrape_lotteries.download_lottery("uk_lotto")

    assert csv_path.read_text(encoding="utf-8").splitlines() == [
        "draw_date,draw_time,draw_id,number_01,number_02,number_03,number_04,number_05,number_06,bonus_01",
        "2026-03-25,,3157,19,22,31,32,34,40,43",
        "2026-03-28,,3158,9,24,35,40,47,52,18",
    ]
