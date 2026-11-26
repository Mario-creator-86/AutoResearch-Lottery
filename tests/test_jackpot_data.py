from __future__ import annotations

from datetime import date, timedelta

from generic_backtest import LotteryDraw
from jackpot_data import load_jackpot_data, load_jackpot_data_with_source
from jackpot_proxy import compute_proxy_jackpot
from lottery_config import LotteryConfig, get_lottery_config


def _make_draws(count: int, config: LotteryConfig) -> list[LotteryDraw]:
    import random

    rng = random.Random(123)
    start = date(2020, 1, 1)
    draws = []
    for index in range(count):
        draws.append(
            LotteryDraw(
                draw_date=start + timedelta(days=index),
                draw_time=None,
                draw_identifier=str(index),
                numbers=tuple(sorted(rng.sample(range(1, config.number_range + 1), config.numbers_drawn))),
                bonuses=(),
            )
        )
    return draws


def test_load_jackpot_data_returns_proxy_when_csv_missing():
    config = get_lottery_config("powerball_us")
    draws = _make_draws(200, config)

    result = load_jackpot_data("powerball_us", draws)

    assert len(result) == len(draws)
    assert all(0.0 <= info.phase <= 1.0 for info in result.values())


def test_load_jackpot_data_reads_real_csv(tmp_path, monkeypatch):
    import lottery_config

    state_dir = tmp_path / "lotto_pl"
    state_dir.mkdir(parents=True)
    (state_dir / "jackpot_history.csv").write_text(
        "draw_date,tier1_winners,tier1_amount,is_rollover,total_pool\n"
        "2020-01-01,0,0,true,100000\n"
        "2020-01-02,1,5000000,false,6000000\n"
        "2020-01-03,0,0,true,200000\n",
        encoding="utf-8",
    )

    original = get_lottery_config("lotto_pl")
    patched = LotteryConfig(
        id=original.id,
        name=original.name,
        numbers_drawn=original.numbers_drawn,
        number_range=original.number_range,
        ticket_size=original.ticket_size,
        min_pick_count=original.min_pick_count,
        max_pick_count=original.max_pick_count,
        bonus_count=original.bonus_count,
        bonus_range=original.bonus_range,
        min_bonus_pick_count=original.min_bonus_pick_count,
        max_bonus_pick_count=original.max_bonus_pick_count,
    )
    monkeypatch.setattr(lottery_config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(lottery_config, "LOTTERIES", {**lottery_config.LOTTERIES, "lotto_pl": patched})

    draws = _make_draws(5, patched)
    result = load_jackpot_data("lotto_pl", draws)

    assert result[date(2020, 1, 2)].draws_since_win == 0
    assert result[date(2020, 1, 3)].draws_since_win == 1
    assert result[date(2020, 1, 2)].jackpot_amount == 5000000.0

    sourced_result, source = load_jackpot_data_with_source("lotto_pl", draws)
    assert source == "real"
    assert sourced_result == result


def test_load_jackpot_data_falls_back_to_proxy_for_header_only_csv(tmp_path, monkeypatch):
    import lottery_config

    state_dir = tmp_path / "lotto_pl"
    state_dir.mkdir(parents=True)
    (state_dir / "jackpot_history.csv").write_text(
        "draw_date,tier1_winners,tier1_amount,is_rollover,total_pool\n",
        encoding="utf-8",
    )

    original = get_lottery_config("lotto_pl")
    patched = LotteryConfig(
        id=original.id,
        name=original.name,
        numbers_drawn=original.numbers_drawn,
        number_range=original.number_range,
        ticket_size=original.ticket_size,
        min_pick_count=original.min_pick_count,
        max_pick_count=original.max_pick_count,
        bonus_count=original.bonus_count,
        bonus_range=original.bonus_range,
        min_bonus_pick_count=original.min_bonus_pick_count,
        max_bonus_pick_count=original.max_bonus_pick_count,
    )
    monkeypatch.setattr(lottery_config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(lottery_config, "LOTTERIES", {**lottery_config.LOTTERIES, "lotto_pl": patched})

    draws = _make_draws(200, patched)
    result = load_jackpot_data("lotto_pl", draws)
    expected = compute_proxy_jackpot(draws, patched)
    sourced_result, source = load_jackpot_data_with_source("lotto_pl", draws)

    assert result == expected
    assert sourced_result == expected
    assert source == "proxy"


def test_load_jackpot_data_backfills_missing_dates_with_proxy(tmp_path, monkeypatch):
    import lottery_config

    state_dir = tmp_path / "lotto_pl"
    state_dir.mkdir(parents=True)
    (state_dir / "jackpot_history.csv").write_text(
        "draw_date,tier1_winners,tier1_amount,is_rollover,total_pool\n"
        "2020-01-01,0,0,true,100000\n"
        "2020-01-03,1,5000000,false,6000000\n",
        encoding="utf-8",
    )

    original = get_lottery_config("lotto_pl")
    patched = LotteryConfig(
        id=original.id,
        name=original.name,
        numbers_drawn=original.numbers_drawn,
        number_range=original.number_range,
        ticket_size=original.ticket_size,
        min_pick_count=original.min_pick_count,
        max_pick_count=original.max_pick_count,
        bonus_count=original.bonus_count,
        bonus_range=original.bonus_range,
        min_bonus_pick_count=original.min_bonus_pick_count,
        max_bonus_pick_count=original.max_bonus_pick_count,
    )
    monkeypatch.setattr(lottery_config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(lottery_config, "LOTTERIES", {**lottery_config.LOTTERIES, "lotto_pl": patched})

    draws = _make_draws(5, patched)
    proxy = compute_proxy_jackpot(draws, patched)
    result, source = load_jackpot_data_with_source("lotto_pl", draws)

    assert source == "real"
    assert result[date(2020, 1, 1)].jackpot_amount == 0.0
    assert result[date(2020, 1, 2)] == proxy[date(2020, 1, 2)]
    assert result[date(2020, 1, 3)].jackpot_amount == 5000000.0
