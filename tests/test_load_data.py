from __future__ import annotations

from lottery_config import LotteryConfig
from lottery_config import get_lottery_config
from load_data import load_lottery_draws


def test_load_lottery_draws_filters_history_to_current_rule_range():
    mini_draws = load_lottery_draws("mini_lotto_pl")
    powerball_draws = load_lottery_draws("powerball_us")
    mega_draws = load_lottery_draws("mega_millions_us")
    loto_649_draws = load_lottery_draws("loto_6_49_ro")
    loto_540_draws = load_lottery_draws("loto_5_40_ro")
    joker_draws = load_lottery_draws("joker_ro")

    mini_config = get_lottery_config("mini_lotto_pl")
    powerball_config = get_lottery_config("powerball_us")
    mega_config = get_lottery_config("mega_millions_us")
    loto_649_config = get_lottery_config("loto_6_49_ro")
    loto_540_config = get_lottery_config("loto_5_40_ro")
    joker_config = get_lottery_config("joker_ro")

    assert mini_draws
    assert powerball_draws
    assert mega_draws
    assert loto_649_draws
    assert loto_540_draws
    assert joker_draws
    assert all(max(draw.numbers) <= mini_config.number_range for draw in mini_draws)
    assert all(max(draw.bonuses or (0,)) <= powerball_config.bonus_range for draw in powerball_draws)
    assert all(max(draw.bonuses or (0,)) <= mega_config.bonus_range for draw in mega_draws)
    assert all(len(draw.numbers) == loto_649_config.numbers_drawn for draw in loto_649_draws)
    assert all(len(draw.numbers) == loto_540_config.numbers_drawn for draw in loto_540_draws)
    assert all(len(draw.bonuses) == joker_config.bonus_count for draw in joker_draws)


def test_load_lottery_draws_uses_positional_columns_when_header_is_stale(tmp_path, monkeypatch):
    import lottery_config

    (tmp_path / "loto_5_40_ro").mkdir(parents=True)
    (tmp_path / "loto_5_40_ro.csv").write_text(
        "draw_date,draw_time,draw_id,number_01,number_02,number_03,number_04,number_05\n"
        "2026-03-26,,20260326,2,6,13,19,22,40\n",
        encoding="utf-8",
    )

    original = get_lottery_config("loto_5_40_ro")
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
    monkeypatch.setattr(lottery_config, "LOTTERIES", {**lottery_config.LOTTERIES, "loto_5_40_ro": patched})

    draws = load_lottery_draws("loto_5_40_ro")

    assert len(draws) == 1
    assert draws[0].numbers == (2, 6, 13, 19, 22, 40)
