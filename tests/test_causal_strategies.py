from collections import deque
from datetime import date, timedelta
import random

import pytest

import auto_loop
import generic_backtest
import jackpot_data
import lottery_config
from generic_backtest import LotteryDraw
from jackpot_proxy import JackpotInfo, compute_proxy_jackpot


def make_draws(count=150):
    rng = random.Random(81)
    return [LotteryDraw(date(2020, 1, 1) + timedelta(days=i), None, str(i),
                        tuple(rng.sample(range(1, 50), 6))) for i in range(count)]


@pytest.mark.parametrize("prefix", [20, 70, 100])
def test_future_draws_do_not_change_past_proxy_phases(prefix):
    config = lottery_config.get_lottery_config("lotto_pl")
    draws = make_draws()
    past = compute_proxy_jackpot(draws[:prefix], config)
    full = compute_proxy_jackpot(draws, config)
    assert past == {day: full[day] for day in past}


def test_future_winners_do_not_change_past_real_phases(tmp_path, monkeypatch):
    monkeypatch.setattr(lottery_config, "DATA_DIR", tmp_path)
    config = lottery_config.get_lottery_config("lotto_pl")
    config.state_dir.mkdir()
    draws = make_draws(20)
    config.jackpot_csv.write_text(
        "draw_date,tier1_winners\n" + "".join(
            f"{d.draw_date},{int(i in (2, 8, 9, 10, 11, 12))}\n"
            for i, d in enumerate(draws)), encoding="utf-8")
    past = jackpot_data.load_jackpot_data(config.id, draws[:8])
    full = jackpot_data.load_jackpot_data(config.id, draws)
    assert past == {day: full[day] for day in past}


def test_missing_real_jackpot_dates_still_advance_cycle(tmp_path, monkeypatch):
    monkeypatch.setattr(lottery_config, "DATA_DIR", tmp_path)
    config = lottery_config.get_lottery_config("lotto_pl")
    config.state_dir.mkdir()
    config.jackpot_csv.write_text(
        "draw_date,tier1_winners\n2020-01-01,1\n2020-01-04,0\n", encoding="utf-8")
    result = jackpot_data.load_jackpot_data(config.id, make_draws(4))
    assert result[date(2020, 1, 4)].draws_since_win == 3


def test_intraday_rank_uses_only_completed_day_metadata(monkeypatch):
    config = lottery_config.get_lottery_config("lotto_pl")
    days = [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 2)]
    phases = {days[0]: JackpotInfo(.2, 1, True), days[1]: JackpotInfo(.9, 2, True)}
    observed = []
    monkeypatch.setattr(generic_backtest, "_phase_weights", lambda phase: observed.append(phase) or (1, 0, 0))
    ranker = generic_backtest.make_jackpot_cycle_rank(phases, days, config)
    ranker({1: 1}, deque(), {}, 2, 25)
    ranker({1: 1}, deque(), {}, 3, 25)
    assert observed == [.2, .9]


def test_live_jackpot_context_is_local_and_refreshes_with_history(monkeypatch):
    draws = make_draws(4)
    foreign = {date(1999, 1, 1): JackpotInfo(.99, 1, True)}
    monkeypatch.setattr(auto_loop, "_JACKPOT_DATA", foreign)
    monkeypatch.setattr(auto_loop, "_DRAW_DATES", list(foreign))
    monkeypatch.setattr(auto_loop.ai_strategy_lab, "promoted_plugins", lambda _: [])
    loaded = []
    observed = []

    def load(lid, history):
        loaded.append((lid, len(history)))
        return {d.draw_date: JackpotInfo(.1, 0, False) for d in history}

    def cycle(data, dates, config):
        observed.append((config.id, list(dates), data))
        return lambda *args: list(range(1, config.number_range + 1))

    monkeypatch.setattr(auto_loop, "load_jackpot_data", load)
    monkeypatch.setattr(auto_loop, "make_jackpot_cycle_rank", cycle)
    state = auto_loop.SearchState(extra_strategies=["jc"], jc=1)
    for lid, history in [("lotto_pl", draws[:3]), ("loto_6_49_ro", draws[:3]), ("lotto_pl", draws)]:
        auto_loop.rank_numbers_from_draws(history, state, lid)
    assert loaded == [("lotto_pl", 3), ("loto_6_49_ro", 3), ("lotto_pl", 4)]
    assert observed[-1][1] == [d.draw_date for d in draws]
    assert all(date(1999, 1, 1) not in row[2] for row in observed)
    assert auto_loop._JACKPOT_DATA is foreign


def test_live_prediction_does_not_mutate_shared_weights(monkeypatch):
    weights = {"frequency": .2, "recent": .7, "delay": .1}
    monkeypatch.setattr(auto_loop, "BACKTEST_WEIGHTS", weights)
    monkeypatch.setattr(generic_backtest, "BACKTEST_WEIGHTS", weights)
    monkeypatch.setattr(auto_loop.ai_strategy_lab, "promoted_plugins", lambda _: [])
    before = dict(weights)
    auto_loop.rank_numbers_from_draws(make_draws(12), auto_loop.SearchState(extra_strategies=[]), "lotto_pl")
    assert weights == before


def test_registered_plugins_with_same_key_are_isolated_per_lottery(tmp_path, monkeypatch):
    import ai_strategy_lab as lab

    monkeypatch.setattr(lab, "AI_STRATEGIES_DIR", tmp_path)
    monkeypatch.setattr(lab, "_loaded_keys", {})
    for lid, reverse in [("lotto_pl", False), ("loto_6_49_ro", True)]:
        directory = lab.strategies_dir(lid)
        directory.mkdir()
        (directory / "zshared.py").write_text(
            'NAME = "zshared"\nDESCRIPTION = "Audit"\nSOURCE = "test"\n'
            'def create(config):\n'
            f'    return lambda *args: sorted(range(1, config.number_range + 1), reverse={reverse})\n',
            encoding="utf-8")
        lab.promote_plugin(lid, "zshared.py", "zshared", 1, .5)
        lab.register_promoted_for_config(lottery_config.get_lottery_config(lid))
    first = lab.make_extra(lottery_config.get_lottery_config("lotto_pl"), "zshared")
    second = lab.make_extra(lottery_config.get_lottery_config("loto_6_49_ro"), "zshared")
    assert callable(first) and callable(second)
    assert first({}, deque(), {}, 0, 1)[0] == 1
    assert second({}, deque(), {}, 0, 1)[0] == 49


def test_live_prediction_loads_promoted_plugin_for_nonactive_lottery(tmp_path, monkeypatch):
    import ai_strategy_lab as lab

    monkeypatch.setattr(lab, "AI_STRATEGIES_DIR", tmp_path)
    monkeypatch.setattr(lab, "_loaded_keys", {})
    monkeypatch.setattr(auto_loop, "ACTIVE_CONFIG", lottery_config.get_lottery_config("multi_multi_pl"))
    directory = lab.strategies_dir("lotto_pl")
    directory.mkdir()
    (directory / "zreverse.py").write_text(
        'NAME = "zreverse"\nDESCRIPTION = "Audit"\nSOURCE = "test"\n'
        'def create(config):\n'
        '    return lambda *args: list(range(config.number_range, 0, -1))\n', encoding="utf-8")
    lab.promote_plugin("lotto_pl", "zreverse.py", "zreverse", 1, .5)
    state = auto_loop.SearchState(tf=0, hc=0, bal=0, extra_strategies=["zreverse"])
    ranked = auto_loop.rank_numbers_from_draws(make_draws(12), state, "lotto_pl")
    assert ranked == list(range(49, 0, -1))

    # A revoked promotion must not remain usable through the in-memory registry.
    lab.save_registry("lotto_pl", [])
    auto_loop.rank_numbers_from_draws(make_draws(12), state, "lotto_pl")
    assert lab.registered_keys("lotto_pl") == []


@pytest.mark.parametrize("options", [
    {"lookback": 0}, {"lookback": -1}, {"lookback": True}, {"lookback": 1.5},
    {"percentile": -1}, {"percentile": 101}, {"percentile": float("nan")},
])
def test_proxy_rejects_invalid_parameters(options):
    with pytest.raises(ValueError):
        compute_proxy_jackpot(make_draws(2), lottery_config.get_lottery_config("lotto_pl"), **options)


def test_built_strategy_keeps_weights_when_shared_defaults_change(monkeypatch):
    config = lottery_config.get_lottery_config("lotto_pl")
    monkeypatch.setattr(generic_backtest, "BACKTEST_WEIGHTS", dict(auto_loop.SEARCH_WEIGHTS))
    state = auto_loop.SearchState(tf=0, hc=0, bal=1, extra_strategies=[])
    ranker = auto_loop.build_rank_fn_from_state(state, config)
    counts = {n: (50 if n < 8 else 1) for n in range(1, 50)}
    recent = deque([{45, 46, 47, 48, 49}], maxlen=5)
    last_seen = {n: 10 for n in range(1, 50)}
    expected = ranker(counts, recent, last_seen, 100, 5)
    generic_backtest.BACKTEST_WEIGHTS.update(frequency=0, recent=1, delay=0)
    assert ranker(counts, recent, last_seen, 100, 5) == expected


def test_walkforward_factory_loads_requested_lottery_context(monkeypatch):
    draws = make_draws(3)
    config = lottery_config.get_lottery_config("lotto_pl")
    loaded = []
    observed = []
    monkeypatch.setattr(auto_loop.ai_strategy_lab, "promoted_plugins", lambda _: [])
    monkeypatch.setattr(auto_loop, "load_lottery_draws", lambda lid: loaded.append(lid) or draws)
    data = {d.draw_date: JackpotInfo(.1, 0, False) for d in draws}
    monkeypatch.setattr(auto_loop, "load_jackpot_data", lambda lid, ds: data)
    monkeypatch.setattr(auto_loop, "make_jackpot_cycle_rank",
                        lambda jd, dates, cfg: observed.append((jd, dates, cfg.id)) or (lambda *args: list(range(1, 50))))
    auto_loop.build_rank_fn_from_state(auto_loop.SearchState(extra_strategies=["jc"]), config)
    assert loaded == [config.id]
    assert observed == [(data, [d.draw_date for d in draws], config.id)]
