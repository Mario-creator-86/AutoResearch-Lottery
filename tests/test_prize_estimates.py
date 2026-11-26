"""Tabelul de premii trebuie să fie realist: un tabel umflat (ex. Kaskada 356%,
Joker 214% randament) făcea bilanțul biletelor AI să pară pe profit."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import prize_estimates as pe
from lottery_config import get_lottery_config


@pytest.mark.parametrize("lottery_id", sorted(pe.TIERS))
def test_expected_return_is_realistic(lottery_id):
    ev, price = pe.expected_value(lottery_id, get_lottery_config(lottery_id))
    assert price > 0
    assert 0.05 <= ev / price <= 0.75, f"{lottery_id}: randament așteptat {ev / price:.0%}"


def test_every_priced_lottery_has_tiers_and_source():
    assert set(pe.TICKET_PRICES) == set(pe.TIERS)
    for lottery_id in pe.TIERS:
        if lottery_id not in {"lotto_pl", "superenalotto_it"}:
            assert lottery_id in pe.SOURCES


@pytest.mark.parametrize("hits", range(13))
def test_kaskada_pays_only_for_displayed_set(hits):
    prize = pe.estimate_ticket_prize("kaskada_pl", hits) or {}
    assert prize.get("amount", 0) == {8: 2, 9: 8, 10: 25, 11: 1000, 12: 250000}.get(hits, 0)
    assert pe.winning_hit_counts("kaskada_pl", 12) == {8, 9, 10, 11, 12}


def test_multi_multi_pays_zero_hits_but_not_five():
    assert pe.estimate_ticket_prize("multi_multi_pl", 0)["amount"] == 4
    assert pe.estimate_ticket_prize("multi_multi_pl", 5)["amount"] == 0
    assert pe.winning_hit_counts("multi_multi_pl", 10) == {0, 6, 7, 8, 9, 10}


def test_joker_categories_need_joker_number_for_low_hits():
    assert pe.estimate_ticket_prize("joker_ro", 2, 0)["amount"] == 0
    assert pe.estimate_ticket_prize("joker_ro", 2, 1)["amount"] > 0
    assert pe.estimate_ticket_prize("joker_ro", 1, 1)["amount"] > 0
    assert pe.estimate_ticket_prize("joker_ro", 3, 1)["amount"] > pe.estimate_ticket_prize("joker_ro", 3, 0)["amount"]


def test_loto_5_40_three_numbers_win_nothing():
    assert pe.estimate_ticket_prize("loto_5_40_ro", 3)["amount"] == 0
    assert pe.estimate_ticket_prize("loto_5_40_ro", 4)["amount"] > 0
