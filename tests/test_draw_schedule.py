"""Zilele de extragere din configurație trebuie să corespundă istoricului real.

Un draw_days greșit face agentul de bilete să propună bilete pentru zile fără
extragere (rămân „pending" pentru totdeauna) și să sară zilele reale.
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from load_data import load_lottery_draws
from lottery_config import list_lottery_configs

RECENT_SINCE = date(2024, 1, 1)
MIN_COVERAGE = 0.95  # tolerează extrageri mutate de sărbători


@pytest.mark.parametrize("config", [c for c in list_lottery_configs() if c.draw_days], ids=lambda c: c.id)
def test_configured_draw_days_match_recent_history(config):
    try:
        recent = [d for d in load_lottery_draws(config.id) if d.draw_date >= RECENT_SINCE]
    except FileNotFoundError:
        pytest.skip("fără fișier de istoric")
    if len(recent) < 100:
        pytest.skip("istoric recent insuficient")
    weekdays = Counter(d.draw_date.weekday() for d in recent)
    covered = sum(weekdays[day] for day in config.draw_days) / len(recent)
    assert covered >= MIN_COVERAGE, (
        f"{config.id}: draw_days={config.draw_days} acoperă doar {covered:.1%} din extrageri; "
        f"zile reale={dict(weekdays.most_common())}"
    )
    # și invers: fiecare zi configurată chiar are extrageri
    for day in config.draw_days:
        assert weekdays[day] >= 0.1 * len(recent) / len(config.draw_days), (
            f"{config.id}: ziua {day} e configurată, dar aproape nu are extrageri"
        )
