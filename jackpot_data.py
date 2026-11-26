"""Unified jackpot data loader."""

from __future__ import annotations

import csv
from bisect import insort
from datetime import date
from pathlib import Path

from generic_backtest import LotteryDraw
from jackpot_proxy import DEFAULT_CYCLE_LENGTH, JackpotInfo, compute_proxy_jackpot
from lottery_config import get_lottery_config


def load_jackpot_data(
    lottery_id: str,
    draws: list[LotteryDraw],
) -> dict[date, JackpotInfo]:
    """Load real jackpot history when available, else fall back to a proxy."""

    result, _source = load_jackpot_data_with_source(lottery_id, draws)
    return result


def load_jackpot_data_with_source(
    lottery_id: str,
    draws: list[LotteryDraw],
) -> tuple[dict[date, JackpotInfo], str]:
    """Load jackpot history and return both the data and the effective source."""

    config = get_lottery_config(lottery_id)
    if config.jackpot_csv.exists():
        real_data = _load_real_jackpot(config.jackpot_csv, draws)
        if real_data:
            return real_data, "real"
    return compute_proxy_jackpot(draws, config), "proxy"


def _load_real_jackpot(csv_path: Path, draws: list[LotteryDraw]) -> dict[date, JackpotInfo]:
    csv_rows: dict[date, dict[str, float | bool | int]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            draw_date = date.fromisoformat(row["draw_date"])
            csv_rows[draw_date] = {
                "tier1_winners": int(row.get("tier1_winners", 0) or 0),
                "tier1_amount": float(row.get("tier1_amount", 0) or 0),
                "is_rollover": str(row.get("is_rollover", "true")).strip().lower() == "true",
                "total_pool": float(row.get("total_pool", 0) or 0),
            }

    if not csv_rows:
        return {}

    config = get_lottery_config(csv_path.parent.name)
    proxy_data = compute_proxy_jackpot(draws, config)

    ordered_dates: list[date] = []
    seen_dates: set[date] = set()
    for draw in draws:
        if draw.draw_date not in seen_dates:
            ordered_dates.append(draw.draw_date)
            seen_dates.add(draw.draw_date)

    if not any(draw_date in csv_rows for draw_date in ordered_dates):
        return {}

    completed_cycles: list[int] = []
    last_win: int | None = None

    result: dict[date, JackpotInfo] = {}
    draws_since_win = 0

    for index, draw_date in enumerate(ordered_dates):
        row = csv_rows.get(draw_date)
        if index:
            draws_since_win += 1
        if row is None:
            proxy_info = proxy_data.get(draw_date)
            if proxy_info is not None:
                result[draw_date] = proxy_info
                continue
        winners = int(row.get("tier1_winners", 0)) if row else 0
        if winners > 0:
            if last_win is not None:
                insort(completed_cycles, index - last_win)
            last_win = index
            draws_since_win = 0
        median_cycle = (completed_cycles[len(completed_cycles) // 2]
                        if completed_cycles else DEFAULT_CYCLE_LENGTH)

        result[draw_date] = JackpotInfo(
            phase=min(draws_since_win / max(median_cycle, 1), 1.0),
            draws_since_win=draws_since_win,
            is_rollover=bool(row.get("is_rollover", draws_since_win > 0)) if row else (draws_since_win > 0),
            jackpot_amount=float(row.get("tier1_amount", 0.0)) if row else 0.0,
        )

    return result
