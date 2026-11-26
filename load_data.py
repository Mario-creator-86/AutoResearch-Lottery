"""Load lottery draws from CSV for autoresearch experiments."""

from __future__ import annotations

import csv
import os
import sys
from functools import lru_cache
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from generic_backtest import LotteryDraw
from lottery_config import LotteryConfig, get_lottery_config

LOTO_POLONIA_SRC = Path(os.environ.get("LOTO_POLONIA_SRC", str(Path(__file__).resolve().parent.parent / "Loto Polonia" / "src")))
if str(LOTO_POLONIA_SRC) not in sys.path:
    sys.path.insert(0, str(LOTO_POLONIA_SRC))

try:
    from loto_polonia.models import DrawRecord
except ImportError:
    @dataclass(frozen=True, slots=True)
    class DrawRecord:
        draw_date: date
        draw_time: str | None
        draw_identifier: str | None
        numbers: tuple[int, ...]
        source: str
        fetched_at: datetime
        plus_number: int | None = None

DEFAULT_CSV = Path(__file__).parent / "multi_multi_official_history_full.csv"


def load_draws(csv_path: Path | None = None) -> list[DrawRecord]:
    """Parse the full history CSV into DrawRecord objects, sorted chronologically."""
    if csv_path is None:
        csv_path = DEFAULT_CSV
    draws: list[DrawRecord] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            numbers = tuple(int(row[f"number_{i:02d}"]) for i in range(1, 21))
            plus_raw = row.get("plus_number", "").strip()
            plus = int(plus_raw) if plus_raw else None
            draw_time = row.get("draw_time", "").strip() or None
            draw_id = row.get("draw_id", "").strip() or None
            draws.append(DrawRecord(
                draw_date=date.fromisoformat(row["draw_date"]),
                draw_time=draw_time,
                draw_identifier=draw_id,
                numbers=numbers,
                source="csv_history",
                fetched_at=datetime.now(),
                plus_number=plus,
            ))
    draws.sort(key=lambda d: (d.draw_date, d.draw_time or ""))
    return draws


def load_lottery_draws(lottery_id: str = "multi_multi_pl") -> list[LotteryDraw]:
    """Load unified CSV draws for any supported lottery."""
    config = get_lottery_config(lottery_id)
    try:
        stat = config.draws_file.stat()
        cached = _load_lottery_draws_cached(
            config.id,
            str(config.draws_file),
            stat.st_mtime_ns,
            stat.st_size,
        )
        return list(cached)
    except FileNotFoundError:
        return load_lottery_draws_from_config(config)


def load_specific_lottery_draws(lottery_id: str, target_dates: set[str] | None = None, recent_days: int = 30) -> list[LotteryDraw]:
    """Filter the shared, validated history without reparsing the CSV."""
    draws = load_lottery_draws(lottery_id)
    if target_dates is not None:
        return [draw for draw in draws if draw.draw_date.isoformat() in target_dates]
    from datetime import timedelta

    recent_cutoff = date.today() - timedelta(days=recent_days)
    return [draw for draw in draws if draw.draw_date >= recent_cutoff]


def _draw_matches_current_rules(config: LotteryConfig, numbers: list[int], bonuses: list[int]) -> bool:
    if len(numbers) != config.numbers_drawn or len(set(numbers)) != len(numbers):
        return False
    if len(bonuses) != config.bonus_count or len(set(bonuses)) != len(bonuses):
        return False
    if any(number < 1 or number > config.number_range for number in numbers):
        return False
    if config.bonus_range is not None and any(bonus < 1 or bonus > config.bonus_range for bonus in bonuses):
        return False
    return True


def load_lottery_draws_from_config(config: LotteryConfig) -> list[LotteryDraw]:
    """Load draws from CSV, handling both multi-column and inline string formats."""
    draws: list[LotteryDraw] = []

    # Detect CSV format by checking header
    header = None
    with config.draws_file.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)

    if header is None:
        return []

    # Check if this is inline format (draw_date,numbers) like multi_multi_pl
    is_inline_format = len(header) == 2 and header[0] == "draw_date" and header[1] == "numbers"

    with config.draws_file.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # Skip header

        for row in reader:
            if not row or not any(cell.strip() for cell in row):
                continue

            try:
                if is_inline_format:
                    # Format: draw_date,"num1,num2,num3,..."
                    draw_date = date.fromisoformat(row[0].strip())
                    # Parse numbers from the quoted string
                    numbers_str = row[1].strip().strip('"')  # Remove quotes if present
                    numbers = [int(n.strip()) for n in numbers_str.split(",") if n.strip()]
                    bonuses = []
                    draw_time = None
                    draw_id = None
                else:
                    # Format: draw_date,draw_time,draw_id,number_01,number_02,...
                    expected_len = 3 + config.numbers_drawn + config.bonus_count
                    if len(row) < expected_len:
                        continue
                    draw_date = date.fromisoformat(row[0].strip())
                    numbers = [int(row[3 + idx].strip()) for idx in range(config.numbers_drawn)]
                    bonuses = []
                    for idx in range(config.bonus_count):
                        value = row[3 + config.numbers_drawn + idx].strip()
                        if value:
                            bonuses.append(int(value))
                    draw_time = row[1].strip() or None
                    draw_id = row[2].strip() or None
            except (IndexError, ValueError, AttributeError) as e:
                continue

            if not _draw_matches_current_rules(config, numbers, bonuses):
                continue
            if config.current_rules_since and draw_date < date.fromisoformat(config.current_rules_since):
                continue

            draws.append(
                LotteryDraw(
                    draw_date=draw_date,
                    draw_time=draw_time,
                    draw_identifier=draw_id,
                    numbers=tuple(numbers),
                    bonuses=tuple(bonuses),
                )
            )

    draws.sort(key=lambda draw: (draw.draw_date, draw.draw_time or ""))
    seen: set[tuple] = set()
    unique: list[LotteryDraw] = []
    for draw in draws:
        key = (draw.draw_date, draw.draw_time, draw.draw_identifier,
               tuple(sorted(draw.numbers)), tuple(sorted(draw.bonuses)))
        if key not in seen:
            seen.add(key)
            unique.append(draw)
    return unique


@lru_cache(maxsize=64)
def _load_lottery_draws_cached(
    lottery_id: str,
    _draws_path: str,
    _mtime_ns: int,
    _size: int,
) -> tuple[LotteryDraw, ...]:
    config = get_lottery_config(lottery_id)
    return tuple(load_lottery_draws_from_config(config))


if __name__ == "__main__":
    all_draws = load_draws()
    print(f"Loaded {len(all_draws)} draws")
    print(f"Date range: {all_draws[0].draw_date} -> {all_draws[-1].draw_date}")
