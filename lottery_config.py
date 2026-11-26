from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"

LEGACY_MULTI_MULTI_RESULTS = PROJECT_DIR / "results.tsv"
LEGACY_MULTI_MULTI_PROGRESS = PROJECT_DIR / "progress.json"
LEGACY_MULTI_MULTI_STATE = PROJECT_DIR / "autoloop_state.json"

RESULTS_HEADER = (
    "commit\taverage_hits\tstatus\tstrategy\thit_rate_2_plus\thit_rate_3_plus\t"
    "portfolio_avg\telapsed_s\tdescription\n"
)


@dataclass(frozen=True, slots=True)
class LotteryConfig:
    id: str
    name: str
    numbers_drawn: int
    number_range: int
    ticket_size: int
    min_pick_count: int | None = None
    max_pick_count: int | None = None
    bonus_count: int = 0
    bonus_range: int | None = None
    min_bonus_pick_count: int | None = None
    max_bonus_pick_count: int | None = None
    current_rules_since: str | None = None  # ISO date: only use draws from this date onward
    # Draw schedule: which weekdays (0=Mon..6=Sun) have draws, and local times (HH:MM)
    # If empty, assume daily draws.
    draw_days: tuple[int, ...] = ()
    draw_times_local: tuple[str, ...] = ()  # e.g. ("14:00", "21:40")

    @property
    def draws_file(self) -> Path:
        return DATA_DIR / f"{self.id}.csv"

    @property
    def state_dir(self) -> Path:
        return DATA_DIR / self.id

    @property
    def results_file(self) -> Path:
        return self.state_dir / "results.tsv"

    @property
    def progress_file(self) -> Path:
        return self.state_dir / "progress.json"

    @property
    def loop_state_file(self) -> Path:
        return self.state_dir / "autoloop_state.json"

    @property
    def history_file(self) -> Path:
        return self.state_dir / "ticket_history.json"

    @property
    def lock_file(self) -> Path:
        return self.state_dir / ".experiment.lock"

    @property
    def stop_request_file(self) -> Path:
        return self.state_dir / ".autoloop.stop"

    @property
    def baseline(self) -> float:
        return self.ticket_size * self.numbers_drawn / self.number_range

    @property
    def expected_rate(self) -> float:
        return self.numbers_drawn / self.number_range

    @property
    def jackpot_csv(self) -> Path:
        return self.state_dir / "jackpot_history.csv"

    @property
    def legacy_best_file(self) -> Path:
        return self.state_dir / "legacy_best_reference.json"

    @property
    def min_ticket_numbers(self) -> int:
        return self.min_pick_count if self.min_pick_count is not None else self.ticket_size

    @property
    def max_ticket_numbers(self) -> int:
        return self.max_pick_count if self.max_pick_count is not None else self.ticket_size

    @property
    def min_bonus_numbers(self) -> int:
        if not self.bonus_range:
            return 0
        if self.min_bonus_pick_count is not None:
            return self.min_bonus_pick_count
        return self.bonus_count

    @property
    def max_bonus_numbers(self) -> int:
        if not self.bonus_range:
            return 0
        if self.max_bonus_pick_count is not None:
            return self.max_bonus_pick_count
        return self.bonus_count

    def next_draw_date(self, today: date | None = None) -> date:
        """Return the date of the next draw for this lottery.

        If the lottery has scheduled draw_days, find the next day with a draw.
        If today is a draw day and the draw hasn't happened yet, return today.
        Otherwise return the next scheduled draw day.
        If draw_days is empty, assume daily draws → return today.
        """
        if today is None:
            today = date.today()

        if not self.draw_days:
            # Daily draws (Multi Multi, Mini Lotto, Kaskada, Romania lotteries)
            return today

        # Scheduled draws: find next day that matches
        # Check if today is a draw day
        weekday = today.weekday()  # 0=Mon..6=Sun
        if weekday in self.draw_days:
            # Today is a draw day — return today
            return today

        # Find next draw day within the next 7 days
        for offset in range(1, 8):
            candidate = today + timedelta(days=offset)
            if candidate.weekday() in self.draw_days:
                return candidate

        # Fallback: shouldn't happen if draw_days is valid
        return today


LOTTERIES: dict[str, LotteryConfig] = {
    "multi_multi_pl": LotteryConfig(
        id="multi_multi_pl",
        name="Multi Multi (Poland)",
        numbers_drawn=20,
        number_range=80,
        ticket_size=10,
        min_pick_count=1,
        max_pick_count=10,
        # Daily draws at 14:00 and 21:40
        draw_days=(),  # empty = daily
        draw_times_local=("14:00", "21:40"),
    ),
    "lotto_pl": LotteryConfig(
        id="lotto_pl",
        name="Lotto (Poland)",
        numbers_drawn=6,
        number_range=49,
        ticket_size=6,
        min_pick_count=6,
        max_pick_count=12,
        # Tue, Thu, Sat
        draw_days=(1, 3, 5),
        draw_times_local=("21:40",),
    ),
    "mini_lotto_pl": LotteryConfig(
        id="mini_lotto_pl",
        name="Mini Lotto (Poland)",
        numbers_drawn=5,
        number_range=42,
        ticket_size=5,
        min_pick_count=5,
        max_pick_count=12,
        # Daily
        draw_days=(),
        draw_times_local=("21:40",),
    ),
    "kaskada_pl": LotteryConfig(
        id="kaskada_pl",
        name="Kaskada 12/24 (Poland)",
        numbers_drawn=12,
        number_range=24,
        ticket_size=12,
        min_pick_count=6,
        max_pick_count=12,
        # Daily at 14:00 and 21:40
        draw_days=(),
        draw_times_local=("14:00", "21:40"),
    ),
    "loto_6_49_ro": LotteryConfig(
        id="loto_6_49_ro",
        name="Loto 6/49 (Romania)",
        numbers_drawn=6,
        number_range=49,
        ticket_size=6,
        min_pick_count=6,
        max_pick_count=10,
        # Thu, Sun
        draw_days=(3, 6),
        draw_times_local=("18:00",),
    ),
    "loto_5_40_ro": LotteryConfig(
        id="loto_5_40_ro",
        name="Loto 5/40 (Romania)",
        numbers_drawn=6,
        number_range=40,
        ticket_size=5,
        min_pick_count=5,
        max_pick_count=10,
        # Thu, Sun (împreună cu 6/49 și Joker — confirmat din istoric; era greșit Sun+Tue,
        # ceea ce genera bilete pentru marți fără extragere, rămase „pending" pentru totdeauna)
        draw_days=(3, 6),
        draw_times_local=("15:00",),
    ),
    "joker_ro": LotteryConfig(
        id="joker_ro",
        name="Joker (Romania)",
        numbers_drawn=5,
        number_range=45,
        ticket_size=5,
        min_pick_count=5,
        max_pick_count=10,
        bonus_count=1,
        bonus_range=20,
        min_bonus_pick_count=1,
        max_bonus_pick_count=1,
        # Thu, Sun
        draw_days=(3, 6),
        draw_times_local=("18:00",),
    ),
    "superenalotto_it": LotteryConfig(
        id="superenalotto_it",
        name="SuperEnalotto (Italy)",
        numbers_drawn=6,
        number_range=90,
        ticket_size=6,
        min_pick_count=6,
        max_pick_count=12,
        bonus_count=1,  # Jolly
        bonus_range=90,
        # Tue, Thu, Fri, Sat
        draw_days=(1, 3, 4, 5),
        draw_times_local=("20:00",),
    ),
    "eurojackpot": LotteryConfig(
        id="eurojackpot",
        name="EuroJackpot",
        numbers_drawn=5,
        number_range=50,
        ticket_size=5,
        min_pick_count=5,
        max_pick_count=12,
        bonus_count=2,
        bonus_range=12,
        min_bonus_pick_count=2,
        max_bonus_pick_count=12,
        # Tue, Fri
        draw_days=(1, 4),
        draw_times_local=("21:00",),
    ),
    "powerball_us": LotteryConfig(
        id="powerball_us",
        name="Powerball (US)",
        numbers_drawn=5,
        number_range=69,
        ticket_size=5,
        bonus_count=1,
        bonus_range=26,
        current_rules_since="2015-10-07",
        # Mon, Wed, Sat
        draw_days=(0, 2, 5),
        draw_times_local=("23:00",),
    ),
    "mega_millions_us": LotteryConfig(
        id="mega_millions_us",
        name="Mega Millions (US)",
        numbers_drawn=5,
        number_range=70,
        ticket_size=5,
        bonus_count=1,
        bonus_range=25,
        current_rules_since="2017-10-28",
        # Tue, Fri
        draw_days=(1, 4),
        draw_times_local=("23:00",),
    ),
    "uk_lotto": LotteryConfig(
        id="uk_lotto",
        name="UK Lotto",
        numbers_drawn=6,
        number_range=59,
        ticket_size=6,
        bonus_count=1,
        bonus_range=59,
        current_rules_since="2025-10-01",
        # Wed, Sat
        draw_days=(2, 5),
        draw_times_local=("20:45",),
    ),
    "euromillions": LotteryConfig(
        id="euromillions",
        name="EuroMillions",
        numbers_drawn=5,
        number_range=50,
        ticket_size=5,
        bonus_count=2,
        bonus_range=12,
        # Tue, Fri
        draw_days=(1, 4),
        draw_times_local=("21:45",),
    ),
}


def get_lottery_config(lottery_id: str) -> LotteryConfig:
    try:
        return LOTTERIES[lottery_id]
    except KeyError as exc:
        raise KeyError(f"Unknown lottery '{lottery_id}'") from exc


# Loterii retrase din exploatare la cererea utilizatorului (2026-09-08).
# Rămân definite pentru istoric / verificări vechi / teste, dar nu mai sunt
# optimizate, jucate sau actualizate automat (sunt filtrate din
# list_lottery_configs, deci toți agenții le ignoră).
RETIRED_LOTTERIES = frozenset({
    "eurojackpot",
    "powerball_us",
    "mega_millions_us",
    "uk_lotto",
    "euromillions",
})


def list_lottery_configs() -> list[LotteryConfig]:
    return [c for c in LOTTERIES.values() if c.id not in RETIRED_LOTTERIES]


def ensure_lottery_storage(config: LotteryConfig) -> None:
    config.state_dir.mkdir(parents=True, exist_ok=True)
    if config.id == "multi_multi_pl":
        if LEGACY_MULTI_MULTI_RESULTS.exists() and not config.results_file.exists():
            shutil.copy2(LEGACY_MULTI_MULTI_RESULTS, config.results_file)
        if LEGACY_MULTI_MULTI_PROGRESS.exists() and not config.progress_file.exists():
            shutil.copy2(LEGACY_MULTI_MULTI_PROGRESS, config.progress_file)
        if LEGACY_MULTI_MULTI_STATE.exists() and not config.loop_state_file.exists():
            shutil.copy2(LEGACY_MULTI_MULTI_STATE, config.loop_state_file)
    if not config.results_file.exists():
        config.results_file.write_text(RESULTS_HEADER, encoding="utf-8")
