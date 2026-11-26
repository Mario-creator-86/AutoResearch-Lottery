"""Scrape jackpot history for Romanian lotteries from the official monthly archive."""

from __future__ import annotations

import argparse
import csv
from datetime import date

from lottery_config import ensure_lottery_storage, get_lottery_config
from romania_lotto import fetch_results_since

ROMANIA_JACKPOT_LOTTERIES = ("loto_6_49_ro", "loto_5_40_ro", "joker_ro")
CSV_HEADER = ["draw_date", "tier1_winners", "tier1_amount", "is_rollover", "total_pool"]


def _last_scraped_date(csv_path) -> date | None:
    if not csv_path.exists():
        return None
    last_seen = None
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                last_seen = date.fromisoformat(row["draw_date"])
            except (KeyError, ValueError):
                continue
    return last_seen


def _existing_scraped_dates(csv_path) -> set[date]:
    if not csv_path.exists():
        return set()
    existing_dates: set[date] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                existing_dates.add(date.fromisoformat(row["draw_date"]))
            except (KeyError, ValueError):
                continue
    return existing_dates


def scrape_lottery(lottery_id: str, dry_run: bool = False) -> int:
    if lottery_id not in ROMANIA_JACKPOT_LOTTERIES:
        raise KeyError(f"Unsupported lottery '{lottery_id}'")

    config = get_lottery_config(lottery_id)
    ensure_lottery_storage(config)
    csv_path = config.jackpot_csv
    if not csv_path.exists():
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_HEADER)

    last_seen = _last_scraped_date(csv_path)
    existing_dates = _existing_scraped_dates(csv_path)
    print(f"{lottery_id}: last jackpot row {last_seen}")
    if dry_run:
        return 0

    results = fetch_results_since(lottery_id, start_date=last_seen, delay_seconds=0.05)
    new_results = []
    seen_dates = set(existing_dates)
    for result in results:
        if result.draw_date in seen_dates:
            continue
        new_results.append(result)
        seen_dates.add(result.draw_date)

    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for result in new_results:
            writer.writerow(
                [
                    result.draw_date.isoformat(),
                    result.tier1_winners,
                    result.tier1_amount,
                    str(result.is_rollover).lower(),
                    result.total_pool,
                ]
            )
    print(f"{lottery_id}: wrote {len(new_results)} jackpot rows")
    return len(new_results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape jackpot history for Romanian lotteries.")
    parser.add_argument("--lottery", choices=sorted(ROMANIA_JACKPOT_LOTTERIES), help="Single lottery id to scrape")
    parser.add_argument("--dry-run", action="store_true", help="Show the last scraped date without fetching pages")
    args = parser.parse_args()

    lottery_ids = [args.lottery] if args.lottery else list(ROMANIA_JACKPOT_LOTTERIES)
    total_rows = 0
    for lottery_id in lottery_ids:
        total_rows += scrape_lottery(lottery_id, dry_run=args.dry_run)
    print(f"Total new rows: {total_rows}")


if __name__ == "__main__":
    main()
