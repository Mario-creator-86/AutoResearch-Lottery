"""Scrape jackpot history for Polish lotteries with rollover jackpots."""

from __future__ import annotations

import argparse
import csv
import re
import time
from datetime import date, timedelta
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover - exercised only when dependency missing
    requests = None

from lottery_config import ensure_lottery_storage, get_lottery_config

LOTTERY_URL_MAP = {
    "lotto_pl": "https://wynikilotto.net.pl/lotto/losowania/{date_str}",
    "eurojackpot": "https://wynikilotto.net.pl/eurojackpot/losowania/{date_str}",
}
DRAW_WEEKDAYS = {
    "lotto_pl": {1, 3, 5},
    "eurojackpot": {1, 4},
}
REQUEST_DELAY_SECONDS = 1.0
CSV_HEADER = ["draw_date", "tier1_winners", "tier1_amount", "is_rollover", "total_pool"]


def _last_scraped_date(csv_path: Path) -> date | None:
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


def _parse_money(raw_value: str) -> float:
    cleaned = raw_value.replace("\xa0", "").replace(" ", "")
    cleaned = cleaned.replace("zł", "").replace("PLN", "").replace("EUR", "").replace("€", "")
    cleaned = cleaned.replace(",", ".")
    match = re.search(r"(\d+(?:\.\d+)?)", cleaned)
    return float(match.group(1)) if match else 0.0


def _parse_prize_page(html: str) -> dict[str, float | int | bool] | None:
    if not html:
        return None
    lowered = html.lower()
    if "wyniki" not in lowered and "wygrane" not in lowered:
        return None

    tier1_winners = 0
    tier1_amount = 0.0
    total_pool = 0.0
    is_rollover = "kumulac" in lowered

    winners_match = re.search(
        r"(?:I\s*stopni[a-z]*|jackpot)[^0-9]{0,50}(\d+)\s*(?:wygran|zwyci)",
        html,
        re.IGNORECASE,
    )
    if winners_match:
        tier1_winners = int(winners_match.group(1))
        if tier1_winners > 0:
            is_rollover = False

    amount_match = re.search(
        r"(?:I\s*stopni[a-z]*|jackpot)[^0-9]{0,50}([\d\s,.]+)\s*(?:zł|PLN|EUR|€)",
        html,
        re.IGNORECASE,
    )
    if amount_match:
        tier1_amount = _parse_money(amount_match.group(1))

    pool_match = re.search(
        r"(?:pula|pool)[^0-9]{0,50}([\d\s,.]+)\s*(?:zł|PLN|EUR|€)",
        html,
        re.IGNORECASE,
    )
    if pool_match:
        total_pool = _parse_money(pool_match.group(1))

    return {
        "tier1_winners": tier1_winners,
        "tier1_amount": tier1_amount,
        "is_rollover": is_rollover,
        "total_pool": total_pool,
    }


def _get_draw_dates(lottery_id: str, start_date: date, end_date: date) -> list[date]:
    weekdays = DRAW_WEEKDAYS[lottery_id]
    current = start_date
    result = []
    while current <= end_date:
        if current.weekday() in weekdays:
            result.append(current)
        current += timedelta(days=1)
    return result


def scrape_lottery(lottery_id: str, dry_run: bool = False) -> int:
    if requests is None and not dry_run:
        raise RuntimeError("requests is not installed. Add it to the environment before scraping.")
    if lottery_id not in LOTTERY_URL_MAP:
        raise KeyError(f"Unsupported lottery '{lottery_id}'")

    config = get_lottery_config(lottery_id)
    ensure_lottery_storage(config)
    csv_path = config.jackpot_csv
    raw_pages_dir = config.state_dir / "raw_pages"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    raw_pages_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_HEADER)

    last_seen = _last_scraped_date(csv_path)
    start_date = last_seen + timedelta(days=1) if last_seen else date(2012, 1, 1)
    end_date = date.today()
    draw_dates = _get_draw_dates(lottery_id, start_date, end_date)
    if not draw_dates:
        print(f"{lottery_id}: already up to date")
        return 0

    print(f"{lottery_id}: {len(draw_dates)} candidate dates from {draw_dates[0]} to {draw_dates[-1]}")
    if dry_run:
        return 0

    new_rows = 0
    failures = 0
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for index, draw_date in enumerate(draw_dates, start=1):
            url = LOTTERY_URL_MAP[lottery_id].format(date_str=draw_date.strftime("%d.%m.%Y"))
            try:
                response = requests.get(url, timeout=20)
                response.raise_for_status()
                raw_page_path = raw_pages_dir / f"{draw_date.isoformat()}.html"
                raw_page_path.write_text(response.text, encoding="utf-8")
                parsed = _parse_prize_page(response.text)
            except Exception as exc:  # pragma: no cover - network behavior
                failures += 1
                if failures <= 5:
                    print(f"{lottery_id}: failed {draw_date} -> {exc}")
                time.sleep(REQUEST_DELAY_SECONDS)
                continue

            if parsed is not None:
                writer.writerow(
                    [
                        draw_date.isoformat(),
                        parsed["tier1_winners"],
                        parsed["tier1_amount"],
                        str(parsed["is_rollover"]).lower(),
                        parsed["total_pool"],
                    ]
                )
                new_rows += 1

            if index % 100 == 0:
                print(f"{lottery_id}: processed {index}/{len(draw_dates)} dates")
            time.sleep(REQUEST_DELAY_SECONDS)

    print(f"{lottery_id}: wrote {new_rows} rows ({failures} failures)")
    return new_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape jackpot history for Polish lotteries.")
    parser.add_argument("--lottery", choices=sorted(LOTTERY_URL_MAP), help="Single lottery id to scrape")
    parser.add_argument("--dry-run", action="store_true", help="Show the date range without requesting pages")
    args = parser.parse_args()

    lottery_ids = [args.lottery] if args.lottery else sorted(LOTTERY_URL_MAP)
    total_rows = 0
    for lottery_id in lottery_ids:
        total_rows += scrape_lottery(lottery_id, dry_run=args.dry_run)
    print(f"Total new rows: {total_rows}")


if __name__ == "__main__":
    main()
