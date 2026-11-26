"""Scrape historical lottery data from official sources (no API keys needed).

Downloads CSV/data from official lottery websites and converts to a unified format.
Each lottery gets its own CSV file in the data/ directory.

Usage:
    python scrape_lotteries.py              # Download all lotteries
    python scrape_lotteries.py lotto_pl     # Download only Polish Lotto
    python scrape_lotteries.py --list       # Show available lotteries

Unified CSV format:
    draw_date,draw_time,draw_id,number_01,number_02,...,number_NN,bonus_01,...
"""

from __future__ import annotations

import csv
import io
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

import requests

from romania_lotto import fetch_results_since

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


# ─── Lottery configurations ───

LOTTERIES = {
    "multi_multi_pl": {
        "name": "Multi Multi (Poland)",
        "numbers_drawn": 20,
        "number_range": 80,
        "bonus_count": 1,  # plus_number
        "draws_per_day": 2,
        "description": "20/80 keno, 2x daily",
    },
    "lotto_pl": {
        "name": "Lotto (Poland)",
        "numbers_drawn": 6,
        "number_range": 49,
        "bonus_count": 0,
        "draws_per_day": 3,  # Tue, Thu, Sat
        "description": "6/49 classic",
    },
    "mini_lotto_pl": {
        "name": "Mini Lotto (Poland)",
        "numbers_drawn": 5,
        "number_range": 42,
        "bonus_count": 0,
        "draws_per_day": 7,  # daily
        "description": "5/42 daily",
    },
    "kaskada_pl": {
        "name": "Kaskada 12/24 (Poland)",
        "numbers_drawn": 12,
        "number_range": 24,
        "bonus_count": 0,
        "draws_per_day": 2,  # 14:00 and 21:40 daily
        "description": "12/24 twice daily",
    },
    "loto_6_49_ro": {
        "name": "Loto 6/49 (Romania)",
        "numbers_drawn": 6,
        "number_range": 49,
        "bonus_count": 0,
        "draws_per_day": 2,
        "description": "6/49 official archive",
    },    "loto_5_40_ro": {
        "name": "Loto 5/40 (Romania)",
        "numbers_drawn": 6,
        "number_range": 40,
        "bonus_count": 0,
        "draws_per_day": 2,
        "description": "5 picked / 6 drawn official archive",
    },
    "joker_ro": {
        "name": "Joker (Romania)",
        "numbers_drawn": 5,
        "number_range": 45,
        "bonus_count": 1,
        "bonus_range": 20,
        "draws_per_day": 2,
        "description": "5/45 + 1/20 official archive",
    },
    "superenalotto_it": {
        "name": "SuperEnalotto (Italy)",
        "numbers_drawn": 6,
        "number_range": 90,
        "bonus_count": 1,  # Jolly
        "bonus_range": 90,
        "draws_per_day": 4,  # Tue, Thu, Fri, Sat
        "description": "6/90 + jolly, Italy",
    },
    "euromillions": {
        "name": "EuroMillions",
        "numbers_drawn": 5,
        "number_range": 50,
        "bonus_count": 2,  # lucky stars 1-12
        "bonus_range": 12,
        "draws_per_day": 2,  # Tue, Fri
        "description": "5/50 + 2/12 stars",
    },
    "eurojackpot": {
        "name": "EuroJackpot",
        "numbers_drawn": 5,
        "number_range": 50,
        "bonus_count": 2,  # euro numbers 1-12
        "bonus_range": 12,
        "draws_per_day": 2,  # Tue, Fri
        "description": "5/50 + 2/12",
    },
    "uk_lotto": {
        "name": "UK Lotto",
        "numbers_drawn": 6,
        "number_range": 59,
        "bonus_count": 1,
        "draws_per_day": 2,  # Wed, Sat
        "description": "6/59 + bonus ball",
    },
    "powerball_us": {
        "name": "Powerball (US)",
        "numbers_drawn": 5,
        "number_range": 69,
        "bonus_count": 1,  # powerball 1-26
        "bonus_range": 26,
        "draws_per_day": 3,  # Mon, Wed, Sat
        "description": "5/69 + 1/26",
    },
    "mega_millions_us": {
        "name": "Mega Millions (US)",
        "numbers_drawn": 5,
        "number_range": 70,
        "bonus_count": 1,  # mega ball 1-25
        "bonus_range": 25,
        "draws_per_day": 2,  # Tue, Fri
        "description": "5/70 + 1/25",
    },
}
ROMANIA_OFFICIAL_LOTTERIES = {"loto_6_49_ro", "loto_5_40_ro", "joker_ro"}
UK_OFFICIAL_LOTTERIES = {"uk_lotto", "euromillions"}
NATIONAL_LOTTERY_UK_DOWNLOADS = {
    "uk_lotto": "https://api-dfe.national-lottery.co.uk/draw-game/results/1/download?interval=ONE_EIGHTY",
    "euromillions": "https://api-dfe.national-lottery.co.uk/draw-game/results/33/download?interval=ONE_EIGHTY",
}


def write_unified_csv(filepath: Path, draws: list[dict], config: dict) -> int:
    """Write draws to unified CSV format. Returns number of rows written."""
    if not draws:
        return 0

    n_drawn = config["numbers_drawn"]
    n_bonus = config.get("bonus_count", 0)

    # Build header
    header = ["draw_date", "draw_time", "draw_id"]
    for i in range(1, n_drawn + 1):
        header.append(f"number_{i:02d}")
    for i in range(1, n_bonus + 1):
        header.append(f"bonus_{i:02d}")

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for draw in draws:
            row = {
                "draw_date": draw["draw_date"],
                "draw_time": draw.get("draw_time", ""),
                "draw_id": draw.get("draw_id", ""),
            }
            numbers = draw["numbers"]
            for i, num in enumerate(numbers[:n_drawn], 1):
                row[f"number_{i:02d}"] = num
            bonuses = draw.get("bonuses", [])
            for i, b in enumerate(bonuses[:n_bonus], 1):
                row[f"bonus_{i:02d}"] = b
            writer.writerow(row)

    return len(draws)


def append_unified_csv(filepath: Path, draws: list[dict], config: dict) -> int:
    """Append draws to unified CSV format. Returns number of rows written."""
    if not draws:
        return 0

    n_drawn = config["numbers_drawn"]
    n_bonus = config.get("bonus_count", 0)

    header = ["draw_date", "draw_time", "draw_id"]
    for i in range(1, n_drawn + 1):
        header.append(f"number_{i:02d}")
    for i in range(1, n_bonus + 1):
        header.append(f"bonus_{i:02d}")

    if not filepath.exists() or not filepath.read_text(encoding="utf-8").strip():
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        for draw in draws:
            row = {
                "draw_date": draw["draw_date"],
                "draw_time": draw.get("draw_time", ""),
                "draw_id": draw.get("draw_id", ""),
            }
            numbers = draw["numbers"]
            for i, num in enumerate(numbers[:n_drawn], 1):
                row[f"number_{i:02d}"] = num
            bonuses = draw.get("bonuses", [])
            for i, b in enumerate(bonuses[:n_bonus], 1):
                row[f"bonus_{i:02d}"] = b
            writer.writerow(row)

    return len(draws)


def get_last_unified_draw_date(filepath: Path) -> date | None:
    if not filepath.exists():
        return None

    last_seen = None
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                last_seen = date.fromisoformat(row["draw_date"])
            except (KeyError, ValueError):
                continue
    return last_seen


def _filter_draws_newer_than(draws: list[dict], last_date: date | None) -> list[dict]:
    if last_date is None:
        return draws
    filtered = []
    for draw in draws:
        try:
            draw_date = date.fromisoformat(draw["draw_date"])
        except (KeyError, ValueError):
            continue
        if draw_date > last_date:
            filtered.append(draw)
    return filtered


# ─── Scrapers ───

def scrape_wynikilotto(game: str) -> list[dict]:
    """Scrape from wynikilotto.net.pl (Polish lotteries). Direct CSV download."""
    url_map = {
        "multi_multi_pl": "https://www.wynikilotto.net.pl/download/multi_multi.csv",
        "lotto_pl": "https://www.wynikilotto.net.pl/download/lotto.csv",
        "mini_lotto_pl": "https://www.wynikilotto.net.pl/download/mini_lotto.csv",
        "kaskada_pl": "https://www.wynikilotto.net.pl/download/kaskada.csv",
    }
    url = url_map.get(game)
    if not url:
        return []

    print(f"  Downloading from {url} ...")
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()

    draws = []
    content = resp.text
    reader = csv.reader(io.StringIO(content))

    # Try to detect format
    first_line = next(reader, None)
    if not first_line:
        return []

    # wynikilotto.net.pl format varies — detect columns
    # Usually: draw_number, date, numbers...
    for row in reader:
        if len(row) < 3:
            continue
        try:
            # Try different date formats
            draw_date = None
            draw_id = ""
            numbers = []
            bonuses = []

            for i, val in enumerate(row):
                val = val.strip()
                # Try to detect date
                if not draw_date:
                    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
                        try:
                            draw_date = datetime.strptime(val, fmt).date()
                            break
                        except ValueError:
                            pass
                    if draw_date:
                        continue
                # Try to detect draw ID (usually first numeric field before date)
                if not draw_id and val.isdigit() and not draw_date and i == 0:
                    draw_id = val
                    continue
                # Remaining numeric fields are numbers
                if val.isdigit():
                    numbers.append(int(val))

            if draw_date and numbers:
                n_drawn = LOTTERIES[game]["numbers_drawn"]
                n_bonus = LOTTERIES[game].get("bonus_count", 0)

                main_nums = sorted(numbers[:n_drawn])
                bonus_nums = numbers[n_drawn:n_drawn + n_bonus] if n_bonus else []

                draws.append({
                    "draw_date": draw_date.isoformat(),
                    "draw_time": "",
                    "draw_id": draw_id,
                    "numbers": main_nums,
                    "bonuses": bonus_nums,
                })
        except Exception:
            continue

    return draws


def scrape_national_lottery_uk(game: str) -> list[dict]:
    """Scrape the official UK National Lottery CSV download."""
    url = NATIONAL_LOTTERY_UK_DOWNLOADS.get(game)
    if not url:
        return []

    print(f"  Downloading from {url} ...")
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()

    draws = []
    reader = csv.DictReader(io.StringIO(resp.text))

    for row in reader:
        try:
            # UK Lotto CSV format: DrawDate, Ball 1-6, Bonus Ball, etc.
            draw_date = None
            for date_key in ("DrawDate", "Draw Date", "drawDate"):
                if date_key in row:
                    for fmt in ("%d-%b-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%B-%Y"):
                        try:
                            draw_date = datetime.strptime(row[date_key].strip(), fmt).date()
                            break
                        except ValueError:
                            pass
                    break

            if not draw_date:
                continue

            if game == "uk_lotto":
                numbers = []
                for key in ("Ball 1", "Ball 2", "Ball 3", "Ball 4", "Ball 5", "Ball 6"):
                    if key in row and row[key].strip().isdigit():
                        numbers.append(int(row[key]))
                bonuses = []
                for key in ("Bonus Ball", "Bonus"):
                    if key in row and row[key].strip().isdigit():
                        bonuses.append(int(row[key]))
                        break

                if len(numbers) == 6:
                    draws.append({
                        "draw_date": draw_date.isoformat(),
                        "draw_time": "",
                        "draw_id": row.get("DrawNumber", ""),
                        "numbers": sorted(numbers),
                        "bonuses": bonuses,
                    })

            elif game == "euromillions":
                numbers = []
                for key in ("Ball 1", "Ball 2", "Ball 3", "Ball 4", "Ball 5"):
                    if key in row and row[key].strip().isdigit():
                        numbers.append(int(row[key]))
                stars = []
                for key in ("Lucky Star 1", "Lucky Star 2", "Star 1", "Star 2"):
                    if key in row and row[key].strip().isdigit():
                        stars.append(int(row[key]))

                if len(numbers) == 5:
                    draws.append({
                        "draw_date": draw_date.isoformat(),
                        "draw_time": "",
                        "draw_id": row.get("DrawNumber", ""),
                        "numbers": sorted(numbers),
                        "bonuses": sorted(stars),
                    })

        except Exception:
            continue

    return draws


def scrape_data_gov_us(game: str) -> list[dict]:
    """Scrape from data.ny.gov (US lotteries). Direct CSV download."""
    url_map = {
        "powerball_us": "https://data.ny.gov/api/views/d6yy-54nr/rows.csv?accessType=DOWNLOAD",
        "mega_millions_us": "https://data.ny.gov/api/views/5xaw-6ayf/rows.csv?accessType=DOWNLOAD",
    }
    url = url_map.get(game)
    if not url:
        return []

    print(f"  Downloading from {url} ...")
    resp = requests.get(url, headers=HEADERS, timeout=120)
    resp.raise_for_status()

    draws = []
    reader = csv.DictReader(io.StringIO(resp.text))

    for row in reader:
        try:
            # Format: Draw Date, Winning Numbers, Mega Ball/Multiplier
            draw_date = None
            for date_key in ("Draw Date",):
                if date_key in row:
                    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
                        try:
                            draw_date = datetime.strptime(row[date_key].strip(), fmt).date()
                            break
                        except ValueError:
                            pass

            if not draw_date:
                continue

            # Winning Numbers field is space-separated
            nums_str = row.get("Winning Numbers", "").strip()
            if not nums_str:
                continue

            numbers = [int(x) for x in nums_str.split() if x.isdigit()]

            bonuses = []
            if game == "mega_millions_us":
                mb = row.get("Mega Ball", "").strip()
                if mb.isdigit():
                    bonuses = [int(mb)]
            elif game == "powerball_us":
                # In NY data, the last number in "Winning Numbers" might be the powerball
                # Or there's a separate "Multiplier" column
                # Check if there's a separate powerball column
                for key in ("Powerball", "PB"):
                    if key in row and row[key].strip().isdigit():
                        bonuses = [int(row[key])]
                        break

            n_drawn = LOTTERIES[game]["numbers_drawn"]
            if len(numbers) >= n_drawn:
                main_nums = sorted(numbers[:n_drawn])
                if not bonuses and len(numbers) > n_drawn:
                    bonuses = [numbers[n_drawn]]

                draws.append({
                    "draw_date": draw_date.isoformat(),
                    "draw_time": "",
                    "draw_id": "",
                    "numbers": main_nums,
                    "bonuses": bonuses,
                })

        except Exception:
            continue

    return draws


def scrape_eurojackpot() -> list[dict]:
    """Scrape EuroJackpot from eurojackpot.org or similar."""
    # Try euro-jackpot.net which has structured results
    url = "https://www.euro-jackpot.net/results-archive"
    print(f"  Attempting EuroJackpot scrape...")

    # Fallback: try lottoland/lottery.net CSV approach
    # EuroJackpot data is harder to get as bulk CSV
    # Use alternative: scrape from results pages year by year
    draws = []
    current_year = datetime.now().year

    for year in range(2012, current_year + 1):
        try:
            url = f"https://www.lottery.net/eurojackpot/results/{year}"
            print(f"    Fetching {year}...")
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                continue

            # Parse HTML for results
            # lottery.net format: date + numbers in structured HTML
            from html.parser import HTMLParser

            text = resp.text
            # Find patterns like "5 - 12 - 23 - 34 - 45" and "1 - 7"
            # This is fragile but works as fallback
            date_pattern = re.compile(r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})')
            number_blocks = re.findall(r'<li class="ball"[^>]*>(\d+)</li>', text)

            # Group numbers: 5 main + 2 euro per draw
            if number_blocks:
                for i in range(0, len(number_blocks), 7):
                    if i + 6 < len(number_blocks):
                        main = sorted([int(number_blocks[i+j]) for j in range(5)])
                        bonus = sorted([int(number_blocks[i+5]), int(number_blocks[i+6])])
                        # Date parsing would need more context
                        # This is a simplified scraper

            time.sleep(0.5)  # Be polite
        except Exception:
            continue

    # If web scraping didn't work well, try wynikilotto for Polish eurojackpot
    if not draws:
        try:
            url = "https://www.wynikilotto.net.pl/download/eurojackpot.csv"
            print(f"  Trying wynikilotto.net.pl for EuroJackpot...")
            resp = requests.get(url, headers=HEADERS, timeout=60)
            if resp.status_code == 200:
                reader = csv.reader(io.StringIO(resp.text))
                next(reader, None)  # skip header
                for row in reader:
                    if len(row) < 8:
                        continue
                    try:
                        draw_date = None
                        numbers = []
                        draw_id = ""

                        for val in row:
                            val = val.strip()
                            if not draw_date:
                                for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
                                    try:
                                        draw_date = datetime.strptime(val, fmt).date()
                                        break
                                    except ValueError:
                                        pass
                                if draw_date:
                                    continue
                            if val.isdigit() and not draw_id and not draw_date:
                                draw_id = val
                                continue
                            if val.isdigit():
                                numbers.append(int(val))

                        if draw_date and len(numbers) >= 7:
                            draws.append({
                                "draw_date": draw_date.isoformat(),
                                "draw_time": "",
                                "draw_id": draw_id,
                                "numbers": sorted(numbers[:5]),
                                "bonuses": sorted(numbers[5:7]),
                            })
                    except Exception:
                        continue
        except Exception:
            pass

    return draws


def scrape_romania_official(game: str, start_date: date | None = None) -> list[dict]:
    draws = []
    for result in fetch_results_since(game, start_date=start_date, delay_seconds=0.05):
        draws.append(
            {
                "draw_date": result.draw_date.isoformat(),
                "draw_time": "",
                "draw_id": result.draw_date.strftime("%Y%m%d"),
                "numbers": list(result.numbers),
                "bonuses": list(result.bonuses),
            }
        )
    return draws


SUPERENALOTTO_ARCHIVE_URL = "https://www.national-lottery.com/superenalotto/results/{year}-archive"
_SUPERENALOTTO_ROW_RE = re.compile(r"<tr>(.*?)</tr>", re.S)
_SUPERENALOTTO_DATE_RE = re.compile(r"/results/(\d{2})-(\d{2})-(\d{4})")
_SUPERENALOTTO_BALL_RE = re.compile(r'<li class="result[^"]*superenalotto ball dark ball">(\d+)</li>')
_SUPERENALOTTO_JOLLY_RE = re.compile(r'ball dark jolly">(\d+)</li>')


def _parse_superenalotto_year(html: str, year: int) -> list[dict]:
    """Parse one yearly SuperEnalotto archive page into draw dicts."""
    draws = []
    for row_html in _SUPERENALOTTO_ROW_RE.findall(html):
        date_match = _SUPERENALOTTO_DATE_RE.search(row_html)
        if not date_match:
            continue
        day, month, draw_year = (int(part) for part in date_match.groups())
        if draw_year != year:
            continue
        balls = [int(value) for value in _SUPERENALOTTO_BALL_RE.findall(row_html)]
        jolly = _SUPERENALOTTO_JOLLY_RE.findall(row_html)
        if len(balls) < 6:
            continue
        bonuses = [int(jolly[0])] if jolly else []
        draws.append({
            "draw_date": date(draw_year, month, day).isoformat(),
            "draw_time": "",
            "draw_id": "",
            "numbers": sorted(balls[:6]),
            "bonuses": bonuses,
        })
    return draws


def scrape_superenalotto_it(years: list[int] | None = None) -> list[dict]:
    """Scrape SuperEnalotto yearly archives from national-lottery.com."""
    if years is None:
        years = list(range(1997, datetime.now().year + 1))
    draws: list[dict] = []
    for year in years:
        url = SUPERENALOTTO_ARCHIVE_URL.format(year=year)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
        except requests.RequestException as exc:
            print(f"    {year}: request failed ({exc})")
            continue
        if resp.status_code != 200:
            print(f"    {year}: HTTP {resp.status_code}")
            continue
        year_draws = _parse_superenalotto_year(resp.text, year)
        print(f"    {year}: {len(year_draws)} draws")
        draws.extend(year_draws)
        time.sleep(0.3)
    return draws


# ─── Main dispatcher ───

SCRAPER_MAP = {
    "multi_multi_pl": lambda: scrape_wynikilotto("multi_multi_pl"),
    "lotto_pl": lambda: scrape_wynikilotto("lotto_pl"),
    "mini_lotto_pl": lambda: scrape_wynikilotto("mini_lotto_pl"),
    "kaskada_pl": lambda: scrape_wynikilotto("kaskada_pl"),
    "loto_6_49_ro": lambda: scrape_romania_official("loto_6_49_ro"),
    "loto_5_40_ro": lambda: scrape_romania_official("loto_5_40_ro"),
    "joker_ro": lambda: scrape_romania_official("joker_ro"),
    "superenalotto_it": scrape_superenalotto_it,
    "euromillions": lambda: scrape_national_lottery_uk("euromillions"),
    "eurojackpot": scrape_eurojackpot,
    "uk_lotto": lambda: scrape_national_lottery_uk("uk_lotto"),
    "powerball_us": lambda: scrape_data_gov_us("powerball_us"),
    "mega_millions_us": lambda: scrape_data_gov_us("mega_millions_us"),
}


def download_lottery(game_key: str) -> None:
    """Download and save a single lottery's data."""
    if game_key not in LOTTERIES:
        print(f"  Unknown lottery: {game_key}")
        return

    config = LOTTERIES[game_key]
    print(f"\n{'='*60}")
    print(f"  {config['name']} ({config['description']})")
    print(f"{'='*60}")

    scraper = SCRAPER_MAP.get(game_key)
    if not scraper:
        print(f"  No scraper available for {game_key}")
        return

    filepath = DATA_DIR / f"{game_key}.csv"
    incremental_last_date = None
    try:
        if game_key in ROMANIA_OFFICIAL_LOTTERIES:
            incremental_last_date = get_last_unified_draw_date(filepath)
            if incremental_last_date:
                print(f"  Incremental Romania scrape from {incremental_last_date} ...")
            draws = scrape_romania_official(game_key, start_date=incremental_last_date)
        elif game_key in UK_OFFICIAL_LOTTERIES:
            incremental_last_date = get_last_unified_draw_date(filepath)
            if incremental_last_date:
                print(f"  Incremental UK scrape from {incremental_last_date} ...")
            draws = _filter_draws_newer_than(scrape_national_lottery_uk(game_key), incremental_last_date)
        elif game_key == "superenalotto_it" and filepath.exists():
            # Incremental: only refresh the current-year archive page
            incremental_last_date = get_last_unified_draw_date(filepath)
            if incremental_last_date:
                print(f"  Incremental SuperEnalotto scrape from {incremental_last_date} ...")
            draws = scrape_superenalotto_it(years=[datetime.now().year])
            draws = _filter_draws_newer_than(draws, incremental_last_date)
        else:
            draws = scraper()
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    if not draws:
        if game_key in ROMANIA_OFFICIAL_LOTTERIES and incremental_last_date:
            print("  No new Romanian draws found.")
        elif game_key in UK_OFFICIAL_LOTTERIES and incremental_last_date:
            print("  No new UK draws found.")
        elif game_key == "superenalotto_it" and incremental_last_date:
            print("  No new SuperEnalotto draws found.")
        else:
            print("  No data retrieved!")
        return

    # Sort chronologically
    draws.sort(key=lambda d: d["draw_date"])

    # Write to CSV
    if game_key in (ROMANIA_OFFICIAL_LOTTERIES | UK_OFFICIAL_LOTTERIES | {"superenalotto_it"}) and incremental_last_date:
        count = append_unified_csv(filepath, draws, config)
    else:
        count = write_unified_csv(filepath, draws, config)

    first = draws[0]["draw_date"]
    last = draws[-1]["draw_date"]
    if game_key in (ROMANIA_OFFICIAL_LOTTERIES | UK_OFFICIAL_LOTTERIES | {"superenalotto_it"}) and incremental_last_date:
        print(f"  Appended {count} new draws to {filepath}")
        print(f"  New date range: {first} -> {last}")
    else:
        print(f"  Saved {count} draws to {filepath}")
        print(f"  Date range: {first} -> {last}")
    print(f"  Numbers: {config['numbers_drawn']}/{config['number_range']}"
          + (f" + {config['bonus_count']} bonus" if config.get('bonus_count') else ""))


def download_all() -> None:
    """Download all available lotteries."""
    print("Downloading all lottery data...")
    print(f"Output directory: {DATA_DIR}")

    for game_key in LOTTERIES:
        download_lottery(game_key)
        time.sleep(1)  # Be polite between requests

    print(f"\n{'='*60}")
    print("Done! All files saved in data/ directory.")
    print("Run analysis with: python analyze_lottery.py")


def list_lotteries() -> None:
    """Show available lotteries."""
    print("Available lotteries:\n")
    for key, config in LOTTERIES.items():
        scraper = "available" if key in SCRAPER_MAP else "not available"
        print(f"  {key:20s} {config['name']:25s} {config['description']:20s} [{scraper}]")


if __name__ == "__main__":
    if "--list" in sys.argv:
        list_lotteries()
    elif len(sys.argv) > 1 and sys.argv[1] != "--list":
        for game in sys.argv[1:]:
            download_lottery(game)
    else:
        download_all()
