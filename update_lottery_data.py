#!/usr/bin/env python3
"""
Download latest lottery draws and append only NEW rows to existing CSV files.
"""

import requests
import io
import csv
from datetime import datetime, date
from pathlib import Path

from lottery_config import get_lottery_config
from romania_lotto import fetch_results_since

BASE_DIR = Path(__file__).resolve().parent


def parse_date_dmy(s):
    """Parse DD.MM.YYYY -> date"""
    return datetime.strptime(s.strip(), "%d.%m.%Y").date()


def parse_date_mdy(s):
    """Parse MM/DD/YYYY -> date"""
    return datetime.strptime(s.strip(), "%m/%d/%Y").date()


def get_last_date_and_id(filepath):
    """Read the last non-empty data line from a CSV and return (last_date, last_id)."""
    path = Path(filepath)
    if not path.exists():
        return None, None
    lines = path.read_text(encoding="utf-8").splitlines()
    # skip header
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith("draw_date"):
            parts = line.split(",")
            last_date = datetime.strptime(parts[0], "%Y-%m-%d").date()
            last_id = parts[2].strip() if len(parts) > 2 else ""
            return last_date, last_id
    return None, None


def download(url, encoding="utf-8"):
    """Download a URL and return text, trying utf-8 then latin-1."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    # try declared encoding first, then latin-1 fallback
    try:
        text = resp.content.decode(encoding)
    except UnicodeDecodeError:
        text = resp.content.decode("latin-1")
    return text


def append_rows(filepath, rows):
    """Append rows (list of strings) to file, ensuring no double newline at end."""
    path = Path(filepath)
    if not path.exists():
        path.write_text("", encoding="utf-8")
    content = path.read_text(encoding="utf-8")
    # ensure file ends with exactly one newline
    if not content.endswith("\n"):
        content += "\n"
    with open(path, "a", encoding="utf-8", newline="") as f:
        for row in rows:
            f.write(row + "\n")


def _expected_unified_header(lottery_id):
    config = get_lottery_config(lottery_id)
    header = ["draw_date", "draw_time", "draw_id"]
    for idx in range(1, config.numbers_drawn + 1):
        header.append(f"number_{idx:02d}")
    for idx in range(1, config.bonus_count + 1):
        header.append(f"bonus_{idx:02d}")
    return ",".join(header)


def ensure_unified_csv(filepath, lottery_id):
    path = Path(filepath)
    expected_header = _expected_unified_header(lottery_id)
    if not path.exists():
        path.write_text(expected_header + "\n", encoding="utf-8")
        return

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        path.write_text(expected_header + "\n", encoding="utf-8")
        return
    if lines[0].strip() == expected_header:
        return
    lines[0] = expected_header
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# 1. MULTI MULTI PL
# ─────────────────────────────────────────────────────────────────────────────
def update_multi_multi():
    filepath = BASE_DIR / "multi_multi_official_history_full.csv"
    last_date, last_id = get_last_date_and_id(filepath)
    print(f"[multi_multi_pl] Last known: date={last_date}, id={last_id}")

    url = "https://www.wynikilotto.net.pl/download/multi_multi.csv"
    text = download(url, encoding="utf-8")

    new_rows = []
    last_id_int = int(last_id) if last_id else 0

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 22:
            continue
        try:
            src_id = int(parts[0].strip())
            src_date = parse_date_dmy(parts[1])
        except (ValueError, IndexError):
            continue

        if src_id <= last_id_int:
            continue  # already have this draw

        # Source: draw_id, DD.MM.YYYY, HH:MM, num1..20, bonus
        # Target: YYYY-MM-DD,,draw_id,bonus,n1..n20,source_notes
        draw_date = src_date.strftime("%Y-%m-%d")
        draw_id = str(src_id)
        # parts[2] = time, parts[3..22] = 20 numbers, parts[23] = bonus
        if len(parts) < 24:
            continue
        numbers = [p.strip() for p in parts[3:23]]  # 20 main numbers
        bonus = parts[23].strip()

        row = f"{draw_date},,{draw_id},{bonus}," + ",".join(numbers) + ",wynikilotto.net.pl scrape"
        new_rows.append(row)

    if new_rows:
        append_rows(filepath, new_rows)
    print(f"[multi_multi_pl] New draws added: {len(new_rows)}  (last: {new_rows[-1].split(',')[0] if new_rows else last_date})")
    return len(new_rows)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOTTO PL
# ─────────────────────────────────────────────────────────────────────────────
def update_lotto_pl():
    filepath = BASE_DIR / "data" / "lotto_pl.csv"
    last_date, last_id = get_last_date_and_id(filepath)
    print(f"[lotto_pl] Last known: date={last_date}, id={last_id}")

    url = "https://www.wynikilotto.net.pl/download/lotto.csv"
    text = download(url, encoding="utf-8")

    new_rows = []
    last_id_int = int(last_id) if last_id else 0

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 8:
            continue
        try:
            src_id = int(parts[0].strip())
            src_date = parse_date_dmy(parts[1])
        except (ValueError, IndexError):
            continue

        if src_id <= last_id_int:
            continue

        # Source: draw_id, DD.MM.YYYY, num1..6
        # Target: YYYY-MM-DD,,draw_id,n1,n2,n3,n4,n5,n6
        draw_date = src_date.strftime("%Y-%m-%d")
        draw_id = str(src_id)
        numbers = [p.strip() for p in parts[2:8]]

        row = f"{draw_date},,{draw_id}," + ",".join(numbers)
        new_rows.append(row)

    if new_rows:
        append_rows(filepath, new_rows)
    print(f"[lotto_pl] New draws added: {len(new_rows)}  (last: {new_rows[-1].split(',')[0] if new_rows else last_date})")
    return len(new_rows)


# ─────────────────────────────────────────────────────────────────────────────
# 3. MINI LOTTO PL
# ─────────────────────────────────────────────────────────────────────────────
def update_mini_lotto_pl():
    filepath = BASE_DIR / "data" / "mini_lotto_pl.csv"
    last_date, last_id = get_last_date_and_id(filepath)
    print(f"[mini_lotto_pl] Last known: date={last_date}, id={last_id}")

    url = "https://www.wynikilotto.net.pl/download/mini_lotto.csv"
    text = download(url, encoding="utf-8")

    new_rows = []
    last_id_int = int(last_id) if last_id else 0

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 7:
            continue
        try:
            src_id = int(parts[0].strip())
            src_date = parse_date_dmy(parts[1])
        except (ValueError, IndexError):
            continue

        if src_id <= last_id_int:
            continue

        # Source: draw_id, DD.MM.YYYY, num1..5
        # Target: YYYY-MM-DD,,draw_id,n1,n2,n3,n4,n5
        draw_date = src_date.strftime("%Y-%m-%d")
        draw_id = str(src_id)
        numbers = [p.strip() for p in parts[2:7]]

        row = f"{draw_date},,{draw_id}," + ",".join(numbers)
        new_rows.append(row)

    if new_rows:
        append_rows(filepath, new_rows)
    print(f"[mini_lotto_pl] New draws added: {len(new_rows)}  (last: {new_rows[-1].split(',')[0] if new_rows else last_date})")
    return len(new_rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4. EUROJACKPOT
# ─────────────────────────────────────────────────────────────────────────────
def update_eurojackpot():
    filepath = BASE_DIR / "data" / "eurojackpot.csv"
    last_date, last_id = get_last_date_and_id(filepath)
    print(f"[eurojackpot] Last known: date={last_date}, id={last_id}")

    url = "https://www.wynikilotto.net.pl/download/eurojackpot.csv"
    text = download(url, encoding="utf-8")

    new_rows = []
    last_id_int = int(last_id) if last_id else 0

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 9:
            continue
        try:
            src_id = int(parts[0].strip())
            src_date = parse_date_dmy(parts[1])
        except (ValueError, IndexError):
            continue

        if src_id <= last_id_int:
            continue

        # Source: draw_id(zero-padded), DD.MM.YYYY, num1..5, bonus1, bonus2
        # Target: YYYY-MM-DD,,draw_id,n1,n2,n3,n4,n5,b1,b2
        draw_date = src_date.strftime("%Y-%m-%d")
        draw_id = f"{src_id:04d}"
        numbers = [p.strip() for p in parts[2:7]]  # 5 main numbers
        bonus1 = parts[7].strip()
        bonus2 = parts[8].strip()

        row = f"{draw_date},,{draw_id}," + ",".join(numbers) + f",{bonus1},{bonus2}"
        new_rows.append(row)

    if new_rows:
        append_rows(filepath, new_rows)
    print(f"[eurojackpot] New draws added: {len(new_rows)}  (last: {new_rows[-1].split(',')[0] if new_rows else last_date})")
    return len(new_rows)


# ─────────────────────────────────────────────────────────────────────────────
# 5. MEGA MILLIONS US
# ─────────────────────────────────────────────────────────────────────────────
def update_mega_millions():
    filepath = BASE_DIR / "data" / "mega_millions_us.csv"
    last_date, _ = get_last_date_and_id(filepath)
    print(f"[mega_millions_us] Last known: date={last_date}")

    url = "https://data.ny.gov/api/views/5xaw-6ayf/rows.csv?accessType=DOWNLOAD"
    text = download(url, encoding="utf-8")

    new_rows = []
    reader = csv.DictReader(io.StringIO(text))

    for row in reader:
        try:
            src_date = parse_date_mdy(row["Draw Date"])
        except (ValueError, KeyError):
            continue

        if last_date and src_date <= last_date:
            continue

        winning = row["Winning Numbers"].strip().split()
        mega_ball = row["Mega Ball"].strip()

        if len(winning) < 5:
            continue

        draw_date = src_date.strftime("%Y-%m-%d")
        numbers = winning[:5]
        row_str = f"{draw_date},,,{','.join(numbers)},{mega_ball}"
        new_rows.append((src_date, row_str))

    # sort by date ascending
    new_rows.sort(key=lambda x: x[0])
    new_rows = [r[1] for r in new_rows]

    if new_rows:
        append_rows(filepath, new_rows)
    print(f"[mega_millions_us] New draws added: {len(new_rows)}  (last: {new_rows[-1].split(',')[0] if new_rows else last_date})")
    return len(new_rows)


# ─────────────────────────────────────────────────────────────────────────────
# 6. POWERBALL US
# ─────────────────────────────────────────────────────────────────────────────
def update_powerball():
    filepath = BASE_DIR / "data" / "powerball_us.csv"
    last_date, _ = get_last_date_and_id(filepath)
    print(f"[powerball_us] Last known: date={last_date}")

    url = "https://data.ny.gov/api/views/d6yy-54nr/rows.csv?accessType=DOWNLOAD"
    text = download(url, encoding="utf-8")

    new_rows = []
    reader = csv.DictReader(io.StringIO(text))

    for row in reader:
        try:
            src_date = parse_date_mdy(row["Draw Date"])
        except (ValueError, KeyError):
            continue

        if last_date and src_date <= last_date:
            continue

        winning = row["Winning Numbers"].strip().split()
        if len(winning) < 6:
            continue

        draw_date = src_date.strftime("%Y-%m-%d")
        numbers = winning[:5]
        powerball = winning[5]

        row_str = f"{draw_date},,,{','.join(numbers)},{powerball}"
        new_rows.append((src_date, row_str))

    # sort by date ascending
    new_rows.sort(key=lambda x: x[0])
    new_rows = [r[1] for r in new_rows]

    if new_rows:
        append_rows(filepath, new_rows)
    print(f"[powerball_us] New draws added: {len(new_rows)}  (last: {new_rows[-1].split(',')[0] if new_rows else last_date})")
    return len(new_rows)


# ─────────────────────────────────────────────────────────────────────────────
# 7. ROMANIA OFFICIAL LOTTERIES
# ─────────────────────────────────────────────────────────────────────────────
def update_romania_lottery(lottery_id):
    filepath = BASE_DIR / "data" / f"{lottery_id}.csv"
    ensure_unified_csv(filepath, lottery_id)
    last_date, _ = get_last_date_and_id(filepath)
    print(f"[{lottery_id}] Last known: date={last_date}")

    config = get_lottery_config(lottery_id)
    results = fetch_results_since(lottery_id, start_date=last_date, delay_seconds=0.05)
    new_rows = []
    for result in results:
        draw_id = result.draw_date.strftime("%Y%m%d")
        row = [result.draw_date.isoformat(), "", draw_id]
        row.extend(str(number) for number in result.numbers[: config.numbers_drawn])
        row.extend(str(bonus) for bonus in result.bonuses[: config.bonus_count])
        new_rows.append(",".join(row))

    if new_rows:
        append_rows(filepath, new_rows)
    print(f"[{lottery_id}] New draws added: {len(new_rows)}  (last: {new_rows[-1].split(',')[0] if new_rows else last_date})")
    return len(new_rows)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Lottery data update — " + date.today().strftime("%Y-%m-%d"))
    print("=" * 60)

    results = {}
    results["multi_multi_pl"] = update_multi_multi()
    print()
    results["lotto_pl"] = update_lotto_pl()
    print()
    results["mini_lotto_pl"] = update_mini_lotto_pl()
    print()
    results["loto_6_49_ro"] = update_romania_lottery("loto_6_49_ro")
    print()
    results["loto_5_40_ro"] = update_romania_lottery("loto_5_40_ro")
    print()
    results["joker_ro"] = update_romania_lottery("joker_ro")
    print()
    results["eurojackpot"] = update_eurojackpot()
    print()
    results["mega_millions_us"] = update_mega_millions()
    print()
    results["powerball_us"] = update_powerball()

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, count in results.items():
        print(f"  {name}: {count} new draws appended")
