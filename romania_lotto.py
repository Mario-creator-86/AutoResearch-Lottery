from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover - exercised only when dependency missing
    requests = None

from bs4 import BeautifulSoup

ROMANIA_BASE_URL = "https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php"
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ro-RO,ro;q=0.9,en-US;q=0.8,en;q=0.7",
    "Cache-Control": "no-cache",
}
REQUEST_RETRIES = 4
REQUEST_RETRY_DELAY_SECONDS = 1.0

ROMANIA_RESULTS_PAGES = {
    "loto_6_49_ro": "/jocuri/649_si_noroc/rezultate_extragere.html",
    "loto_5_40_ro": "/jocuri/540_si_super_noroc/rezultate_extrageri.html",
    "joker_ro": "/jocuri/joker_si_noroc_plus/rezultate_extrageri.html",
}

ROMANIA_DRAW_SPECS = {
    "loto_6_49_ro": {"numbers_drawn": 6, "bonus_count": 0},
    "loto_5_40_ro": {"numbers_drawn": 6, "bonus_count": 0},
    "joker_ro": {"numbers_drawn": 5, "bonus_count": 1},
}
ROMANIA_START_YEARS = {
    "loto_6_49_ro": 1998,
    "loto_5_40_ro": 1998,
    "joker_ro": 2001,
}


@dataclass(frozen=True, slots=True)
class RomaniaDrawResult:
    lottery_id: str
    draw_date: date
    numbers: tuple[int, ...]
    bonuses: tuple[int, ...]
    tier1_winners: int
    tier1_amount: float
    is_rollover: bool
    total_pool: float


def _page_url(lottery_id: str) -> str:
    try:
        return f"{ROMANIA_BASE_URL}{ROMANIA_RESULTS_PAGES[lottery_id]}"
    except KeyError as exc:
        raise KeyError(f"Unsupported Romanian lottery '{lottery_id}'") from exc


def _request_with_retries(session: requests.Session, method: str, url: str, **kwargs: Any):
    last_error = None
    for attempt in range(REQUEST_RETRIES):
        try:
            response = session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except Exception as exc:  # pragma: no cover - network behavior
            last_error = exc
            if attempt >= REQUEST_RETRIES - 1:
                raise
            time.sleep(REQUEST_RETRY_DELAY_SECONDS * (attempt + 1))
    raise last_error


def _parse_money(raw_value: str) -> float:
    cleaned = raw_value.replace("\xa0", "").replace(" ", "")
    cleaned = cleaned.replace("lei", "").replace("RON", "").replace("Lei", "")
    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif cleaned.count(".") > 1:
        parts = cleaned.split(".")
        cleaned = "".join(parts[:-1]) + "." + parts[-1]
    match = re.search(r"(\d+(?:\.\d+)?)", cleaned)
    return float(match.group(1)) if match else 0.0


def _parse_draw_date(raw_value: str) -> date | None:
    raw_value = raw_value.strip()
    for fmt in ("%d.%m.%Y", "%Y-%m-%d"):
        try:
            return date.fromisoformat(raw_value) if fmt == "%Y-%m-%d" else datetime.strptime(raw_value, fmt).date()
        except ValueError:
            continue
    return None


def _parse_number_from_src(src: str) -> int | None:
    match = re.search(r"/(\d+)\.png(?:\?|$)", src)
    if not match:
        return None
    return int(match.group(1))


def _extract_available_years(html: str) -> list[int]:
    soup = BeautifulSoup(html, "html.parser")
    years = []
    for option in soup.select("form#formSubmit select[name='select-year'] option"):
        value = (option.get("value") or "").strip()
        if value.isdigit() and value != "0":
            years.append(int(value))
    return sorted(set(years))


def _parse_tier_data(block) -> tuple[int, float, bool, float]:
    table = block.find("table")
    if table is None:
        return 0, 0.0, False, 0.0

    first_cells = None
    total_pool = 0.0
    for row in table.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]
        if not cells:
            continue
        row_text = " ".join(cells)
        if "Fond total de castiguri" in row_text:
            total_pool = _parse_money(row_text)
            continue
        if first_cells is None and len(cells) >= 4:
            first_cells = cells

    if not first_cells:
        return 0, 0.0, False, total_pool

    winners_text = first_cells[1]
    value_text = first_cells[2]
    report_text = first_cells[3]

    winners_match = re.search(r"\d+", winners_text.replace(".", ""))
    winners = int(winners_match.group(0)) if winners_match else 0
    report_amount = _parse_money(report_text)
    value_amount = _parse_money(value_text)
    is_rollover = "report" in winners_text.lower() or (winners == 0 and report_amount > 0)
    tier1_amount = report_amount if report_amount > 0 else value_amount
    if winners > 0:
        is_rollover = False

    return winners, tier1_amount, is_rollover, total_pool


def parse_results_page(lottery_id: str, html: str) -> list[RomaniaDrawResult]:
    spec = ROMANIA_DRAW_SPECS[lottery_id]
    soup = BeautifulSoup(html, "html.parser")
    results: dict[date, RomaniaDrawResult] = {}

    for block in soup.select("div.rezultate-extrageri-content.resultDiv"):
        date_tag = block.select_one("div.button-open-details span")
        if date_tag is None:
            continue
        draw_date = _parse_draw_date(date_tag.get_text(" ", strip=True))
        if draw_date is None:
            continue

        main_numbers: list[int] = []
        bonus_numbers: list[int] = []
        for image in block.select("div.numere-extrase img"):
            src = image.get("src") or ""
            number = _parse_number_from_src(src)
            if number is None:
                continue
            if "/joker/" in src.lower():
                bonus_numbers.append(number)
            else:
                main_numbers.append(number)

        if len(main_numbers) != spec["numbers_drawn"]:
            continue
        if spec["bonus_count"] != len(bonus_numbers):
            continue

        tier1_winners, tier1_amount, is_rollover, total_pool = _parse_tier_data(block)
        results[draw_date] = RomaniaDrawResult(
            lottery_id=lottery_id,
            draw_date=draw_date,
            numbers=tuple(sorted(main_numbers)),
            bonuses=tuple(sorted(bonus_numbers)),
            tier1_winners=tier1_winners,
            tier1_amount=tier1_amount,
            is_rollover=is_rollover,
            total_pool=total_pool,
        )

    return [results[d] for d in sorted(results)]


def fetch_month_results(
    lottery_id: str,
    year: int,
    month: int,
    session: requests.Session | None = None,
) -> list[RomaniaDrawResult]:
    if requests is None:
        raise RuntimeError("requests is not installed.")
    url = _page_url(lottery_id)
    owned_session = session is None
    session = session or requests.Session()
    if owned_session:
        _request_with_retries(session, "GET", url, headers=REQUEST_HEADERS, timeout=45)
    response = _request_with_retries(
        session,
        "POST",
        url,
        headers={**REQUEST_HEADERS, "Referer": url},
        data={"select-year": str(year), "select-month": str(month), "submitButton": "Cauta"},
        timeout=45,
    )
    results = parse_results_page(lottery_id, response.text)
    if owned_session:
        session.close()
    return [result for result in results if result.draw_date.year == year and result.draw_date.month == month]


def fetch_results_since(
    lottery_id: str,
    start_date: date | None = None,
    end_date: date | None = None,
    *,
    delay_seconds: float = 0.0,
) -> list[RomaniaDrawResult]:
    if requests is None:
        raise RuntimeError("requests is not installed.")

    end_date = end_date or date.today()
    url = _page_url(lottery_id)
    session = requests.Session()
    try:
        landing = _request_with_retries(session, "GET", url, headers=REQUEST_HEADERS, timeout=45)
        years = _extract_available_years(landing.text)
        if not years:
            return []

        results: dict[date, RomaniaDrawResult] = {}
        start_year = start_date.year if start_date else ROMANIA_START_YEARS.get(lottery_id, years[0])
        for year in [value for value in years if start_year <= value <= end_date.year]:
            month_start = start_date.month if start_date and year == start_date.year else 1
            month_end = end_date.month if year == end_date.year else 12
            for month in range(month_start, month_end + 1):
                try:
                    month_results = fetch_month_results(lottery_id, year, month, session=session)
                except Exception as exc:  # pragma: no cover - network behavior
                    print(f"[romania_lotto] {lottery_id} {year}-{month:02d} failed: {exc}")
                    continue
                for result in month_results:
                    if start_date and result.draw_date <= start_date:
                        continue
                    if result.draw_date > end_date:
                        continue
                    results[result.draw_date] = result
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
        return [results[d] for d in sorted(results)]
    finally:
        session.close()
