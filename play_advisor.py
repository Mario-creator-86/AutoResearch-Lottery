"""Consilier de joc — valoarea așteptată a unui bilet în funcție de report.

Șansa de a nimeri nu poate fi crescută, dar valoarea unui bilet NU e constantă:
la loteriile cu report, premiul cel mare crește de la o extragere la alta, deci
aceeași șansă plătește mai mult. Singurele decizii care schimbă cu adevărat
rezultatul sunt CÂND joci (reportul) și CE variante alegi (numere nepopulare,
ca să nu împarți premiul — vezi portfolio_optimizer).

Pentru fiecare loterie:
  - randamentul tipic (cât se întoarce în medie dintr-un leu jucat);
  - reportul de prag: cât ar trebui să fie premiul cel mare ca biletul să aibă
    valoare așteptată ≥ prețul (ignorând împărțirea premiului și impozitul, deci
    pragul REAL e mai sus);
  - randamentul la reportul curent, dacă data/{id}/jackpot_history.csv e la zi.

Probabilitățile sunt exacte (hipergeometrice); premiile vin din prize_estimates.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from math import comb

import prize_estimates as pe

JACKPOT_MAX_AGE_DAYS = 10

# Loto 5/40: se extrag 6 numere, dar reportul (Cat. I) cere toate cele 5 numere
# jucate în PRIMELE 5 extrase — 1/6 din cazurile cu 5 nimerite; restul sunt Cat. II.
# prize_estimates ține o singură medie (100 000 lei) pentru „5 numere"; aici o
# despărțim: 1/6 × ~500 000 (reportul median 2024–2026) + 5/6 × 20 000 = 100 000.
SPLIT_TOP_TIER = {
    "loto_5_40_ro": {"top_fraction": 1 / 6, "typical_top": 500_000.0, "second_amount": 20_000.0},
}


def tier_probabilities(config) -> dict[tuple[int, int], float]:
    """P(h numere nimerite, bonus nimerit=1/0) pentru un bilet simplu."""
    N, K, n = config.number_range, config.numbers_drawn, config.ticket_size
    total = comb(N, n)
    p_bonus = 0.0
    if config.bonus_range:
        p_bonus = min(1.0, max(config.bonus_count, 1) / config.bonus_range)
    out: dict[tuple[int, int], float] = {}
    for h in range(0, n + 1):
        p = comb(K, h) * comb(N - K, n - h) / total
        if p:
            out[(h, 1)] = p * p_bonus
            out[(h, 0)] = p * (1 - p_bonus)
    return out


def _jackpot_tier(lottery_id: str) -> tuple[int, int] | None:
    tiers = pe.TIERS.get(lottery_id)
    return (tiers[0][0], tiers[0][1]) if tiers else None


def ev_breakdown(lottery_id: str, config) -> tuple[float, float]:
    """(valoarea așteptată a premiilor fără cel mare, probabilitatea premiului cel mare)."""
    top = _jackpot_tier(lottery_id)
    rest, p_top = 0.0, 0.0
    for (hits, bonus), p in tier_probabilities(config).items():
        prize = pe.estimate_ticket_prize(lottery_id, hits, bonus) or {}
        tiers = pe.TIERS.get(lottery_id) or []
        is_top = bool(top and tiers and prize.get("category") == tiers[0][2])
        if is_top:
            p_top += p
        else:
            rest += p * float(prize.get("amount", 0.0))
    split = SPLIT_TOP_TIER.get(lottery_id)
    if split:
        rest += p_top * (1 - split["top_fraction"]) * split["second_amount"]
        p_top *= split["top_fraction"]
    return rest, p_top


def latest_jackpot(config, today: date | None = None) -> dict | None:
    """Ultimul report cunoscut, dacă e recent și premiul nu a fost câștigat."""
    path = getattr(config, "jackpot_csv", None)
    if path is None or not path.exists():
        return None
    last = None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                last = row
    except OSError:
        return None
    if not last:
        return None
    try:
        draw_date = date.fromisoformat(last["draw_date"])
        amount = float(last["tier1_amount"])
    except (KeyError, ValueError):
        return None
    if (today or date.today()) - draw_date > timedelta(days=JACKPOT_MAX_AGE_DAYS):
        return None
    if str(last.get("is_rollover", "")).lower() != "true" or amount <= 0:
        return None  # câștigat la ultima extragere → fondul repornește, valoare necunoscută
    return {"draw_date": draw_date.isoformat(), "amount": amount}


def advise(lottery_id: str, today: date | None = None) -> dict | None:
    from lottery_config import get_lottery_config

    if lottery_id not in pe.TIERS:
        return None
    config = get_lottery_config(lottery_id)
    price, currency = pe.ticket_price(lottery_id)
    if price <= 0:
        return None
    typical_ev, _ = pe.expected_value(lottery_id, config)
    rest, p_top = ev_breakdown(lottery_id, config)
    typical_top = float(SPLIT_TOP_TIER.get(lottery_id, {}).get("typical_top", pe.TIERS[lottery_id][0][3]))
    # premii fixe (Multi Multi, Kaskada): premiul cel mare nu crește → randament constant
    has_rollover = getattr(config, "jackpot_csv", None) is not None and lottery_id not in {
        "multi_multi_pl", "kaskada_pl"}
    out = {
        "lottery": lottery_id,
        "name": config.name,
        "currency": currency,
        "ticket_price": price,
        "typical_return_pct": round(typical_ev / price * 100, 1),
        "jackpot_odds": round(1 / p_top) if p_top else None,
        "typical_jackpot": typical_top,
        "has_rollover": has_rollover,
        "prizes_verified_on": pe.VERIFIED_ON,
    }
    if not has_rollover or not p_top:
        out["verdict"] = "fixed"
        out["verdict_text"] = (f"premii fixe: din fiecare {price:g} {currency} jucați se întorc în medie "
                               f"{typical_ev:.2f} — nu există un moment mai bun de joc")
        return out

    breakeven = (price - rest) / p_top
    out["breakeven_jackpot"] = round(breakeven)
    jackpot = latest_jackpot(config, today)
    if jackpot is None:
        out["verdict"] = "unknown"
        out["verdict_text"] = (f"report curent necunoscut; biletul și-ar acoperi prețul doar la un premiu "
                               f"cel mare de peste {breakeven:,.0f} {currency}")
        return out
    ev_now = rest + p_top * jackpot["amount"]
    ratio = ev_now / price
    out.update({
        "current_jackpot": round(jackpot["amount"]),
        "jackpot_as_of": jackpot["draw_date"],
        "current_return_pct": round(ratio * 100, 1),
        "jackpot_vs_breakeven_pct": round(jackpot["amount"] / breakeven * 100, 1),
    })
    # pragul e optimist (fără impozit și fără împărțirea premiului) → cerem o marjă
    if ratio >= 1.25:
        out["verdict"] = "favorable"
    elif ratio >= 0.8:
        out["verdict"] = "near"
    else:
        out["verdict"] = "unfavorable"
    out["verdict_text"] = {
        "favorable": "reportul depășește pragul — moment rar în care biletul are valoare așteptată pozitivă "
                     "(înainte de impozit și de împărțirea premiului)",
        "near": "reportul e aproape de prag — cel mai bun raport din ciclu, dar încă sub cost",
        "unfavorable": "reportul e mult sub prag — în medie biletul pierde",
    }[out["verdict"]] + f" (se întorc ~{ratio * 100:.0f}% din preț; prag {breakeven:,.0f} {currency})"
    return out


def advise_all(today: date | None = None) -> list[dict]:
    out = []
    for lottery_id in pe.TIERS:
        try:
            item = advise(lottery_id, today)
        except Exception as exc:
            item = {"lottery": lottery_id, "verdict": "unknown", "error": str(exc)[:200]}
        if item:
            out.append(item)
    return out


if __name__ == "__main__":
    for item in advise_all():
        print(f"{item.get('name', item['lottery'])}: {item.get('verdict_text', item.get('error'))}")
