"""Estimări de premii pe categorii (raportarea biletelor AI, bilanț, optimizator).

Valorile sunt premii FIXE oficiale sau MEDII TIPICE pentru categoriile cu fond
comun (acelea depind de vânzări și de numărul de câștigători). Verificate la
2026-09-21 din surse publice (vezi SOURCES). Testul tests/test_prize_estimates.py
verifică faptul că randamentul așteptat al fiecărei loterii rămâne realist
(5–75% din preț) — un tabel umflat făcea bilanțul să pară pe profit.

Format tier: (main_hits, min_bonus_hits, etichetă, sumă, monedă).
Un tier se potrivește când hits == main_hits ȘI bonus_hits >= min_bonus_hits;
tier-urile cu bonus stau înaintea celor fără bonus pentru același main_hits.
"""

from __future__ import annotations

from math import comb

VERIFIED_ON = "2026-09-21"

# (main_hits, min_bonus_hits, etichetă, sumă, monedă)
TIERS: dict[str, list[tuple[int, int, str, float, str]]] = {
    "loto_6_49_ro": [
        (6, 0, "Jackpot (Categoria I)", 2_500_000, "RON"),
        (5, 0, "Categoria II", 50_000, "RON"),
        (4, 0, "Categoria III", 1_500, "RON"),
        (3, 0, "Categoria IV (fix)", 50, "RON"),
    ],
    # 6 numere extrase; 3 numere NU câștigă. Cat. I = 5 din primele 5 extrase
    # (jackpot), Cat. II = 5 din toate 6 — nu le distingem, folosim o medie.
    "loto_5_40_ro": [
        (5, 0, "Categoria I/II (5 numere)", 100_000, "RON"),
        (4, 0, "Categoria III", 800, "RON"),
    ],
    # 8 categorii; bonus = numărul JOKER (1 din 20). 2 numere fără Joker NU câștigă.
    "joker_ro": [
        (5, 1, "Jackpot (Categoria I, 5+J)", 3_000_000, "RON"),
        (5, 0, "Categoria II (5)", 120_000, "RON"),
        (4, 1, "Categoria III (4+J)", 35_000, "RON"),
        (4, 0, "Categoria IV (4)", 1_100, "RON"),
        (3, 1, "Categoria V (3+J)", 680, "RON"),
        (3, 0, "Categoria VI (3)", 85, "RON"),
        (2, 1, "Categoria VII (2+J)", 50, "RON"),
        (1, 1, "Categoria VIII (1+J)", 30, "RON"),
    ],
    "superenalotto_it": [
        (6, 0, "Jackpot (6 punti)", 3_000_000, "EUR"),
        (5, 1, "5 punti + Jolly", 400_000, "EUR"),
        (5, 0, "5 punti", 60_000, "EUR"),
        (4, 0, "4 punti", 300, "EUR"),
        (3, 0, "3 punti", 30, "EUR"),
        (2, 0, "2 punti", 5, "EUR"),
    ],
    "lotto_pl": [
        (6, 0, "Jackpot (Grupa I)", 2_500_000, "PLN"),
        (5, 0, "Grupa II", 5_000, "PLN"),
        (4, 0, "Grupa III", 150, "PLN"),
        (3, 0, "Grupa IV (fix)", 24, "PLN"),
    ],
    "mini_lotto_pl": [
        (5, 0, "Jackpot (min. 100 000 zł)", 150_000, "PLN"),
        (4, 0, "4 trafienia", 1_500, "PLN"),
        (3, 0, "3 trafienia (fix)", 30, "PLN"),
    ],
    # Multi Multi, bilet de 10 numere, stawka de bază — premii fixe. 5 trafienia
    # NU câștigă; 0 trafienia câștigă.
    "multi_multi_pl": [
        (10, 0, "10/10", 250_000, "PLN"),
        (9, 0, "9/10", 2_500, "PLN"),
        (8, 0, "8/10", 300, "PLN"),
        (7, 0, "7/10", 25, "PLN"),
        (6, 0, "6/10", 2.5, "PLN"),
        (0, 0, "0/10 (zero)", 4, "PLN"),
    ],
    # Kaskada: premii fixe pentru setul de 12 numere afișat.
    "kaskada_pl": [
        (12, 0, "12/12", 250_000, "PLN"),
        (11, 0, "11/12", 1_000, "PLN"),
        (10, 0, "10/12", 25, "PLN"),
        (9, 0, "9/12", 8, "PLN"),
        (8, 0, "8/12", 2, "PLN"),
    ],
}

# Preț pentru o variantă simplă (fără taxa de bilet de 0,50 lei la loteriile RO).
TICKET_PRICES: dict[str, tuple[float, str]] = {
    "loto_6_49_ro": (8.0, "RON"),
    "loto_5_40_ro": (5.0, "RON"),
    "joker_ro": (7.0, "RON"),
    "superenalotto_it": (1.0, "EUR"),
    "lotto_pl": (3.0, "PLN"),
    "mini_lotto_pl": (2.5, "PLN"),
    "multi_multi_pl": (2.5, "PLN"),
    "kaskada_pl": (2.0, "PLN"),
}

SOURCES: dict[str, str] = {
    "loto_6_49_ro": "loto.ro (preț 8 lei; cat. IV fix 50 lei)",
    "loto_5_40_ro": "loto.ro descriere + rezultate 20.09.2026 (cat. III 794,84 lei; 3 numere nu câștigă)",
    "joker_ro": "loto.ro (preț 7 lei) + rezultate 20.09.2026 (IV 1137, V 678, VI 86,5, VII 51,9, VIII 29,9)",
    "mini_lotto_pl": "lotto-wyniki.pl (2,50 zł; 3 trafienia 30 zł fix; 4 ≈ 1500 zł; 5 min. 100 000 zł)",
    "multi_multi_pl": "lotto-wyniki.pl / wynikilotto (10 liczb: 250000/2500/300/25/2,50, 0 trafień 4 zł)",
    "kaskada_pl": "multipasko losowanie 9427 (8=2, 9=8, 10=25, 11=1000, 12=250000 zł); 2 zł / 2 zestawy",
}


def estimate_ticket_prize(lottery_id: str, hits: int, bonus_hits: int = 0) -> dict | None:
    """Premiu pentru numerele înregistrate, fără un set complementar implicit."""
    hits = int(hits)
    tiers = TIERS.get(lottery_id)
    if tiers:
        for main_req, bonus_req, label, amount, currency in tiers:
            if hits == main_req and bonus_hits >= bonus_req:
                return {"category": label, "amount": amount, "currency": currency}
    if hits <= 0:
        return None
    if not tiers:
        return {"category": f"{hits} lovituri", "amount": 0.0, "currency": ""}
    return {"category": f"{hits} lovituri (fără premiu)", "amount": 0.0, "currency": ""}


def ticket_price(lottery_id: str) -> tuple[float, str]:
    """(preț aproximativ, monedă) pentru o variantă simplă."""
    return TICKET_PRICES.get(lottery_id, (0.0, ""))


def winning_hit_counts(lottery_id: str, ticket_size: int) -> set[int]:
    """Numărul de lovituri (fără bonus) pentru care un bilet primește premiu."""
    out = set()
    for hits in range(0, ticket_size + 1):
        prize = estimate_ticket_prize(lottery_id, hits, 0)
        if prize and prize.get("amount"):
            out.add(hits)
    return out


def expected_value(lottery_id: str, config) -> tuple[float, float]:
    """(câștig așteptat per bilet, preț) — exact pe hipergeometrică.

    Bonusul: probabilitate bonus_count / bonus_range (Joker: 1 din 20), aproximare
    suficientă pentru un control de realism.
    """
    N, K, n = config.number_range, config.numbers_drawn, config.ticket_size
    total = comb(N, n)
    p_bonus = 0.0
    if config.bonus_range:
        p_bonus = min(1.0, max(config.bonus_count, 1) / config.bonus_range)
    ev = 0.0
    for h in range(0, n + 1):
        p = comb(K, h) * comb(N - K, n - h) / total
        if p == 0:
            continue
        with_bonus = estimate_ticket_prize(lottery_id, h, 1) or {}
        without = estimate_ticket_prize(lottery_id, h, 0) or {}
        ev += p * (p_bonus * float(with_bonus.get("amount", 0.0))
                   + (1 - p_bonus) * float(without.get("amount", 0.0)))
    return ev, ticket_price(lottery_id)[0]
