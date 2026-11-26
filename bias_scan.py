"""Detector de anomalii reale în extrageri — singura cale cinstită spre un avantaj.

auto_loop caută strategii care „ar fi mers" pe istoric; cu sute de mii de
configurații testate, cea mai bună arată mereu bine din pură selecție. Întrebarea
corectă e alta: sunt extragerile în sine diferite de hazard? (bile/aparate cu
defect fizic, dependență între extrageri). Dacă da, există un avantaj real și
măsurabil; dacă nu, nicio formulă nu poate exista.

Fiecare test are distribuția nulă exactă (extragere fără întoarcere a K din N),
deci nu are parametri de ajustat și nu poate fi „antrenat" pe zgomot:

  frequency_all      χ² pe frecvențele numerelor, tot istoricul
  frequency_recent   același test pe extragerile recente (aparatele se schimbă)
  persistence        abaterile din prima jumătate se regăsesc în a doua?
                     (un defect fizic stabil → corelație pozitivă; test out-of-sample)
  repeats            numere repetate între extrageri consecutive
  sum                media sumei numerelor extrase
  parity             distribuția numărului de numere impare pe extragere
  hottest_number     cea mai mare abatere individuală (Šidák pe N numere)

Verdictul unei loterii folosește cel mai mic p, corectat Šidák pentru toate
testele rulate (7 × numărul de loterii scanate). Validat în tests/test_bias_scan.py:
pe extrageri uniforme nu dă alarme, pe extrageri cu bile „grele" detectează.

Stocare: data/{id}/bias_scan.json. Rulat automat după actualizarea istoricului.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
from scipy import stats

from atomic_io import read_json_with_recovery, write_json_atomic
from significance import VERDICT_TEXT, normal_sf, sidak_adjust, verdict_for

SCAN_FILE = "bias_scan.json"
MIN_DRAWS = 300
RECENT_MIN, RECENT_MAX = 300, 1500
TESTS = ("frequency_all", "frequency_recent", "persistence", "repeats", "sum", "parity", "hottest_number")


def draws_matrix(draws, number_range: int, numbers_drawn: int) -> np.ndarray:
    """Matrice 0/1 (extrageri × numere); ignoră extragerile cu alt format (reguli vechi)."""
    rows = [d.numbers for d in draws
            if len(d.numbers) == numbers_drawn and len(set(d.numbers)) == numbers_drawn
            and all(1 <= n <= number_range for n in d.numbers)]
    matrix = np.zeros((len(rows), number_range), dtype=np.int8)
    for i, numbers in enumerate(rows):
        matrix[i, np.asarray(numbers) - 1] = 1
    return matrix


def _two_sided(z: float) -> float:
    return min(1.0, 2.0 * normal_sf(abs(z)))


def frequency_test(matrix: np.ndarray) -> dict:
    n, N = matrix.shape
    K = int(matrix[0].sum())
    counts = matrix.sum(axis=0)
    expected = n * K / N
    # fără întoarcere, frecvențele sunt corelate negativ: factorul (N-1)/(N-K)
    # readuce statistica la χ²(N-1)
    chi2 = float(((counts - expected) ** 2).sum() / expected * (N - 1) / (N - K))
    return {"statistic": round(chi2, 3), "df": N - 1, "n_draws": int(n),
            "p_value": float(stats.chi2.sf(chi2, N - 1))}


def number_z_scores(matrix: np.ndarray) -> np.ndarray:
    n, N = matrix.shape
    p = matrix[0].sum() / N
    return (matrix.sum(axis=0) - n * p) / math.sqrt(n * p * (1 - p))


def hottest_number_test(matrix: np.ndarray) -> dict:
    z = number_z_scores(matrix)
    idx = int(np.argmax(np.abs(z)))
    return {"number": idx + 1, "z": round(float(z[idx]), 3),
            "p_value": sidak_adjust(_two_sided(float(z[idx])), matrix.shape[1])}


def persistence_test(matrix: np.ndarray) -> dict:
    """Corelația abaterilor de frecvență între cele două jumătăți ale istoricului."""
    half = matrix.shape[0] // 2
    first, second = matrix[:half].sum(axis=0), matrix[half:].sum(axis=0)
    if first.std() == 0 or second.std() == 0:
        return {"r": 0.0, "p_value": 1.0}
    r = float(np.corrcoef(first, second)[0, 1])
    N = matrix.shape[1]
    t = r * math.sqrt((N - 2) / max(1e-12, 1 - r * r))
    # unilateral: doar corelația pozitivă înseamnă un defect stabil, exploatabil
    return {"r": round(r, 4), "p_value": float(stats.t.sf(t, N - 2))}


def repeats_test(matrix: np.ndarray) -> dict:
    n, N = matrix.shape
    K = int(matrix[0].sum())
    overlaps = (matrix[1:] & matrix[:-1]).sum(axis=1)
    mean, var = K * K / N, K * (K / N) * (1 - K / N) * (N - K) / (N - 1)
    z = (float(overlaps.mean()) - mean) / math.sqrt(var / len(overlaps))
    return {"mean_repeats": round(float(overlaps.mean()), 4), "expected": round(mean, 4),
            "z": round(z, 3), "p_value": _two_sided(z)}


def sum_test(matrix: np.ndarray) -> dict:
    n, N = matrix.shape
    K = int(matrix[0].sum())
    sums = matrix @ np.arange(1, N + 1)
    mean, var = K * (N + 1) / 2, K * (N + 1) * (N - K) / 12
    z = (float(sums.mean()) - mean) / math.sqrt(var / n)
    return {"mean_sum": round(float(sums.mean()), 3), "expected": round(mean, 3),
            "z": round(z, 3), "p_value": _two_sided(z)}


def parity_test(matrix: np.ndarray) -> dict:
    n, N = matrix.shape
    K = int(matrix[0].sum())
    odd_total = (N + 1) // 2
    odd_counts = matrix[:, 0::2].sum(axis=1)  # coloanele 0,2,4… = numerele 1,3,5…
    observed = np.bincount(odd_counts, minlength=K + 1)[:K + 1].astype(float)
    expected = stats.hypergeom.pmf(np.arange(K + 1), N, odd_total, K) * n
    # comasează categoriile rare (așteptat < 5) ca χ² să rămână valid
    obs, exp, acc_o, acc_e = [], [], 0.0, 0.0
    for o, e in zip(observed, expected):
        acc_o, acc_e = acc_o + o, acc_e + e
        if acc_e >= 5:
            obs.append(acc_o)
            exp.append(acc_e)
            acc_o = acc_e = 0.0
    if acc_e and obs:
        obs[-1] += acc_o
        exp[-1] += acc_e
    if len(obs) < 2:
        return {"p_value": 1.0}
    chi2 = float(sum((o - e) ** 2 / e for o, e in zip(obs, exp)))
    return {"statistic": round(chi2, 3), "df": len(obs) - 1,
            "p_value": float(stats.chi2.sf(chi2, len(obs) - 1))}


def scan_matrix(matrix: np.ndarray, n_lotteries: int = 1) -> dict:
    """Rulează toate testele pe o matrice de extrageri și dă verdictul corectat."""
    n, N = matrix.shape
    if n < MIN_DRAWS:
        return {"n_draws": int(n), "verdict": "insufficient", "verdict_text": VERDICT_TEXT["insufficient"],
                "tests": {}}
    recent = matrix[-max(RECENT_MIN, min(RECENT_MAX, n // 4)):]
    tests = {
        "frequency_all": frequency_test(matrix),
        "frequency_recent": frequency_test(recent),
        "persistence": persistence_test(matrix),
        "repeats": repeats_test(matrix),
        "sum": sum_test(matrix),
        "parity": parity_test(matrix),
        "hottest_number": hottest_number_test(matrix),
    }
    family = len(tests) * max(int(n_lotteries), 1)
    for result in tests.values():
        result["p_value"] = round(min(max(result["p_value"], 0.0), 1.0), 8)
        result["p_adjusted"] = round(sidak_adjust(result["p_value"], family), 8)
        result["verdict"] = verdict_for(result["p_adjusted"])
    strongest = min(tests, key=lambda name: tests[name]["p_value"])
    verdict = tests[strongest]["verdict"]
    z = number_z_scores(matrix)
    order = np.argsort(-z)
    return {
        "n_draws": int(n),
        "tests_in_family": family,
        "strongest_test": strongest,
        "p_adjusted": tests[strongest]["p_adjusted"],
        "verdict": verdict,
        "verdict_text": VERDICT_TEXT[verdict],
        "tests": tests,
        # informative; devin o „formulă" doar dacă verdictul e significant ȘI
        # testul de persistență confirmă că abaterile sunt stabile în timp
        "most_frequent": [{"number": int(i + 1), "z": round(float(z[i]), 2)} for i in order[:5]],
        "least_frequent": [{"number": int(i + 1), "z": round(float(z[i]), 2)} for i in order[::-1][:5]],
        "exploitable": bool(verdict == "significant" and tests["persistence"]["verdict"] != "noise"),
    }


def scan_lottery(lottery_id: str, draws=None, n_lotteries: int = 1) -> dict:
    from load_data import load_lottery_draws
    from lottery_config import get_lottery_config

    config = get_lottery_config(lottery_id)
    draws = list(draws) if draws is not None else load_lottery_draws(lottery_id)
    matrix = draws_matrix(draws, config.number_range, config.numbers_drawn)
    out = scan_matrix(matrix, n_lotteries=n_lotteries)
    if len(matrix):
        # Cât ar valora abaterile dacă ar fi reale: lovituri medii jucând cele mai
        # frecvente numere, față de hazard. Estimare OPTIMISTĂ (numerele sunt alese
        # tocmai pentru că au ieșit des) — plafonul, nu câștigul așteptat.
        frequencies = np.sort(matrix.mean(axis=0))[::-1]
        out["face_value_edge_pct"] = round(
            (float(frequencies[:config.ticket_size].sum()) / config.baseline - 1.0) * 100, 2)
    out.update({
        "lottery": lottery_id,
        "name": config.name,
        "last_draw": draws[-1].draw_date.isoformat() if draws else None,
        "scanned_at": datetime.now().isoformat(timespec="seconds"),
    })
    return out


def load_scan(lottery_id: str) -> dict | None:
    from lottery_config import get_lottery_config

    data, _ = read_json_with_recovery(get_lottery_config(lottery_id).state_dir / SCAN_FILE)
    return data if isinstance(data, dict) else None


def run_all(lottery_ids=None, save: bool = True) -> dict:
    """Scanează toate loteriile cu istoric. Returnează {results, newly_significant}."""
    from lottery_config import get_lottery_config, list_lottery_configs

    ids = list(lottery_ids) if lottery_ids else [c.id for c in list_lottery_configs() if c.draws_file.exists()]
    results, newly_significant = [], []
    for lottery_id in ids:
        try:
            previous = load_scan(lottery_id) if save else None
            result = scan_lottery(lottery_id, n_lotteries=len(ids))
        except Exception as exc:
            results.append({"lottery": lottery_id, "verdict": "insufficient", "error": str(exc)[:200]})
            continue
        results.append(result)
        if result["verdict"] == "significant" and (previous or {}).get("verdict") != "significant":
            newly_significant.append(result)
        if save:
            state_dir = get_lottery_config(lottery_id).state_dir
            state_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(state_dir / SCAN_FILE, result, ensure_ascii=False)
    return {"results": results, "newly_significant": newly_significant}


def summary_text(result: dict) -> str:
    """O propoziție pentru Telegram / prompt-urile AI."""
    name = result.get("name") or result.get("lottery")
    if result.get("verdict") in (None, "insufficient"):
        return f"{name}: prea puține extrageri pentru scanarea de anomalii."
    test = result["tests"][result["strongest_test"]]
    text = (f"{name}: {result['n_draws']} extrageri, cel mai puternic semnal «{result['strongest_test']}» "
            f"p ajustat = {test['p_adjusted']:.4f} → {result['verdict_text']}.")
    if result.get("exploitable"):
        hot = ", ".join(str(item["number"]) for item in result["most_frequent"])
        text += f" Abatere stabilă în timp; numere supra-reprezentate: {hot}."
    return text


if __name__ == "__main__":
    for item in run_all(save=False)["results"]:
        print(summary_text(item) if "tests" in item else item)
