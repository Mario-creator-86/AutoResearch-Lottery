"""Reality check — este un „edge" real sau zgomot de selecție?

Optimizatorul testează sute/mii de configurații și păstrează maximul. Chiar pe
extrageri perfect aleatoare, maximul a M configurații depășește baseline-ul cu
~sqrt(2·ln M) erori standard. Modulul acesta pune fiecare metrică în context:

- varianța EXACTĂ a loviturilor unui bilet (hipergeometrică);
- z-score și p-value față de baseline-ul aleator;
- p-value ajustat pentru numărul de experimente (corecție Šidák);
- verdict: „significant" / „suggestive" / „noise".

Folosit de /api/reality-check, de notificările auto_loop, de promovarea
strategiilor scrise de AI și de auto-învățarea agentului de bilete.
"""

from __future__ import annotations

import math
import re

CONFIGS_RE = re.compile(r"\((\d+) configs")

SIGNIFICANT_P = 0.01
SUGGESTIVE_P = 0.10

VERDICT_TEXT = {
    "significant": "semnificativ statistic chiar după corecția pentru numărul de experimente",
    "suggestive": "sugestiv, dar nu concludent — poate fi încă zgomot",
    "noise": "compatibil cu zgomotul (nu se distinge de alegerea aleatoare)",
    "insufficient": "prea puține date pentru o concluzie",
}


def ticket_hits_variance(number_range: int, numbers_drawn: int, ticket_size: int) -> float:
    """Varianța hipergeometrică a numărului de lovituri pentru un bilet."""
    N, K, n = number_range, numbers_drawn, ticket_size
    if N <= 1:
        return 0.0
    return n * (K / N) * ((N - K) / N) * ((N - n) / (N - 1))


def normal_sf(z: float) -> float:
    """P(Z > z) pentru normala standard."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def sidak_adjust(p_value: float, n_tests: int) -> float:
    """p ajustat = 1 − (1 − p)^M, calculat stabil numeric."""
    n_tests = max(int(n_tests), 1)
    p_value = min(max(p_value, 0.0), 1.0)
    if p_value >= 1.0:
        return 1.0
    return float(-math.expm1(n_tests * math.log1p(-p_value)))


def verdict_for(p_adjusted: float | None) -> str:
    if p_adjusted is None:
        return "insufficient"
    if p_adjusted < SIGNIFICANT_P:
        return "significant"
    if p_adjusted < SUGGESTIVE_P:
        return "suggestive"
    return "noise"


def edge_stats(
    *,
    metric: float,
    baseline: float,
    number_range: int,
    numbers_drawn: int,
    ticket_size: int,
    n_draws: int,
    n_experiments: int = 1,
) -> dict:
    """Statistici pentru media loviturilor pe `n_draws` extrageri evaluate."""
    if n_draws <= 0:
        return {"metric": metric, "baseline": baseline, "verdict": "insufficient",
                "verdict_text": VERDICT_TEXT["insufficient"]}
    se = math.sqrt(ticket_hits_variance(number_range, numbers_drawn, ticket_size) / n_draws)
    z = (metric - baseline) / se if se > 0 else 0.0
    p_value = normal_sf(z)
    p_adjusted = sidak_adjust(p_value, n_experiments)
    verdict = verdict_for(p_adjusted)
    return {
        "metric": round(metric, 6),
        "baseline": round(baseline, 6),
        "edge_pct": round((metric / baseline - 1.0) * 100, 3) if baseline else 0.0,
        "n_draws": int(n_draws),
        "n_experiments": int(max(n_experiments, 1)),
        "standard_error": round(se, 6),
        "z": round(z, 3),
        "p_value": round(p_value, 6),
        "p_adjusted": round(p_adjusted, 6),
        # cât de mare ar fi maximul pur întâmplător al atâtor experimente
        "noise_max_z": round(math.sqrt(2.0 * math.log(max(n_experiments, 2))), 3),
        "verdict": verdict,
        "verdict_text": VERDICT_TEXT[verdict],
    }


def config_edge_stats(config, metric: float, n_draws: int, n_experiments: int = 1) -> dict:
    return edge_stats(
        metric=metric, baseline=config.baseline, number_range=config.number_range,
        numbers_drawn=config.numbers_drawn, ticket_size=config.ticket_size,
        n_draws=n_draws, n_experiments=n_experiments,
    )


def count_experiments(results_file) -> int:
    """Numărul total de configurații testate (suma „(N configs" din results.tsv)."""
    total = 0
    try:
        lines = results_file.read_text(encoding="utf-8").splitlines()[1:]
    except OSError:
        return 0
    for line in lines:
        cols = line.split("\t")
        if len(cols) < 3:
            continue
        match = CONFIGS_RE.search(cols[-1]) if cols else None
        total += int(match.group(1)) if match else 1
    return total


def prospective_ticket_stats(lottery_id: str, config=None) -> dict:
    """Recordul REAL (prospectiv) al biletelor AI verificate după extrageri.

    Biletele din aceeași extragere sunt corelate, deci eroarea standard e
    calculată conservator pe numărul de extrageri, nu pe numărul de bilete.
    """
    from ai_tickets_store import load_ticket_store
    from lottery_config import get_lottery_config

    config = config or get_lottery_config(lottery_id)
    checked = [e for e in load_ticket_store(lottery_id).get("entries", []) if e.get("status") == "checked"]
    per_draw: list[float] = []
    per_draw_best: list[int] = []
    sizes: list[int] = []
    for entry in checked:
        results = entry.get("results") or []
        if not results:
            continue
        hits = [int(r.get("hit_count", 0) or 0) for r in results]
        sizes.extend(len(r.get("main") or []) for r in results)
        per_draw.append(sum(hits) / len(hits))
        per_draw_best.append(max(hits))
    if not per_draw:
        return {"draws_checked": 0, "verdict": "insufficient", "verdict_text": VERDICT_TEXT["insufficient"]}
    ticket_size = round(sum(sizes) / len(sizes)) if sizes and all(sizes) else config.ticket_size
    baseline = ticket_size * config.numbers_drawn / config.number_range
    mean_hits = sum(per_draw) / len(per_draw)
    stats = edge_stats(
        metric=mean_hits, baseline=baseline, number_range=config.number_range,
        numbers_drawn=config.numbers_drawn, ticket_size=ticket_size, n_draws=len(per_draw),
    )
    # testăm în ambele direcții: și „mai slab decât aleator" e informație utilă
    stats["p_two_sided"] = round(min(1.0, 2 * normal_sf(abs(stats["z"]))), 6)
    stats.update({
        "draws_checked": len(per_draw),
        "mean_hits_per_ticket": round(mean_hits, 4),
        "mean_best_ticket_hits": round(sum(per_draw_best) / len(per_draw_best), 3),
        "ticket_size": ticket_size,
    })
    if len(per_draw) < 30:
        stats["verdict"] = "insufficient"
        stats["verdict_text"] = (f"doar {len(per_draw)} extrageri verificate — sunt necesare "
                                 f"cel puțin 30 pentru o concluzie")
    return stats


def lottery_reality_check(lottery_id: str) -> dict:
    """Tot contextul statistic pentru o loterie (backtest, walkforward, prospectiv)."""
    import auto_loop
    from ai_tickets_store import load_ticket_store
    from load_data import load_lottery_draws
    from lottery_config import get_lottery_config
    from walkforward import TRAIN_RATIO, load_walkforward_result

    config = get_lottery_config(lottery_id)
    draws = load_lottery_draws(lottery_id)
    n_experiments = count_experiments(config.results_file)
    out: dict = {
        "lottery": lottery_id,
        "name": config.name,
        "draws": len(draws),
        "baseline": round(config.baseline, 6),
        "experiments": n_experiments,
    }

    state, _active = auto_loop.load_search_state_for_lottery(lottery_id)
    if state.best_metric > 0 and draws:
        evaluated = max(len(draws) - int(state.iw), 1)
        out["backtest"] = config_edge_stats(config, state.best_metric, evaluated, max(n_experiments, 1))
        out["backtest"]["strategy"] = state.best_strategy

    wf = load_walkforward_result(config)
    if wf and draws:
        test_draws = max(int(len(draws) * (1 - TRAIN_RATIO)), 1)
        # parametrii au fost selectați dintre toate configurațiile testate pe un istoric
        # care include și ferestrele de test → aceeași corecție ca la backtest
        wf_stats = config_edge_stats(config, float(wf.get("wf_score", 0.0)), test_draws, max(n_experiments, 1))
        wf_stats["caveat"] = ("parametrii au fost aleși privind TOATE extragerile (inclusiv perioada de "
                              "test) — nu este o validare curată out-of-sample")
        wf_stats["timestamp"] = wf.get("timestamp")
        out["walkforward"] = wf_stats

    try:
        import forward_test

        out["forward_test"] = forward_test.summary(lottery_id, draws=draws)
    except Exception as exc:
        out["forward_test"] = {"error": str(exc)[:200]}

    out["ai_tickets_prospective"] = prospective_ticket_stats(lottery_id, config)
    try:
        import bias_scan
        import play_advisor

        # scanarea salvată de ciclul de descoperire; la prima rulare o calculăm pe loc
        out["anomaly_scan"] = bias_scan.load_scan(lottery_id) or bias_scan.scan_lottery(lottery_id, draws=draws)
        out["play_advice"] = play_advisor.advise(lottery_id)
    except Exception as exc:
        out["anomaly_scan"] = {"verdict": "insufficient", "error": str(exc)[:200]}
    latest = [e for e in load_ticket_store(lottery_id).get("entries", []) if e.get("portfolio")]
    if latest:
        out["latest_portfolio"] = {"draw_date": latest[-1].get("draw_date"), **latest[-1]["portfolio"]}
    return out


def reality_summary_text(check: dict) -> str:
    """Rezumat de 1-3 propoziții (pentru prompt-urile AI și Telegram)."""
    parts = []
    bt = check.get("backtest")
    if bt:
        parts.append(
            f"Cel mai bun backtest {bt['metric']:.4f} vs aleator {bt['baseline']:.4f} "
            f"({bt['edge_pct']:+.2f}%, z={bt['z']:+.2f}, după {bt['n_experiments']} experimente "
            f"p_ajustat={bt['p_adjusted']:.3f}) → {bt['verdict_text']}."
        )
    ft = check.get("forward_test") or {}
    if ft.get("n_draws"):
        parts.append(
            f"Forward test (parametri înghețați la {ft.get('frozen_after')}): {ft['metric']:.4f} vs aleator "
            f"{ft['baseline']:.4f} pe {ft['n_draws']} extrageri noi (z={ft['z']:+.2f}) → {ft['verdict_text']}."
        )
    pr = check.get("ai_tickets_prospective") or {}
    if pr.get("draws_checked"):
        parts.append(
            f"Biletele AI reale: {pr['mean_hits_per_ticket']:.3f} lovituri/bilet vs aleator "
            f"{pr['baseline']:.3f} pe {pr['draws_checked']} extrageri → {pr['verdict_text']}."
        )
    scan = check.get("anomaly_scan") or {}
    if scan.get("tests"):
        parts.append(
            f"Scanare anomalii pe {scan['n_draws']} extrageri (7 teste exacte, cel mai puternic "
            f"«{scan['strongest_test']}», p_ajustat={scan['p_adjusted']:.3f}) → {scan['verdict_text']}."
        )
    advice = check.get("play_advice") or {}
    if advice.get("verdict_text"):
        parts.append(f"Valoare așteptată: {advice['verdict_text']}.")
    return " ".join(parts)
