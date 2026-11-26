"""Ollama AI integration: LLM-powered lottery analysis and number proposals.

Uses the local Ollama server (http://localhost:11434) which can proxy to
Ollama Cloud models (e.g. "gpt-oss:120b-cloud", "glm-5.3-flash:cloud",
"deepseek-v4-pro:cloud"). The AI receives real statistics computed from the
downloaded draw history plus the current best auto-loop strategy, and returns:
  1. insights about number patterns / strategy behavior
  2. a concrete ranking of the most promising numbers for the next draw
  3. optional strategy parameter suggestions for the auto-loop to test

All model output is advisory: the deterministic backtest loop keeps authority
over what is actually used (KEEP/DISCARD stays metric-driven).

Usage:
    python ollama_ai.py                      # analyze all lotteries
    python ollama_ai.py loto_6_49_ro         # analyze one lottery
    python ollama_ai.py --status             # show Ollama/model availability
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, deque
from datetime import date, datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from decision_ledger import append_entry
from lottery_config import LotteryConfig, get_lottery_config, list_lottery_configs

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
# Preferred models, in order. Cloud models give the best reasoning; the first
# one that responds wins. Override with OLLAMA_MODEL / OLLAMA_MODELS env var.
PREFERRED_MODELS = [
    name.strip()
    for name in os.environ.get(
        "OLLAMA_MODELS",
        os.environ.get("OLLAMA_MODEL", "glm-5.3-flash:cloud,gpt-oss:120b-cloud,deepseek-v4-flash:cloud,gpt-oss:20b-cloud"),
    ).split(",")
    if name.strip()
]
REQUEST_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "600"))
# Per-attempt chat timeout — a hung cloud model should not stall the sweep;
# on timeout the caller falls back to the next preferred model.
CHAT_TIMEOUT = int(os.environ.get("OLLAMA_CHAT_TIMEOUT", "240"))
MAX_MODEL_ATTEMPTS = 2
MAX_INSIGHTS_LOG = 200

PREDICTIONS_VERSION = 1


# ─── Low-level client ────────────────────────────────────────────────────────


def _post(path: str, payload: dict, timeout: int = REQUEST_TIMEOUT) -> dict:
    url = f"{OLLAMA_HOST}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ollama_available() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=5) as resp:
            return bool(json.loads(resp.read().decode("utf-8")).get("models"))
    except (urllib.error.URLError, OSError, ValueError):
        return False


def list_models() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=5) as resp:
            return [m["name"] for m in json.loads(resp.read().decode("utf-8")).get("models", [])]
    except (urllib.error.URLError, OSError, ValueError):
        return []


def resolve_model(preferred: list[str] | None = None) -> str | None:
    """Return the first preferred model that is actually available."""
    available = list_models()
    if not available:
        return None
    for name in (preferred or PREFERRED_MODELS):
        if name in available:
            return name
    # Any cloud model as fallback, then any local model
    for name in available:
        if name.endswith("-cloud"):
            return name
    return available[0] if available else None


def chat(prompt: str, model: str, system: str | None = None, timeout: int | None = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = _post(
        "/api/chat",
        {"model": model, "messages": messages, "stream": False},
        timeout=timeout or CHAT_TIMEOUT,
    )
    return str(response.get("message", {}).get("content", "")).strip()


def model_candidates(model: str | None = None) -> list[str]:
    """Ordered model list to try: the requested one, then preferred fallbacks."""
    if model:
        return [model]
    available = list_models()
    candidates: list[str] = []
    resolved = resolve_model()
    if resolved:
        candidates.append(resolved)
    for name in PREFERRED_MODELS:
        if name in available and name not in candidates:
            candidates.append(name)
    for name in available:
        if name not in candidates:
            candidates.append(name)
    return candidates[:MAX_MODEL_ATTEMPTS]


def extract_json(text: str) -> dict | None:
    """Best-effort JSON extraction from an LLM answer."""
    text = text.strip()
    # Strip markdown fences if present
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


# ─── Self-improvement: AI-editable prompt overrides ─────────────────────────

SELF_DIR = PROJECT_DIR / "data" / "ai_self"
PROMPT_OVERRIDES_FILE = SELF_DIR / "prompt_overrides.json"


def load_prompt_overrides() -> dict:
    """Instrucțiuni suplimentare pe care AI-ul și le poate scrie singur.

    Fișier: data/ai_self/prompt_overrides.json cu cheile opționale:
      - "analysis_extra": text adăugat la system prompt-ul de analiză
      - "coder_extra": text adăugat la system prompt-ul de generare strategii
      - "ticket_extra": text adăugat la system prompt-ul de generare bilete
    """
    try:
        data = json.loads(PROMPT_OVERRIDES_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_prompt_overrides(overrides: dict) -> bool:
    """Salvează override-uri de prompt (folosit de agentul AI pentru auto-îmbunătățire)."""
    try:
        SELF_DIR.mkdir(parents=True, exist_ok=True)
        PROMPT_OVERRIDES_FILE.write_text(
            json.dumps(overrides, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return True
    except OSError:
        return False


# ─── Web research ────────────────────────────────────────────────────────────

DEFAULT_RESEARCH_URLS = [
    "https://en.wikipedia.org/wiki/Lottery_mathematics",
    "https://en.wikipedia.org/wiki/Gambler%27s_fallacy",
]

MAX_RESEARCH_URLS_PER_RUN = 3
MAX_RESEARCH_CHARS_PER_PAGE = 4000


def fetch_url_text(url: str, max_chars: int = MAX_RESEARCH_CHARS_PER_PAGE) -> str | None:
    """Descarcă o pagină publică (read-only GET) și extrage textul esențial."""
    import re as _re
    import urllib.request

    if not url.lower().startswith(("http://", "https://")):
        return None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 AutoResearch/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read(400_000).decode("utf-8", errors="ignore")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    # Elimină scripturi/stiluri/tag-uri și comprimă spațiile
    text = _re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", raw)
    text = _re.sub(r"(?s)<[^>]+>", " ", text)
    text = _re.sub(r"\s+", " ", text)
    return text[:max_chars] if text.strip() else None


def build_research_block(requested_urls: list[str] | None = None) -> tuple[str, list[str]]:
    """Strânge note de cercetare de pe web pentru prompt-ul AI-ului.

    Returnează (bloc_text, urluri_folosite). Combină paginile implicite cu
    cele cerute explicit, limitat la MAX_RESEARCH_URLS_PER_RUN pagini.
    """
    urls: list[str] = []
    for url in DEFAULT_RESEARCH_URLS:
        urls.append(url)
    for url in (requested_urls or [])[:MAX_RESEARCH_URLS_PER_RUN]:
        if url not in urls:
            urls.append(url)
    urls = urls[: MAX_RESEARCH_URLS_PER_RUN + len(DEFAULT_RESEARCH_URLS)]

    blocks: list[str] = []
    used: list[str] = []
    for url in urls:
        text = fetch_url_text(url)
        if not text:
            continue
        used.append(url)
        blocks.append(f"[sursă: {url}]\n{text}")
        if len(used) >= MAX_RESEARCH_URLS_PER_RUN + 1:
            break
    if not blocks:
        return "(cercetarea web nu a fost disponibilă la această rulare — bazează-te doar pe statistici)", []
    return "\n\n".join(blocks), used


# ─── Statistics payload ──────────────────────────────────────────────────────


def compute_lottery_stats(config: LotteryConfig, draws, best: dict | None = None) -> dict:
    """Compute a compact stats payload describing the lottery for the LLM."""
    number_range = config.number_range
    counts: Counter = Counter()
    recent_counts: Counter = Counter()
    last_seen: dict[int, int] = {}
    for idx, draw in enumerate(draws):
        for n in draw.numbers:
            counts[n] += 1
            last_seen[n] = idx
    recent_window = draws[-min(50, len(draws)):]
    for idx, draw in enumerate(recent_window):
        for n in draw.numbers:
            recent_counts[n] += 1

    total = len(draws)
    expected_per_number = total * config.numbers_drawn / number_range
    gaps = {n: (total - 1 - last_seen.get(n, -1)) for n in range(1, number_range + 1)}
    pairs = Counter()
    for draw in draws[-300:]:
        nums = sorted(draw.numbers)
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                pairs[(nums[i], nums[j])] += 1
    top_pairs = [
        {"a": a, "b": b, "count": c}
        for (a, b), c in pairs.most_common(8)
    ]

    all_counts = [counts[n] for n in range(1, number_range + 1)]
    hottest = sorted(range(1, number_range + 1), key=lambda n: -counts[n])[:10]
    coldest = sorted(range(1, number_range + 1), key=lambda n: counts[n])[:10]
    most_overdue = sorted(range(1, number_range + 1), key=lambda n: -gaps[n])[:10]

    def dev(n: int) -> float:
        return (counts[n] - expected_per_number) / max(1.0, expected_per_number ** 0.5)

    payload = {
        "lottery": config.id,
        "name": config.name,
        "format": f"{config.numbers_drawn}/{number_range}",
        "total_draws": total,
        "history": f"{draws[0].draw_date.isoformat()} .. {draws[-1].draw_date.isoformat()}",
        "expected_hits_per_ticket": round(config.baseline, 4),
        "best_strategy": best or {},
        "frequency": {
            "hottest": [{"n": n, "count": counts[n], "z": round(dev(n), 2)} for n in hottest],
            "coldest": [{"n": n, "count": counts[n], "z": round(dev(n), 2)} for n in coldest],
            "mean": round(sum(all_counts) / len(all_counts), 2),
            "std": round(
                (sum((c - sum(all_counts) / len(all_counts)) ** 2 for c in all_counts) / len(all_counts)) ** 0.5, 2
            ),
        },
        "overdue": [{"n": n, "draws_since": gaps[n]} for n in most_overdue],
        "recent_50_hot": [{"n": n, "count": recent_counts[n]} for n, _ in recent_counts.most_common(8)],
        "top_pairs_recent": top_pairs,
        "last_5_draws": [sorted(d.numbers) for d in draws[-5:]],
    }
    # verdictul statistic (edge vs zgomot) — ca modelul să nu „vadă" tipare în zgomot
    try:
        import significance

        payload["reality_check"] = significance.reality_summary_text(
            significance.lottery_reality_check(config.id)
        )
    except Exception:
        pass
    return payload


# ─── Prompts ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Ești un analist cantitatativ specializat în loterii. Primești statistici reale "
    "calculate din istoricul complet al unei loterii. Rolul tau: (1) investighezi "
    "deviatiile statistice, frecventele, intarzierile si comportamentul strategiei "
    "curente; (2) propui un clasament concret al numerelor pentru urmatoarea extragere; "
    "(3) sugerezi ajustari de parametri pentru strategie. Loteriile sunt in esenta "
    "aleatoare — trateaza orice deviatie ca ipoteza de testat, nu ca certitudine, si "
    "spune mereu cand un pattern e probabil zgomot. Campul reality_check din statistici "
    "contine verdictul statistic (z-score, p ajustat pentru numarul de experimente) al "
    "strategiei si al biletelor tale reale — nu il contrazice. Ce se poate imbunatati real: "
    "acoperirea biletelor (suprapunere mica) si evitarea combinatiilor populare. "
    "Raspunzi EXCLUSIV cu JSON valid."
)

PROMPT_TEMPLATE = """Analizează această loterie: {name} ({format}), {total_draws} extrageri ({history}).

STATISTICI REALE:
{stats_json}

BEST STRATEGIE CURENTĂ (din backtest determinist): {best_json}

Răspunde DOAR cu JSON în acest format exact:
{{
  "analysis": "3-5 propoziții despre ce vezi în date (deviații frecvență, z-scores, întârzieri, pereți frecvente, dacă strategia curentă e bien ajustată)",
  "noise_assessment": "care dintre pattern-urile observate sunt cel mai probabil zgomot statistic și de ce",
  "recommended_numbers": [cel mai promițător număr, ..., exact {range_total} numere ordonate de la cel mai bun la cel mai slab, toate întregi între 1 și {range_max}],
  "confidence": "low|medium|high",
  "strategy_suggestions": [
    {{"param": "delay|momentum|sb|jc|bf|dd|rw|iw", "current": <valoarea curentă>, "suggested": <valoarea propusă>, "reason": ".argument scurt"}}
  ],
  "next_draw_focus": "1-2 propoziții ce strategie de combinare recomanzi pentru următoarea extragere (echilibru hot/overdue/perechi)",
  "proposed_actions": [{{"type": "restart_loop|stop_loop|start_loop|apply_strategy_params|run_walkforward|update_history|run_ai_analysis|set_agent_config", "params": {{{{"lottery": "{lottery_id}", ...}}}}, "why": "motiv scurt"}}]
}}

Reguli:
- recommended_numbers: EXACT {range_total} numere distincte, ordonate after valoare prezisă.
- strategy_suggestions poate fi listă vidă dacă strategia e bine ajustată.
- proposed_actions (opțional, max 2): dacă ai o propunere CONCRETĂ de acțiune pentru sistem,
  folosește un tip din lista permisă: restart_loop, stop_loop, start_loop, apply_strategy_params,
  run_walkforward, update_history, run_ai_analysis, set_agent_config.
- Fii cantitativ: folosește z-scores și dimensiunea eșantionului."""


def build_prompt(stats: dict) -> str:
    range_max = int(stats["format"].split("/")[1])
    best = stats.get("best_strategy") or {}
    best_summary = (
        f"metric={best.get('metric', 'n/a')} ({best.get('name', 'n/a')}), "
        f"weights tf={best.get('tf')} hc={best.get('hc')} bal={best.get('b')} "
        f"sb={best.get('sb')} delay={best.get('delay')} momentum={best.get('momentum')}, "
        f"windows iw={best.get('iw')} rw={best.get('rw')}"
        if best
        else "niciuna încă (auto-loop-ul nu a găsit încă un KEEP)"
    )
    n_recommend = min(range_max, 20)
    return PROMPT_TEMPLATE.format(
        lottery_id=stats["lottery"],
        name=stats["name"],
        format=stats["format"],
        total_draws=stats["total_draws"],
        history=stats["history"],
        stats_json=json.dumps({k: v for k, v in stats.items() if k != "best_strategy"}, ensure_ascii=False, indent=1),
        best_json=best_summary,
        range_total=n_recommend,
        range_max=range_max,
    )


# ─── Persistence ─────────────────────────────────────────────────────────────


def predictions_file(lottery_id: str) -> Path:
    return PROJECT_DIR / "data" / lottery_id / "ai_predictions.json"


def insights_log_file(lottery_id: str) -> Path:
    return PROJECT_DIR / "data" / lottery_id / "ai_insights.json"


def load_predictions(lottery_id: str) -> dict | None:
    path = predictions_file(lottery_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _append_insight(lottery_id: str, entry: dict) -> None:
    path = insights_log_file(lottery_id)
    try:
        log = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        if not isinstance(log, list):
            log = []
    except (json.JSONDecodeError, OSError):
        log = []
    log.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log[-MAX_INSIGHTS_LOG:], ensure_ascii=False, indent=1), encoding="utf-8")


def _sanitize_numbers(raw, config: LotteryConfig) -> list[int]:
    """Coerce the model's number list into a valid ranked list."""
    seen: set[int] = set()
    result: list[int] = []
    if isinstance(raw, list):
        for item in raw:
            try:
                n = int(item)
            except (TypeError, ValueError):
                continue
            if 1 <= n <= config.number_range and n not in seen:
                seen.add(n)
                result.append(n)
    # Fill up to 20 (or full range for small lotteries) with remaining numbers
    target = min(config.number_range, 20)
    if len(result) < target:
        counts_hint = list(range(1, config.number_range + 1))
        for n in counts_hint:
            if len(result) >= target:
                break
            if n not in seen:
                seen.add(n)
                result.append(n)
    return result[:target]


def _sanitize_suggestions(raw, config: LotteryConfig) -> list[dict]:
    allowed_params = {"delay", "momentum", "sb", "jc", "bf", "dd", "rw", "iw"}
    out = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            param = str(item.get("param", "")).lower().strip()
            if param not in allowed_params:
                continue
            try:
                suggested = float(item.get("suggested"))
            except (TypeError, ValueError):
                continue
            if param in {"rw", "iw"}:
                suggested = int(max(5, min(1000, suggested)))
            else:
                suggested = max(0.0, min(10.0, suggested))
            out.append({
                "param": param,
                "suggested": suggested,
                "reason": str(item.get("reason", ""))[:300],
            })
    return out[:6]


def _sanitize_proposed_actions(raw, lottery_id: str) -> list[dict]:
    """Păstrează doar acțiuni din whitelist, cu parametri coerenti (max 2)."""
    if not isinstance(raw, list):
        return []
    allowed = {"restart_loop", "stop_loop", "start_loop", "apply_strategy_params",
               "run_walkforward", "update_history", "run_ai_analysis", "set_agent_config"}
    out: list[dict] = []
    for item in raw[:2]:
        if not isinstance(item, dict):
            continue
        action_type = str(item.get("type", "")).strip().lower()
        params = item.get("params") if isinstance(item.get("params"), dict) else {}
        if action_type not in allowed:
            continue
        # loteria din analiză e implicita; nu lăsa modelul să țintească altele la întâmplare
        params.setdefault("lottery", lottery_id)
        why = str(item.get("why", ""))[:500]
        out.append({"type": action_type, "params": params, "why": why})
    return out


def _system_prompt() -> str:
    base = SYSTEM_PROMPT
    extra = load_prompt_overrides().get("analysis_extra", "")
    if extra:
        base += "\n\nINSTRUCȚIUNI SUPLEMENTARE (auto-învățare): " + str(extra)[:1500]
    return base


def save_prediction(lottery_id: str, model: str, stats: dict, parsed: dict, elapsed_s: float) -> dict:
    config = get_lottery_config(lottery_id)
    now_iso = datetime.now().isoformat(timespec="seconds")
    numbers = _sanitize_numbers(parsed.get("recommended_numbers"), config)
    record = {
        "version": PREDICTIONS_VERSION,
        "lottery": lottery_id,
        "model": model,
        "generated_at": now_iso,
        "elapsed_seconds": round(elapsed_s, 1),
        "confidence": str(parsed.get("confidence", "low"))[:10],
        "analysis": str(parsed.get("analysis", ""))[:2000],
        "noise_assessment": str(parsed.get("noise_assessment", ""))[:2000],
        "next_draw_focus": str(parsed.get("next_draw_focus", ""))[:1000],
        "recommended_numbers": numbers,
        "strategy_suggestions": _sanitize_suggestions(parsed.get("strategy_suggestions"), config),
        "proposed_actions": _sanitize_proposed_actions(parsed.get("proposed_actions"), lottery_id),
        "based_on_draws": stats["total_draws"],
        "last_draw_date": stats["history"].split(" .. ")[-1],
        "next_draw_date": config.next_draw_date().isoformat(),
    }
    path = predictions_file(lottery_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    _append_insight(lottery_id, {
        "generated_at": now_iso,
        "model": model,
        "confidence": record["confidence"],
        "analysis": record["analysis"],
        "top_numbers": numbers[:10],
        "suggestions": record["strategy_suggestions"],
    })
    append_entry(
        "ai_analysis",
        lottery_id=lottery_id,
        thought=record["analysis"] + ("\n\nZgomot probabil: " + record["noise_assessment"] if record["noise_assessment"] else ""),
        action=f"Numere propuse pentru {record['next_draw_date']}: {numbers[:10]}",
        why=record["next_draw_focus"],
        payload={
            "model": model,
            "confidence": record["confidence"],
            "recommended_numbers": numbers,
            "strategy_suggestions": record["strategy_suggestions"],
            "proposed_actions": record["proposed_actions"],
            "based_on_draws": record["based_on_draws"],
            "elapsed_seconds": record["elapsed_seconds"],
        },
    )

    # Propunerile de acțiuni merg pe magistrală: ledger + Telegram cu aprobare
    if record["proposed_actions"]:
        try:
            import ai_actions

            for pa in record["proposed_actions"]:
                ai_actions.propose_action(
                    pa["type"], pa["params"], why=pa["why"], proposed_by=f"strategist:{model}",
                )
        except Exception:
            pass
    return record


# ─── Orchestration ───────────────────────────────────────────────────────────


def _get_best_strategy_summary(lottery_id: str) -> dict | None:
    """Best strategy info from autoloop state (or KEEP replay), same source the UI uses."""
    try:
        import auto_loop
        config = get_lottery_config(lottery_id)
        state, _active = auto_loop.load_search_state_for_lottery(lottery_id)
        if state.best_metric <= 0:
            state = auto_loop.load_best_from_results_for_lottery(lottery_id)
        if state.best_metric <= 0:
            return None
        return {
            "name": state.best_strategy or "auto",
            "metric": round(state.best_metric, 6),
            "tf": state.tf, "hc": state.hc, "b": state.bal, "sb": state.sb,
            "delay": state.delay, "momentum": state.momentum,
            "jc": state.jc, "bf": state.bf, "dd": state.dd,
            "iw": state.iw, "rw": state.rw,
        }
    except Exception:
        return None


def analyze_lottery(lottery_id: str, model: str | None = None) -> dict | None:
    """Run one full AI analysis for a lottery and persist the result.

    Tries up to MAX_MODEL_ATTEMPTS models (requested model first, then
    preferred fallbacks) so a hung or broken model doesn't kill the sweep.
    """
    from load_data import load_lottery_draws

    config = get_lottery_config(lottery_id)
    draws = load_lottery_draws(lottery_id)
    if not draws:
        raise RuntimeError(f"No draws available for {lottery_id}")
    if not ollama_available():
        raise RuntimeError("No Ollama model available (is Ollama running?)")
    candidates = model_candidates(model)
    if not candidates:
        raise RuntimeError("No Ollama model available (is Ollama running?)")
    stats = compute_lottery_stats(config, draws, best=_get_best_strategy_summary(lottery_id))
    prompt = build_prompt(stats)
    started = time.time()
    last_error: Exception | None = None
    parsed: dict | None = None
    used_model = candidates[0]
    for candidate in candidates:
        used_model = candidate
        try:
            answer = chat(prompt, candidate, system=_system_prompt())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            continue
        parsed = extract_json(answer)
        if parsed:
            break
        last_error = RuntimeError(f"Model {candidate} returned non-JSON answer: {answer[:200]}")
    if not parsed:
        raise RuntimeError(str(last_error) if last_error else "All model attempts failed")
    elapsed = time.time() - started
    return save_prediction(lottery_id, used_model, stats, parsed, elapsed)


def analyze_all_lotteries(lottery_ids: list[str] | None = None, model: str | None = None) -> dict:
    ids = lottery_ids or [c.id for c in list_lottery_configs()]
    summary = {"analyzed": [], "failed": [], "model": None}
    if not ollama_available():
        summary["error"] = f"Ollama not reachable at {OLLAMA_HOST}"
        return summary
    model = model or resolve_model()
    summary["model"] = model
    for lottery_id in ids:
        try:
            record = analyze_lottery(lottery_id, model=model)
            summary["analyzed"].append({
                "lottery": lottery_id,
                "confidence": record["confidence"],
                "suggestions": len(record["strategy_suggestions"]),
            })
        except Exception as exc:
            summary["failed"].append({"lottery": lottery_id, "error": str(exc)[:200]})
    return summary


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--status" in sys.argv:
        print("host:", OLLAMA_HOST)
        print("available:", ollama_available())
        print("models:", json.dumps(list_models(), indent=1))
        print("resolved model:", resolve_model())
        sys.exit(0)
    ids = args or None
    result = analyze_all_lotteries(ids)
    print(json.dumps(result, ensure_ascii=False, indent=1))
