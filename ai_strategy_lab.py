"""AI Strategy Lab — sistem de plugin-uri pentru strategii generate de AI.

Plugin-urile trăiesc în `ai_strategies/{lottery_id}/*.py`. Fiecare fișier:

    NAME = "zgap"                      # cheie scurtă [a-z], unică, ÎNCEPE CU "z"
    DESCRIPTION = "..."
    SOURCE = "glm-5.3-flash:cloud"     # modelul care a scris-o

    def create(config):
        # config: lottery_config.LotteryConfig
        def rank(counts, recent_deque, last_seen, draw_index, recent_window):
            # counts: {număr: apariții totale}; recent_deque: deque[set]
            # last_seen: {număr: index ultima apariție}; return: TOATE numerele
            # 1..config.number_range ordonate de la cel mai bun la cel mai slab
            ...
        return rank

Flux: agentul AI scrie fișierul → validate_plugin() → backtest_plugin() →
dacă bate baseline-ul, promote_plugin() îl înscrie în registry.json, iar
auto_loop îl include în spațiul de căutare la următoarea pornire de buclă
(rotirea la 6h garantează preluarea).

Cheile trebuie să înceapă cu "z" ca să nu colideze cu prefixele parsate din
numele de configurări din results.tsv (tf, hc, b, sb, delay, momentum, jc,
bf, dd, iw, rw, ps, el, ac...).
"""

from __future__ import annotations

import importlib.util
import json
import py_compile
import re
import sys
import time
from collections import deque
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from generic_backtest import backtest_strategy
from lottery_config import LotteryConfig, get_lottery_config

AI_STRATEGIES_DIR = PROJECT_DIR / "ai_strategies"

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,20}$")


# ─── Registry ────────────────────────────────────────────────────────────────


def strategies_dir(lottery_id: str) -> Path:
    return AI_STRATEGIES_DIR / lottery_id


def registry_path(lottery_id: str) -> Path:
    return strategies_dir(lottery_id) / "registry.json"


def load_registry(lottery_id: str) -> list[dict]:
    try:
        data = json.loads(registry_path(lottery_id).read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_registry(lottery_id: str, entries: list[dict]) -> None:
    strategies_dir(lottery_id).mkdir(parents=True, exist_ok=True)
    registry_path(lottery_id).write_text(
        json.dumps(entries[-100:], ensure_ascii=False, indent=1), encoding="utf-8"
    )


def promoted_plugins(lottery_id: str) -> list[dict]:
    """Doar plugin-urile promovate (active în căutarea auto_loop)."""
    return [e for e in load_registry(lottery_id) if e.get("promoted")]


def promote_plugin(lottery_id: str, file_name: str, key: str, metric: float,
                   baseline: float, description: str = "", model: str = "") -> dict:
    entries = load_registry(lottery_id)
    entry = {
        "file": file_name,
        "key": key,
        "metric": metric,
        "baseline": baseline,
        "description": description[:500],
        "model": model[:100],
        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "promoted": True,
    }
    # Un singur plugin activ per cheie: demotează celelalte cu aceeași cheie
    for existing in entries:
        if existing.get("key") == key:
            existing["promoted"] = False
    entries.append(entry)
    save_registry(lottery_id, entries)
    return entry


# ─── Loading & validation ────────────────────────────────────────────────────


def _import_plugin(path: Path):
    spec = importlib.util.spec_from_file_location(f"ai_strategy_{int(time.time() * 1000)}", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot create import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_plugin(path: Path):
    """Importă plugin-ul și returnează (module, meta). Ridică excepție la probleme."""
    module = _import_plugin(path)
    name = str(getattr(module, "NAME", "")).strip().lower()
    create = getattr(module, "create", None)
    if not NAME_RE.match(name):
        raise ValueError(f"invalid NAME {name!r} (expected [a-z][a-z0-9_] {{2,21}})")
    if not callable(create):
        raise ValueError("missing create(config) function")
    if not name.startswith("z"):
        raise ValueError("strategy NAME must start with 'z' (avoid config-token collisions)")
    meta = {
        "name": name,
        "description": str(getattr(module, "DESCRIPTION", ""))[:500],
        "source": str(getattr(module, "SOURCE", ""))[:200],
    }
    return module, meta


def validate_plugin(path: Path, lottery_id: str) -> dict:
    """Validează sintaxa + un smoke-test pe date sintetice. Returnează meta.

    Datele sintetice respectă EXACT formele din backtestul real:
    counts: dict[int, int]; recent_deque: deque[set[int]];
    last_seen: dict[int, int]; draw_index: int; recent_window: int.
    """
    py_compile.compile(str(path), doraise=True)
    module, meta = load_plugin(path)
    config = get_lottery_config(lottery_id)
    rank_fn = module.create(config)

    range_n = config.number_range
    counts = {n: (n * 7) % 13 for n in range(1, range_n + 1)}
    # deque de SET-uri (câte un set per extragere), ca în backtestul real
    recent = deque(
        ({((i * 5 + j * 11) % range_n) + 1 for j in range(6)} for i in range(20)),
        maxlen=10,
    )
    last_seen = {n: (n * 3) % 50 for n in range(1, range_n + 1)}
    result = rank_fn(counts, recent, last_seen, 100, 10)
    if not isinstance(result, (list, tuple)):
        raise ValueError("rank must return a list")
    if sorted(result) != list(range(1, range_n + 1)):
        raise ValueError("rank must return a permutation of 1..number_range")
    return meta


def backtest_plugin(path: Path, lottery_id: str, draws) -> dict:
    """Backtest complet al plugin-ului. Returnează metricele."""
    module, meta = load_plugin(path)
    config = get_lottery_config(lottery_id)
    rank_fn = module.create(config)
    result = backtest_strategy(
        config,
        meta["name"],
        rank_fn,
        draws,
        ticket_size=config.ticket_size,
        recent_window=200,
        initial_window=30,
    )
    return {
        "average_hits": round(result.average_hits, 6),
        "hit_rate_2_plus": round(result.hit_rate_2_plus, 4),
        "hit_rate_3_plus": round(result.hit_rate_3_plus, 4),
        "portfolio_avg": round(result.portfolio_average_best_hits, 6),
    }


def backtest_plugin_holdout(path: Path, lottery_id: str, draws, holdout_fraction: float = 0.25) -> dict:
    """Scorul plugin-ului DOAR pe ultimele `holdout_fraction` extrageri.

    Extragerile anterioare servesc drept încălzire (statisticile online), deci
    nu există look-ahead. Împreună cu backtestul complet arată dacă avantajul
    se menține în perioada recentă sau vine doar din istoricul vechi.
    """
    module, meta = load_plugin(path)
    config = get_lottery_config(lottery_id)
    split = max(30, int(len(draws) * (1.0 - holdout_fraction)))
    if split >= len(draws) - 1:
        raise ValueError("prea puține extrageri pentru holdout")
    result = backtest_strategy(
        config,
        f"{meta['name']}_holdout",
        module.create(config),
        draws,
        ticket_size=config.ticket_size,
        recent_window=200,
        initial_window=split,
    )
    return {
        "average_hits": round(result.average_hits, 6),
        "evaluated_draws": result.evaluated_draws,
    }


# ─── auto_loop integration ───────────────────────────────────────────────────

_loaded_keys: dict[tuple[str, str], dict] = {}  # (lottery, key) -> plugin metadata


def registered_keys(lottery_id: str | None = None) -> list[str]:
    return list(dict.fromkeys(key for lid, key in _loaded_keys
                              if lottery_id is None or lid == lottery_id))


def register_promoted_for_config(config: LotteryConfig) -> list[str]:
    """(Re)încarcă plugin-urile promovate ale loteriei date.

    Apelez-o din auto_loop la pornirea buclei. Returnează cheile înregistrate.
    """
    keys = [k for k, info in _loaded_keys.items() if info["lottery"] == config.id]
    for key in keys:
        _loaded_keys.pop(key, None)

    registered: list[str] = []
    for entry in promoted_plugins(config.id):
        file_name = entry.get("file")
        key = entry.get("key")
        if not file_name or not key:
            continue
        path = strategies_dir(config.id) / file_name
        if not path.exists():
            continue
        try:
            validate_plugin(path, config.id)
        except Exception:
            continue  # plugin stricat — rămâne neînregistrat
        _loaded_keys[(config.id, key)] = {"lottery": config.id, "file": file_name}
        registered.append(key)
    return registered


def make_extra(config: LotteryConfig, key: str):
    """Creează rank_fn pentru o cheie AI înregistrată, sau None."""
    info = _loaded_keys.get((config.id, key))
    if not info or info["lottery"] != config.id:
        return None
    path = strategies_dir(config.id) / info["file"]
    if not path.exists():
        return None
    try:
        module, _meta = load_plugin(path)
        return module.create(config)
    except Exception:
        return None


def unique_key(lottery_id: str, suggested: str) -> str:
    """Cheie unică 'z...' bazată pe sugestia modelului, trunchiată la 8 caractere."""
    base = re.sub(r"[^a-z0-9]", "", (suggested or "").lower())[:7] or "gen"
    base = "z" + base[:7]
    existing = {e.get("key") for e in load_registry(lottery_id)}
    candidate = base
    counter = 0
    while candidate in existing or candidate in _loaded_keys:
        counter += 1
        candidate = f"{base}{counter}"
        if len(candidate) > 10:
            candidate = f"zgen{counter}"
    return candidate
