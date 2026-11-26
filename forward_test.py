"""Forward test — singurul test cu adevărat curat al unei strategii.

Backtestul și walkforward-ul folosesc parametri aleși privind TOT istoricul, deci
sunt contaminați de selecție. Aici „înghețăm" parametrii best-ului la o dată și îi
evaluăm DOAR pe extragerile apărute după acea dată (pre-înregistrare): fără
corecție pentru experimente multiple, fără look-ahead.

Evaluarea se face din istoricul descărcat, deci funcționează și când serverul a
fost oprit în momentul extragerilor (spre deosebire de biletele AI).

Stocare: data/{id}/forward_test.json → {"version": 1, "snapshots": [...]}.
Un snapshot nou apare doar când se schimbă parametrii best-ului.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path

from atomic_io import read_json_with_recovery, write_json_atomic
from lottery_config import get_lottery_config

SNAPSHOT_FILE = "forward_test.json"
MAX_SNAPSHOTS = 50
MIN_DRAWS_FOR_VERDICT = 30

PARAM_FIELDS = (
    "tf", "hc", "bal", "sb", "delay", "momentum", "jc", "bf", "dd", "iw", "rw",
    "prior_strength", "ewma_lambda", "use_anticorrelation", "anticorrelation_ratio",
    "extra_strategies",
)


def snapshot_path(lottery_id: str) -> Path:
    return get_lottery_config(lottery_id).state_dir / SNAPSHOT_FILE


def load_snapshots(lottery_id: str, path: Path | None = None) -> list[dict]:
    data, _ = read_json_with_recovery(path or snapshot_path(lottery_id))
    if isinstance(data, dict) and isinstance(data.get("snapshots"), list):
        return [s for s in data["snapshots"] if isinstance(s, dict)]
    return []


def _params_of(state) -> dict:
    data = state.to_dict() if hasattr(state, "to_dict") else dict(state)
    return {field: data.get(field) for field in PARAM_FIELDS}


def _params_key(params: dict) -> str:
    return hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:12]


def ensure_snapshot(lottery_id: str, state, now: datetime | None = None,
                    path: Path | None = None) -> dict | None:
    """Îngheață parametrii curenți dacă diferă de ultimul snapshot. Returnează snapshot-ul nou."""
    if getattr(state, "best_metric", 0) <= 0:
        return None
    params = _params_of(state)
    key = _params_key(params)
    target = path or snapshot_path(lottery_id)
    snapshots = load_snapshots(lottery_id, target)
    if snapshots and snapshots[-1].get("params_key") == key:
        return None
    now = now or datetime.now()
    snapshot = {
        "created_at": now.isoformat(timespec="seconds"),
        # se evaluează extragerile STRICT de după această zi (extragerile de azi pot
        # fi deja în istoric sau cunoscute)
        "frozen_after": now.date().isoformat(),
        "strategy": getattr(state, "best_strategy", ""),
        "stored_metric": round(float(getattr(state, "best_metric", 0.0)), 6),
        "params": params,
        "params_key": key,
    }
    snapshots.append(snapshot)
    write_json_atomic(target, {"version": 1, "snapshots": snapshots[-MAX_SNAPSHOTS:]}, ensure_ascii=True)
    return snapshot


def evaluate_snapshot(lottery_id: str, snapshot: dict, draws=None) -> dict:
    """Scorul parametrilor înghețați pe extragerile de după `frozen_after`."""
    import auto_loop
    import significance
    from generic_backtest import backtest_strategy
    from load_data import load_lottery_draws

    config = get_lottery_config(lottery_id)
    draws = list(draws) if draws is not None else load_lottery_draws(lottery_id)
    frozen_after = date.fromisoformat(snapshot["frozen_after"])
    start = next((i for i, d in enumerate(draws) if d.draw_date > frozen_after), len(draws))
    base = {
        "frozen_after": snapshot["frozen_after"],
        "created_at": snapshot.get("created_at"),
        "strategy": snapshot.get("strategy"),
        "stored_metric": snapshot.get("stored_metric"),
    }
    n_forward = len(draws) - start
    if n_forward <= 0 or start < 1:
        return {**base, "n_draws": 0, "baseline": round(config.baseline, 6), "verdict": "insufficient",
                "verdict_text": "încă nicio extragere după înghețarea parametrilor"}

    state = auto_loop.SearchState.from_dict({**snapshot["params"], "best_metric": 0.0})
    jackpot_data, draw_dates = {}, []
    if "jc" in state.extra_strategies:
        from jackpot_data import load_jackpot_data

        jackpot_data = load_jackpot_data(lottery_id, draws)
        draw_dates = [d.draw_date for d in draws]
    rank_fn = auto_loop.build_rank_fn_from_state(state, config=config,
                                                jackpot_data=jackpot_data, draw_dates=draw_dates)
    result = backtest_strategy(
        config, "forward_test", rank_fn, draws, ticket_size=config.ticket_size,
        recent_window=int(state.rw), initial_window=start,
    )
    stats = significance.config_edge_stats(config, result.average_hits, result.evaluated_draws, 1)
    stats.update(base)
    stats["p_two_sided"] = round(min(1.0, 2 * significance.normal_sf(abs(stats["z"]))), 6)
    if result.evaluated_draws < MIN_DRAWS_FOR_VERDICT:
        stats["verdict"] = "insufficient"
        stats["verdict_text"] = (f"doar {result.evaluated_draws} extrageri după înghețare — sunt necesare "
                                 f"cel puțin {MIN_DRAWS_FOR_VERDICT} pentru o concluzie")
    return stats


def summary(lottery_id: str, draws=None, path: Path | None = None) -> dict:
    """Evaluarea celui mai vechi snapshot încă activ + numărul de snapshot-uri.

    Folosim snapshot-ul cel mai VECHI al parametrilor curenți ca să avem cea mai
    lungă fereastră prospectivă; dacă parametrii s-au schimbat, cel mai recent.
    """
    snapshots = load_snapshots(lottery_id, path)
    if not snapshots:
        return {"n_draws": 0, "snapshots": 0, "verdict": "insufficient",
                "verdict_text": "niciun snapshot încă (se creează la următoarea pornire a buclei)"}
    latest = snapshots[-1]
    # aceiași parametri înghețați mai devreme rămân o pre-înregistrare validă de atunci
    current = next((s for s in snapshots if s.get("params_key") == latest.get("params_key")), latest)
    out = evaluate_snapshot(lottery_id, current, draws=draws)
    out["snapshots"] = len(snapshots)
    return out
