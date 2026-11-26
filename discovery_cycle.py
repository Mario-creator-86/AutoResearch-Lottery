"""Ciclul autonom de descoperire — rulat de server după fiecare actualizare a istoricului.

1. aduce la zi reportul (loteriile RO, arhiva oficială);
2. scanează extragerile pentru anomalii reale (bias_scan);
3. calculează valoarea așteptată a biletelor la reportul curent (play_advisor);
4. alertează (ledger + Telegram) DOAR la schimbări care contează:
   - o anomalie devine semnificativă statistic după toate corecțiile;
   - reportul unei loterii ajunge aproape de / peste pragul de rentabilitate.

Fără schimbări nu trimite nimic — tăcerea înseamnă „nimic exploatabil", nu „nu
merge". Starea ultimului ciclu: data/discovery.json (GET /api/discovery).
"""

from __future__ import annotations

from datetime import datetime

from atomic_io import read_json_with_recovery, write_json_atomic
from lottery_config import DATA_DIR

STATE_FILE = "discovery.json"
ALERT_ADVICE = ("near", "favorable")


def _state_path():
    return DATA_DIR / STATE_FILE


def load_state() -> dict:
    data, _ = read_json_with_recovery(_state_path())
    return data if isinstance(data, dict) else {}


def update_jackpots() -> dict:
    """Reportul la zi pentru loteriile cu arhivă oficială incrementală."""
    out = {}
    try:
        import scrape_jackpot_ro
    except Exception as exc:
        return {"error": str(exc)[:200]}
    for lottery_id in scrape_jackpot_ro.ROMANIA_JACKPOT_LOTTERIES:
        try:
            out[lottery_id] = {"ok": True, "added": int(scrape_jackpot_ro.scrape_lottery(lottery_id) or 0)}
        except Exception as exc:
            out[lottery_id] = {"ok": False, "error": str(exc)[:200]}
    return out


def _compact_scan(result: dict) -> dict:
    keys = ("lottery", "name", "n_draws", "verdict", "verdict_text", "strongest_test", "p_adjusted",
            "exploitable", "face_value_edge_pct", "most_frequent", "last_draw", "error")
    return {key: result[key] for key in keys if key in result}


def find_alerts(previous: dict, scans: list[dict], advice: list[dict]) -> list[dict]:
    """Doar tranzițiile: ce e nou față de ciclul anterior."""
    import bias_scan

    alerts = []
    old_scans = {item.get("lottery"): item for item in previous.get("anomalies", [])}
    for item in scans:
        was = (old_scans.get(item.get("lottery")) or {}).get("verdict")
        if item.get("verdict") == "significant" and was != "significant":
            alerts.append({"kind": "anomaly", "lottery": item["lottery"],
                           "text": "🔬 Anomalie statistică reală detectată — " + bias_scan.summary_text(item)})
    old_advice = {item.get("lottery"): item for item in previous.get("advice", [])}
    for item in advice:
        was = (old_advice.get(item.get("lottery")) or {}).get("verdict")
        now = item.get("verdict")
        if now in ALERT_ADVICE and (was not in ALERT_ADVICE or ALERT_ADVICE.index(now) > ALERT_ADVICE.index(was)):
            alerts.append({"kind": "jackpot", "lottery": item["lottery"],
                           "text": f"💰 {item.get('name', item['lottery'])}: report "
                                   f"{item.get('current_jackpot', 0):,.0f} {item.get('currency', '')} — "
                                   f"{item.get('verdict_text', '')}"})
    return alerts


def run_cycle(notify: bool = True, update: bool = True, save: bool = True) -> dict:
    import bias_scan
    import play_advisor

    previous = load_state()
    jackpots = update_jackpots() if update else {}
    scan = bias_scan.run_all(save=save)
    advice = play_advisor.advise_all()
    alerts = find_alerts(previous, scan["results"], advice)

    state = {
        "ran_at": datetime.now().isoformat(timespec="seconds"),
        "jackpot_update": jackpots,
        "anomalies": [_compact_scan(item) for item in scan["results"]],
        "advice": advice,
        "alerts": alerts,
        "summary": summary_text(scan["results"], advice),
    }
    if save:
        write_json_atomic(_state_path(), state, ensure_ascii=False)
    for alert in alerts:
        try:
            import decision_ledger

            decision_ledger.append_entry(
                "system", lottery_id=alert["lottery"], thought=alert["text"],
                action=f"discovery_alert:{alert['kind']}",
                why="schimbare detectată de ciclul autonom de descoperire", status="done",
            )
        except Exception:
            pass
        if notify:
            try:
                import telegram_bot

                telegram_bot.notify(alert["text"])
            except Exception:
                pass
    return state


def summary_text(scans: list[dict], advice: list[dict]) -> str:
    """Rezumat scurt, onest, al întregului ciclu (UI + prompt-urile AI)."""
    counted = [s for s in scans if s.get("verdict") not in (None, "insufficient")]
    significant = [s for s in counted if s.get("verdict") == "significant"]
    suggestive = [s for s in counted if s.get("verdict") == "suggestive"]
    parts = []
    if significant:
        parts.append("Anomalii semnificative: " + ", ".join(s.get("name", s["lottery"]) for s in significant) + ".")
    else:
        parts.append(f"Nicio anomalie semnificativă în {len(counted)} loterii scanate — extragerile se "
                     "comportă ca hazard pur, deci nicio formulă nu poate crește șansa unui bilet.")
    if suggestive:
        parts.append("De urmărit (sugestiv, neconfirmat): "
                     + ", ".join(s.get("name", s["lottery"]) for s in suggestive) + ".")
    good = [a for a in advice if a.get("verdict") in ALERT_ADVICE]
    if good:
        parts.append("Report aproape de/peste prag: " + ", ".join(a.get("name", a["lottery"]) for a in good) + ".")
    else:
        known = [a for a in advice if a.get("current_return_pct") is not None]
        if known:
            best = max(known, key=lambda a: a["current_return_pct"])
            parts.append(f"Niciun report nu atinge pragul de rentabilitate; cel mai bun raport acum: "
                         f"{best.get('name', best['lottery'])} (~{best['current_return_pct']:.0f}% din preț).")
    return " ".join(parts)


if __name__ == "__main__":
    print(run_cycle(notify=False)["summary"])
