"""AI Ticket Agent — propune bilete în ziua extragerii, le urmărește și învață.

Ciclu per loterie:
  1. GENERARE (doar în ziua extragerii): combină clasamentul celei mai bune
     strategii deterministe cu numerele recomandate de AI, și cere modelului
     Ollama să construiască biletele finale + DE CE le propune. Biletele sunt
     ordonate de la cel mai probabil la cel mai puțin probabil (biletul 1 =
     favoritul) și se trimit complete pe Telegram. Le salvează în
     data/{id}/ai_tickets.json cu status „pending".
  2. VERIFICARE (după extragere, când istoricul s-a actualizat): compară
     biletele cu numerele câștigătoare reale, calculează hits + câștigul
     estimat pe categorii, marchează „checked", trimite raportul complet pe
     Telegram, întreabă modelul CE A ÎNVĂȚAT și scrie totul în Decision
     Ledger + își ajustează propriile instrucțiuni (self-improvement).

Astfel sistemul „se actualizează constant cu numerele câștigătoare" și își
rafinează propunerile de la o zi la alta.

Loteriile cu mai multe extrageri pe zi (ex. Kaskada 12/24 și Multi Multi:
14:00 și 21:40) primesc un set de bilete PENTRU FIECARE extragere, iar
verificarea le potrivește cu extragerea corectă (draw_index).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime, time as dt_time
from pathlib import Path

from agent_base import BaseAgent

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from decision_ledger import append_entry
from draw_matching import match_draw
from ai_tickets_store import TICKETS_VERSION, load_ticket_store, save_ticket_store, tickets_file
from lottery_config import list_lottery_configs
from prize_estimates import estimate_ticket_prize


class AITicketAgent(BaseAgent):

    # politicos: cel mult o actualizare de istoric la 20 min per loterie
    HISTORY_REFRESH_MIN_INTERVAL_SECONDS = 20 * 60

    def __init__(self):
        super().__init__("ai_ticket_agent", poll_interval=120)
        config = self.get_config()
        self.tickets_per_day = int(config.get("tickets_per_day", 4))
        self._history_refresh_at: dict[str, float] = {}
        default_ids = [c.id for c in list_lottery_configs()]
        configured = config.get("lotteries")
        self.lottery_order = configured if isinstance(configured, list) and configured else default_ids

    # ─── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _blend_ranking(strategy_ranked: list[int], ai_numbers: list[int]) -> list[int]:
        """Interclasează clasamentul strategiei cu numerele AI (AI la fiecare 2)."""
        seen: set[int] = set()
        blended: list[int] = []
        ai_iter = iter(ai_numbers)
        for i, n in enumerate(strategy_ranked):
            blended.append(n)
            seen.add(n)
            if i % 2 == 1:
                for a in ai_iter:
                    if a not in seen:
                        blended.append(a)
                        seen.add(a)
                        break
        return blended

    def _draws_for_date(self, draws, target: date):
        return [d for d in draws if d.draw_date == target]

    @staticmethod
    def _upcoming_slot(config, now: datetime | None = None) -> tuple[int, str] | None:
        """Următoarea extragere din AZI ca (index 1-based, oră "HH:MM").

        Loteriile cu mai multe extrageri pe zi (ex. Kaskada/Multi Multi:
        14:00 și 21:40) primesc bilete separate per extragere. None = toate
        extragerile de azi au trecut → nu mai generăm nimic azi.
        """
        times = [t for t in (config.draw_times_local or ()) if t]
        if not times:
            return (1, "")
        if len(times) == 1:
            # o singură extragere: validă doar până la ora ei
            try:
                hh, mm = str(times[0]).split(":")[:2]
            except ValueError:
                return (1, str(times[0]))
            now = now or datetime.now()
            if now.time() >= dt_time(int(hh), int(mm)):
                return None
            return (1, str(times[0]))
        now = now or datetime.now()
        for i, t in enumerate(times, 1):
            try:
                hh, mm = str(t).split(":")[:2]
                slot = dt_time(int(hh), int(mm))
            except ValueError:
                continue
            if now.time() < slot:
                return (i, str(t))
        return None

    @staticmethod
    def _sort_tickets_by_likelihood(tickets: list[dict], strategy_ranked: list[int]) -> list[dict]:
        """Ordonează biletele de la cel mai probabil la cel mai puțin probabil.

        Scor = suma pozițiilor numerelor în clasamentul strategiei deterministe
        (poziții mici = numere bune). La egalitate rămâne ordinea dată de model
        (sortare stabilă). Fiecare bilet primește și câmpul „score" pentru
        transparență.
        """
        if not strategy_ranked:
            return tickets
        pos = {n: i for i, n in enumerate(strategy_ranked)}
        default = len(strategy_ranked)

        def score(t: dict) -> int:
            return sum(pos.get(n, default) for n in t.get("main", []))

        ordered = sorted(tickets, key=score)
        for t in ordered:
            t["score"] = score(t)
        return ordered

    # ─── generare bilete ─────────────────────────────────────────────────

    def generate_tickets(self, lottery_id: str, ollama_ai) -> None:
        from load_data import load_lottery_draws

        config = ollama_ai.get_lottery_config(lottery_id)
        draws = load_lottery_draws(lottery_id)
        if not draws:
            return
        target_date = config.next_draw_date()
        store = load_ticket_store(lottery_id)

        # Propunem DOAR în ziua extragerii: retragem orice propunere făcută
        # prea devreme (pentru o extragere viitoare), ca să fie refăcută atunci.
        today_iso = date.today().isoformat()
        stale = [e for e in store["entries"]
                 if e.get("status") == "pending" and e.get("draw_date", "") > today_iso]
        if stale:
            store["entries"] = [e for e in store["entries"] if e not in stale]
            save_ticket_store(lottery_id, store)
            self.log_warning(f"{lottery_id}: am retras {len(stale)} propuneri făcute "
                             f"înainte de ziua extragerii")
        if target_date != date.today():
            return  # extragerea nu e azi → nu propunem încă
        slot = self._upcoming_slot(config)
        if slot is None:
            return  # toate extragerile de azi au trecut
        slot_idx, slot_time = slot
        n_slots = len([t for t in (config.draw_times_local or ()) if t]) or 1
        already = any(
            e.get("draw_date") == target_date.isoformat()
            and e.get("draw_index", n_slots) == slot_idx
            for e in store["entries"]
        )
        if already:
            return  # deja există propunere pentru această extragere

        prediction = ollama_ai.load_predictions(lottery_id)
        if not prediction:
            try:
                prediction = ollama_ai.analyze_lottery(lottery_id)
            except Exception as exc:
                self.log_warning(f"{lottery_id}: nu am putut genera predicția pentru bilete: {exc}")
                return

        # clasamentul strategiei deterministe
        try:
            import auto_loop

            state, _active = auto_loop.load_search_state_for_lottery(lottery_id)
            if state.best_metric <= 0:
                state = auto_loop.load_best_from_results_for_lottery(lottery_id)
            strategy_ranked = auto_loop.rank_numbers_from_draws(draws, state, lottery_id=lottery_id)
        except Exception as exc:
            self.log_warning(f"{lottery_id}: rank_strategie indisponibil ({exc}); folosesc doar AI")
            strategy_ranked = []

        ai_numbers = prediction.get("recommended_numbers") or []
        blended = self._blend_ranking(strategy_ranked, ai_numbers)
        best = ollama_ai._get_best_strategy_summary(lottery_id) or {}

        n_main = config.ticket_size
        n_bonus = config.min_bonus_numbers if config.bonus_range else 0

        self.update_heartbeat("running", current_task=f"{lottery_id}: AI construiește biletele pentru {target_date.isoformat()}")

        if n_bonus:
            ticket_example = (
                '{"main": [' + str(n_main) + ' numere], "bonus": [' + str(n_bonus)
                + ' numere bonus], "why": "max 12 cuvinte"}'
            )
            bonus_rule = f"- Biletele pot avea și {n_bonus} numere bonus între 1 și {config.bonus_range}."
        else:
            ticket_example = '{"main": [' + str(n_main) + ' numere], "why": "max 12 cuvinte"}'
            bonus_rule = "- Fără numere bonus."

        why_rule = ("Fiecare bilet cu raționament de max 12 cuvinte."
                    if self.tickets_per_day > 6
                    else "Fiecare bilet cu 1-2 propoziții de ce ai ales exact aceste numere.")

        prompt = f"""Construiește {self.tickets_per_day} bilete pentru {config.name} ({config.numbers_drawn}/{config.number_range}),
pentru extragerea de {target_date.isoformat()}{f" (ora {slot_time})" if slot_time else ""}.

Ai la dispoziție două clasamente (de la cel mai bun număr la cel mai slab):
1. Clasamentul strategiei deterministe (antrenată prin backtest pe {len(draws)} extrageri):
{json.dumps(strategy_ranked[:25])}
2. Numerele recomandate de analiza ta AI anterioară:
{json.dumps(ai_numbers[:20])}

Motivația analizei AI: {prediction.get('analysis', '')[:500]}
Concentrarea recomandată: {prediction.get('next_draw_focus', '')[:300]}
Best strategie: metric={best.get('metric')}, weights={json.dumps({k: best.get(k) for k in ('tf','hc','b','sb','delay','momentum')})}

Reguli:
- Fiecare bilet: exact {n_main} numere distincte între 1 și {config.number_range}.
{bonus_rule}
- {why_rule}
- Diversifică: primele bilete = ancoră pe top strategie, apoi echilibrate strategie+AI,
  apoi orientate pe AI/overdue, ultimele = explorație (perechi/sectoare). Nu repeta aceleași numere.
- Ordine obligatorie: biletele de la cel mai probabil la cel mai puțin probabil
  (primul bilet = favoritul nostru, cel pe care l-am juca noi înșiși).
- Alege tipare de numere realiste (nu toate consecutive, nu toate pare).

Răspunde DOAR cu JSON:
{{
  "thoughts": "raționamentul pentru alegerea biletelor",
  "tickets": [
    {ticket_example}
  ]
}}"""

        model_used = None
        tickets = None
        thoughts = ""
        try:
            candidates = ollama_ai.model_candidates(None)
            for candidate in candidates:
                try:
                    answer = ollama_ai.chat(prompt, candidate, system=self._ticket_system_prompt())
                    model_used = candidate
                except Exception:
                    continue
                parsed = ollama_ai.extract_json(answer or "")
                if not parsed:
                    continue
                tickets = self._sanitize_tickets(parsed.get("tickets"), config)
                thoughts = str(parsed.get("thoughts", ""))[:2500]
                if tickets:
                    break
        except Exception as exc:
            self.log_error(f"{lottery_id}: generare bilete eșuată: {exc}")

        if not tickets:
            # fallback determinist: primele n din blend + variante deplasate
            tickets = self._fallback_tickets(blended, n_main, n_bonus, config)
            thoughts = "Fallback determinist (modelul nu a răspuns valid): interclasare strategie+AI."
            model_used = "fallback-blend"

        # garanție deterministă de ordonare: favoritul primul (indiferent de
        # ordinea în care le-a dat modelul)
        tickets = self._sort_tickets_by_likelihood(tickets, strategy_ranked)

        # Portofoliu: aceleași șanse per bilet, dar mai puțină suprapunere (P(≥1
        # premiu/extragere) mai mare) și numere mai puțin populare (parte mai mare
        # din premiile împărțite). Favoritul (biletul 1) rămâne neschimbat.
        portfolio: dict = {}
        try:
            from portfolio_optimizer import optimize_ticket_dicts

            last_draw = tuple(draws[-1].numbers) if draws and hasattr(draws[-1], "numbers") else ()
            tickets, portfolio = optimize_ticket_dicts(tickets, config, lottery_id, last_draw=last_draw)
            if len(tickets) > 1:
                tickets = tickets[:1] + self._sort_tickets_by_likelihood(tickets[1:], strategy_ranked)
        except Exception as exc:
            self.log_warning(f"{lottery_id}: optimizarea portofoliului a eșuat: {exc}")

        entry = {
            "draw_date": target_date.isoformat(),
            "draw_index": slot_idx,
            "draw_time": slot_time,
            "next_draw_times": list(config.draw_times_local or ()),
            "model": model_used or "unknown",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "reasoning": thoughts,
            "strategy_metric": best.get("metric"),
            "tickets": tickets,
            "portfolio": portfolio,
            "status": "pending",
        }
        store["entries"].append(entry)
        save_ticket_store(lottery_id, store)
        append_entry(
            "ai_tickets",
            lottery_id=lottery_id,
            thought=thoughts,
            action=f"Bilete propuse pentru {target_date.isoformat()}: "
                   + "; ".join(f"[{','.join(map(str, t['main']))}]" for t in tickets),
            why="; ".join((t.get("why") or "")[:150] for t in tickets)[:1000],
            payload={"draw_date": entry["draw_date"], "tickets": tickets, "model": entry["model"]},
        )
        self._notify_new_tickets(lottery_id, config, entry)
        self.log_success(f"{lottery_id}: {len(tickets)} bilete AI propuse pentru {target_date.isoformat()}")

    @staticmethod
    def _slot_label(entry: dict) -> str:
        """Etichetă „— extragerea 2 din 2 (ora 21:40)" pentru loteriile multi-extragere."""
        n_times = len(entry.get("next_draw_times") or ())
        if entry.get("draw_index") and n_times > 1:
            return f" — extragerea {entry['draw_index']} din {n_times} (ora {entry.get('draw_time', '?')})"
        if entry.get("draw_time"):
            return f" (ora {entry['draw_time']})"
        return ""

    def _notify_new_tickets(self, lottery_id: str, config, entry: dict) -> None:
        """Telegram: propunerea completă, în ordinea probabilității (favoritul primul)."""
        try:
            import telegram_bot

            lines = [
                f"🎟 <b>{config.name}</b> — biletele AI pentru "
                f"<b>{entry['draw_date']}</b>{self._slot_label(entry)}",
                "Ordinea = probabilitatea estimată (1 = cel mai favorit):",
            ]
            for i, t in enumerate(entry["tickets"], 1):
                nums = " ".join(f"{n:02d}" for n in t["main"])
                bonus = f" + bonus {' '.join(map(str, t['bonus']))}" if t.get("bonus") else ""
                why = f" — <i>{t['why']}</i>" if t.get("why") else ""
                lines.append(f"<b>{i}.</b> <code>{nums}</code>{bonus}{why}")
            lines.append(f"<i>Model: {entry.get('model', '?')} · metrică strategie: "
                         f"{entry.get('strategy_metric', '?')}</i>")
            pf = entry.get("portfolio") or {}
            if pf.get("p_any_prize_after") is not None:
                lines.append(
                    f"<i>Șansa ca măcar un bilet să prindă premiu: "
                    f"{pf['p_any_prize_before'] * 100:.1f}% → <b>{pf['p_any_prize_after'] * 100:.1f}%</b> "
                    f"(optimizare acoperire, {pf.get('simulated_draws', 0)} extrageri simulate)</i>"
                )
            telegram_bot.notify("\n".join(lines))
        except Exception:
            pass

    def _ticket_system_prompt(self) -> str:
        import ollama_ai

        base = (
            "Ești un consilier de joc care construiește bilete la loterie pe baza unor clasamente "
            "cantitative. Explici mereu raționamentul. Răspunzi EXCLUSIV cu JSON valid."
        )
        extra = ollama_ai.load_prompt_overrides().get("ticket_extra", "")
        if extra:
            base += "\n\nINSTRUCȚIUNI SUPLEMENTARE (auto-învățare): " + str(extra)[:1200]
        return base

    def _sanitize_tickets(self, raw, config) -> list[dict]:
        if not isinstance(raw, list):
            return []
        n_main = config.ticket_size
        n_bonus = config.min_bonus_numbers if config.bonus_range else 0
        out: list[dict] = []
        seen_tickets: set[tuple] = set()
        for item in raw[: self.tickets_per_day]:
            if not isinstance(item, dict):
                continue
            try:
                main = sorted({int(n) for n in item.get("main", []) if 1 <= int(n) <= config.number_range})
            except (TypeError, ValueError):
                continue
            if len(main) != n_main:
                continue
            bonus: list[int] = []
            if n_bonus:
                try:
                    bonus = sorted({int(n) for n in item.get("bonus", []) if 1 <= int(n) <= config.bonus_range})[:n_bonus]
                except (TypeError, ValueError):
                    bonus = []
                if len(bonus) < n_bonus:
                    continue
            key = tuple(main) + tuple(bonus)
            if key in seen_tickets:
                continue
            seen_tickets.add(key)
            out.append({"main": main, "bonus": bonus, "why": str(item.get("why", ""))[:400]})
        return out

    def _fallback_tickets(self, blended: list[int], n_main: int, n_bonus: int, config) -> list[dict]:
        tickets = []
        for offset in range(self.tickets_per_day):
            pool = blended[offset:: max(1, n_main // 2)] + blended
            seen: set[int] = set()
            main: list[int] = []
            for n in pool:
                if n not in seen:
                    main.append(n)
                    seen.add(n)
                if len(main) == n_main:
                    break
            if len(main) < n_main:
                main += [n for n in range(1, config.number_range + 1) if n not in seen][: n_main - len(main)]
            bonus = []
            if n_bonus and config.bonus_range:
                bonus = [((offset + 1) * 7) % config.bonus_range + 1]
            tickets.append({"main": sorted(main), "bonus": bonus,
                            "why": "Interclasare automată strategie deterministă + recomandările AI."})
        return tickets[: self.tickets_per_day]

    # ─── verificare post-extragere ───────────────────────────────────────

    def _maybe_refresh_history(self, lottery_id: str, pending: list[dict],
                               draws_by_date: dict) -> bool:
        """Actualizează istoricul dacă o extragere cu bilete pending a trecut
        (ora ei e depășită) dar rezultatul lipsește încă din istoric.

        Fără asta, rezultatele ar apărea abia la următorul ciclu global de
        auto-update (6h). Throttle: max o încercare la 20 min per loterie.
        """
        if not hasattr(self, "_history_refresh_at"):
            self._history_refresh_at = {}
        now = datetime.now()
        needs = False
        try:
            from lottery_config import get_lottery_config

            draw_times = sorted(str(t) for t in (get_lottery_config(lottery_id).draw_times_local or ()) if t)
        except Exception:
            draw_times = []
        for entry in pending:
            draw_date_iso = str(entry.get("draw_date") or "")
            if not draw_date_iso:
                continue
            available = draws_by_date.get(draw_date_iso) or []
            if match_draw(available, draw_time=entry.get("draw_time"),
                          draw_index=entry.get("draw_index")) is not None:
                continue  # extragerea acestui slot e deja în istoric
            # (la loteriile cu 2 extrageri/zi, prezența extragerii de 14:00 nu
            # înseamnă că și cea de 21:40 e în istoric)
            try:
                draw_day = date.fromisoformat(draw_date_iso)
            except ValueError:
                continue
            if draw_day > date.today():
                continue
            if draw_day == date.today():
                # extragerea trebuie să fi trecut deja: folosim ora din intrare,
                # iar pentru intrări vechi fără oră — ultima oră a zilei
                when = str(entry.get("draw_time") or (draw_times[-1] if draw_times else ""))
                if when:
                    try:
                        hh, mm = when.split(":")[:2]
                        if now.time() < dt_time(int(hh), int(mm)):
                            continue  # extragerea nu a avut loc încă
                    except ValueError:
                        pass
            needs = True
            break
        if not needs:
            return False
        last = self._history_refresh_at.get(lottery_id, 0.0)
        if time.time() - last < self.HISTORY_REFRESH_MIN_INTERVAL_SECONDS:
            return False
        self._history_refresh_at[lottery_id] = time.time()
        try:
            from scrape_lotteries import SCRAPER_MAP, download_lottery

            if lottery_id not in SCRAPER_MAP:
                return False
            self.update_heartbeat("running",
                                  current_task=f"{lottery_id}: actualizez istoricul după extragere")
            download_lottery(lottery_id)
            self.log_success(f"{lottery_id}: istoric actualizat după extragere (pentru verificarea biletelor)")
            return True
        except Exception as exc:
            self.log_warning(f"{lottery_id}: actualizarea istoricului a eșuat: {exc}")
            return False

    def check_pending_tickets(self, lottery_id: str, ollama_ai) -> None:
        from load_data import load_lottery_draws

        store = load_ticket_store(lottery_id)
        pending = [e for e in store["entries"] if e.get("status") == "pending"]
        if not pending:
            return
        draws = load_lottery_draws(lottery_id)
        draws_by_date = {}
        for d in draws:
            draws_by_date.setdefault(d.draw_date.isoformat(), []).append(d)

        # extragerea a trecut dar rezultatul lipsește din istoric? actualizăm acum
        if self._maybe_refresh_history(lottery_id, pending, draws_by_date):
            draws = load_lottery_draws(lottery_id)
            if not draws:
                return
            draws_by_date = {}
            for d in draws:
                draws_by_date.setdefault(d.draw_date.isoformat(), []).append(d)

        checked: list[dict] = []
        for entry in pending:
            available = draws_by_date.get(entry.get("draw_date", ""), [])
            if not available:
                continue  # extragerea nu e încă în istoric
            draw = match_draw(available, draw_time=entry.get("draw_time"),
                              draw_index=entry.get("draw_index"))
            if draw is None:
                continue
            win = set(draw.numbers)
            win_bonus = set(getattr(draw, "bonuses", ()) or ())
            results = []
            for t in entry.get("tickets", []):
                hits = sorted(set(t.get("main", [])) & win)
                bonus_hits = sorted(set(t.get("bonus", [])) & win_bonus)
                results.append({
                    "main": t.get("main"),
                    "played_bonus": t.get("bonus"),
                    "hits": hits,
                    "hit_count": len(hits),
                    "bonus_hits": bonus_hits,
                    "prize": estimate_ticket_prize(lottery_id, len(hits), len(bonus_hits)),
                    "why": t.get("why"),
                })
            entry["status"] = "checked"
            entry["checked_at"] = datetime.now().isoformat(timespec="seconds")
            entry["draw_numbers"] = sorted(draw.numbers)
            entry["draw_bonus"] = sorted(win_bonus)
            entry["results"] = results
            total_hits = sum(r["hit_count"] for r in results)
            prized = [r["prize"] for r in results if r.get("prize") and r["prize"].get("amount")]
            entry["estimated_prize"] = (
                {"amount": sum(p["amount"] for p in prized), "currency": prized[0]["currency"]}
                if prized else {"amount": 0.0, "currency": ""}
            )
            checked.append({"entry": entry, "results": results, "total_hits": total_hits})

        if not checked:
            return
        save_ticket_store(lottery_id, store)

        # învățare: cere modelului o reflecție scurtă asupra rezultatelor
        reflection = ""
        try:
            summary = json.dumps(
                [{"draw_date": c["entry"]["draw_date"],
                  "draw_numbers": c["entry"]["draw_numbers"],
                  "tickets": [{"main": r["main"], "hits": r["hits"],
                               "why": r["why"]} for r in c["results"]]}
                 for c in checked][:4],
                ensure_ascii=False)[:2500]
            import significance

            record = significance.prospective_ticket_stats(lottery_id)
            record_text = (
                f"Recordul tău REAL cumulat: {record.get('draws_checked', 0)} extrageri verificate, "
                f"{record.get('mean_hits_per_ticket', 0)} lovituri/bilet vs {record.get('baseline', '?')} "
                f"la alegere aleatoare (z={record.get('z', 0)}) → {record.get('verdict_text', '')}."
            )
            answer = ollama_ai.chat(
                f"Biletele propuse pentru {lottery_id} au fost verificate după extragere:\n{summary}\n\n"
                f"{record_text}\n\n"
                "IMPORTANT: o singură extragere este zgomot — nu trage concluzii despre numere „calde” "
                "sau „reci” dintr-un rezultat. Extragerile sunt independente; ce poți controla real este "
                "acoperirea (bilete puțin suprapuse) și evitarea combinațiilor populare. "
                "În 2-3 propoziții: ce concluzie susține recordul CUMULAT și ce vei păstra/schimba? "
                "Răspunde cu text simplu, fără JSON.",
                (ollama_ai.model_candidates(None) or ["gpt-oss:20b-cloud"])[0],
                system="Ești un analist cantitativ care învață din rezultatele propriilor predicții. Fii concis.",
            )
            reflection = answer[:1500]
        except Exception:
            reflection = ""

        for c in checked:
            append_entry(
                "ticket_result",
                lottery_id=lottery_id,
                thought=reflection,
                action=f"Extragerea {c['entry']['draw_date']}: numere câștigătoare "
                       f"{c['entry']['draw_numbers']}; total hits: {c['total_hits']} din "
                       f"{len(c['results'])} bilete",
                why=reflection or "Bilete verificate automat după extragere.",
                payload={"draw_date": c["entry"]["draw_date"],
                         "draw_numbers": c["entry"]["draw_numbers"],
                         "results": c["results"],
                         "total_hits": c["total_hits"]},
            )
            # Telegram: raport complet după FIECARE extragere (hits + câștig estimat)
            self._notify_results(lottery_id, c, reflection)
        self.log_success(
            f"{lottery_id}: {len(checked)} extrageri verificate "
            f"(total {sum(c['total_hits'] for c in checked)} hits)"
        )
        if reflection:
            try:
                overrides = ollama_ai.load_prompt_overrides()
                # doar ultimele 3 lecții (nu acumulăm anecdote din extrageri izolate)
                lessons = [ln for ln in str(overrides.get("ticket_extra", "")).splitlines()
                           if ln.startswith("- ")][-2:]
                lessons.append("- " + " ".join(reflection.split())[:350])
                overrides["ticket_extra"] = "\n".join(lessons)[-1200:]
                ollama_ai.save_prompt_overrides(overrides)
                append_entry("self_improvement", lottery_id=lottery_id,
                             thought=reflection,
                             action="actualizat prompt-overrides (ticket_extra)",
                             why="Biletele viitoare trebuie să țină cont de ce a funcționat.")
            except Exception:
                pass

    def _notify_results(self, lottery_id: str, checked: dict, reflection: str) -> None:
        """Telegram: rezultatele de după extragere — lovituri + câștig estimat."""
        try:
            import telegram_bot

            from lottery_config import get_lottery_config

            entry = checked["entry"]
            config = get_lottery_config(lottery_id)
            results = checked["results"]
            bonus_txt = (f" (+bonus: {' '.join(map(str, entry.get('draw_bonus') or []))})"
                         if entry.get("draw_bonus") else "")
            lines = [
                f"📊 <b>{config.name}</b> — rezultatele biletelor AI pentru "
                f"{entry['draw_date']}{self._slot_label(entry)}",
                f"Numere câștigătoare: <b>{' '.join(f'{n:02d}' for n in entry['draw_numbers'])}</b>{bonus_txt}",
            ]
            best = max(results, key=lambda r: r["hit_count"])
            if best["hit_count"]:
                prize = best.get("prize") or {}
                if prize.get("amount"):
                    est = (f" ({prize['category']} ≈ {prize['amount']:,.0f} "
                           f"{prize['currency']})").replace(",", ".")
                else:
                    est = f" ({prize.get('category', '')})" if prize.get("category") else ""
                lines.append(
                    f"🏆 Cel mai bun bilet: <code>{' '.join(f'{n:02d}' for n in best['main'])}</code>"
                    f" → <b>{best['hit_count']} lovituri</b>{est}"
                )
            for i, r in enumerate(results, 1):
                mark = "✔" if r["hit_count"] else "—"
                lines.append(f"{i}. <code>{' '.join(f'{n:02d}' for n in r['main'])}</code>"
                             f" → {r['hit_count']} {mark}")
            total = entry.get("estimated_prize") or {}
            if total.get("amount"):
                lines.append(f"💰 Total: <b>{checked['total_hits']} lovituri</b> din "
                             f"{len(results)} bilete · câștig estimat ≈ "
                             f"{total['amount']:,.0f} {total['currency']}".replace(",", "."))
            else:
                lines.append(f"💰 Total: <b>{checked['total_hits']} lovituri</b> din "
                             f"{len(results)} bilete · fără premii estimate")
            if reflection:
                lines.append(f"<i>{reflection[:250]}</i>")
            lines.append("<i>Premiile sunt estimări tipice pe categorii, nu sumele "
                         "oficiale ale extragerii.</i>")
            telegram_bot.notify("\n".join(lines))
        except Exception:
            pass

    # ─── main ────────────────────────────────────────────────────────────

    def process(self) -> None:
        try:
            import ollama_ai

            if not ollama_ai.ollama_available():
                self.update_heartbeat("degraded", current_task="ollama unreachable")
                return
        except ImportError:
            return

        from load_data import load_lottery_draws

        did_something = False
        for lottery_id in self.lottery_order:
            try:
                self.check_pending_tickets(lottery_id, ollama_ai)
            except Exception as exc:
                self.log_error(f"{lottery_id}: verificare bilete eșuată: {exc}")
            try:
                if not load_lottery_draws(lottery_id):
                    continue
                self.update_heartbeat("running", current_task=f"{lottery_id}: verific bilete / propun altele noi")
                self.generate_tickets(lottery_id, ollama_ai)
                did_something = True
            except Exception as exc:
                self.log_error(f"{lottery_id}: generare bilete eșuată: {exc}")
        if did_something:
            self.update_heartbeat("running", current_task="bilete actualizate")


def main():
    agent = AITicketAgent()
    agent.run()


if __name__ == "__main__":
    main()
