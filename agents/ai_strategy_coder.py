"""AI Strategy Coder — agentul care îmbunătățește singur sistemul.

Ciclu (per loterie, la `interval_minutes`):
  1. Culege: statistici reale, best strategie curentă, strategiile AI existente
     + scorurile lor, ultimele decizii din Decision Ledger.
  2. Citește 2-3 pagini publice despre matematica loteriilor / strategii
     (cercetare web read-only) și include notele în context.
  3. Cere modelului Ollama să SCRIE O STRATEGIE NOUĂ ca plugin Python
     (ai_strategies/{lottery}/...) cu raționament complet.
  4. Validează plugin-ul (compilare, semnătură, smoke-test), face backtest
     complet și compară cu baseline + best curent.
  5. Dacă e mai bună decât cea mai bună strategie AI promovată, o promovează
     în registry → auto_loop o include la următoarea buclă. Dacă nu, o lasă
     arhivată cu scorul ei, ca material de învățare pentru rulările următoare.

Toate deciziile ajung în Decision Ledger (decision_ledger.py). Modelul își
poate îmbunătăți propriile prompt-uri prin data/ai_self/prompt_overrides.json.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

from agent_base import BaseAgent

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from decision_ledger import append_entry, read_entries
from lottery_config import list_lottery_configs

import ai_strategy_lab

CODER_DIR = ai_strategy_lab.strategies_dir  # helper


class AIStrategyCoderAgent(BaseAgent):
    """Agent care generează, testează și promovează cod de strategii."""

    def __init__(self):
        super().__init__("ai_strategy_coder", poll_interval=60)
        config = self.get_config()
        self.interval_minutes = int(config.get("interval_minutes", 180))
        self.max_attempts = int(config.get("max_generation_attempts", 2))
        default_ids = [c.id for c in list_lottery_configs()]
        configured = config.get("lotteries")
        self.lottery_order = configured if isinstance(configured, list) and configured else default_ids
        self._last_run: dict[str, float] = {}
        self._last_attempt: float = 0.0

    # ─── util ────────────────────────────────────────────────────────────

    def _prediction_is_fresh(self, path: Path, interval_s: float) -> bool:
        if not path.exists():
            return False
        try:
            return (time.time() - path.stat().st_mtime) < interval_s
        except OSError:
            return False

    def _build_context(self, lottery_id: str, ollama_ai) -> dict:
        """Adună tot contextul necesar generării."""
        from load_data import load_lottery_draws

        config = ollama_ai.get_lottery_config(lottery_id)
        draws = load_lottery_draws(lottery_id)
        stats = ollama_ai.compute_lottery_stats(
            config, draws, best=ollama_ai._get_best_strategy_summary(lottery_id)
        )
        baseline = round(config.baseline, 6)
        best_metric = float((stats.get("best_strategy") or {}).get("metric") or baseline)

        # strategiile AI existente + scoruri
        ai_existing = []
        for entry in ai_strategy_lab.load_registry(lottery_id)[-10:]:
            ai_existing.append({
                "key": entry.get("key"),
                "metric": entry.get("metric"),
                "promoted": bool(entry.get("promoted")),
                "description": (entry.get("description") or "")[:200],
            })

        # ultimele decizii relevante din ledger
        recent = read_entries(lottery_id, limit=8)
        recent_summary = [
            {k: e.get(k) for k in ("ts", "type", "thought", "action", "why")}
            for e in recent
        ]

        # un exemplu de plugin deja funcțional (dacă există) ca referință de stil
        example_code = ""
        promoted = ai_strategy_lab.promoted_plugins(lottery_id)
        if promoted:
            example_path = ai_strategy_lab.strategies_dir(lottery_id) / promoted[-1]["file"]
            try:
                example_code = example_path.read_text(encoding="utf-8")[:3000]
            except OSError:
                pass

        return {
            "config": config,
            "stats": stats,
            "baseline": baseline,
            "best_metric": best_metric,
            "ai_existing": ai_existing,
            "recent_summary": recent_summary,
            "example_code": example_code,
        }

    # ─── prompt & generation ─────────────────────────────────────────────

    def _coder_prompt(self, ctx: dict, research_block: str, error_feedback: str | None) -> str:
        stats = ctx["stats"]
        range_max = int(stats["format"].split("/")[1])
        prompt = f"""Ești un cercetător de strategii pentru loteria {stats['name']} ({stats['format']}).
Sarcina: scrii O STRATEGIE NOUĂ, complet diferită de cele existente, ca plugin Python.

STATISTICI REALE:
{json.dumps({k: v for k, v in stats.items() if k != 'best_strategy'}, ensure_ascii=False)[:3500]}

BEST STRATEGIE CURENTĂ (backtest): {json.dumps(stats.get('best_strategy') or {}, ensure_ascii=False)[:500]}
Baseline aleator (average_hits așteptat): {ctx['baseline']}
Best metric de bătut: {ctx['best_metric']}

NOTE DIN CERCETARE WEB:
{research_block[:4000]}

STRATEGII AI EXISTENTE (evită să le duplichezi):
{json.dumps(ctx['ai_existing'], ensure_ascii=False)[:1200]}

ULTIMELE DECIZII (Decision Ledger, cele mai noi primele):
{json.dumps(ctx['recent_summary'], ensure_ascii=False)[:2500]}
"""
        if ctx["example_code"]:
            prompt += f"\nEXEMPLU DE PLUGIN FUNCȚIONAL (doar pentru stil, nu copia logica):\n```python\n{ctx['example_code']}\n```\n"
        if error_feedback:
            prompt += f"\nÎNCERCAREA ANTERIOARĂ A EȘUAT cu eroarea:\n{error_feedback[:800]}\nCorectează problema.\n"

        prompt += f"""
CERINȚE PENTRU STRATEGIE:
- Idei valide: perechi co-apărute, cicluri de întârziere pe sectoare, EWMA pe frecvențe,
  entropie pe fereastra recentă, deviație față de frecvența așteptată, combinări ne-liniare.
- Fii cantitativ; exploatează formule simple, rapide (fără ML greu — backtestul rulează mii de ori).
- NU folosi biblioteci în afara stdlib + numpy.
- rank trebuie să returneze o permutare completă a numerelor 1..{range_max}.

ȘABLON OBLIGATORIU — copiază exact această interfață (semnăturile NU se schimbă):
```python
NAME = "znume_scurt"
DESCRIPTION = "ce face strategia"
SOURCE = "model + dată"

def create(config):
    # config: LotteryConfig — are .number_range (ex. 49), .numbers_drawn (ex. 6), .ticket_size
    # NU itera peste config și NU-l converti în dict; folosește doar atributele de mai sus.
    def rank(counts, recent_deque, last_seen, draw_index, recent_window):
        # counts: dict {{număr: apariții totale în tot istoricul}}
        # recent_deque: collections.deque de set() — numerele din ultimele `recent_window` extrageri
        # last_seen: dict {{număr: index-ul extragerii în care a apărut ultima dată (lipsă = niciodată)}}
        # draw_index: index-ul întreg al extragerii curente
        # recent_window: întreg — dimensiunea ferestrei recente
        numbers = list(range(1, config.number_range + 1))
        scores = {{n: counts.get(n, 0) for n in numbers}}
        return sorted(numbers, key=lambda n: (-scores[n], n))
    return rank
```

Răspunde DOAR cu JSON:
{{
  "thoughts": "raționamentul tău pas cu pas înainte de cod",
  "name": "znume_scurt (doar litere mici/cifre)",
  "description": "ce face strategia, în 2-3 propoziții",
  "why": "de ce ai motive să bată baseline-ul; ce ipoteză testezi",
  "code": "FIȘIERUL PYTHON COMPLET conform șablonului: NAME, DESCRIPTION, SOURCE + def create(config) -> rank(counts, recent_deque, last_seen, draw_index, recent_window)",
  "self_improvement": "opțional: 1-2 propoziții cum să-ți ajustezi propriile prompt-uri viitoare (sau gol)"
}}"""
        return prompt

    CODER_SYSTEM = (
        "Ești un inginer cantitativ senior care scrie plugin-uri Python pentru un sistem "
        "de optimizare a strategiilor de loterie. Codul trebuie să fie corect, rapid și "
        "determinist. Răspunzi EXCLUSIV cu JSON valid."
    )

    def _coder_system_prompt(self) -> str:
        import ollama_ai

        base = self.CODER_SYSTEM
        extra = ollama_ai.load_prompt_overrides().get("coder_extra", "")
        if extra:
            base += "\n\nINSTRUCȚIUNI SUPLEMENTARE (auto-învățare): " + str(extra)[:1500]
        return base

    # ─── generarea + evaluarea unei strategii ────────────────────────────

    def _generate_and_evaluate(self, lottery_id: str, ollama_ai, model: str | None) -> None:
        ctx = self._build_context(lottery_id, ollama_ai)

        research_block, used_urls = ollama_ai.build_research_block()
        if used_urls:
            append_entry(
                "web_research",
                lottery_id=lottery_id,
                thought=f"Am citit {len(used_urls)} surse publice pentru idei de strategii.",
                action="; ".join(used_urls),
                why="Context nou de idei > ce pot genera doar din statistici locale.",
                payload={"urls": used_urls},
            )

        error_feedback = None
        for attempt in range(1, self.max_attempts + 1):
            self.update_heartbeat("running", current_task=f"{lottery_id}: AI scrie strategia (încercarea {attempt})")
            prompt = self._coder_prompt(ctx, research_block, error_feedback)
            try:
                candidates = ollama_ai.model_candidates(model)
                answer = None
                for candidate in candidates:
                    try:
                        answer = ollama_ai.chat(prompt, candidate, system=self._coder_system_prompt())
                        break
                    except Exception as exc:
                        answer = None
                        continue
                if not answer:
                    raise RuntimeError("modelul nu a răspuns")
                parsed = ollama_ai.extract_json(answer)
            except Exception as exc:
                error_feedback = str(exc)
                append_entry("strategy_code", lottery_id=lottery_id,
                             thought=f"Generare eșuată (încercarea {attempt}): {error_feedback[:300]}",
                             action="none", why="model/timeout", status="failed")
                continue
            if not parsed:
                error_feedback = "răspunsul nu a fost JSON valid"
                continue

            code = str(parsed.get("code", "")).strip()
            thoughts = str(parsed.get("thoughts", ""))[:3000]
            why = str(parsed.get("why", ""))[:2000]
            description = str(parsed.get("description", ""))[:500]
            suggested_name = str(parsed.get("name", ""))

            # curăță eventuale garduri markdown
            if code.startswith("```"):
                code = code.strip("`")
                if code.startswith("python"):
                    code = code[6:]
                code = code.strip()

            key = ai_strategy_lab.unique_key(lottery_id, suggested_name)
            file_name = f"{key}_{time.strftime('%Y%m%d_%H%M%S')}.py"
            path = ai_strategy_lab.strategies_dir(lottery_id) / file_name
            path.parent.mkdir(parents=True, exist_ok=True)

            # Auto-corecție: forțează NAME din cod să fie chiar cheia unică "z..."
            # (modelele uită uneori prefixul; registry-ul are nevoie de consistență)
            import re as _re

            code = _re.sub(
                r'(?m)^(\s*NAME\s*=\s*)["\'][^"\']*["\']',
                f'\\1"{key}"',
                code,
            )
            path.write_text(code, encoding="utf-8")

            # validare
            try:
                meta = ai_strategy_lab.validate_plugin(path, lottery_id)
            except Exception as exc:
                error_feedback = f"validare eșuată: {exc}"
                append_entry("strategy_code", lottery_id=lottery_id,
                             thought=thoughts, action=f"plugin respins la validare: {path.name}",
                             why=why, status="failed",
                             payload={"error": str(exc)[:500]})
                continue

            # backtest complet
            from load_data import load_lottery_draws

            draws = load_lottery_draws(lottery_id)
            self.update_heartbeat("running", current_task=f"{lottery_id}: backtest {meta['name']}")
            try:
                result = ai_strategy_lab.backtest_plugin(path, lottery_id, draws)
            except Exception as exc:
                error_feedback = f"backtest eșuat: {exc}"
                append_entry("strategy_code", lottery_id=lottery_id,
                             thought=thoughts, action=f"plugin respins la backtest: {path.name}",
                             why=why, status="failed",
                             payload={"error": str(exc)[:500]})
                continue

            metric = result["average_hits"]
            beats_best = metric > max(ctx["best_metric"], ctx["baseline"])
            # Poarta de realitate: best-ul e deja maximul multor încercări, deci
            # un plugin care îl „bate" poate fi tot zgomot. Cerem în plus:
            #  - peste baseline și pe holdout (ultimele 25% extrageri);
            #  - edge distinct de zgomot după corecția pentru plugin-urile testate.
            import significance

            n_tested = max(1, len(list(ai_strategy_lab.strategies_dir(lottery_id).glob("*.py"))))
            edge = significance.config_edge_stats(ctx["config"], metric, max(len(draws) - 30, 1), n_tested)
            try:
                holdout = ai_strategy_lab.backtest_plugin_holdout(path, lottery_id, draws)
            except Exception as exc:
                holdout = {"average_hits": 0.0, "error": str(exc)[:200]}
            require_sig = bool(self.get_config().get("promotion_requires_significance", True))
            passes_reality = holdout.get("average_hits", 0.0) > ctx["baseline"] and (
                edge["verdict"] != "noise" or not require_sig
            )
            promoted = beats_best and passes_reality
            result["holdout"] = holdout
            result["significance"] = {k: edge[k] for k in ("z", "p_adjusted", "n_experiments", "verdict")}
            if promoted:
                entry = ai_strategy_lab.promote_plugin(
                    lottery_id, file_name, key, metric, ctx["baseline"],
                    description=description, model=meta.get("source", ""),
                )
                action = (f"STRATEGIE PROMOVATĂ: {key} ({path.name}) cu average_hits={metric} "
                          f"> best {ctx['best_metric']:.6f}; intră în căutarea auto_loop la următoarea buclă")
            elif beats_best:
                action = (f"Strategie respinsă de poarta de realitate: {key} ({path.name}) average_hits={metric} "
                          f"bate best-ul, dar holdout={holdout.get('average_hits')} (baseline {ctx['baseline']}), "
                          f"z={edge['z']:+.2f}, p_ajustat={edge['p_adjusted']:.3f} după {n_tested} plugin-uri → "
                          f"{edge['verdict_text']}")
            else:
                action = (f"Strategie arhivată: {key} ({path.name}) average_hits={metric} "
                          f"(best rămâne {ctx['best_metric']:.6f}, baseline {ctx['baseline']})")

            append_entry("strategy_code", lottery_id=lottery_id,
                         thought=thoughts, action=action, why=why,
                         payload={"file": file_name, "key": key,
                                  "metrics": result, "promoted": promoted,
                                  "model": meta.get("source", "")},
                         status="promoted" if promoted else "tested")

            # utilizatorul primește pe Telegram vestea unei strategii promovate
            if promoted:
                try:
                    import ai_actions as _bus

                    _bus.notify_event(
                        f"🎉 <b>AI a creat o strategie mai bună!</b>\n"
                        f"Loterie: {lottery_id}\n"
                        f"Strategie: <code>{key}</code> ({description[:120]})\n"
                        f"average_hits: <b>{metric}</b> &gt; best anterior {ctx['best_metric']:.6f}\n"
                        f"Intră în căutarea auto-loop la următoarea rotație de buclă."
                    )
                except Exception:
                    pass

            # self-improvement: modelul își poate ajusta prompt-urile
            improvement = str(parsed.get("self_improvement", "")).strip()
            if improvement:
                overrides = ollama_ai.load_prompt_overrides()
                overrides["coder_extra"] = (overrides.get("coder_extra", "") + "\n- " + improvement).strip()[-1500:]
                ollama_ai.save_prompt_overrides(overrides)
                append_entry("self_improvement", lottery_id=lottery_id,
                             thought=improvement,
                             action="actualizat data/ai_self/prompt_overrides.json (coder_extra)",
                             why="Învățare din propriile rezultate pentru generațiile următoare.")
            return

    # ─── main loop integration ───────────────────────────────────────────

    def process(self) -> None:
        if time.time() - self._last_attempt < 300:
            return

        interval_s = self.interval_minutes * 60
        now = time.time()
        due = [
            lottery_id for lottery_id in self.lottery_order
            if now - self._last_run.get(lottery_id, 0) >= interval_s
            and not self._prediction_is_fresh(
                ai_strategy_lab.strategies_dir(lottery_id) / ".last_generation", interval_s
            )
        ]
        if not due:
            self.update_heartbeat("idle", current_task="așteaptă următorul ciclu de generare")
            return

        self._last_attempt = time.time()

        try:
            import ollama_ai

            if not ollama_ai.ollama_available():
                self.update_heartbeat("degraded", current_task="ollama unreachable")
                return

            from load_data import load_lottery_draws

            ran_any = False
            for lottery_id in due:
                try:
                    draws = load_lottery_draws(lottery_id)
                except Exception:
                    draws = []
                if not draws:
                    continue
                self.update_heartbeat(
                    "running",
                    current_task=f"AI coder: creez o strategie nouă pentru {lottery_id}",
                )
                try:
                    self._generate_and_evaluate(lottery_id, ollama_ai, model=None)
                    marker = ai_strategy_lab.strategies_dir(lottery_id) / ".last_generation"
                    marker.parent.mkdir(parents=True, exist_ok=True)
                    marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
                    self._last_run[lottery_id] = time.time()
                    ran_any = True
                except Exception as exc:
                    self.log_error(f"AI coder failed for {lottery_id}: {exc}")
                    append_entry("strategy_code", lottery_id=lottery_id,
                                 thought="", action=f"eroare agent: {str(exc)[:300]}",
                                 why="excepție neașteptată", status="failed")
            if ran_any:
                self.update_heartbeat("running", current_task="ciclu de generare terminat")
        except ImportError as exc:
            self.log_error(f"module unavailable: {exc}")


def main():
    agent = AIStrategyCoderAgent()
    agent.run()


if __name__ == "__main__":
    main()
