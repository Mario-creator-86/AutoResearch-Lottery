"""Daily Ticket Generator Agent

Acest agent:
- Generează bilete zilnice pentru loteriile active (ora 8:00)
- Verifică rezultatele extragerilor anterioare
- Propune îmbunătățiri bazate pe performanța biletelor
- Trimite rapoarte zilnice

Schedule: Zilnic la 8:00 AM (generare) + după fiecare extragere (verificare)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from agent_base import BaseAgent, AGENTS_DIR

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from lottery_config import list_lottery_configs, get_lottery_config, LotteryConfig
from load_data import load_lottery_draws
from generic_backtest import backtest_strategy, make_weighted_ensemble
from atomic_io import read_json_with_recovery, write_json_atomic

try:
    from telegram_bot import send_message as send_telegram_message
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

# Directoare
TICKETS_DIR = AGENTS_DIR / "tickets"
TODAY_DIR = TICKETS_DIR / "today"
RESULTS_DIR = TICKETS_DIR / "results"
TODAY_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class TicketGeneratorAgent(BaseAgent):
    """Agent pentru generarea biletelor zilnice și verificarea câștigurilor."""

    # Orele la care au loc extragerile pentru fiecare loterie (configurabil)
    DRAW_TIMES = {
        "multi_multi_pl": ["14:00", "21:40"],  # Multi Multi are 2 extrageri/zi
        "lotto_pl": ["21:40"],
        "mini_lotto_pl": ["21:40"],
        "kaskada_pl": ["14:00", "21:40"],
        "loto_6_49_ro": ["18:00"],
        "loto_5_40_ro": ["15:00"],
        "joker_ro": ["18:00"],
        "eurojackpot": ["21:00"],
        "powerball_us": [["04:00"], ["04:00"]],  # America (timezone)
        "mega_millions_us": [["04:00"]],
        "uk_lotto": ["21:00"],
        "euromillions": ["21:45"]
    }

    def __init__(self):
        super().__init__("ticket_generator", poll_interval=300)  # 5 minute
        self.config = self.get_config()
        self.tickets_per_lottery = self.config.get("tickets_per_lottery", 5)
        self.loss_threshold = self.config.get("consecutive_losses_threshold", 7)

        self.last_generation_date = None
        self.generated_tickets: dict[str, list] = {}

    def _get_best_strategy(self, lottery_id: str) -> dict | None:
        """Încarcă cea mai bună strategie din results.tsv."""
        config = get_lottery_config(lottery_id)
        results_file = config.results_file

        try:
            if not results_file.exists():
                return None

            with open(results_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if not lines:
                return None

            # Găsește cea mai bună metrică
            best_line = None
            best_metric = 0.0

            for line in lines:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    try:
                        metric = float(parts[1])
                        if metric > best_metric:
                            best_metric = metric
                            best_line = parts
                    except ValueError:
                        continue

            if best_line:
                return {
                    "metric": best_metric,
                    "status": best_line[2] if len(best_line) > 2 else "unknown",
                    "strategy": best_line[3] if len(best_line) > 3 else "default",
                    "config": best_line
                }

        except Exception as e:
            self.log_error(f"Failed to load best strategy for {lottery_id}: {e}")

        return None

    def _generate_tickets_for_lottery(self, lottery_id: str, count: int = 5) -> list[dict]:
        """Generează bilete pentru o loterie specifică."""
        config = get_lottery_config(lottery_id)

        try:
            # Încarcă datele istorice
            draws = load_lottery_draws(lottery_id)
            if not draws:
                self.log_error(f"No data available for {lottery_id}")
                return []

            # Folosește cea mai bună strategie sau default
            strategy_info = self._get_best_strategy(lottery_id)

            # Generează biletele
            tickets = []

            # Simplified ticket generation using backtest strategy
            for i in range(count):
                # Aici ar trebui să folosim strategia efectivă
                # Pentru moment generăm bazat pe frecvență
                ticket = self._generate_single_ticket(config, draws)
                tickets.append({
                    "numbers": ticket,
                    "generated_at": datetime.utcnow().isoformat(),
                    "strategy_used": strategy_info.get("strategy", "default") if strategy_info else "default"
                })

            return tickets

        except Exception as e:
            self.log_error(f"Failed to generate tickets for {lottery_id}: {e}")
            return []

    def _generate_single_ticket(self, config: LotteryConfig, draws: list) -> list[int]:
        """Generează un singur bilet bazat pe frecvență."""
        from collections import Counter

        # Extrage toate numerele din extrageri recente (ultimele 100)
        recent_draws = draws[-100:] if len(draws) > 100 else draws

        all_numbers = []
        for draw in recent_draws:
            all_numbers.extend(draw.numbers)

        # Numără frecvența
        frequency = Counter(all_numbers)

        # Selectează cele mai frecvente numere
        most_common = frequency.most_common(config.ticket_size)

        # Adaugă puțină diversitate random
        ticket = [num for num, _ in most_common[:config.ticket_size]]

        # Completează până la ticket_size dacă e necesar
        while len(ticket) < config.ticket_size:
            import random
            num = random.randint(1, config.number_range)
            if num not in ticket:
                ticket.append(num)

        ticket.sort()
        return ticket

    def _save_tickets(self, lottery_id: str, tickets: list[dict]) -> bool:
        """Salvează biletele generate."""
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        ticket_file = TODAY_DIR / f"{lottery_id}_{date_str}.json"

        data = {
            "lottery": lottery_id,
            "date": date_str,
            "generated_at": datetime.utcnow().isoformat(),
            "ticket_count": len(tickets),
            "tickets": tickets
        }

        try:
            write_json_atomic(ticket_file, data)
            self.log_success(f"Saved {len(tickets)} tickets for {lottery_id}")
            return True
        except Exception as e:
            self.log_error(f"Failed to save tickets: {e}")
            return False

    def _check_draw_results(self) -> dict[str, Any]:
        """Verifică rezultatele extragerilor recente."""
        # TODO: Implementare reală cu scraping
        # Deocamdată returnăm un placeholder

        checked_lotteries = {}

        for lottery_id in self.DRAW_TIMES:
            # Verifică dacă e timpul pentru verificare
            if self._should_check_results(lottery_id):
                self.log_info(f"Checking results for {lottery_id}")
                # Aici ar trebui implementat scraping-ul real
                checked_lotteries[lottery_id] = "pending"

        return checked_lotteries

    def _should_check_results(self, lottery_id: str) -> bool:
        """Determină dacă ar trebui să verificăm rezultatele pentru această loterie."""
        now = datetime.now()
        draw_times = self.DRAW_TIMES.get(lottery_id, [])

        for draw_time in draw_times:
            if isinstance(draw_time, list):
                # American lotteries (timezone conversion needed)
                continue

            try:
                hour, minute = map(int, draw_time.split(":"))
                draw_datetime = now.replace(hour=hour, minute=minute, second=0)

                # Verifică dacă extragerea a avut loc în ultima oră
                time_diff = (now - draw_datetime).total_seconds()
                if 0 < time_diff < 3600:  # Între 0 și 60 minute după extragere
                    return True
            except Exception:
                continue

        return False

    def _check_tickets_against_results(self, lottery_id: str,
                                        winning_numbers: list) -> dict:
        """Verifică biletele generate împotriva numerelor câștigătoare."""
        date_str = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
        ticket_file = TODAY_DIR / f"{lottery_id}_{date_str}.json"

        if not ticket_file.exists():
            return {"error": "No tickets found for yesterday"}

        try:
            with open(ticket_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            tickets = data.get("tickets", [])
            results = []

            for ticket in tickets:
                numbers = ticket.get("numbers", [])
                matches = len(set(numbers) & set(winning_numbers))
                results.append({
                    "ticket": numbers,
                    "matches": matches,
                    "win": matches >= 2  # Considerăm câștig la 2+ numere
                })

            # Salvează rezultatele
            result_data = {
                "lottery": lottery_id,
                "date": date_str,
                "winning_numbers": winning_numbers,
                "checked_at": datetime.utcnow().isoformat(),
                "results": results,
                "total_tickets": len(tickets),
                "winning_tickets": sum(1 for r in results if r["win"])
            }

            result_file = RESULTS_DIR / f"{lottery_id}_{date_str}_results.json"
            write_json_atomic(result_file, result_data)

            return result_data

        except Exception as e:
            self.log_error(f"Failed to check tickets: {e}")
            return {"error": str(e)}

    def _analyze_performance(self, lottery_id: str) -> dict:
        """Analizează performanța ultimelor bilete."""
        # Încarcă istoricul rezultatelor
        results = []

        for result_file in RESULTS_DIR.glob(f"{lottery_id}_*_results.json"):
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    results.append(data)
            except Exception:
                continue

        if not results:
            return {"status": "no_data"}

        # Sortează după dată
        results.sort(key=lambda x: x.get("date", ""))

        # Calculează statistici
        total_tickets = sum(r.get("total_tickets", 0) for r in results)
        winning_tickets = sum(r.get("winning_tickets", 0) for r in results)
        consecutive_losses = 0

        for r in reversed(results):
            if r.get("winning_tickets", 0) == 0:
                consecutive_losses += 1
            else:
                break

        return {
            "total_tickets": total_tickets,
            "winning_tickets": winning_tickets,
            "win_rate": winning_tickets / total_tickets if total_tickets > 0 else 0,
            "consecutive_losses": consecutive_losses,
            "suggest_strategy_change": consecutive_losses >= self.loss_threshold
        }

    def _propose_improvements(self) -> list[dict]:
        """Propune îmbunătățiri bazate pe performanță."""
        proposals = []

        for config in list_lottery_configs():
            lottery_id = config.id
            performance = self._analyze_performance(lottery_id)

            if performance.get("suggest_strategy_change"):
                proposal = {
                    "lottery": lottery_id,
                    "reason": f"{performance['consecutive_losses']} consecutive losses",
                    "suggested_action": "try_new_strategy",
                    "priority": "high" if performance["consecutive_losses"] > 10 else "medium"
                }
                proposals.append(proposal)

                # Adaugă task în queue pentru lottery_manager
                self.send_message("lottery_manager", "add_task", {
                    "lottery": lottery_id,
                    "task_type": "experiment",
                    "priority": 7,
                    "params": {"reason": "consecutive_losses", "count": performance["consecutive_losses"]}
                })

        return proposals

    def _send_daily_report(self) -> None:
        """Trimite raportul zilnic."""
        if not TELEGRAM_AVAILABLE:
            return

        report_lines = ["📊 <b>Daily Lottery Report</b>\n"]

        for config in list_lottery_configs():
            lottery_id = config.id
            performance = self._analyze_performance(lottery_id)

            if performance.get("status") != "no_data":
                report_lines.append(
                    f"\n<b>{lottery_id}</b>: "
                    f"{performance['winning_tickets']}/{performance['total_tickets']} câștiguri "
                    f"({performance['win_rate']:.1%})"
                )

                if performance["suggest_strategy_change"]:
                    report_lines.append(f"⚠️ {performance['consecutive_losses']} pierderi consecutive!")

        try:
            send_telegram_message("\n".join(report_lines))
        except Exception as e:
            self.log_error(f"Failed to send report: {e}")

    def _is_generation_time(self) -> bool:
        """Verifică dacă e timpul să generăm bilete (8:00 AM)."""
        now = datetime.now()
        return now.hour == 8 and now.minute < 5

    def _should_send_report(self) -> bool:
        """Verifică dacă ar trebui să trimitem raportul (22:00)."""
        now = datetime.now()
        return now.hour == 22 and now.minute < 5

    def process(self) -> None:
        """Procesare principală."""
        current_date = datetime.utcnow().date()

        # 1. Generează bilete noi (8:00 AM)
        if self._is_generation_time() and current_date != self.last_generation_date:
            self.log_info("Starting daily ticket generation...")
            self.update_heartbeat("running", current_task="generating_tickets", progress_percent=10)

            total_generated = 0
            configs = list_lottery_configs()

            for i, config in enumerate(configs):
                tickets = self._generate_tickets_for_lottery(config.id, self.tickets_per_lottery)
                if tickets:
                    self._save_tickets(config.id, tickets)
                    total_generated += len(tickets)

                progress = int((i + 1) / len(configs) * 90)
                self.update_heartbeat("running", current_task=f"generating for {config.id}",
                                      progress_percent=progress)

            self.last_generation_date = current_date
            self.log_success(f"Generated {total_generated} tickets for {len(configs)} lotteries")

            self.update_heartbeat("idle", current_task="waiting for results", progress_percent=100)

        # 2. Verifică rezultate extrageri
        self._check_draw_results()

        # 3. Trimite raport zilnic (22:00)
        if self._should_send_report():
            self._send_daily_report()

        # 4. Analizează performanță și propune îmbunătățiri
        if int(time.time()) % 3600 < 300:  # O dată pe oră
            proposals = self._propose_improvements()
            if proposals:
                self.log_info(f"Proposed {len(proposals)} strategy changes based on performance")


def main():
    """Pornește Ticket Generator Agent."""
    agent = TicketGeneratorAgent()
    agent.run()


if __name__ == "__main__":
    main()
