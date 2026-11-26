"""Strategy Research Agent

Acest agent caută strategii noi de loterie din surse online:
- Forumuri: lotterypost.com, reddit r/lottery, forumuri loto românești
- Site-uri statistici: lottery analysis websites
- Research papers: arXiv, Google Scholar

Schedule: La fiecare 6 ore (00:00, 06:00, 12:00, 18:00)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_base import BaseAgent, AGENTS_DIR

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# Director pentru strategii descoperite
STRATEGIES_DIR = AGENTS_DIR / "strategies"
STRATEGIES_DIR.mkdir(exist_ok=True)


class StrategyResearcherAgent(BaseAgent):
    """Agent pentru descoperirea strategiilor de loterie din surse online."""

    # Surse de încredere pentru cercetare
    RESEARCH_SOURCES = {
        "forums": [
            "https://www.lotterypost.com/search",
            "https://www.reddit.com/r/lottery/search",
            "https://www.reddit.com/r/Lotto/search"
        ],
        "statistics": [
            "lottery statistics analysis frequency",
            "lottery number patterns hot cold analysis",
            "wheeling systems lottery strategies"
        ],
        "academic": [
            "arxiv.org lottery prediction statistical analysis",
            "random number generation patterns"
        ]
    }

    def __init__(self):
        super().__init__("strategy_researcher", poll_interval=60)
        self.last_research_time = 0
        self.research_interval = 6 * 3600  # 6 ore în secunde

    def _extract_strategy_from_text(self, text: str, source: str) -> dict | None:
        """Extrage o strategie din text folosind pattern matching."""
        strategy = None

        # Pattern pentru Gap Analysis
        if any(kw in text.lower() for kw in ["gap", "missed", "not drawn", "overdue", "absent"]):
            gap_match = self._extract_gap_pattern(text)
            if gap_match:
                strategy = {
                    "type": "gap_analysis",
                    "name": f"Gap Analysis - {gap_match['min_gap']}+ games",
                    "description": f"Select numbers not drawn in {gap_match['min_gap']}+ consecutive games",
                    "parameters": gap_match,
                    "confidence": self._assess_confidence(text, source),
                    "complexity": "medium"
                }

        # Pattern pentru Hot/Cold
        elif any(kw in text.lower() for kw in ["hot numbers", "cold numbers", "frequency", "often drawn"]):
            freq_match = self._extract_frequency_pattern(text)
            if freq_match:
                strategy = {
                    "type": "frequency_weighted",
                    "name": "Hot/Cold Weighted Frequency",
                    "description": "Weight numbers by recent frequency and gap since last appearance",
                    "parameters": freq_match,
                    "confidence": self._assess_confidence(text, source),
                    "complexity": "low"
                }

        # Pattern pentru Wheeling Systems
        elif any(kw in text.lower() for kw in ["wheeling system", "wheel", "combinations"]):
            wheel_match = self._extract_wheel_pattern(text)
            if wheel_match:
                strategy = {
                    "type": "wheeling_system",
                    "name": f"Wheel {wheel_match.get('numbers', 'N')}/{wheel_match.get('guarantee', 'N')}",
                    "description": f"Wheeling system covering {wheel_match.get('numbers', '?')} numbers",
                    "parameters": wheel_match,
                    "confidence": self._assess_confidence(text, source),
                    "complexity": "high"
                }

        # Pattern pentru Delta System
        elif any(kw in text.lower() for kw in ["delta", "difference", "spacing"]):
            delta_match = self._extract_delta_pattern(text)
            if delta_match:
                strategy = {
                    "type": "delta_system",
                    "name": "Delta Number Spacing",
                    "description": "Select numbers based on spacing patterns between consecutive draws",
                    "parameters": delta_match,
                    "confidence": self._assess_confidence(text, source),
                    "complexity": "medium"
                }

        # Pattern pentru Sum Analysis
        elif any(kw in text.lower() for kw in ["sum", "total", "addition"]):
            sum_match = self._extract_sum_pattern(text)
            if sum_match:
                strategy = {
                    "type": "sum_analysis",
                    "name": f"Sum Range {sum_match.get('min_sum', '?')}-{sum_match.get('max_sum', '?')}",
                    "description": "Select combinations whose sum falls within optimal range",
                    "parameters": sum_match,
                    "confidence": self._assess_confidence(text, source),
                    "complexity": "low"
                }

        # Pattern pentru Odd/Even Balance
        elif any(kw in text.lower() for kw in ["odd even balance", "parity", "mixed"]):
            balance_match = self._extract_balance_pattern(text)
            if balance_match:
                strategy = {
                    "type": "parity_balance",
                    "name": f"Odd/Even {balance_match.get('odd', '?')}/{balance_match.get('even', '?')}",
                    "description": "Maintain specific odd/even number ratio in selections",
                    "parameters": balance_match,
                    "confidence": self._assess_confidence(text, source),
                    "complexity": "low"
                }

        return strategy

    def _extract_gap_pattern(self, text: str) -> dict | None:
        """Extrage parametri pentru Gap Analysis."""
        import re

        # Caută numere în text (ex: "30 games", "15 draws")
        numbers = re.findall(r'(\d+)\s*(?:games|draws|times|consecutive)', text.lower())
        if numbers:
            min_gap = int(numbers[0])
            max_gap = int(numbers[1]) if len(numbers) > 1 else min_gap + 30
            return {"min_gap": min_gap, "max_gap": max_gap, "weight": 1.0}

        # Default dacă găsim cuvinte cheie
        if "overdue" in text.lower() or "long absent" in text.lower():
            return {"min_gap": 20, "max_gap": 50, "weight": 1.0}

        return None

    def _extract_frequency_pattern(self, text: str) -> dict | None:
        """Extrage parametri pentru Frequency Analysis."""
        import re

        # Caută mențiuni de frecvență
        hot_window = 20  # default
        cold_threshold = 10  # default

        matches = re.findall(r'(\d+)\s*(?:draws|games|days|weeks)', text.lower())
        if matches:
            hot_window = int(matches[0])
            if len(matches) > 1:
                cold_threshold = int(matches[1])

        return {
            "hot_window": hot_window,
            "cold_threshold": cold_threshold,
            "frequency_weight": 0.6,
            "recency_weight": 0.4
        }

    def _extract_wheel_pattern(self, text: str) -> dict | None:
        """Extrage parametri pentru Wheeling Systems."""
        import re

        # Pattern: "wheel 20 numbers" sau "full wheel 10"
        matches = re.findall(r'wheel\s+(\d+)', text.lower())
        if matches:
            numbers = int(matches[0])
            return {
                "numbers": numbers,
                "guarantee": 3,  # default
                "combinations": numbers * (numbers - 1) // 2
            }

        return None

    def _extract_delta_pattern(self, text: str) -> dict | None:
        """Extrage parametri pentru Delta System."""
        import re

        # Caută range pentru diferențe
        matches = re.findall(r'(\d+)\s*(?:to|-)\s*(\d+)', text)
        if matches:
            return {
                "min_delta": int(matches[0][0]),
                "max_delta": int(matches[0][1]),
                "target_sum_delta": (int(matches[0][0]) + int(matches[0][1])) // 2
            }

        return {"min_delta": 1, "max_delta": 10, "target_sum_delta": 5}

    def _extract_sum_pattern(self, text: str) -> dict | None:
        """Extrage parametri pentru Sum Analysis."""
        import re

        # Caută sume în text
        matches = re.findall(r'(\d{3,4})\s*(?:to|-)\s*(\d{3,4})', text)
        if matches:
            return {
                "min_sum": int(matches[0][0]),
                "max_sum": int(matches[0][1]),
                "target_sum": (int(matches[0][0]) + int(matches[0][1])) // 2
            }

        return None

    def _extract_balance_pattern(self, text: str) -> dict | None:
        """Extrage parametri pentru Odd/Even Balance."""
        import re

        # Caută pattern-uri ca "3 odd 3 even" sau "2/4 split"
        matches = re.findall(r'(\d+)\s*(?:odd|even)', text.lower())
        if len(matches) >= 2:
            return {
                "odd": int(matches[0]),
                "even": int(matches[1]),
                "ratio": int(matches[0]) / (int(matches[1]) if int(matches[1]) > 0 else 1)
            }

        return {"odd": 3, "even": 3, "ratio": 1.0}

    def _assess_confidence(self, text: str, source: str) -> str:
        """Evaluează încrederea în strategie."""
        confidence = "low"

        # Crește încrederea pentru surse academice
        if "arxiv" in source or "scholar" in source:
            confidence = "high"
        elif "lotterypost" in source or "forum" in source:
            confidence = "medium"

        # Crește pentru texte cu detalii numerice
        import re
        if re.search(r'\d+\s*(?:%|percent|draws|games)', text):
            confidence = "medium" if confidence == "low" else "high"

        # Crește pentru texte lungi cu exemple
        if len(text) > 500 and ("example" in text.lower() or "test" in text.lower()):
            confidence = "high"

        return confidence

    def _save_strategy(self, strategy: dict, source: str) -> bool:
        """Salvează strategia descoperită."""
        strategy_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "source": source,
            "status": "pending_review",
            "reviewed_by": None,
            "review_notes": None,
            "strategy": strategy
        }

        # Încarcă strategii existente
        pending_file = STRATEGIES_DIR / "pending_review.json"
        try:
            if pending_file.exists():
                with open(pending_file, "r", encoding="utf-8") as f:
                    strategies = json.load(f)
            else:
                strategies = []
        except Exception:
            strategies = []

        # Verifică duplicat (după tip + parametri cheie)
        for existing in strategies:
            existing_strategy = existing.get("strategy", {})
            if (existing_strategy.get("type") == strategy.get("type") and
                existing_strategy.get("name") == strategy.get("name")):
                self.log_info(f"Duplicate strategy found: {strategy.get('name')}, skipping")
                return False

        # Adaugă strategia nouă
        strategies.append(strategy_record)

        # Salvează
        try:
            with open(pending_file, "w", encoding="utf-8") as f:
                json.dump(strategies, f, indent=2)

            self.log_success(f"New strategy saved: {strategy.get('name', 'Unknown')}")
            return True
        except Exception as e:
            self.log_error(f"Failed to save strategy: {e}")
            return False

    def _simulate_research(self) -> list[dict]:
        """Simulează o căutare (pentru demo/test)."""
        # Aici ar trebui să folosim Firecrawl pentru căutare reală
        # Pentru moment, generăm strategii simulate bazate pe config

        config = self.get_config()
        discovered = []

        # Strategii template bazate pe config
        templates = [
            {
                "type": "gap_analysis",
                "name": "Long Absent Numbers Strategy",
                "description": "Focus on numbers not drawn in 25+ consecutive games",
                "parameters": {"min_gap": 25, "max_gap": 60, "weight": 1.2},
                "confidence": "medium",
                "complexity": "medium"
            },
            {
                "type": "frequency_weighted",
                "name": "Recent Frequency Boost",
                "description": "Weight recent draws more heavily in frequency calculation",
                "parameters": {"hot_window": 15, "recency_decay": 0.9, "boost_factor": 1.5},
                "confidence": "high",
                "complexity": "low"
            },
            {
                "type": "clustering",
                "name": "Number Cluster Detection",
                "description": "Identify clusters of frequently co-occurring numbers",
                "parameters": {"min_cluster_size": 3, "min_cooccurrence": 5, "lookback": 100},
                "confidence": "medium",
                "complexity": "high"
            },
            {
                "type": "sum_analysis",
                "name": "Optimal Sum Range",
                "description": "Target combinations with sum in 180-220 range",
                "parameters": {"min_sum": 180, "max_sum": 220, "target": 200},
                "confidence": "low",
                "complexity": "low"
            }
        ]

        return templates

    def _perform_research(self) -> int:
        """Execută căutarea de strategii."""
        self.log_info("Starting strategy research...")
        self.update_heartbeat("running", current_task="researching_strategies", progress_percent=10)

        discovered_count = 0

        # TODO: Aici ar trebui integrare cu Firecrawl skill pentru căutare reală
        # Deocamdată folosim simulare

        strategies = self._simulate_research()

        self.update_heartbeat("running", current_task="analyzing_results", progress_percent=50)

        for strategy in strategies:
            if self._save_strategy(strategy, "simulated_research"):
                discovered_count += 1

        self.update_heartbeat("running", current_task="complete", progress_percent=100)

        self.log_success(f"Research complete. Discovered {discovered_count} new strategies.")

        if discovered_count > 0:
            # Notifică prin Telegram
            try:
                from telegram_bot import send_message
                send_message(f"🔍 <b>Strategy Researcher</b>\n{discovered_count} strategii noi descoperite!")
            except Exception:
                pass

        return discovered_count

    def process(self) -> None:
        """Procesare principală - verifică dacă e timpul pentru cercetare."""
        current_time = time.time()

        # Verifică dacă a trecut intervalul de cercetare
        if current_time - self.last_research_time > self.research_interval:
            self._perform_research()
            self.last_research_time = current_time
        else:
            # Doar update heartbeat
            time_to_next = int(self.research_interval - (current_time - self.last_research_time))
            hours = time_to_next // 3600
            mins = (time_to_next % 3600) // 60
            self.update_heartbeat("idle",
                                  current_task=f"waiting ({hours}h {mins}m to next research)",
                                  progress_percent=0)


def main():
    """Pornește Strategy Researcher Agent."""
    agent = StrategyResearcherAgent()
    agent.run()


if __name__ == "__main__":
    main()
