#!/usr/bin/env python3
"""
Strategy Optimizer Agent

Acest agent se ocupă cu optimizarea și dezvoltarea continuă a strategiilor pentru loterii.
Analizează performanțele actuale, generează noi combinații și optimizează parametrii.

Schedule: Runs continuously, checking for optimization opportunities every hour
"""

from __future__ import annotations

import json
import os
import sys
import time
import random
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project directory to path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from agents.agent_base import BaseAgent
from lottery_config import list_lottery_configs, get_lottery_config
from atomic_io import read_json_with_recovery, write_json_atomic

# Constants
AGENTS_DIR = PROJECT_DIR / "agents"
STRATEGIES_DIR = AGENTS_DIR / "strategies"
PENDING_STRATEGIES_FILE = AGENTS_DIR / "tasks" / "pending_strategies.json"
COMPLETED_STRATEGIES_FILE = AGENTS_DIR / "tasks" / "completed_strategies.json"


class StrategyOptimizerAgent(BaseAgent):
    """Agent pentru optimizarea și dezvoltarea strategiilor."""

    def __init__(self):
        super().__init__("strategy_optimizer", poll_interval=3600)  # Run every hour
        self.strategies_dir = STRATEGIES_DIR
        self.strategies_dir.mkdir(parents=True, exist_ok=True)

        # Ensure task files exist
        AGENTS_DIR.joinpath("tasks").mkdir(parents=True, exist_ok=True)
        for task_file in [PENDING_STRATEGIES_FILE, COMPLETED_STRATEGIES_FILE]:
            if not task_file.exists():
                write_json_atomic(task_file, [])

    def _load_historical_results(self, lottery_id: str) -> List[Dict[str, Any]]:
        """Încarcă rezultatele istorice pentru o loterie."""
        try:
            config = get_lottery_config(lottery_id)
            results_file = config.results_file

            if results_file.exists():
                data, _ = read_json_with_recovery(results_file)
                if isinstance(data, list):
                    return data
        except Exception as e:
            self.log_error(f"Failed to load historical results for {lottery_id}: {e}")

        return []

    def _analyze_current_strategies(self, lottery_id: str) -> Dict[str, Any]:
        """Analizează strategiile curente și performanțele lor."""
        try:
            # Load experiment results
            results = self._load_historical_results(lottery_id)

            if not results:
                return {"total_experiments": 0, "best_metric": 0, "avg_metric": 0}

            # Extract metrics
            metrics = [float(r.get("metric", 0)) for r in results if r.get("status") == "KEEP"]

            if not metrics:
                return {"total_experiments": len(results), "best_metric": 0, "avg_metric": 0}

            return {
                "total_experiments": len(results),
                "best_metric": max(metrics),
                "avg_metric": sum(metrics) / len(metrics),
                "recent_metrics": metrics[-10:] if len(metrics) >= 10 else metrics
            }
        except Exception as e:
            self.log_error(f"Failed to analyze current strategies for {lottery_id}: {e}")
            return {"total_experiments": 0, "best_metric": 0, "avg_metric": 0}

    def _generate_new_strategy_variations(self, lottery_id: str, current_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generează variații ale strategiilor existente."""
        variations = []

        try:
            # Load existing strategies from results
            results = self._load_historical_results(lottery_id)
            keep_results = [r for r in results if r.get("status") == "KEEP"]

            # Generate variations based on top performing strategies
            top_strategies = sorted(keep_results, key=lambda x: float(x.get("metric", 0)), reverse=True)[:5]

            for i, strategy in enumerate(top_strategies):
                # Extract strategy parameters
                description = strategy.get("description", "")
                strategy_params = self._extract_strategy_params(description)

                # Generate variations
                for j in range(3):  # 3 variations per strategy
                    variation = self._mutate_strategy(strategy_params, j)
                    if variation:
                        variations.append({
                            "lottery_id": lottery_id,
                            "base_strategy": description,
                            "variation_params": variation,
                            "priority": "high" if float(strategy.get("metric", 0)) > current_analysis.get("avg_metric", 0) else "normal",
                            "generated_at": datetime.utcnow().isoformat(),
                            "estimated_improvement": self._estimate_improvement(variation, strategy_params)
                        })

        except Exception as e:
            self.log_error(f"Failed to generate strategy variations: {e}")

        return variations

    def _extract_strategy_params(self, description: str) -> Dict[str, Any]:
        """Extrage parametrii din descrierea strategiei."""
        # This is a simplified extraction - in practice, you might want to parse
        # the actual train.py code or use more sophisticated parsing
        params = {}

        # Look for common patterns
        if "ticket_size=" in description:
            try:
                start = description.find("ticket_size=") + len("ticket_size=")
                end = description.find(" ", start)
                if end == -1:
                    end = len(description)
                params["ticket_size"] = int(description[start:end])
            except:
                params["ticket_size"] = 10

        if "lookback=" in description:
            try:
                start = description.find("lookback=") + len("lookback=")
                end = description.find(" ", start)
                if end == -1:
                    end = len(description)
                params["lookback"] = int(description[start:end])
            except:
                params["lookback"] = 50

        # Default values
        if "ticket_size" not in params:
            params["ticket_size"] = 10
        if "lookback" not in params:
            params["lookback"] = 50

        return params

    def _mutate_strategy(self, params: Dict[str, Any], variation_index: int) -> Optional[Dict[str, Any]]:
        """Mutatează parametrii strategiei pentru a genera variații."""
        try:
            mutated = params.copy()

            # Apply different mutations based on variation_index
            if variation_index == 0:
                # Increase ticket size
                mutated["ticket_size"] = min(mutated["ticket_size"] + 2, 20)
            elif variation_index == 1:
                # Decrease lookback
                mutated["lookback"] = max(mutated["lookback"] - 10, 10)
            elif variation_index == 2:
                # Combine both
                mutated["ticket_size"] = min(mutated["ticket_size"] + 1, 20)
                mutated["lookback"] = max(mutated["lookback"] - 5, 10)

            # Add some randomness
            if random.random() < 0.3:
                mutated["ticket_size"] = max(5, min(mutated["ticket_size"] + random.randint(-3, 3), 25))

            if random.random() < 0.3:
                mutated["lookback"] = max(5, mutated["lookback"] + random.randint(-15, 15))

            return mutated
        except Exception as e:
            self.log_error(f"Failed to mutate strategy: {e}")
            return None

    def _estimate_improvement(self, new_params: Dict[str, Any], old_params: Dict[str, Any]) -> float:
        """Estimează potențialul de îmbunătățire al unei strategii."""
        # Simple heuristic - larger ticket sizes might perform better but with diminishing returns
        # Lookback periods that are too short or too long might hurt performance

        ticket_improvement = 0
        lookback_improvement = 0

        # Ticket size improvement (optimal around 10-15)
        old_ticket = old_params.get("ticket_size", 10)
        new_ticket = new_params.get("ticket_size", 10)

        if 8 <= new_ticket <= 15 and not (8 <= old_ticket <= 15):
            ticket_improvement = 0.05
        elif new_ticket > 15 and old_ticket <= 15:
            ticket_improvement = -0.02  # Too large might hurt

        # Lookback improvement (optimal around 30-80)
        old_lookback = old_params.get("lookback", 50)
        new_lookback = new_params.get("lookback", 50)

        if 30 <= new_lookback <= 80 and not (30 <= old_lookback <= 80):
            lookback_improvement = 0.03
        elif new_lookback < 10 or new_lookback > 150:
            lookback_improvement = -0.05  # Too extreme

        return max(-0.1, min(0.1, ticket_improvement + lookback_improvement))

    def _generate_fresh_strategies(self, lottery_id: str) -> List[Dict[str, Any]]:
        """Generează strategii complet noi bazate pe analiza datelor."""
        fresh_strategies = []

        try:
            # Define strategy templates
            templates = [
                {
                    "name": "frequency_based",
                    "description": "Frequency-based selection with adaptive weighting",
                    "params": {"method": "frequency", "weight_decay": 0.95, "min_frequency": 2}
                },
                {
                    "name": "hot_cold_analysis",
                    "description": "Hot/cold number analysis with trend following",
                    "params": {"hot_window": 20, "cold_threshold": 30, "trend_weight": 0.7}
                },
                {
                    "name": "pattern_matching",
                    "description": "Pattern matching with recent draw similarity",
                    "params": {"pattern_window": 15, "similarity_threshold": 0.8}
                },
                {
                    "name": "statistical_arbitrage",
                    "description": "Statistical arbitrage based on number correlations",
                    "params": {"correlation_window": 25, "arb_threshold": 0.15}
                }
            ]

            # Generate variations of each template
            for template in templates:
                for i in range(2):  # 2 variations per template
                    strategy = template.copy()
                    strategy["lottery_id"] = lottery_id
                    strategy["generated_at"] = datetime.utcnow().isoformat()
                    strategy["priority"] = "normal"

                    # Add some randomization to parameters
                    if "params" in strategy:
                        for param_name, param_value in strategy["params"].items():
                            if isinstance(param_value, (int, float)):
                                # Add ±20% variation
                                variation = param_value * (0.8 + random.random() * 0.4)
                                if isinstance(param_value, int):
                                    strategy["params"][param_name] = int(variation)
                                else:
                                    strategy["params"][param_name] = round(variation, 2)

                    fresh_strategies.append(strategy)

        except Exception as e:
            self.log_error(f"Failed to generate fresh strategies: {e}")

        return fresh_strategies

    def _save_pending_strategies(self, strategies: List[Dict[str, Any]]) -> None:
        """Salvează strategiile în lista de sarcini în așteptare."""
        try:
            # Load existing pending strategies
            existing, _ = read_json_with_recovery(PENDING_STRATEGIES_FILE)
            if not isinstance(existing, list):
                existing = []

            # Add new strategies
            existing.extend(strategies)

            # Sort by priority
            priority_order = {"high": 0, "normal": 1, "low": 2}
            existing.sort(key=lambda x: priority_order.get(x.get("priority", "normal"), 1))

            # Save updated list
            write_json_atomic(PENDING_STRATEGIES_FILE, existing)
            self.log_info(f"Saved {len(strategies)} new strategies to pending list")

        except Exception as e:
            self.log_error(f"Failed to save pending strategies: {e}")

    def _check_for_completed_optimizations(self) -> None:
        """Verifică dacă au fost completate optimizări și actualizează statistici."""
        try:
            # This would be called to check if any of our suggested strategies
            # have been implemented and tested by the main experiment loop
            pass  # Implementation would depend on how results are tracked
        except Exception as e:
            self.log_error(f"Failed to check for completed optimizations: {e}")

    def process(self) -> None:
        """Procesare principală pentru optimizarea strategiilor."""
        self.log_info("Starting strategy optimization cycle...")

        # Check for completed optimizations first
        self._check_for_completed_optimizations()

        all_new_strategies = []

        # Process each lottery
        for config in list_lottery_configs():
            lottery_id = config.id
            self.log_info(f"Analyzing strategies for {lottery_id}...")

            try:
                # Analyze current performance
                current_analysis = self._analyze_current_strategies(lottery_id)
                self.log_info(f"  Current best metric: {current_analysis.get('best_metric', 0):.4f}")

                # Generate variations of existing successful strategies
                strategy_variations = self._generate_new_strategy_variations(lottery_id, current_analysis)
                self.log_info(f"  Generated {len(strategy_variations)} strategy variations")

                # Generate completely fresh strategies
                fresh_strategies = self._generate_fresh_strategies(lottery_id)
                self.log_info(f"  Generated {len(fresh_strategies)} fresh strategies")

                # Combine all strategies
                all_lottery_strategies = strategy_variations + fresh_strategies
                all_new_strategies.extend(all_lottery_strategies)

                # Add lottery context to each strategy
                for strategy in all_lottery_strategies:
                    strategy["lottery_id"] = lottery_id

            except Exception as e:
                self.log_error(f"Failed to process {lottery_id}: {e}")

        # Save all new strategies to pending list
        if all_new_strategies:
            self._save_pending_strategies(all_new_strategies)
            self.log_success(f"Total of {len(all_new_strategies)} new strategies saved for testing")
        else:
            self.log_info("No new strategies generated in this cycle")

        # Update heartbeat
        self.update_heartbeat(
            "running",
            current_task=f"analyzed {len(list_lottery_configs())} lotteries, generated {len(all_new_strategies)} strategies",
            progress_percent=100,
            additional_info={
                "pending_strategies": len(all_new_strategies),
                "last_analysis": datetime.utcnow().isoformat()
            }
        )

        self.log_info("Strategy optimization cycle completed")


def main():
    """Pornește Strategy Optimizer Agent."""
    agent = StrategyOptimizerAgent()
    agent.run()


if __name__ == "__main__":
    main()