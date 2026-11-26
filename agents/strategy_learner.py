#!/usr/bin/env python3
"""
Strategy Learning Agent

Acest agent analizează rezultatele experimentelor finalizate pentru a învăța
ce combinații de parametri funcționează cel mai bine și pentru a identifica
modele în succesul strategic.

Schedule: Runs daily to analyze completed experiments
"""

from __future__ import annotations

import json
import os
import sys
import time
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Add project directory to path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from agents.agent_base import BaseAgent
from lottery_config import list_lottery_configs, get_lottery_config
from atomic_io import read_json_with_recovery, write_json_atomic

# Constants
AGENTS_DIR = PROJECT_DIR / "agents"
LEARNING_DIR = AGENTS_DIR / "learning"
MODELS_DIR = LEARNING_DIR / "models"
INSIGHTS_DIR = LEARNING_DIR / "insights"


class StrategyLearningAgent(BaseAgent):
    """Agent pentru învățarea din strategiile testate."""

    def __init__(self):
        super().__init__("strategy_learner", poll_interval=86400)  # Run daily (24 hours)
        self.learning_dir = LEARNING_DIR
        self.models_dir = MODELS_DIR
        self.insights_dir = INSIGHTS_DIR

        # Create directories
        for directory in [self.learning_dir, self.models_dir, self.insights_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _load_experiment_results(self, lottery_id: str) -> List[Dict[str, Any]]:
        """Încarcă toate rezultatele experimentelor pentru o loterie."""
        try:
            config = get_lottery_config(lottery_id)
            results_file = config.results_file

            if results_file.exists():
                data, _ = read_json_with_recovery(results_file)
                if isinstance(data, list):
                    return data
        except Exception as e:
            self.log_error(f"Failed to load experiment results for {lottery_id}: {e}")

        return []

    def _extract_strategy_features(self, description: str) -> Dict[str, Any]:
        """Extrage caracteristicile unei strategii din descrierea ei."""
        features = {
            "ticket_size": 10,
            "lookback_period": 50,
            "algorithm_type": "unknown",
            "complexity_score": 1.0,
            "parameters_count": 0
        }

        try:
            # Extract ticket size
            if "ticket_size=" in description:
                start = description.find("ticket_size=") + len("ticket_size=")
                end = description.find(" ", start)
                if end == -1:
                    end = len(description)
                features["ticket_size"] = int(description[start:end])

            # Extract lookback period
            if "lookback=" in description or "window=" in description:
                lookback_match = description.find("lookback=")
                window_match = description.find("window=")

                if lookback_match != -1:
                    start = lookback_match + len("lookback=")
                    end = description.find(" ", start)
                    if end == -1:
                        end = len(description)
                    features["lookback_period"] = int(description[start:end])
                elif window_match != -1:
                    start = window_match + len("window=")
                    end = description.find(" ", start)
                    if end == -1:
                        end = len(description)
                    features["lookback_period"] = int(description[start:end])

            # Determine algorithm type based on keywords
            description_lower = description.lower()
            if "frequency" in description_lower or "freq" in description_lower:
                features["algorithm_type"] = "frequency_based"
            elif "hot" in description_lower and "cold" in description_lower:
                features["algorithm_type"] = "hot_cold_analysis"
            elif "pattern" in description_lower:
                features["algorithm_type"] = "pattern_matching"
            elif "correlat" in description_lower or "arb" in description_lower:
                features["algorithm_type"] = "statistical_arbitrage"
            elif "random" in description_lower:
                features["algorithm_type"] = "random_selection"
            else:
                features["algorithm_type"] = "hybrid"

            # Estimate complexity based on description length and parameter count
            features["parameters_count"] = description.count("=")
            features["complexity_score"] = min(10.0, len(description) / 50.0)

        except Exception as e:
            self.log_error(f"Failed to extract features from description: {e}")

        return features

    def _analyze_performance_patterns(self, results: List[Dict[str, Any]], lottery_id: str) -> Dict[str, Any]:
        """Analizează modelele de performanță în rezultatele experimentelor."""
        if not results:
            return {}

        try:
            # Filter for successful experiments (KEEP status)
            keep_results = [r for r in results if r.get("status") == "KEEP"]

            if not keep_results:
                return {"message": "No successful experiments to analyze"}

            # Extract features and metrics
            features_list = []
            metrics_list = []

            for result in keep_results:
                description = result.get("description", "")
                features = self._extract_strategy_features(description)
                metric = float(result.get("metric", 0))

                features_list.append(features)
                metrics_list.append(metric)

            # Convert to numpy arrays for analysis
            metrics_array = np.array(metrics_list)

            # Basic statistics
            analysis = {
                "total_successful_experiments": len(keep_results),
                "best_metric": float(np.max(metrics_array)),
                "average_metric": float(np.mean(metrics_array)),
                "median_metric": float(np.median(metrics_array)),
                "std_deviation": float(np.std(metrics_array)),
                "percentile_25": float(np.percentile(metrics_array, 25)),
                "percentile_75": float(np.percentile(metrics_array, 75)),
                "improvement_over_baseline": float(np.mean(metrics_array) - 2.5)  # Assuming 2.5 baseline for ticket_size=10
            }

            # Parameter correlation analysis
            ticket_sizes = [f["ticket_size"] for f in features_list]
            lookbacks = [f["lookback_period"] for f in features_list]
            complexities = [f["complexity_score"] for f in features_list]
            alg_types = [f["algorithm_type"] for f in features_list]

            # Correlation with performance
            if len(metrics_array) > 1:
                ticket_corr = np.corrcoef(ticket_sizes, metrics_array)[0, 1] if np.std(ticket_sizes) > 0 else 0
                lookback_corr = np.corrcoef(lookbacks, metrics_array)[0, 1] if np.std(lookbacks) > 0 else 0
                complexity_corr = np.corrcoef(complexities, metrics_array)[0, 1] if np.std(complexities) > 0 else 0

                analysis["parameter_correlations"] = {
                    "ticket_size_correlation": float(ticket_corr),
                    "lookback_correlation": float(lookback_corr),
                    "complexity_correlation": float(complexity_corr)
                }

            # Algorithm type performance
            alg_performance = defaultdict(list)
            for features, metric in zip(features_list, metrics_list):
                alg_performance[features["algorithm_type"]].append(metric)

            alg_stats = {}
            for alg_type, performances in alg_performance.items():
                if performances:
                    alg_stats[alg_type] = {
                        "count": len(performances),
                        "average": float(np.mean(performances)),
                        "best": float(np.max(performances)),
                        "std_dev": float(np.std(performances))
                    }

            analysis["algorithm_performance"] = alg_stats

            # Best performing configurations
            sorted_results = sorted(zip(keep_results, metrics_list), key=lambda x: x[1], reverse=True)
            top_5 = sorted_results[:5]

            analysis["top_performers"] = [
                {
                    "description": result[0].get("description", ""),
                    "metric": float(result[1]),
                    "ticket_size": self._extract_strategy_features(result[0].get("description", "")).get("ticket_size", 10)
                }
                for result in top_5
            ]

            return analysis

        except Exception as e:
            self.log_error(f"Failed to analyze performance patterns: {e}")
            return {"error": str(e)}

    def _generate_insights(self, analysis: Dict[str, Any], lottery_id: str) -> Dict[str, Any]:
        """Generează insights din analiza performanței."""
        insights = {
            "lottery_id": lottery_id,
            "generated_at": datetime.utcnow().isoformat(),
            "recommendations": [],
            "warnings": [],
            "observations": []
        }

        try:
            # Check if we have meaningful data
            if "total_successful_experiments" not in analysis:
                insights["observations"].append("Insufficient data for analysis")
                return insights

            # Performance observations
            avg_metric = analysis.get("average_metric", 0)
            best_metric = analysis.get("best_metric", 0)

            if avg_metric > 2.7:
                insights["observations"].append(f"Average performance ({avg_metric:.3f}) is significantly above random baseline (2.5)")
            elif avg_metric > 2.5:
                insights["observations"].append(f"Average performance ({avg_metric:.3f}) is above random baseline (2.5)")

            if best_metric > avg_metric + 0.2:
                insights["observations"].append(f"Best performer ({best_metric:.3f}) shows significant potential over average ({avg_metric:.3f})")

            # Parameter recommendations
            if "parameter_correlations" in analysis:
                corrs = analysis["parameter_correlations"]

                if abs(corrs.get("ticket_size_correlation", 0)) > 0.3:
                    direction = "increase" if corrs["ticket_size_correlation"] > 0 else "decrease"
                    insights["recommendations"].append(f"Consider {direction} ticket sizes based on correlation analysis")

                if abs(corrs.get("lookback_correlation", 0)) > 0.3:
                    direction = "increase" if corrs["lookback_correlation"] > 0 else "decrease"
                    insights["recommendations"].append(f"Consider {direction} lookback periods based on correlation analysis")

            # Algorithm recommendations
            if "algorithm_performance" in analysis:
                alg_perf = analysis["algorithm_performance"]
                if alg_perf:
                    best_alg = max(alg_perf.keys(), key=lambda k: alg_perf[k]["average"])
                    insights["recommendations"].append(f"Focus on {best_alg} algorithms (best average performance)")

            # Warnings
            std_dev = analysis.get("std_deviation", 0)
            if std_dev > 0.5:
                insights["warnings"].append(f"High variance in results (σ={std_dev:.3f}) suggests instability")

            if analysis.get("total_successful_experiments", 0) < 10:
                insights["warnings"].append("Limited successful experiments for robust analysis")

        except Exception as e:
            self.log_error(f"Failed to generate insights: {e}")
            insights["warnings"].append(f"Error generating insights: {e}")

        return insights

    def _save_learning_model(self, analysis: Dict[str, Any], lottery_id: str) -> None:
        """Salvează modelul de învățare pentru utilizare ulterioară."""
        try:
            model_file = self.models_dir / f"{lottery_id}_learning_model.json"
            model_data = {
                "lottery_id": lottery_id,
                "created_at": datetime.utcnow().isoformat(),
                "analysis": analysis,
                "version": "1.0"
            }

            write_json_atomic(model_file, model_data)
            self.log_info(f"Learning model saved for {lottery_id}")
        except Exception as e:
            self.log_error(f"Failed to save learning model: {e}")

    def _save_insights(self, insights: Dict[str, Any]) -> None:
        """Salvează insights pentru utilizare ulterioară."""
        try:
            lottery_id = insights.get("lottery_id", "unknown")
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            insights_file = self.insights_dir / f"{lottery_id}_insights_{timestamp}.json"

            write_json_atomic(insights_file, insights)
            self.log_info(f"Insights saved for {lottery_id}")
        except Exception as e:
            self.log_error(f"Failed to save insights: {e}")

    def process(self) -> None:
        """Procesare principală pentru învățarea din strategii."""
        self.log_info("Starting strategy learning cycle...")

        all_analyses = {}

        # Process each lottery
        for config in list_lottery_configs():
            lottery_id = config.id
            self.log_info(f"Analyzing learning patterns for {lottery_id}...")

            try:
                # Load experiment results
                results = self._load_experiment_results(lottery_id)
                self.log_info(f"  Loaded {len(results)} experiment results")

                if not results:
                    self.log_info(f"  No results to analyze for {lottery_id}")
                    continue

                # Analyze performance patterns
                analysis = self._analyze_performance_patterns(results, lottery_id)
                all_analyses[lottery_id] = analysis

                if analysis:
                    self.log_info(f"  Analysis completed for {lottery_id}")

                    # Generate insights
                    insights = self._generate_insights(analysis, lottery_id)

                    # Save learning model and insights
                    self._save_learning_model(analysis, lottery_id)
                    self._save_insights(insights)

                    # Log key findings
                    if "average_metric" in analysis:
                        self.log_info(f"  Average performance: {analysis['average_metric']:.4f}")
                    if "best_metric" in analysis:
                        self.log_info(f"  Best performance: {analysis['best_metric']:.4f}")

                else:
                    self.log_info(f"  No analysis generated for {lottery_id}")

            except Exception as e:
                self.log_error(f"Failed to process {lottery_id}: {e}")

        # Update heartbeat with summary
        total_lotteries = len(all_analyses)
        successful_analyses = sum(1 for analysis in all_analyses.values() if analysis)

        self.update_heartbeat(
            "running",
            current_task=f"analyzed {total_lotteries} lotteries, {successful_analyses} successful analyses",
            progress_percent=100,
            additional_info={
                "analyses_completed": successful_analyses,
                "total_lotteries": total_lotteries,
                "last_analysis": datetime.utcnow().isoformat()
            }
        )

        self.log_info("Strategy learning cycle completed")


def main():
    """Pornește Strategy Learning Agent."""
    agent = StrategyLearningAgent()
    agent.run()


if __name__ == "__main__":
    main()