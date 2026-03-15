"""
MonitorAgent — performance monitor and self-improvement trigger.

Self-improve loop:
  1. RiskAgent.run()         — check portfolio metrics
  2. If degraded coin found → OptimizerAgent.run(coin)
  3. If optimizer finds improvement > 5% Sharpe → BacktestAgent.run(coin) for full retrain
  4. All actions stored in agent_memory.db
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from agents.base import BaseAgent
from agents.memory import memory


class MonitorAgent(BaseAgent):
    SYSTEM_PROMPT = (
        "You are a performance monitoring agent for a crypto trading system. "
        "Check portfolio metrics and trigger optimization when performance degrades."
    )

    def __init__(self):
        super().__init__("MonitorAgent", self.SYSTEM_PROMPT)

    def run(self, task: str = "check") -> dict:
        """
        Full self-improvement check:
        1. Risk report
        2. Optimize degraded coins
        3. Full retrain if optimizer finds significant improvement
        """
        from agents.evaluation.risk_agent import RiskAgent
        from agents.evaluation.optimizer_agent import OptimizerAgent
        from agents.evaluation.backtest_agent import BacktestAgent

        self.log("Starting self-improvement check")
        actions_taken = []

        # Step 1 — Risk check
        risk_result = RiskAgent().run("report")
        degraded    = risk_result.get("degraded_coins", {})

        if not degraded:
            self.log("All coins healthy — no action needed")
            return {
                "status":        "healthy",
                "risk_report":   risk_result,
                "actions_taken": [],
            }

        self.log(f"Degraded: {list(degraded.keys())}")

        # Step 2 — Optimize degraded coins
        retrain_queue = []
        for coin in degraded:
            if coin.startswith("PORTFOLIO"):
                continue  # Portfolio-level alert — handled by per-coin loop

            opt_result = OptimizerAgent().run(coin)
            improvement = opt_result.get("sharpe_improvement", 0)

            if opt_result.get("recommend_change") and improvement > 0.05:
                self.log(
                    f"{coin}: threshold change recommended "
                    f"({opt_result.get('current_threshold')} → {opt_result.get('best_threshold')}, "
                    f"Sharpe +{improvement:.3f})"
                )
                actions_taken.append({
                    "type":      "threshold_updated",
                    "coin":      coin,
                    "old":       opt_result.get("current_threshold"),
                    "new":       opt_result.get("best_threshold"),
                    "sharpe_delta": improvement,
                })
                memory.store(
                    f"threshold_update_{coin}",
                    {"threshold": opt_result.get("best_threshold")},
                    agent_name="MonitorAgent",
                )

            # Step 3 — Full retrain if improvement is major (>15%)
            if improvement > 0.15:
                retrain_queue.append(coin)

        for coin in retrain_queue:
            self.log(f"{coin}: triggering full walk-forward retrain")
            bt_result = BacktestAgent().run(coin)
            verdict   = bt_result.get("verdict", "UNKNOWN")
            self.log(f"{coin} retrain result: {verdict}")
            actions_taken.append({
                "type":    "retrained",
                "coin":    coin,
                "verdict": verdict,
                "sharpe":  bt_result.get("sharpe", 0),
            })

        result = {
            "status":        "improved" if actions_taken else "degraded_no_fix",
            "degraded_coins": degraded,
            "risk_report":   risk_result,
            "actions_taken": actions_taken,
        }
        memory.store("last_monitor_run", result, agent_name="MonitorAgent")
        self.log(f"Monitor done: {len(actions_taken)} actions taken")
        return result
