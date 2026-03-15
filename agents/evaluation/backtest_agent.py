"""
BacktestAgent — runs expanding walk-forward validation for a coin.
Wraps walk_forward_validation.run_walk_forward().
Saves model version to SQLite on VIABLE/MARGINAL result.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from agents.base import BaseAgent
from agents.memory import memory


class BacktestAgent(BaseAgent):
    SYSTEM_PROMPT = (
        "You are a backtesting agent. Run walk-forward validation for crypto ML models "
        "and report Sharpe, win rate, and verdict (VIABLE/MARGINAL/NOT_VIABLE)."
    )

    def __init__(self):
        super().__init__("BacktestAgent", self.SYSTEM_PROMPT)

    def run(self, task: str = "BTC") -> dict:
        """
        Run full expanding WF for the specified coin.
        task: "BTC", "BTC_USDT", "backtest SOL"
        """
        from walk_forward_validation import run_walk_forward

        coin = self._resolve_coin(task)
        self.log(f"Running walk-forward validation for {coin}")

        try:
            result = run_walk_forward(coin)

            if not result:
                return {"coin": coin, "verdict": "ERROR", "error": "run_walk_forward returned None"}

            verdict = result.get("verdict", "UNKNOWN")
            sharpe  = result.get("sharpe", 0.0)
            wr      = result.get("win_rate", 0.0)
            thresh  = result.get("best_threshold", 0.55)

            self.log(f"{coin}: {verdict} — Sharpe={sharpe:.3f}, WR={wr:.1f}%, threshold={thresh}")

            # Save model version if viable
            if verdict in ("VIABLE", "MARGINAL"):
                model_path = os.path.join(BASE_DIR, "models", coin, "wf_decision_model_v2.pkl")
                memory.save_model_version(
                    coin=coin,
                    model_path=model_path,
                    sharpe=sharpe,
                    wr=wr,
                    threshold=thresh,
                )
                self.log(f"Model version saved for {coin} (Sharpe={sharpe:.3f})")

            return result

        except Exception as exc:
            self.log(f"Error: {exc}")
            return {"coin": coin, "verdict": "ERROR", "error": str(exc)}

    def _resolve_coin(self, task: str) -> str:
        for c in ["BTC", "ETH", "SOL", "PEPE", "AVAX", "BNB", "LINK"]:
            if c.upper() in task.upper():
                return f"{c}_USDT"
        if "_USDT" in task.upper():
            return task.upper()
        return f"{task.upper()}_USDT"
