"""BacktestAgent — Runs walk-forward validation for one symbol."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from agents.base import BaseAgent

class BacktestAgent(BaseAgent):
    name = "BacktestAgent"

    def run(self, task: dict) -> dict:
        symbol = task.get("symbol")
        try:
            from walk_forward import run_wf
            result = run_wf(symbol)
            self.remember(f"wf_result_{symbol}", result)
            return {"status": "ok", "verdict": result.get("verdict"),
                    "sharpe": result.get("avg_sharpe"),
                    "wr": result.get("avg_wr")}
        except Exception as e:
            return {"status": "error", "error": str(e)}
