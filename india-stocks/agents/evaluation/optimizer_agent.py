"""OptimizerAgent — Sweeps thresholds on saved WF model."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from agents.base import BaseAgent

class OptimizerAgent(BaseAgent):
    name = "OptimizerAgent"

    def run(self, task: dict) -> dict:
        symbol = task.get("symbol")
        try:
            from walk_forward import run_wf
            result = run_wf(symbol)
            return {"status": "ok", "best_threshold": result.get("best_threshold"),
                    "verdict": result.get("verdict")}
        except Exception as e:
            return {"status": "error", "error": str(e)}
