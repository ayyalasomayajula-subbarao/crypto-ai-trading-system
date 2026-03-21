"""RiskAgent — Reports paper trading metrics."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from agents.base import BaseAgent

class RiskAgent(BaseAgent):
    name = "RiskAgent"

    def run(self, task: dict) -> dict:
        try:
            from paper_trader import PaperTrader
            pt = PaperTrader()
            metrics = pt.get_metrics()
            self.remember("paper_metrics", metrics)
            return {"status": "ok", **metrics}
        except Exception as e:
            return {"status": "error", "error": str(e)}
