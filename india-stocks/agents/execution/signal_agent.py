"""SignalAgent — Loads model and generates verdict for one symbol."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from agents.base import BaseAgent

class SignalAgent(BaseAgent):
    name = "SignalAgent"

    def run(self, task: dict) -> dict:
        symbol = task.get("symbol")
        try:
            from precision_verdict import VerdictEngine
            engine = VerdictEngine()
            result = engine.get_verdict(symbol, force=True)
            self.remember(f"last_verdict_{symbol}", result)
            return {"status": "ok", "verdict": result.get("verdict"),
                    "score": result.get("score")}
        except Exception as e:
            return {"status": "error", "error": str(e)}
