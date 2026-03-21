"""FeatureAgent — Runs feature engineering pipeline for one symbol."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from agents.base import BaseAgent

class FeatureAgent(BaseAgent):
    name = "FeatureAgent"

    def run(self, task: dict) -> dict:
        symbol = task.get("symbol")
        try:
            from feature_engineering import update_symbol
            ok = update_symbol(symbol)
            return {"status": "ok" if ok else "no_data"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
