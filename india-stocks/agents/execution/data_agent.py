"""DataAgent — Fetches OHLCV + FII/DII + option chain for one symbol."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from agents.base import BaseAgent

class DataAgent(BaseAgent):
    name = "DataAgent"

    def run(self, task: dict) -> dict:
        symbol = task.get("symbol")
        try:
            from collect_nse_data import update_symbol
            r = update_symbol(symbol)
            return {"status": "ok", "rows": r}
        except Exception as e:
            return {"status": "error", "error": str(e)}
