"""
DataAgent — incremental market data updater.
Wraps update_data.IncrementalUpdater.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from agents.base import BaseAgent

ALL_COINS = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "PEPE_USDT", "AVAX_USDT", "BNB_USDT", "LINK_USDT"]


class DataAgent(BaseAgent):
    SYSTEM_PROMPT = (
        "You are a market data agent for a crypto trading system. "
        "Your job is to ensure OHLCV data is current for all supported coins."
    )

    def __init__(self):
        super().__init__("DataAgent", self.SYSTEM_PROMPT)

    def run(self, task: str = "update") -> dict:
        """
        Update market data for all coins (or a specific coin if named in task).
        task: "update", "update BTC", "update full"
        """
        from update_data import IncrementalUpdater

        coins = self._parse_coins(task)                      # ["BTC_USDT", ...]
        ccxt_coins = [c.replace("_", "/") for c in coins]   # ccxt needs "BTC/USDT"
        full = "full" in task.lower()

        self.log(f"Updating {len(ccxt_coins)} coin(s) (full={full})")

        try:
            updater = IncrementalUpdater()
            results = {}
            for sym in ccxt_coins:
                key = sym.replace("/", "_")
                try:
                    counts = updater.update_coin(sym, full=full)
                    total = sum(counts.values()) if isinstance(counts, dict) else (counts or 0)
                    results[key] = {"new_candles": total, "status": "ok"}
                    self.log(f"  {key}: +{total} candles")
                except Exception as e:
                    results[key] = {"status": "error", "error": str(e)}
                    self.log(f"  {key}: ERROR {e}")

            return {"status": "ok", "coins_updated": results}

        except Exception as e:
            self.log(f"Fatal error: {e}")
            return {"status": "error", "error": str(e)}

    def _parse_coins(self, task: str) -> list:
        for c in ["BTC", "ETH", "SOL", "PEPE", "AVAX", "BNB", "LINK"]:
            if c.upper() in task.upper():
                return [f"{c}_USDT"]
        return ALL_COINS
