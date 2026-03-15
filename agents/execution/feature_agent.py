"""
FeatureAgent — multi-timeframe feature engineering.
Wraps collect_multi_timeframe.create_multi_timeframe_features().
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from agents.base import BaseAgent

ALL_COINS = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "PEPE_USDT", "AVAX_USDT", "BNB_USDT", "LINK_USDT"]


class FeatureAgent(BaseAgent):
    SYSTEM_PROMPT = (
        "You are a feature engineering agent for a crypto trading ML system. "
        "Build multi-timeframe technical features with no lookahead bias."
    )

    def __init__(self):
        super().__init__("FeatureAgent", self.SYSTEM_PROMPT)

    def run(self, task: str = "rebuild") -> dict:
        """
        Regenerate feature CSVs for all coins (or specific coin).
        task: "rebuild", "rebuild BTC", "rebuild SOL"
        """
        from collect_multi_timeframe import create_multi_timeframe_features

        coins = self._parse_coins(task)
        self.log(f"Rebuilding features for {len(coins)} coin(s)")

        results = {}
        for coin in coins:
            try:
                df = create_multi_timeframe_features(coin)
                ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
                results[coin] = {
                    "status": "ok",
                    "rows": len(df),
                    "features": len(df.columns),
                    "date_from": str(df[ts_col].iloc[0])[:10],
                    "date_to": str(df[ts_col].iloc[-1])[:10],
                }
                self.log(f"  {coin}: {len(df)} rows, {len(df.columns)} cols")
            except Exception as e:
                results[coin] = {"status": "error", "error": str(e)}
                self.log(f"  {coin}: ERROR {e}")

        return {"status": "ok", "results": results}

    def _parse_coins(self, task: str) -> list:
        for c in ["BTC", "ETH", "SOL", "PEPE", "AVAX", "BNB", "LINK"]:
            if c.upper() in task.upper():
                return [f"{c}_USDT"]
        return ALL_COINS
