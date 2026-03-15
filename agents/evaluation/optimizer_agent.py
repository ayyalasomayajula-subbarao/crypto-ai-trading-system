"""
OptimizerAgent — threshold sweep on the last test fold using the saved model.

Faster than full WF: reuses the already-saved wf_decision_model_v2.pkl and
tests a range of thresholds on the last fold's test period.
Does NOT retrain — use BacktestAgent for full retraining.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from agents.base import BaseAgent
from agents.memory import memory

THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


class OptimizerAgent(BaseAgent):
    SYSTEM_PROMPT = (
        "You are a parameter optimization agent for a crypto ML trading system. "
        "Find the optimal threshold for the saved model without retraining."
    )

    def __init__(self):
        super().__init__("OptimizerAgent", self.SYSTEM_PROMPT)

    def run(self, task: str = "BTC") -> dict:
        """
        Sweep thresholds on last fold's test window using the saved model.
        Returns best threshold and whether a change is recommended.
        """
        from walk_forward_validation import FOLDS, COIN_PARAMS, simulate_trades

        coin = self._resolve_coin(task)
        self.log(f"Optimizing threshold for {coin}")

        # Get current threshold from memory or use default
        active            = memory.get_active_model(coin)
        current_threshold = active.get("threshold", 0.55) if active else 0.55

        model_path = os.path.join(BASE_DIR, "models", coin, "wf_decision_model_v2.pkl")
        feat_path  = os.path.join(BASE_DIR, "models", coin, "decision_features_v2.txt")

        if not os.path.exists(model_path):
            return {"coin": coin, "error": "no saved model found"}
        if not os.path.exists(feat_path):
            feat_path = os.path.join(BASE_DIR, "models", coin, "decision_features.txt")
        if not os.path.exists(feat_path):
            return {"coin": coin, "error": "no features file found"}

        try:
            model = joblib.load(model_path)
            with open(feat_path) as f:
                feature_cols = [l.strip() for l in f if l.strip()]

            folds = FOLDS.get(coin, [])
            if not folds:
                return {"coin": coin, "error": "no folds defined"}

            # Use last fold's test window
            last_fold = folds[-1]
            params    = COIN_PARAMS.get(coin, {"tp": 0.05, "sl": 0.025, "time_limit": 48})

            csv_path = os.path.join(BASE_DIR, "data", f"{coin}_multi_tf_features.csv")
            df = pd.read_csv(csv_path, parse_dates=["timestamp"]).sort_values("timestamp")
            feature_cols = [c for c in feature_cols if c in df.columns]

            test_df = df[
                (df["timestamp"] >= last_fold["test"][0]) &
                (df["timestamp"] <= last_fold["test"][1])
            ]

            if len(test_df) < 30:
                return {"coin": coin, "error": f"insufficient test data ({len(test_df)} rows)"}

            classes = list(model.classes_)

            threshold_results = {}
            for thresh in THRESHOLDS:
                trades, _, _ = simulate_trades(
                    model, test_df, feature_cols, classes,
                    thresh, params["tp"], params["sl"],
                    coin=coin, time_limit=params["time_limit"],
                )
                if len(trades) >= 3:
                    pnls = [t.get("net_pnl_pct", 0) for t in trades]
                    wins = [t for t in trades if t.get("result") == "WIN"]
                    wr   = len(wins) / len(trades)
                    sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(252)) if np.std(pnls) > 0 else 0.0
                    threshold_results[thresh] = {
                        "sharpe": round(sharpe, 3),
                        "wr":     round(wr * 100, 1),
                        "trades": len(trades),
                    }

            if not threshold_results:
                return {"coin": coin, "current_threshold": current_threshold, "error": "no valid threshold results"}

            best_threshold = max(threshold_results, key=lambda t: threshold_results[t]["sharpe"])
            best_sharpe    = threshold_results[best_threshold]["sharpe"]
            current_sharpe = threshold_results.get(current_threshold, {}).get("sharpe", 0.0)
            improvement    = best_sharpe - current_sharpe

            self.log(
                f"{coin}: best={best_threshold} (Sharpe={best_sharpe:.3f}), "
                f"current={current_threshold} (Sharpe={current_sharpe:.3f}), "
                f"delta={improvement:+.3f}"
            )

            return {
                "coin":               coin,
                "current_threshold":  current_threshold,
                "best_threshold":     best_threshold,
                "sharpe_improvement": round(improvement, 3),
                "all_results":        {str(k): v for k, v in threshold_results.items()},
                "recommend_change":   improvement > 0.05,
            }

        except Exception as exc:
            self.log(f"Error: {exc}")
            return {"coin": coin, "error": str(exc)}

    def _resolve_coin(self, task: str) -> str:
        for c in ["BTC", "ETH", "SOL", "PEPE", "AVAX", "BNB", "LINK"]:
            if c.upper() in task.upper():
                return f"{c}_USDT"
        if "_USDT" in task.upper():
            return task.upper()
        return f"{task.upper()}_USDT"
