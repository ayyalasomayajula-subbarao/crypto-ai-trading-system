"""
StrategyDiscoveryAgent — autonomous strategy parameter discovery.

Phase 1: Ask Groq to propose parameter variations (TP/SL/time_limit/threshold).
Phase 2: Mini-backtest each variation on last 2 folds (fast, retrains model).
Phase 3: Keep improvements > 10% Sharpe delta — register in model_versions.

This is how the system discovers better configurations over time.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from agents.base import BaseAgent
from agents.memory import memory


class StrategyDiscoveryAgent(BaseAgent):
    SYSTEM_PROMPT = (
        "You are a quantitative strategy researcher for a crypto trading system. "
        "Propose conservative parameter variations to test: "
        "TP%, SL%, time_limit (hours), threshold (0.40-0.70). "
        "Keep TP:SL ratio >= 2:1. Return JSON array of variation objects only."
    )

    def __init__(self):
        super().__init__("StrategyDiscoveryAgent", self.SYSTEM_PROMPT)

    def run(self, task: str = "BTC") -> dict:
        from walk_forward_validation import FOLDS, COIN_PARAMS, train_model, simulate_trades

        coin   = self._resolve_coin(task)
        params = COIN_PARAMS.get(coin, {"tp": 0.05, "sl": 0.025, "time_limit": 48})
        folds  = FOLDS.get(coin, [])

        if len(folds) < 2:
            return {"coin": coin, "error": "need at least 2 folds for discovery"}

        self.log(f"Strategy discovery for {coin} (base: {params})")

        # Phase 1: Ask LLM for parameter variations
        variations = self._propose_variations(coin, params)
        self.log(f"Testing {len(variations)} variations on last 2 folds")

        # Load feature data
        csv_path = os.path.join(BASE_DIR, "data", f"{coin}_multi_tf_features.csv")
        feat_path = os.path.join(BASE_DIR, "models", coin, "decision_features_v2.txt")

        if not os.path.exists(csv_path) or not os.path.exists(feat_path):
            return {"coin": coin, "error": "missing CSV or features file"}

        df = pd.read_csv(csv_path, parse_dates=["timestamp"]).sort_values("timestamp")
        with open(feat_path) as f:
            feature_cols = [l.strip() for l in f if l.strip()]
        feature_cols = [c for c in feature_cols if c in df.columns]

        # Phase 2: Mini-backtest each variation on last 2 folds
        base_sharpe  = self._test_variation(params, coin, df, feature_cols, folds[-2:], train_model, simulate_trades)
        improvements = []

        for var in variations:
            try:
                var_sharpe = self._test_variation(var, coin, df, feature_cols, folds[-2:], train_model, simulate_trades)
                delta = var_sharpe - base_sharpe
                result = {
                    "params":      var,
                    "sharpe":      round(var_sharpe, 3),
                    "base_sharpe": round(base_sharpe, 3),
                    "delta":       round(delta, 3),
                }
                if delta > 0.10:
                    improvements.append(result)
                    self.log(f"  Improvement found: {var} → Sharpe +{delta:.3f}")
            except Exception as exc:
                self.log(f"  Variation {var} failed: {exc}")

        # Phase 3: Register best improvement in memory
        best = max(improvements, key=lambda x: x["delta"]) if improvements else None
        if best:
            memory.store(
                f"discovery_{coin}",
                best,
                agent_name="StrategyDiscoveryAgent",
            )
            self.log(f"Best discovery for {coin}: {best['params']} (Sharpe +{best['delta']:.3f})")

        return {
            "coin":         coin,
            "base_sharpe":  round(base_sharpe, 3),
            "tested":       len(variations),
            "improvements": improvements,
            "best_params":  best["params"] if best else None,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _propose_variations(self, coin: str, base: dict) -> list:
        prompt = (
            f"Current params for {coin}: {json.dumps(base)}. "
            "Suggest exactly 5 parameter variations to test. "
            "Each: {\"tp\": float, \"sl\": float, \"time_limit\": int, \"threshold\": float}. "
            "TP:SL must be >= 2:1. threshold between 0.40 and 0.70. "
            "Return JSON array only."
        )
        result = self.think_json(prompt, max_tokens=400)

        if isinstance(result, list) and result:
            validated = []
            for v in result:
                v = self._validate_params(v, base)
                if v is not None:
                    validated.append(v)
            if validated:
                return validated

        # Fallback: hardcoded conservative variations
        tp, sl, tl = base["tp"], base["sl"], base["time_limit"]
        return [
            {"tp": tp * 1.2,  "sl": sl,       "time_limit": tl,    "threshold": 0.55},
            {"tp": tp,        "sl": sl * 0.8,  "time_limit": tl,    "threshold": 0.55},
            {"tp": tp * 1.1,  "sl": sl * 0.9,  "time_limit": tl,    "threshold": 0.60},
            {"tp": tp,        "sl": sl,        "time_limit": tl + 24, "threshold": 0.55},
            {"tp": tp,        "sl": sl,        "time_limit": tl,    "threshold": 0.60},
        ]

    def _test_variation(self, params: dict, coin: str, df: pd.DataFrame,
                        feature_cols: list, test_folds: list,
                        train_model_fn, simulate_fn) -> float:
        """Run mini-backtest on last N folds, return mean Sharpe."""
        sharpes = []
        threshold = params.get("threshold", 0.55)

        for fold in test_folds:
            train_df = df[
                (df["timestamp"] >= fold["train"][0]) &
                (df["timestamp"] <= fold["train"][1])
            ].copy()
            test_df = df[
                (df["timestamp"] >= fold["test"][0]) &
                (df["timestamp"] <= fold["test"][1])
            ].copy()

            if len(train_df) < 200 or len(test_df) < 30:
                continue

            model, classes, _ = train_model_fn(train_df, feature_cols)
            if model is None:
                continue

            trades, _, _ = simulate_fn(
                model, test_df, feature_cols, classes,
                threshold, params["tp"], params["sl"],
                coin=coin, time_limit=params["time_limit"],
            )

            if len(trades) >= 5:
                pnls = [t.get("net_pnl_pct", 0) for t in trades]
                sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(252)) if np.std(pnls) > 0 else 0.0
                sharpes.append(sharpe)

        return float(np.mean(sharpes)) if sharpes else -1.0

    def _validate_params(self, v: dict, base: dict):
        """
        Validate and sanitize an LLM-proposed parameter dict.
        Returns a clean dict or None if fundamentally invalid.
        """
        if not isinstance(v, dict):
            return None

        try:
            tp        = float(v.get("tp",         base.get("tp", 0.05)))
            sl        = float(v.get("sl",         base.get("sl", 0.025)))
            time_limit = int(v.get("time_limit",  base.get("time_limit", 48)))
            threshold  = float(v.get("threshold", 0.55))
        except (TypeError, ValueError):
            return None

        # Hard bounds — reject clearly bad values
        if tp <= 0 or tp > 0.50:          # TP must be 0–50%
            return None
        if sl <= 0 or sl > 0.25:          # SL must be 0–25%
            return None
        if time_limit < 1 or time_limit > 336:   # 1h–2 weeks
            return None
        if threshold < 0.40 or threshold > 0.70:
            return None
        if sl == 0 or tp / sl < 2.0:      # enforce TP:SL >= 2:1
            return None

        return {"tp": tp, "sl": sl, "time_limit": time_limit, "threshold": threshold}

    def _resolve_coin(self, task: str) -> str:
        for c in ["BTC", "ETH", "SOL", "PEPE", "AVAX", "BNB", "LINK"]:
            if c.upper() in task.upper():
                return f"{c}_USDT"
        if "_USDT" in task.upper():
            return task.upper()
        return f"{task.upper()}_USDT"
