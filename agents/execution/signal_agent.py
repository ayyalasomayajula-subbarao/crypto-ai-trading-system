"""
SignalAgent — real-time signal generator for all 7 coins.

Loads wf_decision_model_v2.pkl per coin, reads last row of feature CSV,
runs predict_proba(), applies WF-validated thresholds + ADX/regime gates.
Scans 7 coins in parallel via ThreadPoolExecutor.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from agents.base import BaseAgent
from agents.memory import memory

DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# WF-validated thresholds (from walk_forward_validation results)
# Update these after running a new WF round
COIN_THRESHOLDS = {
    "BTC_USDT":  0.60,
    "ETH_USDT":  0.65,
    "SOL_USDT":  0.55,
    "PEPE_USDT": 0.55,
    "AVAX_USDT": 0.60,
    "BNB_USDT":  0.55,
    "LINK_USDT": 0.55,
}

ALL_COINS = list(COIN_THRESHOLDS.keys())


class SignalAgent(BaseAgent):
    SYSTEM_PROMPT = (
        "You are a signal generation agent for a crypto trading system. "
        "Load ML models and produce ranked trade signals with direction and confidence."
    )

    def __init__(self):
        super().__init__("SignalAgent", self.SYSTEM_PROMPT)

    def run(self, task: str = "scan") -> list:
        """
        Return ranked signal list for all (or specific) coins.
        Signals sorted: LONG > SHORT > WAIT, then by signal_prob desc.
        """
        coins = self._parse_coins(task)
        self.log(f"Scanning {len(coins)} coin(s) in parallel")

        with ThreadPoolExecutor(max_workers=min(7, len(coins))) as ex:
            futures = {ex.submit(self._scan_coin, c): c for c in coins}
            results = []
            for f in as_completed(futures):
                r = f.result()
                if r:
                    results.append(r)

        # Sort: actionable first, then by signal probability descending
        priority = {"LONG": 0, "SHORT": 1, "WAIT": 2, "ERROR": 3}
        results.sort(
            key=lambda x: (priority.get(x.get("verdict", "ERROR"), 3), -x.get("signal_prob", 0))
        )

        # Store in memory for StrategyAgent to consume
        memory.store("last_scan", results, agent_name="SignalAgent")
        self.log(f"Scan complete: {sum(1 for r in results if r['verdict'] in ['LONG','SHORT'])} actionable signals")

        return results

    def _scan_coin(self, coin: str) -> dict:
        try:
            csv_path = os.path.join(DATA_DIR, f"{coin}_multi_tf_features.csv")
            if not os.path.exists(csv_path):
                return {"coin": coin, "verdict": "ERROR", "error": "no feature CSV"}

            df = pd.read_csv(csv_path, parse_dates=["timestamp"]).sort_values("timestamp")
            last = df.iloc[-1]
            ts = str(last.get("timestamp", "unknown"))[:19]

            # Load model — prefer v2
            model = self._load_model(coin)
            if model is None:
                return {"coin": coin, "verdict": "ERROR", "error": "no model found"}

            feature_cols = self._load_features(coin)
            if not feature_cols:
                return {"coin": coin, "verdict": "ERROR", "error": "no features file"}

            available = [c for c in feature_cols if c in df.columns]
            if len(available) < len(feature_cols) * 0.8:
                return {
                    "coin": coin,
                    "verdict": "ERROR",
                    "error": f"only {len(available)}/{len(feature_cols)} features available",
                }

            X = pd.DataFrame([last[available]], columns=available)
            probs = model.predict_proba(X)[0]
            classes = list(model.classes_)

            up_prob = down_prob = sideways_prob = 0.0
            for i, cls in enumerate(classes):
                cs = str(cls).upper()
                if cs in ["UP", "1", "WIN"]:
                    up_prob = float(probs[i])
                elif cs in ["DOWN", "-1", "LOSS"]:
                    down_prob = float(probs[i])
                elif cs in ["SIDEWAYS", "0", "NEUTRAL"]:
                    sideways_prob = float(probs[i])

            threshold   = COIN_THRESHOLDS.get(coin, 0.55)
            adx         = float(last.get("1h_adx", 0))
            adx_ok      = adx >= 20
            regime_col  = "1w_dist_sma_50" if coin != "PEPE_USDT" else "1d_dist_sma_50"
            regime_ok   = float(last.get(regime_col, 0)) > 0
            close_price = float(last.get("close", 0))

            verdict     = "WAIT"
            direction   = None
            signal_prob = max(up_prob, down_prob)

            if adx_ok:
                if up_prob >= threshold and down_prob >= threshold:
                    direction = "LONG" if up_prob >= down_prob else "SHORT"
                elif up_prob >= threshold:
                    direction = "LONG"
                elif down_prob >= threshold:
                    direction = "SHORT"

                if direction:
                    verdict = direction if regime_ok else "WAIT"

            return {
                "coin":          coin,
                "verdict":       verdict,
                "direction":     direction,
                "up_prob":       round(up_prob * 100, 1),
                "down_prob":     round(down_prob * 100, 1),
                "sideways_prob": round(sideways_prob * 100, 1),
                "signal_prob":   round(signal_prob * 100, 1),
                "threshold_pct": int(threshold * 100),
                "adx":           round(adx, 1),
                "adx_ok":        adx_ok,
                "regime_ok":     regime_ok,
                "price":         round(close_price, 8),
                "timestamp":     ts,
            }

        except Exception as exc:
            self.log(f"{coin}: {exc}")
            return {"coin": coin, "verdict": "ERROR", "error": str(exc)}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_model(self, coin: str):
        coin_dir = os.path.join(MODELS_DIR, coin)
        for fname in ["wf_decision_model_v2.pkl", "decision_model_v2.pkl",
                      "wf_decision_model.pkl", "decision_model.pkl"]:
            p = os.path.join(coin_dir, fname)
            if os.path.exists(p):
                return joblib.load(p)
        return None

    def _load_features(self, coin: str) -> list:
        coin_dir = os.path.join(MODELS_DIR, coin)
        for fname in ["decision_features_v2.txt", "decision_features.txt", "feature_list.txt"]:
            p = os.path.join(coin_dir, fname)
            if os.path.exists(p):
                with open(p) as f:
                    return [line.strip() for line in f if line.strip()]
        return []

    def _parse_coins(self, task: str) -> list:
        for c in ["BTC", "ETH", "SOL", "PEPE", "AVAX", "BNB", "LINK"]:
            if c.upper() in task.upper():
                return [f"{c}_USDT"]
        return ALL_COINS
