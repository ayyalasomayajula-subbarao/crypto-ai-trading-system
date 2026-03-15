"""
RiskAgent — portfolio performance analyser and degradation detector.
Reads data/paper_trading_state.json and computes Sharpe, WR, drawdown per coin.
Flags degraded coins that need optimizer attention.
"""

import os
import sys
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from agents.base import BaseAgent
from agents.memory import memory

STATE_PATH = os.path.join(BASE_DIR, "data", "paper_trading_state.json")


class RiskAgent(BaseAgent):
    SYSTEM_PROMPT = (
        "You are a risk management agent. Analyse portfolio performance metrics "
        "and flag coins whose model performance has degraded."
    )

    def __init__(self):
        super().__init__("RiskAgent", self.SYSTEM_PROMPT)

    def run(self, task: str = "report") -> dict:
        if not os.path.exists(STATE_PATH):
            return {"status": "no_state", "message": "Paper trading not started yet"}

        with open(STATE_PATH) as f:
            state = json.load(f)

        # Cache state for StrategyAgent
        memory.store("paper_trading_state", state, agent_name="RiskAgent")

        trades        = state.get("trades", [])
        equity_curve  = state.get("equity_curve", [])
        positions     = state.get("positions", {})
        initial       = state.get("config", {}).get("initial_capital", 10000)
        current       = state.get("equity", initial)

        metrics     = self._portfolio_metrics(trades, equity_curve, initial, current)
        per_coin    = self._per_coin_metrics(trades)
        degraded    = self._detect_degradation(per_coin, metrics)
        weights     = self._compute_portfolio_weights()

        result = {
            "status":           "ok",
            "equity":           round(current, 2),
            "total_return_pct": round((current - initial) / initial * 100, 2),
            "open_positions":   len(positions),
            "metrics":          metrics,
            "per_coin":         per_coin,
            "degraded_coins":   degraded,
            "alerts":           [f"{c}: {r}" for c, r in degraded.items()],
            "portfolio_weights": weights,
        }

        self.log(
            f"Portfolio: ${current:.0f} ({result['total_return_pct']:+.1f}%) | "
            f"{len(trades)} trades | {len(degraded)} degraded coin(s)"
        )
        return result

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _portfolio_metrics(self, trades, equity_curve, initial, current) -> dict:
        if not trades:
            return {"total_trades": 0, "win_rate": 0, "sharpe": 0, "max_drawdown_pct": 0, "profit_factor": 0}

        wins   = [t for t in trades if t.get("result") == "WIN"]
        losses = [t for t in trades if t.get("result") == "LOSS"]

        win_rate     = len(wins) / len(trades) * 100
        gross_profit = sum(t.get("net_pnl_pct", 0) for t in wins)
        gross_loss   = abs(sum(t.get("net_pnl_pct", 0) for t in losses))
        pf           = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        sharpe = 0.0
        if equity_curve and len(equity_curve) >= 5:
            eq   = [e["equity"] for e in equity_curve]
            rets = np.diff(eq) / np.array(eq[:-1])
            if rets.std() > 0:
                sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(365))

        max_dd = 0.0
        if equity_curve:
            eq   = [e["equity"] for e in equity_curve]
            peak = eq[0]
            for v in eq:
                peak  = max(peak, v)
                max_dd = min(max_dd, (v - peak) / peak * 100)

        return {
            "total_trades":    len(trades),
            "wins":            len(wins),
            "losses":          len(losses),
            "win_rate":        round(win_rate, 1),
            "profit_factor":   round(pf, 2),
            "sharpe":          round(sharpe, 3),
            "max_drawdown_pct": round(max_dd, 1),
        }

    def _per_coin_metrics(self, trades) -> dict:
        coins = {t.get("coin") for t in trades if t.get("coin")}
        result = {}
        for coin in coins:
            ct   = [t for t in trades if t.get("coin") == coin]
            wins = [t for t in ct if t.get("result") == "WIN"]
            pnls = [t.get("net_pnl_pct", 0) for t in ct]
            result[coin] = {
                "trades":   len(ct),
                "win_rate": round(len(wins) / len(ct) * 100, 1) if ct else 0,
                "avg_pnl":  round(float(np.mean(pnls)), 2) if pnls else 0,
            }
        return result

    def _compute_portfolio_weights(self) -> dict:
        """
        Quality-weighted capital allocation using WF model metrics from SQLite.

        Score per coin = Sharpe × WinRate (both from last backtest).
        Coins with no model or negative Sharpe get weight 0.
        Weights sum to 1.0.

        Example: BTC Sharpe=0.37 WR=58% → score=0.215
                 SOL Sharpe=0.30 WR=44% → score=0.133
                 Equal weight would give each 25% — quality-weighting tilts toward BTC.
        """
        TRADEABLE = ["BTC_USDT", "SOL_USDT", "AVAX_USDT", "LINK_USDT"]
        scores = {}

        for coin in TRADEABLE:
            model_info = memory.get_active_model(coin)
            if model_info:
                sharpe = model_info.get("sharpe", 0.0)
                wr     = model_info.get("wr", 0.0) / 100.0  # convert % to fraction
                score  = max(0.0, sharpe) * max(0.0, wr)
                scores[coin] = score
            else:
                scores[coin] = 0.0

        total = sum(scores.values())
        if total == 0:
            # No model data yet — fall back to equal weight
            equal = round(1.0 / len(TRADEABLE), 3)
            return {c: equal for c in TRADEABLE}

        return {
            coin: round(score / total, 3)
            for coin, score in scores.items()
        }

    def _detect_degradation(self, per_coin, metrics) -> dict:
        degraded = {}

        if metrics["total_trades"] >= 10:
            if metrics["sharpe"] < 0.2:
                degraded["PORTFOLIO"] = f"Sharpe={metrics['sharpe']:.3f} < 0.2"
            if metrics["win_rate"] < 40:
                degraded["PORTFOLIO_WR"] = f"WR={metrics['win_rate']:.1f}% < 40%"

        for coin, m in per_coin.items():
            if m["trades"] >= 5 and m["win_rate"] < 35:
                degraded[coin] = f"WR={m['win_rate']}% < 35%"

        return degraded
