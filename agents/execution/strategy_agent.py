"""
StrategyAgent — LLM-powered signal ranker and trade recommendation engine.

Reads last_scan from memory, asks Groq to rank by risk-adjusted priority,
returns ranked recommendations + best_trade.
"""

import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from agents.base import BaseAgent
from agents.memory import memory


class StrategyAgent(BaseAgent):
    SYSTEM_PROMPT = (
        "You are a crypto strategy analyst. Given ML model signals and portfolio state, "
        "rank trade opportunities by risk-adjusted priority. "
        "Consider: signal probability, ADX trend strength, portfolio concentration, regime. "
        "Return JSON: {\"ranked\": [...signals...], \"best_trade\": {...}, \"summary\": \"...\"}"
    )

    def __init__(self):
        super().__init__("StrategyAgent", self.SYSTEM_PROMPT)

    def run(self, task: str = "rank") -> dict:
        signals   = memory.retrieve("last_scan") or []
        portfolio = memory.retrieve("paper_trading_state") or {}

        if not signals:
            return {"error": "no signals in memory — run SignalAgent first"}

        actionable = [s for s in signals if s.get("verdict") in ["LONG", "SHORT"]]
        waiting    = [s for s in signals if s.get("verdict") == "WAIT"]

        if not actionable:
            return {
                "ranked": [],
                "best_trade": None,
                "summary": f"No actionable signals. {len(waiting)} coins in WAIT state.",
            }

        # Load quality weights from last RiskAgent run (or compute fallback)
        weights = self._get_portfolio_weights()

        # Attach weight and composite score to each signal
        for s in actionable:
            coin   = s.get("coin", "")
            w      = weights.get(coin, weights.get(coin.replace("_USDT", "") + "_USDT", 0.25))
            prob   = s.get("signal_prob", 0.5)
            adx_n  = min(s.get("adx", 20), 60) / 60  # normalise ADX 0→1
            s["portfolio_weight"]  = w
            s["composite_score"]   = round(prob * w * (1 + adx_n), 4)

        # Sort by composite score descending (pre-LLM ranking)
        actionable.sort(key=lambda x: -x["composite_score"])

        # Compact signal summary for LLM (include weight + score)
        sig_compact = json.dumps(
            [
                {k: v for k, v in s.items()
                 if k in ["coin", "verdict", "signal_prob", "adx", "regime_ok",
                          "portfolio_weight", "composite_score"]}
                for s in actionable
            ],
            indent=2,
        )

        positions = list(portfolio.get("positions", {}).keys())
        equity    = portfolio.get("equity", "unknown")
        context   = (
            f"Open positions: {positions}. Portfolio equity: {equity}. "
            f"Weights reflect model quality (Sharpe × WinRate) — higher weight = better model."
        )

        prompt = (
            f"Rank these {len(actionable)} actionable signals by risk-adjusted priority.\n"
            f"composite_score already incorporates model quality weights:\n{sig_compact}\n"
            "Return JSON only."
        )

        result = self.think_json(prompt, context=context, max_tokens=600)

        # Fallback if LLM fails or returns empty
        if not result or not isinstance(result, dict):
            return {
                "ranked":           actionable,
                "best_trade":       actionable[0],
                "portfolio_weights": weights,
                "summary":          f"{len(actionable)} signals ranked by composite score (model quality × probability × ADX)",
            }

        # Ensure best_trade and weights are always present
        if not result.get("best_trade") and actionable:
            result["best_trade"] = actionable[0]
        result["portfolio_weights"] = weights

        return result

    def _get_portfolio_weights(self) -> dict:
        """
        Retrieve quality weights from last RiskAgent output in memory.
        Falls back to equal weight (0.25) if not yet computed.
        """
        risk_output = memory.retrieve("RiskAgent_output") or {}
        weights = risk_output.get("portfolio_weights", {})
        if weights:
            return weights

        # Equal weight fallback
        TRADEABLE = ["BTC_USDT", "SOL_USDT", "AVAX_USDT", "LINK_USDT"]
        return {c: 0.25 for c in TRADEABLE}
