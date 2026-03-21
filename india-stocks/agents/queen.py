"""
QueenAgent — Routes tasks to the correct specialist agent.
Hardcoded plans for known tasks: scan / report / update-data / backtest / optimize.
"""
import logging
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.base import BaseAgent, llm_call
from config import ALL_SYMBOLS

log = logging.getLogger(__name__)

HARDCODED_PLANS = {
    "scan":        ["data_agent", "feature_agent", "signal_agent", "strategy_agent"],
    "report":      ["risk_agent"],
    "update-data": ["data_agent", "feature_agent"],
    "backtest":    ["backtest_agent"],
    "optimize":    ["optimizer_agent"],
    "discover":    ["discovery_agent"],
    "self-improve": ["backtest_agent", "optimizer_agent", "risk_agent"],
}


class QueenAgent(BaseAgent):
    name = "QueenAgent"

    def __init__(self):
        from agents.execution.data_agent     import DataAgent
        from agents.execution.feature_agent  import FeatureAgent
        from agents.execution.signal_agent   import SignalAgent
        from agents.execution.strategy_agent import StrategyAgent
        from agents.evaluation.backtest_agent  import BacktestAgent
        from agents.evaluation.risk_agent      import RiskAgent
        from agents.evaluation.optimizer_agent import OptimizerAgent
        from agents.evaluation.discovery_agent import DiscoveryAgent

        self.registry = {
            "data_agent":      DataAgent(),
            "feature_agent":   FeatureAgent(),
            "signal_agent":    SignalAgent(),
            "strategy_agent":  StrategyAgent(),
            "backtest_agent":  BacktestAgent(),
            "risk_agent":      RiskAgent(),
            "optimizer_agent": OptimizerAgent(),
            "discovery_agent": DiscoveryAgent(),
        }

    def route(self, command: str, symbol: str | None = None) -> list[dict]:
        plan = HARDCODED_PLANS.get(command.lower())
        if plan is None:
            # LLM-guided decomposition for unknown commands
            prompt = (
                f"India stocks trading system. Command: '{command}'. "
                f"Available agents: {list(self.registry.keys())}. "
                f"Return JSON list of agent names to call in order."
            )
            raw = llm_call(prompt, max_tokens=100)
            try:
                plan = json.loads(raw)
            except Exception:
                log.warning(f"LLM plan parse failed, defaulting to scan")
                plan = HARDCODED_PLANS["scan"]

        symbols = [symbol] if symbol else ALL_SYMBOLS
        results = []

        for agent_name in plan:
            agent = self.registry.get(agent_name)
            if not agent:
                log.warning(f"Unknown agent: {agent_name}")
                continue
            for sym in symbols:
                try:
                    r = agent.run({"symbol": sym, "command": command})
                    results.append({"agent": agent_name, "symbol": sym, "result": r})
                    log.info(f"[{agent_name}] {sym} → {r.get('status', 'ok')}")
                except Exception as e:
                    log.error(f"[{agent_name}] {sym} failed: {e}")
                    results.append({"agent": agent_name, "symbol": sym, "error": str(e)})

        return results

import json  # needed for LLM plan parse
