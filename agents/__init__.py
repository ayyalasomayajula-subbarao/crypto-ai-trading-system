"""
agents/ — Hierarchical multi-agent trading system.

Architecture:
  QueenAgent
    ├── PlannerAgent          (task decomposition, JSON contracts)
    ├── AgentMemory           (SQLite: memory + model_versions)
    ├── TaskQueue             (parallel + sequential execution, timeout protection)
    ├── Execution Hub
    │     DataAgent           (market data update)
    │     FeatureAgent        (multi-TF feature engineering)
    │     SignalAgent         (ML signal generation, 7 coins parallel)
    │     StrategyAgent       (LLM signal ranking)
    └── Evaluation Hub
          BacktestAgent       (walk-forward validation)
          RiskAgent           (portfolio risk/performance)
          OptimizerAgent      (threshold sweep on saved model)
          StrategyDiscovery   (LLM-driven param discovery + mini-backtest)
          MonitorAgent        (self-improve loop)

Quick start:
  from agents.queen import QueenAgent
  result = QueenAgent().run("scan")
"""

from agents.queen import QueenAgent, AGENT_REGISTRY
from agents.memory import memory, AgentMemory

__all__ = ["QueenAgent", "AGENT_REGISTRY", "memory", "AgentMemory"]


def run_task(task: str) -> dict:
    """Convenience wrapper — create Queen, run task, shutdown, return result."""
    queen = QueenAgent()
    try:
        return queen.run(task)
    finally:
        queen.shutdown()
