"""
QueenAgent — top-level orchestrator.

Architecture:
  Queen → PlannerAgent (decompose task)
        → TaskQueue.submit_batch() (parallel + sequential execution)
        → collect results
        → store in AgentMemory
        → return consolidated dict

AGENT_REGISTRY maps string names (from Planner JSON) to agent classes.
Bug fix vs user skeleton: Planner returns strings, not instances — registry resolves them.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from agents.base import BaseAgent
from agents.memory import memory
from agents.planner import PlannerAgent
from agents.queue_worker import TaskQueue

# Lazy imports to avoid circular dependency issues at module load time
from agents.execution.data_agent import DataAgent
from agents.execution.feature_agent import FeatureAgent
from agents.execution.signal_agent import SignalAgent
from agents.execution.strategy_agent import StrategyAgent
from agents.evaluation.backtest_agent import BacktestAgent
from agents.evaluation.risk_agent import RiskAgent
from agents.evaluation.optimizer_agent import OptimizerAgent
from agents.evaluation.discovery_agent import StrategyDiscoveryAgent
from agents.monitor import MonitorAgent

# ── AGENT_REGISTRY: string → class (Upgrade 3 bug fix) ───────────────────────
AGENT_REGISTRY = {
    "DataAgent":               DataAgent,
    "FeatureAgent":            FeatureAgent,
    "SignalAgent":             SignalAgent,
    "StrategyAgent":           StrategyAgent,
    "BacktestAgent":           BacktestAgent,
    "RiskAgent":               RiskAgent,
    "OptimizerAgent":          OptimizerAgent,
    "StrategyDiscoveryAgent":  StrategyDiscoveryAgent,
    "MonitorAgent":            MonitorAgent,
}


class QueenAgent(BaseAgent):
    SYSTEM_PROMPT = (
        "You are the master orchestrator for a crypto trading AI system. "
        "Route tasks to specialized agents and consolidate results."
    )

    def __init__(self):
        super().__init__("Queen", self.SYSTEM_PROMPT)
        self.planner = PlannerAgent()
        self.queue   = TaskQueue(max_workers=8)

    def run(self, task: str) -> dict:
        """
        Orchestrate a high-level task end-to-end.
        Returns consolidated result dict.
        """
        import uuid
        task_id = uuid.uuid4().hex[:8]
        self.log(f"Task [{task_id}]: {task!r}")

        # 1. Decompose task into agent jobs
        subtasks = self.planner.decompose(task)
        self.log(f"Plan ({len(subtasks)} jobs): {[s['agent'] for s in subtasks]}")

        if not subtasks:
            return {"task": task, "error": "Planner returned empty plan"}

        # 2. Execute via TaskQueue (parallel + sequential handled internally)
        task_map = self.queue.submit_batch(subtasks)

        # 3. Collect results — separate successes from errors
        raw = self.queue.collect_results(task_map)
        results, errors = self._split_results(raw)

        # 4. Store in memory (successful results only)
        memory.store("last_task_result", results, agent_name="Queen", task_id=task_id)
        for agent_name, result in results.items():
            memory.summarize_and_store(agent_name, result, task_id)

        # 5. Return consolidated output with clear error visibility
        consolidated = {
            "task":    task,
            "task_id": task_id,
            "agents":  list(results.keys()),
            "results": results,
        }
        if errors:
            consolidated["errors"] = errors
            consolidated["partial"] = True
            self.log(f"Task [{task_id}] partial — {len(errors)} error(s): {list(errors.keys())}")
        else:
            self.log(f"Task [{task_id}] complete — {len(results)} agent(s) finished")

        return consolidated

    @staticmethod
    def _split_results(raw: dict) -> tuple:
        """
        Separate agent results into (successes, errors).
        An agent result is an error if it's a dict with only an 'error' key,
        or if it has both 'error' and 'agent' keys (timeout/crash pattern).
        """
        results = {}
        errors  = {}
        for name, result in raw.items():
            if isinstance(result, dict) and "error" in result and len(result) <= 2:
                errors[name] = result["error"]
            else:
                results[name] = result
        return results, errors

    def shutdown(self):
        self.queue.shutdown()
