"""
PlannerAgent — decomposes high-level tasks into structured agent job lists.

Upgrade 3: Structured JSON contracts — every job has agent/task/timeout/parallel.
           Uses json.loads() — NEVER eval() (code-injection risk).
           Falls back to hardcoded plans for common tasks (saves LLM call).
"""

import json
from agents.base import BaseAgent

# ── Hardcoded plans for common tasks (skip LLM, save tokens) ─────────────────
HARDCODED_PLANS = {
    "scan": [
        {"agent": "SignalAgent",   "task": "scan",   "timeout": 90,  "parallel": True},
        {"agent": "RiskAgent",     "task": "report", "timeout": 30,  "parallel": True},
    ],
    "report": [
        {"agent": "RiskAgent",     "task": "report", "timeout": 30, "parallel": False},
    ],
    "update-data": [
        {"agent": "DataAgent",    "task": "update",  "timeout": 300, "parallel": False},
        {"agent": "FeatureAgent", "task": "rebuild", "timeout": 300, "parallel": False},
    ],
    "update": [
        {"agent": "DataAgent",    "task": "update",  "timeout": 300, "parallel": False},
        {"agent": "FeatureAgent", "task": "rebuild", "timeout": 300, "parallel": False},
    ],
    "self-improve": [
        {"agent": "MonitorAgent", "task": "check",   "timeout": 120, "parallel": False},
    ],
}

# ── JSON contract schema (for documentation / validation) ────────────────────
#  Each subtask:
#  {
#    "agent":    str   — must be a key in AGENT_REGISTRY
#    "task":     str   — passed as the first arg to agent.run()
#    "timeout":  int   — seconds before task is cancelled
#    "parallel": bool  — True = run concurrently with other parallel tasks
#  }


class PlannerAgent(BaseAgent):
    SYSTEM_PROMPT = (
        "You are a task planner for a crypto trading AI system. "
        "Decompose the given task into a JSON array of agent jobs. "
        "Each job must be: "
        "{\"agent\": name, \"task\": description, \"timeout\": seconds, \"parallel\": bool}. "
        "Available agents: DataAgent FeatureAgent SignalAgent StrategyAgent "
        "BacktestAgent RiskAgent OptimizerAgent MonitorAgent StrategyDiscoveryAgent. "
        "Output ONLY valid JSON array. No markdown, no explanation."
    )

    def __init__(self):
        super().__init__("Planner", self.SYSTEM_PROMPT)

    # ── Public ────────────────────────────────────────────────────────────────

    def decompose(self, task: str) -> list:
        """
        Decompose task into a list of agent job dicts.
        Tries hardcoded plan first (common tasks), then LLM, then keyword fallback.
        """
        base_task = task.strip().lower().split()[0]

        # 1. Hardcoded plan (zero LLM tokens, no coin needed)
        if base_task in HARDCODED_PLANS:
            self.log(f"Using hardcoded plan for '{base_task}'")
            return HARDCODED_PLANS[base_task]

        # 1b. Keyword actions — skip LLM, preserve coin in task string
        KEYWORD_ACTIONS = {"backtest", "retrain", "optimize", "discover", "analyze"}
        if base_task in KEYWORD_ACTIONS:
            self.log(f"Using keyword plan for '{task}'")
            return self._keyword_plan(task)

        # 2. LLM decomposition
        prompt = (
            f"Task: {task}\n\n"
            "Return a minimal JSON array of agent jobs to complete this task. "
            "Use only the agents required."
        )
        result = self.think_json(prompt, max_tokens=400)

        if isinstance(result, list) and result:
            validated = self._validate(result)
            if validated:
                self.log(f"LLM plan: {len(validated)} jobs for '{task}'")
                return validated

        # 3. Keyword fallback
        self.log(f"Falling back to keyword plan for '{task}'")
        return self._keyword_plan(task)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _validate(self, plan: list) -> list:
        """Validate and fill defaults in a planner output list."""
        valid = []
        for item in plan:
            if not isinstance(item, dict):
                continue
            if "agent" not in item or "task" not in item:
                continue
            item.setdefault("timeout", 120)
            item.setdefault("parallel", True)
            valid.append(item)
        return valid

    def _keyword_plan(self, task: str) -> list:
        t = task.lower()
        coin = self._extract_coin(task)

        if "backtest" in t or "retrain" in t:
            return [{"agent": "BacktestAgent", "task": coin or task, "timeout": 600, "parallel": False}]

        if "optimize" in t:
            return [{"agent": "OptimizerAgent", "task": coin or task, "timeout": 300, "parallel": False}]

        if "discover" in t:
            return [{"agent": "StrategyDiscoveryAgent", "task": coin or task, "timeout": 600, "parallel": False}]

        if "analyze" in t:
            return [
                {"agent": "SignalAgent",   "task": coin or "all", "timeout": 60,  "parallel": True},
                {"agent": "RiskAgent",     "task": "report",       "timeout": 30,  "parallel": True},
                {"agent": "StrategyAgent", "task": "rank",         "timeout": 30,  "parallel": False},
            ]

        # Default: scan
        return [{"agent": "SignalAgent", "task": task, "timeout": 90, "parallel": False}]

    @staticmethod
    def _extract_coin(task: str) -> str:
        for c in ["BTC", "ETH", "SOL", "PEPE", "AVAX", "BNB", "LINK"]:
            if c.upper() in task.upper():
                return c
        return ""
