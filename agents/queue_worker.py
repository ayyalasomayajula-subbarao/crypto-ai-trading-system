"""
TaskQueue — decoupled task queue with timeout protection. (Upgrade 1 + 2)

Queen enqueues subtasks → workers dequeue and execute agents concurrently.

Timeout: daemon thread + join(timeout) — avoids nested-future pool deadlock.
CPU-bound tasks (BacktestAgent, StrategyDiscoveryAgent): routed to
ProcessPoolExecutor so LightGBM training bypasses the Python GIL.
"""

import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Dict, List, Optional

# CPU_BOUND_AGENTS: these bypass the GIL by running in a subprocess.
# The agent is re-instantiated inside the process — no pickling of LLM clients.
CPU_BOUND_AGENTS = {"BacktestAgent", "StrategyDiscoveryAgent"}


def _run_agent_in_process(agent_class_name: str, task_args) -> Any:
    """
    Module-level function (required for ProcessPoolExecutor pickling).
    Re-instantiates the agent inside the subprocess — avoids pickling LLM clients.
    """
    from agents.queen import AGENT_REGISTRY
    cls = AGENT_REGISTRY.get(agent_class_name)
    if cls is None:
        return {"error": f"unknown agent in subprocess: {agent_class_name}"}
    agent = cls()
    args = task_args if isinstance(task_args, tuple) else (task_args,)
    return agent.run(*args)


class TaskQueue:

    def __init__(self, max_workers: int = 8):
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="AgentWorker",
        )
        self._cpu_executor = ProcessPoolExecutor(max_workers=2)
        self._futures: Dict[str, Any] = {}
        self._lock = threading.Lock()

    # ── Submit ────────────────────────────────────────────────────────────────

    def submit(
        self,
        agent_instance,
        task_args,
        timeout: int = 120,
    ) -> str:
        """
        Submit one agent job.
        Returns task_id immediately (non-blocking).
        """
        task_id = uuid.uuid4().hex[:8]
        future = self._executor.submit(
            self._run_with_timeout, agent_instance, task_args, timeout
        )
        with self._lock:
            self._futures[task_id] = future
        return task_id

    def _run_with_timeout(self, agent, task_args, timeout: int) -> Any:
        """
        Enforce per-task timeout using a daemon thread + join(timeout).

        NOT nested futures — nested self._executor.submit() inside a worker
        can deadlock when the pool is saturated (outer holds a slot, inner
        can never get one).  A daemon thread has no pool quota.
        """
        _result: list = [None]
        _exc:    list = [None]
        agent_name = getattr(agent, "name", type(agent).__name__)

        def _target():
            try:
                args = task_args if isinstance(task_args, tuple) else (task_args,)
                _result[0] = agent.run(*args)
            except Exception as e:
                _exc[0] = e

        t = threading.Thread(target=_target, daemon=True, name=f"Agent-{agent_name}")
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            # Thread still running — timeout elapsed
            return {"error": f"timeout after {timeout}s", "agent": agent_name}

        if _exc[0] is not None:
            return {"error": str(_exc[0]), "agent": agent_name}

        return _result[0]

    # ── Collect ───────────────────────────────────────────────────────────────

    def get_result(
        self,
        task_id: str,
        block: bool = True,
        timeout: int = 300,
    ) -> Any:
        """Wait for result. Returns result dict or error dict."""
        with self._lock:
            future = self._futures.get(task_id)
        if future is None:
            return {"error": f"unknown task_id: {task_id}"}
        try:
            return future.result(timeout=timeout if block else 0.001)
        except FutureTimeout:
            return {"error": "still running"}
        except Exception as exc:
            return {"error": str(exc)}

    # ── Batch ─────────────────────────────────────────────────────────────────

    def submit_batch(self, subtasks: List[Dict]) -> Dict[str, str]:
        """
        Submit a planner-produced subtask list.
        parallel=True items run concurrently.
        parallel=False items run sequentially after all parallel items finish.
        CPU-bound agents (Backtest, Discovery) run in ProcessPoolExecutor.

        Returns {agent_name: task_id}.
        """
        from agents.queen import AGENT_REGISTRY

        parallel_tasks   = [s for s in subtasks if s.get("parallel", True)]
        sequential_tasks = [s for s in subtasks if not s.get("parallel", True)]

        task_map: Dict[str, str] = {}

        def _submit_one(s: Dict) -> Optional[str]:
            agent_name = s["agent"]
            task_args  = s.get("task", "")
            timeout    = s.get("timeout", 120)

            if agent_name in CPU_BOUND_AGENTS:
                # Run in subprocess — bypasses GIL for CPU-heavy ML work
                return self._submit_cpu(agent_name, task_args, timeout)
            else:
                agent_cls = AGENT_REGISTRY.get(agent_name)
                if agent_cls is None:
                    return None
                return self.submit(agent_cls(), task_args, timeout=timeout)

        # Submit all parallel tasks at once
        for s in parallel_tasks:
            tid = _submit_one(s)
            if tid:
                task_map[s["agent"]] = tid

        # Wait for all parallel tasks before starting sequential
        for tid in list(task_map.values()):
            self.get_result(tid, block=True, timeout=600)

        # Run sequential tasks one by one
        for s in sequential_tasks:
            tid = _submit_one(s)
            if tid:
                task_map[s["agent"]] = tid
                self.get_result(tid, block=True, timeout=s.get("timeout", 600))

        return task_map

    def _submit_cpu(self, agent_class_name: str, task_args, timeout: int) -> str:
        """Submit a CPU-bound task to ProcessPoolExecutor."""
        task_id = uuid.uuid4().hex[:8]
        future = self._cpu_executor.submit(
            _run_agent_in_process, agent_class_name, task_args
        )
        with self._lock:
            self._futures[task_id] = future
        return task_id

    def collect_results(self, task_map: Dict[str, str]) -> Dict[str, Any]:
        """Collect all finished results keyed by agent name."""
        return {name: self.get_result(tid) for name, tid in task_map.items()}

    def shutdown(self, wait: bool = False):
        self._executor.shutdown(wait=wait)
        self._cpu_executor.shutdown(wait=wait)
