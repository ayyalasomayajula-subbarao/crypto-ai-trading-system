#!/usr/bin/env python3
"""
agent_runner.py — CLI entry point for the multi-agent trading system.

Usage:
  python agent_runner.py scan
  python agent_runner.py report
  python agent_runner.py analyze BTC
  python agent_runner.py backtest BTC
  python agent_runner.py retrain SOL
  python agent_runner.py optimize AVAX
  python agent_runner.py discover BTC
  python agent_runner.py update-data
  python agent_runner.py self-improve
  python agent_runner.py memory-history BTC   (show model version history)
  python agent_runner.py memory-search <keyword>

Direct agent shortcuts (bypass Queen):
  python agent_runner.py --agent SignalAgent scan
  python agent_runner.py --agent RiskAgent report

Cron mode (structured logs, exit code on error):
  python agent_runner.py --cron scan
  python agent_runner.py --cron update-data
"""

import sys
import json
import logging
import argparse
from datetime import datetime

# ── Cron logger — writes structured lines to stdout (captured by >> log) ─────
_cron_log = logging.getLogger("cron")


def _setup_cron_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [CRON] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    _cron_log.addHandler(handler)
    _cron_log.setLevel(logging.INFO)


def print_result(result, indent: bool = True):
    """Pretty-print a result dict."""
    print(json.dumps(result, indent=2 if indent else None, default=str))


def run_direct(agent_name: str, task: str, cron: bool = False):
    """Bypass Queen — run a single agent directly."""
    from agents.queen import AGENT_REGISTRY
    cls = AGENT_REGISTRY.get(agent_name)
    if cls is None:
        print(f"Unknown agent: {agent_name}")
        print(f"Available: {list(AGENT_REGISTRY.keys())}")
        sys.exit(1)
    agent = cls()
    result = agent.run(task)
    if cron:
        _cron_summarize(agent_name, result)
    else:
        print_result(result)


def run_queen(task: str, cron: bool = False):
    """Run full orchestration via QueenAgent."""
    from agents import run_task

    if cron:
        _cron_log.info(f"START task={task!r}")
        try:
            result = run_task(task)
            _cron_summarize(task, result)
            # Non-zero exit if any agent errored
            if result.get("partial"):
                _cron_log.warning(f"PARTIAL errors={list(result.get('errors', {}).keys())}")
                sys.exit(2)
            _cron_log.info("DONE ok")
        except Exception as e:
            _cron_log.error(f"FATAL {e}")
            sys.exit(1)
    else:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting task: {task!r}\n")
        result = run_task(task)
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done.\n")
        print_result(result["results"] if "results" in result else result)


def _cron_summarize(label: str, result: dict):
    """Log a one-line summary of the result suitable for cron logs."""
    if not isinstance(result, dict):
        _cron_log.info(f"{label}: {str(result)[:120]}")
        return

    # Pull the most useful top-level scalars
    sub = result.get("results", result)
    summary_parts = []

    # SignalAgent — count verdicts
    signals = sub.get("SignalAgent", [])
    if isinstance(signals, list):
        longs  = sum(1 for s in signals if s.get("verdict") == "LONG")
        shorts = sum(1 for s in signals if s.get("verdict") == "SHORT")
        waits  = sum(1 for s in signals if s.get("verdict") == "WAIT")
        summary_parts.append(f"signals: {longs}L/{shorts}S/{waits}W")

    # RiskAgent — equity + sharpe
    risk = sub.get("RiskAgent", {})
    if isinstance(risk, dict) and "equity" in risk:
        eq  = risk.get("equity", 0)
        ret = risk.get("total_return_pct", 0)
        sh  = risk.get("metrics", {}).get("sharpe", 0)
        summary_parts.append(f"equity=${eq:.0f}({ret:+.1f}%) sharpe={sh:.3f}")

    # DataAgent — new candles
    data = sub.get("DataAgent", {})
    if isinstance(data, dict):
        candles = sum(
            v.get("new_candles", 0) for v in data.get("coins_updated", {}).values()
            if isinstance(v, dict)
        )
        summary_parts.append(f"new_candles={candles}")

    # Errors
    errors = result.get("errors", {})
    if errors:
        summary_parts.append(f"ERRORS={list(errors.keys())}")

    _cron_log.info(f"{label}: {' | '.join(summary_parts) or json.dumps(result, default=str)[:200]}")


def cmd_memory_history(coin: str):
    from agents.memory import memory
    c = coin.upper()
    if "_USDT" not in c:
        c += "_USDT"
    rows = memory.list_model_history(c)
    if not rows:
        print(f"No model history for {c}")
        return
    print(f"\nModel history for {c}:")
    print(f"{'Active':<8} {'Sharpe':<8} {'WR%':<7} {'Thresh':<8} {'Created':<22} Path")
    print("-" * 80)
    for r in rows:
        ts = datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d %H:%M") if r["created_at"] else "unknown"
        active = "✅" if r["active"] else "  "
        print(f"{active:<8} {r['sharpe']:<8.3f} {r['wr']:<7.1f} {r['threshold']:<8.2f} {ts:<22} {r['model_path']}")


def cmd_memory_search(keyword: str):
    from agents.memory import memory
    results = memory.search(keyword, limit=10)
    if not results:
        print(f"No memory entries matching '{keyword}'")
        return
    print(f"\nMemory search results for '{keyword}':")
    print_result(results)


def main():
    parser = argparse.ArgumentParser(description="Crypto AI Multi-Agent System")
    parser.add_argument("task",   nargs="+", help="Task or command")
    parser.add_argument("--agent", default=None, help="Run specific agent directly")
    parser.add_argument("--json",  action="store_true", help="Output raw JSON (no indent)")
    parser.add_argument("--cron",  action="store_true",
                        help="Cron mode: structured logs, non-zero exit on error")
    args = parser.parse_args()

    if args.cron:
        _setup_cron_logging()

    task_str = " ".join(args.task)

    # Special commands (not meaningful in cron mode, but harmless)
    if args.task[0] == "memory-history":
        coin = args.task[1] if len(args.task) > 1 else "BTC"
        cmd_memory_history(coin)
        return

    if args.task[0] == "memory-search":
        keyword = args.task[1] if len(args.task) > 1 else ""
        cmd_memory_search(keyword)
        return

    if args.task[0] == "agents":
        from agents.queen import AGENT_REGISTRY
        print("Available agents:")
        for name in sorted(AGENT_REGISTRY.keys()):
            print(f"  {name}")
        return

    # Direct agent invocation
    if args.agent:
        task_arg = " ".join(args.task)
        run_direct(args.agent, task_arg, cron=args.cron)
        return

    # Full orchestration via Queen
    run_queen(task_str, cron=args.cron)


if __name__ == "__main__":
    main()
