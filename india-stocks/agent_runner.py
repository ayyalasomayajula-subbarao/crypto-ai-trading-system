"""
India Stocks Agent Runner — CLI entry point

Usage:
    python agent_runner.py scan
    python agent_runner.py scan --symbol NIFTY50
    python agent_runner.py report
    python agent_runner.py update-data
    python agent_runner.py backtest --symbol NIFTY50
    python agent_runner.py optimize --symbol BANKNIFTY
    python agent_runner.py self-improve
    python agent_runner.py discover --symbol RELIANCE
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="India Stocks AI Agent Runner")
    parser.add_argument("command", choices=[
        "scan", "report", "update-data", "backtest",
        "optimize", "discover", "self-improve",
    ])
    parser.add_argument("--symbol", default=None,
                        help="Target symbol (e.g. NIFTY50, BANKNIFTY, RELIANCE)")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON")
    args = parser.parse_args()

    from agents.queen import QueenAgent
    queen = QueenAgent()

    log.info(f"Running: {args.command}  symbol={args.symbol or 'ALL'}")
    results = queen.route(args.command, symbol=args.symbol)

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print(f"\n{'='*60}")
        print(f"Command: {args.command}  |  Symbol: {args.symbol or 'ALL'}")
        print(f"{'='*60}")
        for r in results:
            status = r.get("result", {}).get("status", "?")
            sym    = r.get("symbol", "")
            agent  = r.get("agent", "")
            extra  = {k: v for k, v in r.get("result", {}).items()
                      if k != "status" and v is not None}
            print(f"  [{agent}] {sym:14} → {status}  {extra}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
