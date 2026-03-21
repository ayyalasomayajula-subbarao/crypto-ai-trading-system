"""DiscoveryAgent — Mini backtests with param variations to find improvements."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from agents.base import BaseAgent
import logging
log = logging.getLogger(__name__)

class DiscoveryAgent(BaseAgent):
    name = "DiscoveryAgent"

    # Param variations to try
    VARIATIONS = [
        {"tp_pct_mult": 1.2, "sl_pct_mult": 1.0, "label": "wider_tp"},
        {"tp_pct_mult": 0.8, "sl_pct_mult": 0.8, "label": "tighter"},
        {"tp_pct_mult": 1.0, "sl_pct_mult": 1.2, "label": "wider_sl"},
    ]

    def run(self, task: dict) -> dict:
        symbol = task.get("symbol")
        results = []
        best = None

        for var in self.VARIATIONS:
            try:
                from walk_forward import run_wf
                from config import INSTRUMENTS
                orig_tp = INSTRUMENTS[symbol]["tp_pct"]
                orig_sl = INSTRUMENTS[symbol]["sl_pct"]
                INSTRUMENTS[symbol]["tp_pct"] = orig_tp * var["tp_pct_mult"]
                INSTRUMENTS[symbol]["sl_pct"] = orig_sl * var["sl_pct_mult"]

                r = run_wf(symbol)
                results.append({"label": var["label"], **r})

                # Restore
                INSTRUMENTS[symbol]["tp_pct"] = orig_tp
                INSTRUMENTS[symbol]["sl_pct"] = orig_sl

            except Exception as e:
                log.error(f"Discovery var {var}: {e}")

        if results:
            best = max(results, key=lambda x: x.get("avg_sharpe", -99))
            self.remember(f"discovery_{symbol}", {"best": best, "all": results})

        return {"status": "ok", "best_variant": best.get("label") if best else None,
                "best_sharpe": best.get("avg_sharpe") if best else None}
