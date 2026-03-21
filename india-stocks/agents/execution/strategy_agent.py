"""StrategyAgent — LLM ranks signals and provides market commentary."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from agents.base import BaseAgent, llm_call

SYSTEM_PROMPT = (
    "You are an expert Indian stock market analyst. Given signal data for a stock/index, "
    "provide a concise trading rationale in 2-3 sentences. Focus on key signals. "
    "Be direct and actionable."
)

class StrategyAgent(BaseAgent):
    name = "StrategyAgent"

    def run(self, task: dict) -> dict:
        symbol = task.get("symbol")
        try:
            from precision_verdict import VerdictEngine
            engine  = VerdictEngine()
            verdict = engine.get_verdict(symbol)
            sigs    = {s["name"]: s["value"] for s in verdict.get("signals", [])}

            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Symbol: {symbol}\n"
                f"Verdict: {verdict.get('verdict')} (score={verdict.get('score')})\n"
                f"Key signals: PCR={sigs.get('pcr',0):.2f}, "
                f"VIX={sigs.get('india_vix',0):.2f}, "
                f"FII={sigs.get('fii_net',0):.2f}, "
                f"ML_UP={verdict.get('ml_p_up',0):.2f}, "
                f"ML_DN={verdict.get('ml_p_down',0):.2f}\n"
                f"Write 2-3 sentence rationale:"
            )
            rationale = llm_call(prompt, max_tokens=150)
            self.remember(f"rationale_{symbol}", {"text": rationale, "verdict": verdict.get("verdict")})
            return {"status": "ok", "rationale": rationale}
        except Exception as e:
            return {"status": "error", "error": str(e)}
