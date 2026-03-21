# Crypto AI Trading System — Agent Rules

## Project
AI-powered crypto trading system. Multi-agent hierarchical architecture.
7 supported coins: BTC_USDT ETH_USDT SOL_USDT PEPE_USDT AVAX_USDT BNB_USDT LINK_USDT

## Core Rules (ALL agents must follow)
- Python only. Use pandas, numpy, lightgbm, joblib.
- All ML operations: vectorised, no Python loops over rows.
- NO lookahead in features — 4H/1D/1W use PREVIOUS completed candle (floor-subtract).
- LightGBM + isotonic calibration is the standard model stack.
- Walk-forward validated thresholds are FROZEN — do not change without running full WF.
- Parallel ML pipeline: data collection, training, backtesting run concurrently.

## Agent Roles
| Agent | Scope | Must NOT |
|-------|-------|----------|
| DataAgent | Binance OHLCV fetch + CSV update | modify feature CSVs |
| FeatureAgent | Multi-TF feature engineering | fetch live prices |
| SignalAgent | Load model + predict_proba | retrain models |
| StrategyAgent | LLM ranking of signals | make execution decisions |
| BacktestAgent | Walk-forward validation | use test data for training |
| RiskAgent | Paper trading metrics | modify positions |
| OptimizerAgent | Threshold sweep on saved model | re-train from scratch |
| DiscoveryAgent | Mini-backtest param variations | deploy to paper trading directly |
| MonitorAgent | Trigger self-improve pipeline | run WF without RiskAgent approval |
| QueenAgent | Route + orchestrate | execute trades |

## File Paths (do not change)
- Models: `models/{COIN}_USDT/wf_decision_model_v2.pkl`
- Features: `data/{COIN}_USDT_multi_tf_features.csv`
- Paper state: `data/paper_trading_state.json`
- Agent memory: `data/agent_memory.db`

## WF-Validated Thresholds (frozen)
BTC=0.60, SOL=0.55, AVAX=0.60, LINK=0.55 (MARGINAL)
ETH=INSUFFICIENT_DATA, PEPE/BNB=NOT_VIABLE

## Coin-Specific TP/SL
BTC: TP=3% SL=1.5% 48h | ETH: TP=4.5% SL=1.5% 48h
SOL: TP=7.5% SL=2.5% 72h | PEPE: TP=15% SL=5% 48h
AVAX: TP=7.5% SL=2.5% 72h | BNB: TP=6% SL=2% 48h | LINK: TP=7.5% SL=2.5% 72h

## Parallel Execution (mandatory for swarm tasks)
- Signal scanning: 7 coins simultaneously via ThreadPoolExecutor(max_workers=7)
- Data collection: all timeframes in parallel
- Backtest folds: parallel where fold data doesn't overlap

## Token Efficiency Rules
- Agent system prompts: 50-100 words max
- Pass only relevant inputs, not full codebase context
- Compress outputs before storing in memory
- Use hardcoded plans for known tasks (scan, report, update-data) — skip LLM decomposition

## Prohibited
- `eval()` for JSON parsing — use `json.loads()` with try/except
- `signal.alarm` for timeouts — use `future.result(timeout=N)`
- Shared SQLite connection across threads — create new connection per call
- Full feature CSV in LLM context — summarise to key stats only

## Production (EC2)
- Server: ubuntu@13.51.159.80, key: ~/Desktop/TradeWise.pem, region: eu-north-1
- Crypto API: port 8000 (also serves React frontend + proxies /stocks/* to 8001)
- India Stocks API: port 8001 (internal only, proxied through 8000)
- EC2 is NOT a git repo — deploy via rsync/scp from Mac
- Full deploy reference: see EC2_DEPLOYMENT.md
- Python 3.9 venv — all `X | None` type hints require `from __future__ import annotations`
- REACT_APP_WS_URL must be empty in .env.production (fallback handles /ws/prices path)
- Viable models on EC2: BTC+LINK (crypto), NIFTY50+BANKNIFTY+NIFTYIT+TITAN+AXISBANK (stocks)
