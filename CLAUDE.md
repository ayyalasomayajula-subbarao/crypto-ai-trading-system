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
- Viable models on EC2: BTC+LINK (crypto), 25 India stocks symbols (v9 expanded universe)
- Restart stocks API: `pkill -f 'uvicorn api_stocks'` then `nohup uvicorn api_stocks:app --host 0.0.0.0 --port 8001 >> /tmp/stocks_api.log 2>&1 &`
- Deploy frontend: build locally → rsync dashboard/build/ to EC2 (no API restart needed)

## India Stocks — Position Sizing Formula (v9)
```
trade_capital = ₹10L × 15% × WF_multiplier × score_multiplier
qty = int(trade_capital / entry_price)   # raw shares — NO forced lot minimum
```
- score_multiplier: LONG score≥65 or SHORT score≤25 → 1.0x | BUY/SELL → 0.75x | LEAN → 0.5x
- WF_multiplier: 5+ valid folds → 0.7x | 3-4 → 0.5x | 2 → 0.3x | NOT_VIABLE → 0.0x (blocked)
- DO NOT use `max(1 lot)` — forces oversized positions (RELIANCE lot=250 shares = ₹3.5L vs intended ₹52K)
- Max 5 open positions, max daily loss 3% of capital

## India Stocks — Key API Fields (v9)
- `/stocks/verdict/{symbol}` returns: `wf_tier` (VIABLE/MARGINAL/NOT_VIABLE) + `wf_multiplier` (float)
- `paper_trader.py` exports: `get_wf_multiplier(symbol)`, `get_wf_tier(symbol)` (public helpers)
- Position object includes: `trade_value`, `entry_time`, `verdict_score`, `score_mult`, `wf_size_mult`

## India Stocks — Auto-Scanner (v9)
- Background loop in `api_stocks.py`: scans all 25 symbols every 15 min during market hours (09:15–15:30 IST)
- LONG entry: score≥55 + direction=LONG | SHORT entry: score≤40 + direction=SHORT
- Activity ring buffer (100 events) at `/stocks/paper-trading/activity`
- Live intraday prices: yfinance 1m batch fetch during market hours (60s cache), 1D CSV outside hours (300s cache)

## India Stocks — Instruments (v9, 25 symbols)
3 indices: NIFTY50, BANKNIFTY, NIFTYIT
22 stocks: RELIANCE, TCS, HDFCBANK, ICICIBANK, INFY, HCLTECH, BAJFINANCE, SBIN, AXISBANK,
           MARUTI, TITAN, LT, DRREDDY, BAJAJFINSV, ULTRACEMCO, ASIANPAINT, KOTAKBANK,
           CIPLA, WIPRO, BHARTIARTL, SUNPHARMA, TECHM
