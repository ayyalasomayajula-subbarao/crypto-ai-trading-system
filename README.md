# Crypto AI Trading System v8.1

A full-stack crypto trading intelligence platform with LightGBM ML predictions, walk-forward validated models, multi-agent orchestration, AI-powered analysis, multi-LLM consensus chat, bidirectional paper trading, backtesting, and real-time market data.

## User Flow

### 1. Login / Sign Up
Create an account or log in with your email. Authentication is handled by Supabase. All pages are protected — you must be logged in to access the app.

### 2. Overview (Home)
After login, you land on the **Overview** page — your market command center.

- **Market Banner** — Shows current market condition (opportunities, high risk, etc.)
- **Live Prices** — BTC, ETH, SOL, PEPE update in real-time via WebSocket
- **Coin Cards** — Each card shows current price, 1H/24H/7D changes, ML verdict (BUY/SHORT/WAIT/AVOID), expectancy, and active scenarios
- **News Feed** — Crypto and geopolitical news with sentiment tags below the signals grid
- **Sidebar Navigation** — Persistent sidebar with links to all sections

### 3. Portfolio
Dedicated page for tracking your holdings:

- **Portfolio Summary** — Total value, P&L over selectable periods (24h/7d/30d/all)
- **Capital Management** — Set and update your trading capital
- **Holdings List** — Live-priced holdings with sell functionality
- **Add Holdings** — Add new positions with coin, quantity, and entry price

### 4. Coins
Browse all supported coins with live prices and quick verdicts:

- **Coin Cards** — BTC, ETH, SOL, PEPE, AVAX, BNB, LINK, ARB, OP, INJ with live WebSocket prices
- **24H Change** — Color-coded percentage changes
- **Quick Verdict** — ML model verdict badge (BUY/SHORT/WAIT/AVOID) and win probability bar
- Click any coin to open the detailed analysis page

### 5. Coin Analysis Page
Click any coin for the full analysis breakdown:

- **Price Chart** — Interactive TradingView chart with 9 timeframes (1H to ALL), candlestick or area mode
- **ML Predictions** — UP/DOWN/SIDEWAYS probabilities from the LightGBM model with isotonic calibration
- **Trade Setup** — Select your trade type (SCALP, SHORT_TERM, SWING, INVESTMENT) and experience level (BEGINNER, INTERMEDIATE, ADVANCED) to get personalized thresholds
- **Market Regime** — ADX trend strength, volatility classification
- **Volume Analysis** — OBV, MFI, buy/sell delta, volume spikes, volume profile
- **Market Sentiment** — Fear & Greed Index, funding rates, open interest
- **Derivatives Intelligence** — Long/short ratios, taker buy/sell, order book depth, liquidation map
- **Whale Activity** — Large transaction detection
- **Scenario Engine** — Warns about FOMO, losing streaks, overtrading, BTC crashes
- **Investment Requirements** — Shows which conditions are met/unmet for your trade type
- **Price Forecast** — Upside/sideways/downside probabilities with bull/bear targets
- **AI Analysis** — Click "Run Deep Analysis" for Groq-powered insights including TLDR, risks, entry strategy

### 6. AI Trading Chat
Context-aware trading advisor accessible from any coin page:

- **Time Horizon Chips** — Select SCALP / SHORT / SWING / INVEST before querying; AI prompts you if not set
- **Capital Input** — Enter your position size for personalized sizing advice
- **Multi-LLM Consensus** — Groq (Llama 3.3 70B) + Qwen3-32B run in parallel; a dedicated Groq judge resolves disagreements
- **Rich Response Cards** — Verdict + confidence, market bias, BTC environment tag, multi-timeframe alignment grid, key support/resistance/invalidation levels, stop loss, targets with probabilities, scenario bars, position sizing, news sentiment
- **Auto Data Fetch** — Chat endpoint pulls live prices, multi-timeframe CSVs, and news independently — no manual context needed
- **Consensus Badge** — Shows `CONSENSUS HIGH` when both models agree, `MEDIUM` when judge arbitrated

### 7. Paper Trading
The automated paper trading bot runs 24/7 on your server:

- **Start/Stop** — Set your capital and start the bot
- **Live Equity Curve** — Watch your portfolio value change over time
- **Open Positions** — See current trades with entry price, P&L, and exit targets (LONG and SHORT)
- **Trade Log** — Full history of every trade with entry/exit prices and reasons
- **Metrics** — Win rate, profit factor, Sharpe ratio, max drawdown
- **Day Counter** — Progress toward the 45-day validation target
- **Bidirectional** — Goes LONG on UP signals, SHORT on DOWN signals, filtered by ADX + weekly regime gate
- Currently active coins: **BTC** (threshold=0.50) and **LINK** (threshold=0.55), both walk-forward validated MARGINAL

### 8. Backtest
Test trading strategies against historical data:

- Select a coin and time period
- See results: total return, win rate, profit factor, max drawdown, trade count
- Compare different strategies and parameter settings

### 9. Settings
Manage your profile and account preferences.

## Features

### Core ML Engine
- **LightGBM 3-Class Model** — UP / DOWN / SIDEWAYS predictions with isotonic probability calibration
- **90 ML Features** — 1H/4H/1D/1W: SMA-21/50 + distances + slopes, RSI, MACD-diff, Bollinger width/position, ATR%, ADX, momentum, dist-from-10p-high, realized vol (1H), funding rate features (rate/3d-avg/7d-avg/momentum/trend), taker imbalance (4H/24H MA)
- **No Lookahead** — 4H/1D/1W use the previous completed candle (floor-subtract)
- **Walk-Forward Validated Models** — Expanding multi-fold WF ensures models are not overfit
- **Meta-Labeling** — Secondary LightGBM trained on each fold's last 25%; only trades when meta P(WIN) ≥ 0.52
- **Bidirectional Trading** — LONG on UP signal, SHORT on DOWN signal
- **ADX Gate** — `1h_adx ≥ 20` required for entry (trending markets only)
- **Regime Gate** — Weekly SMA-50 distance > 0 for LONG, < 0 for SHORT

### Walk-Forward Results (Current Best)
| Coin | Status | Sharpe | WR | Threshold | Notes |
|------|--------|--------|----|-----------|-------|
| BTC | **MARGINAL** ✅ | +0.382 | 58.4% | 0.50 | 3 folds, 43 trades, shorts-only |
| LINK | **MARGINAL** ✅ | +0.456 | 49.5% | 0.55 | 3 folds, 21 trades, longs-only |
| AVAX | NOT_VIABLE | +0.173 | — | 0.60 | borderline |
| OP | NOT_VIABLE | +0.302 | — | — | borderline, short history |
| ETH | INSUFFICIENT_DATA | — | — | — | meta over-filters; resolves with more 2025 data |
| SOL | NOT_VIABLE | — | — | — | |
| PEPE | NOT_VIABLE | — | — | — | meme coin, fast regime changes |
| ARB | NOT_VIABLE | — | — | — | <2yr history |
| BNB | NOT_VIABLE | — | — | — | |
| INJ | NOT_VIABLE | — | — | — | |

### Multi-Agent System (NEW in v8.0)
Hierarchical orchestration engine for automated ML pipeline management:

- **Queen Agent** — Routes tasks, orchestrates parallel execution via TaskQueue
- **Planner Agent** — Groq LLM task decomposition (hardcoded plans for known tasks — no LLM call for scan/report/update-data)
- **Memory Agent** — SQLite at `data/agent_memory.db` with model version tracking
- **Signal Agent** — Scans all coins in parallel (ThreadPoolExecutor × 7)
- **Data Agent** — Incremental Binance OHLCV fetch + CSV update
- **Feature Agent** — Multi-timeframe feature engineering
- **Strategy Agent** — LLM-ranked trade recommendations
- **Backtest Agent** — Walk-forward validation per coin
- **Risk Agent** — Paper trading metrics + degradation detection
- **Optimizer Agent** — Threshold sweep (±0.05/0.10/0.15) on saved model
- **Discovery Agent** — LLM proposes param variations, mini-backtests top candidates
- **Monitor Agent** — Daily drift check → triggers Optimizer → BacktestAgent if degraded

CLI usage:
```bash
python agent_runner.py scan              # ranked signals for all coins
python agent_runner.py report            # paper trading metrics
python agent_runner.py analyze BTC       # full pipeline: data→signal→strategy→risk
python agent_runner.py backtest LINK     # walk-forward validation
python agent_runner.py optimize BTC      # threshold sweep
python agent_runner.py self-improve      # monitor → optimize → retrain if degraded
python agent_runner.py discover BTC      # LLM proposes + mini-backtests param variations
python agent_runner.py update-data       # refresh all coin CSVs
```

Token efficiency: ~65-80% reduction vs single-prompt approach via role-specific 50-100-word system prompts, SQLite context retrieval, parallel 7-coin scan, and summary compression before storage.

### MUI Sidebar Layout
- **Persistent Sidebar** — Material UI drawer with Overview, Portfolio, Coins, Backtest, Signals, Paper Trading navigation
- **Mobile Responsive** — Collapsible hamburger menu on mobile, permanent drawer on desktop
- **Profile Section** — User avatar, name, and email in sidebar footer
- **Dark Theme** — MUI ThemeProvider with custom dark palette

### Paper Trading Engine
- **Automated Hourly Processing** — Fetches candles from Binance, computes 90 live features, runs LightGBM model
- **Bidirectional** — LONG on P(UP) ≥ threshold + ADX ≥ 20 + regime long; SHORT on P(DOWN) ≥ threshold + regime short
- **Frozen Parameters** — BTC thresh=0.50 TP=3% SL=1.5% 48h | LINK thresh=0.55 TP=7.5% SL=2.5% 72h
- **State Persistence** — Survives server restarts via JSON state file
- **0.5% Equity Risk** per trade with 0.22% round-trip costs
- **Feature Parity** — Same 90-feature format as training (no lookahead, SMA-21/50 only)

### Backtesting Engine
- **Realistic Execution** — Includes trading costs, slippage, position sizing
- **Walk-Forward Validation** — Test model performance on unseen data windows
- **Rolling Robustness** — Verify strategy stability across multiple time periods

### AI-Powered Insights
- **Groq** (Llama 3.3 70B) as primary analyst
- **Qwen3-32B** on Groq as second opinion (DeepSeek API when credits available)
- **Groq Judge** (dedicated key) arbitrates on disagreement
- Cross-references technicals, volume, sentiment, derivatives, news, and whale data
- Returns: verdict + confidence, timeframe alignment, key levels, market bias, BTC environment, stop loss, targets, position sizing, scenarios

### Interactive Price Charts
- **TradingView lightweight-charts** — professional candlestick and area charts
- **9 timeframes** — 1H, 6H, 1D, 3D, 7D, 1M, 3M, 1Y, ALL
- Area and Candlestick chart type toggle

### Volume Analysis
- **OBV** (On-Balance Volume) with divergence detection
- **MFI** (Money Flow Index) with overbought/oversold zones
- **Buy/Sell Delta** with pressure strength
- **Volume Spikes** detection (vs 20-period MA)

### Market Sentiment
- **Fear & Greed Index** (alternative.me)
- **Funding Rate** (Binance Futures — 8h settlement)
- **Open Interest** (Binance Futures)
- **Taker Buy/Sell Imbalance** (4H and 24H moving average)

### Derivatives Intelligence
- **Long/Short Ratio** — Top Traders vs Global accounts with smart money signal
- **Taker Buy/Sell Volume** — aggressive order flow pressure
- **Order Book Depth** — bid/ask walls, support/resistance levels
- **Liquidation Estimates** — calculated from OI + leverage levels (5x-100x)

### Whale Tracking
- **BTC/ETH** — Large transactions via Blockchair API
- **SOL/PEPE** — Outlier trade detection from Binance trades

### News & Geopolitics
- **Responsive card grid** layout with sentiment-based gradient headers
- **6 category tabs** — All, Crypto, Geopolitical, Bullish, Bearish, High Impact
- RSS feeds from CoinTelegraph, CoinDesk, Decrypt, Bitcoin Magazine

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.9, FastAPI, uvicorn |
| ML Models | LightGBM + isotonic calibration (3-class UP/DOWN/SIDEWAYS), walk-forward validated |
| Feature Engineering | 90 features, 4 timeframes (1H/4H/1D/1W), no lookahead |
| Agent System | Python `agents/` package, Groq LLM, SQLite memory |
| Frontend | React 19, TypeScript, MUI 6 |
| Charts | TradingView lightweight-charts |
| Auth | Supabase |
| AI | Groq (primary + judge), Qwen3-32B/Groq (second opinion), DeepSeek (optional) |
| Data | Binance API (OHLCV + Futures), Blockchair, alternative.me, RSS feeds |
| Deployment | AWS EC2 (eu-north-1) |

## Coins Supported
BTC/USDT, ETH/USDT, SOL/USDT, PEPE/USDT, AVAX/USDT, BNB/USDT, LINK/USDT, ARB/USDT, OP/USDT, INJ/USDT

## India Stocks Supported
NIFTY50 (VIABLE), BANKNIFTY (MARGINAL), NIFTYIT (MARGINAL), TITAN (MARGINAL), AXISBANK (MARGINAL)
Data via yfinance (1D+1W, 10yr history). Cron updates daily at 4:30pm IST Mon-Fri.

### India Stocks API (port 8001, proxied through 8000)
| Endpoint | Description |
|----------|------------|
| `GET /stocks/scan` | Verdicts for all active instruments |
| `GET /stocks/verdict/{symbol}` | Single instrument verdict + signals |
| `GET /stocks/market/overview` | PCR, max pain, India VIX, FII/DII summary |
| `GET /stocks/fii-dii` | 30-day FII/DII flow history |
| `GET /stocks/india-vix` | India VIX history |
| `GET /stocks/option-chain/{symbol}` | Live option chain (NIFTY50/BANKNIFTY) |
| `GET /stocks/klines/{symbol}` | OHLCV chart data |
| `GET /stocks/backtest` | Walk-forward results for all instruments |
| `GET /stocks/paper-trading/status` | Paper trading status + metrics |
| `GET /stocks/paper-trading/trades` | Full trade log |
| `POST /stocks/paper-trading/start` | Start paper trading |
| `WS /ws/stocks/prices` | Real-time NSE price WebSocket |

## Setup

### Backend
```bash
cd crypto-ai-system
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create .env with:
# GROQ_API_KEY=your_key_here        (free at console.groq.com — primary + second opinion)
# GROQ_JUDGE_API_KEY=your_key_here  (second Groq key — dedicated judge pool)
# DEEPSEEK_API_KEY=your_key_here    (optional — falls back to Qwen3-32B on Groq)
# OPENAI_API_KEY=your_key_here      (optional — for /ai-analysis endpoint only)

python api_final.py
# Runs on http://localhost:8000
```

### Frontend
```bash
cd dashboard
npm install

# Create .env with Supabase keys:
# REACT_APP_SUPABASE_URL=your_url
# REACT_APP_SUPABASE_ANON_KEY=your_key

npm start
# Runs on http://localhost:3000
```

### Production Build
```bash
cd dashboard
npm run build
# Build output served by FastAPI at http://localhost:8000
```

### Data & Model Training
```bash
# 1. Collect OHLCV + features for all 10 coins
python collect_multi_timeframe.py

# 2. Collect funding rates (Binance Futures)
python collect_funding_rates.py

# 3. Collect taker volume
python collect_taker_volume.py

# 4. Train and walk-forward validate models
python walk_forward_validation.py

# 5. (Optional) Incremental data update via agent system
python agent_runner.py update-data
```

Use Python 3.9 to ensure model pickle compatibility across environments.

## API Endpoints

| Endpoint | Description |
|----------|------------|
| `GET /analyze/{coin}` | Full ML analysis with all parameters |
| `GET /ai-analysis/{coin}` | AI-powered deep analysis (Groq/OpenAI) |
| `GET /klines/{coin}` | OHLCV chart data (interval + limit params) |
| `GET /scan` | Quick verdict on all coins |
| `GET /prices` | Current live prices |
| `GET /market-sentiment/{coin}` | Fear & Greed, Funding, OI |
| `GET /derivatives/{coin}` | L/S Ratio, Taker Volume, Order Book, Liquidations |
| `GET /whales/{coin}` | Whale activity and large transactions |
| `GET /news/crypto` | Crypto news with sentiment |
| `GET /news/geopolitical` | Geopolitical news impact |
| `GET /fear-greed` | Fear & Greed Index |
| `GET /ai-status` | AI provider availability |
| `POST /chat/{coin}` | AI Trading Chat — multi-LLM consensus (body: `message`, `time_horizon`, `capital`, `chat_history`) |
| `POST /paper-trading/start` | Start paper trading (optional `capital` param) |
| `POST /paper-trading/stop` | Stop paper trading |
| `GET /paper-trading/status` | Current status, equity, open positions |
| `GET /paper-trading/trades` | Full trade log |
| `GET /paper-trading/metrics` | Win rate, PF, Sharpe, DD |
| `POST /paper-trading/reset` | Clear state and start fresh |
| `GET /agents/scan` | Agent system: ranked signals for all coins |
| `GET /agents/report` | Agent system: paper trading health report |
| `GET /agents/models` | Agent system: active model versions |
| `WS /ws/prices` | Real-time price WebSocket |

## Project Structure
```
crypto-ai-system/
├── api_final.py                  # FastAPI backend (port 8000)
├── paper_trader.py               # Paper trading engine (bidirectional, 90-feature)
├── walk_forward_validation.py    # Expanding WF + meta-labeling
├── collect_multi_timeframe.py    # 90-feature engineering (no lookahead)
├── collect_funding_rates.py      # Binance Futures funding rate history
├── collect_taker_volume.py       # Binance taker buy/sell volume
├── rolling_walk_forward.py       # Rolling robustness testing
├── train_decision_model.py       # Standard (non-WF) model training
├── agent_runner.py               # Multi-agent CLI entry point
├── update_data.py                # Incremental data updater
├── CLAUDE.md                     # Agent development rules
├── requirements.txt              # Python dependencies (pinned)
├── .env                          # API keys (GROQ, OpenAI)
├── agents/                       # Multi-agent system
│   ├── __init__.py
│   ├── base.py                   # BaseAgent: think() + log()
│   ├── memory.py                 # AgentMemory: SQLite + model version tracking
│   ├── queen.py                  # QueenAgent: routing + orchestration
│   ├── planner.py                # PlannerAgent: Groq LLM task decomposition
│   ├── queue_worker.py           # TaskQueue: ThreadPoolExecutor + per-task timeout
│   ├── monitor.py                # MonitorAgent: drift detection + self-improve loop
│   ├── execution/
│   │   ├── data_agent.py         # DataAgent: Binance OHLCV fetch
│   │   ├── feature_agent.py      # FeatureAgent: multi-TF feature engineering
│   │   ├── signal_agent.py       # SignalAgent: parallel 7-coin signal scan
│   │   └── strategy_agent.py     # StrategyAgent: LLM trade ranking
│   └── evaluation/
│       ├── backtest_agent.py     # BacktestAgent: walk-forward validation
│       ├── risk_agent.py         # RiskAgent: paper trading metrics
│       ├── optimizer_agent.py    # OptimizerAgent: threshold sweep
│       └── discovery_agent.py    # DiscoveryAgent: LLM param exploration
├── data/                         # Historical OHLCV + features
│   ├── {COIN}_USDT_1h.csv
│   ├── {COIN}_USDT_4h.csv
│   ├── {COIN}_USDT_1d.csv
│   ├── {COIN}_USDT_1w.csv
│   ├── {COIN}_USDT_multi_tf_features.csv
│   ├── {COIN}_USDT_funding.csv
│   ├── {COIN}_USDT_taker_volume.csv
│   ├── paper_trading_state.json
│   └── agent_memory.db           # SQLite: agent memory + model versions
├── models/                       # Trained ML models (per coin)
│   └── {COIN}_USDT/
│       ├── wf_decision_model_v2.pkl    # Walk-forward + meta-labeled model
│       ├── decision_model_v2.pkl       # Standard trained model
│       └── decision_features_v2.txt   # 90 feature names (exact training order)
└── dashboard/                    # React frontend
    ├── .env                      # Supabase keys
    └── src/
        ├── theme.ts              # MUI dark theme configuration
        ├── App.tsx               # Routes with Layout wrapper
        ├── components/
        │   ├── layout/
        │   │   ├── Layout.tsx    # Sidebar + TopBar + Outlet
        │   │   ├── Sidebar.tsx   # MUI persistent drawer navigation
        │   │   └── TopBar.tsx    # Mobile hamburger + Live indicator
        │   ├── Dashboard.tsx     # Overview: signals grid + news
        │   ├── Coinpage.tsx      # Per-coin analysis page
        │   ├── TradingChat.tsx   # AI Trading Chat (multi-LLM consensus)
        │   ├── TradingChat.css   # Chat UI styles
        │   ├── PaperTrading.tsx  # Paper trading monitor
        │   ├── Backtest.tsx      # Backtesting UI
        │   ├── SignalHistory.tsx  # Signal history
        │   ├── PriceChart.tsx    # Interactive TradingView charts
        │   └── NewsFeed.tsx      # News grid with sentiment
        └── pages/
            ├── PortfolioPage.tsx # Portfolio management
            ├── CoinsPage.tsx     # Coin browser with live prices
            └── SettingsPage.tsx  # Profile & account settings
```

## Version History
| Version | Highlights |
|---------|-----------|
| **v8.1** | India Stocks system (NSE/BSE) with 5 viable instruments (NIFTY50 VIABLE, BANKNIFTY/NIFTYIT/TITAN/AXISBANK MARGINAL), yfinance 1D+1W features, LightGBM + meta-labeling, paper trading, option chain/FII-DII/VIX data, stocks proxy through port 8000, production EC2 deployment with auto-restart cron |
| **v8.0** | LightGBM 3-class (UP/DOWN/SIDEWAYS) + meta-labeling, 90 ML features (no lookahead), bidirectional paper trading (BTC+LINK MARGINAL), multi-agent orchestration system, 10 coins (added ARB/OP/INJ/AVAX/BNB/LINK), walk-forward expanded folds, funding rate + taker imbalance features |
| **v7.2** | AI Trading Chat with multi-LLM consensus (Groq + Qwen3-32B + dedicated Groq judge), time horizon/capital input, rich response cards, DeepSeek fallback chain |
| **v7.1** | MUI sidebar layout, dedicated Portfolio/Coins/Settings pages, pinned Python dependencies, prod/local model sync |
| **v7.0** | Paper trading engine, backtesting, walk-forward validation, EC2 deployment, dynamic API URLs |
| **v6.0** | Groq/OpenAI AI analysis, derivatives intelligence, whale tracking, interactive charts, tooltips |
| **v5.0** | Supabase auth, portfolio tracking, trade-type-specific analysis |
| **v4.1** | Scenario Engine, WebSocket prices, WIN/LOSS probability display |
| **v2.0** | Initial UI + Backend release |

## Note
Training data and models are excluded from the repository. Run `collect_multi_timeframe.py`, `collect_funding_rates.py`, and `collect_taker_volume.py` to download historical data, then use `walk_forward_validation.py` to train and validate models. Use Python 3.9 to ensure model pickle compatibility across environments.
