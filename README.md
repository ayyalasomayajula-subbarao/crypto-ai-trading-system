# Crypto AI Trading System v7.0

A full-stack crypto trading intelligence platform with ML predictions, AI-powered analysis, paper trading, backtesting, and real-time market data.

## User Flow

### 1. Login / Sign Up
Create an account or log in with your email. Authentication is handled by Supabase. All pages are protected — you must be logged in to access the app.

### 2. Dashboard (Home)
After login, you land on the **Dashboard** — your command center.

- **Live Prices** — BTC, ETH, SOL, PEPE update in real-time via WebSocket
- **Coin Cards** — Each card shows current price, 24h change, and a quick ML verdict (BUY/WAIT/AVOID)
- **Navigation** — Click any coin card to dive into detailed analysis, or use the top buttons:
  - **Backtest** — Test strategies against historical data
  - **Signals** — View signal history
  - **Paper Trading** — Monitor the live paper trading bot

### 3. Coin Analysis Page
Click any coin to see the full analysis breakdown:

- **Price Chart** — Interactive TradingView chart with 9 timeframes (1H to ALL), candlestick or area mode
- **ML Predictions** — WIN/LOSS/SIDEWAYS probabilities from the trained model
- **Trade Setup** — Select your trade type (SCALP, SHORT_TERM, SWING, INVESTMENT) and experience level (BEGINNER, INTERMEDIATE, ADVANCED) to get personalized thresholds
- **Market Regime** — ADX trend strength, volatility classification
- **Volume Analysis** — OBV, MFI, buy/sell delta, volume spikes, volume profile
- **Market Sentiment** — Fear & Greed Index, funding rates, open interest
- **Derivatives Intelligence** — Long/short ratios, taker buy/sell, order book depth, liquidation map
- **Whale Activity** — Large transaction detection
- **Scenario Engine** — Warns about FOMO, losing streaks, overtrading, BTC crashes
- **Investment Requirements** — Shows which conditions are met/unmet for your trade type
- **Price Forecast** — Upside/sideways/downside probabilities with bull/bear targets
- **AI Analysis** — Click "Run Deep Analysis" for Groq/OpenAI-powered insights including TLDR, risks, entry strategy

### 4. Paper Trading
The automated paper trading bot runs 24/7 on your server:

- **Start/Stop** — Set your capital and start the bot
- **Live Equity Curve** — Watch your portfolio value change over time
- **Open Positions** — See current trades with entry price, P&L, and exit targets
- **Trade Log** — Full history of every trade with entry/exit prices and reasons
- **Metrics** — Win rate, profit factor, Sharpe ratio, max drawdown
- **Day Counter** — Progress toward the 45-day validation target
- The bot uses frozen walk-forward validated models — no parameter changes during the test

### 5. Backtest
Test trading strategies against historical data:

- Select a coin and time period
- See results: total return, win rate, profit factor, max drawdown, trade count
- Compare different strategies and parameter settings

### 6. News Feed
Visible on the Dashboard below the coin cards:

- **Crypto News** — From CoinTelegraph, CoinDesk, Decrypt, Bitcoin Magazine
- **Geopolitical News** — Fed decisions, regulations, macro events
- **Sentiment Tags** — Each article tagged BULLISH/BEARISH/NEUTRAL with impact score
- **Category Filters** — All, Crypto, Geopolitical, Bullish, Bearish, High Impact

## Features

### Core Analysis Engine
- **ML Model Predictions** — RandomForest trained on multi-timeframe data (1H/4H/1D/1W)
- **Walk-Forward Validated Models** — Rolling validation ensures models aren't overfit
- **Trade-Type-Specific Analysis** — SCALP, SHORT_TERM, SWING, INVESTMENT with different thresholds
- **Experience-Level Adjustments** — BEGINNER, INTERMEDIATE, ADVANCED modify required probabilities
- **Scenario Engine** — Detects FOMO, losing streaks, overtrading, BTC crashes, extreme volatility
- **Position Sizing** — Kelly Criterion + ATR-based stop loss/take profit

### Paper Trading Engine (NEW in v7)
- **Automated Hourly Processing** — Fetches candles from Binance, computes features, runs model
- **Frozen Parameters** — SOL threshold 0.35, PEPE threshold 0.40, TP 5%, SL 3%, TIME 48h
- **State Persistence** — Survives server restarts via JSON state file
- **Auto-Start** — Set `AUTO_START_PAPER_TRADING=true` to start on boot
- **0.5% Equity Risk** per trade with 0.22% round-trip costs

### Backtesting Engine (NEW in v7)
- **Realistic Execution** — Includes trading costs, slippage, position sizing
- **Walk-Forward Validation** — Test model performance on unseen data windows
- **Rolling Robustness** — Verify strategy stability across multiple time periods

### AI-Powered Insights (FREE)
- **Groq** (Llama 3.3 70B) as primary
- **OpenAI** (GPT-3.5-turbo) as fallback
- Cross-references technicals, volume, sentiment, derivatives, news, and whale data
- Returns: TLDR, positives/risks with severity, market pulse, entry strategy, conviction reason

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
- **Funding Rate** (Binance Futures)
- **Open Interest** (Binance Futures)

### Derivatives Intelligence
- **Long/Short Ratio** — Top Traders vs Global accounts with smart money signal
- **Taker Buy/Sell Volume** — aggressive order flow pressure
- **Order Book Depth** — bid/ask walls, support/resistance levels
- **Liquidation Estimates** — calculated from OI + leverage levels (5x-100x)

### Whale Tracking
- **BTC/ETH** — Large transactions via Blockchair API
- **SOL/PEPE** — Outlier trade detection from Binance trades

### News & Geopolitics
- **Card grid** layout with sentiment-based gradient headers
- **6 category tabs** — All, Crypto, Geopolitical, Bullish, Bearish, High Impact
- RSS feeds from CoinTelegraph, CoinDesk, Decrypt, Bitcoin Magazine

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, FastAPI, uvicorn |
| ML Models | scikit-learn (RandomForest), walk-forward validated |
| Frontend | React 19, TypeScript |
| Charts | TradingView lightweight-charts |
| Auth | Supabase |
| AI | Groq (free), OpenAI (fallback) |
| Data | Binance API, Blockchair, alternative.me, RSS feeds |

## Coins Supported
BTC/USDT, ETH/USDT, SOL/USDT, PEPE/USDT

## Setup

### Local Development

#### Backend
```bash
cd crypto-ai-system
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create .env with:
# GROQ_API_KEY=your_key_here (free at console.groq.com)
# OPENAI_API_KEY=your_key_here (optional fallback)

python api_final.py
# Runs on http://localhost:8000
```

#### Frontend
```bash
cd dashboard
npm install

# Create .env with Supabase keys:
# REACT_APP_SUPABASE_URL=your_url
# REACT_APP_SUPABASE_ANON_KEY=your_key

npm start
# Runs on http://localhost:3000
```

### Production (EC2)

The app runs as a single process — FastAPI serves both the API and the React frontend on port 8000.

```bash
# Build frontend (on your local machine)
cd dashboard
REACT_APP_API_URL=http://YOUR_EC2_IP:8000 REACT_APP_WS_URL=ws://YOUR_EC2_IP:8000/ws/prices npm run build

# Upload to EC2
scp -r api_final.py paper_trader.py backtesting_engine.py requirements.txt .env models/ dashboard/build/ data/ ubuntu@YOUR_EC2_IP:~/crypto-ai-system/

# On EC2: install and run
cd ~/crypto-ai-system
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Start with paper trading auto-enabled
tmux new -s trading
AUTO_START_PAPER_TRADING=true PAPER_TRADING_CAPITAL=1000 python3 api_final.py
# Ctrl+B, D to detach
```

Open port 8000 in your AWS Security Group, then visit `http://YOUR_EC2_IP:8000/`

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
| `POST /paper-trading/start` | Start paper trading (optional `capital` param) |
| `POST /paper-trading/stop` | Stop paper trading |
| `GET /paper-trading/status` | Current status, equity, open positions |
| `GET /paper-trading/trades` | Full trade log |
| `GET /paper-trading/metrics` | Win rate, PF, Sharpe, DD |
| `POST /paper-trading/reset` | Clear state and start fresh |
| `WS /ws/prices` | Real-time price WebSocket |

## Project Structure
```
crypto-ai-system/
├── api_final.py              # FastAPI backend (port 8000)
├── paper_trader.py           # Paper trading engine
├── backtesting_engine.py     # Backtesting with realistic execution
├── walk_forward_validation.py # Walk-forward model validation
├── rolling_walk_forward.py   # Rolling robustness testing
├── collect_multi_timeframe.py # Data collection script
├── requirements.txt          # Python dependencies
├── .env                      # API keys (GROQ, OpenAI)
├── data/                     # Historical OHLCV + features
│   ├── {COIN}_USDT_1h.csv
│   ├── {COIN}_USDT_multi_tf_features.csv
│   └── paper_trading_state.json
├── models/                   # Trained ML models
│   └── {COIN}_USDT/
│       ├── wf_decision_model.pkl    # Walk-forward validated
│       ├── decision_model.pkl       # Standard trained
│       └── decision_features.txt
└── dashboard/                # React frontend
    ├── .env                  # Supabase keys
    ├── .env.production       # Production build config
    └── src/components/
        ├── Dashboard.tsx     # Main portfolio dashboard
        ├── Coinpage.tsx      # Per-coin analysis page
        ├── PaperTrading.tsx  # Paper trading monitor
        ├── Backtest.tsx      # Backtesting UI
        ├── SignalHistory.tsx  # Signal history
        ├── PriceChart.tsx    # Interactive TradingView charts
        └── NewsFeed.tsx      # News grid with sentiment
```

## Version History
| Version | Highlights |
|---------|-----------|
| **v7.0** | Paper trading engine, backtesting, walk-forward validation, EC2 deployment, dynamic API URLs |
| **v6.0** | Groq/OpenAI AI analysis, derivatives intelligence, whale tracking, interactive charts, tooltips |
| **v5.0** | Supabase auth, portfolio tracking, trade-type-specific analysis |
| **v4.1** | Scenario Engine, WebSocket prices, WIN/LOSS probability display |
| **v2.0** | Initial UI + Backend release |

## Note
Training data and models are excluded from the repository. Run `collect_multi_timeframe.py` to download historical data, then use `walk_forward_validation.py` to train and validate models.
