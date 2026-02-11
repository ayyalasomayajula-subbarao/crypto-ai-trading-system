# Crypto AI Trading System v6.0

A full-stack crypto trading intelligence platform with ML predictions, AI-powered analysis, and real-time market data.

## Features

### Core Analysis Engine
- **ML Model Predictions** — RandomForest + XGBoost trained on multi-timeframe data (1H/4H/1D/1W)
- **Trade-Type-Specific Analysis** — SCALP, SHORT_TERM, SWING, INVESTMENT with different thresholds
- **Experience-Level Adjustments** — BEGINNER, INTERMEDIATE, ADVANCED modify required probabilities
- **Scenario Engine** — Detects FOMO, losing streaks, overtrading, BTC crashes, extreme volatility
- **Position Sizing** — Kelly Criterion + ATR-based stop loss/take profit

### AI-Powered Insights (FREE)
- **Groq** (Llama 3.3 70B) as primary
- **OpenAI** (GPT-3.5-turbo) as fallback
- Cross-references technicals, volume, sentiment, derivatives, news, and whale data
- Returns: TLDR, positives/risks with severity, market pulse, entry strategy, conviction reason

### Interactive Price Charts
- **TradingView lightweight-charts** — professional candlestick and area charts
- **9 timeframes** — 1H, 6H, 1D, 3D, 7D, 1M, 3M, 1Y, ALL
- Area and Candlestick chart type toggle
- Always visible on coin pages, loads instantly from Binance API

### Volume Analysis
- **OBV** (On-Balance Volume) with divergence detection
- **MFI** (Money Flow Index) with overbought/oversold zones
- **Buy/Sell Delta** with pressure strength
- **Volume Spikes** detection (vs 20-period MA)
- **Volume Profile** — POC, Value Area, HVN/LVN from live Binance data

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
- Accumulation/Distribution signal

### News & Geopolitics (Arkham-Style)
- **Full-width card grid** layout below coin cards
- **6 category tabs** — All, Crypto, Geopolitical, Bullish, Bearish, High Impact
- **Sentiment-based gradient headers** — green (bullish), red (bearish), purple (neutral)
- **Source watermarks** and category/sentiment badges on each card
- Responsive grid: 3 cols → 2 cols → 1 col

### UX
- **Hover tooltips** on every field explaining what each metric means
- **Real-time prices** via WebSocket
- **Supabase auth** — login/signup with portfolio tracking
- **Dark theme** throughout
- **Responsive** design for mobile

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, FastAPI, uvicorn |
| ML Models | scikit-learn (RandomForest), XGBoost |
| Frontend | React 19, TypeScript |
| Charts | TradingView lightweight-charts |
| Auth | Supabase |
| AI | Groq (free), OpenAI (fallback) |
| Data | Binance API, Blockchair, alternative.me, RSS feeds |

## Coins Supported
BTC/USDT, ETH/USDT, SOL/USDT, PEPE/USDT

## Setup

### Backend
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

### Data Updates (Automated)
```bash
# Manual update
python update_data.py

# Update specific coin
python update_data.py --coin BTC

# Auto-update every 6 hours (cron)
crontab -e
0 */6 * * * cd /path/to/crypto-ai-system && venv/bin/python update_data.py >> data/update.log 2>&1
```

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
| `WS /ws/prices` | Real-time price WebSocket |

## Project Structure
```
crypto-ai-system/
├── api_final.py              # FastAPI backend (port 8000)
├── update_data.py            # Automated data updater
├── requirements.txt          # Python dependencies
├── .env                      # API keys (GROQ, OpenAI)
├── data/                     # Historical OHLCV + features
│   └── {COIN}_USDT_*.csv
├── models/                   # Trained ML models
│   └── {COIN}_USDT/
│       ├── random_forest_*.pkl
│       └── xgboost_*.pkl
└── dashboard/                # React frontend (port 3000)
    ├── .env                  # Supabase keys
    └── src/components/
        ├── Dashboard.tsx     # Main portfolio dashboard
        ├── Coinpage.tsx      # Per-coin analysis page
        ├── PriceChart.tsx    # Interactive TradingView charts
        └── NewsFeed.tsx      # Arkham-style news grid
```

## Version History
| Version | Highlights |
|---------|-----------|
| **v6.0** | Groq/OpenAI AI analysis, derivatives intelligence, whale tracking, interactive charts, tooltips, Arkham-style news grid |
| **v5.0** | Supabase auth, portfolio tracking, trade-type-specific analysis |
| **v4.1** | Scenario Engine, WebSocket prices, WIN/LOSS probability display |
| **v2.0** | Initial UI + Backend release |

## Note
Training data and models are excluded from the repository. Run `collect_multi_timeframe.py` to download historical data, then train models using the training scripts.
