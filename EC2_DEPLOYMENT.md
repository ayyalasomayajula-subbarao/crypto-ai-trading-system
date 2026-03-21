# EC2 Deployment Reference

## Server Details
- **IP**: 13.51.159.80 (eu-north-1, Stockholm)
- **User**: ubuntu
- **Key**: ~/Desktop/TradeWise.pem
- **OS**: Ubuntu 24.04 LTS
- **Disk**: 20GB EBS gp3 (expanded from 8GB on 2026-03-21)

## SSH
```bash
ssh -i ~/Desktop/TradeWise.pem ubuntu@13.51.159.80
```

## Directory Structure (EC2)
```
~/crypto-ai-system/
  api_final.py          # Crypto API + proxy to stocks API
  dashboard/build/      # React static build (served by FastAPI)
  models/{COIN}_USDT/   # Only MARGINAL coins: BTC, LINK (pkl files)
  data/                 # Feature CSVs + paper trading state
  venv/                 # Python 3.9 venv (600MB)
  india-stocks/
    api_stocks.py
    models/             # Only MARGINAL: NIFTY50, BANKNIFTY, NIFTYIT, TITAN, AXISBANK
    data/               # Feature CSVs (copied from Mac, cron updates daily)
```

## Services
| Service | Port | Process | Log |
|---------|------|---------|-----|
| Crypto API + Frontend | 8000 | uvicorn api_final:app | /tmp/api.log |
| India Stocks API | 8001 | uvicorn api_stocks:app | /tmp/stocks_api.log |

**Port 8001 is internal only** — api_final.py proxies `/stocks/*` and `/ws/stocks/prices` to localhost:8001.
Only port 8000 needs to be open in the AWS security group.

## Start/Restart Commands (EC2)

### Crypto API (port 8000)
```bash
pkill -f "api_final" 2>/dev/null; sleep 1
cd ~/crypto-ai-system
nohup env AUTO_START_PAPER_TRADING=true PAPER_TRADING_CAPITAL=10000 \
  ~/crypto-ai-system/venv/bin/uvicorn api_final:app \
  --host 0.0.0.0 --port 8000 --workers 1 >> /tmp/api.log 2>&1 &
sleep 4 && curl -s http://localhost:8000/health
```

### India Stocks API (port 8001)
```bash
pkill -f "api_stocks" 2>/dev/null; sleep 1
cd ~/crypto-ai-system/india-stocks
nohup env ACTIVE_SYMBOLS="NIFTY50,BANKNIFTY,NIFTYIT,TITAN,AXISBANK" \
  ~/crypto-ai-system/venv/bin/uvicorn api_stocks:app \
  --host 0.0.0.0 --port 8001 --workers 1 >> /tmp/stocks_api.log 2>&1 &
sleep 4 && curl -s http://localhost:8001/health
```

### Check both running
```bash
ss -tlnp | grep -E "8000|8001"
curl -s http://localhost:8000/health
curl -s http://localhost:8001/health
```

## Deploy from Mac (full deploy)

### 1. Build dashboard
```bash
cd ~/Desktop/crypto-ai-system/dashboard
npm run build
```

### 2. Push code to EC2 (excludes venv, data CSVs, node_modules)
```bash
rsync -avz --progress \
  -e "ssh -i ~/Desktop/TradeWise.pem" \
  --exclude='venv/' --exclude='venv_old/' \
  --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='data/*.csv' --exclude='data/*.json' \
  --exclude='node_modules/' \
  --exclude='india-stocks/data/sector_cache/' \
  ~/Desktop/crypto-ai-system/ \
  ubuntu@13.51.159.80:~/crypto-ai-system/
```

### 3. Push dashboard build (clean push)
```bash
ssh -i ~/Desktop/TradeWise.pem ubuntu@13.51.159.80 \
  "rm -rf ~/crypto-ai-system/dashboard/build"

scp -i ~/Desktop/TradeWise.pem -r \
  ~/Desktop/crypto-ai-system/dashboard/build \
  ubuntu@13.51.159.80:~/crypto-ai-system/dashboard/
```

### 4. Push feature CSVs (when updated)
```bash
# Crypto
scp -i ~/Desktop/TradeWise.pem \
  ~/Desktop/crypto-ai-system/data/BTC_USDT_multi_tf_features.csv \
  ~/Desktop/crypto-ai-system/data/LINK_USDT_multi_tf_features.csv \
  ubuntu@13.51.159.80:~/crypto-ai-system/data/

# India stocks (viable symbols only)
scp -i ~/Desktop/TradeWise.pem \
  ~/Desktop/crypto-ai-system/india-stocks/data/NIFTY50_features.csv \
  ~/Desktop/crypto-ai-system/india-stocks/data/BANKNIFTY_features.csv \
  ~/Desktop/crypto-ai-system/india-stocks/data/NIFTYIT_features.csv \
  ~/Desktop/crypto-ai-system/india-stocks/data/TITAN_features.csv \
  ~/Desktop/crypto-ai-system/india-stocks/data/AXISBANK_features.csv \
  ubuntu@13.51.159.80:~/crypto-ai-system/india-stocks/data/
```

### 5. Restart APIs after deploy
```bash
# Crypto
pkill -f "api_final" 2>/dev/null; sleep 1
cd ~/crypto-ai-system
nohup ~/crypto-ai-system/venv/bin/uvicorn api_final:app \
  --host 0.0.0.0 --port 8000 --workers 1 >> /tmp/api.log 2>&1 &

# Stocks
pkill -f "api_stocks" 2>/dev/null; sleep 1
cd ~/crypto-ai-system/india-stocks
nohup env ACTIVE_SYMBOLS="NIFTY50,BANKNIFTY,NIFTYIT,TITAN,AXISBANK" \
  ~/crypto-ai-system/venv/bin/uvicorn api_stocks:app \
  --host 0.0.0.0 --port 8001 --workers 1 >> /tmp/stocks_api.log 2>&1 &
```

## Python Venv (EC2)
- Path: `~/crypto-ai-system/venv` (Python 3.9)
- Key packages installed: fastapi, uvicorn, lightgbm, numpy>=2.0, pandas>=2.0,
  scikit-learn, joblib, aiohttp, websockets, pyotp, yfinance, groq, openai, python-dotenv
- libgomp1 system lib required for lightgbm (`sudo apt-get install -y libgomp1`)

## Cron Jobs (EC2)
Auto-installed via `india-stocks/ec2_setup.sh`:
- **11:00 UTC (4:30pm IST) Mon-Fri** — NSE data collection
- **11:30 UTC (5:00pm IST) Mon-Fri** — FII/DII data
- **11:45 UTC (5:15pm IST) Mon-Fri** — Feature engineering
- **04:00 + 10:00 UTC Mon-Fri** — Option chain snapshots
- **Every 5 min** — Auto-restart api_stocks if crashed

## EC2 is NOT a git repo
Files are deployed via rsync/scp from Mac. To update code on EC2, always push from Mac.

## env.production (dashboard)
```
REACT_APP_API_URL=http://13.51.159.80:8000
REACT_APP_WS_URL=                          # leave empty — auto-builds from window.location
REACT_APP_STOCKS_API_URL=http://13.51.159.80:8000
REACT_APP_STOCKS_WS_URL=ws://13.51.159.80:8000/ws/stocks/prices
```

## Known Issues Fixed
- Python 3.9 on EC2: all `X | None` type hints need `from __future__ import annotations`
- numpy 2.0 + pandas binary incompatibility: fix with `pip install --force-reinstall numpy pandas scikit-learn lightgbm joblib`
- libgomp1 missing: `sudo apt-get install -y libgomp1`
- REACT_APP_WS_URL must be empty (not ws://host:port) — let fallback handle `/ws/prices` path

## Disk Management
- Total: 20GB — keep below 80% (~16GB)
- Large items: venv (600MB), data CSVs (400MB), models (viable only ~200MB)
- Do NOT copy: NOT_VIABLE model pkl files, node_modules, venv_old, sector_cache
- Free space: `df -h /` and `du -sh ~/crypto-ai-system/*/`
